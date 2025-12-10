import os
import re
import sys
from typing import List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from rdp import rdp
from tqdm import tqdm

from datatools import config, utils


class PossDetector:
    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame):
        self.tracking, self.events = utils.label_frames_and_episodes(tracking, events)
        self.atomic_events = PossDetector.get_atomic_events(self.events, self.tracking)

        self.home_players = [c[:-2] for c in tracking.columns if re.match(r"home_\d+_x", c)]
        self.away_players = [c[:-2] for c in tracking.columns if re.match(r"away_\d+_x", c)]
        self.team_players = {"home": self.home_players, "away": self.away_players}

    @staticmethod
    def get_atomic_events(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        if "frame_id" in tracking.columns:
            tracking = tracking.copy().set_index("frame_id")

        home_players = [c[:-2] for c in tracking.columns if re.match(r"home_\d+_x", c)]
        away_players = [c[:-2] for c in tracking.columns if re.match(r"away_\d+_x", c)]
        players = home_players + away_players

        event_starts = events[["frame_id", "synced_ts", "player_id", "spadl_type"]].copy()
        event_ends = events[["receive_frame_id", "receive_ts", "receiver_id"]].copy()
        event_ends.columns = ["frame_id", "synced_ts", "player_id"]
        event_ends["spadl_type"] = "dribble"

        atomic_events = pd.concat([event_starts, event_ends]).sort_values("frame_id").dropna(ignore_index=True)
        grouped = atomic_events.groupby(["frame_id", "player_id"])
        has_event_start = grouped["spadl_type"].transform(lambda s: (s != "dribble").any())
        drop_mask = (atomic_events["spadl_type"] == "dribble") & has_event_start
        atomic_events = atomic_events[~drop_mask].reset_index(drop=True)

        atomic_events["frame_id"] = atomic_events["frame_id"].astype(int)
        atomic_events["episode_id"] = 0
        atomic_events["x"] = np.nan
        atomic_events["y"] = np.nan

        for i in atomic_events.index:
            frame_id = atomic_events.at[i, "frame_id"]
            player_id = atomic_events.at[i, "player_id"]

            if player_id in players:
                event_x, event_y = tracking.loc[frame_id, [f"{player_id}_x", f"{player_id}_y"]]
            else:
                event_x, event_y = tracking.loc[frame_id, ["ball_x", "ball_y"]]

            atomic_events.at[i, "episode_id"] = tracking.at[frame_id, "episode_id"]
            atomic_events.at[i, "x"] = event_x
            atomic_events.at[i, "y"] = event_y

        return atomic_events

    @staticmethod
    def dtw(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
        n, m = len(seq1), len(seq2)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        i, j = n, m
        path = []
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            step = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

        path.reverse()
        return np.array(path)

    @staticmethod
    def clean_dtw_path(
        dtw_path: List[Tuple[int, int]],
        rdp_points: np.ndarray,
        events: pd.DataFrame,
        max_dist=10.0,
    ) -> pd.DataFrame:
        pairs = np.array(dtw_path)  # (K, 2)
        rdp_idx = pairs[:, 0]
        event_idx = pairs[:, 1]

        ball_xy = rdp_points[:, 1:]  # (N, 2)
        event_xy = events[["x", "y"]].to_numpy()  # (M, 2)
        dists = np.linalg.norm(ball_xy[rdp_idx] - event_xy[event_idx], axis=1)

        df = pd.DataFrame({"rdp_idx": rdp_idx, "event_idx": event_idx, "dist": dists})

        # Clean duplicated RDP indices
        df = df.sort_values("dist")
        df = df.drop_duplicates(subset="rdp_idx", keep="first")

        # Clean duplicated event indices
        df = df.sort_values("dist")
        df = df.drop_duplicates(subset="event_idx", keep="first")

        # Filter out pairs below the threshold distance
        df = df[df["dist"] < max_dist].reset_index(drop=True)
        return df.sort_values("rdp_idx", ignore_index=True)

    @staticmethod
    def simplify_trajectory(
        xy: pd.DataFrame,
        frame_scale=config.RDP_FRAME_SCALE,
        min_angle=config.RDP_MIN_ANGLE,
    ) -> np.ndarray:
        fxy = xy.reset_index().copy()  # Columns: [frame_id, ball_x, ball_y]
        fxy["frame_id"] *= frame_scale  # To eliminate the influence of frame_id in RDP
        rdp_points = np.array(rdp(fxy, epsilon=0.5))

        dirs = np.diff(rdp_points[:, 1:], axis=0)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs_norm = dirs / (norms + 1e-8)

        v1 = dirs_norm[:-1]
        v2 = dirs_norm[1:]
        cos_theta = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
        angles = np.arccos(cos_theta)

        change_idx = np.where(angles >= np.deg2rad(min_angle))[0] + 1
        change_idx = np.concatenate([[0], change_idx, [len(rdp_points) - 1]])
        rdp_points = rdp_points[change_idx]
        rdp_points[:, 0] = (rdp_points[:, 0] / frame_scale).round()

        return rdp_points

    def detect_touches_episode(
        self,
        episode: int,
        rdp_min_angle=config.RDP_MIN_ANGLE,
        max_dist=config.POSS_MAX_DIST,
    ) -> pd.DataFrame:
        ep_tracking = self.tracking[self.tracking["episode_id"] == episode].copy().set_index("frame_id")
        ep_events = self.atomic_events[self.atomic_events["episode_id"] == episode]

        if len(ep_tracking) < 25 or ep_events.empty:
            return None

        ball_xy = ep_tracking[["ball_x", "ball_y"]].copy()
        touches = PossDetector.simplify_trajectory(ball_xy, min_angle=rdp_min_angle)

        dtw_path = PossDetector.dtw(touches, ep_events[["frame_id", "x", "y"]].values)
        dtw_path = PossDetector.clean_dtw_path(dtw_path, touches, ep_events)

        rdp_idx = dtw_path["rdp_idx"].values
        event_idx = dtw_path["event_idx"].values
        matched_events = ep_events.iloc[event_idx]

        touches = pd.DataFrame(touches, columns=["frame_id", "x", "y"])
        touches["episode_id"] = episode
        touches.loc[rdp_idx, "player_id"] = matched_events["player_id"].values
        touches.loc[rdp_idx, "spadl_type"] = matched_events["spadl_type"].values
        touches.loc[rdp_idx, "elastic_frame_id"] = matched_events["frame_id"].values
        touches.loc[rdp_idx, "elastic_ts"] = matched_events["synced_ts"].values
        event_mask = touches["player_id"].notna()

        for i, row in touches[~event_mask].iterrows():
            frame_id = int(round(row["frame_id"]))
            tracking_row: pd.Series = ep_tracking.loc[frame_id]
            cand_players = self.home_players + self.away_players

            prev_events = touches.loc[(touches.index < i) & event_mask]
            next_events = touches.loc[(touches.index > i) & event_mask]

            if not prev_events.empty and not next_events.empty:
                prev_idx = prev_events.index[-1]
                next_idx = next_events.index[0]

                prev_frame = int(round(touches.at[prev_idx, "frame_id"]))
                next_frame = int(round(touches.at[next_idx, "frame_id"]))
                prev_team = touches.at[prev_idx, "player_id"][:4]
                next_team = touches.at[next_idx, "player_id"][:4]

                if next_frame - prev_frame < 50 and prev_team == next_team and prev_team in self.team_players:
                    cand_players = self.team_players[prev_team]

            team_x_cols = [f"{p}_x" for p in cand_players]
            team_y_cols = [f"{p}_y" for p in cand_players]
            player_x = tracking_row[team_x_cols].dropna().astype(float)
            player_y = tracking_row[team_y_cols].dropna().astype(float)

            dists = np.sqrt((player_x.values - row["x"]) ** 2 + (player_y.values - row["y"]) ** 2)
            min_idx = dists.argmin()

            if dists[min_idx] >= max_dist:
                continue

            touches.at[i, "player_id"] = player_x.index[min_idx][:-2]

        return touches

    def detect_touches(self, rdp_min_angle=config.RDP_MIN_ANGLE, max_dist=config.POSS_MAX_DIST) -> pd.DataFrame:
        touches: List[pd.DataFrame] = []

        for episode in tqdm(self.tracking["episode_id"].unique(), desc="Detecting touches per episode"):
            if episode == 0:
                continue

            ep_touches = self.detect_touches_episode(episode, rdp_min_angle, max_dist)

            if ep_touches is not None and not ep_touches.empty:
                touches.append(ep_touches)

        touches: pd.DataFrame = pd.concat(touches, ignore_index=True).dropna(subset="player_id")

        out_mask = touches["player_id"] == "out"
        goal_mask = touches["player_id"] == "goal"

        touches.loc[out_mask, "spadl_type"] = "out"
        touches.loc[goal_mask, "spadl_type"] = "goal"

        out_l = out_mask & (touches["x"] < 0)
        out_r = out_mask & (touches["x"] > config.PITCH_X)
        out_b = out_mask & (touches["y"] < 0)
        out_t = out_mask & (touches["y"] > config.PITCH_Y)
        goal_l = goal_mask & (touches["x"] < 5)
        goal_r = goal_mask & (touches["x"] > config.PITCH_X - 5)

        touches.loc[out_l, "player_id"] = "out_left"
        touches.loc[out_r, "player_id"] = "out_right"
        touches.loc[out_b, "player_id"] = "out_bottom"
        touches.loc[out_t, "player_id"] = "out_top"
        touches.loc[goal_l, "player_id"] = "goal_left"
        touches.loc[goal_r, "player_id"] = "goal_right"

        return touches

    def merge_tracking_poss(self, touches: pd.DataFrame) -> pd.DataFrame:
        tracking_poss = pd.merge(self.tracking, touches[["frame_id", "player_id"]], how="left")

        for episode, ep_tracking in tracking_poss.groupby("episode_id"):
            if episode == 0:
                continue

            ep_tracking = ep_tracking.copy()
            valid_poss = ep_tracking["player_id"].dropna()

            if not valid_poss.empty:
                prev_poss = ep_tracking["player_id"].ffill()
                next_poss = ep_tracking["player_id"].bfill()
                ep_tracking["player_id"] = np.where(prev_poss == next_poss, prev_poss, np.nan)

                first_poss = valid_poss.index[0]
                last_poss = valid_poss.index[-1]
                ep_tracking.loc[:first_poss, "player_id"] = valid_poss.at[first_poss]
                ep_tracking.loc[last_poss:, "player_id"] = valid_poss.at[last_poss]

                tracking_poss.loc[ep_tracking.index, "player_id"] = ep_tracking["player_id"]

        return tracking_poss


if __name__ == "__main__":
    EVENT_DIR = "data/sportec/event_synced"
    TRACKING_DIR = "data/sportec/tracking_parquet"
    OUTPUT_DIR = "data/sportec/tracking_processed"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    match_ids = [f.split(".")[0] for f in os.listdir(EVENT_DIR)]

    for match_id in match_ids:
        print()
        print(match_id)

        events = pd.read_parquet(f"{EVENT_DIR}/{match_id}.parquet")
        tracking = pd.read_parquet(f"{TRACKING_DIR}/{match_id}.parquet")
        tracking[["timestamp", "ball_x", "ball_y"]] = tracking[["timestamp", "ball_x", "ball_y"]].round(2)

        detector = PossDetector(events, tracking)
        touches = detector.detect_touches()
        tracking = detector.merge_tracking_poss(touches)

        tracking_processed = utils.calculate_running_features(tracking)
        tracking_processed.to_parquet(f"{OUTPUT_DIR}/{match_id}.parquet")
