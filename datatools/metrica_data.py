import os
import re
import sys
from typing import Optional, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

from datatools import config


class MetricaData:
    def __init__(
        self,
        home_tracking: pd.DataFrame = None,
        away_tracking: pd.DataFrame = None,
        tracking_from_txt: pd.DataFrame = None,
        events: pd.DataFrame = None,
    ):
        events = events.copy()

        if home_tracking is not None:
            assert away_tracking is not None and tracking_from_txt is None
            home_tracking = home_tracking.copy()
            away_tracking = away_tracking.copy()
            tracking = MetricaData._parse_tracking_from_csv(home_tracking, away_tracking)
        else:
            assert away_tracking is None and tracking_from_txt is not None
            tracking = tracking_from_txt.copy()

        tracking = MetricaData.format_tracking(tracking).dropna(axis=1, how="all")
        x_cols = [c for c in tracking.columns if c.endswith("_x")]
        y_cols = [c for c in tracking.columns if c.endswith("_y")]
        tracking[x_cols] *= config.PITCH_X
        tracking[y_cols] *= config.PITCH_Y

        players = [c[:-2] for c in tracking.columns if c.endswith("_x") and not c.startswith("ball")]
        events = MetricaData.format_events(events, players)

        self.tracking, self.events = MetricaData._rebase_timestamps(tracking, events)

    @staticmethod
    def _get_team(player_id: str) -> str:
        if isinstance(player_id, str) and (player_id.startswith("home_") or player_id.startswith("away_")):
            return player_id.split("_", 1)[0]
        else:
            return np.nan

    @staticmethod
    def _parse_player_id(player_id: str, team: Optional[str] = None) -> str:
        if pd.isna(player_id):
            return np.nan
        player_id = str(player_id).strip()
        if player_id.startswith("home_") or player_id.startswith("away_"):
            return player_id
        match = re.match(r"^(A|B)(\d+)$", player_id)
        if match:
            team_prefix = "home" if match.group(1) == "A" else "away"
            return f"{team_prefix}_{int(match.group(2))}"
        match = re.match(r"^Player\s*(\d+)$", player_id, flags=re.IGNORECASE)
        if match and team in ["home", "away"]:
            return f"{team}_{int(match.group(1))}"
        return player_id

    @staticmethod
    def _parse_tracking_from_csv(home_tracking: pd.DataFrame, away_tracking: pd.DataFrame) -> pd.DataFrame:
        home_players = [f"home_{int(c[2][6:])}" for c in home_tracking.columns[3:-2:2]]
        home_xy_cols = np.array([[f"{p}_x", f"{p}_y"] for p in home_players]).flatten().tolist()
        home_tracking.columns = ["period_id", "frame_id", "timestamp"] + home_xy_cols + ["ball_x", "ball_y"]
        home_tracking = home_tracking.set_index("frame_id").astype(float)
        home_tracking["period_id"] = home_tracking["period_id"].astype(int)

        away_players = [f"away_{int(c[2][6:])}" for c in away_tracking.columns[3:-2:2]]
        away_xy_cols = np.array([[f"{p}_x", f"{p}_y"] for p in away_players]).flatten().tolist()
        away_tracking.columns = ["period_id", "frame_id", "timestamp"] + away_xy_cols + ["ball_x", "ball_y"]
        away_tracking = away_tracking.set_index("frame_id").astype(float)
        away_tracking["period_id"] = away_tracking["period_id"].astype(int)

        header = home_tracking.columns[:-2].tolist() + away_tracking.columns[2:].tolist()
        tracking = pd.merge(home_tracking, away_tracking)[header]
        tracking.index = home_tracking.index.astype(int)
        tracking.index.name = "frame_id"
        return tracking

    @staticmethod
    def format_events(events: pd.DataFrame, players: list) -> pd.DataFrame:
        events.columns = [
            "team",
            "type",
            "subtype",
            "period_id",
            "start_frame",
            "start_time",
            "end_frame",
            "end_time",
            "from",
            "to",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]

        events["team"] = events["team"].replace({"Team A": "Home", "Team B": "Away"}).str.lower()
        events.loc[events["subtype"].isna(), "subtype"] = events.loc[events["subtype"].isna(), "type"]

        player_numbers = []
        for p in players:
            match = re.search(r"(\d+)$", p)
            if match:
                player_numbers.append((p, int(match.group(1))))

        player_dict1 = dict(zip([f"Player {num}" for p, num in player_numbers], [p for p, num in player_numbers]))
        player_dict2 = dict(zip([f"Player{num}" for p, num in player_numbers], [p for p, num in player_numbers]))
        player_dict = {**player_dict1, **player_dict2}
        player_dict[np.nan] = np.nan

        def map_player_id(player_id, team):
            if pd.isna(player_id):
                return np.nan
            mapped = MetricaData._parse_player_id(player_id, team=team)
            if mapped != player_id:
                return mapped
            return player_dict.get(player_id, player_id)

        events["from"] = events.apply(lambda row: map_player_id(row["from"], row["team"]), axis=1)
        events["to"] = events.apply(lambda row: map_player_id(row["to"], row["team"]), axis=1)

        return events.drop("team", axis=1)

    @staticmethod
    def format_tracking(tracking: pd.DataFrame) -> pd.DataFrame:
        tracking = tracking.copy()
        rename_map = {}
        if "frame" in tracking.columns:
            rename_map["frame"] = "frame_id"
        if "session" in tracking.columns:
            rename_map["session"] = "period_id"
        if "time" in tracking.columns:
            rename_map["time"] = "timestamp"
        if rename_map:
            tracking = tracking.rename(columns=rename_map)
        if tracking.index.name == "frame":
            tracking.index.name = "frame_id"

        rename_map = {}
        for col in tracking.columns:
            match = re.match(r"^(A|B)(\d+)_(.+)$", col)
            if match:
                team_prefix = "home" if match.group(1) == "A" else "away"
                rename_map[col] = f"{team_prefix}_{int(match.group(2))}_{match.group(3)}"
        if rename_map:
            tracking = tracking.rename(columns=rename_map)

        for col in ["player_id", "event_player"]:
            if col in tracking.columns:
                tracking[col] = tracking[col].apply(MetricaData._parse_player_id)
        if "team_poss" in tracking.columns:
            tracking["team_poss"] = tracking["team_poss"].replace({"A": "home", "B": "away"})

        if "frame_id" in tracking.columns and tracking.index.name != "frame_id":
            tracking = tracking.set_index("frame_id")
        elif tracking.index.name is None and "frame_id" not in tracking.columns:
            tracking.index.name = "frame_id"
        return tracking

    @staticmethod
    def _rebase_timestamps(
        tracking: pd.DataFrame,
        events: pd.DataFrame = None,
        frame_dt: float = 0.04,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tracking = tracking.copy()
        events = events.copy() if events is not None else None

        frame_id_series = None
        if "frame_id" in tracking.columns:
            tracking["frame_id"] = tracking["frame_id"].astype(int)
            tracking["frame_id"] = (tracking["frame_id"] - 1).clip(lower=0)
            frame_id_series = tracking["frame_id"]
        elif tracking.index.name == "frame_id":
            tracking.index = tracking.index.astype(int)
            tracking.index = pd.Index(np.maximum(tracking.index - 1, 0), name="frame_id")
            frame_id_series = tracking.index.to_series(index=tracking.index)

        period_start_frames = None
        recomputed_timestamp = False
        if "period_id" in tracking.columns and frame_id_series is not None:
            period_start_frames = frame_id_series.groupby(tracking["period_id"]).min().to_dict()
            period_offsets = tracking["period_id"].map(period_start_frames)
            tracking["timestamp"] = ((frame_id_series - period_offsets) * frame_dt).round(2)
            recomputed_timestamp = True

        if not recomputed_timestamp and "timestamp" in tracking.columns:
            tracking["timestamp"] = (tracking["timestamp"] - frame_dt).clip(lower=0).round(2)

        if events is not None:
            recomputed_event_times = False
            for col in ["start_frame", "end_frame"]:
                if col in events.columns:
                    events[col] = pd.to_numeric(events[col], errors="coerce")
                    events[col] = (events[col] - 1).clip(lower=0)

            if period_start_frames is not None and "period_id" in events.columns:
                period_offsets = events["period_id"].map(period_start_frames)
                if "start_frame" in events.columns:
                    events["start_time"] = ((events["start_frame"] - period_offsets) * frame_dt).round(2)
                if "end_frame" in events.columns:
                    events["end_time"] = ((events["end_frame"] - period_offsets) * frame_dt).round(2)
                recomputed_event_times = True

            if not recomputed_event_times:
                for col in ["start_time", "end_time"]:
                    if col in events.columns:
                        events[col] = (events[col] - frame_dt).clip(lower=0).round(2)

        return tracking, events

    @staticmethod
    def construct_phase_records(tracking: pd.DataFrame) -> pd.DataFrame:
        home_players = sorted({c[:-2] for c in tracking.columns if re.match(r"home_\d+_x", c)})
        away_players = sorted({c[:-2] for c in tracking.columns if re.match(r"away_\d+_x", c)})
        player_x_cols = [f"{p}_x" for p in home_players + away_players]

        play_records = []
        phase_records = []

        for p in home_players + away_players:
            valid_player_idx = tracking[tracking[f"{p}_x"].notna()].index
            f0 = valid_player_idx[0]
            f1 = valid_player_idx[-1]

            if len(tracking.loc[f1, player_x_cols].dropna()) > 22:
                play_records.append([p, f0, f1 - 1])
            else:
                play_records.append([p, f0, f1])

        play_records = pd.DataFrame(play_records, columns=["object", "start_frame", "end_frame"]).set_index("object")

        change_frames = play_records["start_frame"].tolist()
        last_frame = play_records["end_frame"].max()
        change_frames.extend([tracking[tracking["period_id"] == 2].index[0], last_frame + 1])
        change_frames = list(set(change_frames))
        change_frames.sort()

        for i, f0 in enumerate(change_frames[:-1]):
            f1 = change_frames[i + 1] - 1
            period_id = tracking.loc[f0, "period_id"]
            start_time = round(tracking.at[f0, "timestamp"], 2)
            end_time = round(tracking.at[f1, "timestamp"], 2)

            inplay_flags = tracking.loc[f0:f1, player_x_cols].notna().any()
            player_ids = [c[:-2] for c in inplay_flags[inplay_flags].index]

            phase_records.append([i + 1, period_id, start_time, end_time, player_ids])

        header = ["phase_id", "period_id", "start_time", "end_time", "player_ids"]
        return pd.DataFrame(phase_records, columns=header).set_index("phase_id")

    @staticmethod
    def label_phases(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        tracking = tracking.copy()

        events["phase_id"] = 0
        tracking["phase_id"] = 0

        phase_records = MetricaData.construct_phase_records(tracking)

        for phase_id, row in phase_records.iterrows():
            period_id = row["period_id"]
            start_time = row["start_time"]
            end_time = row["end_time"]

            e_mask = (events["period_id"] == period_id) & events["start_time"].between(start_time, end_time)
            t_mask = (tracking["period_id"] == period_id) & tracking["timestamp"].between(start_time, end_time)
            events.loc[e_mask, "phase_id"] = phase_id
            tracking.loc[t_mask, "phase_id"] = phase_id

        return events, tracking

    @staticmethod
    def label_episodes(
        events: pd.DataFrame,
        tracking: pd.DataFrame,
        margin_sec: float = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        events = events.copy()
        tracking = tracking.copy()

        tracking["episode_id"] = 0
        count = 0

        for phase in tracking["phase_id"].unique():
            phase_events = events[(events["phase_id"] == phase) & (events["type"] != "CARD")]
            phase_tracking = tracking[tracking["phase_id"] == phase]
            assert isinstance(phase_events, pd.DataFrame)
            assert isinstance(phase_tracking, pd.DataFrame)

            time_diffs = phase_events["start_time"].diff().fillna(60)
            episodes = (time_diffs > 10).astype(int).cumsum() + count
            count = episodes.max() if not episodes.empty else count

            grouped = phase_events.groupby(episodes)["start_time"]
            first_event_times = grouped.min()
            last_event_times = grouped.max()

            for episode in first_event_times.index:
                first_time = round(first_event_times.loc[episode] - margin_sec, 2)
                last_time = round(last_event_times.loc[episode] + margin_sec, 2)
                episode_idxs = phase_tracking[phase_tracking["timestamp"].between(first_time, last_time)].index
                tracking.loc[episode_idxs, "episode_id"] = episode

        tracking["ball_state"] = np.where(tracking["episode_id"] > 0, "alive", "dead")
        return events, tracking

    def merge_for_animation(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        tracking = tracking.copy()
        events = events[["start_frame", "from", "type"]].copy()
        events.columns = ["frame_id", "player_id", "event_type"]
        merged = pd.merge(tracking.drop("player_id", axis=1), events, how="left")

        merged["event_x"] = np.nan
        merged["event_y"] = np.nan

        mask = merged["event_type"].notna()
        rows = np.flatnonzero(mask)

        player_ids = merged.loc[mask, "player_id"].astype(str)
        x_cols = (player_ids + "_x").to_numpy()
        y_cols = (player_ids + "_y").to_numpy()
        col_idx_x = merged.columns.get_indexer(x_cols)
        col_idx_y = merged.columns.get_indexer(y_cols)

        merged.loc[mask, "event_x"] = merged.to_numpy()[rows, col_idx_x].astype(float)
        merged.loc[mask, "event_y"] = merged.to_numpy()[rows, col_idx_y].astype(float)

        merged["player_id"] = merged["player_id"].ffill()
        merged["event_type"] = merged["event_type"].ffill()
        merged["event_x"] = merged["event_x"].ffill().bfill()
        merged["event_y"] = merged["event_y"].ffill().bfill()

        return merged

    @staticmethod
    def label_possessions(
        events: pd.DataFrame,
        tracking: pd.DataFrame,
        out_gap_frames: int = 50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        events = events.copy()
        tracking = tracking.copy()

        tracking["player_id"] = pd.Series(np.nan, index=tracking.index, dtype="object")
        if "frame_id" in tracking.columns:
            tracking.set_index("frame_id", inplace=True)

        events = events[
            ~(events["type"].isin(["CARRY", "CARD", "SET PIECE"]))
            & ~((events["type"] == "BALL LOST") & (events["subtype"] == "THEFT"))
            & ~((events["type"] == "CHALLENGE") & (events["subtype"].str.endswith("-LOST")))
        ].copy()

        type_order = ["BALL LOST", "CHALLENGE", "RECOVERY", "FAULT RECEIVED", "PASS", "SHOT", "BALL OUT"]
        events["type"] = pd.Categorical(events["type"], categories=type_order)
        events.sort_values(["start_frame", "end_frame", "type"], inplace=True)

        out_frame = 0

        for i in events.index:
            event_type = events.at[i, "type"]
            event_subtype = events.at[i, "subtype"]
            start_frame = int(events.at[i, "start_frame"])
            end_frame = int(events.at[i, "end_frame"])

            if start_frame in tracking.index:
                tracking.at[start_frame, "player_id"] = events.at[i, "from"]

            to_player = events.at[i, "to"]
            if pd.notna(to_player) and end_frame in tracking.index:
                tracking.at[end_frame, "player_id"] = to_player

            if event_type == "BALL OUT" or event_subtype.endswith("-OUT") or event_subtype.endswith("-GOAL"):
                out_x = events.at[i, "end_x"]
                out_y = events.at[i, "end_y"]
                if out_x < 0:
                    out_label = "goal_left" if event_subtype.endswith("-GOAL") else "out_left"
                elif out_x > 1:
                    out_label = "goal_right" if event_subtype.endswith("-GOAL") else "out_right"
                elif out_y < 0:
                    out_label = "out_bottom"
                elif out_y > 1:
                    out_label = "out_top"
                else:
                    continue

                out_frame = end_frame
                if i == events.index[-1]:
                    tracking.loc[out_frame:, "player_id"] = out_label
                else:
                    i_next = events[events["start_frame"] > out_frame + out_gap_frames].index[0]
                    next_frame = events.at[i_next, "start_frame"]
                    tracking.loc[out_frame : next_frame - (out_gap_frames + 1), "player_id"] = out_label
                    tracking.loc[next_frame - out_gap_frames : next_frame, "player_id"] = events.at[i_next, "from"]

        poss_prev = tracking["player_id"].ffill()
        poss_next = tracking["player_id"].bfill()
        tracking["player_id"] = poss_prev.where(poss_prev == poss_next, np.nan)
        tracking["ball_owning_team_id"] = tracking["player_id"].apply(MetricaData._get_team).bfill().ffill()

        tracking_cols = config.TRACKING_COLS + [c for c in tracking.columns if c not in config.TRACKING_COLS]
        return events[config.EVENT_COLS], tracking[tracking_cols].reset_index()

    @staticmethod
    def find_nearest_player(snapshot, players, team_code=None):
        if team_code is None:
            team_players = players
        else:
            team_players = [p for p in players if MetricaData._get_team(p) == team_code]
            if not team_players:
                team_players = players

        x_cols = [f"{p}_x" for p in team_players]
        y_cols = [f"{p}_y" for p in team_players]

        ball_dists_x = (snapshot[x_cols] - snapshot["ball_x"]).astype(float).values
        ball_dists_y = (snapshot[y_cols] - snapshot["ball_y"]).astype(float).values

        ball_dists = pd.Series(np.sqrt(ball_dists_x**2 + ball_dists_y**2), index=team_players)

        return ball_dists.idxmin()

    def correct_event_player_ids(self):
        print("\nCorrecting event player IDs:")
        players = self.home_players + self.away_players
        player_x_cols = [f"{p}_x" for p in players]
        valid_types = ["BALL LOST", "BALL OUT", "CHALLENGE", "PASS", "RECOVERY", "SET PIECE", "SHOT"]

        for phase in self.events["phase_id"].unique():
            phase_events = self.events[self.events["phase_id"] == phase]
            phase_tracking = self.tracking[self.tracking["phase_id"] == phase]
            phase_players = [c[:-2] for c in phase_tracking[player_x_cols].dropna(axis=1).columns]

            switch_counts = pd.DataFrame(0, index=players, columns=players)
            for i in tqdm(phase_events.index, desc=f"Phase {phase}"):
                event_type = phase_events.at[i, "type"]
                event_subtype = phase_events.at[i, "subtype"]

                if event_type in valid_types:
                    if event_type == "BALL LOST" and event_subtype == "THEFT":
                        continue
                    if event_type == "CHALLENGE" and not event_subtype.endswith("-WON"):
                        continue

                    start_frame = phase_events.at[i, "start_frame"]
                    end_frame = phase_events.at[i, "end_frame"]

                    recorded_p_from = phase_events.at[i, "from"]
                    recorded_team = MetricaData._get_team(recorded_p_from)
                    if pd.isna(recorded_team):
                        recorded_team = None
                    detected_p_from = MetricaData.find_nearest_player(
                        phase_tracking.loc[start_frame - 1], phase_players, recorded_team
                    )
                    switch_counts.at[recorded_p_from, detected_p_from] += 1

                    if event_type == "PASS":
                        recorded_p_to = phase_events.at[i, "to"]
                        recorded_team = MetricaData._get_team(recorded_p_to)
                        if pd.isna(recorded_team):
                            recorded_team = None
                        detected_p_to = MetricaData.find_nearest_player(
                            phase_tracking.loc[end_frame - 1], phase_players, recorded_team
                        )
                        switch_counts.at[recorded_p_to, detected_p_to] += 1

            mapping = switch_counts[switch_counts.sum(axis=1) > 0].idxmax(axis=1).to_dict()
            self.events.loc[phase_events.index, "from"] = phase_events["from"].replace(mapping)
            self.events.loc[phase_events.index, "to"] = phase_events["to"].replace(mapping)
            self.tracking.loc[phase_tracking.index, "event_player"] = phase_tracking["event_player"].replace(mapping)
            self.tracking.loc[phase_tracking.index, "player_id"] = phase_tracking["player_id"].replace(mapping)

    def construct_pass_records(self, frames: pd.Series = None):
        events = self.events[self.events["start_frame"].isin(frames)] if frames is not None else self.events

        valid_types = ["BALL LOST", "RECOVERY", "PASS"]
        events = events[events["type"].isin(valid_types)].copy()
        events["type"] = pd.Categorical(events["type"], categories=valid_types)
        events.sort_values(["start_time", "type"], ignore_index=True, inplace=True)

        passes = []

        for i in events.index[:-1]:
            event_type = events.at[i, "type"]
            event_subtype = events.at[i, "subtype"]

            if event_type == "PASS" or (event_type == "BALL LOST" and "INTERCEPTION" in event_subtype):
                episode = events.at[i, "episode_id"]
                start_frame = events.at[i, "start_frame"]
                passer = events.at[i, "from"]

                if event_type == "PASS":
                    end_frame = events.at[i, "end_frame"]
                    receiver = events.at[i, "to"]
                    success = True

                else:  # event_type == BALL LOST and event_subtype contains INTERCEPTION
                    next_events = events[i + 1 : i + 5]
                    recovery = next_events[(next_events["type"] == "RECOVERY") & (next_events["from"] != passer)]
                    if recovery.empty:
                        continue
                    else:
                        i_next = recovery.index[0]
                        end_frame = events.at[i_next, "start_frame"]
                        receiver = events.at[i_next, "from"]
                        success = False

                passes.append([episode, start_frame, end_frame, passer, receiver, success])

        pass_cols = ["episode_id", "start_frame", "end_frame", "passer", "receiver", "success"]
        return pd.DataFrame(passes, columns=pass_cols)
