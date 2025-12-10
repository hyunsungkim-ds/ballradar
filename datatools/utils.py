import os
import re
import sys
from typing import List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

from datatools import config


def detect_keepers(period_tracking: pd.DataFrame):
    home_x_cols = [c for c in period_tracking.columns if re.match(r"home_.*_x", c)]
    away_x_cols = [c for c in period_tracking.columns if re.match(r"away_.*_x", c)]

    home_gk = (period_tracking[home_x_cols].mean() - config.PITCH_X / 2).abs().idxmax()[:-2]
    away_gk = (period_tracking[away_x_cols].mean() - config.PITCH_Y / 2).abs().idxmax()[:-2]

    home_gk_x = period_tracking[f"{home_gk}_x"].mean()
    away_gk_x = period_tracking[f"{away_gk}_x"].mean()

    return (home_gk, away_gk) if home_gk_x < away_gk_x else (away_gk, home_gk)


def find_active_players(traces: pd.DataFrame, frame: int = None, team: str = None, include_goals=False) -> dict:
    if pd.isna(frame):
        snapshot = traces.dropna(how="all", axis=1).copy()
    else:
        snapshot = traces.loc[frame:frame].dropna(how="all", axis=1).copy()

    if include_goals:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_.*_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_.*_x", c)]
    else:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_\d+_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_\d+_x", c)]

    if not pd.isna(frame):
        team = team or traces.at[frame, "ball_owning_home_away"]
    else:
        team = team or "home"

    if team == "home":
        players = [home_players, away_players]
    else:
        players = [away_players, home_players]

    return players


def label_frames_and_episodes(
    tracking: pd.DataFrame,
    events: pd.DataFrame = None,
    fps: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tracking = tracking.copy().sort_values(["period_id", "timestamp"], ignore_index=True)

    if "frame_id" not in tracking.columns:
        tracking["frame_id"] = (tracking["timestamp"] * fps).round().astype(int)
        n_prev_frames = 0

        for i in tracking["period_id"].unique():
            period_tracking = tracking[tracking["period_id"] == i]
            tracking.loc[period_tracking.index, "frame_id"] += n_prev_frames
            n_prev_frames += len(period_tracking)

    tracking["episode_id"] = 0
    n_prev_episodes = 0

    for i in tracking["period_id"].unique():
        period_tracking = tracking[tracking["period_id"] == i].copy()
        alive_tracking = period_tracking[period_tracking["ball_state"] == "alive"].copy()

        frame_diffs = np.diff(alive_tracking["frame_id"].values, prepend=-5)
        period_episode_ids = (frame_diffs >= 5).astype(int).cumsum() + n_prev_episodes
        tracking.loc[alive_tracking.index, "episode_id"] = period_episode_ids

        n_prev_episodes = period_episode_ids.max()

    tracking = tracking.set_index("frame_id")

    if events is not None:
        events = events.copy()
        events["episode_id"] = np.nan

        for i in events.index:
            frame_id = events.at[i, "frame_id"]
            if not pd.isna(frame_id):
                events.at[i, "episode_id"] = tracking.at[frame_id, "episode_id"]

    return tracking.reset_index(), events


def label_phases(tracking: pd.DataFrame) -> pd.DataFrame:
    phases = summarize_phases(tracking)

    tracking = tracking.copy()
    tracking["phase_id"] = 0

    for i in phases.index:
        start_frame = phases.at[i, "start_frame_id"]
        end_frame = phases.at[i, "end_frame_id"]
        phase_mask = tracking["frame_id"].between(start_frame, end_frame)
        tracking.loc[phase_mask, "phase_id"] = i

    return tracking


def summarize_playing_times(tracking: pd.DataFrame) -> pd.DataFrame:
    if "frame_id" in tracking.columns:
        tracking = tracking.copy().set_index("frame_id")

    players = [c[:-2] for c in tracking.columns if c[:4] in ["home", "away"] and c.endswith("_x")]
    play_records = dict()

    for p in players:
        player_x = tracking[f"{p}_x"].dropna()
        if not player_x.empty:
            play_records[p] = {"in_frame_id": player_x.index[0], "out_frame_id": player_x.index[-1]}

    return pd.DataFrame(play_records).T


def summarize_phases(tracking: pd.DataFrame, keepers: List[str] = None) -> pd.DataFrame:
    if "frame_id" in tracking:
        tracking = tracking.copy().set_index("frame_id")

    keepers = [] if keepers is None else list(keepers)

    play_records = summarize_playing_times(tracking)
    player_in_frames = play_records["in_frame_id"].unique().tolist()
    player_out_frames = (play_records["out_frame_id"].unique() + 1).tolist()
    period_start_frames = tracking.reset_index().groupby("period_id")["frame_id"].first().values.tolist()
    phase_changes = np.sort(np.unique(player_in_frames + player_out_frames + period_start_frames))

    phases = []

    for i, start_frame in enumerate(phase_changes[:-1]):
        end_frame = phase_changes[i + 1] - 1

        alive_tracking = tracking[tracking["ball_state"] == "alive"].loc[start_frame:end_frame].copy()
        if len(alive_tracking) < 250:
            continue

        active_players = find_active_players(alive_tracking)
        home_keepers = [p for p in keepers if p in active_players[0]]
        away_keepers = [p for p in keepers if p in active_players[1]]
        home_x_cols = [f"{p}_x" for p in active_players[0]]
        away_x_cols = [f"{p}_x" for p in active_players[1]]
        home_keeper = home_keepers[0] if home_keepers else alive_tracking[home_x_cols].mean().idxmin()[:-2]
        away_keeper = away_keepers[0] if away_keepers else alive_tracking[away_x_cols].mean().idxmax()[:-2]

        phase_dict = {
            "period_id": alive_tracking["period_id"].iloc[0],
            "start_frame_id": start_frame,
            "end_frame_id": end_frame,
            "active_players": active_players[0] + active_players[1],
            "active_keepers": [home_keeper, away_keeper],
        }
        phases.append(phase_dict)

    phases = pd.DataFrame(phases)
    phases.index.name = "phase"
    phases.index += 1

    return phases


def calculate_running_features(tracking: pd.DataFrame, fps=25) -> pd.DataFrame:
    from scipy.signal import savgol_filter

    tracking = tracking.copy()

    if "episode_id" not in tracking.columns:
        tracking = label_frames_and_episodes(tracking)

    if "phase_id" not in tracking.columns:
        tracking = label_phases(tracking)

    home_players = [c[:-2] for c in tracking.dropna(axis=1, how="all").columns if re.match(r"home_.*_x", c)]
    away_players = [c[:-2] for c in tracking.dropna(axis=1, how="all").columns if re.match(r"away_.*_x", c)]
    objects = home_players + away_players + ["ball"]
    physical_features = ["x", "y", "vx", "vy", "speed", "accel"]

    state_cols = ["frame_id", "period_id", "timestamp", "phase_id", "episode_id", "ball_state", "ball_owning_team_id"]
    feature_cols = [f"{p}_{f}" for p in objects for f in physical_features] + ["ball_z"]

    if "player_id" in tracking.columns:
        state_cols.append("player_id")

    for p in tqdm(objects, desc="Calculating running features per player"):
        new_cols = [f"{p}_{x}" for x in physical_features[2:]]
        new_features = pd.DataFrame(np.nan, index=tracking.index, columns=new_cols)

        # Drop pre-existing columns to avoid duplicate column names during concat/assign
        tracking = tracking.drop(columns=[c for c in new_cols if c in tracking.columns], errors="ignore")
        tracking = pd.concat([tracking, new_features], axis=1)

        for i in tracking["period_id"].unique():
            x: pd.Series = tracking.loc[tracking["period_id"] == i, f"{p}_x"].dropna()
            y: pd.Series = tracking.loc[tracking["period_id"] == i, f"{p}_y"].dropna()
            if x.empty:
                continue

            vx = savgol_filter(np.diff(x.values) * fps, window_length=15, polyorder=2)
            vy = savgol_filter(np.diff(y.values) * fps, window_length=15, polyorder=2)
            ax = savgol_filter(np.diff(vx) * fps, window_length=9, polyorder=2)
            ay = savgol_filter(np.diff(vy) * fps, window_length=9, polyorder=2)

            tracking.loc[x.index[1:], f"{p}_vx"] = vx
            tracking.loc[x.index[1:], f"{p}_vy"] = vy
            tracking.loc[x.index[1:], f"{p}_speed"] = np.sqrt(vx**2 + vy**2)
            tracking.loc[x.index[1:-1], f"{p}_accel"] = np.sqrt(ax**2 + ay**2)

            tracking.at[x.index[0], f"{p}_vx"] = tracking.at[x.index[1], f"{p}_vx"]
            tracking.at[x.index[0], f"{p}_vy"] = tracking.at[x.index[1], f"{p}_vy"]
            tracking.at[x.index[0], f"{p}_speed"] = tracking.at[x.index[1], f"{p}_speed"]
            tracking.loc[[x.index[0], x.index[-1]], f"{p}_accel"] = 0

    return tracking[state_cols + feature_cols].copy()
