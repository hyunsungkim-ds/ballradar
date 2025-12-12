import os
import re
import sys
from typing import List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
        if len(alive_tracking) < 100:
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


def inference(
    model: nn.Module,
    tracking: pd.DataFrame,
    masking_prob: float = 1.0,
    evaluate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run inference for a trained PlayerBall model on tracking data in the Sportec-like format.

    Args:
        model: Trained PlayerBall model.
        tracking: Tracking DataFrame with columns such as frame_id, period_id, timestamp, phase_id, episode_id,
                  ball_state, ball_owning_team_id, player_id, and player physical features.
        masking_prob: Probability used for the same random masking behaviour as TraceHelper.
        evaluate: Whether to compute simple accuracy/error stats while running inference.

    Returns:
        macro_pred_df, micro_pred_df, stats
    """
    tracking = tracking.copy().dropna(axis=1, how="all")

    if "episode_id" not in tracking.columns or "frame_id" not in tracking.columns:
        tracking, _ = label_frames_and_episodes(tracking)
    if "phase_id" not in tracking.columns:
        tracking = label_phases(tracking)

    tracking = tracking.set_index("frame_id")

    macro_type = getattr(model, "macro_type", None)
    target_type = model.target_type

    n_features = model.params["n_features"]
    n_team_players = model.params.get("n_players", 11)

    feature_types = ["_x", "_y", "_vx", "_vy", "_speed", "_accel"][:n_features]
    outside_xy = {
        "out_left": (0, config.PITCH_Y / 2),
        "out_right": (config.PITCH_X, config.PITCH_Y / 2),
        "out_bottom": (config.PITCH_X / 2, 0),
        "out_top": (config.PITCH_X / 2, config.PITCH_Y),
    }
    outside_labels = list(outside_xy.keys())

    home_players = sorted({c[:-2] for c in tracking.columns if re.match(r"home_.*_x", c)})
    away_players = sorted({c[:-2] for c in tracking.columns if re.match(r"away_.*_x", c)})
    all_players = home_players + away_players

    if macro_type == "team_poss":
        macro_pred_df = pd.DataFrame(index=tracking.index, columns=["home", "away"], dtype=float)
    elif macro_type == "player_poss":
        macro_pred_df = pd.DataFrame(index=tracking.index, columns=all_players + outside_labels, dtype=float)
        for k, xy in outside_xy.items():
            tracking[f"{k}_x"] = xy[0]
            tracking[f"{k}_y"] = xy[1]
            tracking[[f"{k}_vx", f"{k}_vy", f"{k}_speed", f"{k}_accel"]] = 0
    else:
        macro_pred_df = None

    if target_type == "player_poss":
        micro_pred_df = pd.DataFrame(index=tracking.index, columns=all_players + outside_labels, dtype=float)
        for k, xy in outside_xy.items():
            tracking[f"{k}_x"] = xy[0]
            tracking[f"{k}_y"] = xy[1]
            tracking[[f"{k}_vx", f"{k}_vy", f"{k}_speed", f"{k}_accel"]] = 0
    elif target_type == "ball":
        micro_pred_df = pd.DataFrame(index=tracking.index, columns=["ball_x", "ball_y"], dtype=float)
        if masking_prob < 1:
            tracking["masked_ball_x"] = np.nan
            tracking["masked_ball_y"] = np.nan
    else:
        micro_pred_df = pd.DataFrame(index=tracking.index, dtype=float)

    stats = {"n_frames": 0}
    correct_team_poss = 0
    correct_player_poss = 0
    sum_pos_error = 0
    device = next(model.parameters()).device

    for phase in tracking["phase_id"].unique():
        phase_tracking = tracking[tracking["phase_id"] == phase]
        if phase == 0 or phase_tracking.empty:
            continue

        phase_x_cols = phase_tracking[[f"{p}_x" for p in all_players]].dropna(axis=1, how="all").columns
        phase_players = [c[:-2] for c in phase_x_cols]

        left_gk, right_gk = detect_keepers(phase_tracking)
        left_team, right_team = left_gk.split("_")[0], right_gk.split("_")[0]
        left_players = [p for p in phase_players if p.startswith(left_team)]
        right_players = [p for p in phase_players if p.startswith(right_team)]

        if min(len(left_players), len(right_players)) < n_team_players:
            continue

        ordered_players = left_players + right_players
        input_cols = [f"{p}{ft}" for p in ordered_players for ft in feature_types]

        if macro_type == "player_poss" or target_type == "player_poss":
            outside_labels = ["out_left", "out_right", "out_bottom", "out_top"]
            input_cols += [f"{label}{ft}" for label in outside_labels for ft in feature_types]
            macro_cols = ordered_players + outside_labels

            team_poss_dict = {left_team: 0, right_team: 1, "out": 2, "goal": 2}
            player_poss_dict = dict(zip(macro_cols, np.arange(len(macro_cols))))
            player_poss_dict["goal_left"] = len(macro_cols) - 4
            player_poss_dict["goal_right"] = len(macro_cols) - 3

        if target_type == "ball":
            output_cols = ["ball_x", "ball_y"]
        elif target_type == "player_poss":
            output_cols = macro_cols

        episodes = [e for e in phase_tracking["episode_id"].unique() if e > 0]
        for episode in tqdm(episodes, desc=f"Phase {phase}"):
            ep_tracking = phase_tracking[phase_tracking["episode_id"] == episode]
            if ep_tracking[input_cols].isna().any().any():
                continue

            ep_input = torch.tensor(ep_tracking[input_cols].values, dtype=torch.float32)
            random_mask = torch.empty((1, len(ep_input), 1)).uniform_() > masking_prob

            macro_target = None
            if macro_type == "team_poss":
                macro_target = torch.tensor((ep_tracking["ball_owning_team_id"] == right_team).values, dtype=torch.long)
            elif macro_type == "player_poss":
                player_poss = ep_tracking["player_id"].bfill().ffill().map(player_poss_dict)
                if player_poss.isna().any():
                    continue
                macro_target = torch.tensor(player_poss.values, dtype=torch.long)

            micro_target = None
            if target_type == "player_poss":
                player_poss = ep_tracking["player_id"].bfill().ffill().map(player_poss_dict)
                if player_poss.isna().any():
                    continue
                micro_target = torch.tensor(player_poss.values, dtype=torch.long)
            elif target_type == "ball":
                micro_target = torch.tensor(ep_tracking[output_cols].values, dtype=torch.float32)

            if macro_type == "player_poss" and target_type == "ball" and masking_prob < 1:
                random_mask_np = random_mask.numpy()[0, :, 0]
                tracking.loc[ep_tracking.index, "masked_poss"] = player_poss.where(random_mask_np)
                tracking.loc[ep_tracking.index, "masked_ball_x"] = ep_tracking["ball_x"].where(random_mask_np)
                tracking.loc[ep_tracking.index, "masked_ball_y"] = ep_tracking["ball_y"].where(random_mask_np)

            with torch.no_grad():
                ep_input_t = ep_input.unsqueeze(0).to(device)
                macro_target_t = macro_target.unsqueeze(0).to(device) if macro_target is not None else None
                micro_target_t = micro_target.unsqueeze(0).to(device) if micro_target is not None else None
                random_mask_t = random_mask.to(device) if random_mask is not None else None

                ep_pred = model.forward(ep_input_t, macro_target_t, micro_target_t, random_mask_t)
                ep_pred = ep_pred.squeeze(0).detach().cpu()

                if macro_type is None:
                    micro_pred = ep_pred
                else:
                    macro_pred = ep_pred[:, :-2]
                    micro_pred = ep_pred[:, -2:]
                    macro_pred_probs = nn.Softmax(dim=-1)(macro_pred).numpy()
                    if macro_type == "team_poss":
                        macro_pred_df.loc[ep_tracking.index, left_team] = macro_pred_probs[:, 0]
                        macro_pred_df.loc[ep_tracking.index, right_team] = macro_pred_probs[:, 1]
                    elif macro_type == "player_poss":
                        macro_pred_df.loc[ep_tracking.index, macro_cols] = macro_pred_probs

                if target_type in ["team_poss", "player_poss"]:
                    micro_pred = nn.Softmax(dim=-1)(micro_pred)

            micro_pred_df.loc[ep_tracking.index, output_cols] = micro_pred.numpy()

            stats["n_frames"] += micro_pred.shape[0]

            if evaluate:
                if macro_type == "team_poss":
                    correct_team_poss += ((macro_pred_probs[:, 1] > 0.5) == macro_target.numpy()).astype(int).sum()
                elif macro_type == "player_poss":
                    team_poss_pred = np.argmax(macro_pred.numpy(), axis=1) // n_team_players
                    team_poss_target = player_poss.apply(
                        lambda x: x.split("_")[0] if isinstance(x, str) else np.nan
                    ).map(team_poss_dict)
                    correct_team_poss += (team_poss_pred == team_poss_target).sum()
                    correct_player_poss += (np.argmax(macro_pred_probs, axis=1) == macro_target.numpy()).sum()

                if target_type == "player_poss":
                    team_poss_pred = np.argmax(micro_pred.numpy(), axis=1) // n_team_players
                    team_poss_target = player_poss.apply(
                        lambda x: x.split("_")[0] if isinstance(x, str) else np.nan
                    ).map(team_poss_dict)
                    correct_team_poss += (team_poss_pred == team_poss_target).sum()
                    correct_player_poss += (np.argmax(micro_pred.numpy(), axis=1) == micro_target.numpy()).sum()
                elif target_type == "ball":
                    sum_pos_error += np.linalg.norm(micro_pred.numpy() - micro_target.numpy(), axis=1).sum()

            del ep_input_t, macro_target_t, micro_target_t, random_mask_t, ep_pred

        if macro_type is not None:
            phase_macro = macro_pred_df.loc[phase_tracking.index]
            macro_pred_df.loc[phase_tracking.index] = phase_macro.interpolate(limit_direction="both")

        phase_micro = micro_pred_df.loc[phase_tracking.index]
        micro_pred_df.loc[phase_tracking.index] = phase_micro.interpolate(limit_direction="both")

    # if macro_type is not None:
    #     valid_poss_mask = macro_pred_df.notna().any(axis=1)
    #     tracking["pred_poss"] = np.nan
    #     tracking.loc[valid_poss_mask, "pred_poss"] = macro_pred_df.loc[valid_poss_mask].idxmax(axis=1, skipna=True)
    #     argmax_idxs = np.argpartition(-macro_pred_df.values, range(3), axis=1)[:, :3]
    #     player_poss_top3 = pd.DataFrame(np.array(macro_pred_df.columns)[argmax_idxs])
    #     tracking["pred_poss_top3"] = player_poss_top3.apply(lambda x: x.tolist(), axis=1)
    #     # tracking["pred_poss_top3"] = macro_pred_df.apply(lambda row: row.nlargest(3).index.tolist(), axis=1)

    # if {"ball_x", "ball_y"}.issubset(micro_pred_df.columns):
    #     tracking["pred_ball_x"] = micro_pred_df["ball_x"]
    #     tracking["pred_ball_y"] = micro_pred_df["ball_y"]

    if stats["n_frames"] == 0:
        return macro_pred_df, micro_pred_df, stats

    if evaluate:
        if correct_team_poss > 0:
            stats["correct_team_poss"] = correct_team_poss
        if correct_player_poss > 0:
            stats["correct_player_poss"] = correct_player_poss
        if target_type == "ball":
            stats["sum_pos_error"] = sum_pos_error
        print({k: round(v / stats["n_frames"], 4) if "correct" in k or "error" in k else v for k, v in stats.items()})

    return macro_pred_df, micro_pred_df, stats
