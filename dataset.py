import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datatools import config, utils


class SoccerDataset(Dataset):
    def __init__(
        self,
        data_paths: str,
        target_type: str = "ball",  # Choices: [team_poss, player_poss, gk, ball]
        macro_type: str = "player_poss",  # Choices: [None, team_poss, player_poss]
        n_features: int = 6,
        fps: float = 25.0,
        window_seconds: float = 10.0,
        window_stride: int = 5,
        target_speed: bool = False,
        flip_pitch: bool = False,
    ):
        self.macro_type = macro_type
        self.target_type = target_type
        targets = [target_type]  # "gk" will be modified later

        self.feature_types = ["_x", "_y", "_vx", "_vy", "_speed", "_accel"]  # Total features to save as npy files
        self.n_features = n_features  # Number of features among self.feature_types to use in model training
        self.fps = fps
        self.window_size = int(window_seconds * fps)
        self.window_stride = int(window_stride)
        self.flip_pitch = flip_pitch

        team_size = 10 if target_type == "gk" else 11  # Number of input players per team
        outside_xy = {
            "out_left": (0, config.PITCH_Y / 2),
            "out_right": (config.PITCH_X, config.PITCH_Y / 2),
            "out_bottom": (config.PITCH_X / 2, 0),
            "out_top": (config.PITCH_X / 2, config.PITCH_Y),
        }

        input_data_list = []
        target_data_list = []
        if macro_type is not None:
            macro_data_list = []

        for f in tqdm(data_paths):
            tracking = pd.read_parquet(f)

            if macro_type == "player_poss" or target_type == "player_poss":
                for k, xy in outside_xy.items():
                    tracking[f"{k}_x"] = xy[0]
                    tracking[f"{k}_y"] = xy[1]
                    tracking[[f"{k}_vx", f"{k}_vy", f"{k}_speed", f"{k}_accel"]] = 0

            phases = utils.summarize_phases(tracking)

            for phase, row in phases.iterrows():
                active_players = row["active_players"]
                if len(active_players) < 22:
                    continue

                phase_tracking = tracking[tracking["phase_id"] == phase].copy()
                player_cols = [f"{p}{x}" for p in active_players for x in self.feature_types]

                left_gk, right_gk = utils.detect_keepers(phase_tracking)
                left_team, right_team = left_gk.split("_")[0], right_gk.split("_")[0]
                if target_type == "gk":
                    targets = [left_gk, right_gk]

                input_cols = [c for c in player_cols if re.sub(r"_[^_]+$", "", c) not in targets]
                left_cols = [c for c in input_cols if c.startswith(left_team)]
                right_cols = [c for c in input_cols if c.startswith(right_team)]
                input_cols = left_cols + right_cols  # Reorder teams so that the left team comes first

                if macro_type == "player_poss" or target_type == "player_poss":
                    input_cols += [f"{k}{x}" for k in outside_xy.keys() for x in self.feature_types]
                    object_order = [re.sub(r"_[^_]+$", "", c) for c in input_cols[::n_features]]
                    poss_dict = dict(zip(object_order, np.arange(len(object_order))))
                    poss_dict["goal_left"] = len(outside_xy) - 4  # Same as out_left
                    poss_dict["goal_right"] = len(outside_xy) - 3  # Same as out_right

                if target_type in ["gk", "ball"]:
                    target_cols = [f"{p}{t}" for p in targets for t in ["_x", "_y"]]

                for episode in phase_tracking["episode_id"].unique():
                    if episode == 0:
                        continue

                    ep_tracking = tracking[tracking["episode_id"] == episode]
                    ep_input = ep_tracking[input_cols].values

                    if macro_type == "team_poss":
                        ep_macro = (ep_tracking["ball_owning_team_id"] == right_team).astype(int).values

                    elif macro_type == "player_poss":
                        player_poss = ep_tracking["player_id"].bfill().ffill()
                        ep_macro = player_poss.map(poss_dict).values

                    if target_type == "team_poss":
                        ep_target = (ep_tracking["ball_owning_team_id"] == right_team).astype(int).values

                    elif target_type == "player_poss":
                        player_poss = ep_tracking["player_id"].bfill().ffill()
                        ep_target = player_poss.map(poss_dict).values

                    else:  # target_type in ["gk", "ball"]
                        ep_target = ep_tracking[target_cols].values
                        if target_speed:
                            x = ep_target[:, 0]
                            y = ep_target[:, 1]
                            vx = np.diff(x, prepend=x[0]) * fps
                            vy = np.diff(y, prepend=y[0]) * fps
                            speed = np.sqrt(vx**2 + vy**2)
                            ep_target = np.stack([x, y, speed], axis=-1)

                    if len(ep_tracking) >= self.window_size:
                        for i in range(0, len(ep_tracking) - self.window_size + 1, self.window_stride):
                            input_data_list.append(ep_input[i : i + self.window_size])
                            target_data_list.append(ep_target[i : i + self.window_size])
                            if macro_type is not None:
                                macro_data_list.append(ep_macro[i : i + self.window_size])

        input_data = np.stack(input_data_list, axis=0)
        target_data = np.stack(target_data_list, axis=0)
        if macro_type is not None:
            macro_data = np.stack(macro_data_list, axis=0)

        if n_features < 6:
            input_data = input_data.reshape(input_data.shape[0], self.window_size, -1, len(self.feature_types))
            input_data = input_data[:, :, :, :n_features].reshape(input_data.shape[0], self.window_size, -1)

        if flip_pitch:
            flip_x = np.random.choice(2, (input_data.shape[0], 1, 1))
            flip_y = np.random.choice(2, (input_data.shape[0], 1, 1))
            valid_dim = n_features * (team_size * 2)  # valid input dimension only including player features

            # (ref, mul) = (PITCH_X or PITCH_Y, -1) if flip == 1 else (0, 1)
            ref_x = flip_x * config.PITCH_X
            ref_y = flip_y * config.PITCH_Y
            mul_x = 1 - flip_x * 2
            mul_y = 1 - flip_y * 2

            # Flip x and y
            input_data[:, :, 0:valid_dim:n_features] = input_data[:, :, 0:valid_dim:n_features] * mul_x + ref_x
            input_data[:, :, 1:valid_dim:n_features] = input_data[:, :, 1:valid_dim:n_features] * mul_y + ref_y
            if target_type == "gk":
                target_data[:, :, 0::2] = target_data[:, :, 0::2] * mul_x + ref_x
                target_data[:, :, 1::2] = target_data[:, :, 1::2] * mul_y + ref_y
            elif target_type == "ball":
                target_data[:, :, [0]] = target_data[:, :, [0]] * mul_x + ref_x
                target_data[:, :, [1]] = target_data[:, :, [1]] * mul_y + ref_y

            # Flip vx and vy
            if n_features > 2:
                input_data[:, :, 2:valid_dim:n_features] = input_data[:, :, 2:valid_dim:n_features] * mul_x
                input_data[:, :, 3:valid_dim:n_features] = input_data[:, :, 3:valid_dim:n_features] * mul_y

            # If flip_x == 1, reorder left_team and right_team features
            left_input = input_data[:, :, : n_features * team_size]
            right_input = input_data[:, :, n_features * team_size : valid_dim]
            if macro_type == "player_poss" or target_type == "player_poss":
                outside_input = input_data[:, :, valid_dim:]
                input_permuted = np.concatenate([right_input, left_input, outside_input], -1)
            else:
                input_permuted = np.concatenate([right_input, left_input], -1)
            input_data = np.where(flip_x, input_permuted, input_data)

            if macro_type == "team_poss":
                # If flip_x == 1, switch left_team (0) and right_team (1)
                macro_data = np.where(flip_x.squeeze(-1), 1 - macro_data, macro_data)

            elif macro_type == "player_poss":
                # If flip_x == 1, switch left_team (0-10) and right_team (11-21)
                left_team_mask = macro_data < team_size
                right_team_mask = (macro_data >= team_size) & (macro_data < team_size * 2)
                left_permuted = np.where(left_team_mask, macro_data + team_size, 0)
                right_permuted = np.where(right_team_mask, macro_data - team_size, 0)

                # If flip_x == 1, switch out_left (22) and out_right (23)
                out_l_to_r = np.where(macro_data == team_size * 2, team_size * 2 + 1, 0)
                out_r_to_l = np.where(macro_data == team_size * 2 + 1, team_size * 2, 0)
                out_bt = np.where(np.isin(macro_data, [team_size * 2 + 2, team_size * 2 + 3]), macro_data, 0)
                macro_permuted = left_permuted + right_permuted + out_l_to_r + out_r_to_l + out_bt
                macro_data = np.where(flip_x.squeeze(-1), macro_permuted, macro_data)

                # If flip_y == 1, switch out_bottom (24) and out_top (25)
                out_lr = np.where(macro_data < team_size * 2 + 2, macro_data, 0)
                out_b_to_t = np.where(macro_data == team_size * 2 + 2, team_size * 2 + 3, 0)
                out_t_to_b = np.where(macro_data == team_size * 2 + 3, team_size * 2 + 2, 0)
                macro_permuted = out_lr + out_b_to_t + out_t_to_b
                macro_data = np.where(flip_y.squeeze(-1), macro_permuted, macro_data)

            if target_type == "team_poss":
                # If flip_x == 1, switch left_team (0) and right_team (1)
                target_data = np.where(flip_x.squeeze(-1), 1 - target_data, target_data)

            elif target_type == "player_poss":
                # If flip_x == 1, switch left_team (0-10) and right_team (11-21),
                left_team_mask = target_data < team_size
                right_team_mask = (target_data >= team_size) & (target_data < team_size * 2)
                left_permuted = np.where(left_team_mask, target_data + team_size, 0)
                right_permuted = np.where(right_team_mask, target_data - team_size, 0)

                # If flip_x == 1, switch out_left (22) and out_right (23)
                out_l_to_r = np.where(target_data == team_size * 2, team_size * 2 + 1, 0)
                out_r_to_l = np.where(target_data == team_size * 2 + 1, team_size * 2, 0)
                out_bt = np.where(np.isin(target_data, [team_size * 2 + 2, team_size * 2 + 3]), target_data, 0)
                target_permuted = left_permuted + right_permuted + out_l_to_r + out_r_to_l + out_bt
                target_data = np.where(flip_x.squeeze(-1), target_permuted, target_data)

                # If flip_y == 1, switch out_bottom (24) and out_top (25)
                target_invariant = np.where(target_data < team_size * 2 + 2, target_data, 0)
                out_b_to_t = np.where(target_data == team_size * 2 + 2, team_size * 2 + 3, 0)
                out_t_to_b = np.where(target_data == team_size * 2 + 3, team_size * 2 + 2, 0)
                target_permuted = target_invariant + out_b_to_t + out_t_to_b
                target_data = np.where(flip_y.squeeze(-1), target_permuted, target_data)

            elif target_type == "gk":
                # If flip_x == 1, switch left_gk and right_gk
                target_permuted = np.concatenate([target_data[:, :, 2:], target_data[:, :, :2]], -1)
                target_data = np.where(flip_x, target_permuted, target_data)

        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        if macro_type in ["team_poss", "player_poss"]:
            self.macro_data = torch.tensor(macro_data, dtype=torch.long)
        if target_type in ["team_poss", "player_poss"]:
            self.target_data = torch.tensor(target_data, dtype=torch.long)
        else:  # target_type in ["gk", "ball"]
            self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.macro_type is None:
            return self.input_data[i], None, self.target_data[i]
        else:
            return self.input_data[i], self.macro_data[i], self.target_data[i]

    def __len__(self) -> int:
        return len(self.input_data)


if __name__ == "__main__":
    dir = "data/metrica_traces"
    filepaths = [f"{dir}/{f}" for f in os.listdir(dir) if f.endswith(".csv")]
    filepaths.sort()
    dataset = SoccerDataset(filepaths[-1:], target_type="gk", train=False, save=False)
    print(dataset[10000][2])
