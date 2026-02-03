import os
import sys
from datetime import timedelta
from typing import Dict, Optional, Union

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, axes, collections, lines, patches, text

import datatools.matplotsoccer as mps
from datatools import config

anim_config = {
    "sports": "soccer",  # soccer or basketball
    "figsize": (9, 6),
    "fontsize": 15,
    "player_size": 400,
    "ball_size": 150,
    "star_size": 150,
    "annot_size": 100,
    "player_history": 20,
    "ball_history": 50,
    "edge_threshold": 0.05,
    "self_loop_scale": 2.0,
    "self_loop_linewidth_base": 0.8,
    "arrow_head_scale": 2.5,
    "arrow_tail_scale": 1.0,
}


class Animator:
    def __init__(
        self,
        track_dict: Optional[Dict[str, pd.DataFrame]] = None,
        events: Optional[pd.DataFrame] = None,
        heatmaps: Optional[np.ndarray] = None,
        player_sizes: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        edge_seq: Optional[pd.DataFrame] = None,
        edge_weights: Optional[pd.DataFrame] = None,
        show_times: bool = True,
        show_episodes: bool = False,
        show_events: bool = False,
        text_cols: bool = None,  # column names for additional annotation
        rotate_pitch: bool = False,
        anonymize: bool = False,
        small_image: bool = False,
        play_speed: int = 1,
    ):
        self.track_dict = track_dict
        self.heatmaps = heatmaps
        self.sizes = player_sizes
        self.edge_seq = edge_seq
        self.edge_weights = edge_weights

        if events is None or events.empty:
            self.events = None
        else:
            self.events = events[events["frame_id"].isin(track_dict["main"].index)]

        self.sports = anim_config["sports"]
        self.show_times = show_times
        self.show_episodes = show_episodes
        self.show_events = show_events
        self.text_cols = text_cols
        self.rotate_pitch = rotate_pitch
        self.anonymize = anonymize

        self.small_image = small_image
        self.play_speed = play_speed
        self.arg_dict = dict()

    @staticmethod
    def plot_players(
        tracking: pd.DataFrame,
        ax: axes.Axes,
        sizes=750,
        alpha=1,
        anonymize=False,
        edgecolors=None,
        linewidths=None,
    ):
        if len(tracking.columns) == 0:
            return None

        color = "tab:red" if tracking.columns[0].startswith("home_") else "tab:blue"
        x = tracking[tracking.columns[0::2]].values
        y = tracking[tracking.columns[1::2]].values
        size = sizes[0, 0] if isinstance(sizes, np.ndarray) else sizes
        lw = linewidths[0, 0] if isinstance(linewidths, np.ndarray) else linewidths
        scatter_kwargs = {}
        if edgecolors is not None:
            scatter_kwargs["edgecolors"] = edgecolors
        if lw is not None:
            scatter_kwargs["linewidths"] = lw
        scat = ax.scatter(x[0], y[0], s=size, c=color, alpha=alpha, zorder=2, **scatter_kwargs)

        players = [c[:-2] for c in tracking.columns[0::2]]
        player_dict = dict(zip(players, np.arange(len(players)) + 1))
        plots = dict()
        annots = dict()

        for p in players:
            (plots[p],) = ax.plot([], [], c=color, alpha=alpha, ls=":", zorder=0)

            player_id = player_dict[p] if anonymize else int(p.split("_")[-1])
            annots[p] = ax.annotate(
                player_id,
                xy=tracking.loc[0, [f"{p}_x", f"{p}_y"]],
                ha="center",
                va="center",
                color="w",
                fontsize=anim_config["fontsize"] - 2,
                fontweight="bold",
                annotation_clip=False,
                zorder=3,
            )
            annots[p].set_animated(True)

        return tracking, sizes, scat, plots, annots, linewidths

    @staticmethod
    def animate_players(
        t: int,
        inplay_records: pd.DataFrame,
        tracking: pd.DataFrame,
        sizes: np.ndarray,
        scat: collections.PatchCollection,
        plots: Dict[str, lines.Line2D],
        annots: Dict[str, text.Annotation],
        linewidths=None,
    ):
        x = tracking[tracking.columns[0::2]].values
        y = tracking[tracking.columns[1::2]].values
        scat.set_offsets(np.stack([x[t], y[t]]).T)

        if isinstance(sizes, np.ndarray):
            scat.set_sizes(sizes[t])
        if isinstance(linewidths, np.ndarray):
            scat.set_linewidths(linewidths[t])
        elif linewidths is not None:
            scat.set_linewidths(linewidths)

        for p in plots.keys():
            inplay_start = inplay_records.at[p, "start_index"]
            inplay_end = inplay_records.at[p, "end_index"]

            if t >= inplay_start:
                if t <= inplay_end:
                    t_from = max(t - anim_config["player_history"] + 1, inplay_start)
                    plots[p].set_data(tracking.loc[t_from:t, f"{p}_x"], tracking.loc[t_from:t, f"{p}_y"])
                    annots[p].set_position(tracking.loc[t, [f"{p}_x", f"{p}_y"]].values)
                elif t == inplay_end + 1:
                    plots[p].set_alpha(0)
                    annots[p].set_alpha(0)

    @staticmethod
    def plot_ball(xy: pd.DataFrame, ax=axes.Axes, color="w", edgecolor="k", marker="o", show_path=True):
        x = xy.values[:, 0]
        y = xy.values[:, 1]
        scat = ax.scatter(
            x,
            y,
            s=anim_config["ball_size"],
            c=color,
            edgecolors=edgecolor,
            marker=marker,
            zorder=4,
        )

        if show_path:
            pathcolor = "k" if color == "w" else color
            (plot,) = ax.plot([], [], pathcolor, zorder=3)
        else:
            plot = None

        return x, y, scat, plot

    @staticmethod
    def animate_ball(
        t: int,
        x: np.ndarray,
        y: np.ndarray,
        scat: collections.PatchCollection,
        plot: lines.Line2D = None,
    ):
        scat.set_offsets(np.array([x[t], y[t]]))

        if plot is not None:
            t_from = max(t - anim_config["ball_history"], 0)
            plot.set_data(x[t_from : t + 1], y[t_from : t + 1])

    @staticmethod
    def plot_events(xy: pd.DataFrame, ax=axes.Axes, color="orange", edgecolor="k", marker="*"):
        x = xy.values[:, 0]
        y = xy.values[:, 1]
        scat = ax.scatter(
            x[0],
            y[0],
            s=anim_config["star_size"] if marker == "*" else anim_config["annot_size"],
            c=color,
            edgecolors=edgecolor if marker != "x" else None,
            marker=marker,
            zorder=100,
        )
        return x, y, scat

    @staticmethod
    def animate_events(t: int, x: np.ndarray, y: np.ndarray, scat: collections.PatchCollection):
        scat.set_offsets(np.array([x[t], y[t]]))

    @staticmethod
    def plot_event_table(events: pd.DataFrame, fig: plt.Figure, table_ax: plt.Axes, frame_times: np.ndarray) -> dict:
        if events is None or table_ax is None or frame_times is None:
            return None

        assert "timestamp" in events.columns
        events = events.copy()
        if isinstance(events["timestamp"].iloc[0], timedelta):
            events["_ts"] = events["timestamp"].dt.total_seconds()
        else:
            events["_ts"] = pd.to_numeric(events["timestamp"], errors="coerce")

        events = events.dropna(subset="_ts").sort_values("_ts").reset_index(drop=True)
        event_times = events["_ts"].to_numpy()
        event_players = events.get("player_id", pd.Series([], dtype=str)).astype(str).to_numpy()
        event_types = events.get("event_type", pd.Series([], dtype=str)).astype(str).to_numpy()

        fig_h_px = fig.get_size_inches()[1] * fig.dpi
        ax_h_px = table_ax.get_position().height * fig_h_px
        row_h_px = anim_config["fontsize"] * 1.2
        header_lines = 2
        available_px = max(0, ax_h_px - row_h_px * header_lines)
        max_rows = max(1, int(available_px / row_h_px))
        line_step = row_h_px / ax_h_px if ax_h_px > 0 else 0.05

        header_text = "time     | player  | event_type"
        header_artist = table_ax.text(
            0,
            1,
            header_text,
            fontsize=anim_config["fontsize"] - 2,
            fontfamily="monospace",
            ha="left",
            va="top",
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = header_artist.get_window_extent(renderer=renderer)
        x0, _ = table_ax.transAxes.inverted().transform((bbox.x0, bbox.y0))
        x1, _ = table_ax.transAxes.inverted().transform((bbox.x1, bbox.y0))
        _, y_gap = table_ax.transAxes.inverted().transform((0, bbox.y0 - 2))
        table_ax.plot(
            [x0, x1],
            [1 - line_step * 1.3, 1 - line_step * 1.3],
            transform=table_ax.transAxes,
            color="black",
            linewidth=1.0,
            clip_on=False,
        )

        event_rows = []
        start_y = 1 - line_step * 1.6
        for i in range(max_rows):
            y = start_y - i * line_step
            row = table_ax.text(
                0,
                y,
                "",
                fontsize=anim_config["fontsize"] - 2,
                fontfamily="monospace",
                ha="left",
                va="top",
            )
            event_rows.append(row)

        return {
            "event_rows": event_rows,
            "event_times": event_times,
            "event_players": event_players,
            "event_types": event_types,
            "event_queue": [],
            "event_ptr": 0,
        }

    @staticmethod
    def animate_event_table(t: int, frame_times: np.ndarray, event_attr: dict):
        if event_attr is None or frame_times is None:
            return

        event_rows = event_attr["event_rows"]
        event_times = event_attr["event_times"]
        event_players = event_attr["event_players"]
        event_types = event_attr["event_types"]
        event_queue = event_attr["event_queue"]
        event_ptr = event_attr["event_ptr"]

        current_time = frame_times[t]
        if np.isfinite(current_time):
            while event_ptr < len(event_times) and event_times[event_ptr] <= current_time + 1e-9:
                sec = float(event_times[event_ptr])
                mm = int(sec // 60)
                ss = sec % 60
                time_str = f"{mm:02d}:{ss:05.2f}"
                node_id = event_players[event_ptr] if event_ptr < len(event_players) else ""
                event_type = event_types[event_ptr] if event_ptr < len(event_types) else ""
                event_queue.append(f"{time_str} | {node_id:7s} | {event_type}")
                if len(event_queue) > len(event_rows):
                    event_queue.pop(0)
                event_ptr += 1

            for i, row in enumerate(event_rows):
                row.set_text(event_queue[i] if i < len(event_queue) else "")

        event_attr["event_ptr"] = event_ptr

    def plot_init(self, ax: axes.Axes, track_key: str):
        tracking = self.track_dict[track_key].iloc[:: self.play_speed].copy()
        tracking = tracking.dropna(axis=1, how="all").reset_index(drop=True)
        xy_cols = [c for c in tracking.columns if c.endswith("_x") or c.endswith("_y")]

        if self.rotate_pitch:
            tracking[xy_cols[0::2]] = config.PITCH_X - tracking[xy_cols[0::2]]
            tracking[xy_cols[1::2]] = config.PITCH_Y - tracking[xy_cols[1::2]]

        inplay_records = []
        for c in xy_cols[::2]:
            inplay_index = tracking[tracking[c].notna()].index
            inplay_records.append([c[:-2], inplay_index[0], inplay_index[-1]])
        inplay_records = pd.DataFrame(inplay_records, columns=["object", "start_index", "end_index"])

        home_tracking = tracking[[c for c in xy_cols if c.startswith("home_")]].fillna(-100)
        away_tracking = tracking[[c for c in xy_cols if c.startswith("away_")]].fillna(-100)
        base_size = anim_config["player_size"]
        home_linewidths = None
        away_linewidths = None
        home_edgecolors = None
        away_edgecolors = None

        if track_key == "main" and self.edge_weights is not None:
            edge_weights = self.edge_weights.reset_index(drop=True).loc[tracking.index].copy()
            home_players = [c[:-2] for c in home_tracking.columns[0::2]]
            away_players = [c[:-2] for c in away_tracking.columns[0::2]]
            player_order = home_players + away_players

            base_lw = anim_config["self_loop_linewidth_base"]
            lw_mat = np.full((tracking.shape[0], len(player_order)), base_lw, dtype=float)
            for i, p in enumerate(player_order):
                if f"{p}-{p}" in edge_weights.columns:
                    w = edge_weights[f"{p}-{p}"].to_numpy(dtype=float)
                    w = np.where(np.isnan(w), 0.0, w)
                    mask = w >= anim_config["edge_threshold"]
                    lw_mat[mask, i] = base_lw + w[mask] * anim_config["self_loop_scale"]

            n_players = len(home_players)
            home_sizes = base_size
            away_sizes = base_size
            home_linewidths = lw_mat[:, :n_players]
            away_linewidths = lw_mat[:, n_players:]
            home_edgecolors = "w"
            away_edgecolors = "w"

        elif track_key == "main" and self.sizes is not None:
            sizes = self.sizes

            if isinstance(sizes, pd.DataFrame):
                player_order = [c[:-2] for c in xy_cols[::2]]
                sizes = sizes.reindex(columns=player_order)

            if sizes.shape[1] == 2:  # team_poss
                sizes = sizes.fillna(0.5).values[(self.play_speed - 1) :: self.play_speed]
                home_sizes = np.repeat(sizes[:, [0]] * 500 + base_size, home_tracking.shape[1], axis=1)
                away_sizes = np.repeat(sizes[:, [1]] * 500 + base_size, away_tracking.shape[1], axis=1)
            else:  # player_poss
                n_players = home_tracking.shape[1] // 2
                sizes = pd.DataFrame(sizes).fillna(1 / sizes.shape[1]).values[(self.play_speed - 1) :: self.play_speed]
                home_sizes = sizes[:, :n_players] * 1000 + base_size
                away_sizes = sizes[:, n_players : n_players * 2] * 1000 + base_size

        else:
            home_sizes = base_size
            away_sizes = base_size

        alpha = 1 if track_key == "main" else 0.5
        home_state = self.plot_players(
            home_tracking,
            ax,
            home_sizes,
            alpha,
            self.anonymize,
            edgecolors=home_edgecolors,
            linewidths=home_linewidths,
        )
        away_state = self.plot_players(
            away_tracking,
            ax,
            away_sizes,
            alpha,
            self.anonymize,
            edgecolors=away_edgecolors,
            linewidths=away_linewidths,
        )

        ball_state = None
        if "ball_x" in tracking.columns and tracking["ball_x"].notna().any():
            ball_xy = tracking[["ball_x", "ball_y"]]
            if track_key == "main":
                if self.sports == "soccer":
                    ball_state = Animator.plot_ball(ball_xy, ax, "w", "k", "o")
                else:
                    ball_state = Animator.plot_ball(ball_xy, ax, "darkorange", "k", "o")
            else:
                ball_state = Animator.plot_ball(ball_xy, ax, track_key, None, "*")

        self.track_dict[track_key] = tracking
        self.arg_dict[track_key] = {
            "inplay_records": inplay_records.set_index("object"),
            "home": home_state,
            "away": away_state,
            "ball": ball_state,
        }

    def run(self, cmap="jet", vmin=0, vmax=1, max_frames=np.inf, fps=10):
        if self.edge_weights is None and self.edge_seq is not None and "main" in self.track_dict:
            main_tracking = self.track_dict["main"].iloc[:: self.play_speed].copy()
            main_tracking = main_tracking.dropna(axis=1, how="all").reset_index(drop=True)

            edge_seq = self.edge_seq.iloc[:: self.play_speed].copy().reset_index(drop=True)
            edge_seq = edge_seq.reindex(range(main_tracking.shape[0])).bfill()

            if {"edge_src", "edge_dst"}.issubset(edge_seq.columns):
                valid = edge_seq["edge_src"].notna() & edge_seq["edge_dst"].notna()
                edges = pd.Series(np.where(valid, edge_seq["edge_src"] + "-" + edge_seq["edge_dst"], np.nan))
                if edges.notna().any():
                    edges_oh = pd.get_dummies(edges).astype(float)
                    self.edge_weights = edges_oh.reindex(range(main_tracking.shape[0])).fillna(0.0)

        table_ax = None
        if self.sports == "soccer":
            if self.show_events and self.events is not None:
                fig = plt.figure(figsize=(anim_config["figsize"][0] * 1.35, anim_config["figsize"][1]))
                gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.5], wspace=0.05)
                ax = fig.add_subplot(gs[0, 0])
                table_ax = fig.add_subplot(gs[0, 1])
                table_ax.axis("off")
                table_ax.set_xlim(0, 1)
                table_ax.set_ylim(0, 1)
            else:
                fig, ax = plt.subplots(figsize=anim_config["figsize"])
            mps.field("green", config.PITCH_X, config.PITCH_Y, fig, ax, show=False)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
            fig.set_tight_layout(True)
        else:
            if self.events is not None:
                fig = plt.figure(figsize=(10 * 1.35, 5.2))
                gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.5], wspace=0.05)
                ax = fig.add_subplot(gs[0, 0])
                table_ax = fig.add_subplot(gs[0, 1])
                table_ax.axis("off")
                table_ax.set_xlim(0, 1)
                table_ax.set_ylim(0, 1)
            else:
                fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.set_xlim(-2, config.PITCH_X + 2)
            ax.set_ylim(-1, config.PITCH_Y + 1)
            ax.axis("off")
            ax.grid(False)
            court = plt.imread("images/bball_court.png")
            ax.imshow(court, zorder=0, extent=[0, config.PITCH_X, config.PITCH_Y, 0])

        for key in self.track_dict.keys():
            self.plot_init(ax, key)

        main_tracking = self.track_dict["main"]
        text_y = config.PITCH_Y + 1
        edge_weights = None
        edge_tuples = None
        arrows = None
        node_xy = None

        if self.edge_weights is not None:
            edge_weights = self.edge_weights.iloc[:: self.play_speed].copy().reset_index(drop=True)

            parsed = []
            for col in edge_weights.columns:
                edge = tuple(col.split("-", 1))
                if len(edge) == 2:
                    parsed.append((edge[0], edge[1], col))

            node_xy = {}
            for node in {e[0] for e in parsed} | {e[1] for e in parsed}:
                x_col = f"{node}_x"
                y_col = f"{node}_y"
                if x_col in main_tracking.columns and y_col in main_tracking.columns:
                    node_xy[node] = (main_tracking[x_col].to_numpy(), main_tracking[y_col].to_numpy())

            edge_tuples = [(s, d, c) for s, d, c in parsed if s in node_xy and d in node_xy]
            arrows = []
            hs = anim_config["arrow_head_scale"]
            ts = anim_config["arrow_tail_scale"]

            for _ in edge_tuples:
                arrow = patches.FancyArrowPatch(
                    (0, 0),
                    (0, 0),
                    arrowstyle=f"Simple,head_length={hs},head_width={hs},tail_width={ts}",
                    mutation_scale=10,
                    facecolor="w",
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=0.7,
                    zorder=1,
                    shrinkA=5,
                    shrinkB=5,
                )
                arrow.set_visible(False)
                ax.add_patch(arrow)
                arrows.append(arrow)

        if self.heatmaps is not None:
            hm_extent = (0, config.PITCH_X, 0, config.PITCH_Y)
            hm = ax.imshow(self.heatmaps[0], extent=hm_extent, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.7)

        timestamps = None
        frame_times = None
        if "timestamp" in main_tracking.columns:
            timestamps = main_tracking["timestamp"]
        elif "time_left" in main_tracking.columns:
            timestamps = main_tracking["time_left"]

        if timestamps is not None and len(timestamps) > 0:
            if isinstance(timestamps.iloc[0], timedelta):
                frame_times = timestamps.dt.total_seconds().to_numpy()
            else:
                frame_times = pd.to_numeric(timestamps, errors="coerce").to_numpy()

        if self.show_times:
            timestamps = main_tracking["timestamp"] if self.sports == "soccer" else main_tracking["time_left"]
            timestamps = timestamps.dt.total_seconds() if isinstance(timestamps.iloc[0], timedelta) else timestamps
            timestamps_str = timestamps.apply(lambda x: f"{int(x // 60):02d}:{x % 60:05.2f}").values
            time_text = ax.text(
                0,
                text_y,
                timestamps_str[0],
                fontsize=anim_config["fontsize"],
                ha="left",
                va="bottom",
            )
            time_text.set_animated(True)

        if self.show_episodes:
            episodes_str = main_tracking["episode_id"].apply(lambda x: f"Episode {x}")
            episodes_str = np.where(episodes_str == "Episode 0", "", episodes_str)
            text_x = config.PITCH_X
            episode_text = ax.text(
                text_x,
                text_y,
                episodes_str[0],
                fontsize=anim_config["fontsize"],
                ha="right",
                va="bottom",
            )
            episode_text.set_animated(True)

        if self.show_events:
            if self.events is not None:
                event_attr = Animator.plot_event_table(self.events, fig, table_ax, frame_times)

            elif "event_type" in main_tracking.columns:
                events_str = main_tracking.apply(lambda x: f"{x['event_type']} by {x['player_id']}", axis=1)
                events_str = np.where(events_str == "nan by nan", "", events_str)

                text_x = config.PITCH_X / 2
                event_text = ax.text(
                    text_x,
                    text_y,
                    str(events_str[0]),
                    fontsize=anim_config["fontsize"],
                    ha="center",
                    va="bottom",
                )
                event_text.set_animated(True)

                if "event_x" in main_tracking.columns:
                    event_xy = main_tracking[["event_x", "event_y"]]
                    event_state = Animator.plot_events(event_xy, ax, color="orange", marker="*")

                if "annot_x" in main_tracking.columns:
                    annot_xy = main_tracking[["annot_x", "annot_y"]]
                    annot_state = Animator.plot_events(annot_xy, ax, color="k", marker="X")

        if self.text_cols is not None:
            str_dict = {}
            text_dict = {}
            for i, col in enumerate(self.text_cols):
                str_dict[col] = f"{col}: " + np.where(main_tracking[col].isna(), "", main_tracking[col].astype(str))
                text_x = config.PITCH_X * i / 2
                text_dict[col] = ax.text(
                    text_x,
                    -1,
                    str(str_dict[col][0]),
                    fontsize=anim_config["fontsize"],
                    ha="left",
                    va="top",
                )
                text_dict[col].set_animated(True)

        def animate(t):
            for key in self.track_dict.keys():
                inplay_records = self.arg_dict[key]["inplay_records"]
                home_state = self.arg_dict[key]["home"]
                away_state = self.arg_dict[key]["away"]
                ball_state = self.arg_dict[key]["ball"]

                if home_state is not None:
                    Animator.animate_players(t, inplay_records, *home_state)
                if away_state is not None:
                    Animator.animate_players(t, inplay_records, *away_state)
                if ball_state is not None:
                    Animator.animate_ball(t, *ball_state)

            if self.heatmaps is not None:
                hm.set_array(self.heatmaps[t])

            if edge_weights is not None and edge_tuples is not None:
                weights = edge_weights.iloc[t]
                for i, (src, dst, col) in enumerate(edge_tuples):
                    w = weights[col]
                    if pd.isna(w) or w < anim_config["edge_threshold"]:
                        arrows[i].set_visible(False)
                        continue

                    src_xy = node_xy[src]
                    dst_xy = node_xy[dst]
                    x0, y0 = src_xy[0][t], src_xy[1][t]
                    x1, y1 = dst_xy[0][t], dst_xy[1][t]

                    if not np.isfinite([x0, y0, x1, y1]).all():
                        arrows[i].set_visible(False)
                        continue
                    if src == dst:
                        arrows[i].set_visible(False)
                        continue

                    arrow = arrows[i]
                    arrow.set_positions((x0, y0), (x1, y1))
                    hs = max(0.1, w) * anim_config["arrow_head_scale"]
                    ts = max(0.1, w) * anim_config["arrow_tail_scale"]
                    arrow.set_arrowstyle(f"Simple,head_length={hs},head_width={hs},tail_width={ts}")
                    arrow.set_mutation_scale(10)
                    arrow.set_visible(True)

            if self.show_times:
                time_text.set_text(str(timestamps_str[t]))

            if self.show_episodes:
                episode_text.set_text(str(episodes_str[t]))

            if self.show_events:
                if self.events is not None:
                    Animator.animate_event_table(t, frame_times, event_attr)
                elif "event_type" in main_tracking.columns:
                    event_text.set_text(events_str[t])
                    if "event_x" in main_tracking.columns:
                        Animator.animate_events(t, *event_state)
                    if "annot_x" in main_tracking.columns:
                        Animator.animate_events(t, *annot_state)

            if self.text_cols is not None:
                for col in self.text_cols:
                    text_dict[col].set_text(str(str_dict[col][t]))

        frames = min(max_frames, main_tracking.shape[0])
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 / fps)
        plt.close(fig)

        return anim
