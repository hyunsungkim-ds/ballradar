PITCH_X, PITCH_Y = 105, 68
RDP_FRAME_SCALE = 1e-6
RDP_MIN_ANGLE = 15.0
POSS_MAX_DIST = 5.0

EVENT_COLS = [
    "period_id",
    "phase_id",
    "start_frame",
    "start_time",
    "end_frame",
    "end_time",
    "from",
    "to",
    "type",
    "subtype",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]

TRACKING_COLS = [
    "period_id",
    "timestamp",
    "phase_id",
    "episode_id",
    "ball_state",
    "ball_owning_team_id",
    "player_id",
]
