"""The experiment's fixed setup: the HV9 target grid, the setup geometry, and the camera, eye and participant names."""

CENTER_TARGET = 0
TARGET_LABELS = {
    0: "Center",
    1: "Top-C",
    2: "Bot-C",
    3: "Left",
    4: "Right",
    5: "Top-L",
    6: "Top-R",
    7: "Bot-L",
    8: "Bot-R",
}
ECCENTRIC_TARGETS = tuple(target for target in TARGET_LABELS if target != CENTER_TARGET)
# (horizontal, vertical) grid step of each target, screen right and up positive.
GRID = {
    0: (0, 0),
    1: (0, 1),
    2: (0, -1),
    3: (-1, 0),
    4: (1, 0),
    5: (-1, 1),
    6: (1, 1),
    7: (-1, -1),
    8: (1, -1),
}
# (horizontal, vertical) position of each target on the screen in mm from its center, screen right and up
# positive. The grid is not perfectly symmetric because the targets sit on pixel centers.
SCREEN_POSITIONS = {
    0: (0.0, 0.0),
    1: (0.0, 124.298),
    2: (0.0, -124.021),
    3: (-163.836, 0.0),
    4: (163.559, 0.0),
    5: (-122.877, 93.293),
    6: (122.600, 93.293),
    7: (-122.877, -93.016),
    8: (122.600, -93.016),
}
RIGHT, LEFT = 4, 3

# Setup geometry in mm as (x, height, distance): x positive toward the participant's right, height above the screen
# center, distance in front of the screen plane.
EYE_POSITIONS_MM = {"left_eye": (-30.0, 100.0, 814.0), "right_eye": (30.0, 100.0, 814.0)}
CAMERA_POSITIONS_MM = {"cam1": (-160.0, -95.0, 315.0), "cam2": (0.0, 105.0, 195.0)}
ILLUMINATOR_POSITION_MM = (75.0, -97.5, 435.0)
# Width and height of the monitor's active area.
SCREEN_SIZE_MM = (531.36, 298.98)

CAMERAS = ("cam1", "cam2")
# Row order for the camera-by-eye figures, eye-level first, matching the order the paper reports them in.
PANEL_CAMERAS = ("cam2", "cam1")
EYES = ("left_eye", "right_eye")
CAMERA_LABEL = {"cam1": "Bottom-left camera", "cam2": "Eye-level camera"}
# How each camera is named in figure file names.
CAMERA_FILE_NAME = {"cam1": "bottom_left_camera", "cam2": "eye_level_camera"}
EYE_LABEL = {"left_eye": "Left eye", "right_eye": "Right eye"}
# Horizontal grid direction that is nasal for each eye: screen right for the left eye, screen left for the right.
NASAL_DIRECTION = {"left_eye": 1, "right_eye": -1}
# Each hue from a different family than its neighbors or clearly apart in lightness, so any two participants can be
# told apart wherever their arrows land.
PARTICIPANT_COLORS = (
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#469990",  # teal
    "#9a6324",  # brown
    "#808000",  # olive
    "#000075",  # navy
)


def nasal_or_temporal(eye: str, target: int) -> str | None:
    """'nasal' or 'temporal' for a target left or right of the center column, None for a target in it."""
    horizontal = GRID[target][0]
    if horizontal == 0:
        return None
    return "nasal" if horizontal == NASAL_DIRECTION[eye] else "temporal"


def participant_colors(participants: set[str]) -> dict[str, str]:
    """{participant: color}. Pass every participant in the recordings, so each keeps its color in every figure."""
    if len(participants) > len(PARTICIPANT_COLORS):
        raise SystemExit(f"{len(participants)} participants but only {len(PARTICIPANT_COLORS)} distinct colors")
    return dict(zip(sorted(participants), PARTICIPANT_COLORS, strict=False))
