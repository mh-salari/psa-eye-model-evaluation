"""The measured apparent pupil center shift at the center target, one figure per camera.

    uv run python measured/plot_apparent_pupil_center_shift_center_target_measured.py

Measurement only. One panel per eye and one arrow per participant, from one origin. The arrows are the apparent pupil
center shift per mm of pupil diameter change, in screen axes. A participant has the same color in both cameras'
figures.

Writes output/apparent_pupil_center_shift_center_target_measured_<camera>.png, <camera> being eye_level_camera or
bottom_left_camera.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from psa_eye_model_evaluation.center_target_figure import MEASURED, LegendArrow, add_arrow, eye_panels, legend_below
from psa_eye_model_evaluation.experiment_setup import (
    CAMERA_FILE_NAME,
    CENTER_TARGET,
    EYES,
    PANEL_CAMERAS,
    participant_colors,
)
from psa_eye_model_evaluation.recordings import load_measured_apparent_pupil_center_shift
from psa_eye_model_evaluation.target_grid_figure import MEASURED_RATIO, center_target_label

OUTPUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    """Draw the center-target shift, one figure per camera and one panel per eye."""
    center = {
        (camera, eye, participant): vector
        for (camera, participant, eye, target), vector in load_measured_apparent_pupil_center_shift().items()
        if target == CENTER_TARGET
    }
    colors = participant_colors({participant for _camera, _eye, participant in center})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for camera in PANEL_CAMERAS:
        tips = {eye: np.array([v for (c, e, _p), v in center.items() if (c, e) == (camera, eye)]) for eye in EYES}
        fig = Figure(figsize=(9.0, 5.0))
        panels = eye_panels(fig, tips)
        for (c, eye, participant), vector in center.items():
            if c == camera:
                add_arrow(panels[eye], (0.0, 0.0), (float(vector[0]), float(vector[1])), colors[participant], MEASURED)
        legend_below(fig, [LegendArrow(MEASURED)], [center_target_label(MEASURED_RATIO)])
        fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.155, wspace=0.08)
        path = OUTPUT / f"apparent_pupil_center_shift_center_target_measured_{CAMERA_FILE_NAME[camera]}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
