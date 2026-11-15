"""The apparent pupil center shift at the center target, measured and simulated, one figure per camera.

    uv run python simulators/apparent_pupil_center_shift_center_target.py

One panel per eye, with one arrow per participant for the measured apparent pupil center shift and for each
simulator's, all from one origin. The arrows are the apparent pupil center shift per mm of apparent pupil diameter
change, in screen axes. Each participant has one color, the same as in the other figures, and each source its own
line style and arrowhead: the measured arrow is solid with a filled head, the simulated ones dashed, dotted and
dash-dotted with an open, a hollow and a swept head.

Writes output/apparent_pupil_center_shift_center_target_<camera>.png, <camera> being eye_level_camera or
bottom_left_camera.
"""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
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
from psa_eye_model_evaluation.simulator_results import SIMULATORS, Shifts, load_simulator_result_file

OUTPUT = Path(__file__).resolve().parent / "output"
LABELS = {MEASURED: "Measured", **SIMULATORS}
# The measured and the three simulated arrows of one eye lie close together, so each is drawn partly transparent.
ARROW_ALPHA = 0.8


def center_target_shifts(camera: str, measured: Shifts) -> dict[str, Shifts]:
    """{source: shift at the center target} on one camera, for the measurement and every simulator."""
    shifts = {
        MEASURED: {key: vector for key, vector in measured.items() if key[0] == camera and key[3] == CENTER_TARGET}
    }
    for simulator, name in SIMULATORS.items():
        simulated = {
            key: v for key, v in load_simulator_result_file(simulator, camera).items() if key[3] == CENTER_TARGET
        }
        if simulated.keys() != shifts[MEASURED].keys():
            raise SystemExit(f"the {name} result for {camera} covers other eyes than the recordings")
        shifts[simulator] = simulated
    return shifts


def draw_shifts(panels: dict[str, Axes], per_camera: dict[str, Shifts], colors: dict[str, str]) -> None:
    """Every source's arrow for every participant and eye, from the origin of that eye's panel."""
    for source, vectors in per_camera.items():
        for (_camera, participant, eye, _target), vector in vectors.items():
            add_arrow(
                panels[eye], (0.0, 0.0), (float(vector[0]), float(vector[1])), colors[participant], source, ARROW_ALPHA
            )


def main() -> None:
    """Draw the measured and every simulator's center-target shift, one figure per camera and one panel per eye."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    measured = load_measured_apparent_pupil_center_shift()
    colors = participant_colors({participant for _camera, participant, _eye, _target in measured})
    for camera in PANEL_CAMERAS:
        per_camera = center_target_shifts(camera, measured)
        tips = {
            eye: np.array([
                vector for source in per_camera.values() for key, vector in source.items() if key[2] == eye
            ])
            for eye in EYES
        }
        fig = Figure(figsize=(9.0, 5.0))
        panels = eye_panels(fig, tips)
        draw_shifts(panels, per_camera, colors)

        legend_below(fig, [LegendArrow(source) for source in per_camera], [LABELS[source] for source in per_camera])
        fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.155, wspace=0.08)
        path = OUTPUT / f"apparent_pupil_center_shift_center_target_{CAMERA_FILE_NAME[camera]}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
