"""The apparent pupil center shift at each eccentric target minus that at the center target, for every participant.

    uv run python measured/plot_apparent_pupil_center_shift_difference_by_target.py
    uv run python measured/plot_apparent_pupil_center_shift_difference_by_target.py --simulators

One figure per camera and one panel per eye, with the targets drawn at their screen positions. At each eccentric
target, one arrow per participant is that participant's shift at the target minus their shift at the center target,
per mm of pupil diameter change and in screen axes, drawn from the target's position.

Each figure's arrow scale and panel extent are set by its own arrows. Without options it draws the measured apparent
pupil center shift. With --simulators it draws the same figures from each simulator's population eye and from
PyEtSimul's second test (each eye fitted to that camera, with the physical pupil diameters and pupil center shift
inferred at the center target), for the eyes each covers, and prints each figure's arrow scale as a multiple of the
measured figure's of the same camera. A reference arrow gives the scale in mm per mm, and a participant has the same
color in every figure.

Writes output/apparent_pupil_center_shift_difference_by_target_<camera>.png for the measurement and
output/apparent_pupil_center_shift_difference_by_target_<figure>_<camera>.png for the simulators, <figure> being the
simulator or pyetsimul_fitted for the second test, and <camera> being eye_level_camera or bottom_left_camera.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from psa_eye_model_evaluation.apparent_pupil_center_shift_change import difference_from_center_target
from psa_eye_model_evaluation.center_target_figure import THIN, LegendArrow, legend_below
from psa_eye_model_evaluation.experiment_setup import (
    CAMERA_FILE_NAME,
    CENTER_TARGET,
    ECCENTRIC_TARGETS,
    EYES,
    PANEL_CAMERAS,
    SCREEN_POSITIONS,
    participant_colors,
)
from psa_eye_model_evaluation.recordings import load_measured_apparent_pupil_center_shift
from psa_eye_model_evaluation.simulator_results import FITTED_RUN, load_simulator_result_file
from psa_eye_model_evaluation.target_grid_figure import (
    MEASURED_RATIO,
    SIMULATED_RATIO,
    arrow,
    difference_label,
    draw_reference_arrow,
    panel_span,
    reference_arrow_for_width,
    smallest_target_gap,
    style_panels,
    target_position,
)

OUTPUT = Path(__file__).resolve().parent / "output"
NAME = "apparent_pupil_center_shift_difference_by_target"
POPULATION_EYE = dict.fromkeys(PANEL_CAMERAS, "")
# The simulator figures, as (name in the file name, simulator, run per camera): each simulator's population eye, and
# PyEtSimul's second test.
SIMULATOR_FIGURES = (
    ("pyetsimul", "pyetsimul", POPULATION_EYE),
    ("et_simul", "et_simul", POPULATION_EYE),
    ("gkamodeleye", "gkamodeleye", POPULATION_EYE),
    ("pyetsimul_fitted", "pyetsimul", FITTED_RUN),
)
# Targets are drawn at this share of their screen positions, which keeps the layout's proportions and leaves room for
# the arrows.
LAYOUT_SCALE = 0.75
# The arrow of this percentile length reaches a fill share of the smallest gap between two targets, so the few longest
# arrows do not set the scale for all the others.
SCALE_PERCENTILE = 90
MEASURED_FILL = 0.85
# The simulator figures are drawn with longer arrows than the measured ones, for readability.
SIMULATOR_FILL = 1.3
# Margin beyond everything drawn, as a share of the drawn extent.
MARGIN = 0.03
# The reference arrow is drawn at about this share of the panel width in every figure, with its length rounded to this
# many significant digits.
REFERENCE_SHARE = 0.1
REFERENCE_DIGITS = 2
TARGET_COLOR = "0.45"
ECCENTRIC_TARGET_MARKER = {"marker": "o", "markersize": 1.6, "color": TARGET_COLOR}
CENTER_TARGET_MARKER = {"marker": "+", "markersize": 5, "markeredgewidth": 0.8, "color": TARGET_COLOR}

type Differences = dict[tuple[str, str, str, int], np.ndarray]


def draw_panel(axis: Axes, differences: Differences, colors: dict[str, str], eye: str, scale: float) -> None:
    """Every participant's difference arrow at each eccentric target, for one eye."""
    for target in ECCENTRIC_TARGETS:
        axis.plot(*target_position(target, LAYOUT_SCALE), zorder=3, **ECCENTRIC_TARGET_MARKER)
    for (_camera, participant, e, target), difference in differences.items():
        if e != eye:
            continue
        origin = target_position(target, LAYOUT_SCALE)
        arrow(axis, origin, origin + difference * scale, colors[participant], linewidth=0.9, alpha=0.85, head_size=6)
    axis.plot(*target_position(CENTER_TARGET, LAYOUT_SCALE), zorder=3, **CENTER_TARGET_MARKER)


def on_camera(differences: Differences, camera: str) -> Differences:
    """The differences recorded or simulated for one camera."""
    return {key: difference for key, difference in differences.items() if key[0] == camera}


def arrow_scale(differences: Differences, fill: float) -> tuple[float, float, tuple[float, float]]:
    """(arrow scale, reference arrow length in mm per mm, panel span) that fit every arrow in ``differences``."""
    magnitudes = [np.linalg.norm(difference) for difference in differences.values()]
    typical_magnitude = float(np.percentile(magnitudes, SCALE_PERCENTILE))
    scale = fill * smallest_target_gap(LAYOUT_SCALE) / typical_magnitude
    drawn = np.array(
        [target_position(target, LAYOUT_SCALE) for target in SCREEN_POSITIONS]
        + [
            target_position(target, LAYOUT_SCALE) + difference * scale
            for (_c, _p, _e, target), difference in differences.items()
        ]
    )
    span = panel_span(drawn, MARGIN)
    reference = reference_arrow_for_width(2 * span[0], scale, REFERENCE_SHARE, REFERENCE_DIGITS)
    return scale, reference, span


def write_figure(
    differences: Differences,
    colors: dict[str, str],
    camera: str,
    scaling: tuple[float, float, tuple[float, float]],
    name: str,
    ratio: str,
) -> None:
    """One camera's figure, a panel per eye, with ``ratio`` in the legend, written to output/<name>_<camera>.png."""
    scale, reference, span = scaling
    fig = Figure(figsize=(9.5, 4.8))
    axes = fig.subplots(1, len(EYES), sharex=True, sharey=True, squeeze=False)
    for column, eye in enumerate(EYES):
        draw_panel(axes[0][column], differences, colors, eye, scale)
    style_panels(axes[0], EYES, span)
    draw_reference_arrow(axes[0][0], reference, scale, span, REFERENCE_DIGITS)
    # A panel's height follows from its data and its equal aspect, so the panels are held at the bottom of their slot,
    # right above the legend.
    for axis in axes[0]:
        axis.set_anchor("S")
    legend_below(
        fig,
        [
            LegendArrow(THIN),
            Line2D([], [], linestyle="None", **ECCENTRIC_TARGET_MARKER),
            Line2D([], [], linestyle="None", **CENTER_TARGET_MARKER),
        ],
        [difference_label(ratio), "Eccentric target", "Center target"],
        handlelength=2.5,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.927, bottom=0.085, wspace=0.05)
    path = OUTPUT / f"{name}_{CAMERA_FILE_NAME[camera]}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"wrote {path}: reference arrow {reference:#.{REFERENCE_DIGITS}g} mm per mm")


def main() -> None:
    """Draw the measured or every simulator's differences from the center target, one figure per camera."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulators",
        action="store_true",
        help="draw every simulator figure instead of the measured one",
    )
    args = parser.parse_args()

    measured = load_measured_apparent_pupil_center_shift()
    # Colored over every recorded participant, so a participant a run leaves out does not shift the others' colors.
    colors = participant_colors({participant for _camera, participant, _eye, _target in measured})
    measured_differences = {
        camera: on_camera(difference_from_center_target(measured), camera) for camera in PANEL_CAMERAS
    }
    measured_scaling = {
        camera: arrow_scale(differences, MEASURED_FILL) for camera, differences in measured_differences.items()
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not args.simulators:
        for camera, differences in measured_differences.items():
            write_figure(differences, colors, camera, measured_scaling[camera], NAME, MEASURED_RATIO)
        return

    for figure, simulator, runs in SIMULATOR_FIGURES:
        shift = {}
        for camera in PANEL_CAMERAS:
            shift.update(load_simulator_result_file(simulator, camera, runs[camera]))
        differences = difference_from_center_target(shift)
        for camera in PANEL_CAMERAS:
            camera_differences = on_camera(differences, camera)
            scaling = arrow_scale(camera_differences, SIMULATOR_FILL)
            write_figure(camera_differences, colors, camera, scaling, f"{NAME}_{figure}", SIMULATED_RATIO)
            print(f"  arrow scale {scaling[0] / measured_scaling[camera][0]:.1f} times that of the measured figure")


if __name__ == "__main__":
    main()
