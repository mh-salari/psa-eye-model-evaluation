"""How the apparent pupil center shift difference is built from the two shifts, for one participant.

    uv run python measured/plot_apparent_pupil_center_shift_difference_construction.py
    uv run python measured/plot_apparent_pupil_center_shift_difference_construction.py --participant 346

One figure per camera and one panel per eye. At every target it draws three arrows from the target's screen position:
the eye's apparent pupil center shift at the center target (dashed), its shift at that target, and the difference
between them, drawn from the tip of the first to the tip of the second. All arrows in a figure share one scale, set by
that camera's longest arrow, so the difference is seen at its magnitude relative to the shift.

Without --participant it draws the participant whose median center-target shift magnitude, over their eyes and
cameras, is closest to the median over participants. With --simulator it draws that simulator's shift for the same
participant.

Writes output/apparent_pupil_center_shift_difference_construction_<participant>_<camera>.png, <camera> being
eye_level_camera or bottom_left_camera, with _<simulator> before the camera for --simulator, and prints, per camera
and eye, the range of the direction change and at how many of the nasal targets the shift's magnitude increased and
at how many of the temporal targets it decreased.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from psa_eye_model_evaluation.apparent_pupil_center_shift_change import changes_from_center
from psa_eye_model_evaluation.center_target_figure import LINESTYLES, THIN, THIN_DASHED, LegendArrow, legend_below
from psa_eye_model_evaluation.experiment_setup import (
    CAMERA_FILE_NAME,
    CAMERA_LABEL,
    CENTER_TARGET,
    EYE_LABEL,
    EYES,
    PANEL_CAMERAS,
    SCREEN_POSITIONS,
    TARGET_LABELS,
    nasal_or_temporal,
)
from psa_eye_model_evaluation.recordings import load_measured_apparent_pupil_center_shift
from psa_eye_model_evaluation.simulator_results import SIMULATORS, load_simulated_apparent_pupil_center_shift
from psa_eye_model_evaluation.target_grid_figure import (
    MEASURED_RATIO,
    SIMULATED_RATIO,
    arrow,
    center_target_label,
    difference_label,
    draw_reference_arrow,
    eccentric_target_label,
    panel_span,
    reference_arrow_length,
    smallest_target_gap,
    style_panels,
    target_position,
)

OUTPUT = Path(__file__).resolve().parent / "output"
# The longest arrow reaches this share of the smallest gap between two targets.
FILL = 0.85
# Margin beyond everything drawn, as a share of the drawn extent.
MARGIN = 0.05
CENTER_TARGET_SHIFT_COLOR = "0.45"
SHIFT_COLOR = "#0057E7"
DIFFERENCE_COLOR = "#E4002B"


def typical_participant(shift: dict[tuple[str, str, str, int], np.ndarray]) -> str:
    """The participant whose median center-target shift magnitude is closest to the median over participants."""
    magnitudes: dict[str, list[float]] = {}
    for (_camera, participant, _eye, target), vector in shift.items():
        if target == CENTER_TARGET:
            magnitudes.setdefault(participant, []).append(float(np.linalg.norm(vector)))
    per_participant = {participant: float(np.median(values)) for participant, values in magnitudes.items()}
    overall = float(np.median(list(per_participant.values())))
    return min(per_participant, key=lambda participant: abs(per_participant[participant] - overall))


def label_side(vectors: dict[int, np.ndarray], target: int) -> int:
    """+1 to put a target's label right of its dot, -1 to put it left: the side its arrows do not lean toward."""
    drawn = vectors[CENTER_TARGET] if target == CENTER_TARGET else vectors[CENTER_TARGET] + vectors[target]
    return -1 if drawn[0] > 0 else 1


def draw_panel(axis: Axes, vectors: dict[int, np.ndarray], scale: float) -> None:
    """The center-target shift, the shift and their difference at every target, for one camera and eye."""
    label_offset = 0.06 * smallest_target_gap()
    for target in SCREEN_POSITIONS:
        origin = target_position(target)
        axis.plot(*origin, marker="o", markersize=1.8, color="0.3", zorder=4)
        side = label_side(vectors, target)
        axis.text(
            origin[0] + side * label_offset,
            origin[1],
            TARGET_LABELS[target],
            fontsize=7,
            color="0.35",
            ha="left" if side > 0 else "right",
            va="center",
        )
        center_target_tip = origin + vectors[CENTER_TARGET] * scale
        arrow(axis, origin, center_target_tip, CENTER_TARGET_SHIFT_COLOR, linestyle=LINESTYLES[THIN_DASHED])
        if target == CENTER_TARGET:
            continue
        tip = origin + vectors[target] * scale
        arrow(axis, origin, tip, SHIFT_COLOR)
        arrow(axis, center_target_tip, tip, DIFFERENCE_COLOR)


def print_participant_changes(camera: str, per_eye: dict[str, dict[int, np.ndarray]]) -> None:
    """Per eye, the direction change range and how often the magnitude increased nasally and decreased temporally."""
    for eye, vectors in per_eye.items():
        changes = changes_from_center(vectors)
        directions = [change.direction_change for change in changes.values()]
        counts = {"nasal": [0, 0], "temporal": [0, 0]}
        for target, change in changes.items():
            side = nasal_or_temporal(eye, target)
            if side is None:
                continue
            expected = change.magnitude_change > 0 if side == "nasal" else change.magnitude_change < 0
            counts[side][0] += int(expected)
            counts[side][1] += 1
        print(
            f"{CAMERA_LABEL[camera]}, {EYE_LABEL[eye]}: direction change {min(directions):.1f} to "
            f"{max(directions):.1f} deg over {len(directions)} eccentric targets; magnitude increased at "
            f"{counts['nasal'][0]} of {counts['nasal'][1]} nasal targets, decreased at {counts['temporal'][0]} of "
            f"{counts['temporal'][1]} temporal targets"
        )


def construction_figure(
    per_eye: dict[str, dict[int, np.ndarray]],
    scale: float,
    span: tuple[float, float],
    reference: float,
    ratio: str,
) -> Figure:
    """One camera's figure: a panel per eye (marked when not recorded), the reference arrow and ``ratio``'s legend."""
    fig = Figure(figsize=(10.0, 4.6))
    axes = fig.subplots(1, len(EYES), sharex=True, sharey=True, squeeze=False)
    for column, eye in enumerate(EYES):
        if eye in per_eye:
            draw_panel(axes[0][column], per_eye[eye], scale)
        else:
            axes[0][column].text(0, 0, "not recorded", fontsize=8, color="0.5", ha="center", va="center")
    style_panels(axes[0], EYES, span)
    draw_reference_arrow(axes[0][0], reference, scale, span)
    legend_below(
        fig,
        [
            LegendArrow(THIN_DASHED, CENTER_TARGET_SHIFT_COLOR),
            LegendArrow(THIN, SHIFT_COLOR),
            LegendArrow(THIN, DIFFERENCE_COLOR),
        ],
        [center_target_label(ratio), eccentric_target_label(ratio), difference_label(ratio)],
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.12, wspace=0.05)
    return fig


def main() -> None:
    """Draw how the difference from the center target is built, for one participant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", help="participant to draw (default: the most typical one)")
    parser.add_argument(
        "--simulator",
        choices=sorted(SIMULATORS),
        help="draw this simulator's apparent pupil center shift instead of the measured one",
    )
    args = parser.parse_args()

    shift = (
        load_simulated_apparent_pupil_center_shift(args.simulator)
        if args.simulator
        else load_measured_apparent_pupil_center_shift()
    )
    participant = args.participant or typical_participant(shift)
    if participant not in {p for _c, p, _e, _t in shift}:
        raise SystemExit(f"participant {participant} is not in the recordings")

    suffix = f"_{args.simulator}" if args.simulator else ""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for camera in PANEL_CAMERAS:
        per_eye = {}
        for eye in EYES:
            vectors = {t: v for (c, p, e, t), v in shift.items() if (c, p, e) == (camera, participant, eye)}
            if vectors:
                per_eye[eye] = vectors
        if not per_eye:
            print(f"{CAMERA_LABEL[camera]}: participant {participant} was not recorded, no figure written")
            continue
        print_participant_changes(camera, per_eye)

        longest = max(float(np.linalg.norm(v)) for vectors in per_eye.values() for v in vectors.values())
        scale = FILL * smallest_target_gap() / longest
        reference = reference_arrow_length(longest)
        drawn = np.array(
            [target_position(target) for target in SCREEN_POSITIONS]
            + [target_position(target) + vectors[target] * scale for vectors in per_eye.values() for target in vectors]
            + [
                target_position(target) + vectors[CENTER_TARGET] * scale
                for vectors in per_eye.values()
                for target in SCREEN_POSITIONS
            ]
        )
        span = panel_span(drawn, MARGIN)

        ratio = SIMULATED_RATIO if args.simulator else MEASURED_RATIO
        fig = construction_figure(per_eye, scale, span, reference, ratio)
        name = f"apparent_pupil_center_shift_difference_construction_{participant}{suffix}_{CAMERA_FILE_NAME[camera]}"
        path = OUTPUT / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}: reference arrow {reference:g} mm per mm")


if __name__ == "__main__":
    main()
