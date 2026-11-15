"""Figures that draw data at the nine target positions: layout, arrows, the reference arrow and the panel style."""

import numpy as np
from matplotlib.axes import Axes

from psa_eye_model_evaluation.experiment_setup import EYE_LABEL, SCREEN_POSITIONS

REFERENCE_ARROW_STEPS = (1.0, 2.0, 5.0)
REFERENCE_ARROW_COLOR = "0.2"
# The relative apparent pupil center shift as measured and as a simulator produces it, in mathtext without the $ signs.
MEASURED_RATIO = r"\Delta\mathbf{PC}^{*}/\Delta\mathrm{PD}^{*}"
SIMULATED_RATIO = r"\Delta\mathbf{PC}^{*}_{\mathrm{sim}}/\Delta\mathrm{PD}^{*}_{\mathrm{sim}}"
CENTER_TARGET_SUBSCRIPT = r"_{\mathrm{center\;target}}"
ECCENTRIC_TARGET_SUBSCRIPT = r"_{\mathrm{eccentric\;target}}"


def center_target_label(ratio: str) -> str:
    """Legend text for ``ratio`` at the center target."""
    return f"$({ratio}){CENTER_TARGET_SUBSCRIPT}$"


def eccentric_target_label(ratio: str) -> str:
    """Legend text for ``ratio`` at an eccentric target."""
    return f"$({ratio}){ECCENTRIC_TARGET_SUBSCRIPT}$"


def difference_label(ratio: str) -> str:
    """Legend text for ``ratio`` at an eccentric target minus ``ratio`` at the center target."""
    return f"$({ratio}){ECCENTRIC_TARGET_SUBSCRIPT} - ({ratio}){CENTER_TARGET_SUBSCRIPT}$"


def target_position(target: int, layout_scale: float = 1.0) -> np.ndarray:
    """Where a target is drawn: its screen position in mm, multiplied by ``layout_scale``."""
    return layout_scale * np.array(SCREEN_POSITIONS[target], dtype=float)


def smallest_target_gap(layout_scale: float = 1.0) -> float:
    """The shortest distance between any two targets as drawn."""
    positions = [target_position(target, layout_scale) for target in SCREEN_POSITIONS]
    return min(
        float(np.linalg.norm(first - second))
        for index, first in enumerate(positions)
        for second in positions[index + 1 :]
    )


def reference_arrow_length(value: float) -> float:
    """The largest 1, 2 or 5 times a power of ten that is at most ``value``."""
    if value <= 0:
        raise ValueError(f"reference arrow length needs a positive value, got {value}")
    power = 10.0 ** np.floor(np.log10(value))
    return float(max(step * power for step in REFERENCE_ARROW_STEPS if step * power <= value))


def reference_arrow_for_width(panel_width: float, scale: float, share: float, digits: int) -> float:
    """The reference arrow length (mm per mm) whose drawn length is closest to ``share`` of ``panel_width``.

    Rounded to ``digits`` significant digits, so figures drawn at different scales keep about the same drawn reference
    arrow and differ in its label.
    """
    value = share * panel_width / scale
    if value <= 0:
        raise ValueError(f"reference arrow length needs a positive value, got {value}")
    return float(round(value, digits - 1 - int(np.floor(np.log10(value)))))


def panel_span(points: np.ndarray, margin: float) -> tuple[float, float]:
    """Half-widths (x, y) of centered panels that hold every point, plus ``margin`` of the larger extent."""
    padding = margin * max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    return float(np.abs(points[:, 0]).max()) + padding, float(np.abs(points[:, 1]).max()) + padding


def arrow(
    axis: Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    *,
    linestyle: str | tuple = "-",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    head_size: float = 7.0,
) -> None:
    """One arrow from ``start`` to ``end``."""
    axis.annotate(
        "",
        xy=tuple(end),
        xytext=tuple(start),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": linewidth,
            "linestyle": linestyle,
            "color": color,
            "alpha": alpha,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": head_size,
        },
    )


def draw_reference_arrow(
    axis: Axes, length: float, scale: float, span: tuple[float, float], digits: int | None = None
) -> None:
    """A horizontal arrow of ``length`` mm per mm in the lower left corner of the panel, labeled with its length.

    With ``digits``, the label shows that many significant digits, trailing zeros included.
    """
    span_x, span_y = span
    start = np.array([-0.94 * span_x, -0.92 * span_y])
    arrow(axis, start, start + np.array([length * scale, 0.0]), REFERENCE_ARROW_COLOR)
    label = f"{length:#.{digits}g}" if digits else f"{length:g}"
    axis.text(start[0], start[1] + 0.05 * span_y, f"Scale: {label} mm per mm", fontsize=8, color=REFERENCE_ARROW_COLOR)


def style_panels(axes: np.ndarray, eyes: tuple[str, ...], span: tuple[float, float]) -> None:
    """One eye per panel: shared limits, equal aspect, no ticks, and the eye as the panel title."""
    span_x, span_y = span
    for axis, eye in zip(axes, eyes, strict=True):
        axis.set_xlim(-span_x, span_x)
        axis.set_ylim(-span_y, span_y)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
        axis.set_title(EYE_LABEL[eye], fontsize=10)
