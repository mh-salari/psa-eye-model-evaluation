"""Figures of the apparent pupil center shift at the center target: panels per eye, the arrows and their legend."""

from dataclasses import dataclass
from typing import override

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Transform

from psa_eye_model_evaluation.experiment_setup import EYE_LABEL, EYES
from psa_eye_model_evaluation.target_grid_figure import MEASURED_RATIO

# Fraction of each panel's span added as margin beyond both the data and zero.
PADDING = 0.08
X_LABEL = rf"Horizontal ${MEASURED_RATIO}$ (mm per mm)"
Y_LABEL = rf"Vertical ${MEASURED_RATIO}$ (mm per mm)"
MEASURED = "measured"
THIN = "thin"
THIN_DASHED = "thin_dashed"
# Line style per kind of arrow as (offset, (on, off, ...) in points), spaced so the patterns stay apart at thin widths.
LINESTYLES = {
    MEASURED: "-",
    THIN: "-",
    THIN_DASHED: (0, (3, 2)),
    "pyetsimul": (0, (5, 4)),
    "et_simul": (0, (1, 2)),
    "gkamodeleye": (0, (5, 2.5, 1, 2.5)),
}
LINEWIDTHS = {MEASURED: 1.4, THIN: 0.9, THIN_DASHED: 0.9, "pyetsimul": 1.0, "et_simul": 1.0, "gkamodeleye": 1.0}
# matplotlib arrowhead per kind of arrow: a filled triangle for the measurement and the thin arrows; an open V, a
# hollow triangle and a swept head for the simulators. Head dimensions are fractions of HEAD_SIZE.
HEAD_STYLES = {
    MEASURED: "-|>,head_length=0.8,head_width=0.3",
    THIN: "-|>,head_length=0.35,head_width=0.18",
    THIN_DASHED: "-|>,head_length=0.35,head_width=0.18",
    "pyetsimul": "->,head_length=0.7,head_width=0.35",
    "et_simul": "-|>,head_length=0.8,head_width=0.3",
    "gkamodeleye": "fancy,head_length=0.8,head_width=0.4,tail_width=0.15",
}
HOLLOW_HEAD = frozenset({"et_simul"})
HEAD_SIZE = 7.0
# The head is drawn as its own solid patch over the last part of the shaft, so the dash pattern never breaks it up
# and the shaft does not show inside a hollow head.
HEAD_SHAFT_FRACTION = 0.02
LEGEND_COLOR = "0.25"


def head_patch(
    start: tuple[float, float], end: tuple[float, float], color: str, kind: str, alpha: float, joinstyle: str
) -> FancyArrowPatch:
    """A patch from ``start`` to ``end`` with the head of ``kind`` at ``end``."""
    return FancyArrowPatch(
        start,
        end,
        arrowstyle=HEAD_STYLES[kind],
        mutation_scale=HEAD_SIZE,
        linewidth=LINEWIDTHS[kind],
        edgecolor=color,
        facecolor="white" if kind in HOLLOW_HEAD else color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        joinstyle=joinstyle,
    )


def arrow_artists(
    start: tuple[float, float], end: tuple[float, float], color: str, kind: str, alpha: float
) -> list[Artist]:
    """One arrow: a single patch if solid, otherwise a shaft in the dash pattern of its kind and a solid head."""
    if LINESTYLES[kind] == "-":
        # A separate solid shaft would reach the tip at full width and show past the narrowing head. The miter join
        # brings the tip of the single patch to ``end``.
        return [head_patch(start, end, color, kind, alpha, joinstyle="miter")]
    (x0, y0), (x1, y1) = start, end
    shaft = Line2D(
        [x0, x1],
        [y0, y1],
        color=color,
        linestyle=LINESTYLES[kind],
        linewidth=LINEWIDTHS[kind],
        alpha=alpha,
        solid_capstyle="butt",
    )
    head_start = (x1 - HEAD_SHAFT_FRACTION * (x1 - x0), y1 - HEAD_SHAFT_FRACTION * (y1 - y0))
    return [shaft, head_patch(head_start, end, color, kind, alpha, joinstyle="round")]


def add_arrow(
    axis: Axes, start: tuple[float, float], end: tuple[float, float], color: str, kind: str, alpha: float = 1.0
) -> None:
    """Draw one arrow on ``axis``."""
    for artist in arrow_artists(start, end, color, kind, alpha):
        axis.add_artist(artist)


@dataclass(frozen=True)
class LegendArrow:
    """A legend entry drawn by ArrowHandler as an arrow of ``kind`` in ``color``."""

    kind: str
    color: str = LEGEND_COLOR


class ArrowHandler(HandlerBase):
    """Draws a LegendArrow entry as the same arrow as in the panels, spanning the entry's width."""

    @override
    def create_artists(
        self,
        legend: Legend,
        orig_handle: LegendArrow,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> list[Artist]:
        """The entry's arrow, placed in the legend's coordinates."""
        del legend, fontsize
        start = (xdescent, ydescent + height / 2)
        end = (xdescent + width, ydescent + height / 2)
        artists = arrow_artists(start, end, orig_handle.color, orig_handle.kind, alpha=1.0)
        for artist in artists:
            artist.set_transform(trans)
        return artists


def legend_below(
    fig: Figure, handles: list[Artist | LegendArrow], labels: list[str], handlelength: float = 4.0
) -> None:
    """A frameless one-row legend centered below the panels, with handles ``handlelength`` font sizes long."""
    fig.legend(
        handles=handles,
        labels=labels,
        handler_map={LegendArrow: ArrowHandler()},
        handlelength=handlelength,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )


def padded_range(low: float, high: float) -> tuple[float, float]:
    """The range from ``low`` to ``high``, extended to include zero and padded by PADDING of its span."""
    low, high = min(0.0, low), max(0.0, high)
    margin = PADDING * (high - low)
    return low - margin, high + margin


def eye_panels(fig: Figure, tips: dict[str, np.ndarray]) -> dict[str, Axes]:
    """{eye: panel}, one panel per eye that holds that eye's arrow tips (an n x 2 array) and the origin.

    The panels share one vertical range, so the origin sits at the same height in both, and each panel's width is
    proportional to its own horizontal range, so the horizontal scale is the same in both.
    """
    y_range = padded_range(min(t[:, 1].min() for t in tips.values()), max(t[:, 1].max() for t in tips.values()))
    x_ranges = {eye: padded_range(tips[eye][:, 0].min(), tips[eye][:, 0].max()) for eye in EYES}
    axes = fig.subplots(1, len(EYES), gridspec_kw={"width_ratios": [high - low for low, high in x_ranges.values()]})
    panels = {}
    for axis, eye in zip(axes, EYES, strict=True):
        axis.axhline(0, lw=0.4, color="0.75")
        axis.axvline(0, lw=0.4, color="0.75")
        axis.set_xlim(*x_ranges[eye])
        axis.set_ylim(*y_range)
        axis.grid(True, lw=0.3, color="0.9")
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
        axis.set_title(EYE_LABEL[eye], fontsize=10)
        axis.set_xlabel(X_LABEL, fontsize=9)
        panels[eye] = axis
    axes[0].set_ylabel(Y_LABEL, fontsize=9)
    # Both panels share one vertical range, so only the leftmost keeps its tick labels.
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    return panels
