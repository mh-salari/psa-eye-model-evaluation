"""How the apparent pupil center shift changes from the center target to each eccentric target."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from psa_eye_model_evaluation.experiment_setup import (
    CENTER_TARGET,
    ECCENTRIC_TARGETS,
    EYES,
    SCREEN_POSITIONS,
    TARGET_LABELS,
    nasal_or_temporal,
)

# A difference less than this angle from the direction to the center target has a component pointing back toward it.
TOWARD_CENTER_TARGET_DEG = 90.0


@dataclass(frozen=True)
class ChangeSummary:
    """The change from the center target for one camera and eye, over participants.

    median_direction_change               median direction change (deg) over every participant and eccentric target
    direction_change_mad                  median absolute deviation (deg) of the same values
    median_direction_change_by_target     {eccentric target: median direction change (deg) over participants}
    median_magnitude_change_by_target     {eccentric target: median magnitude change (mm per mm) over participants}
    nasal_magnitude_change                median magnitude change (mm per mm) over every participant and nasal target
    nasal_magnitude_change_mad            median absolute deviation (mm per mm) of the same values
    temporal_magnitude_change             median magnitude change (mm per mm) over every participant and temporal target
    temporal_magnitude_change_mad         median absolute deviation (mm per mm) of the same values
    nasal_magnitude_change_percent        median magnitude change (% of the center-target magnitude) over every
                                          participant and nasal target
    temporal_magnitude_change_percent     the same over every participant and temporal target
    n_eyes                                number of eyes summarized
    """

    median_direction_change: float
    direction_change_mad: float
    median_direction_change_by_target: dict[int, float]
    median_magnitude_change_by_target: dict[int, float]
    nasal_magnitude_change: float
    nasal_magnitude_change_mad: float
    temporal_magnitude_change: float
    temporal_magnitude_change_mad: float
    nasal_magnitude_change_percent: float
    temporal_magnitude_change_percent: float
    n_eyes: int

    @property
    def direction_change_range(self) -> tuple[float, float]:
        """(min, max) across the eccentric targets of the per-target median direction change."""
        values = self.median_direction_change_by_target.values()
        return min(values), max(values)

    @property
    def magnitude_change_range(self) -> tuple[float, float]:
        """(min, max) across the eccentric targets of the per-target median magnitude change."""
        values = self.median_magnitude_change_by_target.values()
        return min(values), max(values)


def angle_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Unsigned angle (deg) between two shift vectors."""
    cosine = first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def median_absolute_deviation(values: list[float]) -> float:
    """Median of the absolute deviations from the median, unscaled."""
    array = np.asarray(values, dtype=float)
    return float(np.median(np.abs(array - np.median(array))))


def difference_from_center_target(
    shift: dict[tuple[str, str, str, int], np.ndarray],
) -> dict[tuple[str, str, str, int], np.ndarray]:
    """{(camera, participant, eye, eccentric target): shift at that target minus shift at the center target}."""
    return {
        (camera, participant, eye, target): vector - shift[camera, participant, eye, CENTER_TARGET]
        for (camera, participant, eye, target), vector in shift.items()
        if target != CENTER_TARGET
    }


def angle_to_center_target_deg(difference: np.ndarray, target: int) -> float:
    """Angle (deg) between a shift difference and the direction from the target's screen position to the center's.

    The shift is in screen axes, so the direction on the screen and the direction of the difference compare directly.
    """
    to_center = np.array(SCREEN_POSITIONS[CENTER_TARGET], dtype=float) - np.array(
        SCREEN_POSITIONS[target], dtype=float
    )
    return angle_between_deg(difference, to_center)


def share_toward_center_target(shift: dict[tuple[str, str, str, int], np.ndarray], camera: str, eye: str) -> float:
    """Percentage of one camera and eye's differences from the center target that point back toward it."""
    angles = [
        angle_to_center_target_deg(difference, target)
        for (c, _participant, e, target), difference in difference_from_center_target(shift).items()
        if (c, e) == (camera, eye)
    ]
    if not angles:
        raise ValueError(f"no {eye} on {camera} in the shift")
    return float(np.mean(np.array(angles) < TOWARD_CENTER_TARGET_DEG) * 100)


class Change(NamedTuple):
    """How one eye's shift at one eccentric target differs from its shift at the center target.

    direction_change            angle between the two shifts (deg)
    magnitude_change            eccentric-target shift magnitude minus center-target shift magnitude (mm per mm)
    magnitude_change_percent    the magnitude change as a percentage of the center-target shift magnitude
    """

    direction_change: float
    magnitude_change: float
    magnitude_change_percent: float


def changes_from_center(vectors: dict[int, np.ndarray]) -> dict[int, Change]:
    """{eccentric target: change from the center target} for one eye's shift vectors at all nine targets.

    The vectors are the apparent pupil center shift per millimeter of apparent pupil diameter change (mm per mm).
    """
    center = vectors[CENTER_TARGET]
    center_magnitude = float(np.linalg.norm(center))
    changes = {}
    for target in ECCENTRIC_TARGETS:
        magnitude_change = float(np.linalg.norm(vectors[target])) - center_magnitude
        changes[target] = Change(
            direction_change=angle_between_deg(center, vectors[target]),
            magnitude_change=magnitude_change,
            magnitude_change_percent=100 * magnitude_change / center_magnitude,
        )
    return changes


def summarize_changes(per_participant: dict[str, dict[int, Change]], eye: str) -> ChangeSummary:
    """Summary of ``changes_from_center`` over the participants of one camera and eye."""
    if not per_participant:
        raise ValueError(f"no participants to summarize for {eye}")
    participants = list(per_participant.values())
    direction = {
        target: [changes[target].direction_change for changes in participants] for target in ECCENTRIC_TARGETS
    }
    magnitude = {
        target: [changes[target].magnitude_change for changes in participants] for target in ECCENTRIC_TARGETS
    }
    by_side: dict[str, dict[str, list[float]]] = {
        side: {"mm_per_mm": [], "percent": []} for side in ("nasal", "temporal")
    }
    for target in ECCENTRIC_TARGETS:
        side = nasal_or_temporal(eye, target)
        if side is not None:
            by_side[side]["mm_per_mm"].extend(changes[target].magnitude_change for changes in participants)
            by_side[side]["percent"].extend(changes[target].magnitude_change_percent for changes in participants)
    all_directions = [value for values in direction.values() for value in values]
    return ChangeSummary(
        median_direction_change=float(np.median(all_directions)),
        direction_change_mad=median_absolute_deviation(all_directions),
        median_direction_change_by_target={target: float(np.median(values)) for target, values in direction.items()},
        median_magnitude_change_by_target={target: float(np.median(values)) for target, values in magnitude.items()},
        nasal_magnitude_change=float(np.median(by_side["nasal"]["mm_per_mm"])),
        nasal_magnitude_change_mad=median_absolute_deviation(by_side["nasal"]["mm_per_mm"]),
        temporal_magnitude_change=float(np.median(by_side["temporal"]["mm_per_mm"])),
        temporal_magnitude_change_mad=median_absolute_deviation(by_side["temporal"]["mm_per_mm"]),
        nasal_magnitude_change_percent=float(np.median(by_side["nasal"]["percent"])),
        temporal_magnitude_change_percent=float(np.median(by_side["temporal"]["percent"])),
        n_eyes=len(participants),
    )


def summarize_changes_by_eye(
    shift: dict[tuple[str, str, str, int], np.ndarray], camera: str
) -> dict[str, ChangeSummary]:
    """{eye: summary} for one camera, from shift vectors keyed (camera, participant, eye, target)."""
    summaries = {}
    for eye in EYES:
        participants = sorted({participant for c, participant, e, _target in shift if c == camera and e == eye})
        per_participant = {}
        for participant in participants:
            vectors = {target: shift[camera, participant, eye, target] for target in TARGET_LABELS}
            per_participant[participant] = changes_from_center(vectors)
        summaries[eye] = summarize_changes(per_participant, eye)
    return summaries
