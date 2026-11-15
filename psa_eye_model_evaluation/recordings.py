"""The measured data: P-CR and pupil diameter per phase, their condition means, and the apparent pupil center shift.

Everything comes from data/raw_eye_features.csv's per-phase rows. For each eye and target, the P-CR vector (pupil
center minus corneal reflection) and the pupil diameter are averaged over the three phases of each condition. The
apparent pupil center shift per millimeter of pupil diameter change at a target is

    IMAGE_TO_SCREEN_AXES * (dilated mean P-CR - constricted mean P-CR)
        / (dilated mean diameter - constricted mean diameter)

with both differences taken at that target.

The P-CR vector stands in for the pupil center after corneal-reflection alignment: aligning every phase's corneal
reflection onto a reference phase's and then differencing two phases is algebraically identical to differencing their
P-CR vectors, because the reference position cancels.
"""

import csv
import math
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np

from psa_eye_model_evaluation.experiment_setup import TARGET_LABELS

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_eye_features.csv"

# Image x already increases toward screen right (the P-CR moves to +x as the eye looks right). Image y increases
# downward and screen y upward, so only the vertical component is negated.
IMAGE_TO_SCREEN_AXES = np.array([1.0, -1.0])

type Phases = dict[tuple[str, str, str, int, str, str], np.ndarray]
type Means = dict[tuple[str, str, str, int, str], np.ndarray]


def load_phases() -> Phases:
    """{(camera, participant, eye, target, condition, phase): (pcr_x, pcr_y, pupil diameter)} in mm, per phase."""
    phases: Phases = {}
    with RAW.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["estimator"] != "phase":
                continue
            key = (row["camera"], row["participant"], row["eye"], int(row["target"]), row["state"], row["phase"])
            if key in phases:
                raise ValueError(f"more than one row for {key}")
            phases[key] = np.array([float(row["pcr_mm_x"]), float(row["pcr_mm_y"]), float(row["pupil_diameter_mm"])])
    return phases


@cache
def camera_roll_deg(camera: str) -> float:
    """The camera's roll about its view axis (deg), as seen in its images.

    The mean angle, in the camera's image axes, of the line from the left eye's corneal reflection to the right eye's,
    over every per-phase row in which both eyes were recorded.
    """
    reflections: dict[tuple[str, str, str, str], np.ndarray] = {}
    with RAW.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["estimator"] == "phase" and row["camera"] == camera:
                key = (row["participant"], row["target"], row["phase"], row["eye"])
                reflections[key] = np.array([float(row["glint_px_x"]), float(row["glint_px_y"])])
    angles = []
    for (participant, target, phase, eye), left in reflections.items():
        right = reflections.get((participant, target, phase, "right_eye"))
        if eye == "left_eye" and right is not None:
            dx, dy = right - left
            angles.append(math.degrees(math.atan2(dy, dx)))
    if not angles:
        raise ValueError(f"no rows with both eyes' corneal reflections for {camera}")
    return float(np.mean(angles))


def condition_means(phases: Phases) -> Means:
    """{(camera, participant, eye, target, condition): (pcr_x, pcr_y, pupil diameter)}, averaged over the phases."""
    per_condition: dict[tuple[str, str, str, int, str], list[np.ndarray]] = defaultdict(list)
    for (camera, participant, eye, target, condition, _phase), values in phases.items():
        per_condition[camera, participant, eye, target, condition].append(values)
    return {key: np.mean(values, axis=0) for key, values in per_condition.items()}


def nine_target_mean_apparent_pupil_diameters(means: Means) -> dict[tuple[str, str, str, str], float]:
    """{(camera, participant, eye, condition): pupil diameter (mm)}, averaged over the nine targets."""
    per_eye: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (camera, participant, eye, _target, condition), values in means.items():
        per_eye[camera, participant, eye, condition].append(float(values[2]))
    for key, diameters in per_eye.items():
        if len(diameters) != len(TARGET_LABELS):
            raise ValueError(f"{key} has {len(diameters)} targets, expected {len(TARGET_LABELS)}")
    return {key: float(np.mean(diameters)) for key, diameters in per_eye.items()}


def recorded_eyes(means: Means) -> list[tuple[str, str, str]]:
    """Every (camera, participant, eye) in the recordings."""
    return sorted({(camera, participant, eye) for camera, participant, eye, _target, _condition in means})


def apparent_pupil_center_shift_mm(means: Means, camera: str, participant: str, eye: str, target: int) -> np.ndarray:
    """Dilated mean P-CR minus constricted mean P-CR (mm) at one target, in the camera's image axes."""
    dilated = means[camera, participant, eye, target, "dilated"][:2]
    return dilated - means[camera, participant, eye, target, "constricted"][:2]


def apparent_pupil_diameter_change_mm(means: Means, camera: str, participant: str, eye: str, target: int) -> float:
    """Dilated mean pupil diameter minus constricted mean pupil diameter (mm) at one target."""
    dilated = means[camera, participant, eye, target, "dilated"][2]
    return float(dilated - means[camera, participant, eye, target, "constricted"][2])


def load_measured_apparent_pupil_center_shift() -> dict[tuple[str, str, str, int], np.ndarray]:
    """{(camera, participant, eye, target): apparent pupil center shift per mm of pupil diameter change}."""
    means = condition_means(load_phases())
    out: dict[tuple[str, str, str, int], np.ndarray] = {}
    for eye in recorded_eyes(means):
        for target in TARGET_LABELS:
            shift = IMAGE_TO_SCREEN_AXES * apparent_pupil_center_shift_mm(means, *eye, target)
            out[*eye, target] = shift / apparent_pupil_diameter_change_mm(means, *eye, target)
    return out
