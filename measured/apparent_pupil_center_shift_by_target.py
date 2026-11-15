"""How the measured apparent pupil center shift changes from the center target to each eccentric target.

    uv run python measured/apparent_pupil_center_shift_by_target.py

Measurement only. For each camera, participant, eye and eccentric target, the direction change and the magnitude
change of the shift from the center target, the magnitude change also as a percentage of the center-target shift
magnitude, and the angle between the difference of the two shifts and the direction from the target's screen position
to the center target's. Per camera and eye it prints the median direction change over participants and eccentric
targets, the range across the eight targets of the per-target median direction change and magnitude change, and the
median magnitude change over the nasal and over the temporal targets, in mm per mm and in percent. It then prints, per
camera and eye, the share of the differences that point back toward the center target and their median angle to the
direction of it. Left and right eyes are never pooled.

Writes output/apparent_pupil_center_shift_by_target.csv, one row per camera, participant, eye and eccentric target.
"""

import csv
from pathlib import Path

import numpy as np

from psa_eye_model_evaluation.apparent_pupil_center_shift_change import (
    TOWARD_CENTER_TARGET_DEG,
    angle_to_center_target_deg,
    changes_from_center,
    summarize_changes,
)
from psa_eye_model_evaluation.experiment_setup import (
    CAMERA_LABEL,
    CENTER_TARGET,
    EYE_LABEL,
    EYES,
    PANEL_CAMERAS,
    TARGET_LABELS,
    nasal_or_temporal,
)
from psa_eye_model_evaluation.recordings import load_measured_apparent_pupil_center_shift

OUT = Path(__file__).resolve().parent / "output" / "apparent_pupil_center_shift_by_target.csv"
COLUMNS = (
    "camera",
    "participant",
    "eye",
    "target",
    "target_label",
    "nasal_or_temporal",
    "direction_change_deg",
    "magnitude_change_mm_per_mm",
    "magnitude_change_percent",
    "angle_to_center_target_deg",
)


def main() -> None:
    """Write every participant's changes, and print their summary per camera and eye."""
    shift = load_measured_apparent_pupil_center_shift()
    rows = []
    angles_to_center: dict[tuple[str, str], list[float]] = {}
    print(
        f"{'':<32}{'direction change':>17}{'range':>16}{'magnitude change range':>24}{'nasal':>18}{'temporal':>18}"
        f"{'eyes':>6}"
    )
    for camera in PANEL_CAMERAS:
        for eye in EYES:
            participants = sorted({p for c, p, e, _target in shift if (c, e) == (camera, eye)})
            per_participant = {
                participant: changes_from_center({t: shift[camera, participant, eye, t] for t in TARGET_LABELS})
                for participant in participants
            }
            angles_to_center[camera, eye] = []
            for participant, changes in per_participant.items():
                for target, change in changes.items():
                    difference = (
                        shift[camera, participant, eye, target] - shift[camera, participant, eye, CENTER_TARGET]
                    )
                    angle_to_center = angle_to_center_target_deg(difference, target)
                    angles_to_center[camera, eye].append(angle_to_center)
                    rows.append((
                        camera,
                        participant,
                        eye,
                        target,
                        TARGET_LABELS[target],
                        nasal_or_temporal(eye, target) or "",
                        change.direction_change,
                        change.magnitude_change,
                        change.magnitude_change_percent,
                        angle_to_center,
                    ))
            summary = summarize_changes(per_participant, eye)
            low, high = summary.direction_change_range
            magnitude_low, magnitude_high = summary.magnitude_change_range
            print(
                f"{CAMERA_LABEL[camera] + ', ' + EYE_LABEL[eye]:<32}{summary.median_direction_change:16.1f}°"
                f"{low:9.1f}°-{high:4.1f}°{magnitude_low:+14.4f} to {magnitude_high:+.4f}"
                f"{summary.nasal_magnitude_change:+10.4f} ({summary.nasal_magnitude_change_percent:+4.0f}%)"
                f"{summary.temporal_magnitude_change:+10.4f} ({summary.temporal_magnitude_change_percent:+4.0f}%)"
                f"{summary.n_eyes:6d}"
            )

    print("\nDifference from the center target: share pointing back toward it, and median angle to its direction")
    for (camera, eye), angles in angles_to_center.items():
        toward = np.mean(np.array(angles) < TOWARD_CENTER_TARGET_DEG) * 100
        print(
            f"{CAMERA_LABEL[camera] + ', ' + EYE_LABEL[eye]:<32}{toward:5.0f}% of {len(angles)}"
            f"   median angle {np.median(angles):5.1f}°"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"\nwrote {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
