"""The measured apparent pupil center shift at the center target: magnitude, direction, and agreement between cameras.

    uv run python measured/center_target_summary.py

Measurement only, on the apparent pupil center shift per mm of pupil diameter change in screen axes. Every number is
per eye; left and right eyes are never pooled.

    magnitude     median and range of the shift's magnitude per camera
    direction     per camera, how many eyes have a horizontal component pointing temporally or nasally, and a vertical
                  component pointing up or down
    two cameras   for eyes recorded by both cameras, the angle between the two cameras' shifts and the absolute
                  difference in their magnitudes, as median and interquartile range; the rank correlation between that
                  angle and the eye's shift magnitude (the mean over both cameras); and every eye's values, sorted by
                  shift magnitude

Prints only; writes nothing.
"""

from operator import itemgetter

import numpy as np
from scipy.stats import spearmanr

from psa_eye_model_evaluation.apparent_pupil_center_shift_change import angle_between_deg
from psa_eye_model_evaluation.experiment_setup import (
    CAMERA_LABEL,
    CAMERAS,
    CENTER_TARGET,
    EYE_LABEL,
    EYES,
    NASAL_DIRECTION,
)
from psa_eye_model_evaluation.recordings import load_measured_apparent_pupil_center_shift

type CenterShift = dict[tuple[str, str, str], np.ndarray]


def print_magnitudes(center: CenterShift) -> None:
    """Median and range of the shift's magnitude per eye and camera."""
    print("magnitude at the center target (mm per mm)")
    for eye in EYES:
        for camera in CAMERAS:
            magnitudes = [
                float(np.linalg.norm(v)) for (c, _participant, e), v in center.items() if (c, e) == (camera, eye)
            ]
            print(
                f"  {EYE_LABEL[eye]}, {CAMERA_LABEL[camera]:<20}median {np.median(magnitudes):.3f}"
                f"   range {min(magnitudes):.3f}-{max(magnitudes):.3f}   n {len(magnitudes)}"
            )


def print_directions(center: CenterShift) -> None:
    """How many eyes point temporally or nasally, and up or down, per eye and camera."""
    print("\ndirection at the center target")
    for eye in EYES:
        for camera in CAMERAS:
            label = CAMERA_LABEL[camera]
            vectors = [v for (c, _participant, e), v in center.items() if (c, e) == (camera, eye)]
            temporal = sum(1 for v in vectors if np.sign(v[0]) == -NASAL_DIRECTION[eye])
            nasal = sum(1 for v in vectors if np.sign(v[0]) == NASAL_DIRECTION[eye])
            up = sum(1 for v in vectors if v[1] > 0)
            down = sum(1 for v in vectors if v[1] < 0)
            print(
                f"  {EYE_LABEL[eye]}, {label:<20}horizontal: temporal {temporal:2d}, nasal {nasal:2d}"
                f"   vertical: up {up:2d}, down {down:2d}   of {len(vectors)}"
            )


def print_camera_agreement(center: CenterShift) -> None:
    """Angle and magnitude difference between the two cameras' shifts for each eye recorded by both."""
    print("\ntwo cameras, eyes recorded by both")
    first_camera, second_camera = CAMERAS
    for eye in EYES:
        participants = sorted(
            participant
            for camera, participant, e in center
            if (camera, e) == (first_camera, eye) and (second_camera, participant, eye) in center
        )
        rows = []
        for participant in participants:
            first, second = center[first_camera, participant, eye], center[second_camera, participant, eye]
            magnitude = float((np.linalg.norm(first) + np.linalg.norm(second)) / 2)
            difference = float(abs(np.linalg.norm(first) - np.linalg.norm(second)))
            rows.append((participant, magnitude, angle_between_deg(first, second), difference))
        angles = [row[2] for row in rows]
        differences = [row[3] for row in rows]
        a25, a50, a75 = np.percentile(angles, [25, 50, 75])
        d25, d50, d75 = np.percentile(differences, [25, 50, 75])
        rho, p_value = spearmanr([row[1] for row in rows], angles)
        print(f"  {EYE_LABEL[eye]}, n {len(rows)}")
        print(f"    angle between cameras      median {a50:.0f}° (IQR {a25:.0f}°-{a75:.0f}°)")
        print(f"    magnitude difference       median {d50:.3f} (IQR {d25:.3f}-{d75:.3f}) mm per mm")
        print(f"    angle vs shift magnitude   Spearman rho {rho:.2f}, p = {p_value:.3f}")
        print(f"    {'participant':<13}{'magnitude':>10}{'angle':>8}{'difference':>12}")
        for participant, magnitude, angle, difference in sorted(rows, key=itemgetter(1)):
            print(f"    {participant:<13}{magnitude:10.3f}{angle:7.0f}°{difference:12.3f}")


def main() -> None:
    """Print the center-target magnitude, direction and camera agreement per eye."""
    center = {
        (camera, participant, eye): vector
        for (camera, participant, eye, target), vector in load_measured_apparent_pupil_center_shift().items()
        if target == CENTER_TARGET
    }
    print_magnitudes(center)
    print_directions(center)
    print_camera_agreement(center)


if __name__ == "__main__":
    main()
