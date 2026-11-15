"""Apparent pupil diameter in the dilated and the constricted condition, per camera, participant and eye.

    uv run python measured/pupil_size.py

Each eye's diameter in each condition is its mean over the three phases and the nine targets. Writes
output/pupil_size.csv for pupil_size.R, and prints the diameters and the dilation per camera, the right eye against
the left, the spread over the three phases, and each target's diameter against the center target's.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from psa_eye_model_evaluation.experiment_setup import CENTER_TARGET, ECCENTRIC_TARGETS, EYE_LABEL, EYES, TARGET_LABELS
from psa_eye_model_evaluation.recordings import (
    Means,
    Phases,
    condition_means,
    load_phases,
    nine_target_mean_apparent_pupil_diameters,
)

OUT = Path(__file__).resolve().parent / "output" / "pupil_size.csv"
CONDITIONS = ("dilated", "constricted")

type Diameters = dict[tuple[str, str, str, str], float]


def write_diameters(diameters: Diameters) -> None:
    """Write one row per camera, participant, eye and condition."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("camera", "participant", "eye", "condition", "apparent_pupil_diameter_mm"))
        writer.writerows(sorted((*key, diameter) for key, diameter in diameters.items()))
    print(f"wrote {OUT}: {len(diameters)} rows\n")


def print_by_camera(diameters: Diameters) -> None:
    """Mean and SD of the diameter per camera and condition, and of the dilation per eye."""
    print(f"{'camera':<8}{'dilated':>16}{'constricted':>16}{'dilation':>26}")
    for camera in sorted({camera for camera, _participant, _eye, _condition in diameters}):
        by_condition = {
            condition: [d for (c, _p, _e, cond), d in diameters.items() if c == camera and cond == condition]
            for condition in CONDITIONS
        }
        dilation = [
            diameters[camera, participant, eye, "dilated"] - diameters[camera, participant, eye, "constricted"]
            for c, participant, eye, condition in diameters
            if c == camera and condition == "dilated"
        ]
        conditions = "".join(f"{np.mean(v):6.2f} +- {np.std(v, ddof=1):4.2f} mm" for v in by_condition.values())
        dilation_text = f"{np.mean(dilation):6.2f} +- {np.std(dilation, ddof=1):4.2f} mm"
        print(f"{camera:<8}{conditions}{dilation_text} ({min(dilation):.2f}-{max(dilation):.2f})")


def print_right_against_left(diameters: Diameters) -> None:
    """Mean diameter of each eye per camera and condition, and how much smaller or larger the right eye is."""
    print(f"\n{'camera':<8}{'condition':<13}{'left':>12}{'right':>12}{'right vs left':>16}")
    for camera in sorted({camera for camera, _participant, _eye, _condition in diameters}):
        for condition in CONDITIONS:
            left, right = (
                np.mean([d for (c, _p, e, cond), d in diameters.items() if (c, e, cond) == (camera, eye, condition)])
                for eye in ("left_eye", "right_eye")
            )
            print(f"{camera:<8}{condition:<13}{left:9.2f} mm{right:9.2f} mm{100 * (right - left) / left:15.1f}%")


def print_phase_spread(phases: Phases, diameters: Diameters) -> None:
    """Median SD of the diameter over the three phases at one target, over participants and the nine targets.

    Per camera, eye and condition, also as % of that eye's mean diameter in the condition.
    """
    per_target: dict[tuple[str, str, str, int, str], list[float]] = defaultdict(list)
    for (camera, participant, eye, target, condition, _phase), values in phases.items():
        per_target[camera, participant, eye, target, condition].append(float(values[2]))
    print(f"\n{'camera':<8}{'eye':<11}{'condition':<13}{'phase spread (SD)':>20}{'as % of the mean':>20}")
    for camera in sorted({key[0] for key in per_target}):
        for eye in EYES:
            for condition in CONDITIONS:
                spread = np.median([
                    np.std(v, ddof=1)
                    for key, v in per_target.items()
                    if (key[0], key[2], key[4]) == (camera, eye, condition)
                ])
                mean = np.mean([
                    d for key, d in diameters.items() if (key[0], key[2], key[3]) == (camera, eye, condition)
                ])
                print(f"{camera:<8}{EYE_LABEL[eye]:<11}{condition:<13}{spread:17.3f} mm{100 * spread / mean:19.1f}%")


def print_target_against_center(means: Means) -> None:
    """Each target's diameter minus the center target's, median over participants, per camera, eye and condition.

    Printed in mm and in % of the center target's diameter. Kept per eye, because each eye views the camera from a
    different angle, so the foreshortening differs between them.
    """
    per_eye: dict[tuple[str, str, str, str], dict[int, float]] = defaultdict(dict)
    for (camera, participant, eye, target, condition), values in means.items():
        per_eye[camera, participant, eye, condition][target] = float(values[2])
    columns = [(eye, condition) for eye in EYES for condition in CONDITIONS]
    for camera in sorted({camera for camera, _participant, _eye, _condition in per_eye}):
        print(f"\n{camera}: diameter at each target minus the center target's, median over participants")
        print(f"{'':<8}" + "".join(f"{EYE_LABEL[eye] + ', ' + condition:>26}" for eye, condition in columns))
        for target in ECCENTRIC_TARGETS:
            cells = []
            for eye, condition in columns:
                eyes = [d for (c, _p, e, cond), d in per_eye.items() if (c, e, cond) == (camera, eye, condition)]
                difference = np.median([d[target] - d[CENTER_TARGET] for d in eyes])
                percent = np.median([100 * (d[target] - d[CENTER_TARGET]) / d[CENTER_TARGET] for d in eyes])
                cells.append(f"{difference:+.3f} mm ({percent:+.1f}%)")
            print(f"{TARGET_LABELS[target]:<8}" + "".join(f"{cell:>26}" for cell in cells))


def main() -> None:
    """Write the diameters and print their summaries."""
    phases = load_phases()
    means = condition_means(phases)
    diameters = nine_target_mean_apparent_pupil_diameters(means)
    write_diameters(diameters)
    print_by_camera(diameters)
    print_right_against_left(diameters)
    print_phase_spread(phases, diameters)
    print_target_against_center(means)


if __name__ == "__main__":
    main()
