"""The measured input for the simulator scripts, taken from the recordings.

    uv run python measured/simulator_input.py

simulator_input.csv is the input of the nine-target runs of all three simulators, one row per eye and target: the
apparent pupil center shift in mm and the apparent pupil diameters at that target. The simulators take the physical
pupil diameters and pupil center shift from the center target's row. The diameters at each target give the measured
apparent pupil diameter change that the measured apparent pupil center shift at that target is divided by. The shift is
in mm, not per mm of diameter change, because the simulators place the physical pupil center shift from it. Each row also carries its
camera's roll about the view axis (deg), measured from the recordings, so every simulator can give its camera the real
camera's image axes.

fit_input.csv is the input of the eye-parameter fit: each eye's P-CR and apparent pupil diameter at every target, in
the camera's image axes, both in the constricted condition only. The apparent pupil center shift lives in the
dilated-minus-constricted difference, so a fit to the constricted P-CR cannot tune it.

Both files cover both cameras, so every simulator script reads identical input. Writes to measured/output/.
"""

import csv
from pathlib import Path

from psa_eye_model_evaluation.experiment_setup import TARGET_LABELS
from psa_eye_model_evaluation.recordings import (
    IMAGE_TO_SCREEN_AXES,
    apparent_pupil_center_shift_mm,
    camera_roll_deg,
    condition_means,
    load_phases,
    recorded_eyes,
)

OUTPUT = Path(__file__).resolve().parent / "output"
SIMULATOR_COLUMNS = (
    "camera",
    "participant",
    "eye",
    "target",
    "target_label",
    "apparent_pupil_center_shift_x_mm",
    "apparent_pupil_center_shift_y_mm",
    "apparent_pupil_diameter_dilated_mm",
    "apparent_pupil_diameter_constricted_mm",
    "camera_roll_deg",
)
FIT_COLUMNS = (
    "camera",
    "participant",
    "eye",
    "target",
    "target_label",
    "pcr_constricted_x_mm",
    "pcr_constricted_y_mm",
    "apparent_pupil_diameter_constricted_mm",
)


def write(name: str, columns: tuple[str, ...], rows: list[tuple]) -> None:
    """Write one CSV file to the output directory."""
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows")


def main() -> None:
    """Write both input files for every eye on both cameras."""
    means = condition_means(load_phases())
    simulator_rows, fit_rows = [], []
    for camera, participant, eye in recorded_eyes(means):
        for target, label in TARGET_LABELS.items():
            shift_mm = apparent_pupil_center_shift_mm(means, camera, participant, eye, target)
            shift_x, shift_y = IMAGE_TO_SCREEN_AXES * shift_mm
            pcr_x, pcr_y, constricted = means[camera, participant, eye, target, "constricted"]
            dilated = means[camera, participant, eye, target, "dilated"][2]
            simulator_rows.append((
                camera,
                participant,
                eye,
                target,
                label,
                float(shift_x),
                float(shift_y),
                float(dilated),
                float(constricted),
                camera_roll_deg(camera),
            ))
            fit_rows.append((camera, participant, eye, target, label, float(pcr_x), float(pcr_y), float(constricted)))
    write("simulator_input.csv", SIMULATOR_COLUMNS, simulator_rows)
    write("fit_input.csv", FIT_COLUMNS, fit_rows)


if __name__ == "__main__":
    main()
