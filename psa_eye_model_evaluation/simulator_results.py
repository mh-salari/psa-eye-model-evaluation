"""Simulator result files, and their agreement with the measured apparent pupil center shift at the center target."""

import csv
from pathlib import Path

import numpy as np

from psa_eye_model_evaluation.apparent_pupil_center_shift_change import angle_between_deg
from psa_eye_model_evaluation.experiment_setup import CENTER_TARGET
from psa_eye_model_evaluation.recordings import load_measured_apparent_pupil_center_shift

# Simulator key, as used in the result file names -> display name.
SIMULATORS = {"pyetsimul": "PyEtSimul", "et_simul": "et_simul", "gkamodeleye": "gkaModelEye"}
# Result file suffix per camera; the eye-level camera is each simulator script's default run and has none.
RESULT_FILE_CAMERA_SUFFIX = {"cam2": "", "cam1": "_cam1"}
# Each simulator's second test per camera: the eye fitted to that camera alone, with the physical pupil diameters and
# pupil center shift inferred at the center target.
FITTED_RUN = {"cam2": "eye_level_physical", "cam1": "bottom_left_physical"}
RESULTS_DIR = Path(__file__).resolve().parents[1] / "simulators" / "output"
# A result file's copy of the measured apparent pupil center shift matches the recordings up to CSV round-trip
# precision; a larger difference means the file was written from other input.
MEASURED_TOLERANCE = 1e-6

type Shifts = dict[tuple[str, str, str, int], np.ndarray]
type Pcrs = dict[tuple[str, str, str, int, str], np.ndarray]

CONDITIONS = ("constricted", "dilated")


def result_path(simulator: str, camera: str, run: str = "") -> Path:
    """The result file of one simulator run on one camera; ``run`` names a variant, empty for the default."""
    if simulator not in SIMULATORS:
        raise SystemExit(f"unknown simulator {simulator!r}; expected one of {', '.join(SIMULATORS)}")
    run_suffix = f"_{run}" if run else ""
    name = f"apparent_pupil_center_shift_9targets_{simulator}{run_suffix}{RESULT_FILE_CAMERA_SUFFIX[camera]}.csv"
    return RESULTS_DIR / name


def load_simulator_result_file(simulator: str, camera: str, run: str = "") -> Shifts:
    """One result file's simulated apparent pupil center shift per mm of simulated apparent pupil diameter change.

    Keyed by (camera, participant, eye, target). Every row's copy of the measured apparent pupil center shift is checked
    against the recordings, so a file written from other input stops the caller instead of being compared against the
    wrong measurement.
    """
    path = result_path(simulator, camera, run)
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run the {SIMULATORS[simulator]} nine-target script for {camera}")
    measured = load_measured_apparent_pupil_center_shift()
    simulated: Shifts = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (camera, row["participant"], row["eye"], int(row["target"]))
            if key not in measured:
                raise SystemExit(f"{path.name} has {key}, which the recordings do not")
            measured_copy = np.array([
                float(row["measured_apparent_pupil_center_shift_x_mm_per_mm"]),
                float(row["measured_apparent_pupil_center_shift_y_mm_per_mm"]),
            ])
            difference = float(np.max(np.abs(measured_copy - measured[key])))
            if difference > MEASURED_TOLERANCE:
                raise SystemExit(
                    f"{path.name}: the measured apparent pupil center shift of {key} differs from the recordings by "
                    f"{difference:.2e}"
                )
            simulated[key] = np.array([
                float(row["simulated_apparent_pupil_center_shift_x_mm_per_mm"]),
                float(row["simulated_apparent_pupil_center_shift_y_mm_per_mm"]),
            ])
    return simulated


def load_simulated_pcr(simulator: str, camera: str, run: str) -> Pcrs:
    """{(camera, participant, eye, target, condition): simulated P-CR (mm at the eye, image axes)} of a simulator run.

    The file first goes through load_simulator_result_file, so a file written from other input stops the caller here
    as well.
    """
    load_simulator_result_file(simulator, camera, run)
    path = result_path(simulator, camera, run)
    pcr: Pcrs = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path.name} is empty")
        missing = [
            f"simulated_pcr_{condition}_{axis}_mm"
            for condition in CONDITIONS
            for axis in ("x", "y")
            if f"simulated_pcr_{condition}_{axis}_mm" not in reader.fieldnames
        ]
        if missing:
            raise SystemExit(f"{path.name} has no {', '.join(missing)}; rerun the {SIMULATORS[simulator]} nine-target script")
        for row in reader:
            for condition in CONDITIONS:
                pcr[camera, row["participant"], row["eye"], int(row["target"]), condition] = np.array([
                    float(row[f"simulated_pcr_{condition}_x_mm"]),
                    float(row[f"simulated_pcr_{condition}_y_mm"]),
                ])
    return pcr


def load_simulated_apparent_pupil_center_shift(simulator: str) -> Shifts:
    """Both cameras' simulated apparent pupil center shift per mm of simulated apparent pupil diameter change."""
    simulated: Shifts = {}
    for camera in RESULT_FILE_CAMERA_SUFFIX:
        simulated.update(load_simulator_result_file(simulator, camera))
    return simulated


def center_target_agreement(measured: Shifts, simulated: Shifts, camera: str, eye: str) -> tuple[float, float, int]:
    """(median ratio |simulated| / |measured|, median angle between them in deg, eyes) at the center target.

    Covers the eyes that ``simulated`` has. The ratio is taken per eye before the median, so an eye with a near-zero
    measured apparent pupil center shift cannot dominate the way it would in a ratio of two medians.
    """
    ratios, angles = [], []
    for key, simulated_vector in simulated.items():
        if key[0] != camera or key[2] != eye or key[3] != CENTER_TARGET:
            continue
        measured_vector = measured[key]
        ratios.append(float(np.linalg.norm(simulated_vector) / np.linalg.norm(measured_vector)))
        angles.append(angle_between_deg(measured_vector, simulated_vector))
    if not ratios:
        raise ValueError(f"no simulated {eye} on {camera} at the center target")
    return float(np.median(ratios)), float(np.median(angles)), len(ratios)
