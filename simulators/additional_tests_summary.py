"""Each simulator's two additional tests against its population eye and the measured apparent pupil center shift.

    uv run python simulators/additional_tests_summary.py

For each camera and eye, compares the change from the center target in the measured apparent pupil center shift and in
three runs of each simulator: the population eye, the first test (the physical pupil diameters and pupil center shift
inferred at the center target) and the second test (the first test with each eye's anatomy fitted to that camera). A
fitted run covers only the eyes its fit covers, so every run is set against the population-eye run and the measurement
on its own eyes.

The first block is the paper's table: the median magnitude change at the nasal and at the temporal targets, in mm per
mm, and the median direction change, each with its median absolute deviation (MAD) over the same values, all
participants and the targets on that side. The second block gives each run's median as a percentage of the measured
one, the magnitude change in absolute value. The third block gives, per run, the median per-eye magnitude ratio of the
simulated to the measured apparent pupil center shift at the center target and the median angle between the two, the
largest difference from the population eye in the per-target median magnitude change, and the percentage of the
differences from the center target that point back toward it.

For each run with inferred inputs, it prints the median ratio of the inferred physical pupil diameter to the measured
apparent one at the center target, per camera, eye and condition. From each simulator's fit on each camera, it prints
the median P-CR residual before and after fitting, and how many eyes ended inside, at the lower bound and at the upper
bound of each fitted parameter.

Prints the report and writes it to output/additional_tests_summary.txt.
"""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from psa_eye_model_evaluation.apparent_pupil_center_shift_change import (
    ChangeSummary,
    share_toward_center_target,
    summarize_changes_by_eye,
)
from psa_eye_model_evaluation.experiment_setup import (
    CAMERA_LABEL,
    CENTER_TARGET,
    ECCENTRIC_TARGETS,
    EYE_LABEL,
    EYES,
    PANEL_CAMERAS,
)
from psa_eye_model_evaluation.recordings import load_measured_apparent_pupil_center_shift
from psa_eye_model_evaluation.simulator_results import (
    FITTED_RUN,
    RESULTS_DIR,
    SIMULATORS,
    Shifts,
    center_target_agreement,
    load_simulator_result_file,
    result_path,
)

SIMULATOR_INPUT = Path(__file__).resolve().parents[1] / "measured" / "output" / "simulator_input.csv"
REPORT = RESULTS_DIR / "additional_tests_summary.txt"
CONDITIONS = ("dilated", "constricted")
# Simulators in the order of the paper's tables.
TABLE_ORDER = ("et_simul", "gkamodeleye", "pyetsimul")
MEASURED = "measured"
POPULATION_EYE = "population eye"
FIRST_TEST = "first test"
SECOND_TEST = "second test"
# (label, run name) of each simulator's runs per camera; the three simulators share the run names.
RUNS = {
    camera: ((POPULATION_EYE, ""), (FIRST_TEST, "physical"), (SECOND_TEST, FITTED_RUN[camera]))
    for camera in PANEL_CAMERAS
}
# The fit table behind each simulator's second test on each camera.
FIT_TABLES = {
    "pyetsimul": {"cam2": "eye_parameters_eye_level.csv", "cam1": "eye_parameters_bottom_left.csv"},
    "et_simul": {"cam2": "eye_parameters_et_simul_eye_level.csv", "cam1": "eye_parameters_et_simul_bottom_left.csv"},
    "gkamodeleye": {
        "cam2": "eye_parameters_gkamodeleye_eye_level.csv",
        "cam1": "eye_parameters_gkamodeleye_bottom_left.csv",
    },
}
# A fit table has one column per fitted parameter saying which bound, if any, the fitted value reached.
RAILED_SUFFIX = "_railed"
UM_PER_MM = 1000.0
LABEL_WIDTH = 30
TABLE_HEADER = (
    f"{'':<{LABEL_WIDTH}}{'nasal (mm per mm)':>20}{'temporal (mm per mm)':>22}{'direction change (deg)':>24}"
    f"{'eyes':>6}"
)
SHARE_HEADER = f"{'':<{LABEL_WIDTH}}{'nasal':>10}{'temporal':>10}{'direction':>11}"
REFERENCE_HEADER = f"{'':<{LABEL_WIDTH}}{'ratio':>8}{'angle (deg)':>13}{'vs population eye':>19}{'toward center':>15}"


@dataclass(frozen=True)
class RunOnEye:
    """One run's change from the center target for one eye, with its references on the same eyes.

    run           the run's summary
    population    the population eye's summary on the run's eyes
    measurement   the measurement's summary on the run's eyes
    ratio         median per-eye magnitude ratio of the simulated to the measured shift at the center target
    angle         median angle (deg) between the two at the center target
    largest       largest difference from the population eye in the per-target median magnitude change (mm per mm)
    toward        percentage of the run's differences from the center target that point back toward it
    """

    run: ChangeSummary
    population: ChangeSummary
    measurement: ChangeSummary
    ratio: float
    angle: float
    largest: float
    toward: float


def restricted(shifts: Shifts, eyes: set[tuple[str, str]]) -> Shifts:
    """The shifts of the given (participant, eye) pairs only."""
    return {key: vector for key, vector in shifts.items() if (key[1], key[2]) in eyes}


def run_on_eye(shifts: Shifts, population_shifts: Shifts, measured: Shifts, camera: str, eye: str) -> RunOnEye:
    """Summarize one run for one eye against the population eye and the measurement on the run's own eyes."""
    eyes = {(participant, e) for _camera, participant, e, _target in shifts}
    run = summarize_changes_by_eye(shifts, camera)[eye]
    population = summarize_changes_by_eye(restricted(population_shifts, eyes), camera)[eye]
    measurement = summarize_changes_by_eye(restricted(measured, eyes), camera)[eye]
    ratio, angle, _n_eyes = center_target_agreement(measured, shifts, camera, eye)
    largest = max(
        abs(run.median_magnitude_change_by_target[t] - population.median_magnitude_change_by_target[t])
        for t in ECCENTRIC_TARGETS
    )
    return RunOnEye(
        run, population, measurement, ratio, angle, largest, share_toward_center_target(shifts, camera, eye)
    )


def magnitude_cell(median: float, mad: float) -> str:
    """A magnitude change as 'median (MAD)' in mm per mm."""
    return f"{median:+.4f} ({mad:.4f})"


def direction_cell(median: float, mad: float) -> str:
    """A direction change as 'median (MAD)' in degrees."""
    return f"{median:.1f} ({mad:.1f})"


def table_row(label: str, summary: ChangeSummary) -> str:
    """One row of the paper's table."""
    return (
        f"{label:<{LABEL_WIDTH}}"
        f"{magnitude_cell(summary.nasal_magnitude_change, summary.nasal_magnitude_change_mad):>20}"
        f"{magnitude_cell(summary.temporal_magnitude_change, summary.temporal_magnitude_change_mad):>22}"
        f"{direction_cell(summary.median_direction_change, summary.direction_change_mad):>24}"
        f"{summary.n_eyes:>6d}"
    )


def share_row(label: str, result: RunOnEye) -> str:
    """The run's medians as a percentage of the measured ones, the magnitude change in absolute value."""
    nasal = 100 * abs(result.run.nasal_magnitude_change / result.measurement.nasal_magnitude_change)
    temporal = 100 * abs(result.run.temporal_magnitude_change / result.measurement.temporal_magnitude_change)
    direction = 100 * result.run.median_direction_change / result.measurement.median_direction_change
    return f"{label:<{LABEL_WIDTH}}{nasal:9.1f}%{temporal:9.1f}%{direction:10.1f}%"


def reference_row(label: str, result: RunOnEye, is_population_eye: bool) -> str:
    """The run's center-target agreement, distance from the population eye and share pointing toward the center."""
    largest = "" if is_population_eye else f"{result.largest:.4f}"
    return f"{label:<{LABEL_WIDTH}}{result.ratio:8.3f}{result.angle:13.1f}{largest:>19}{result.toward:14.0f}%"


def run_label(simulator: str, label: str) -> str:
    """The simulator's name for its population eye, the test's name indented under it."""
    return SIMULATORS[simulator] if label == POPULATION_EYE else f"  {label}"


def camera_lines(camera: str, measured: Shifts) -> list[str]:
    """The three blocks for one camera: the table, the shares of the measured values, and the references."""
    measured_camera = {key: vector for key, vector in measured.items() if key[0] == camera}
    runs = {
        (simulator, label): load_simulator_result_file(simulator, camera, run)
        for simulator in TABLE_ORDER
        for label, run in RUNS[camera]
    }
    for simulator in TABLE_ORDER:
        if runs[simulator, POPULATION_EYE].keys() != measured_camera.keys():
            raise SystemExit(
                f"the {SIMULATORS[simulator]} population-eye run for {camera} covers other eyes or targets than the "
                "recordings"
            )
    results = {
        (simulator, label, eye): run_on_eye(shifts, runs[simulator, POPULATION_EYE], measured_camera, camera, eye)
        for (simulator, label), shifts in runs.items()
        for eye in EYES
    }
    measured_summaries = summarize_changes_by_eye(measured_camera, camera)

    lines = [f"\n{CAMERA_LABEL[camera]}", "", "Change from the center target, median (MAD)", TABLE_HEADER]
    for eye in EYES:
        lines.extend((EYE_LABEL[eye], table_row(f"  {MEASURED}", measured_summaries[eye])))
        lines.extend(
            table_row(f"  {run_label(simulator, label)}", results[simulator, label, eye].run)
            for simulator in TABLE_ORDER
            for label, _run in RUNS[camera]
        )

    lines += ["", "Each run's median as a percentage of the measured one, the magnitude change in absolute value"]
    lines.append(SHARE_HEADER)
    for eye in EYES:
        lines.append(EYE_LABEL[eye])
        lines.extend(
            share_row(f"  {run_label(simulator, label)}", results[simulator, label, eye])
            for simulator in TABLE_ORDER
            for label, _run in RUNS[camera]
        )

    lines += [
        "",
        "Center target: median per-eye magnitude ratio simulated / measured and median angle between them; largest",
        "per-target difference from the population eye in the median magnitude change (mm per mm); differences from",
        "the center target pointing back toward it",
        REFERENCE_HEADER,
    ]
    for eye in EYES:
        toward = share_toward_center_target(measured_camera, camera, eye)
        lines.extend((EYE_LABEL[eye], f"{'  ' + MEASURED:<{LABEL_WIDTH}}{'':>40}{toward:14.0f}%"))
        lines.extend(
            reference_row(f"  {run_label(simulator, label)}", results[simulator, label, eye], label == POPULATION_EYE)
            for simulator in TABLE_ORDER
            for label, _run in RUNS[camera]
        )
    return lines


def center_target_rows(path: Path, camera: str | None = None) -> dict[tuple[str, str], dict[str, str]]:
    """{(participant, eye): row} of a CSV's center-target rows, of one camera when the file covers several."""
    if not path.is_file():
        raise SystemExit(f"{path} is missing")
    with path.open(encoding="utf-8") as handle:
        return {
            (row["participant"], row["eye"]): row
            for row in csv.DictReader(handle)
            if int(row["target"]) == CENTER_TARGET and (camera is None or row["camera"] == camera)
        }


def physical_diameter_lines() -> list[str]:
    """Median ratio of the inferred physical pupil diameter to the measured apparent one at the center target."""
    lines = ["", "Inferred physical pupil diameter over measured apparent pupil diameter at the center target, median"]
    for camera in PANEL_CAMERAS:
        measured = center_target_rows(SIMULATOR_INPUT, camera)
        for simulator in TABLE_ORDER:
            for label, run in RUNS[camera]:
                if label == POPULATION_EYE:
                    continue
                simulated = center_target_rows(result_path(simulator, camera, run))
                for eye in EYES:
                    ratios = {
                        condition: np.median([
                            float(row[f"physical_pupil_diameter_{condition}_mm"])
                            / float(measured[key][f"apparent_pupil_diameter_{condition}_mm"])
                            for key, row in simulated.items()
                            if key[1] == eye
                        ])
                        for condition in CONDITIONS
                    }
                    row_label = f"{CAMERA_LABEL[camera]}, {EYE_LABEL[eye]}, {SIMULATORS[simulator]}, {label}"
                    lines.append(
                        f"  {row_label:<60}"
                        + "".join(f"{condition} {ratio:.3f}   " for condition, ratio in ratios.items())
                    )
    return lines


def fit_lines() -> list[str]:
    """Residuals and bound hits of each simulator's eye-parameter fit on each camera."""
    lines = ["", "Eye-parameter fits"]
    for simulator in TABLE_ORDER:
        for camera in PANEL_CAMERAS:
            path = RESULTS_DIR / FIT_TABLES[simulator][camera]
            if not path.is_file():
                raise SystemExit(f"{path} is missing; run the {SIMULATORS[simulator]} fit for that camera")
            with path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise SystemExit(f"{path.name} is empty")
                parameters = [
                    column.removesuffix(RAILED_SUFFIX)
                    for column in reader.fieldnames
                    if column.endswith(RAILED_SUFFIX)
                ]
                rows = list(reader)
            before = np.median([float(row["rms_before_um"]) for row in rows]) / UM_PER_MM
            after = np.median([float(row["rms_after_um"]) for row in rows]) / UM_PER_MM
            lines.append(
                f"  {SIMULATORS[simulator]} fit to the {CAMERA_LABEL[camera].lower()}: {len(rows)} eyes, median P-CR "
                f"residual {before:.3f} mm before, {after:.3f} mm after"
            )
            for parameter in parameters:
                counts = Counter(row[f"{parameter}{RAILED_SUFFIX}"] or "inside" for row in rows)
                if counts["held"]:
                    lines.append(f"    {parameter:<22}held   {counts['held']:2d}")
                else:
                    lines.append(
                        f"    {parameter:<22}inside {counts['inside']:2d}   lower bound {counts['lower']:2d}"
                        f"   upper bound {counts['upper']:2d}"
                    )
    return lines


def main() -> None:
    """Print the report for both cameras, the inferred pupil diameters and the fit statistics, and write it."""
    measured = load_measured_apparent_pupil_center_shift()
    lines: list[str] = []
    for camera in PANEL_CAMERAS:
        lines.extend(camera_lines(camera, measured))
    lines.extend(physical_diameter_lines())
    lines.extend(fit_lines())
    report = "\n".join(lines).lstrip("\n") + "\n"
    print(report, end="")
    REPORT.write_text(report, encoding="utf-8")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
