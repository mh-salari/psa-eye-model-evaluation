"""The PSA a P-CR eye tracker reports, from the recordings and from each simulator with parameters fitted to each eye.

    uv run python simulators/psa_equivalent.py

For each camera and eye, the nine-point biquadratic calibration with per-quadrant corner correction of Stampe, D. M.
(1993), "Heuristic filtering and reliable calibration methods for video-based pupil-tracking systems", Behavior
Research Methods, Instruments, & Computers, 25(2), 137-142, is fitted to the eye's P-CR at the nine targets in the
constricted condition, with the center and edge targets as inner points and the corner targets as outer points. The
P-CR in the dilated condition is mapped through that calibration, and the PSA at a target is the distance between the
gaze points predicted in the two conditions, the apparent gaze shift of Salari, M., Niehorster, D. C., Nystrom, M., &
Bednarik, R. (2025), "The effect of pupil size on data quality in head-mounted eye trackers", Behavior Research Methods,
58(1), 17.

The recordings are calibrated on each eye's mean P-CR per condition. Each simulator is calibrated on its own simulated
P-CR, from its second-test run: each eye's anatomy fitted to that camera, and the physical pupil diameters and pupil
center shift inferred at the center target. A simulator's error at a target is the distance between its PSA vector and
the measured one. Both calibrations put the constricted gaze points on the targets, so this is the distance between the
two dilated gaze points, and an error in direction counts as much as one in magnitude. It is given in degrees and as a
percentage of the measured PSA at that target.

Per camera and eye, the medians are taken over participants and the eight eccentric targets, and over participants at
the center target. Left and right eyes are never pooled.

Prints only; writes nothing.
"""

import numpy as np
from stampe1993_gaze_mapping import StampeModel, angular_error

from psa_eye_model_evaluation.experiment_setup import (
    CAMERA_LABEL,
    CENTER_TARGET,
    ECCENTRIC_TARGETS,
    EYE_LABEL,
    EYE_POSITIONS_MM,
    EYES,
    GRID,
    PANEL_CAMERAS,
    SCREEN_POSITIONS,
    TARGET_LABELS,
)
from psa_eye_model_evaluation.recordings import condition_means, load_phases
from psa_eye_model_evaluation.simulator_results import FITTED_RUN, SIMULATORS, load_simulated_pcr

CALIBRATION_CONDITION, TEST_CONDITION = "constricted", "dilated"
TARGETS = tuple(TARGET_LABELS)
# Center and edge targets lie on a grid axis; the four corners are the outer points of the corner correction.
INNER_TARGETS = tuple(target for target in TARGETS if 0 in GRID[target])
OUTER_TARGETS = tuple(target for target in TARGETS if 0 not in GRID[target])
# The screen positions are already in mm, so one screen unit is one mm on both axes.
SCREEN_UNIT_MM = 1.0


def psa_vectors_mm(calibration_pcr: np.ndarray, test_pcr: np.ndarray) -> np.ndarray:
    """(9, 2) on-screen PSA (mm): the test condition's gaze points minus the calibration condition's, per target.

    Both P-CR arrays are (9, 2) in ``TARGETS`` order. The calibration is fitted to ``calibration_pcr``.
    """
    targets = np.array([SCREEN_POSITIONS[target] for target in TARGETS])
    inner = [TARGETS.index(target) for target in INNER_TARGETS]
    outer = [TARGETS.index(target) for target in OUTER_TARGETS]
    model = StampeModel(degree=2)
    model.fit(calibration_pcr[inner], targets[inner], calibration_pcr[outer], targets[outer])
    return model.predict(test_pcr) - model.predict(calibration_pcr)


def magnitude_deg(offsets_mm: np.ndarray, eye: str) -> np.ndarray:
    """The visual angle (deg) of on-screen gaze offsets (mm), at the eye's viewing distance."""
    deg_x, deg_y, _ = angular_error(offsets_mm, SCREEN_UNIT_MM, SCREEN_UNIT_MM, EYE_POSITIONS_MM[eye][2])
    return np.hypot(deg_x, deg_y)


def main() -> None:
    """Print, per camera and eye, the measured PSA and how far each simulator's PSA is from it."""
    means = condition_means(load_phases())
    center = TARGETS.index(CENTER_TARGET)
    eccentric = [TARGETS.index(target) for target in ECCENTRIC_TARGETS]
    for camera in PANEL_CAMERAS:
        simulated = {simulator: load_simulated_pcr(simulator, camera, FITTED_RUN[camera]) for simulator in SIMULATORS}
        print(f"\n{CAMERA_LABEL[camera]}")
        for eye in EYES:
            covered = {
                simulator: sorted({participant for _camera, participant, e, _target, _condition in pcr if e == eye})
                for simulator, pcr in simulated.items()
            }
            participants = next(iter(covered.values()))
            if any(eyes != participants for eyes in covered.values()):
                raise SystemExit(f"the second-test runs on {camera} cover different participants' {eye}")
            measured_psa = []
            errors: dict[str, list[np.ndarray]] = {simulator: [] for simulator in SIMULATORS}
            for participant in participants:
                measured = psa_vectors_mm(
                    *(
                        np.array([means[camera, participant, eye, target, condition][:2] for target in TARGETS])
                        for condition in (CALIBRATION_CONDITION, TEST_CONDITION)
                    )
                )
                measured_psa.append(magnitude_deg(measured, eye))
                for simulator, pcr in simulated.items():
                    simulated_psa = psa_vectors_mm(
                        *(
                            np.array([pcr[camera, participant, eye, target, condition] for target in TARGETS])
                            for condition in (CALIBRATION_CONDITION, TEST_CONDITION)
                        )
                    )
                    errors[simulator].append(magnitude_deg(simulated_psa - measured, eye))
            psa_deg = np.array(measured_psa)
            print(
                f"{EYE_LABEL[eye]}, {len(participants)} eyes: measured PSA median {np.median(psa_deg[:, eccentric]):.2f}"
                f" deg at the eccentric targets, {np.median(psa_deg[:, center]):.2f} deg at the center target"
            )
            for simulator, simulator_errors in errors.items():
                error_deg = np.array(simulator_errors)
                share = 100 * error_deg / psa_deg
                print(
                    f"  {SIMULATORS[simulator]} off from the measured PSA: {np.median(error_deg[:, eccentric]):.2f} deg "
                    f"({np.median(share[:, eccentric]):.0f}%) at the eccentric targets, "
                    f"{np.median(error_deg[:, center]):.2f} deg ({np.median(share[:, center]):.0f}%) at the center "
                    "target"
                )


if __name__ == "__main__":
    main()
