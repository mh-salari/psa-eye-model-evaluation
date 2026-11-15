"""The apparent pupil center shift that PyEtSimul produces at the nine targets, next to the measured one.

    uv run python simulators/apparent_pupil_center_shift_9targets_pyetsimul.py [--camera cam1] [--solve-physical]
        [--eye-parameters simulators/output/eye_parameters_eye_level.json]

Each eye is simulated twice at every target: constricted, and dilated with its pupil center moved by a physical pupil
center shift. The physical pupil diameters and the physical pupil center shift are the measured apparent ones at the
center target, the target the magnitude and direction change are measured from. They are the same at every target, so
the eye orientation is all that differs between targets.

--solve-physical instead infers the physical pupil diameters and the physical pupil center shift for which the
simulated apparent ones at the center target equal the measured ones. --eye-parameters takes each eye's anatomy from a
fit_eye_parameters.py run on the same camera instead of the population eye, and leaves out any eye the fit does not
cover, naming it in the output.

The simulated apparent pupil center shift and apparent pupil diameter change at a target are the dilated minus the
constricted pupil center and pupil diameter in the two simulated images, in mm at the eye, the diameter being the mean
of the fitted ellipse's two axes as in the recordings. The measured and the simulated apparent pupil center shift at a
target are each divided by their own apparent pupil diameter change at that target, measured or simulated.

Writes apparent_pupil_center_shift_9targets_pyetsimul{_run}{_cam1}.csv to simulators/output/, where the run names the
fit, followed by "physical" for --solve-physical. Each row also carries the simulated apparent pupil diameter change,
the eye's two physical pupil diameters and its simulated P-CR (pupil center minus corneal reflection) in both
conditions, in mm at the eye and in the camera's image axes.
"""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from pyetsimul.core import Camera, Light
from pyetsimul.core.camera import CameraImage
from pyetsimul.core.eye_model import EyeModel
from pyetsimul.core.off_axis_pupil import OffAxisPupilConfig
from tqdm import tqdm

from psa_eye_model_evaluation.apparent_pupil_center_shift_change import angle_between_deg
from psa_eye_model_evaluation.experiment_setup import CAMERAS, CENTER_TARGET, TARGET_LABELS
from psa_eye_model_evaluation.pyetsimul_scene import (
    PUPIL_CENTER_METHOD,
    TARGET_POSITIONS,
    apparent_pupil_diameter_mm,
    build_camera,
    build_eye,
    build_eye_model,
    build_illuminator,
    mm_per_pixel,
    physical_pupil_diameter_mm,
)
from psa_eye_model_evaluation.recordings import IMAGE_TO_SCREEN_AXES
from psa_eye_model_evaluation.simulator_results import result_path

SIMULATOR_INPUT = Path(__file__).resolve().parents[1] / "measured" / "output" / "simulator_input.csv"
# Eye-local step of the two probe simulations that measure how a physical pupil center shift appears in the image.
PROBE_MM = 0.05
RESULT_COLUMNS = (
    "participant",
    "eye",
    "target",
    "target_label",
    "measured_apparent_pupil_center_shift_x_mm_per_mm",
    "measured_apparent_pupil_center_shift_y_mm_per_mm",
    "simulated_apparent_pupil_center_shift_x_mm_per_mm",
    "simulated_apparent_pupil_center_shift_y_mm_per_mm",
    "simulated_apparent_pupil_diameter_change_mm",
    "physical_pupil_diameter_dilated_mm",
    "physical_pupil_diameter_constricted_mm",
    "simulated_pcr_constricted_x_mm",
    "simulated_pcr_constricted_y_mm",
    "simulated_pcr_dilated_x_mm",
    "simulated_pcr_dilated_y_mm",
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuredEye:
    """One eye's measured input on one camera. The shifts are in mm, in the screen axes."""

    apparent_pupil_center_shift_mm: dict[int, np.ndarray]
    apparent_pupil_diameter_change_mm: dict[int, float]
    apparent_pupil_diameter_dilated_center_target_mm: float
    apparent_pupil_diameter_constricted_center_target_mm: float


def load_simulator_input(camera: str) -> dict[tuple[str, str], MeasuredEye]:
    """{(participant, eye): that eye's measured input} on one camera, from measured/simulator_input.py."""
    if not SIMULATOR_INPUT.is_file():
        raise SystemExit(f"{SIMULATOR_INPUT} is missing; run measured/simulator_input.py")
    rows: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    with SIMULATOR_INPUT.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["camera"] == camera:
                rows[row["participant"], row["eye"]][int(row["target"])] = row
    if not rows:
        raise SystemExit(f"no {camera} rows in {SIMULATOR_INPUT}")
    eyes = {}
    for key, targets in sorted(rows.items()):
        if targets.keys() != TARGET_LABELS.keys():
            raise SystemExit(f"{camera} {key} has targets {sorted(targets)}, expected all nine")
        center = targets[CENTER_TARGET]
        eyes[key] = MeasuredEye(
            apparent_pupil_center_shift_mm={
                target: np.array([
                    float(row["apparent_pupil_center_shift_x_mm"]),
                    float(row["apparent_pupil_center_shift_y_mm"]),
                ])
                for target, row in targets.items()
            },
            apparent_pupil_diameter_change_mm={
                target: float(row["apparent_pupil_diameter_dilated_mm"])
                - float(row["apparent_pupil_diameter_constricted_mm"])
                for target, row in targets.items()
            },
            apparent_pupil_diameter_dilated_center_target_mm=float(center["apparent_pupil_diameter_dilated_mm"]),
            apparent_pupil_diameter_constricted_center_target_mm=float(
                center["apparent_pupil_diameter_constricted_mm"]
            ),
        )
    return eyes


def load_fitted_anatomy(path: Path, camera: str) -> dict[tuple[str, str], dict[str, float]]:
    """{(participant, eye): fitted anatomy} from a fit_eye_parameters.py run on ``camera``.

    A fit to another camera stops the run, because each camera's simulations use eyes fitted to that camera's view.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records.values():
        if record["camera"] != camera:
            raise SystemExit(f"{path.name} was fitted to {record['camera']}, not to {camera}")
    return {(record["participant"], record["eye"]): record["parameters"] for record in records.values()}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def off_axis_pupil_with_shift(model: EyeModel, eye: str, shift_mm: np.ndarray) -> OffAxisPupilConfig:
    """The model's off-axis pupil in raw eye-local mm, with ``shift_mm`` added to its horizontal and vertical parts.

    The shift is added to the off-axis pupil rather than replacing it, so two simulations built with different shifts
    differ by exactly the difference between the shifts.
    """
    resolved = replace(model.off_axis_pupil)
    resolved.resolve_defaults()
    offset = resolved.offset_for_eye(eye.removesuffix("_eye"))
    return OffAxisPupilConfig(
        enabled=True,
        nasal_mm=offset.x + float(shift_mm[0]),
        superior_mm=offset.y + float(shift_mm[1]),
        raw=True,
        axial_mm=offset.z,
    )


@dataclass(frozen=True)
class SimulatedEye:
    """One eye in PyEtSimul, seen by one camera under the setup's illuminator, with its physical pupil diameters."""

    camera: Camera
    illuminator: Light
    model: EyeModel
    eye: str
    dilated_physical_diameter_mm: float
    constricted_physical_diameter_mm: float

    def image(self, target: int, physical_diameter_mm: float, physical_shift_mm: np.ndarray) -> CameraImage:
        """The simulated image, with the eye looking at one target."""
        model = replace(self.model, off_axis_pupil=off_axis_pupil_with_shift(self.model, self.eye, physical_shift_mm))
        simulated = build_eye(model, self.eye)
        simulated.set_pupil_diameter(physical_diameter_mm)
        simulated.look_at(TARGET_POSITIONS[target])
        return self.camera.take_image(simulated, [self.illuminator], center_method=PUPIL_CENTER_METHOD)

    def apparent_pupil(
        self, target: int, physical_diameter_mm: float, physical_shift_mm: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """The pupil center (px) and pupil diameter (mm at the eye) in one simulated image, looking at one target."""
        image = self.image(target, physical_diameter_mm, physical_shift_mm)
        center = image.pupil_center
        return np.array([center.x, center.y]), apparent_pupil_diameter_mm(image, self.camera, self.eye)

    def pcr_mm(self, target: int, physical_diameter_mm: float, physical_shift_mm: np.ndarray) -> np.ndarray:
        """Pupil center minus corneal reflection at one target, in mm at the eye and in the camera's image axes."""
        image = self.image(target, physical_diameter_mm, physical_shift_mm)
        reflection = image.corneal_reflections[0]
        if reflection is None:
            raise ValueError(f"the {self.eye} shows no corneal reflection at target {target}")
        center = image.pupil_center
        return mm_per_pixel(self.camera, self.eye) * np.array([center.x - reflection.x, center.y - reflection.y])

    def apparent_change(self, target: int, physical_shift_mm: np.ndarray) -> tuple[np.ndarray, float]:
        """Dilated minus constricted pupil center (mm at the eye, screen axes) and pupil diameter (mm) at one target."""
        dilated_center, dilated_diameter = self.apparent_pupil(
            target, self.dilated_physical_diameter_mm, physical_shift_mm
        )
        constricted_center, constricted_diameter = self.apparent_pupil(
            target, self.constricted_physical_diameter_mm, np.zeros(2)
        )
        shift_mm = mm_per_pixel(self.camera, self.eye) * IMAGE_TO_SCREEN_AXES * (dilated_center - constricted_center)
        return shift_mm, dilated_diameter - constricted_diameter


def infer_physical_pupil_center_shift(simulated_eye: SimulatedEye, apparent_mm: np.ndarray) -> np.ndarray:
    """The physical pupil center shift (eye-local mm) that appears as ``apparent_mm`` at the center target.

    The map from a physical to an apparent pupil center shift is treated as affine over this range. Its constant part
    is the apparent shift without a physical shift, which the pupil diameter change produces on its own. Two probe
    simulations give its linear part, which also carries the rotation between the eye's axes and the image's.
    """
    zero, _diameter_change = simulated_eye.apparent_change(CENTER_TARGET, np.zeros(2))
    columns = [
        (simulated_eye.apparent_change(CENTER_TARGET, PROBE_MM * axis)[0] - zero) / PROBE_MM for axis in np.identity(2)
    ]
    return np.linalg.solve(np.column_stack(columns), apparent_mm - zero)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_name(eye_parameters: Path | None, solve_physical: bool) -> str:
    """The run's name in its result file: the fit it used, then "physical" for --solve-physical; empty by default."""
    parts = []
    if eye_parameters is not None:
        parts.append(eye_parameters.stem.removeprefix("eye_parameters_"))
    if solve_physical:
        parts.append("physical")
    return "_".join(parts)


def main() -> None:
    """Simulate every eye one camera recorded and write its simulated and measured apparent pupil center shift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", choices=CAMERAS, default="cam2", help="cam2 (eye-level, default) or cam1 (bottom-left)"
    )
    parser.add_argument(
        "--solve-physical",
        action="store_true",
        help="infer the physical pupil diameters and pupil center shift that reproduce the measured ones at the center"
        " target",
    )
    parser.add_argument(
        "--eye-parameters",
        type=Path,
        help="a fit_eye_parameters.py JSON, to simulate each eye with its fitted anatomy",
    )
    args = parser.parse_args()

    measured = load_simulator_input(args.camera)
    fitted = load_fitted_anatomy(args.eye_parameters, args.camera) if args.eye_parameters is not None else None
    if fitted is not None:
        for participant, eye in sorted(key for key in measured if key not in fitted):
            print(f"left out: {participant} {eye} has no fitted anatomy in {args.eye_parameters}")
        measured = {key: measured_eye for key, measured_eye in measured.items() if key in fitted}
    camera, illuminator = build_camera(args.camera), build_illuminator()
    rows = []
    for (participant, eye), measured_eye in tqdm(measured.items(), unit="eye"):
        model = build_eye_model(fitted[participant, eye] if fitted is not None else None)
        dilated_mm = measured_eye.apparent_pupil_diameter_dilated_center_target_mm
        constricted_mm = measured_eye.apparent_pupil_diameter_constricted_center_target_mm
        if args.solve_physical:
            dilated_mm = physical_pupil_diameter_mm(model, camera, eye, dilated_mm)
            constricted_mm = physical_pupil_diameter_mm(model, camera, eye, constricted_mm)
        simulated_eye = SimulatedEye(camera, illuminator, model, eye, dilated_mm, constricted_mm)
        measured_center = measured_eye.apparent_pupil_center_shift_mm[CENTER_TARGET]
        physical_shift_mm = (
            infer_physical_pupil_center_shift(simulated_eye, measured_center)
            if args.solve_physical
            else measured_center
        )
        simulated = {target: simulated_eye.apparent_change(target, physical_shift_mm) for target in TARGET_LABELS}
        for target, label in TARGET_LABELS.items():
            simulated_shift_mm, simulated_diameter_change_mm = simulated[target]
            rows.append((
                participant,
                eye,
                target,
                label,
                *(
                    measured_eye.apparent_pupil_center_shift_mm[target]
                    / measured_eye.apparent_pupil_diameter_change_mm[target]
                ),
                *(simulated_shift_mm / simulated_diameter_change_mm),
                simulated_diameter_change_mm,
                simulated_eye.dilated_physical_diameter_mm,
                simulated_eye.constricted_physical_diameter_mm,
                *simulated_eye.pcr_mm(target, simulated_eye.constricted_physical_diameter_mm, np.zeros(2)),
                *simulated_eye.pcr_mm(target, simulated_eye.dilated_physical_diameter_mm, physical_shift_mm),
            ))
        measured_center_rate = measured_center / measured_eye.apparent_pupil_diameter_change_mm[CENTER_TARGET]
        simulated_center_rate = simulated[CENTER_TARGET][0] / simulated[CENTER_TARGET][1]
        tqdm.write(
            f"{participant} {eye:<9} physical pupil diameter "
            f"dilated {simulated_eye.dilated_physical_diameter_mm:.2f} mm, "
            f"constricted {simulated_eye.constricted_physical_diameter_mm:.2f} mm; "
            f"center target, simulated against measured per mm of pupil diameter change: "
            f"magnitude ratio {np.linalg.norm(simulated_center_rate) / np.linalg.norm(measured_center_rate):.3f}, "
            f"angle {angle_between_deg(measured_center_rate, simulated_center_rate):.1f} deg"
        )

    path = result_path("pyetsimul", args.camera, run_name(args.eye_parameters, args.solve_physical))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_COLUMNS)
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
