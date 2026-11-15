"""Fit each eye's anatomy in PyEtSimul to that eye's measured P-CR on one camera, in the constricted condition.

    uv run python simulators/fit_eye_parameters.py [--camera eye-level|bottom-left]

The simulator otherwise runs PyEtSimul's population eye. This fits, per eye, the corneal apex radius and the nasal and
superior off-axis pupil, within published human bounds, so that the simulated constricted P-CR matches the measured one
at all nine targets. Angle kappa is held at its population value, because a search of it did not converge. The held
values are written out with the fitted ones, so an output file describes the whole eye that was simulated. For every
anatomy the fit tries, the simulated pupil's physical diameter is the one whose apparent diameter at the center target
equals the measured constricted one, inferred as in the nine-target runs with --solve-physical.

Only the constricted condition is fitted: the apparent pupil center shift lives in the dilated-minus-constricted
difference, so a fit to the constricted P-CR cannot tune it.

Each fit matches one camera's P-CR at the nine targets, 9 x 2 = 18 constraints for the three fitted parameters, so the
simulations of a camera use eyes fitted to that camera's view alone. Eyes the camera did not record are left out and
named in the output.

Writes eye_parameters_<camera>.csv and .json to simulators/output/, <camera> being eye_level or bottom_left.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyetsimul.core import Camera, Light
from scipy.optimize import least_squares
from tqdm import tqdm

from psa_eye_model_evaluation.experiment_setup import CENTER_TARGET, TARGET_LABELS
from psa_eye_model_evaluation.pyetsimul_scene import (
    PUPIL_CENTER_METHOD,
    TARGET_POSITIONS,
    build_camera,
    build_eye,
    build_eye_model,
    build_illuminator,
    mm_per_pixel,
    physical_pupil_diameter_mm,
    population_anatomy,
)

FIT_INPUT = Path(__file__).resolve().parents[1] / "measured" / "output" / "fit_input.csv"
OUT = Path(__file__).resolve().parent / "output"
FIT_CAMERAS = {"eye-level": "cam2", "bottom-left": "cam1"}

# Published human mean and SD of each fitted parameter. Each is bounded to its mean +- BOUND_SDS SD, so a fitted value
# at a bound means the fit wants anatomy outside the range reported for human eyes.
#   anterior_radius: corneal apex radius, Table 2.2 of Atchison, D. A. (2023). Optics of the Human Eye (2nd ed.).
#       CRC Press.
#   off_axis_nasal_mm, off_axis_superior_mm: the pupil center's offset, photopic row (the fit uses the constricted
#       pupil), Table 1 of Yang, Y., Thompson, K., & Burns, S. A. (2002). Pupil location under mesopic, photopic, and
#       pharmacologically dilated conditions. Investigative Ophthalmology & Visual Science, 43(7), 2508-2512.
HUMAN_MEAN_SD = {
    "anterior_radius": (7.79, 0.25),
    "off_axis_nasal_mm": (0.202, 0.120),
    "off_axis_superior_mm": (0.087, 0.151),
}
BOUND_SDS = 3
BOUNDS = {name: (mean - BOUND_SDS * sd, mean + BOUND_SDS * sd) for name, (mean, sd) in HUMAN_MEAN_SD.items()}
FITTED = tuple(BOUNDS)
POPULATION = population_anatomy()
HELD = {name: POPULATION[name] for name in ("alpha_deg", "beta_deg")}
# A fitted value within this distance of a bound, in the parameter's units, counts as having reached it.
RAIL_TOLERANCE = 1e-3

type Views = dict[str, tuple[np.ndarray, float]]


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def load_fit_input() -> dict[tuple[str, str], Views]:
    """{(participant, eye): {camera: (9 x 2 constricted P-CR, center-target diameter)}}, in mm and image axes."""
    if not FIT_INPUT.is_file():
        raise SystemExit(f"{FIT_INPUT} is missing; run measured/simulator_input.py")
    rows: dict[tuple[str, str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    with FIT_INPUT.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["participant"], row["eye"], row["camera"]][int(row["target"])] = row
    views: dict[tuple[str, str], Views] = defaultdict(dict)
    for (participant, eye, camera), targets in sorted(rows.items()):
        if targets.keys() != TARGET_LABELS.keys():
            raise SystemExit(f"{camera} {participant} {eye} has targets {sorted(targets)}, expected all nine")
        pcr = np.array([
            [float(targets[t]["pcr_constricted_x_mm"]), float(targets[t]["pcr_constricted_y_mm"])]
            for t in TARGET_LABELS
        ])
        diameter = float(targets[CENTER_TARGET]["apparent_pupil_diameter_constricted_mm"])
        views[participant, eye][camera] = (pcr, diameter)
    return views


# ---------------------------------------------------------------------------
# Simulation and fit
# ---------------------------------------------------------------------------


def simulated_pcr(
    anatomy: dict[str, float], eye: str, camera: Camera, illuminator: Light, apparent_diameter_mm: float
) -> np.ndarray:
    """9 x 2 simulated constricted P-CR (mm at the eye, image axes), in target order.

    The physical pupil diameter is inferred for this anatomy, so that the apparent one at the center target equals
    ``apparent_diameter_mm``.
    """
    model = build_eye_model(anatomy)
    simulated = build_eye(model, eye)
    simulated.set_pupil_diameter(physical_pupil_diameter_mm(model, camera, eye, apparent_diameter_mm))
    scale = mm_per_pixel(camera, eye)
    pcr = []
    for target in TARGET_LABELS:
        simulated.look_at(TARGET_POSITIONS[target])
        image = camera.take_image(simulated, [illuminator], center_method=PUPIL_CENTER_METHOD)
        center, reflection = image.pupil_center, image.corneal_reflections[0]
        pcr.append(scale * np.array([center.x - reflection.x, center.y - reflection.y]))
    return np.array(pcr)


def pcr_error(
    anatomy: dict[str, float], eye: str, views: Views, cameras: dict[str, Camera], illuminator: Light
) -> np.ndarray:
    """Flat simulated-minus-measured P-CR residual in mm, over every camera of ``cameras`` and all nine targets."""
    parts = []
    for name, camera in cameras.items():
        pcr, diameter = views[name]
        parts.append((simulated_pcr(anatomy, eye, camera, illuminator, diameter) - pcr).ravel())
    return np.concatenate(parts)


def rms(flat: np.ndarray) -> float:
    """Root-mean-square P-CR error in mm from a flat vector of (x, y) pairs."""
    return float(np.sqrt(np.mean(np.sum(flat.reshape(-1, 2) ** 2, axis=1))))


def railed(name: str, value: float) -> str | None:
    """Which bound a fitted value has reached, or None when it lies inside its range."""
    low, high = BOUNDS[name]
    if value - low <= RAIL_TOLERANCE:
        return "lower"
    if high - value <= RAIL_TOLERANCE:
        return "upper"
    return None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def write_results(stem: str, results: dict[str, dict]) -> None:
    """The fit as JSON, read by the nine-target script, and as a CSV table."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (OUT / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "participant",
            "eye",
            "camera",
            *POPULATION,
            *(f"{name}_railed" for name in POPULATION),
            "rms_before_um",
            "rms_after_um",
        ))
        for row in results.values():
            writer.writerow([
                row["participant"],
                row["eye"],
                row["camera"],
                *(f"{row['parameters'][name]:.6f}" for name in POPULATION),
                *(row["railed"].get(name, "held" if name in HELD else "") for name in POPULATION),
                f"{row['rms_before_mm'] * 1000:.2f}",
                f"{row['rms_after_mm'] * 1000:.2f}",
            ])
    print(f"\nwrote {OUT / f'{stem}.json'}\nwrote {OUT / f'{stem}.csv'}")


def main() -> None:
    """Fit every eye recorded by the chosen camera, stream each fit, and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", choices=tuple(FIT_CAMERAS), default="eye-level", help="the camera whose P-CR the fit matches"
    )
    args = parser.parse_args()
    camera_name = FIT_CAMERAS[args.camera]
    cameras = {camera_name: build_camera(camera_name)}
    illuminator = build_illuminator()

    views = load_fit_input()
    for participant, eye in sorted(key for key, eye_views in views.items() if camera_name not in eye_views):
        print(f"left out: {participant} {eye} is not recorded by the {args.camera} camera")
    jobs = [
        (participant, eye, eye_views)
        for (participant, eye), eye_views in sorted(views.items())
        if camera_name in eye_views
    ]
    print(
        f"Fitting {len(FITTED)} parameters for {len(jobs)} eyes on the {args.camera} camera "
        f"({2 * len(TARGET_LABELS)} constraints each); angle kappa held at {HELD}",
        flush=True,
    )

    lower = np.array([BOUNDS[name][0] for name in FITTED])
    upper = np.array([BOUNDS[name][1] for name in FITTED])
    start = np.array([POPULATION[name] for name in FITTED])
    results = {}
    bar = tqdm(jobs, unit="eye")
    for participant, eye, eye_views in bar:
        bar.set_description(f"{participant} {eye}")
        before = pcr_error(POPULATION, eye, eye_views, cameras, illuminator)
        evaluations = {"n": 0}

        def objective(
            vector: np.ndarray, eye: str = eye, eye_views: Views = eye_views, evaluations: dict[str, int] = evaluations
        ) -> np.ndarray:
            anatomy = {**POPULATION, **dict(zip(FITTED, (float(v) for v in vector), strict=True))}
            flat = pcr_error(anatomy, eye, eye_views, cameras, illuminator)
            evaluations["n"] += 1
            bar.set_postfix_str(f"eval {evaluations['n']:3d}  rms {rms(flat) * 1000:6.1f} um")
            return flat

        solution = least_squares(objective, start, bounds=(lower, upper))
        values = {**POPULATION, **{name: float(v) for name, v in zip(FITTED, solution.x, strict=True)}}
        rails = {name: side for name in FITTED if (side := railed(name, values[name])) is not None}
        results[f"{participant}/{eye}"] = {
            "participant": participant,
            "eye": eye,
            "camera": camera_name,
            "parameters": values,
            "held": dict(HELD),
            "railed": rails,
            "rms_before_mm": rms(before),
            "rms_after_mm": rms(np.asarray(solution.fun)),
            "n_evaluations": evaluations["n"],
        }
        summary = "  ".join(f"{name}={values[name]:+.3f}{'!' if name in rails else ''}" for name in FITTED)
        tqdm.write(
            f"{participant} {eye:<9} rms {rms(before) * 1000:6.1f} -> "
            f"{rms(np.asarray(solution.fun)) * 1000:6.1f} um   {summary}"
        )
    bar.close()

    write_results(f"eye_parameters_{args.camera.replace('-', '_')}", results)
    rows = list(results.values())
    print(
        f"\nP-CR error over {len(rows)} eyes, median: {np.median([r['rms_before_mm'] for r in rows]) * 1000:.1f} -> "
        f"{np.median([r['rms_after_mm'] for r in rows]) * 1000:.1f} um"
    )
    for name in FITTED:
        fitted_values = [r["parameters"][name] for r in rows]
        low, high = BOUNDS[name]
        print(
            f"  {name:<22} median {np.median(fitted_values):+.3f}  "
            f"range {min(fitted_values):+.3f}..{max(fitted_values):+.3f}  bounds {low:+.3f}..{high:+.3f}  "
            f"railed {sum(1 for r in rows if name in r['railed'])}/{len(rows)}"
        )
    for name, value in HELD.items():
        print(f"  {name:<22} held   {value:+.3f}")


if __name__ == "__main__":
    main()
