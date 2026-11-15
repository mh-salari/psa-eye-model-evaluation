"""The change from the center target in the measured apparent pupil center shift and in each simulator's.

    uv run python simulators/apparent_pupil_center_shift_9targets_summary.py

For every camera and eye, over participants: the median direction change over all participants and the eight
eccentric targets, the range across the eight targets of the per-target median direction change, the same range for
the magnitude change, and the median magnitude change at the nasal and at the temporal targets, in mm per mm and as a
percentage of the magnitude at the center target. Each median carries its median absolute deviation (MAD) over the same
values in parentheses. For each simulator, the measured median direction change is also given
as a multiple of the simulated one, and the simulated one as a percentage of the measured one. The simulated median
magnitude change at the nasal and at the temporal targets is given as a percentage of the measured one, in absolute
value, because the two have opposite signs. At the center target, each simulator's median per-eye ratio of the
magnitudes of the simulated and the measured apparent pupil center shift and the median angle between the two are
printed as well, and, for the measurement and each simulator, the share of the differences from the center target that
point back toward it. Left and right eyes are never pooled.
"""

from psa_eye_model_evaluation.apparent_pupil_center_shift_change import (
    ChangeSummary,
    share_toward_center_target,
    summarize_changes_by_eye,
)
from psa_eye_model_evaluation.experiment_setup import CAMERA_LABEL, EYE_LABEL, EYES, PANEL_CAMERAS
from psa_eye_model_evaluation.recordings import load_measured_apparent_pupil_center_shift
from psa_eye_model_evaluation.simulator_results import SIMULATORS, center_target_agreement, load_simulator_result_file

HEADER = (
    f"{'':<24}{'direction change (deg)':^38}{'magnitude change (mm per mm, MAD, % of center-target magnitude)':^72}\n"
    f"{'':<24}{'median (MAD)':>14}{'x':>7}{'per-target range':>17}{'per-target range':>20}{'nasal':>26}"
    f"{'temporal':>26}{'eyes':>6}"
)


def summary_line(label: str, summary: ChangeSummary, factor: str = "") -> str:
    """One printed row of a change summary; ``factor`` is the measured median direction change over this one."""
    direction_low, direction_high = summary.direction_change_range
    magnitude_low, magnitude_high = summary.magnitude_change_range
    return (
        f"{label:<24}{summary.median_direction_change:8.1f} ({summary.direction_change_mad:3.1f}){factor:>7}"
        f"{direction_low:10.1f} to {direction_high:4.1f}{magnitude_low:+11.4f} to {magnitude_high:+.4f}"
        f"{summary.nasal_magnitude_change:+10.4f} ({summary.nasal_magnitude_change_mad:.4f}, "
        f"{summary.nasal_magnitude_change_percent:+4.0f}%)"
        f"{summary.temporal_magnitude_change:+10.4f} ({summary.temporal_magnitude_change_mad:.4f}, "
        f"{summary.temporal_magnitude_change_percent:+4.0f}%)"
        f"{summary.n_eyes:6d}"
    )


def main() -> None:
    """Print the measured and each simulator's change from the center target, and the center-target agreement."""
    measured = load_measured_apparent_pupil_center_shift()
    for camera in PANEL_CAMERAS:
        recorded = {key for key in measured if key[0] == camera}
        simulated = {simulator: load_simulator_result_file(simulator, camera) for simulator in SIMULATORS}
        for simulator, shifts in simulated.items():
            if shifts.keys() != recorded:
                raise SystemExit(
                    f"the {SIMULATORS[simulator]} result for {camera} covers other eyes or targets than the recordings"
                )
        measured_summaries = summarize_changes_by_eye(measured, camera)
        simulated_summaries = {
            simulator: summarize_changes_by_eye(shifts, camera) for simulator, shifts in simulated.items()
        }

        print(f"\n{CAMERA_LABEL[camera]}\n{HEADER}")
        for eye in EYES:
            print(summary_line(f"{EYE_LABEL[eye]}, measured", measured_summaries[eye]))
            for simulator, name in SIMULATORS.items():
                summary = simulated_summaries[simulator][eye]
                factor = measured_summaries[eye].median_direction_change / summary.median_direction_change
                print(summary_line(f"{EYE_LABEL[eye]}, {name}", summary, f"{factor:.1f}x"))

        print("\nSimulated median direction change as a percentage of the measured one")
        for eye in EYES:
            for simulator, name in SIMULATORS.items():
                share = (
                    100
                    * simulated_summaries[simulator][eye].median_direction_change
                    / measured_summaries[eye].median_direction_change
                )
                print(f"{EYE_LABEL[eye] + ', ' + name:<24}{share:8.1f}%")

        print("\nSimulated median magnitude change as a percentage of the measured one, in absolute value")
        print(f"{'':<24}{'nasal':>9}{'temporal':>9}")
        for eye in EYES:
            for simulator, name in SIMULATORS.items():
                summary = simulated_summaries[simulator][eye]
                nasal = 100 * abs(summary.nasal_magnitude_change / measured_summaries[eye].nasal_magnitude_change)
                temporal = 100 * abs(
                    summary.temporal_magnitude_change / measured_summaries[eye].temporal_magnitude_change
                )
                print(f"{EYE_LABEL[eye] + ', ' + name:<24}{nasal:8.1f}%{temporal:8.1f}%")

        print("\nDifferences from the center target pointing back toward it, over participants and eccentric targets")
        for eye in EYES:
            print(f"{EYE_LABEL[eye] + ', measured':<24}{share_toward_center_target(measured, camera, eye):8.0f}%")
            for simulator, name in SIMULATORS.items():
                share = share_toward_center_target(simulated[simulator], camera, eye)
                print(f"{EYE_LABEL[eye] + ', ' + name:<24}{share:8.0f}%")

        print("\nCenter target: median per-eye magnitude ratio simulated / measured, and median angle between them")
        for eye in EYES:
            for simulator, name in SIMULATORS.items():
                ratio, angle, n_eyes = center_target_agreement(measured, simulated[simulator], camera, eye)
                print(f"{EYE_LABEL[eye] + ', ' + name:<24}{ratio:8.3f}{angle:8.1f} deg{n_eyes:6d} eyes")


if __name__ == "__main__":
    main()
