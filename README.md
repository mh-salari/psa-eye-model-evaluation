# psa-eye-model-evaluation

Code and data of a manuscript that is under peer review.

> **Not licensed for use.** This repository is public only so that the editor and the reviewers of the manuscript can inspect and run its contents. Until the manuscript is published, all rights are reserved, and the code and the data may not be used, copied, modified or redistributed for any other purpose. An open-source license will replace this notice when the paper is published.

This repository holds the measurements of the apparent pupil center shift at nine gaze targets, and the test of whether three eye-tracking simulators (PyEtSimul, et_simul and gkaModelEye) reproduce it.

## Contents

| Folder | What it holds |
| --- | --- |
| `data/` | The measured eye features of all participants (`raw_eye_features.csv`), and `SOURCE.md`, which describes the file and where it comes from. |
| `psa_eye_model_evaluation/` | Definitions shared by all scripts: the experiment setup, the measured data, the setup as built in PyEtSimul, and the figure helpers. |
| `measured/` | Analyses and figures of the measurements. Results are in `measured/output/`. |
| `simulators/` | The simulations and their comparison with the measurements. Results are in `simulators/output/`. The MATLAB packages `+et_simul_scene/` and `+gkamodeleye_scene/` build the setup in et_simul and in gkaModelEye. |

The two `output/` folders hold the results reported in the manuscript. Running a script overwrites its outputs.

## Requirements

Python 3.12 and [uv](https://docs.astral.sh/uv/). All code was written and run on macOS; it is not tested on other operating systems. The dependencies, including PyEtSimul 6.0.0, are installed with:

```
uv sync
```

The et_simul and gkaModelEye scripts need MATLAB with the Optimization Toolbox (the manuscript used MATLAB R2026a) and the two simulators, which are not part of this repository:

- et_simul version 1.01, archived at <https://github.com/mh-salari/et_simul-1.01>, placed in `simulators/et_simul-1.01/`.
- gkaModelEye commit 15cabea (20 November 2025) from <https://github.com/gkaguirrelab/gkaModelEye>, placed in `simulators/gkaModelEye/`, with a sign bug in its camera rotation matrix fixed (see <https://github.com/gkaguirrelab/gkaModelEye/pull/39>).

`measured/pupil_size.R` needs R.

## Running the scripts

Python scripts are run from the repository root, for example:

```
uv run python measured/simulator_input.py
```

Each script starts with a description of what it computes, what it reads and what it writes.

### Measurements (`measured/`)

| Script | What it does |
| --- | --- |
| `simulator_input.py` | Writes the measured input for the simulator scripts. |
| `center_target_summary.py` | The measured apparent pupil center shift at the center target: magnitude, direction, and agreement between cameras. |
| `apparent_pupil_center_shift_by_target.py` | How the measured apparent pupil center shift changes from the center target to each eccentric target. |
| `pupil_size.py`, `pupil_size.R` | The apparent pupil diameter in the dilated and the constricted condition. |
| `plot_apparent_pupil_center_shift_center_target_measured.py` | Figure of the measured apparent pupil center shift at the center target. |
| `plot_apparent_pupil_center_shift_difference_by_target.py` | Figures of the difference between each eccentric target and the center target, for the measurements and, with `--simulators`, for the simulators. |
| `plot_apparent_pupil_center_shift_difference_construction.py` | Figure of how this difference is built, for one participant. |
| `wildenmann_comparison.py` | The pupil center shift per mm of pupil diameter reported by Wildenmann and Schaeffel (2013), for comparison. |

### Simulations (`simulators/`)

| Script | What it does |
| --- | --- |
| `apparent_pupil_center_shift_9targets_pyetsimul.py` | The apparent pupil center shift that PyEtSimul produces at the nine targets. Options select the camera (`--camera cam1`), the first additional test (`--solve-physical`) and the second additional test (`--solve-physical --eye-parameters <file>`). |
| `fit_eye_parameters.py` | Fits each eye's anatomy in PyEtSimul to that eye's measured P-CR on one camera (`--camera eye-level` or `--camera bottom-left`), for the second additional test. |
| `apparent_pupil_center_shift_9targets_et_simul.m`, `apparent_pupil_center_shift_9targets_gkamodeleye.m` | The same simulation in et_simul and in gkaModelEye. |
| `fit_eye_parameters_et_simul.m`, `fit_eye_parameters_gkamodeleye.m` | The same fit in et_simul and in gkaModelEye. |
| `apparent_pupil_center_shift_9targets_summary.py` | The change from the center target in the measured apparent pupil center shift and in each simulator's. |
| `additional_tests_summary.py` | Each simulator's two additional tests against its default eye and the measurements. |
| `psa_equivalent.py` | The PSA that a P-CR eye tracker reports, from the measurements and from each simulator. |
| `apparent_pupil_center_shift_center_target.py` | Figure of the apparent pupil center shift at the center target, measured and simulated. |
| `check_setup_pyetsimul.py` | A 3D drawing of the simulated setup, to check it against the real one. |

## Data

`data/raw_eye_features.csv` holds the pupil center, the corneal reflection, the pupil-corneal reflection vector and the pupil diameter for each camera, participant, eye, target and recording phase, as annotated in the eye images. It is produced by the recording project [psa-3cams-dark-bright-calibration](https://github.com/mh-salari/psa-3cams-dark-bright-calibration). Participants are identified by random three-digit numbers. The eye images and the recordings themselves are personal data and are not public.

## License

All rights reserved until the manuscript is published, see `LICENSE`.
