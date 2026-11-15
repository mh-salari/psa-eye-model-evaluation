# Where the data files came from

Copied on 2026-08-18 from the recording project `psa-3cams-dark-bright-calibration`, which owns the recordings and every step that turns them into measurements. Re-copy after any change to the script named below.

| file | rows | written by |
| --- | --- | --- |
| `raw_eye_features.csv` | 3024 | `eye_images_processing/export_raw_eye_features_csv.py` |

It covers 21 participant-camera recordings: 11 participants, cam1 and cam2, less the one camera excluded below.

## What the file holds

`raw_eye_features.csv` is one row per camera, participant, eye, target and phase, as detected, with no alignment applied, plus two per-state median rows. Positions are in the detected image frame, so the glint still sits where it was found in that phase.

## Exclusions

The exporter applies `data/exclusions.json` in the source repository, so excluded recordings never reach this file:

- participant 935 is excluded outright: the dark calibration failed and the recording was stopped before usable data was collected
- participant 111 is excluded for cam2: the upper eyelid covered the pupil in the dark condition
