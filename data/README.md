# Data — acquisition notes

The raw datasets are **NOT redistributed** in this repository (third-party
licenses + privacy of facial images). The `data/` directory is git-ignored.
This file explains how to obtain the data the experiments use.

## FERAC (facial expression images — Exp2 perception)

- Used by: Exp2 (emotion recognition) and the end-to-end walkthrough (image retrieval).
- Layout expected by the code:

  ```
  data/ferac/
  ├── train/{Natural,anger,fear,joy}/*.jpg
  └── test/{Natural,anger,fear,joy}/*.jpg
  ```

- 4 classes (Natural / Anger / Fear / Joy), ~770 images total.
- Obtain from the original FERAC source and place under `data/ferac/` with the
  layout above. We do not redistribute the images.

## Engagnition (engagement behavioural data — Exp1 calibration)

- Used by: Exp1 (LLM-ASD child simulator calibration targets).
- 38 records (LPE 19 + HPE 19) → K-means k=3 → persona targets P1/P2/P3.
- Obtain from the original Engagnition dataset source. We do not redistribute it.
- The derived, non-identifying calibration targets we computed are stored under
  `artifacts/exp1/calibration_final.json` (no raw subject data).

## What IS shipped in this repo

- **Model checkpoints** (`models/`, ~17 MB): trained surrogate / SAC / Exp2 perception weights.
- **Walkthrough trace** (`results/walkthrough.json`): a recorded end-to-end episode (no raw faces).
- Derived/aggregate artifacts under `artifacts/` (metrics, figures).

> Honest scope: all data use here is for a **simulation research framework**, not a
> clinical study. See the root `README.md`.
