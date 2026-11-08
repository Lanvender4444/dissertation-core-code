# Code and evidence package for the dissertation

*Multi-View Object Verification and Post-Detection Fusion for a Vehicle-Mounted Pinhole–Fisheye Camera System* (Zhang Chengwei, NTU EEE MSc Computer Control and Automation, 2026)

This package is the code attachment of the dissertation, and every path quoted in **Appendix A (Evidence Map and Reproduction)** is relative to its root. The folder also holds every source file named in Appendices B and C (Table C.1 maps each implementation component to its files), together with the result folders of the supplementary experiments and the viewer audits cited in Appendices B and C. All files were copied byte for byte from the project repository and keep their original relative paths, so any file here can be compared with, or put back into, the full repository.

The package holds code and evidence. The 79 GB image set, the annotations, the calibration files, the detector weights and the intermediate caches stay in the repository; Section 2 gives the location of each.

---

## Contents

1. [Directory layout](#1-directory-layout)
2. [Data, calibration, weights and caches](#2-data-calibration-weights-and-caches)
3. [Environments](#3-environments)
4. [How to run](#4-how-to-run)
5. [Pipeline stages and the code that implements them](#5-pipeline-stages-and-the-code-that-implements-them)
6. [File and function reference](#6-file-and-function-reference)
7. [Evidence files and the thesis results they support](#7-evidence-files-and-the-thesis-results-they-support)
8. [Integrity rules built into the code](#8-integrity-rules-built-into-the-code)
9. [Scope notes](#9-scope-notes)

---

## 1. Directory layout

```text
<package root>/
├── README.md                         this file
├── THESIS_DATA_LEDGER.json           machine-readable data ledger (file index, SHA-256, XML checks, splits)
├── launch_viewers.py                 menu launcher for all viewers (Appendix C.1)
├── start_viewers.cmd                 Windows double-click wrapper for launch_viewers.py
│
├── portable/                         standalone demo package; runs without the dataset (named code/ in the repository)
│   ├── demo.py, fusion_core.py, verify.py, run_project.py
│   ├── tests/test_submission.py      14 contract and replay tests
│   ├── models/recovery_single.json   exported F5-P singleton recovery head (numeric trees)
│   ├── examples/*.json|*.npz         prepared features of frames 00000641, 00000650, 00001200
│   ├── metadata/                     source manifest, extraction record, hashes
│   ├── original_layout/              100-file source snapshot with all local imports (Section 6.8)
│   └── README.md, SOURCE_MAP.md, EXTERNAL_DATA.md, THIRD_PARTY_NOTICES.md, pyproject.toml, uv.lock
│
├── scripts/
│   ├── supp/                         supplementary experiments E1–E7 (Appendix B)
│   ├── experiment_five_camera_capacity_recovery.py   F5-R stage 1: transferred recovery heads
│   ├── experiment_five_camera_repair_recovery.py     F5-R stage 2: bounded repair + recovery (final)
│   ├── protected_conditional_inputs.py               signed input loader for F5-P
│   ├── fit_protected_conditional_recovery.py         fits the F5-P recovery arms (early segment)
│   ├── evaluate_protected_conditional_recovery.py    development lock + full replay of F5-P
│   ├── audit_protected_conditional_recovery.py       independent recount of F5-P
│   ├── audit_thesis_recovery_intervals.py            paired block-bootstrap intervals
│   └── test_viewer_launcher.py, test_surround_dashboard_v09.py   viewer regression tests
│
├── ultralytics/
│   ├── pyproject.toml, uv.lock       uv environment of the full project
│   ├── core_algorithms/run_depth_bev.py      three-camera depth/BEV driver incl. the E1 rule `dedup3d`
│   ├── research/                     scene-level decision modules and surround rendering
│   ├── decided/                      viewer entry points used in Chapter 5
│   └── algorithm/
│       ├── *.md                      experiment notes, one per study
│       └── evaluation_reports/*.json signed results and independent audits
│
├── thesis/                           writing-evidence snapshot, audits, figure manifests, F5-R summary
└── docs/archive/root_markdown_20261020/
    ├── DISPLAY_ENTRY_POINTS.md       viewer commands and key bindings (in Chinese)
    └── THESIS_DATA_LEDGER.md         human-readable data ledger (in Chinese)
```

This package root is laid out like the project repository: `scripts/`, `ultralytics/`, `thesis/` and `docs/` sit at the same relative paths as in the repository. The one exception is `portable/`, the small self-contained package that the repository keeps at `code/`; it was renamed here so that the root folder and the package do not share a name. Files inside `portable/` (its own README and SOURCE_MAP) still call it `code/`, and its commands written for the repository root read `--project code`; from the package root use `--project portable`.

Several modules import neighbours that are not copied as separate files here (for example `retained_reference.py`, `residual_recovery.py`, `joint_identity_admission.py`, `experiment_road_support_admission.py`). Their exact sources are in `portable/original_layout/`, which keeps each file's original path and bytes; `portable/metadata/source_manifest.json` records the import graph and the SHA-256 of every file.

---

## 2. Data, calibration, weights and caches

All paths in this section are relative to the **repository root** (`D:\Yolo-Dissertation` on the author's machine).

### 2.1 Images and annotations

| Item | Location | Format and size |
|---|---|---|
| Native camera images | `ultralytics/mycar/images/{1,2,3,4,5}/{frame}.png` | PNG, 2464 × 2056 px; 1,655 frames per camera, 8,275 images, about 79 GB |
| Per-camera annotations | `ultralytics/mycar/cam_{1,2,3,4,5}/{frame}.xml` | one XML file per image: boxes, class labels and instance IDs |
| Original recording archive | `(2024-04-29) One-North Route 1 Day-011.zip` | about 52 GB; the supplied One-North Route 1 recording |
| LiDAR files supplied with the recording | `lidar/` | about 17 MB; not used by the reported experiments |
| Provider's format description | `Inputs and Outputs Format.pdf` | — |

Folder `images/1` … `images/5` and `cam_1` … `cam_5` correspond to Cam1 … Cam5. Frame names are eight-digit indices (`00000000` … `00001700`); the supplied pairing covers 1,655 of the indices 0–1700, and frames are paired across cameras by this index.

Camera roles: **Cam2, Cam3, Cam4** are forward pinhole cameras; **Cam1** and **Cam5** are lateral fisheye cameras following the Mei unified model.

Annotation counts (from `THESIS_DATA_LEDGER.json`):

| Camera | PNG | XML | Annotated boxes | Notes |
|---|---:|---:|---:|---|
| Cam1 | 1,655 | 1,655 | 9,242 | |
| Cam2 | 1,655 | 1,655 | 14,890 | |
| Cam3 | 1,655 | 1,655 | 14,843 | |
| Cam4 | 1,655 | 1,655 | 14,798 | |
| Cam5 | 1,655 | 1,655 | 11,201 | 29 zero-height boxes (id 39 in frames 00000303–00000330, id 5 in 00001395), kept as supplied |

Evaluation unions (each object-frame instance counted once per frame, with at least one box 25 px or taller, identities taken as shared across cameras): **13,473** instances for Cam234 and **24,845** for all five cameras. The ledger also lists 5 (Cam234) and 7 (Cam12345) identities whose class differs between cameras, and boxes that extend past the image border.

### 2.2 Calibration

| File | Content | Loaded by |
|---|---|---|
| `Intrinsics_cam.json` | intrinsics and distortion of the five cameras, including the Mei parameter ξ of Cam1 and Cam5 | `ultralytics/core_algorithms/multiview_geometry.py` (`calibration_list(ROOT/'Intrinsics_cam.json')`) |
| `T_imu_cam(1).json` | camera-to-IMU extrinsics `T_imu_cam`: rotation R_c and centre t_c of Section 3.3 | same loader |

### 2.3 Detector weights

| File | Use |
|---|---|
| `ultralytics/yolo11n.pt` | the fixed YOLO11n detector with its released weights. It is needed only for fresh inference; the cached-replay viewers and every evaluation in the thesis read the signed detection caches. |

### 2.4 Intermediate caches and frozen models of the five-camera pipeline

These also appear in `portable/EXTERNAL_DATA.md`.

| Stage | Location in the repository |
|---|---|
| Crop policy (two 1312 × 568 road-band crops per camera) | `ultralytics/experiments/budget_fovea_20261009/probe01/policy.json` |
| Prepared geometry and detector cache (confidence floor 0.05) | `ultralytics/experiments/budget_fovea_20261009/fusion01/shard*/frames/` |
| ROI descriptor cache | `ultralytics/experiments/budget_fovea_20261009/detector_roi01/shard*/frames/` |
| Frozen output-box calibration | `ultralytics/algorithm/evaluation_reports/box_error_calibration_results_20261009.json` |
| Upstream group model and protocol (group score, 1,056 features) | `ultralytics/experiments/budget_fovea_20261009/detector_roi_admission01/` |
| Reliable-reference upstream report | `ultralytics/algorithm/evaluation_reports/detector_roi_admission_results_20261009.json` |
| Residual reference model (672 features) | `ultralytics/experiments/retained_reference_20261010/run01/` |
| Cross-fitted references of the early segment | `ultralytics/experiments/protected_conditional_20261010/run01/reference01/` |
| Pairwise identity head | `ultralytics/experiments/identity_pair_20261010/early01/` |
| F5-P recovery model, protocol and signed per-frame records | `ultralytics/experiments/protected_conditional_20261010/run01/recovery01/` |
| F5-R stage 1 (transferred heads) | `ultralytics/experiments/five_camera_capacity_recovery_20261017/run01/` |
| F5-R stage 2 (repair + recovery, final configuration) | `ultralytics/experiments/five_camera_repair_recovery_20261017/run01/` |
| Supplementary snapshot and results E1–E7 | `ultralytics/experiments/supp_20261101/` |

Each protocol records the SHA-256 of its sources and inputs, and the loaders refuse a cache whose hashes differ (Section 8).

---

## 3. Environments

| Environment | Definition | Contents | Used for |
|---|---|---|---|
| Portable | `portable/pyproject.toml`, `portable/uv.lock` | Python 3.12, NumPy 1.26.4, SciPy 1.17.1 | `portable/demo.py`, tests, `verify.py` |
| Full project | `ultralytics/pyproject.toml`, `ultralytics/uv.lock` | Python 3.12.1, Ultralytics 8.3.184, opencv-python 4.11.0.86, NumPy 1.26.4, SciPy 1.17.1, PyTorch 2.14.0 | viewers and experiment runners |
| Recovery fitting | inside the full project | scikit-learn **1.7.2** exactly (`nonlinear_admission.fit` checks the version) | refitting any boosted head |
| Supplementary experiments | Python ≥ 3.10 with NumPy, SciPy, scikit-learn 1.7.2, matplotlib | — | `scripts/supp/*` |

---

## 4. How to run

### 4.1 Portable check (no dataset needed)

From the package root:

```powershell
uv run --project portable python portable/demo.py                       # replays 3 frames: score -> select -> count
uv run --project portable python -m unittest discover -s portable/tests -v
uv run --project portable python portable/verify.py                     # hash audit of the package
```

Expected `demo.py` result: frame 00000641 goes from TP/FP/FN 16/5/31 (reference A) to 17/5/30 with one recovered object; frames 00000650 and 00001200 keep 20/4/25 and 4/0/2. Recovery scores match the signed originals to 1e-12.

### 4.2 Viewers (full repository with images, calibration and caches)

Run inside the repository root:

```powershell
uv run --offline --no-sync --project ultralytics python launch_viewers.py                              # menu
uv run --offline --no-sync --project ultralytics python launch_viewers.py --view surround --mode step  # look-around (Section 5.1)
uv run --offline --no-sync --project ultralytics python launch_viewers.py --view five --mode step      # five native views + panorama + BEV
uv run --offline --no-sync --project ultralytics python launch_viewers.py --view bev --mode step       # bowl surround + textured BEV
uv run --offline --no-sync --project ultralytics python launch_viewers.py --view three --mode step     # Cam2/3/4 + forward fusion (F3)
```

Look-around keys (`final_immersive_cam.py`): drag to turn, wheel to zoom, `M` perspective, `P` sphere, `C` native cameras, `G` BEV, `1`–`4` front/left/rear/right, `R` reset, `F` full screen, `T` BEV inset, `N`/`B` next/previous frame, `Space` play/pause, `Q`/`Esc` quit. The complete list is in `docs/archive/root_markdown_20261020/DISPLAY_ENTRY_POINTS.md`.

The five-camera viewers replay **F5-P** (P 90.45 %, R 42.69 %). F5-R comes from the research runner in Section 4.3 and uses the same object record.

### 4.3 Final configuration F5-R

```powershell
uv run --project ultralytics python scripts/experiment_five_camera_capacity_recovery.py --stage <stage>   # stage 1
uv run --project ultralytics python scripts/experiment_five_camera_repair_recovery.py --stage early        # protocol lock + 200 development frames
uv run --project ultralytics python scripts/experiment_five_camera_repair_recovery.py --stage later        # full 1,655-frame replay, audit, report
```

The runners write to append-only directories and stop if the run directory already exists, so a reproduction uses a new run directory.

### 4.4 Supplementary experiments E1–E7 (Appendix B)

Order of execution, from `scripts/supp/`:

```bash
python3 extract.py          # read-only extraction of the frozen per-frame snapshot (resumable)
python3 selfcheck.py        # must reproduce A 10366/1044, F5-P 10606/1120, F5-R 11143/1124
python3 e1_e6_error_analysis.py
python3 e2_simple_baselines.py
python3 e3_feature_ablation.py
python3 e3b_fisheye_output.py
python3 e4_association.py
python3 e4_audit_sample.py && python3 e4_render_audit.py CASES OUTDIR BUDGET_S && python3 e4_audit_score.py
python3 e5_ap.py
python3 e7_learned_baseline.py   # learned verifier without scene context + bootstrap block-length check
python3 make_figures.py && python3 make_tables.py
```

Results are written to `ultralytics/experiments/supp_20261101/`. This package includes those results (all runs, figures and the identity-check labels); only the 71 MB per-frame snapshot written by `extract.py` (`extract/`) is left out and is regenerated from the repository data.

---

## 5. Pipeline stages and the code that implements them

| Stage (thesis section) | What happens | Main code |
|---|---|---|
| Detector input (4.4) | YOLO11n on two road-band crops per camera, each resized to 960 × 416 and stacked to 960 × 832; seam-crossing boxes dropped; boxes mapped to native coordinates | `portable/original_layout/ultralytics/research/budget_fovea.py`, `crop_measurement.py` |
| Candidate preparation, box calibration (4.5) | ego exclusion; frozen learned correction of detector boxes towards the annotation convention | `.../core_algorithms/final_pipeline.py`, `.../research/box_error_calibration.py` |
| Projection and rays (3.3) | pinhole and Mei projection and back-projection in the IMU frame | `.../core_algorithms/multiview_geometry.py`, `.../research/shared_geometry.py` |
| Forward association, stereo, E1 (4.6–4.8) | triple/pair/singleton grouping, SGBM box-local depth, 23-feature logistic verification, duplicate rule E1 | `.../core_algorithms/tri-compare.py`, `stereo_depth.py`, `depth_bev_fusion.py`; `ultralytics/core_algorithms/run_depth_bev.py` (`build`, `predict`, `dedup3d`) |
| Lateral evidence and reference A (4.9) | pinhole–Mei pair features and pair head; group score ≥ 0.70 plus residual singletons ≥ 0.55 | `.../research/identity_pair_features.py`, `joint_identity_admission.py`; `ultralytics/research/protected_reference.py` (`assemble`) |
| F5-P recovery (4.10–4.12) | singleton candidates from unused observations, 24 context features, admission at 0.55, append-only | `ultralytics/research/protected_conditional_recovery.py`, `nonlinear_admission.py`; `scripts/fit_/evaluate_/audit_protected_conditional_recovery.py` |
| F5-R bounded repair (4.15) | at most one edit per frame ranked by J(a) over five gain/harm model pairs, edit threshold 0.05 | `scripts/experiment_five_camera_repair_recovery.py` (`Inputs.frame`), `.../research/shared_edit_uncertainty.py` |
| F5-R current-scene recovery (4.15) | greedy rollout of the 575-feature residual head on the repaired scene, τ = 0.5559089514906035 | `ultralytics/research/five_camera_capacity_recovery.py` (`features`, `rollout`) |
| Evaluation (6.1) | any-member compatible-class IoU ≥ 0.5, cardinality-first linear assignment, 25 px ignore rule | `.../core_algorithms/validate_multiview.py` (`evaluate`); `scripts/supp/supp_lib.py` (`evaluate`) |
| Displays (Chapter 5) | look-around, sphere, panorama, bowl and BEV, native pages | `ultralytics/research/immersive_surround.py`, `immersive_dashboard.py`; `ultralytics/decided/*` |

`...` stands for `portable/original_layout/ultralytics`.

---

## 6. File and function reference

### 6.1 `portable/` — standalone demo package

**`demo.py`** — replays three bundled frames from prepared features to final counts; ground truth plays no part in admission.
- `digest(path)` — SHA-256 of a file; confirms the bundled `.npz` fixture is unchanged.
- `replay(frame)` — loads the frame's features, predicts recovery scores with `models/recovery_single.json`, compares them with the stored scores (maximum error 1e-12), selects additions at τ = 0.55, checks that reference A is unchanged, then counts TP/FP/FN on the stored IoU graph and checks that matching capacity did not fall. Returns counts and P/R/F1 for the frame.

**`fusion_core.py`** — function bodies extracted unchanged from the research modules; `metadata/extractions.json` lists their source lines and hashes.
- `compile_model(model)` — converts an exported JSON tree model into NumPy arrays.
- `predict(xs, model)` — evaluates the exported boosted trees: baseline logit plus leaf values, then the sigmoid.
- `objects(groups, pool)` — turns groups of observation IDs into object records (source cameras, class, member detections).
- `validate(groups, pool)` — source exclusivity: every group non-empty, every observation used once and known, at most one member per camera.
- `append(reference, extra, pool)` — appends recovered groups to the protected reference and keeps every existing group as it is.
- `overlap(a, b)` — IoU of two boxes.
- `conflicts(a, b, pool, probabilities)` — conservative duplicate test between two groups: with a shared camera it needs IoU ≥ 0.7 there and all pair scores ≥ 0.95; with no shared camera all pair scores ≥ 0.99.
- `select(reference, groups, scores, tau, pool, probabilities, suppress)` — ranks candidates by score, admits those with score ≥ τ, optionally retires batch duplicates, and returns the appended scene with a trace.
- `counts(self, objects)` — offline evaluator with cardinality-first one-to-one matching and the small-annotation ignore rule; returns TP, FP, FN and two diagnostic counts.

**`verify.py`** — `sha(path)` hashes a file; `verify(project)` checks every payload file against `metadata/` and, given a repository root, checks that the repository sources still match the snapshot.

**`run_project.py`** — starts one of the full-project entries (`three-no-band`, `three-band`, `cam2`, `cam3`, `cam4`, `five`, `spatial`, `immersive`) after `verify(root)` succeeds and the image folder exists; `--check-only` runs the verification alone.

**`tests/test_submission.py`** — class `RecoveryContracts` with 14 tests: reference immutability, acceptance with unknown location, refusal of protected, duplicate or unknown observation IDs, one member per camera, threshold inclusivity, non-additive marginal labels, the small-GT ignore rule and the three frozen replays.

### 6.2 `ultralytics/research/` — scene-level decisions and rendering

**`protected_reference.py`** — builds and protects reference A; takes no ground-truth input.
- `objects(groups, pool)` — as in `fusion_core`.
- `validate(groups, pool)` — source exclusivity and one active box per camera; returns the set of used observation IDs.
- `contextualize(uids, base_x, candidates, kept, pool, unknown, cameras)` — recomputes the seven context features (columns 278–284) of each candidate against the kept groups; the other 665 of the 672 features stay exactly as computed upstream.
- `candidate_features(seed, group_scores, uids, base_x, pool, unknown, cameras, group_tau=0.7)` — splits the seed groups into kept groups (score ≥ 0.70) and residual singleton candidates, contextualizes the candidates and checks the threshold and role columns of the frozen schema.
- `assemble(..., residual_model, group_tau=0.7, residual_tau=0.55)` — scores the residual candidates and returns reference A: kept groups plus singletons with score ≥ 0.55.
- `append(reference, extra, pool)` — append-only extension of A; refuses reuse of a protected observation.

**`protected_conditional_recovery.py`** — F5-P recovery conditioned on A.
- `partition(reference, pool, probabilities, duplicate_iou=.7, tau=.75, margin=.1, duplicate_tau=.9)` — groups the observations unused by A into candidate groups from pairwise identity scores (pair arm).
- `context(group, reference, pool, unknown, cameras, probabilities)` — the 24 context features of Section 4.11: mean and maximum of seven relative-space features, plus best, mean, minimum and spread of pair support to A, the shares above 0.75 and 0.95, evidence availability, the best anchor, the number of strong anchors and the best-versus-second margin.
- `features(groups, reference, pool, cameras, offsets, descriptors, flags, probabilities, unknown)` — full rows: baseline group features, the 24 context features and, for the shared-state arm, eight more (1,087 or 1,095 columns).
- `conflicts(a, b, pool, probabilities)` — conservative batch-duplicate test (as in `fusion_core`).
- `select(reference, groups, scores, tau, pool, probabilities, suppress=False)` — independent admission at τ, optional batch suppression, append to A.

**`nonlinear_admission.py`** — portable histogram gradient boosting.
- `export(estimator)` — converts a fitted scikit-learn `HistGradientBoostingClassifier` into JSON trees with numeric splits.
- `compile_model(model)`, `predict(xs, model)` — NumPy inference identical to the library's `predict_proba`.
- `fit(xs, ys, params)` — fits with scikit-learn 1.7.2 on one thread, exports, and refuses the result if the exported predictions differ from the library by more than 1e-12.

**`five_camera_capacity_recovery.py`** — F5-R current-scene recovery heads: capacity (529 features), residual (575) and verification (613).
- `residual_features(data, indices)` — the 46 support and competition features of unused observations. For each other camera: best and second-best pair score, their margin, confidence and ROI similarity of the best partner, mutual-best flag and pair availability (8 × 5 = 40). Then overall support count, strong-support count, maximum IoU with the current scene, triangle-closure score and flag, and witness share.
- `appearance_features(data, scene, indices)` — 14 ROI-appearance competition features against the current scene, used by the 613-feature verification head.
- `attach(data, q_known, predictive)` — attaches the pair-availability mask, the 24 forward-prediction features and normalised ROI cosine similarities; refuses any collinear forward prediction on a fisheye observation.
- `features(data, scene, head)` — feature matrix of the chosen head for every unused observation of the current scene.
- `rollout(data, scene, head, predict, floor=.02)` — greedy recovery: score all unused observations, append the best, recompute features on the grown scene, stop when the best score falls below `floor`. Returns the added IDs, the running minimum score and the raw scores; the caller keeps the prefix with score ≥ τ.

**`immersive_surround.py`** — look-around display (Section 5.1).
- class `ImmersiveSphere` — builds the 1536 × 768 equirectangular texture with per-texel source-camera ownership (`pixels`, `render`).
- class `LookController` — drag, wheel, turn and reset state of the virtual view, testable without a window.
- class `LookViewport` — perspective viewport of width W, height H and field of view φ; `maps` builds the ray grid and lookup maps, `render` samples the camera pixels directly.
- `overview(texture, sphere, control, width)` — the aspect-correct sphere page.

**`immersive_dashboard.py`** — class `SurroundDashboard`: page layout (`render`), hit-testing of buttons and insets (`hit`) and mouse routing (`mouse`). It changes presentation only.

### 6.3 `ultralytics/core_algorithms/run_depth_bev.py` — forward three-camera pipeline (F3)

Resumable driver of the depth/BEV experiments.
- `splits()` — frame lists of the fitting and evaluation splits.
- class `Frame` — one frame's detections, stereo maps and metadata.
- `slim(objs)`, `params_of(detset, name)`, `pkl(detset, name, stem)` — cache helpers.
- `build(frame, cfg, cams, stereo, rd, parent_cfg, parent_params, previous, T_ego)` — forms forward object hypotheses with the geometric gates and box-local stereo evidence.
- `stage_cand(detset, name, lists, prev, limit_s)` — candidate-generation stage over all frames.
- `_zs(o)`, `_ios(a, b)` — group depth statistics and intersection over the smaller area.
- `dedup3d(objs, ios_thr, zfrac)` — **E1 duplicate rule** (Section 4.8, Eq. 4.1): ranks groups by support and score, and suppresses a lower-ranked group that shares an observation identifier, or that is class-compatible, contained in every shared camera and inside the depth gate.
- `predict(objs, params)` — verification score: regularised logistic regression on 23 features (boosted trees in one variant).
- `admit_slim(objs, cfg, params)` — applies the frozen threshold and E1.
- `labels(objs, kept, gt, milp)` — fitting labels from the annotations.
- `stage_fit`, `stage_eval`, `main()` — fitting and evaluation stages and the command-line entry.

### 6.4 `ultralytics/decided/` — viewer entry points

- **`final_five_cam.py`** — class `CachedScene` loads the signed F5-P report, protocol and per-frame records, re-verifies their hashes and the native-image signatures, and rebuilds the F5-P scene with `protected_conditional_recovery.select(..., .55, ...)`. `run()` draws the five native views, the panorama (`--panorama rotation|natural`) and the BEV, and recomputes TP/FP/FN for visited frames.
- **`final_immersive_cam.py`** — `natives(images, objects, width)` draws the native-camera strip; `run()` is the look-around application.
- **`final_spatial_cam.py`** — `run()`: bowl surround and textured road BEV from the same cached scene.
- **`final_three_cam.py`** — `main()`: Cam2/Cam3/Cam4 panels plus the BAND + depth + E1 fusion; `--no-eval` hides the statistics.
- **`final_three_cam_no_band.py`** — the same four panels with full-image 960 input and no BAND (configuration F3).

### 6.5 `scripts/` — research runners and audits

**`protected_conditional_inputs.py`** — class `ProtectedInputs`. It loads the signed observation pool, corrected boxes, descriptors, flags and crops of a frame. `frame(s, early=False)` rebuilds reference A (or loads the cross-fitted early-segment reference) and checks it against the frozen one; `probabilities(pool, descriptor, flags)` predicts pairwise identity scores.

**`fit_protected_conditional_recovery.py`** — `labels(groups, reference, ev)` computes the marginal-capacity target y(g | A) = ν(A ∪ {g}) − ν(A) on early frames; `run()` builds the singleton, pair and shared arms (34,153 singleton examples, 3,552 positives) and fits them.

**`evaluate_protected_conditional_recovery.py`** — `select(dev)` applies the development selection rule; `run(domain)` replays development or the full sequence with A kept immutable and every output counted.

**`audit_protected_conditional_recovery.py`** — `independent_labels(...)`, `run(domain)`: independent recount of protection, models, sources and counts.

**`audit_thesis_recovery_intervals.py`** — `metrics(counts)`, `run(args)`: paired 25-frame block-bootstrap intervals from frozen per-frame counts.

**`experiment_five_camera_capacity_recovery.py`** — F5-R stage 1. `init()` writes the protocol; `config()` loads and hash-checks it; `parent(s, kind)` and `extra(...)` load the five-view evidence of a frame; `prepare(scope)` builds the feature banks; `load(s, kind)` reads them; `fit()` fits the capacity, residual and verification heads on the 601 early frames (130,355 examples, 12,335 positives); `evaluate(domain)`, `audit()` and `report()` replay, recount and summarise.

**`experiment_five_camera_repair_recovery.py`** — F5-R stage 2, the final configuration. `init()` locks the protocol (λ = 1, κ = 1, edit threshold 0.05, at most one edit per frame). Class `Inputs`, method `frame(s)`, applies the frozen repair decision and refreshes the forward predictions when a group with two forward members is retired. `run(domain)` performs the recovery rollouts, locks τ on development at P ≥ 0.91 and, on the full run, recomputes features, paths and XML counts from scratch and checks them. `report()` writes `FIVE_CAMERA_REPAIR_RECOVERY_20261017_v1.md`.

**`test_viewer_launcher.py`**, **`test_surround_dashboard_v09.py`** — unit tests of the launcher (entry paths, modes, argument safety) and of the surround pages (sphere aspect ratio, page sizes, mouse regions).

### 6.6 `scripts/supp/` — supplementary experiments (Appendix B)

**`supp_lib.py`** — shared library.
- `cls_match(label, ycls)` — class compatibility (car, bus and truck mutually compatible).
- `iou(a, b)` — box IoU.
- `segments()` — early, development, guard and late frame lists.
- `frame(s)` — loads one frame of the frozen snapshot.
- `det(d, corrected)` — detection view with the native or the box-calibrated box, class and confidence.
- `as_objects(groups, F, corrected, recovered_from)` — object records for evaluation.
- `f5r_tau()` — the locked F5-R threshold.
- `configs(s, tau)` — frozen outputs of one frame: raw nodes, A, F5-P, repaired root, F5-R.
- `build_union(gt_by_cam, min_h)`, `gt_subset(gt_by_cam, cams)` — evaluation unions.
- `evaluate(objects, gt_by_cam, thr, min_h, return_assignment)` — the dissertation protocol: any-member IoU, cardinality-first `linear_sum_assignment`, 25 px ignore rule.
- `prf(c)` — P/R/F1 from counts; `block_bootstrap(counts_a, counts_b, block, reps, seed)` — paired block bootstrap; `new_run`, `dump`, `h_bin` — run folders, JSON output, height bins.

**`common.py`** — the earlier shared loaders: the same protocol functions plus `raw_pool`, `load_f5p`, `f5r_record`, `f5r_groups`, `f5r_objects`, `evaluate_at`.

**`extract.py`** — resumable, read-only extraction of the per-frame snapshot with count verification.
**`selfcheck.py`** — reproduces the frozen F5-P and F5-R counts frame by frame.
**`winpath_shim.py`** — normalises backslash paths of Windows-authored protocol files when they are read on Linux (`_pa`, `_fix`, `loads`, `load`); files and hashes stay as stored.
**`verify_lib.py`** — checks the library against the frozen totals.

**`e1_e6_error_analysis.py`** — `match_camera` (per-camera one-to-one matching), `rnorm` and `rbin` (normalised radial distance and its bins), `crop_cover` (fraction of a GT box inside the crop tiles), `run` (E1: per-camera, size, class, radial and cross-view statistics) and `e6` (recall by size, class group and visible-camera set).
**`e2_simple_baselines.py`** — `angle`, `ray_nms` (B2), `variants` (all grid points of B1, B2, B3 and B3′ for one frame), `sweep`, `select` (development lock).
**`e3_feature_ablation.py`** — `arrays`, `load`, `mask` (removes one feature group), `training_set`, `fit`, `predictor`, `rollout_frame`, `run_domain`, `at_tau`, `lock` (P ≥ 0.91 on development), `full`, `verify` (the frozen model must reproduce the audited paths), `scene_edges`, `counts`, `out`, `frozen_residual`.
**`e3b_fisheye_output.py`** — removes Cam1 and Cam5 members from the frozen outputs and re-evaluates.
**`e4_association.py`** — `label_members` (per-camera identity of each member), `pair_stats` (correct, wrong, unknown and split pairs), `config_groups`, `summarise`.
**`e4_audit_sample.py`**, **`e4_render_audit.py`**, **`e4_audit_score.py`** (`wilson`) — blind visual identity check: stratified sampling, crop-sheet rendering and scoring with Wilson intervals.
**`e5_ap.py`** — `coco_ap` (101-point AP), `single_camera`, `group_events`, `prefix_counts`, `group_ap`.
**`e7_learned_baseline.py`** (`own_features`, `agreement_features`, `labels`, `frame_arrays`, `ray_nms_scored`, `sweep`, `per_frame`) — E7: per-detection HistGB verifier trained on the 601 early frames without scene context (L1: 34 own-evidence features; L2: + 9 local multi-view agreement features), grouped by ray NMS on the learned score, locked on development at P≥0.90/0.91, replayed on full and guard+late with paired bootstrap against F5-R; also the block-length sensitivity (10/25/50/100) of F5-R minus F5-P. Writes `e7_learned_baseline/run01/results.json`.
**`make_figures.py`** (`save`), **`make_tables.py`** (`write`, `tab`, `agg`, `r6`, `row2`, `rec_at`, `prec_at`) — figures and LaTeX tables of Appendix B and Chapter 6.

### 6.7 Top level

**`launch_viewers.py`** — `runner()` selects uv or the project `.venv`; `command(view, mode, start, fps, cache)` builds the child command for a view; `start_viewer(cmd)` starts it without a shell; class `Launcher` is the Tk menu; `main(argv)` handles command-line use (`--view three|five|bev|surround|…`, `--mode auto|step`, `--start`, `--fps`, `--cache`).

### 6.8 `portable/original_layout/` — full source snapshot

100 files at their original paths; `portable/SOURCE_MAP.md` maps them to thesis sections.

- **Three-camera foundation** (`ultralytics/core_algorithms/`): `multiview_geometry.py` (calibration loading, projection, collinear constraint), `tri-compare.py`, `legacy_adapter.py`, `stereo_depth.py`, `depth_bev_fusion.py`, `run_depth_bev.py`, `final_pipeline.py` and `final_viewer.py` (ego exclusion, single-camera controls), `validate_multiview.py` (evaluator), and historical variants such as `multiview_dedup.py`, `multiview_fusion.py`, `soft_tracking.py`, `temporal_bev.py`, `guided_redetect.py`, `yolo_hr_detect.py`.
- **Five-camera evidence and scene decisions** (`ultralytics/research/`): `shared_geometry.py`, `cam15_scoring.py`, `identity_pair_features.py`, `joint_identity_admission.py`, `identity_measurement_policy.py`, `retained_reference.py`, `residual_recovery.py`, `marginal_capacity.py` (the ν(S) target), `box_error_calibration.py`, `budget_fovea.py`, `crop_measurement.py`, `detector_roi_admission.py`, `protected_reference.py`, `protected_conditional_recovery.py`, `nonlinear_admission.py`, and the alternative branches `spatial_conditional_recovery.py`, `road_support.py`, `road_support_admission.py`.
- **Displays**: `five_camera_display.py`, `natural_panorama.py`, `spatial_surround.py`, `spatial_observation_overlay.py`, `spatial_dashboard.py`, `immersive_surround.py`.
- **Runners and audits** (`scripts/`): input loaders (`joint_identity_inputs.py`, `protected_conditional_inputs.py`, `crossfit_protected_reference.py`), fitting (`fit_retained_reference.py`, `fit_identity_pair_evidence.py`, `fit_protected_conditional_recovery.py`), experiments (`experiment_*_admission.py`, `experiment_spatial_conditional_recovery.py`, `experiment_cam15_shared.py`), audits (`audit_*.py`) and thesis figure builders (`build_thesis_figures.py`, `render_thesis_screenshots.py`, `render_thesis_cases.py`).

---

## 7. Evidence files and the thesis results they support

| Thesis result | Evidence file (relative to the package root) |
|---|---|
| F3 forward results, Table 6.3 (P 93.25 %, R 36.82 %) | `ultralytics/algorithm/NO_BAND_EXPERIMENT_20261007.md`, `ultralytics/algorithm/evaluation_reports/no_band_results_20261007.json` |
| Single-camera and BAND results, Tables 6.4–6.5 | `ultralytics/algorithm/evaluation_reports/global_fusion_results.json`, `low_conf_no_band_results_20261009.json`, `ultralytics/algorithm/LOW_CONF_NO_BAND_EXPERIMENT_20261009.md` |
| F3 rematched to the five-view GT, Table 6.6 | `ultralytics/algorithm/evaluation_reports/original_on_five_gt_20261008_retest01.json` |
| F5-P results, Tables 6.6, 6.11 and 6.12 | `ultralytics/algorithm/evaluation_reports/protected_conditional_full_results_20261010.json`, `protected_conditional_independent_audit_20261010.json` |
| Matching-capacity losses, Figure 6.6 | `ultralytics/algorithm/evaluation_reports/protected_conditional_full_stage_losses_20261010.json` |
| F5-R final result, Tables 6.13 and 6.14 | `ultralytics/algorithm/evaluation_reports/five_camera_repair_recovery_{full,development,audit,evidence}_20261017_v1.json`, `ultralytics/algorithm/FIVE_CAMERA_REPAIR_RECOVERY_20261017_v1.md`, `thesis/FIVE_CAMERA_RECOVERY_RESULTS_20261017.md` |
| Direct recovery without repair (F5-R stage 1), Table 6.13 | `ultralytics/algorithm/evaluation_reports/five_camera_capacity_recovery_*_20261017_v1.json`, `ultralytics/algorithm/FIVE_CAMERA_CAPACITY_RECOVERY_20261017_v1.md` |
| Alternative recovery designs, Table 6.16 | `ultralytics/algorithm/evaluation_reports/spatial_*_20261012.json`, `temporal_*_20261012.json`, `thesis/temporal_evidence_audit_20261016_v1.json` |
| Paired block intervals (Sections 6.2 and 6.5) | `thesis/recovery_block_intervals_v1.json`, produced by `scripts/audit_thesis_recovery_intervals.py` |
| Whole-group support audit (Section 6.4.3) | `thesis/group_support_audit_20261016_v1.json`, `thesis/group_support_verification_20261016_v1.json` |
| Three-camera current-scene recovery (Section 6.3) | `ultralytics/algorithm/THREE_CAMERA_{PREDICTIVE_RECOVERY,LOW_CONF_RECOVERY}_20261016_v1.md`, `THREE_CAMERA_BOX_LOCALIZATION_20261017_v1.md` |
| BEV position availability, Figure 5.6 | `ultralytics/algorithm/evaluation_reports/five_camera_viewer_run_1791807403777670200_audit_20261012.json` |
| Overlay drift correction (Section 5.5) | `ultralytics/algorithm/SPATIAL_OVERLAY_ALIGNMENT_20261013.md` |
| Figures and screenshots | `thesis/figures/{vector_v3,project_screenshots_v2,native_cases_v3}/manifest.json`, `thesis/case_audit_v1.json` |
| Data and annotation facts (Section 3.2) | `THESIS_DATA_LEDGER.json`, `docs/archive/root_markdown_20261020/THESIS_DATA_LEDGER.md` |
| Version record of the writing evidence | `thesis/evidence_snapshot_v6.json` |
| Supplementary results E1–E7 (Sections 3.7, 6.2, 6.4, 6.5.3, 6.6, 6.7) | scripts in `scripts/supp/`; result JSON under `ultralytics/experiments/supp_20261101/` (included in this package) |

---

## 8. Integrity rules built into the code

- **Source exclusivity**: every observation belongs to at most one object, and every object has at most one member per camera (`protected_reference.validate`).
- **Protected reference**: recovery appends to A and keeps A's groups, boxes and scores as they are (`protected_reference.append`, `protected_conditional_recovery.select`).
- **Ground-truth isolation**: candidate construction, features, scoring and selection read camera evidence only; annotations enter fitting labels and offline counting (`demo.py` loads the IoU graph after selection).
- **Hash-checked inputs**: protocols store SHA-256 values of sources, models, caches and native images, and the loaders refuse a mismatch.
- **Append-only runs**: each runner writes to a new directory, so earlier results stay as they were.
- **Portable inference**: exported trees reproduce the scikit-learn predictions to 1e-12, and inference loads JSON trees.

## 9. Scope notes

- The viewers replay F5-P from signed caches; F5-R comes from `scripts/experiment_five_camera_repair_recovery.py` and uses the same object record.
- All results come from the single supplied route. The chronological segments (early 0–600, development 601–800, guard 801–1099, late 1100–1700) separate fitting, threshold selection and later assessment.
- The dataset, weights and external packages are subject to their providers' licence conditions (`portable/THIRD_PARTY_NOTICES.md`).
