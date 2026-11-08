# Dissertation code package

This folder accompanies the English dissertation pre-draft. It contains exact source snapshots, a mechanically extracted portable recovery kernel, the frozen singleton head and three prepared-feature examples. The selected five-view method is `single/independent/tau0.55`.

这是论文代码交付目录，不是另一个 Git 仓库。先运行下面的小型独立样例；完整三目/五目看图需要原项目数据和缓存。源码不是只复制了几个入口：`original_layout/` 保留对应算法及本地导入依赖，`SOURCE_MAP.md` 给出论文章节和文件对应关系。

## Quick start: no raw dataset required

From the repository root, or the parent of this extracted `code/` folder:

```powershell
uv run --project code python code/demo.py
uv run --project code python -m unittest discover -s code/tests -v
uv run --project code python code/verify.py
```

From inside `code/`, the equivalent commands are:

```powershell
uv run python demo.py
uv run python -m unittest discover -s tests -v
uv run python verify.py
```

The standalone environment is Python 3.12, NumPy 1.26.4 and SciPy 1.17.1, recorded in `pyproject.toml` and `uv.lock`. It does not require Torch, Ultralytics, OpenCV, raw RGB images or pickle files. `uv` may install these two numerical dependencies and obtain a compatible Python if unavailable.

## Exactly what the standalone demo reproduces

The example inputs begin **after** detector/geometry/descriptor preparation and reliable-reference construction. The demo predicts the frozen recovery score from the included 1,087-dimensional features, admits residual singletons at 0.55, preserves A and performs offline evaluation on the included eligible IoU graph. The graph is loaded for counting only after selection. It never influences admission.

The selected kernel is mechanically extracted from the original `nonlinear_admission.py`, `protected_reference.py`, `protected_conditional_recovery.py` and evaluator. Function bodies remain unchanged; the extraction only supplies minimal imports and P/J namespaces. `metadata/extractions.json` records function names, source lines and hashes. The full original modules also remain in `original_layout/`.

| Frozen frame | Reference TP / FP / FN | Final TP / FP / FN | Added objects |
|---|---|---|---|
| 00000641 | 16 / 5 / 31 | 17 / 5 / 30 | 1 |
| 00000650 | 20 / 4 / 25 | 20 / 4 / 25 | 0 |
| 00001200 | 4 / 0 / 2 | 4 / 0 / 2 | 0 |

All three score vectors reproduce the signed originals with maximum absolute discrepancy zero in the inspected environment. The corresponding memberships and five-field count vectors agree. These examples are replay checks, not a new benchmark, unseen-scene test or full-sequence inference. Their printed P/R/F1 values must not replace the thesis aggregate P90.45/R42.69/F1 58.00.

The 14 unit tests check source exclusivity, protected-set immutability, one active member per view, invalid inputs, inclusive thresholds, independent competitor admission, unknown-location acceptance, non-additive marginal positives, the small-GT-only ignore rule and the three frozen examples. Their passing does not establish physical identity or 3D accuracy.

## Package layout

```text
code/
  README.md                      scope and executable commands
  SOURCE_MAP.md                  chapter-to-algorithm filename mapping
  EXTERNAL_DATA.md                exact external input paths and limitations
  THIRD_PARTY_NOTICES.md          preserved attribution and release cautions
  pyproject.toml / uv.lock        small standalone environment
  demo.py                        frozen score -> selection -> offline count replay
  fusion_core.py                 unchanged extracted function bodies
  verify.py                      payload and optional source-project hash checks
  run_project.py                 checked launcher for the full external workspace
  models/recovery_single.json    original exported singleton tree head
  examples/*.json / *.npz        three numeric prepared-feature examples, no pixels
  tests/test_submission.py       executable contract/replay tests
  metadata/                      source, extraction, sample and package hashes
  original_layout/
    scripts/                     fitting, evaluation, audits and packaging tools
    ultralytics/decided/          three/single/five/spatial/immersive entries
    ultralytics/core_algorithms/ historical front association, depth, E1, evaluation
    ultralytics/research/         five-view features, reference, recovery and display
    ultralytics/pyproject.toml    original full-project dependency declaration
    ultralytics/uv.lock          original full-project dependency lock
    ultralytics/LICENSE         original AGPL-3.0 licence text, not removed
    thesis/                      frozen evidence/intervals/reference records
    Intrinsics_cam.json
    T_imu_cam(1).json
```

The source snapshot includes 94 Python files selected through the entry-point import closure and explicit local `.py` literals, plus the packaging/export tools. It is not a claim that every function in those files is part of the selected method; some imported modules retain historical or unsuccessful research branches. External libraries and the unrelated Fast-BEV code repository are not vendored here.

## Full project viewers and experiments

Complete viewers need the original dataset/caches and full project environment. With the current workspace available, from its root:

```powershell
uv run --project code python code/verify.py --against-project .
uv run --project ultralytics python code/run_project.py --project-root . --entry five --check-only
uv run --project ultralytics python code/run_project.py --project-root . --entry five -- --mode step --panorama natural
uv run --project ultralytics python code/run_project.py --project-root . --entry spatial -- --mode step
uv run --project ultralytics python code/run_project.py --project-root . --entry three-no-band -- --mode auto
```

`--check-only` checks source/calibration/evidence agreement and does not launch a detector or viewer. The full launcher uses the matching external workspace, not the prepared-feature demo. Its source checksum check intentionally refuses a changed project; create a new package version after algorithm changes. Full three-view fresh inference and five-view signed-cache replay are different workflows.

Other entry choices are `three-band`, `cam2`, `cam3`, `cam4` and `immersive`. For reviewers without the external workspace, `original_layout/` preserves the code for inspection or reconstruction, but **does not** contain all inputs necessary to run the full pipeline. Do not describe that snapshot as self-contained end-to-end inference. Its original historical path conventions are retained instead of silently changing frozen source signatures.

## Exclusions and release boundary

Raw RGB images, XML annotations, full per-frame geometry/descriptor caches, detector weights, predecessor models, virtual environments and Git metadata are not bundled. The derived numeric examples/model also require review of dataset/provider and code redistribution rights before public release. This local package is not permission to publish the dataset or a School-approved submission certificate.

The complete thesis result is reproduced from saved full-domain reports, not inferred from the three examples. `original_layout/thesis/evidence_snapshot_v6.json` retains the common-domain totals and their source hashes. Follow `EXTERNAL_DATA.md` for the actual external dependencies and `SOURCE_MAP.md` for fitting versus inference boundaries. No speed, power or independent-scene result is implied by this package.
