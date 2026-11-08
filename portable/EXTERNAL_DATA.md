# Inputs excluded from the small code package

The standalone demo needs no external files. The complete three/five-camera pipelines do. Paths below are relative to a matching full project root, not to the minimal demo's `models/` folder. They were inspected from the actual loader constants.

| Input | Exact existing path |
|---|---|
| Native images and annotations | `ultralytics/mycar/images/{1..5}/{frame}.png`; `ultralytics/mycar/cam_{1..5}/{frame}.xml` |
| Reliable-reference upstream report | `ultralytics/algorithm/evaluation_reports/detector_roi_admission_results_20261009.json` |
| Upstream group model/protocol | `ultralytics/experiments/budget_fovea_20261009/detector_roi_admission01/` |
| Crop-parent report | `ultralytics/algorithm/evaluation_reports/capacity_admission_results_20261009.json` |
| Descriptor cache | `ultralytics/experiments/budget_fovea_20261009/detector_roi01/shard*/frames/` |
| Prepared geometry/detector cache | `ultralytics/experiments/budget_fovea_20261009/fusion01/shard*/frames/` |
| Frozen output-box model | `ultralytics/algorithm/evaluation_reports/box_error_calibration_results_20261009.json` |
| Crop policy | `ultralytics/experiments/budget_fovea_20261009/probe01/policy.json` |
| Nested scoring-reference records | `ultralytics/experiments/protected_conditional_20261010/run01/reference01/` |
| Existing residual reference model | `ultralytics/experiments/retained_reference_20261010/run01/` |
| Pair head | `ultralytics/experiments/identity_pair_20261010/early01/` |
| Full recovery model/protocol and per-frame signed records | `ultralytics/experiments/protected_conditional_20261010/run01/recovery01/` |
| Frozen five-view evaluation report | `ultralytics/algorithm/evaluation_reports/protected_conditional_full_results_20261010.json` |
| Spatial display-plane reference | Signed paths selected by `ultralytics/research/spatial_surround.py:display_plane`; referenced by `road_support_admission_results_20261010.json` and `road_support_20261010/full01/protocol.json` |
| Historical front detector/depth caches and frozen score | Read from the historical no-BAND/BAND viewer configuration; not the new portable singleton model |
| Detector weights for fresh inference | YOLO11n file expected by the selected full-project entry; not required by `demo.py` |

This table identifies major loader boundaries, not a promise that copying only these directory names supplies every signed dependency. The original protocols also record source hashes, predecessor reports, calibration and input signatures. Full recreation must satisfy those checks; do not remove them to make a partial cache appear valid.

Full data, weights and external packages remain subject to their provider/licence conditions. No model downloading, new fitting, full-domain detector inference or data publication was performed while preparing the small package. The three numerical fixtures are derived prepared features and an offline IoU graph, not a replacement annotated driving dataset.
