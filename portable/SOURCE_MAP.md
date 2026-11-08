# Thesis section to code mapping

Paths in this table are relative to `original_layout/`, which preserves original filenames and file bytes. The full per-file import graph and SHA256 values are in `metadata/source_manifest.json`.

| Thesis role | Original source files |
|---|---|
| Three-view geometric foundation | `ultralytics/core_algorithms/tri-compare.py`, `legacy_adapter.py`, `multiview_geometry.py` |
| Historical depth evidence and E1 | `ultralytics/core_algorithms/run_depth_bev.py` (`dedup3d`), `stereo_depth.py`, `depth_bev_fusion.py` |
| Ego exclusion and individual-camera controls | `ultralytics/core_algorithms/final_pipeline.py`, `final_viewer.py`; `ultralytics/decided/final_cam2.py`, `final_cam3.py`, `final_cam4.py` |
| Three-view no-BAND and BAND entries | `ultralytics/decided/final_three_cam_no_band.py`, `no_band_depth_viewer.py`, `final_three_cam.py`, `band_depth_viewer.py` |
| Five-view observation models and grouping | `ultralytics/research/shared_geometry.py`, `cam15_scoring.py`, `joint_identity_admission.py`, `identity_measurement_policy.py` |
| Protected reference A and context | `ultralytics/research/protected_reference.py`, `retained_reference.py`, `residual_recovery.py` |
| Conditional singleton/pair/shared arms | `ultralytics/research/protected_conditional_recovery.py` |
| Numerical exported tree prediction | `ultralytics/research/nonlinear_admission.py` |
| Marginal supervision graph | `ultralytics/research/marginal_capacity.py`; fitting-only GT use |
| Prepared input loader and reference cross-fitting | `scripts/protected_conditional_inputs.py`, `joint_identity_inputs.py`, `crossfit_protected_reference.py` |
| Recovery fitting | `scripts/fit_protected_conditional_recovery.py`; not the deployment entry |
| Locked evaluation and independent audit | `scripts/evaluate_protected_conditional_recovery.py`, `audit_protected_conditional_recovery.py`, `experiment_cam15_shared.py` |
| Negative spatial branch | `scripts/experiment_spatial_conditional_recovery.py`, `audit_spatial_conditional_recovery.py`; not the retained replacement |
| Five-view frozen scene and panorama | `ultralytics/decided/final_five_cam.py`; `ultralytics/research/five_camera_display.py`, `natural_panorama.py` |
| Bowl/BEV rendering and compact interface | `ultralytics/decided/final_spatial_cam.py`; `ultralytics/research/spatial_surround.py`, `spatial_observation_overlay.py`, `spatial_dashboard.py` |
| Look-around display | `ultralytics/decided/final_immersive_cam.py` and its imported renderer |
| Thesis images and source-bound plots | `scripts/render_thesis_screenshots.py`, `build_thesis_figures.py` |
| Thesis numerical/source checks | `scripts/audit_thesis_evidence.py`, `review_thesis_draft.py`, `audit_thesis_cases.py`, `audit_thesis_recovery_intervals.py`, `audit_thesis_illustrations.py` |

The portable `fusion_core.py` is a mechanically extracted subset: compile/predict, objects/validate/append, overlap/conflicts/select and the unchanged offline count function. It deliberately does not pretend to compute images, detector candidates, descriptors, reference A or feature extraction. `demo.py` tests the last score/selection/count boundary with original prepared inputs.

Implementation provenance is separate from research novelty. Camera models, stereo, assignment, histogram boosting and suppression foundations are established methods. The thesis contribution concerns the reference-relative recovery target, immutable-output contract, selection/evaluation design and measured operating point. Imported experimental alternatives are not automatically components or original contributions of the selected algorithm.
