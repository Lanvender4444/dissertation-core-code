# 论文数据总账（自动核验版）

生成：2026-10-05T20:04:23.541428+00:00。根目录：D:/Yolo-Dissertation。

## 0. 使用方式与本轮范围

先读[需要的数据清单](THESIS_DATA_REQUIREMENTS.md)。完整文件索引、SHA256、每张原图size/mtime、所有原XML哈希、数据切分、历史结果快照与核验详情见[机器总账](THESIS_DATA_LEDGER.json)。

本轮实际执行本地盘点、XML检查、哈希校验和逐帧TP/FP/FN重汇总；没有重新运行YOLO/深度，没有新拟合、没有修改GT或算法。原始数据不搬动，不删除。

## 1. 五相机输入盘点

| 相机 | PNG | XML | 分辨率数量 | 缺XML | 缺PNG | 标注对象框数 | 重复ID | 退化框 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Cam1 | 1655 | 1655 | {'2464x2056': 1655} | 0 | 0 | 9242 | 0 | 0 |
| Cam2 | 1655 | 1655 | {'2464x2056': 1655} | 0 | 0 | 14890 | 0 | 0 |
| Cam3 | 1655 | 1655 | {'2464x2056': 1655} | 0 | 0 | 14843 | 0 | 0 |
| Cam4 | 1655 | 1655 | {'2464x2056': 1655} | 0 | 0 | 14798 | 0 | 0 |
| Cam5 | 1655 | 1655 | {'2464x2056': 1655} | 0 | 0 | 11201 | 0 | 29 |

- cam234：配对完整1655帧；假设跨相机ID一致后，≥25px GT联集13473个对象帧实例（不是独立物体数量）。

- cam12345：配对完整1655帧；假设跨相机ID一致后，≥25px GT联集24845个对象帧实例（不是独立物体数量）。

跨相机同ID类别冲突：{'cam234': 5, 'cam12345': 7}。详细框、帧号与相机见JSON。只是审计标记，不自动改标签或身份。

新增质量问题：Cam5有29个零高度框（00000303–00000330中的id39，以及00001395中的id5）；原XML保持不变。它们本视角达不到25px门槛，但五相机联集可能被其他视角保留，后续五相机协议必须显式处理。

部分标注越过图像边界，全部坐标已记录；可能是截断/标注约定，不自动视为错误或裁剪。Cam234未发现退化框，本次没有改动其评分结果。

输入位置：`ultralytics/mycar/images/1..5`、`ultralytics/mycar/cam_1..5`。PNG头已全查，未全量解码/内容哈希；文件修改时间不是拍摄时间。

## 2. 论文主对照：同一Cam234整体GT

全部1655帧、固定YOLO11n、兼容类别IoU≥0.5、GT任一视角框高≥25px、Hungarian一对一。单相机这一张表也对照Cam234联集，不能叫“该相机自身召回”。

融合对象任一成员命中即可TP，不保证其余成员身份正确。P/R/F1不是分类Accuracy或mAP；FP数量下降也不是传统FP/(FP+TN)假阳性率。

| 方法 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 960/cam2 | 4830 | 1715 | 8643 | 73.80% | 35.85% | 48.26% |
| 960/cam3 | 4475 | 987 | 8998 | 81.93% | 33.21% | 47.27% |
| 960/cam4 | 4375 | 709 | 9098 | 86.05% | 32.47% | 47.15% |
| 960/old_fusion | 5148 | 750 | 8325 | 87.28% | 38.21% | 53.15% |
| 960/base | 5693 | 2018 | 7780 | 73.83% | 42.25% | 53.75% |
| 960/d | 5559 | 1369 | 7914 | 80.24% | 41.26% | 54.50% |
| BAND/cam2 | 6685 | 1325 | 6788 | 83.46% | 49.62% | 62.24% |
| BAND/cam3 | 6629 | 1954 | 6844 | 77.23% | 49.20% | 60.11% |
| BAND/cam4 | 6885 | 1424 | 6588 | 82.86% | 51.10% | 63.22% |
| BAND/old_fusion | 7809 | 2050 | 5664 | 79.21% | 57.96% | 66.94% |
| BAND/base | 7744 | 1853 | 5729 | 80.69% | 57.48% | 67.13% |
| BAND/d | 7589 | 1294 | 5884 | 85.43% | 56.33% | 67.89% |
| BAND/d_E1 | 7507 | 1092 | 5966 | 87.30% | 55.72% | 68.02% |
| BAND/reliable_dedup | 7573 | 1266 | 5900 | 85.68% | 56.21% | 67.88% |
| BAND/rescue | 7509 | 1093 | 5964 | 87.29% | 55.73% | 68.03% |
| BAND/dfo_E1 | 7814 | 1953 | 5659 | 80.00% | 58.00% | 67.25% |
| ADAPTIVE/old_fusion | 7602 | 1922 | 5871 | 79.82% | 56.42% | 66.11% |
| 960/raw_union | 6373 | 19553 | 7100 | 24.58% | 47.30% | 32.35% |
| BAND/raw_union | 8977 | 29788 | 4496 | 23.16% | 66.63% | 34.37% |

## 3. 单相机自身视场口径（不得混用上表）

| 方法 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 960/cam2 | 4825 | 1712 | 7642 | 73.81% | 38.70% | 50.78% |
| 960/cam3 | 4472 | 986 | 8015 | 81.93% | 35.81% | 49.84% |
| 960/cam4 | 4371 | 708 | 8061 | 86.06% | 35.16% | 49.92% |
| BAND/cam2 | 6656 | 1324 | 5811 | 83.41% | 53.39% | 65.10% |
| BAND/cam3 | 6596 | 1951 | 5891 | 77.17% | 52.82% | 62.72% |
| BAND/cam4 | 6841 | 1423 | 5591 | 82.78% | 55.03% | 66.11% |

## 4. 拟合与评价范围

当前BAND_d_E1评分拟合/开发帧50；各历史开发范围联集110；完整帧号列表在JSON。YOLO未重训，但23维后处理评分曾拟合。

全部1655帧已用于研究。排除开发帧/后段/冲突帧属于同路线敏感性检查，不是独立场景盲测。历史known_regression_frames也记录在JSON，不代表未查看。

| 范围 | 帧数 | 主方法TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_frames | 1655 | 7507 | 1092 | 5966 | 87.30% | 55.72% | 68.02% |
| not_used_for_fitting | 1545 | 6890 | 1004 | 5429 | 87.28% | 55.93% | 68.17% |
| late_sequence_ge1000 | 655 | 2236 | 280 | 1435 | 88.87% | 60.91% | 72.28% |
| previously_reported240 | 240 | 824 | 119 | 538 | 87.38% | 60.50% | 71.50% |
| additional_late_frames | 415 | 1412 | 161 | 897 | 89.76% | 61.15% | 72.75% |
| ordered_block_00 | 200 | 992 | 162 | 1058 | 85.96% | 48.39% | 61.92% |
| ordered_block_01 | 200 | 1451 | 229 | 1417 | 86.37% | 50.59% | 63.81% |
| ordered_block_02 | 200 | 908 | 171 | 760 | 84.15% | 54.44% | 66.11% |
| ordered_block_03 | 200 | 1052 | 125 | 397 | 89.38% | 72.60% | 80.12% |
| ordered_block_04 | 200 | 868 | 125 | 899 | 87.41% | 49.12% | 62.90% |
| ordered_block_05 | 200 | 974 | 86 | 649 | 91.89% | 60.01% | 72.61% |
| ordered_block_06 | 200 | 643 | 91 | 395 | 87.60% | 61.95% | 72.57% |
| ordered_block_07 | 200 | 454 | 67 | 235 | 87.14% | 65.89% | 75.04% |
| ordered_block_08 | 55 | 165 | 36 | 156 | 82.09% | 51.40% | 63.22% |

## 5. 验证对象存在与组内身份必须分开

| 方法 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| raw | 8747 | 11422 | 4726 | 43.37% | 64.92% | 52.00% |
| score_gate | 7610 | 1342 | 5863 | 85.01% | 56.48% | 67.87% |
| multiview_iou | 7589 | 1294 | 5884 | 85.43% | 56.33% | 67.89% |
| residual | 7579 | 1259 | 5894 | 85.75% | 56.25% | 67.94% |
| final | 7507 | 1092 | 5966 | 87.30% | 55.72% | 68.02% |

最终连接：正确18732、已知不同GT 318、未知5248；总数24298。

已知不同GT中：182条至少一端有重叠GT对齐歧义，136条两端唯一兼容IoU命中。只人工图片复核4个代表病例，不冒充所有318条人工确认。

早期99.05%与最新98.33%采用不同身份对齐规则，不能当作版本准确率升降。FULL/PART/UNVERIFIED是支持标签，不是真实/假阳性结论。

所有分层、FP分类、区间、消融、外观与关联纠错结果均保留来源JSON快照；报告入口见第8节，不拼成不同设置混合排行榜。

## 6. 本轮校验（不是新检测实验）

| 检查 | 结果 |
|---|---|
| protocol_current_cam234_domain | 通过 |
| previous_cam234_GT_hashes | 通过 |
| code_sha256 | 通过 |
| calibration_sha256 | 通过 |
| model_sha256 | 通过 |
| main_cache_input_size_mtime | 通过 |
| global_fusion_frame_counts.json:counts | 通过 |
| global_fusion_frame_counts.json:native_counts | 通过 |
| topic3_stage1_frame_counts_20261006.json:counts | 通过 |
| topic3_geometry_frame_counts_20261006.json:counts | 通过 |
| topic3_association_constraints_frame_counts_20261006.json:counts | 通过 |
| GT_union_recomputed | 通过 |

## 7. 查找结果与暂缺数据

### 实际发现的数据关系

- 原One-North压缩包只有8275张PNG及目录，没有XML、CSV或同步说明。相机/帧名集合与mycar一致；15张代表图的大小/CRC与原包记录已核对，不能把此包当另一个测试序列。
- MultiViewFusion/data的XML与mycar逐文件SHA256比较，图片核对帧名及15张代表PNG哈希。比较结果见JSON related_data；未对所有PNG做全文哈希，重复导出不作为新独立序列。
- resource_data/20250120有4组Camera3 JPG：578、1154、807、486张，共3025张。该目录未找到Cam2/4配对、GT或标定，因此暂不拿它做有标签多视角评价。
- INS CSV含76929条10列数值记录，首列单调；单位、曝光映射仍未确认，它不是点云或3D目标真值。

### 额外找到的项目计划范围

[本地项目计划](杂项/Chengwei%20Zhang_Project%20Plan.md)不仅谈原始Topic3，还明确写了鱼眼/针孔联合验证、BEV，以及mAP/BEV completeness。仅记录要求上下文，不把计划当完成证明。若以这份计划作为最终验收范围，当前三相机F1实验不能替代五相机、BEV和AP/mAP验证。

搜索范围及每个候选文件见JSON discovery。本轮不重新解析PDF/DOCX正文；PDF此前13页审读结论见原数据审计。其他候选文档尚未建立正式证据，不把文件名当内容结论。

| 项目 | 当前证据 | 能否用于结论 |
|---|---|---|
| 相机内外参 | 根目录Intrinsics_cam.json、T_imu_cam(1).json及冻结协议哈希 | 可说明使用了标定；不可证明绝对标定误差 |
| 标定格式 | 原文件不是严格标准JSON；现有相机解析器按既有结构读取 | 原文件保持不变；不要直接json.load或默默换参数 |
| 同步 | 同帧名；尚无确认曝光时间映射/误差 | 暂不声称已实证硬件同步 |
| 来源/AISG | 原始One-North包、INS与说明文档候选 | 文件名线索不是提供方/采集证明，正式来源暂缺 |
| 标注身份 | 已检查XML；已有冲突和拥挤歧义 | GT-ID物理语义暂缺；关联正确率只能说代理诊断 |
| 新独立序列 | 当前包/各导出目录是否同源尚需来源映射 | 暂无确认独立序列，不把复制/导出当新测试 |
| 3D/深度真值 | 现有lidar目录CSV是INS/姿态候选 | 没有确认图像对齐3D真值，不报绝对深度/BEV精度 |
| Cam1/5 | 图片与XML已盘点；已有历史五相机报告 | 未完成当前BAND+d_E1同口径五相机全域验证 |
| 实时/功耗 | 有历史端到端单进程17帧计时及并发诊断计时 | 仅按原范围引用；没有功耗数据，不报最低功耗/实时 |
| 使用许可/隐私 | 本地候选尚未建立数据许可证据 | 论文定稿需补正式说明，不默认可以公开原图 |

找到候选≠建立证明。以上暂缺项不开展依赖它们的实验、不造替代GT。原包仅查中央目录，未全量解包，不能据此否定其他资料存在。

## 8. 所有已有实验来源索引

| JSON来源 | 大小（字节） | 数据范围提醒 |
|---|---:|---|
| [association_recheck.json](ultralytics/algorithm/evaluation_reports/association_recheck.json) | 27068 | SHA256与摘要在机器总账；按各自协议引用 |
| [baselines_BAND.json](ultralytics/algorithm/evaluation_reports/baselines_BAND.json) | 1962 | SHA256与摘要在机器总账；按各自协议引用 |
| [bev_verification_results.json](ultralytics/algorithm/evaluation_reports/bev_verification_results.json) | 7659 | SHA256与摘要在机器总账；按各自协议引用 |
| [dedup_fp_audit.json](ultralytics/algorithm/evaluation_reports/dedup_fp_audit.json) | 132585 | SHA256与摘要在机器总账；按各自协议引用 |
| [dedup_fp_motorcycle_sensitivity.json](ultralytics/algorithm/evaluation_reports/dedup_fp_motorcycle_sensitivity.json) | 98061 | SHA256与摘要在机器总账；按各自协议引用 |
| [dedup_results.json](ultralytics/algorithm/evaluation_reports/dedup_results.json) | 658360 | SHA256与摘要在机器总账；按各自协议引用 |
| [depth_bev_results.json](ultralytics/algorithm/evaluation_reports/depth_bev_results.json) | 31613 | SHA256与摘要在机器总账；按各自协议引用 |
| [efficient_fusion_frame_counts.json](ultralytics/algorithm/evaluation_reports/efficient_fusion_frame_counts.json) | 4128417 | SHA256与摘要在机器总账；按各自协议引用 |
| [efficient_fusion_results.json](ultralytics/algorithm/evaluation_reports/efficient_fusion_results.json) | 1339866 | SHA256与摘要在机器总账；按各自协议引用 |
| [efficient_fusion_timing.json](ultralytics/algorithm/evaluation_reports/efficient_fusion_timing.json) | 12088 | SHA256与摘要在机器总账；按各自协议引用 |
| [failure_audit_summary.json](ultralytics/algorithm/evaluation_reports/failure_audit_summary.json) | 1691 | SHA256与摘要在机器总账；按各自协议引用 |
| [final_four_pipeline_results.json](ultralytics/algorithm/evaluation_reports/final_four_pipeline_results.json) | 319640 | SHA256与摘要在机器总账；按各自协议引用 |
| [final_fp_cases.json](ultralytics/algorithm/evaluation_reports/final_fp_cases.json) | 88995 | SHA256与摘要在机器总账；按各自协议引用 |
| [final_verification_manifest.json](ultralytics/algorithm/evaluation_reports/final_verification_manifest.json) | 14920 | SHA256与摘要在机器总账；按各自协议引用 |
| [final_verification_results.json](ultralytics/algorithm/evaluation_reports/final_verification_results.json) | 26915 | SHA256与摘要在机器总账；按各自协议引用 |
| [four_ideas_f1_intervals.json](ultralytics/algorithm/evaluation_reports/four_ideas_f1_intervals.json) | 651 | SHA256与摘要在机器总账；按各自协议引用 |
| [four_ideas_results.json](ultralytics/algorithm/evaluation_reports/four_ideas_results.json) | 456351 | SHA256与摘要在机器总账；按各自协议引用 |
| [front_fp_diagnosis_summary.json](ultralytics/algorithm/evaluation_reports/front_fp_diagnosis_summary.json) | 2973 | SHA256与摘要在机器总账；按各自协议引用 |
| [global_fusion_audit.json](ultralytics/algorithm/evaluation_reports/global_fusion_audit.json) | 7844666 | SHA256与摘要在机器总账；按各自协议引用 |
| [global_fusion_frame_counts.json](ultralytics/algorithm/evaluation_reports/global_fusion_frame_counts.json) | 5160267 | SHA256与摘要在机器总账；按各自协议引用 |
| [global_fusion_protocol.json](ultralytics/algorithm/evaluation_reports/global_fusion_protocol.json) | 58218 | SHA256与摘要在机器总账；按各自协议引用 |
| [global_fusion_results.json](ultralytics/algorithm/evaluation_reports/global_fusion_results.json) | 127802 | SHA256与摘要在机器总账；按各自协议引用 |
| [global_fusion_timing.json](ultralytics/algorithm/evaluation_reports/global_fusion_timing.json) | 10251 | SHA256与摘要在机器总账；按各自协议引用 |
| [multiview_holdout.json](ultralytics/algorithm/evaluation_reports/multiview_holdout.json) | 10229 | SHA256与摘要在机器总账；按各自协议引用 |
| [multiview_v2.json](ultralytics/algorithm/evaluation_reports/multiview_v2.json) | 5423 | SHA256与摘要在机器总账；按各自协议引用 |
| [multiview_yolo11_640.json](ultralytics/algorithm/evaluation_reports/multiview_yolo11_640.json) | 10745 | SHA256与摘要在机器总账；按各自协议引用 |
| [multiview_yolo11_960.json](ultralytics/algorithm/evaluation_reports/multiview_yolo11_960.json) | 10799 | SHA256与摘要在机器总账；按各自协议引用 |
| [retention_fresh_manifest.json](ultralytics/algorithm/evaluation_reports/retention_fresh_manifest.json) | 4737 | SHA256与摘要在机器总账；按各自协议引用 |
| [retention_fresh_results.json](ultralytics/algorithm/evaluation_reports/retention_fresh_results.json) | 26214 | SHA256与摘要在机器总账；按各自协议引用 |
| [single_camera_comparison.json](ultralytics/algorithm/evaluation_reports/single_camera_comparison.json) | 20043 | SHA256与摘要在机器总账；按各自协议引用 |
| [staged_regression.json](ultralytics/algorithm/evaluation_reports/staged_regression.json) | 25866 | SHA256与摘要在机器总账；按各自协议引用 |
| [support_regression.json](ultralytics/algorithm/evaluation_reports/support_regression.json) | 28890 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_appearance_results_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_appearance_results_20261006.json) | 37471 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_association_constraints_frame_counts_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_association_constraints_frame_counts_20261006.json) | 5436240 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_association_constraints_results_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_association_constraints_results_20261006.json) | 34820 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_association_origins_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_association_origins_20261006.json) | 749538 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_data_audit_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_data_audit_20261006.json) | 581818 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_depth_signals_results_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_depth_signals_results_20261006.json) | 37762 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_geometry_frame_counts_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_geometry_frame_counts_20261006.json) | 15266670 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_geometry_results_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_geometry_results_20261006.json) | 72136 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_gt_geometry_audit_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_gt_geometry_audit_20261006.json) | 90382 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_quality_E1_results_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_quality_E1_results_20261006.json) | 45339 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_stage1_frame_counts_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_stage1_frame_counts_20261006.json) | 6559537 | SHA256与摘要在机器总账；按各自协议引用 |
| [topic3_stage1_results_20261006.json](ultralytics/algorithm/evaluation_reports/topic3_stage1_results_20261006.json) | 32567 | SHA256与摘要在机器总账；按各自协议引用 |
| [yolo_aware_results.json](ultralytics/algorithm/evaluation_reports/yolo_aware_results.json) | 5795 | SHA256与摘要在机器总账；按各自协议引用 |

算法MD、所有experiment manifest、缓存帧数与完成状态也已索引在JSON。experiments与原图不在Git备份范围，总账不能替代原文件备份。

## 9. 复现与保留

```powershell
uv run --project ultralytics python scripts/build_thesis_data_ledger.py
```

保留：原图、原XML、标定、原包、YOLO权重、uv.lock、代码、冻结protocol、检测缓存、逐帧对象及计数、manifest、完成证明、人工复核图。

JSON保存当前Git HEAD/dirty状态、实际依赖与平台、历史实验环境、主方法完整参数。当前目录有未提交研究文件，本轮不自动commit。

论文目前可引用同域对象级对照与消融；同步、全局ID、独立泛化、绝对3D精度、功耗等未经确认的结论暂不写成已证实。
