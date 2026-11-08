# 空间环视标记漂移：投影一致性修复（2026-10-13）

## 原因和边界

用户截图同时存在碗形图像变形和目标文字/标记偏移。本轮找到了一个明确的**显示坐标不一致**问题：

- 图像：虚拟观察射线与固定碗形表面相交，交点再投影到某个原相机采样RGB。
- 原标记：把原双目估计的真实3D框中心直接投影到虚拟相机，另画从假设地面到3D点的线。

真实目标通常不在显示碗面上，所以二者不是同一种投影。原位置估计即使合理，也不能拿来直接标注被碗面变形的图像。恢复真实目标纹理后，这个错位更显眼；全屏按比例放大只会放大已有偏差，不会自己修改深度或目标坐标。

此前可执行的自然全景使用图像同源映射绘制检测轮廓；碗形空间图是另一种显示。**本次不把误差归因于YOLO退化，不修改相机标定、原融合或深度。**

## 修复原理

记屏幕像素为s、固定碗形交点为Q(s)、实际采样相机为c(s)。显示图像为：

\[
I_{display}(s)=I_{c(s)}(\pi_{c(s)}(Q(s))).
\]

属于相机c的原检测框B，其可画区域应为：

\[
\Omega_B=\{s:c(s)=c,\ \pi_c(Q(s))\in B\}.
\]

旧方法使用真实位置P的虚拟投影 `project(P)`，这不等于上述框在显示图中的位置。

新模块 `ultralytics/research/spatial_observation_overlay.py` 的 `BowlObservations` 直接读取图像的同一套固定点采样LUT和owner，在原图检测框范围内查找实际可见的显示像素，绘制对应轮廓。从实际可见框区域选最接近原框中心的像素作为标记锚点；多路成员优先可见中心、Cam3及较高置信度。只在自身相机提供的纹理上画，不拿另一相机的框覆盖不对应画面。

文本使用多个偏移候选避免重叠，并用指示线连接实际锚点。轮廓是原2D框的显示变换，不是物体轮廓分割、3D包围盒或新定位结果。

空间图取消错误的真实3D点/落地线叠加。真实XY和协方差仍在BEV保持原值；深度未知的检测也可以在可见原图纹理上画框，但BEV仍只显示未知方位。无对应可见纹理时不强行画漂浮标记，完整对象仍在Objects页、原相机页和日志中。

文件关系：

- `ultralytics/research/spatial_surround.py`：调用新观察空间轮廓绘制；原碗面、纹理LUT、BEV不改。
- `ultralytics/research/spatial_observation_overlay.py`：同源索引、可见框选择、轮廓/锚点/标签。
- `ultralytics/research/spatial_dashboard.py`：总览字幕说明为图像对齐轮廓，而非真实3D点。
- `ultralytics/decided/final_spatial_cam.py`：入口不变，配置新增新模块源码签名。

## 启动和使用

退出旧窗口后重新运行：

```powershell
uv run --project ultralytics python ultralytics/decided/final_spatial_cam.py --mode auto
```

需要之前更自然的前向画面时，按V，或直接启动：

```powershell
uv run --project ultralytics python ultralytics/decided/final_spatial_cam.py --mode auto --view panorama
```

这不会改变当前五目预测，只切换显示。

## 八帧前后验证（追加保存，不覆盖）

帧：0、10、17、30、100、650、1200、1600。

修复前诊断：`ultralytics/experiments/spatial_alignment_20261013/before01/`。33个原可定位且有图像锚点的比较中，真实点投影与图像对应锚点的距离中位数24.47像素，20个大于20像素；单位为原960×540碗形画布，放大窗口后屏幕差值更大。这是两种显示投影的差，不是检测误差或3D定位精度。

修复后：`ultralytics/experiments/spatial_alignment_20261013/after02/`。58个实际绘制锚点全部满足同相机owner和原检测框采样范围，并与同LUT求得的预期锚点相同（显示域坐标差0）。这说明绘制与自身图像采样一致，**不等于GT目标定位误差为0**。结果中仍保存24.47像素旧物理投影与纹理投影的差，作为错误旧方式的对照，不是修复后还按旧方式绘制。

八帧原对象/位置指纹、冻结计数一致，BEV整幅图像字节SHA与修复前完全一致。合计58TP/9FP/56FN，样本不作为新全域提升；原FP没有被显示修复消除。

脚本：`scripts/validate_spatial_alignment.py`。保留before01、第一轮after01以及完整锚点核查after02，协议记录当时源码签名。新增观察空间4项测试，加原空间10项、布局8项和入口控制3项，共25项通过。覆盖同源采样、不可见框不画、unknown可以在原图标注、显示不随虚构深度移动、原BEV不变。

实际新入口第17帧保存于 `ultralytics/experiments/spatial_alignment_20261013/run_1791824402840738800/`；9对象、5有原定位，6TP/3FP/2FN，原来源/成员/BEV数值审核通过，报告 `algorithm/evaluation_reports/five_camera_viewer_run_1791824402840738800_audit_20261012.json`。这是单帧统计，不是全域效果。

旧/新独立八帧显示渲染中位数约0.135/0.138秒，仅为单遍小样本渲染时间，不包含IO/定位/评价/窗口，不宣称速度提升或端到端性能保证。新增索引只缓存几何采样，不引入网络。

## 未解决的问题

- 楼房、树木、车辆并不贴在同一碗面上；图像仍可能拉伸、重影和跨相机接缝错位。
- 不同路面坡度下，固定参考平面未必符合当前路面。
- 原框中心测距与真实目标接地点不同，BEV位置和不确定性可能仍不精确。
- ID是帧内编号，不是持续跟踪身份；本轮没有时序跟踪/稳定化。

所以这里只声明**修复同源显示标记的漂移**，不声明真实3D重建、全域无缝或物理位置漂移已全部解决。

## 同帧前后图

修复前（第17帧）：

![修复前](../experiments/spatial_alignment_20261013/before01/00000017/space.png)

修复后：

![修复后](../experiments/spatial_alignment_20261013/after02/00000017/space.png)
