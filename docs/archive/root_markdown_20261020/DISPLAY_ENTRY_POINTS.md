# 展示入口与启动命令

## 新增：根目录一键菜单（2026-10-20）

双击 **[start_viewers.cmd](start_viewers.cmd)**，或运行：

```powershell
uv run --offline --no-sync --project ultralytics python launch_viewers.py
```

菜单默认是**三目无BAND四画面：Cam2、Cam3、Cam4各自检测＋融合结果**。另有五目自然全景＋BEV＋五路原图、空间环视＋纹理BEV、电影式可拖动环视，以及历史BAND和三个单路入口。可选自动／逐帧、起始帧和播放FPS；三目／单路可显式复用验证缓存，仍按原入口处理深度。五目本来就是缓存重放。每次只从菜单启动一个查看器，先按Q退出再换；启动器关闭不强制终止查看器。

启动器只调整初始显示上限为1280×720，不修改原检测/融合/自车排除/统计/模型。**五目三个显示选择不是新算法，仍为原P90.45/R42.69配置；不把研究P90.84/R44.85写进旧画面。** 三目继续实际计算本次访问帧的四路统计，五目统计分母不同。

不打开菜单也可以直接启动：

```powershell
python launch_viewers.py --view three --mode auto
python launch_viewers.py --view five --mode step --start 30
python launch_viewers.py --view bev --mode auto
python launch_viewers.py --view surround --mode auto
```

这里只需标准库运行菜单；子进程优先使用uv的原项目环境，离线且不更新依赖；找不到uv时使用项目已有`.venv`。原有下列入口和命令全部保留。

更新：2026-10-13。以下命令均在项目根目录 `D:\Yolo-Dissertation` 执行；Git根目录在这里，uv项目仍在 `ultralytics/`。

```powershell
Set-Location 'D:\Yolo-Dissertation'
```

## 1. 先选你要看的画面

| 想看什么 | Python入口 | 选择方式 |
|---|---|---|
| 全景电影式环视，可以拖动转头、缩放 | [final_immersive_cam.py](ultralytics/decided/final_immersive_cam.py) | 默认画面；加 `--dashboard` 同时显示BEV和五路原图 |
| 斜俯视、中央有示意自车的空间环视＋地面纹理BEV | [final_spatial_cam.py](ultralytics/decided/final_spatial_cam.py) | 默认画面；就是之前BEV目标黑块已修复的入口 |
| 更自然的正前方全景＋BEV雷达图＋五路原图 | [final_five_cam.py](ultralytics/decided/final_five_cam.py) | 加 `--panorama natural` |
| 原始标定旋转全景＋BEV雷达图＋五路原图 | [final_five_cam.py](ultralytics/decided/final_five_cam.py) | 默认 `--panorama rotation` |
| 以前的Cam2、Cam3、Cam4＋三目融合四画面 | [final_three_cam.py](ultralytics/decided/final_three_cam.py) | 默认BAND＋深度＋E1；不是当前五目方案 |
| 同样的三目四画面，但不使用BAND | [final_three_cam_no_band.py](ultralytics/decided/final_three_cam_no_band.py) | 默认当前评分冻结＋E1；不是五目方案 |
| Cam2 / Cam3 / Cam4各自单独检测 | [final_cam2.py](ultralytics/decided/final_cam2.py)、[final_cam3.py](ultralytics/decided/final_cam3.py)、[final_cam4.py](ultralytics/decided/final_cam4.py) | 单路对照，不是融合结果 |

前四种画面共享当前冻结五目输出，**区别是展示，不是四种不同的检测算法**。采用签名预测缓存重放，不重新运行YOLO；仍会读取标注并统计本次访问帧的五目GT结果。完整1655帧对应P90.45%、R42.69%、F1 58.00%；窗口刚打开时的累计指标只对应已访问帧，不是全域指标。

后三种为历史三目／单路入口，算法、输入和统计范围不同，不直接与五目数字混比。三目入口默认实际检测和双目计算，可显式使用其已验证缓存选项，详见旧运行指南。

## 2. 全景电影式拖动环视

```powershell
# 自动播放：可拖动看四周
uv run --project ultralytics python ultralytics/decided/final_immersive_cam.py --mode auto

# 同时显示BEV雷达图和五路原图
uv run --project ultralytics python ultralytics/decided/final_immersive_cam.py --mode auto --dashboard

# 逐帧：暂停时仍能转头、缩放
uv run --project ultralytics python ultralytics/decided/final_immersive_cam.py --mode step --dashboard
```

鼠标左键拖动转头，滚轮缩放；F切换仪表盘；R重置；1前／2左／3后／4右；A/D左右转头，W/S抬头低头。N下一帧、B上一帧，空格自动／暂停，Q或Esc退出。键盘字母使用小写键值，不需按Shift。

这是可自由转头的视窗，不是围绕自车旋转的第三人称3D重建。可信投影水平约343°，没有可靠图像的方向显示缺口，不能补成假的360°场景。

详细说明：[360°入口、实现和验证](ultralytics/algorithm/IMMERSIVE_360_LOOK_AROUND_20261013.md)。

## 3. 空间环视＋地面纹理BEV（黑块修复后的入口）

```powershell
uv run --project ultralytics python ultralytics/decided/final_spatial_cam.py --mode auto
uv run --project ultralytics python ultralytics/decided/final_spatial_cam.py --mode step
```

2026-10-13已改成小屏紧凑界面：初始窗口默认最大1280×720，并按Windows当前显示器可用空间缩小。最大化／拖动调整窗口后，界面跟随窗口重排，不再停在原小画布。按F或右上角Fullscreen按钮进入／退出全屏，全屏中Esc先恢复窗口。默认只看“空间环视＋纹理BEV”，不再把所有内容堆在下面。P/R/F1独立显示。

按 `1` 看总览，`2` 看五路原图，`3` 看目标信息，也可点击页签；目标列表用 `[`、`]` 翻页。V切总览中的原自然全景；N/B翻帧、空格自动／暂停、Q或Esc退出。底部Back、Play/Pause、Next、Exit支持点击。

```powershell
# 更小窗口
uv run --project ultralytics python ultralytics/decided/final_spatial_cam.py --mode auto --max-width 1024 --max-height 640

# 直接打开五路原图页
uv run --project ultralytics python ultralytics/decided/final_spatial_cam.py --mode step --page cameras
```

当前修复：环视保留真实目标图像；BEV用类别图标、编号和原不确定性椭圆显示可定位目标，斜线区表示非地面目标纹理被移除，不是其实际占地范围。暗棋盘表示没有可靠地面纹理，包括超出近处纹理范围和没有相机覆盖的位置。距离未知只显示方位，不编造位置。

标记漂移修复：环视轮廓/锚点改用图像本身的相同采样映射，不再把原真实3D点直接叠加到假设碗面上；真实位置仍在BEV显示。楼房/树木的碗面拉伸和拼缝不等于已解决，需要更自然的前向图可按V。[原因与八帧前后验证](ultralytics/algorithm/SPATIAL_OVERLAY_ALIGNMENT_20261013.md)。

它不是FastBEV网络推理，也不是真实稠密3D重建。已打开的旧窗口需要退出并重新启动，才能加载新代码。

详细说明：[小屏界面与验证](ultralytics/algorithm/SPATIAL_COMPACT_UI_20261013.md)、[空间视图原理](ultralytics/algorithm/SPATIAL_SURROUND_AND_TEXTURE_BEV_20261012.md)、[黑块修复与验证](ultralytics/algorithm/BEV_OBJECT_DISPLAY_FIX_20261013.md)。

## 4. 自然前向全景＋BEV雷达图

```powershell
uv run --project ultralytics python ultralytics/decided/final_five_cam.py --mode auto --panorama natural
uv run --project ultralytics python ultralytics/decided/final_five_cam.py --mode step --panorama natural
```

Cam3主体画面＋侧向补图＋接缝融合，右侧为目标位置／不确定性雷达式BEV，下方为五路原图。不是上一个入口的地面纹理BEV，也不是可拖动视窗。

N/B翻帧、空格自动／暂停、Q或Esc退出。自然拼接开销较原旋转模式大；近处目标仍可能存在视差，不能保证全序列无缝、无闪烁。

详细说明：[自然全景拼接](ultralytics/algorithm/NATURAL_PANORAMA_STITCHING_20261012.md)。

## 5. 原旋转全景＋BEV雷达图

```powershell
# 不指定panorama时默认就是rotation
uv run --project ultralytics python ultralytics/decided/final_five_cam.py --mode auto
uv run --project ultralytics python ultralytics/decided/final_five_cam.py --mode step
```

显示五目全景、原近似位置雷达图、五路原图及对象列表。只用标定旋转进行图像映射，近景拼缝容易出现错位，但显示开销较低。它与自然模式使用相同目标输出和统计。

详细说明：[五目全景／BEV和统计](ultralytics/algorithm/FIVE_CAMERA_PANORAMA_BEV_AND_RECALL_STUDY_20261012.md)。

## 6. 历史三目四画面入口

```powershell
# BAND＋深度＋E1，Cam234与融合画面
uv run --project ultralytics python ultralytics/decided/final_three_cam.py --mode auto
uv run --project ultralytics python ultralytics/decided/final_three_cam.py --mode step

# 无BAND，当前评分冻结＋E1
uv run --project ultralytics python ultralytics/decided/final_three_cam_no_band.py --mode auto
uv run --project ultralytics python ultralytics/decided/final_three_cam_no_band.py --mode step
```

无BAND入口默认 `--score-profile frozen`；历史Cam234分母下P93.25%、R36.82%。`--score-profile 960`是另一已保存实验的评分配置，不是默认，不能把两者混成一个结果。

三目窗口及终端有Cam234和融合的统计；主表采用Cam234整体GT，自身视场统计另报。不存在“打开就固定显示完整全域分数”：只累计本次唯一已访问帧。

## 7. 三个单路对照

```powershell
uv run --project ultralytics python ultralytics/decided/final_cam2.py --mode auto
uv run --project ultralytics python ultralytics/decided/final_cam3.py --mode auto
uv run --project ultralytics python ultralytics/decided/final_cam4.py --mode auto
```

逐帧把 `--mode auto` 改为 `--mode step`。这些是各自YOLO、自车排除的历史对照入口，不包含其他视角的融合补盲，也不能当作当前五目冻结系统的同输入消融。

三目与单路的完整参数、缓存、评价和历史范围：[FINAL_RUN_GUIDE.md](ultralytics/algorithm/FINAL_RUN_GUIDE.md)。

## 8. 查看指定帧、保存结果

最近三个五目Python入口均支持 `--start`、`--limit`、`--stride`、`--save`、`--headless`（无窗口需 `--mode auto`）。例如：

```powershell
# 打开第30帧，逐帧模式
uv run --project ultralytics python ultralytics/decided/final_spatial_cam.py --mode step --start 00000030

# 无窗口处理第30帧并保存修复后的环视／BEV
uv run --project ultralytics python ultralytics/decided/final_spatial_cam.py --mode auto --headless --start 00000030 --limit 1 --save

# 保存电影式视窗、BEV及五路原图布局
uv run --project ultralytics python ultralytics/decided/final_immersive_cam.py --mode auto --headless --dashboard --start 00000030 --limit 1 --save
```

默认日志位于以下目录的新 `run_<时间戳>` 子目录，不覆盖旧实验；加 `--save` 才保存PNG：

- 原／自然全景：`ultralytics/experiments/five_camera_viewer_20261012/`。
- 空间环视／纹理BEV：`ultralytics/experiments/spatial_viewer_20261012/`。
- 拖动环视：`ultralytics/experiments/immersive_viewer_20261013/`。
- 历史三目／单路：`ultralytics/experiments/final_runs/`。

日期是入口默认目录名，不代表每次运行日期。输出配置、对象日志、累计统计分别为 `config.json`、`objects.jsonl`、`metrics.json`。PNG不是电影文件；目前没有自动导出MP4或VR视频元数据。
