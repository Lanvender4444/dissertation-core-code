# -*- coding: utf-8 -*-
"""
tri-compare.py —— 三相机共线约束的框级比对 / 分类 / 投影 / 统计
=================================================================
需求实现（逐条对应）：
  1) 对每个框先判断是否位于三相机视野交集处；
  2) 交集处采用"比对"法：
       对面两个都有  -> 完全比对框（完全正确, 三个相机都看到）
       只有两个有    -> 一半比对框（另外一个相机有闪失, 并报告是哪一个）
       只有一个有    -> 假阳性框
  3) 根据"共有且比对完成部分"建立的几何（旋转校正 + 视差-深度），
     将左右 cam 独有的框投影到中间 cam3 的画布上；
  4) 统计四类框的占比：完全比对框 / 一半比对框 / 假阳性框 / 左右独有框。

数学原理（区别于之前 decided/ 与 test/ 目录中的拼接、SGBM 方案）：
  由 T_imu_cam 可验证 cam2/cam3/cam4 三光心严格共线（残差<1mm），
  基线几乎完全沿 cam3 图像 x 轴：
      b2 = R3ᵀ(t2−t3) ≈ [−0.252, 0, 0] m   (cam2 = 左)
      b4 = R3ᵀ(t4−t3) ≈ [+0.499, 0, 0] m   (cam4 = 右)
  将三图去畸变并只做旋转校正（消除 ~2° 朝向差）到统一的 cam3 视角后，
  同一物体必然满足：
      (1) 极线约束：  v2′ = v3 = v4′       （竖直坐标相等）
      (2) 尺寸约束：  h2′ = h3 = h4′       （像素高度相等）
      (3) 视差比约束：δi = ui′ − u3 = −fu·bix/Z
          ⇒ δ2/δ4 = b2x/b4x = −0.505 （与物体无关的常数）
          ⇒ û3 = (b4x·u2′ − b2x·u4′)/(b4x−b2x)  （cam3 位置可被左右相机精确预测）
          ⇒ Z  = fu·(b4x−b2x)/(u2′−u4′)          （比对成功即得深度）
  这三条硬约束取代了地面单应 + IoU 的匹配，不再依赖"物体贴地"假设，
  且显式处理了 k1≈−0.23 的镜头畸变。

用法：
  python tri-compare.py                      # 单帧 demo: ./decided/cam2-4/camX.png
  python tri-compare.py --seq <目录>          # 序列: <目录>/camera_{2,3,4}/*.png
  python tri-compare.py --nogui --save 5     # 无窗口, 保存前5帧渲染结果
"""
import os
import glob
import argparse
import cv2
import numpy as np

# ============================================================
# 相机标定（与 decided/ 下其余脚本一致的内联写法）
# ============================================================
camera_params = {
    2: {
        "width": 2464, "height": 2056,
        "fu": 1443.81646341, "fv": 1445.51112485,
        "cu": 1229.04954947, "cv": 1071.27421460,
        "k1": -0.22868332, "k2": 0.06160952, "p1": -0.00029277, "p2": -0.00028234,
        "T_imu_cam": [
            0.0058857061,-0.0340465292,0.9994029179,2.2127853863,
            -0.9999396530,0.0090706493,0.0061978757,0.3732880221,
            -0.0092762496,-0.9993790858,-0.0339910874,1.4387146391,
            0.0,0.0,0.0,1.0
        ]
    },
    3: {
        "width": 2464, "height": 2056,
        "fu": 1442.25928458, "fv": 1443.53427172,
        "cu": 1222.99206886, "cv": 1054.31641512,
        "k1": -0.22944378, "k2": 0.06277848, "p1": -0.00014530, "p2": -0.00031366,
        "T_imu_cam": [
            0.0010341544,-0.0312462017,0.9995111832,2.2129897274,
            -0.9996543161,-0.0262907445,0.0002124149,0.1211855457,
            0.0262712559,-0.9991658879,-0.0312625891,1.4398252164,
            0.0,0.0,0.0,1.0
        ]
    },
    4: {
        "width": 2464, "height": 2056,
        "fu": 1442.33363296, "fv": 1446.01690618,
        "cu": 1249.99894949, "cv": 1081.05951768,
        "k1": -0.22757254, "k2": 0.06116986, "p1": -0.00029809, "p2": -0.00059963,
        "T_imu_cam": [
            -0.0016349029,-0.0417161259,0.9991281659,2.2099962858,
            -0.9999754382,0.0068777006,-0.0013491279,-0.3780594324,
            -0.0068154240,-0.9991058313,-0.0417263456,1.4404701487,
            0.0,0.0,0.0,1.0
        ]
    }
}

CAM_IDS, REF = (2, 3, 4), 3

# ============================================================
# 自适应准入（新融合算法）
#   检测阶段放宽到 DET_CONF, 弱检测也进入比对池;
#   准入门槛随"证据量"递减、随"位置该被看见的程度"递增:
#     三框共识    -> 门槛最低 (三相机同时误检同一位置几乎不可能)
#     双框共识    -> 略高
#     单框在盲区/边缘(其他两cam看不到) -> 较高 (无从印证, 但也无从反驳)
#     单框在交集中心(其他两cam本应看到) -> 最高 (被反驳了)
# ============================================================
DET_CONF           = 0.20   # YOLO 检测阶段阈值(放宽, 由准入门槛把关)
YOLO_CONF          = DET_CONF                # 兼容旧名
# 五类门槛经 GT 五维网格搜索标定 (42帧抽样, min-gt-h=60, IoU=0.5,
# 完整搜索结果见 .review/gate_search_report.md)。
# 两个反直觉发现:
#  (1) 三相机视角高度相似 => 会在同一位置"相关性误检"(同一棵树三个cam一起看错);
#  (2) 共识框的 conf 是成员最大值, 连最大值都 <0.5 的三框共识几乎全是
#      相关性误检 => 三框门槛反而要高。
# 结论: 门槛分级由"证据数量"转为"证据质量": 强共识最可信,
#       弱共识不如强单框; 盲区双框(2+4侧对)极可靠可放行。
GATE_TRI           = 0.50   # 三框(FULL): 成员最大conf需过此值
GATE_DUAL_BLIND    = 0.20   # 双框·第三cam在盲区: 样本中100%正确, 放行
GATE_DUAL_CENTER   = 0.20   # 双框·第三cam本应看到 (F1最优; 高精度场景建议0.60)
GATE_SINGLE_EDGE   = 0.35   # 单框·盲区(UNIQ/UNVER): 无从印证也无从反驳
GATE_SINGLE_CENTER = 0.45   # 单框·交集中心: 被两台相机"反驳" (高精度场景建议0.60)
GATE_DUAL          = GATE_DUAL_CENTER        # 兼容旧名
EDGE_CONF          = GATE_SINGLE_EDGE        # 兼容旧名

# 输出模式: adaptive=自适应准入(新算法)  all=全部输出
#          full+half=只留多相机共识     full=只留三目共识
OUTPUT_MODES = ('adaptive', 'all', 'full+half', 'full')
CLASS_NAMES  = {0: "Person", 1: "Bike", 2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}
CLASS_HEIGHT = {0: 1.70, 1: 1.50, 2: 1.55, 3: 1.60, 5: 3.20, 7: 3.50}
CLASS_WIDTH  = {0: 0.60, 1: 0.80, 2: 1.90, 3: 0.90, 5: 2.60, 7: 2.60}
VEHICLE_GRP  = {2, 5, 7}

# ---- 匹配容差（校正后像素坐标系） ----
V_TOL      = lambda h: max(0.35 * h, 22)   # 极线容差
H_RATIO    = (0.50, 2.00)                  # 高度比
H_RATIO_TR = (0.33, 3.00)                  # 截断框放宽
Z_RANGE    = (0.5, 250.0)                  # 合理深度(m)
TRUNC_PX   = 25                            # 边缘截断判定(含检测量化误差)
SIZE_TOL   = (0.30, 3.3)
U_TOL_TRI  = lambda w: max(0.45 * w, 28)   # û3 预测容差
WRONG_SIGN_SLACK = 8.0

# 四类框（用户需求的分类）+ 一个兜底类
LABELS = {
    "FULL":   ((0, 255,   0), "FULL",   "完全比对框(三个都有)"),
    "HALF":   ((0, 255, 255), "HALF",   "一半比对框(两个有)"),
    "FP":     ((120, 120, 120), "FALSE-POS", "假阳性框(只有一个)"),
    "UNIQ":   ((0,  80, 255), "UNIQUE", "左右cam独有框(交集外)"),
    "UNVER":  ((200, 120, 200), "UNVER", "单框但他cam物理上看不到"),
}


# ============================================================
# 共线三目校正器
# ============================================================
class TriRectifier:
    def __init__(self, params):
        self.K, self.dist, self.R_rel, self.b = {}, {}, {}, {}
        Ts = {cid: np.array(p["T_imu_cam"]).reshape(4, 4) for cid, p in params.items()}
        R_ref, t_ref = Ts[REF][:3, :3], Ts[REF][:3, 3]
        for cid, p in params.items():
            self.K[cid] = np.array([[p["fu"], 0, p["cu"]],
                                    [0, p["fv"], p["cv"]],
                                    [0, 0, 1]], dtype=np.float64)
            self.dist[cid] = np.array([p["k1"], p["k2"], p["p1"], p["p2"]], np.float64)
            self.R_rel[cid] = R_ref.T @ Ts[cid][:3, :3]
            self.b[cid] = R_ref.T @ (Ts[cid][:3, 3] - t_ref)
        self.K3 = self.K[REF]
        self.fu, self.fv = self.K3[0, 0], self.K3[1, 1]
        self.W, self.H = params[REF]["width"], params[REF]["height"]
        self.bx = {cid: self.b[cid][0] for cid in params}
        self.u_bounds = {}
        for cid in params:
            corners = np.array([[0, 0], [self.W-1, 0], [0, self.H-1], [self.W-1, self.H-1]], np.float64)
            rc = self.rect_points(corners, cid)
            self.u_bounds[cid] = (rc[:, 0].min(), rc[:, 0].max())
        self._maps = {}

    def rect_points(self, pts, cid):
        """原始(带畸变)像素 -> 校正后统一 cam3 像素坐标"""
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(pts, self.K[cid], self.dist[cid]).reshape(-1, 2)
        d = np.column_stack([n, np.ones(len(n))])
        d = (self.R_rel[cid] @ d.T).T
        u = self.fu * d[:, 0] / d[:, 2] + self.K3[0, 2]
        v = self.fv * d[:, 1] / d[:, 2] + self.K3[1, 2]
        return np.column_stack([u, v])

    def rect_box(self, box, cid):
        x1, y1, x2, y2 = box
        xm, ym = (x1+x2)/2, (y1+y2)/2
        rc = self.rect_points([(x1,y1),(x2,y1),(x1,y2),(x2,y2),(xm,y1),(xm,y2),(x1,ym),(x2,ym)], cid)
        return np.array([rc[:,0].min(), rc[:,1].min(), rc[:,0].max(), rc[:,1].max()])

    def rect_image(self, img, cid):
        if cid not in self._maps:
            self._maps[cid] = cv2.initUndistortRectifyMap(
                self.K[cid], self.dist[cid], self.R_rel[cid], self.K3,
                (self.W, self.H), cv2.CV_16SC2)
        m1, m2 = self._maps[cid]
        return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)

    # ---- 数学规律 ----
    def disparity(self, cid, Z):                 # δ = u_i′ − u3
        return -self.fu * self.bx[cid] / Z

    def depth_from_pair(self, u_a, ca, u_b, cb): # 比对成功 -> 深度
        du = u_a - u_b
        if abs(du) < 1e-6:
            return np.inf
        return self.fu * (self.bx[cb] - self.bx[ca]) / du

    def predict_u3_from_sides(self, u2, u4):     # 左右相机 -> 预测 cam3 位置
        b2x, b4x = self.bx[2], self.bx[4]
        return (b4x * u2 - b2x * u4) / (b4x - b2x)

    def predict_u(self, u3, cid, Z):
        return u3 + self.disparity(cid, Z)

    def visible_in(self, cid, u3, Z_lo, Z_hi, margin=15):
        lo, hi = self.u_bounds[cid]
        return all(lo + margin <= self.predict_u(u3, cid, Z) <= hi - margin
                   for Z in (Z_lo, Z_hi))

    def in_intersection(self, u3, Z_lo, Z_hi):
        """需求第1步：该位置是否处于三相机视野交集"""
        return all(self.visible_in(c, u3, Z_lo, Z_hi) for c in CAM_IDS)


# ============================================================
# 匹配工具
# ============================================================
def cls_ok(c1, c2):
    return c1 == c2 or ({c1, c2} <= VEHICLE_GRP)

def center(b):
    return (b[0]+b[2])/2, (b[1]+b[3])/2

def is_trunc(d):
    return d['tl'] or d['tr'] or d['tb']

def gate(dA, dB):
    """极线 + 高度硬约束"""
    if not cls_ok(dA['cls'], dB['cls']):
        return False
    bA, bB = dA['rbox'], dB['rbox']
    hA, hB = bA[3]-bA[1], bB[3]-bB[1]
    if abs(center(bA)[1] - center(bB)[1]) > V_TOL(min(hA, hB)):
        return False
    lo, hi = H_RATIO_TR if (is_trunc(dA) or is_trunc(dB)) else H_RATIO
    return lo <= hA/(hB+1e-6) <= hi

def match_u(*dets):
    """截断感知的代表横坐标：截断框中心偏移, 用未截断的那条竖直边"""
    if any(d['tr'] for d in dets):
        return [d['rbox'][0] for d in dets]
    if any(d['tl'] for d in dets):
        return [d['rbox'][2] for d in dets]
    return [(d['rbox'][0]+d['rbox'][2])/2 for d in dets]

def size_ok(rect, det, Z):
    """像素高度反推物理高度 vs 类别先验（截断框豁免）"""
    if det['tb'] or (det['tl'] and det['tr']):
        return True
    if not np.isfinite(Z) or Z <= 0:
        return False
    H_est = (det['rbox'][3]-det['rbox'][1]) * Z / rect.fv
    H_exp = CLASS_HEIGHT.get(det['cls'], 1.6)
    return SIZE_TOL[0]*H_exp <= H_est <= SIZE_TOL[1]*H_exp

def pair_depth(rect, d_side, d3):
    u_s, u_3 = match_u(d_side, d3)
    delta = u_s - u_3
    want = -rect.bx[d_side['src']]
    if delta * want < 0:
        if abs(delta) > WRONG_SIGN_SLACK:
            return None
        delta = np.sign(want) * 1e-3
    Z = rect.fu * abs(rect.bx[d_side['src']]) / max(abs(delta), 1e-3)
    return Z if Z_RANGE[0] <= Z <= Z_RANGE[1]*4 else None

def score(dA, dB):
    bA, bB = dA['rbox'], dB['rbox']
    hA, hB = bA[3]-bA[1], bB[3]-bB[1]
    h = min(hA, hB) + 1e-6
    return abs(center(bA)[1]-center(bB)[1])/h + abs(hA-hB)/h

def est_depth_single(rect, d):
    """单框深度先验：底部截断用宽度, 否则用高度"""
    rb = d['rbox']
    if d['tb']:
        Z = rect.fu * CLASS_WIDTH.get(d['cls'], 1.6) / max(rb[2]-rb[0], 1e-3)
    else:
        Z = rect.fv * CLASS_HEIGHT.get(d['cls'], 1.6) / max(rb[3]-rb[1], 1e-3)
    return float(np.clip(Z, Z_RANGE[0], Z_RANGE[1]))


# ============================================================
# 比对 + 分类主算法
# ============================================================
def compare_and_classify(rect, dets):
    for d in dets:
        x1, y1, x2, y2 = d['box']
        d['tl'] = x1 < TRUNC_PX
        d['tr'] = x2 > rect.W - TRUNC_PX
        d['tb'] = y2 > rect.H - TRUNC_PX

    by_cam = {c: [d for d in dets if d['src'] == c] for c in CAM_IDS}
    used = {c: [False]*len(by_cam[c]) for c in CAM_IDS}
    objs = []

    # ---- 1. 三目比对：û3 = f(u2,u4) 预测 + 残差检验 -> 完全比对框 ----
    cands = []
    for i3, d3 in enumerate(by_cam[3]):
        w3 = d3['rbox'][2] - d3['rbox'][0]
        for i2, d2 in enumerate(by_cam[2]):
            if not gate(d2, d3):
                continue
            for i4, d4 in enumerate(by_cam[4]):
                if not gate(d4, d3) or not gate(d2, d4):
                    continue
                u2, u3, u4 = match_u(d2, d3, d4)
                res = abs(rect.predict_u3_from_sides(u2, u4) - u3)
                if res > U_TOL_TRI(w3):
                    continue
                Z = rect.depth_from_pair(u2, 2, u4, 4)
                if not (Z > 0 and size_ok(rect, d3, min(Z, Z_RANGE[1]))):
                    continue
                cands.append((res/(w3+1e-6) + score(d2,d3) + score(d4,d3), i2, i3, i4, Z))
    for s, i2, i3, i4, Z in sorted(cands):
        if used[2][i2] or used[3][i3] or used[4][i4]:
            continue
        used[2][i2] = used[3][i3] = used[4][i4] = True
        d3 = by_cam[3][i3]
        objs.append({'box': d3['rbox'], 'cls': d3['cls'], 'Z': Z, 'src': {2,3,4},
                     'conf': max(by_cam[2][i2]['conf'], d3['conf'], by_cam[4][i4]['conf']),
                     'members': {2:i2, 3:i3, 4:i4}, 'label': 'FULL', 'missing': None})

    # ---- 2. 侧-中比对 -> 一半比对框（第三个相机有闪失） ----
    cands = []
    for side in (2, 4):
        for i_s, ds in enumerate(by_cam[side]):
            for i3, d3 in enumerate(by_cam[3]):
                if used[side][i_s] or used[3][i3] or not gate(ds, d3):
                    continue
                Z = pair_depth(rect, ds, d3)
                if Z is None or not size_ok(rect, d3, min(Z, Z_RANGE[1])):
                    continue
                cands.append((score(ds, d3), side, i_s, i3, Z))
    for s, side, i_s, i3, Z in sorted(cands):
        if used[side][i_s] or used[3][i3]:
            continue
        used[side][i_s] = used[3][i3] = True
        d3 = by_cam[3][i3]
        other = 4 if side == 2 else 2
        u3 = center(d3['rbox'])[0]
        oov = not rect.visible_in(other, u3, Z*0.8, Z*1.25)
        objs.append({'box': d3['rbox'], 'cls': d3['cls'], 'Z': Z, 'src': {side, 3},
                     'conf': max(by_cam[side][i_s]['conf'], d3['conf']),
                     'members': {side:i_s, 3:i3}, 'label': 'HALF',
                     'missing': None if oov else other})

    # ---- 3. 左右比对（cam3 漏检 或 交集外） ----
    cands = []
    for i2, d2 in enumerate(by_cam[2]):
        for i4, d4 in enumerate(by_cam[4]):
            if used[2][i2] or used[4][i4] or not gate(d2, d4):
                continue
            u2, u4 = match_u(d2, d4)
            Z = rect.depth_from_pair(u2, 2, u4, 4)
            if not (Z > 0 and Z_RANGE[0] <= Z <= Z_RANGE[1]*4):
                continue
            if not size_ok(rect, d2, min(Z, Z_RANGE[1])):
                continue
            cands.append((score(d2, d4), i2, i4, Z))
    for s, i2, i4, Z in sorted(cands):
        if used[2][i2] or used[4][i4]:
            continue
        used[2][i2] = used[4][i4] = True
        d2, d4 = by_cam[2][i2], by_cam[4][i4]
        u2, u4 = match_u(d2, d4)
        u3_hat = rect.predict_u3_from_sides(u2, u4)
        delta2 = rect.disparity(2, Z)                       # 投影: 消除视差
        box = d2['rbox'] - np.array([delta2, 0, delta2, 0])
        inside = 0 <= u3_hat < rect.W
        objs.append({'box': box, 'cls': d2['cls'], 'Z': Z, 'src': {2, 4},
                     'conf': max(d2['conf'], d4['conf']),
                     'members': {2:i2, 4:i4},
                     'label': 'HALF' if inside else 'UNIQ',
                     'missing': 3 if inside else None})

    # ---- 3.5 吸附：落单框并入已比对组（候选被占用的情况） ----
    for cid in CAM_IDS:
        for i, d in enumerate(by_cam[cid]):
            if used[cid][i]:
                continue
            best = None
            for o in objs:
                if cid in o['src'] or not cls_ok(d['cls'], o['cls']):
                    continue
                ob = o['box']; oh = ob[3]-ob[1]
                dh = d['rbox'][3]-d['rbox'][1]
                if abs(center(d['rbox'])[1] - (ob[1]+ob[3])/2) > V_TOL(min(oh, dh)):
                    continue
                lo, hi = H_RATIO_TR if is_trunc(d) else H_RATIO
                if not (lo <= dh/(oh+1e-6) <= hi):
                    continue
                u_pred = rect.predict_u((ob[0]+ob[2])/2, cid, max(o['Z'], Z_RANGE[0]))
                res = abs(center(d['rbox'])[0] - u_pred)
                if res > max(0.55*(ob[2]-ob[0]), 30):
                    continue
                if best is None or res < best[0]:
                    best = (res, o)
            if best is not None:
                o = best[1]
                used[cid][i] = True
                o['src'] |= {cid}
                o['members'][cid] = i
                if len(o['src']) >= 3:
                    o['label'], o['missing'] = 'FULL', None

    # ---- 4. 剩余单框：先判交集, 再分类 ----
    for cid in CAM_IDS:
        for i, d in enumerate(by_cam[cid]):
            if used[cid][i]:
                continue
            Z = est_depth_single(rect, d)
            u3_pos = center(d['rbox'])[0] - rect.disparity(cid, Z)   # 投影到 cam3 系
            inter = rect.in_intersection(u3_pos, Z*0.55, Z*1.8)
            inside3 = 0 <= u3_pos < rect.W

            if inter or (cid == 3):
                # 交集处只有一个相机有 -> 假阳性
                others = [c for c in CAM_IDS if c != cid]
                n_vis = sum(rect.visible_in(c, u3_pos, Z*0.55, Z*1.8) for c in others)
                label = 'FP' if n_vis >= 1 else 'UNVER'
            elif not inside3:
                # cam3 视野之外, 左/右相机独有 -> 投影到中间画布
                # (置信度门槛交给 admission() 按盲区/中心自适应把关)
                label = 'UNIQ'
            else:
                label = 'FP'

            delta = rect.disparity(cid, Z)
            box = d['rbox'] - np.array([delta, 0, delta, 0]) if cid != 3 else d['rbox']
            objs.append({'box': box, 'cls': d['cls'], 'Z': Z, 'src': {cid},
                         'conf': d['conf'], 'members': {cid: i},
                         'label': label, 'missing': None})
    return objs, by_cam


# ============================================================
# 准入过滤：三种固定输出模式 + 自适应准入(新算法)
# ============================================================
def admission(objs, mode='adaptive'):
    """
    mode:
      'all'       全部输出（不过滤, 旧行为）
      'full'      只留完全比对框
      'full+half' 只留多相机共识框
      'adaptive'  新算法: 证据越多门槛越低, 单框按位置区分
    返回 (kept, rejected)
    """
    if mode == 'all':
        return objs, []
    if mode == 'full':
        keep = [o for o in objs if o['label'] == 'FULL']
        return keep, [o for o in objs if o['label'] != 'FULL']
    if mode == 'full+half':
        keep = [o for o in objs if o['label'] in ('FULL', 'HALF')]
        return keep, [o for o in objs if o['label'] not in ('FULL', 'HALF')]
    # ---- adaptive: 五类分级准入 ----
    kept, rejected = [], []
    for o in objs:
        n = len(o['src'])
        if n >= 3:
            gate_c = GATE_TRI                 # 三框
        elif n == 2:
            #  missing 有值 = 第三台相机本应看到(有闪失) => 双框·中心
            #  missing 为空 = 第三台相机在盲区/视野外   => 双框·盲区
            gate_c = GATE_DUAL_CENTER if o.get('missing') else GATE_DUAL_BLIND
        elif o['label'] in ('UNIQ', 'UNVER'):
            gate_c = GATE_SINGLE_EDGE         # 单框·盲区: 其他cam本来就看不到
        else:
            gate_c = GATE_SINGLE_CENTER       # 单框·中心: 其他cam本应看到却没有
        (kept if o['conf'] >= gate_c else rejected).append(o)
    return kept, rejected


# ============================================================
# 统计
# ============================================================
class Stats:
    def __init__(self):
        self.counts = {k: 0 for k in LABELS}
        self.miss = {2: 0, 3: 0, 4: 0}
        self.frames = 0

    def update(self, objs):
        self.frames += 1
        for o in objs:
            self.counts[o['label']] += 1
            if o['label'] == 'HALF' and o.get('missing'):
                self.miss[o['missing']] += 1

    def overlay(self):
        t = sum(self.counts.values()) + 1e-7
        return "  ".join(f"{k}:{v}({v/t*100:.0f}%)" for k, v in self.counts.items())

    def report(self):
        t = sum(self.counts.values())
        print("\n" + "="*62)
        print("  框种类占比统计  (collinear-trifocal compare)")
        print("="*62)
        for k, (_, en, zh) in LABELS.items():
            n = self.counts[k]
            pct = n/(t+1e-7)*100
            print(f"  {en:<10} {zh:<18} | {n:5d} ({pct:5.1f}%) {'#'*int(pct/2)}")
        print(f"  {'TOTAL':<10}{'':<20} | {t:5d} (100.0%)")
        print("-"*62)
        print("  一半比对框中有闪失的相机分布: "
              + "  ".join(f"cam{c}:{self.miss[c]}" for c in (2, 3, 4)))
        print(f"  处理帧数: {self.frames}")
        print("="*62)


# ============================================================
# 渲染：cam3 校正图放中央, 独有框投影后落在画布延伸区
# ============================================================
def render(rect, objs, img3):
    cam3r = rect.rect_image(img3, 3)
    H3, W3 = cam3r.shape[:2]
    cw, ch = int(W3*2.2), int(H3*1.1)
    canvas = np.zeros((ch, cw, 3), np.uint8)
    ox, oy = (cw-W3)//2, (ch-H3)//2
    canvas[oy:oy+H3, ox:ox+W3] = cam3r
    for o in objs:
        b = o['box']
        color, en, _ = LABELS[o['label']]
        tag = f"{en} {CLASS_NAMES.get(o['cls'],'')} {o['Z']:.0f}m"
        if o['label'] == 'HALF' and o.get('missing'):
            tag += f" miss-cam{o['missing']}"
        p1 = (int(b[0]+ox), int(b[1]+oy)); p2 = (int(b[2]+ox), int(b[3]+oy))
        cv2.rectangle(canvas, p1, p2, color, 3)
        ts = cv2.getTextSize(tag, 0, 0.9, 2)[0]
        cv2.rectangle(canvas, (p1[0], p1[1]-ts[1]-8), (p1[0]+ts[0], p1[1]), color, -1)
        cv2.putText(canvas, tag, (p1[0], p1[1]-6), 0, 0.9, (255, 255, 255), 2)
    return canvas


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=str, default='',
                    help='序列目录(含 camera_2/3/4 子目录); 缺省跑单帧 demo')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--nogui', action='store_true')
    ap.add_argument('--save', type=int, default=0)
    ap.add_argument('--mode', type=str, default='adaptive', choices=OUTPUT_MODES,
                    help='输出模式: adaptive(自适应准入,新算法) all full+half full')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    rect = TriRectifier(camera_params)
    print(f"[calib] b2x={rect.bx[2]:+.4f}m  b4x={rect.bx[4]:+.4f}m"
          f"  视差比常数 δ2/δ4 = {rect.bx[2]/rect.bx[4]:+.4f}")

    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')

    def detect(img):
        out = []
        res = model(img, verbose=False, conf=YOLO_CONF)[0]
        for bb in res.boxes:
            c = int(bb.cls[0])
            if c in CLASS_NAMES:
                out.append((bb.xyxy.cpu().numpy()[0], float(bb.conf[0]), c))
        return out

    # 帧列表
    if args.seq:
        f3 = sorted(glob.glob(os.path.join(args.seq, 'camera_3', '*.png')))[::args.stride]
        if args.limit:
            f3 = f3[:args.limit]
        frames_iter = [(os.path.basename(p),
                        {c: os.path.join(args.seq, f'camera_{c}', os.path.basename(p))
                         for c in CAM_IDS}) for p in f3]
    else:
        demo = os.path.join(here, 'cam2-4')
        frames_iter = [('demo', {c: os.path.join(demo, f'cam{c}.png') for c in CAM_IDS})]

    if args.save:
        os.makedirs(os.path.join(here, 'tri_compare_out'), exist_ok=True)

    stats = Stats()
    for fi, (fname, paths) in enumerate(frames_iter):
        imgs = {c: cv2.imread(p) for c, p in paths.items()}
        if imgs[3] is None:
            continue
        dets = []
        for cid in CAM_IDS:
            if imgs[cid] is None:
                continue
            for box, conf, cls in detect(imgs[cid]):
                dets.append({'src': cid, 'cls': cls, 'conf': conf,
                             'box': box, 'rbox': rect.rect_box(box, cid)})

        objs, _ = compare_and_classify(rect, dets)
        objs, rejected = admission(objs, args.mode)
        stats.update(objs)

        canvas = render(rect, objs, imgs[3])
        if args.save and fi < args.save:
            cv2.imwrite(os.path.join(here, 'tri_compare_out', f'vis_{fname}'), canvas)
        if not args.nogui:
            top = cv2.hconcat([cv2.resize(imgs[c], (480, 320)) for c in (4, 3, 2)])
            bottom = cv2.resize(canvas, (top.shape[1], 520))
            ui = cv2.vconcat([top, bottom])
            bar = np.zeros((60, ui.shape[1], 3), np.uint8)
            cv2.putText(bar, stats.overlay(), (10, 38), 0, 0.55, (255, 255, 255), 1)
            cv2.imshow('tri-compare', cv2.vconcat([ui, bar]))
            key = cv2.waitKey(0 if not args.seq else 1) & 0xFF
            if key == ord('q'):
                break

    stats.report()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
