# -*- coding: utf-8 -*-
"""
tri-eval-gt.py —— 以人工标注 XML 为真值的三级准确率对比评估
================================================================
思路（按需求逐条实现）：
  1) 综合三个镜头的 XML（mycar/cam_{2,3,4}/*.xml, instanceId 跨相机一致）,
     按 instanceId 取并集, 生成每帧"这一场存在的全部物体" => 总体集；
  2) 计算并对比三种准确率：
       [A] 单 cam 检测  vs 本镜头 XML     —— 该相机检测水平的上限参照
       [B] 单 cam 检测  vs 总体集 XML     —— 一个相机能覆盖整个场景的多少
       [C] 本算法融合结果 vs 总体集 XML    —— 共线三目融合带来的提升
  3) 额外输出: 融合分类(FULL/HALF/FP/UNIQ)与真值的对应关系,
     验证"假阳性框"是否真的不在真值里。

匹配规则：
  det-GT 匹配在各相机原始(带畸变)像素坐标下进行, IoU >= IOU_TH 且类别相容；
  融合物体与总体集匹配: 其任一成员检测框与该 instanceId 在对应相机的
  GT 框匹配即算命中；
  GT 过滤: 物体在所有相机中框高均 < MIN_GT_H 像素时不计入(不可检测的远小目标),
  过滤阈值会在报告中注明。

用法：
  python tri-eval-gt.py                      # 全部 1655 帧
  python tri-eval-gt.py --stride 10          # 抽样
  python tri-eval-gt.py --min-gt-h 25 --iou 0.5
"""
import os
import sys
import glob
import argparse
import importlib.util
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# ---- 载入 tri-compare.py 核心 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "tri_compare_core", os.path.join(_HERE, "tri-compare.py"))
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

TriRectifier         = core.TriRectifier
compare_and_classify = core.compare_and_classify
camera_params        = core.camera_params
CLASS_NAMES          = core.CLASS_NAMES
YOLO_CONF            = core.YOLO_CONF
CAM_IDS              = core.CAM_IDS
LABELS               = core.LABELS

# GT label -> 可接受的 YOLO 类别（rider=骑车人, YOLO 会出 person/bike）
GT_COMPAT = {
    'car':     {2, 5, 7},
    'bus':     {5, 7, 2},
    'truck':   {7, 5, 2},
    'person':  {0},
    'rider':   {0, 1, 3},
    'bicycle': {1, 3},
    'motorcycle': {3},
}


# ============================================================
# XML 解析与总体集
# ============================================================
def parse_xml(path):
    """返回 [(instanceId, gt_label, [x1,y1,x2,y2]), ...]"""
    out = []
    if not os.path.exists(path):
        return out
    for o in ET.parse(path).getroot().findall('object'):
        lab = o.find('label').text
        iid = int(o.find('instanceId').text)
        d = list(map(float, o.find('vertices/data').text.split()))
        xs, ys = d[0::2], d[1::2]
        out.append((iid, lab, [min(xs), min(ys), max(xs), max(ys)]))
    return out


def build_union(gt_by_cam, min_h):
    """
    总体集: instanceId -> {'label':..., 'boxes': {cid: box}}
    过滤: 所有相机中框高都 < min_h 的物体不计入
    """
    union = {}
    for cid, lst in gt_by_cam.items():
        for iid, lab, box in lst:
            u = union.setdefault(iid, {'label': lab, 'boxes': {}})
            u['boxes'][cid] = box
    return {iid: u for iid, u in union.items()
            if max(b[3] - b[1] for b in u['boxes'].values()) >= min_h}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-9)


def cls_match(gt_label, yolo_cls):
    return yolo_cls in GT_COMPAT.get(gt_label, set())


def match_dets_to_gt(dets, gts, iou_th):
    """
    贪心 IoU 匹配（一对一）。
    dets: [{'box':..,'cls':..}]  gts: [(iid,label,box)]
    返回: det_idx -> iid  的字典
    """
    cand = []
    for di, d in enumerate(dets):
        for gi, (iid, lab, gbox) in enumerate(gts):
            if not cls_match(lab, d['cls']):
                continue
            v = iou(d['box'], gbox)
            if v >= iou_th:
                cand.append((v, di, gi))
    cand.sort(reverse=True)
    used_d, used_g, out = set(), set(), {}
    for v, di, gi in cand:
        if di in used_d or gi in used_g:
            continue
        used_d.add(di); used_g.add(gi)
        out[di] = gts[gi][0]
    return out


# ============================================================
# 主流程
# ============================================================
def main(detector=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-root', type=str,
                    default=os.path.join(_HERE, '..', 'mycar', 'images'))
    ap.add_argument('--gt-root', type=str,
                    default=os.path.join(_HERE, '..', 'mycar'))
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--min-gt-h', type=float, default=25.0,
                    help='GT 最小框高(px), 所有相机都小于此高度的物体不计入')
    ap.add_argument('--out', type=str, default=os.path.join(_HERE, 'tri_eval_out'))
    args = ap.parse_args()

    img_dirs = {c: os.path.join(args.img_root, str(c)) for c in CAM_IDS}
    if not all(os.path.isdir(d) for d in img_dirs.values()):
        img_dirs = {c: os.path.join(args.img_root, f'camera_{c}') for c in CAM_IDS}
    gt_dirs = {c: os.path.join(args.gt_root, f'cam_{c}') for c in CAM_IDS}
    for d in list(img_dirs.values()) + list(gt_dirs.values()):
        if not os.path.isdir(d):
            sys.exit(f"[错误] 目录不存在: {d}")

    frames = sorted(os.path.basename(p) for p in
                    glob.glob(os.path.join(img_dirs[3], '*.png')))[::max(args.stride, 1)]
    if args.limit:
        frames = frames[:args.limit]
    print(f"[frames] {len(frames)} 帧   IoU>={args.iou}   GT最小框高 {args.min_gt_h}px")

    rect = TriRectifier(camera_params)

    if detector is None:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')

        def detector(img):
            out = []
            for bb in model(img, verbose=False, conf=YOLO_CONF)[0].boxes:
                c = int(bb.cls[0])
                if c in CLASS_NAMES:
                    out.append((bb.xyxy.cpu().numpy()[0], float(bb.conf[0]), c))
            return out

    try:
        from tqdm import tqdm
        it = tqdm(frames)
    except ImportError:
        it = frames

    # ---- 累计器 ----
    A = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in CAM_IDS}      # 单cam vs 本镜头
    B = {c: {'hit': 0} for c in CAM_IDS}                       # 单cam vs 总体集(召回)
    MODES = list(core.OUTPUT_MODES)                            # 四种输出模式同时评估
    C = {m: {'obj_tp': 0, 'obj_fp': 0, 'hit': 0} for m in MODES}
    union_total = 0
    label_vs_gt = {k: {'tp': 0, 'fp': 0} for k in LABELS}      # 融合分类与真值对应

    for fname in it:
        stem = os.path.splitext(fname)[0]
        gt_by_cam = {c: parse_xml(os.path.join(gt_dirs[c], stem + '.xml'))
                     for c in CAM_IDS}
        union = build_union(gt_by_cam, args.min_gt_h)
        union_total += len(union)
        # 本镜头 GT 同样过滤: 只保留属于总体集的条目
        gt_f = {c: [(i, l, b) for i, l, b in gt_by_cam[c] if i in union]
                for c in CAM_IDS}

        imgs = {c: cv2.imread(os.path.join(img_dirs[c], fname)) for c in CAM_IDS}
        if imgs[3] is None:
            continue
        dets_by_cam = {}
        all_dets = []
        for cid in CAM_IDS:
            lst = []
            if imgs[cid] is not None:
                for box, conf, cls in detector(imgs[cid]):
                    d = {'src': cid, 'cls': cls, 'conf': conf,
                         'box': np.asarray(box, float),
                         'rbox': rect.rect_box(box, cid)}
                    lst.append(d)
                    all_dets.append(d)
            dets_by_cam[cid] = lst

        # ---- [A]/[B] 单相机 ----
        det2iid = {}
        for cid in CAM_IDS:
            m = match_dets_to_gt(dets_by_cam[cid], gt_f[cid], args.iou)
            det2iid[cid] = m
            tp = len(m)
            A[cid]['tp'] += tp
            A[cid]['fp'] += len(dets_by_cam[cid]) - tp
            A[cid]['fn'] += len(gt_f[cid]) - tp
            B[cid]['hit'] += len(set(m.values()))          # 命中的唯一物体

        # ---- [C] 融合算法 vs 总体集 (四种输出模式同时评估) ----
        objs, by_cam = compare_and_classify(rect, all_dets)
        for o in objs:                       # 每个融合物体先匹配一次真值
            o['_iid'] = None
            for cid, di in o['members'].items():
                if di in det2iid[cid]:
                    o['_iid'] = det2iid[cid][di]
                    break
            k = 'tp' if o['_iid'] is not None else 'fp'
            label_vs_gt[o['label']][k] += 1
        for m in MODES:
            kept, _rej = core.admission(objs, m)
            hit_iids = set()
            for o in kept:
                if o['_iid'] is not None:
                    C[m]['obj_tp'] += 1
                    hit_iids.add(o['_iid'])
                else:
                    C[m]['obj_fp'] += 1
            C[m]['hit'] += len(hit_iids)

    # ============================================================
    # 报告
    # ============================================================
    L = []
    L.append("=" * 70)
    L.append("  三级准确率对比  (真值 = 三镜头 XML 合成的总体集)")
    L.append(f"  帧数 {len(frames)}   总体集物体总数 {union_total}"
             f"   IoU>={args.iou}   GT框高>={args.min_gt_h}px")
    L.append("=" * 70)
    L.append("  [A] 单 cam vs 本镜头 XML  (该相机自己视野内的检测水平)")
    L.append(f"      {'cam':<6}{'TP':>7}{'FP':>7}{'FN':>7}{'精确率':>10}{'召回率':>10}{'F1':>8}")
    for c in CAM_IDS:
        tp, fp, fn = A[c]['tp'], A[c]['fp'], A[c]['fn']
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        L.append(f"      cam{c:<3}{tp:>7}{fp:>7}{fn:>7}{p:>10.1%}{r:>10.1%}{f1:>8.3f}")
    L.append("-" * 70)
    L.append("  [B] 单 cam vs 总体集 XML  (一个相机能覆盖整个场景的多少)")
    L.append(f"      {'cam':<6}{'命中物体':>9}{'总体集':>8}{'场景召回率':>12}")
    for c in CAM_IDS:
        L.append(f"      cam{c:<3}{B[c]['hit']:>9}{union_total:>8}"
                 f"{B[c]['hit']/(union_total+1e-9):>12.1%}")
    L.append("-" * 70)
    L.append("  [C] 本算法(共线三目融合) vs 总体集 XML  —— 四种输出模式对比")
    L.append(f"      {'模式':<12}{'TP':>6}{'FP':>6}{'精确率':>9}{'场景召回率':>11}{'F1':>8}")
    zh_mode = {'adaptive': '自适应准入', 'all': '全部输出',
               'full+half': 'FULL+HALF', 'full': '仅FULL'}
    for m in MODES:
        p = C[m]['obj_tp'] / (C[m]['obj_tp'] + C[m]['obj_fp'] + 1e-9)
        r = C[m]['hit'] / (union_total + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        L.append(f"      {zh_mode.get(m, m):<12}{C[m]['obj_tp']:>6}{C[m]['obj_fp']:>6}"
                 f"{p:>9.1%}{r:>11.1%}{f1:>8.3f}")
    L.append("-" * 70)
    L.append("  [D] 融合分类 与 真值 的对应 (验证四类框的语义)")
    L.append(f"      {'label':<12}{'在真值中':>9}{'不在真值中':>11}{'真值命中率':>12}")
    for k, (_, en, zh) in LABELS.items():
        tp, fp = label_vs_gt[k]['tp'], label_vs_gt[k]['fp']
        if tp + fp == 0:
            continue
        L.append(f"      {en:<12}{tp:>9}{fp:>11}{tp/(tp+fp+1e-9):>12.1%}")
    L.append("=" * 70)
    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'gt_eval.txt'), 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n[输出] {os.path.join(args.out, 'gt_eval.txt')}")


if __name__ == "__main__":
    main()
