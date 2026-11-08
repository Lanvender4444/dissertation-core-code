"""YOLO-aware three-camera fusion (v2).  GT is never read by this module.

Pipeline (see algorithm/YOLO_AWARE_FUSION.md):
  1. detections from yolo_hr_detect (1920 full frame + 2x2 tiles @1600)
  2. same bottom-large-vehicle ego gate as final_pipeline (applied to singles too)
  3. per-camera vehicle-group NMS: YOLO's class-aware NMS keeps car+truck boxes
     on one vehicle; car/bus/truck are merged before association
  4. association: unchanged tri-compare (trinocular -> binocular -> residual),
     fed with ALL detections >= 0.10 so weak-but-consistent views can join
  5. evidence scoring: YOLO scores are per-view sigmoid class scores that fall
     with object size, truncation and duplication.  A logistic evidence model
     combines the per-view logits, the weakest view, support count, box scale,
     border truncation and "should-have-been-seen" negative evidence into one
     probability.  Weights are fitted on DEVELOPMENT frames only and frozen in
     algorithm/yolo_aware_config.json
  6. admission p >= tau, then existing complete-link multiview dedup
  7. residual suppression: a single-view object whose box overlaps a member of
     an admitted multi-view object in the same camera (compatible class) is a
     leftover duplicate of that object and is dropped.
"""
import json
from pathlib import Path
import numpy as np
from final_pipeline import filter_ego, core
from legacy_adapter import legacy
from multiview_dedup import deduplicate, box_iou
from multiview_fusion import compatible
from yolo_hr_detect import merge_nms

CONFIG = Path(__file__).resolve().parents[1] / 'algorithm/yolo_aware_config.json'
W, H = 2464, 2056


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(np.log(p / (1 - p)))


def vehicle_nms(dets, iou_thr=0.7):
    out = []
    for c in sorted({d['src'] for d in dets}):
        out += merge_nms([d for d in dets if d['src'] == c], iou_thr=iou_thr, ios_thr=2.0)
    return out


FEATURES = ['bias', 'views1', 'views2', 'views3', 'logit_max', 'logit_min', 'logit_mean',
            'log_height', 'person', 'truncated', 'expected_but_missing', 'dup_overlap']


def features(o, all_dets):
    cs = [d['conf'] for d in o['dets']]; n = len(cs)
    hs = [d['box'][3] - d['box'][1] for d in o['dets']]
    trunc = any(d['box'][0] < 5 or d['box'][2] > W - 5 or d['box'][3] > H - 5 for d in o['dets'])
    # overlap with a DIFFERENT, stronger same-camera detection: NMS leftover signature
    dup = 0.
    for d in o['dets']:
        for e in all_dets:
            if e is d or e['src'] != d['src'] or e['conf'] <= d['conf'] or not compatible(e['cls'], d['cls']): continue
            dup = max(dup, box_iou(e['box'], d['box']))
    # legacy 'FP' label = single box inside the common FOV where other cams saw nothing;
    # 'HALF' with missing != None = the third camera should have seen it
    missing = 1. if (n == 1 and o['label'] == 'FP') else 0.5 if (n == 2 and o.get('missing')) else 0.
    return np.array([1, n == 1, n == 2, n >= 3, logit(max(cs)), logit(min(cs)), np.mean([logit(c) for c in cs]),
                     np.log(max(np.median(hs), 1)), o['cls'] == 0 if 'cls' in o else
                     max(o['dets'], key=lambda d: d['conf'])['cls'] == 0, trunc, missing, dup], float)


def fit_logistic(X, y, l2=1.0, iters=50):
    w = np.zeros(X.shape[1]); R = l2 * np.eye(X.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + R @ w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + R + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(Hm, g); w -= step
        if np.abs(step).max() < 1e-8: break
    return w


def associate(dets, cameras, rect=None, ego=True):
    active = [d for d in dets if d['src'] in (2, 3, 4)]
    kept, removed = filter_ego(active, cameras, ego)
    kept = vehicle_nms(kept)
    if rect is None: rect = core.TriRectifier(core.camera_params)
    _, raw = legacy(kept, rect)
    for o in raw:
        o['cls'] = max(o['dets'], key=lambda d: d['conf'])['cls']
    return raw, kept, removed


def suppress_residuals(objects, iou_thr):
    multi = [o for o in objects if len(o['src']) > 1]; out = []
    for o in objects:
        if len(o['src']) == 1:
            d = o['dets'][0]
            if any(m['src'] == d['src'] and compatible(m['cls'], d['cls']) and box_iou(m['box'], d['box']) >= iou_thr
                   for k in multi for m in k['dets']):
                continue
        out.append(o)
    return out


def score_and_admit(raw, kept, cfg):
    w = np.asarray(cfg['weights']); objs = []
    for o in raw:
        if np.median([d['box'][3] - d['box'][1] for d in o['dets']]) < cfg['min_height']: continue
        p = float(1 / (1 + np.exp(-features(o, kept) @ w)))
        if p >= cfg['tau']:
            o = dict(o); o['conf'] = p; o['yolo_max_conf'] = max(d['conf'] for d in o['dets']); objs.append(o)
    objs = deduplicate(objs, iou_threshold=cfg['dedup_iou'], min_shared=2)
    return suppress_residuals(objs, cfg['residual_iou'])


def run_pipeline_v2(dets, cameras, rect=None, ego=True, cfg=None):
    cfg = cfg or json.loads(CONFIG.read_text())
    raw, kept, removed = associate(dets, cameras, rect, ego)
    objects = score_and_admit(raw, kept, cfg)
    for i, o in enumerate(objects):
        o['id'] = i + 1
        o['label'] = 'FULL' if len(o['src']) == 3 else 'PART' if len(o['src']) == 2 else 'UNVERIFIED'
        o['unmatched_cameras'] = [c for c in (2, 3, 4) if c not in o['src']]
    return objects, removed
