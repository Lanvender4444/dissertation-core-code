"""Self-contained analysis library for the supplementary experiments (2026-11-01).

Operates only on the compact frozen snapshot written by scripts/supp/extract.py
(ultralytics/experiments/supp_20261101/extract/run01/frames/*.json).  The matching
protocol replicates ultralytics/core_algorithms/validate_multiview.evaluate exactly:
compatible-class any-member IoU>=0.5, GT union over cameras with max box height >=25 px,
small-GT-only predictions ignored, cardinality-first linear assignment.
"""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = ROOT/'ultralytics/experiments/supp_20261101/extract/run01/frames'
REPAIR = ROOT/'ultralytics/experiments/five_camera_repair_recovery_20261017/run01'
OUT_ROOT = ROOT/'ultralytics/experiments/supp_20261101'
CAMS = (1, 2, 3, 4, 5)
MIN_H = 25

# identical to tri-eval-gt.py GT_COMPAT
GT_COMPAT = {'car': {2, 5, 7}, 'bus': {5, 7, 2}, 'truck': {7, 5, 2}, 'person': {0},
             'rider': {0, 1, 3}, 'bicycle': {1, 3}, 'motorcycle': {3}}
GROUP = {'person': 'person', 'rider': 'two-wheeler', 'bicycle': 'two-wheeler', 'motorcycle': 'two-wheeler',
         'car': 'vehicle', 'bus': 'vehicle', 'truck': 'vehicle'}
YGROUP = {0: 'person', 1: 'two-wheeler', 3: 'two-wheeler', 2: 'vehicle', 5: 'vehicle', 7: 'vehicle'}
H_BINS = [(25, 50), (50, 100), (100, 200), (200, 1e9)]
H_LABELS = ['25-50', '50-100', '100-200', '>=200']


def cls_match(label, ycls):
    return ycls in GT_COMPAT.get(label, set())


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1)*max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/(ua+1e-9)


def segments():
    p = json.loads((REPAIR/'protocol.json').read_text())
    seg = {k: list(p[k]) for k in ('train', 'development', 'guard', 'late')}
    seg['full'] = list(p['frames']); seg['guard_late'] = seg['guard'] + seg['late']
    return seg


@lru_cache(maxsize=None)
def frame(s):
    d = json.loads((EXTRACT/f'{s}.json').read_text())
    d['gt'] = {int(c): [(int(i), l, b) for i, l, b in v] for c, v in d['gt'].items()}
    d['dets'] = {int(u): v for u, v in d['dets'].items()}
    for u, v in d['dets'].items():
        v['uid'] = u
    return d


def det(d, corrected=True):
    """Detection view used by evaluation; corrected=True uses box-calibrated box/class/conf."""
    if corrected:
        return dict(uid=d['uid'], src=d['src'], cls=d['ccls'], conf=d['cconf'], box=d['cbox'])
    return dict(uid=d['uid'], src=d['src'], cls=d['cls'], conf=d['conf'], box=d['box'])


def as_objects(groups, F, corrected=True, recovered_from=None):
    out = []
    for k, g in enumerate(groups):
        ds = [det(F['dets'][u], corrected) for u in g]
        best = max(ds, key=lambda x: x['conf'])
        out.append(dict(dets=ds, src=sorted({x['src'] for x in ds}), cls=best['cls'], conf=best['conf'],
                        uids=list(g), recovered=recovered_from is not None and k >= recovered_from))
    return out


def f5r_tau():
    return json.loads((REPAIR/'development_lock.json').read_text())['selection']['residual/P0.91']['tau']


def configs(s, tau=None):
    """Frozen outputs of one frame: raw nodes, A, F5-P, repaired root, F5-R."""
    F = frame(s); tau = f5r_tau() if tau is None else tau
    p_groups, rec = F['f5p_groups'], F['f5p_recovered']
    A = [g for g, r in zip(p_groups, rec) if not r]
    uids, cummin, _ = F['f5r_paths']['control/residual']
    root = [list(g) for g in F['f5r_root']]
    add = [[u] for u, z in zip(uids, cummin) if z >= tau]
    return dict(raw=[[u] for u in F['dets']], A=A, F5P=p_groups, repair=root, F5R=root+add,
                _A_n=len(A), _root_n=len(root))


def build_union(gt_by_cam, min_h):
    union = {}
    for c, lst in gt_by_cam.items():
        for iid, lab, box in lst:
            u = union.setdefault(iid, {'label': lab, 'boxes': {}})
            u['boxes'][c] = box
    return {iid: u for iid, u in union.items() if max(b[3]-b[1] for b in u['boxes'].values()) >= min_h}


def gt_subset(gt_by_cam, cams):
    return {c: v for c, v in gt_by_cam.items() if c in cams}


def evaluate(objects, gt_by_cam, thr=0.5, min_h=MIN_H, return_assignment=False):
    union = build_union(gt_by_cam, min_h); ignored = build_union(gt_by_cam, 0)
    keys = list(union); S = np.zeros((len(objects), len(keys))); ign = set()
    for i, o in enumerate(objects):
        for d in o['dets']:
            for j, iid in enumerate(keys):
                u = union[iid]
                if d['src'] in u['boxes'] and cls_match(u['label'], d['cls']):
                    S[i, j] = max(S[i, j], iou(d['box'], u['boxes'][d['src']]))
        if not keys or S[i].max() < thr:
            for iid, u in ignored.items():
                if iid in union:
                    continue
                if any(d['src'] in u['boxes'] and cls_match(u['label'], d['cls'])
                       and iou(d['box'], u['boxes'][d['src']]) >= thr for d in o['dets']):
                    ign.add(i); break
    reward = (S >= thr)*1000 + S*(S >= thr)
    rr, cc = linear_sum_assignment(-reward) if S.size else ([], [])
    pairs = [(int(r), keys[c]) for r, c in zip(rr, cc) if S[r, c] >= thr]
    tp = len(pairs); fp = len(objects)-len(ign)-tp; fn = len(keys)-tp
    if return_assignment:
        return np.array([tp, fp, fn]), pairs, union, ign
    return np.array([tp, fp, fn])


def prf(c):
    t, p, f = (float(x) for x in list(c)[:3])
    P = t/max(t+p, 1); R = t/max(t+f, 1)
    return dict(tp=int(t), fp=int(p), fn=int(f), precision=P, recall=R, f1=2*P*R/max(P+R, 1e-12))


def block_bootstrap(counts_a, counts_b, block=25, reps=2000, seed=20261101):
    a = np.asarray(counts_a, float); b = np.asarray(counts_b, float)
    n = len(a); starts = np.arange(0, n, block); rng = np.random.default_rng(seed)
    blocks = [np.arange(s, min(s+block, n)) for s in starts]
    def m(c):
        t, p, f = c[:, :3].sum(0); P = t/max(t+p, 1); R = t/max(t+f, 1)
        return np.array([P, R, 2*P*R/max(P+R, 1e-12)])
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        diffs.append(m(b[idx])-m(a[idx]))
    diffs = np.array(diffs)
    return dict(point=(m(b)-m(a)).tolist(), lo=np.percentile(diffs, 2.5, 0).tolist(),
                hi=np.percentile(diffs, 97.5, 0).tolist(), block=block, reps=reps, seed=seed,
                order=['precision', 'recall', 'f1'])


def new_run(name, protocol):
    out = OUT_ROOT/name/'run01'
    out.mkdir(parents=True, exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=1))
    return out


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def h_bin(h):
    for k, (a, b) in enumerate(H_BINS):
        if a <= h < b:
            return k
    return None
