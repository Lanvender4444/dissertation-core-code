"""Shared read-only loaders for supplementary thesis experiments (2026-11-01).
No detector inference, no GT edits, no changes to frozen experiments."""
import sys, json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'scripts'), str(ROOT/'ultralytics/research'),
                str(ROOT/'ultralytics/core_algorithms'), str(ROOT/'ultralytics/decided')]
if os.name != 'nt':
    import winpath_shim  # noqa: F401
import numpy as np
from scipy.optimize import linear_sum_assignment
import experiment_road_support_admission as E
from validate_multiview import truth, evaluate, ev
from protected_conditional_inputs import ProtectedInputs

BASE = ROOT/'ultralytics'
F5P_LOG = BASE/'experiments/five_camera_viewer_20261012/run_1791807403777670200/objects.jsonl'
REPAIR = BASE/'experiments/five_camera_repair_recovery_20261017/run01'
OUT_ROOT = BASE/'experiments/supp_20261101'
CAMS = (1, 2, 3, 4, 5)
MIN_H = 25
GROUP = {'person': 'person', 'rider': 'two-wheeler', 'bicycle': 'two-wheeler', 'motorcycle': 'two-wheeler',
         'car': 'vehicle', 'bus': 'vehicle', 'truck': 'vehicle'}
YGROUP = {0: 'person', 1: 'two-wheeler', 3: 'two-wheeler', 2: 'vehicle', 5: 'vehicle', 7: 'vehicle'}


def segments():
    p = json.loads((REPAIR/'protocol.json').read_text())
    seg = {k: list(p[k]) for k in ('train', 'development', 'guard', 'late')}
    seg['full'] = list(p['frames']); seg['guard_late'] = seg['guard'] + seg['late']
    return seg


def gt(frame, cams=CAMS):
    return truth(frame, cams)


_INPUTS = None
def inputs():
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = ProtectedInputs()
    return _INPUTS


def raw_pool(frame, corrected=False):
    pool, changed, descriptor, flags, crops, _ = inputs().load(frame)
    out = {}
    for u, d in (changed if corrected else pool).items():
        d = dict(d); d['uid'] = u
        if corrected:
            for k in ('ray', 'geometry_valid', 'angular_h', 'feature'):
                if k in pool[u] and k not in d: d[k] = pool[u][k]
        out[u] = d
    return out


def load_f5p():
    rows = {}
    with F5P_LOG.open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            objs = [dict(dets=o['members'], src=sorted(o['sources']), cls=o['cls'],
                         conf=max(m['conf'] for m in o['members']), recovered=o['recovered'])
                    for o in r['objects']]
            rows[r['frame']] = dict(objects=objs, counts=r.get('counts'))
    return rows


def f5r_tau():
    lock = json.loads((REPAIR/'development_lock.json').read_text())
    return lock['selection']['residual/P0.91']['tau']


def f5r_record(frame):
    return json.loads((REPAIR/f'evaluation_full/{frame}.json').read_text())


def f5r_groups(rec, tau):
    uids, cummin, _raw = rec['paths']['control/residual']
    root = [list(g) for g in rec['root']]
    return root, [[u] for u, z in zip(uids, cummin) if z >= tau]


def as_objects(groups, dets_by_uid, recovered_from=None):
    out = []
    for k, g in enumerate(groups):
        ds = [dets_by_uid[u] for u in g]
        best = max(ds, key=lambda d: d['conf'])
        out.append(dict(dets=ds, src=sorted({d['src'] for d in ds}), cls=best['cls'],
                        conf=best['conf'], recovered=recovered_from is not None and k >= recovered_from, uids=list(g)))
    return out


def f5r_objects(frame, tau=None, changed=None):
    tau = f5r_tau() if tau is None else tau
    rec = f5r_record(frame); root, add = f5r_groups(rec, tau)
    changed = raw_pool(frame, corrected=True) if changed is None else changed
    return as_objects(root + add, changed, recovered_from=len(root)), rec['counts']


def evaluate_at(objects, gt_by_cam, thr=0.5, min_h=MIN_H, return_assignment=False):
    union = ev.build_union(gt_by_cam, min_h); ignored = ev.build_union(gt_by_cam, 0)
    keys = list(union); S = np.zeros((len(objects), len(keys))); ign = set()
    for i, o in enumerate(objects):
        for d in o['dets']:
            for j, iid in enumerate(keys):
                u = union[iid]
                if d['src'] in u['boxes'] and ev.cls_match(u['label'], d['cls']):
                    S[i, j] = max(S[i, j], ev.iou(d['box'], u['boxes'][d['src']]))
        if not keys or S[i].max() < thr:
            for iid, u in ignored.items():
                if iid in union: continue
                if any(d['src'] in u['boxes'] and ev.cls_match(u['label'], d['cls'])
                       and ev.iou(d['box'], u['boxes'][d['src']]) >= thr for d in o['dets']):
                    ign.add(i); break
    reward = (S >= thr)*1000 + S*(S >= thr)
    rr, cc = linear_sum_assignment(-reward) if S.size else ([], [])
    pairs = [(int(r), keys[c]) for r, c in zip(rr, cc) if S[r, c] >= thr]
    tp = len(pairs); fp = len(objects) - len(ign) - tp; fn = len(keys) - tp
    if return_assignment:
        return np.array([tp, fp, fn]), pairs, union, ign
    return np.array([tp, fp, fn])


def prf(c):
    t, p, f = (float(x) for x in c[:3])
    P = t/max(t+p, 1); R = t/max(t+f, 1)
    return dict(tp=int(t), fp=int(p), fn=int(f), precision=P, recall=R, f1=2*P*R/max(P+R, 1e-12))


def block_bootstrap(counts_a, counts_b, block=25, reps=2000, seed=20261101):
    a = np.asarray(counts_a, float); b = np.asarray(counts_b, float)
    n = len(a); starts = np.arange(0, n, block); rng = np.random.default_rng(seed)
    def m(c):
        t, p, f = c[:, :3].sum(0); P = t/max(t+p, 1); R = t/max(t+f, 1)
        return np.array([P, R, 2*P*R/max(P+R, 1e-12)])
    diffs = np.array([m(b[idx]) - m(a[idx]) for idx in
                      (np.concatenate([np.arange(s, min(s+block, n)) for s in rng.choice(starts, len(starts))])
                       for _ in range(reps))])
    return dict(point=(m(b)-m(a)).tolist(), lo=np.percentile(diffs, 2.5, 0).tolist(),
                hi=np.percentile(diffs, 97.5, 0).tolist(), block=block, reps=reps, seed=seed)


def new_run(name, protocol, exist_ok=False):
    out = OUT_ROOT/name/'run01'; out.mkdir(parents=True, exist_ok=exist_ok)
    if not (out/'protocol.json').exists():
        (out/'protocol.json').write_text(json.dumps(protocol, indent=1))
    return out


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
