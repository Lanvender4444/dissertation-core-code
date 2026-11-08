"""E3: evidence-type ablation of the five-camera residual recovery head (575 features).

Repair stage frozen (repaired roots from the audited composition). Only the residual head is
re-fitted, on the identical 601-frame early training samples/targets/parameters, with one group
of feature columns held at zero in fitting and inference (equivalent to removal for trees).
Development-only threshold lock: maximise recall subject to precision>=0.91 on the exact score
curve down to floor 0.3 (the rule that produced the retained tau=0.5559). Then full replay.
Usage: python3 e3_feature_ablation.py verify|fit ARM|dev ARM|full ARM|report
"""
import sys, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'ultralytics/research'), str(ROOT/'ultralytics/core_algorithms'), str(ROOT/'scripts')]
import five_camera_capacity_recovery as S
import marginal_capacity as M
import nonlinear_admission as N
from supp_lib import segments, frame, block_bootstrap, prf, dump, OUT_ROOT

RECON = ROOT/'ultralytics/experiments/innovation_convergence_20261016/reconditioned01'
CAP = ROOT/'ultralytics/experiments/five_camera_capacity_recovery_20261017/run01'
CFG = json.loads((CAP/'protocol.json').read_text())
PARAMS = CFG['params']
FLOOR = 0.3
TARGET = 0.91
D = 575

GROUPS = {
    'none': [],
    'no_appearance': list(range(91, 475)) + list(range(475, 523)),
    'no_geometry': list(range(23, 28)),
    'no_crossview': list(range(70, 83)) + list(range(529, 575)),
    'no_camera_id': list(range(8, 13)) + list(range(59, 64)),
    'detector_only': sorted(set(range(D)) - set(list(range(0, 8)) + list(range(13, 23)) + [56, 57] + list(range(64, 70)) + list(range(83, 91)))),
}
# detector_only lists the columns REMOVED: everything except detector/box/class/crop-flag/box-calibration columns


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load(s, kind):
    base = arrays(RECON/f'prepared/{kind}/{s}.npz'); lab = arrays(RECON/f'labels/{kind}/{s}.npz')
    meta = json.loads((RECON/f'prepared/{kind}/{s}.json').read_text())
    aux = arrays(CAP/f'prepared/{kind}/{s}.npz')
    return S.attach(base, **aux), lab, meta


def scene_edges(scene, data, lab):
    lookup = {int(u): i for i, u in enumerate(data['uids'])}
    return np.asarray([np.maximum.reduce([lab['edges'][lookup[u]] for u in g]) for g in scene], bool).reshape(len(scene), lab['edges'].shape[1])


def counts(scene, data, lab):
    lookup = {int(u): i for i, u in enumerate(data['uids'])}; edges = scene_edges(scene, data, lab)
    rr, cc = linear_sum_assignment(-edges.astype(int)) if edges.size else ([], [])
    tp = int(edges[rr, cc].sum()) if edges.size else 0
    ignored = sum(not row.any() and any(lab['small'][lookup[u]] for u in g) for g, row in zip(scene, edges))
    return [tp, len(scene)-ignored-tp, edges.shape[1]-tp]


def mask(x, arm):
    x = np.array(x, float, copy=True)
    if GROUPS[arm]:
        x[:, GROUPS[arm]] = 0.
    return x


def out(arm):
    p = OUT_ROOT/'e3_feature_ablation'/arm/'run01'; p.mkdir(parents=True, exist_ok=True); return p


def frozen_residual():
    doc = json.loads((CAP/'models.json').read_text())
    return N.compile_model(doc['models']['residual'])


def training_set():
    cache = OUT_ROOT/'e3_feature_ablation'/'training_samples.npz'
    if cache.exists():
        z = arrays(cache); return z['x'], z['y']
    seg = segments(); xs = []; ys = []; t = time.time()
    for n, s in enumerate(seg['train'], 1):
        d, lab, m = load(s, 'train'); root = m['scenes']['control']
        idx, _ = S.features(d, root, 'capacity')
        ranking = sorted(idx.tolist(), key=lambda i: (-d['conf'][i], int(d['uids'][i]))); seen = set()
        for length in CFG['prefix_lengths']:
            scene = root + [[int(d['uids'][i])] for i in ranking[:length]]; ident = tuple(tuple(g) for g in scene)
            if ident in seen:
                continue
            seen.add(ident); idx, x = S.features(d, scene, 'residual'); e = lab['edges'][idx]
            y = M.marginal_labels(scene_edges(scene, d, lab), e)['labels']; keep = ~(~e.any(axis=1) & lab['small'][idx])
            xs.append(x[keep]); ys.append(y[keep])
        if n % 100 == 0:
            print('train', n, round(time.time()-t), flush=True)
    x = np.concatenate(xs).astype(float); y = np.concatenate(ys).astype(int)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def fit(arm):
    x, y = training_set()
    doc = N.fit(mask(x, arm), y, PARAMS)
    dump(out(arm)/'model.json', dict(arm=arm, removed_columns=GROUPS[arm], **doc))
    print(arm, 'fitted', doc['samples'], doc['positive'], flush=True)
    return doc


def predictor(arm):
    if arm == 'frozen':
        model = frozen_residual()
        return lambda x: N.predict(x, model)
    model = N.compile_model(json.loads((out(arm)/'model.json').read_text()))
    return lambda x: N.predict(mask(x, arm), model)


def rollout_frame(s, pred, floor):
    d, lab, _ = load(s, 'replay'); F = frame(s); root = [list(g) for g in F['f5r_root']]
    added, cummin, raw = S.rollout(d, root, 'residual', pred, floor)
    prefix = [counts(root+[[u] for u in added[:k]], d, lab) for k in range(len(added)+1)]
    return dict(frame=s, added=[int(u) for u in added], cummin=[float(z) for z in cummin], prefix_counts=prefix)


def run_domain(arm, domain, floor):
    seg = segments(); frames = seg['development'] if domain == 'dev' else seg['full']
    path = out(arm)/f'rollout_{domain}.jsonl'; done = set()
    if path.exists():
        done = {json.loads(l)['frame'] for l in path.read_text().splitlines() if l.strip()}
    pred = predictor(arm); t = time.time()
    with path.open('a') as f:
        for n, s in enumerate(frames, 1):
            if s in done:
                continue
            f.write(json.dumps(rollout_frame(s, pred, floor))+'\n'); f.flush()
            if n % 100 == 0:
                print(arm, domain, n, round(time.time()-t), flush=True)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def at_tau(rows, tau):
    tot = np.zeros(3, int); per = []
    for r in rows:
        k = sum(1 for z in r['cummin'] if z >= tau)
        c = np.array(r['prefix_counts'][k]); tot += c; per.append(c.tolist())
    return tot, per


def lock(arm):
    rows = run_domain(arm, 'dev', FLOOR)
    taus = sorted({z for r in rows for z in r['cummin']} | {1.0000001}, reverse=True)
    best = None; curve = []
    for tau in taus:
        tot, _ = at_tau(rows, tau); m = prf(tot); curve.append(dict(tau=tau, **m))
        if m['precision'] >= TARGET and (best is None or (m['recall'], m['precision']) > (best['recall'], best['precision'])):
            best = dict(tau=tau, **m)
    dump(out(arm)/'development_lock.json', dict(arm=arm, target_precision=TARGET, selection=best, curve=curve[::5]))
    print(arm, 'lock', best, flush=True)
    return best


def full(arm):
    best = json.loads((out(arm)/'development_lock.json').read_text())['selection']
    rows = run_domain(arm, 'full', min(best['tau'], 1.0) if best else 1.0)
    rows = {r['frame']: r for r in rows}; seg = segments(); res = {}
    for name in ('full', 'guard_late', 'development'):
        rr = [rows[s] for s in seg[name]]
        tot, per = at_tau(rr, best['tau'] if best else 1.0000001)
        res[name] = dict(**prf(tot), per_frame=per)
    dump(out(arm)/'results.json', dict(arm=arm, tau=best['tau'] if best else None, **{k: {a: b for a, b in v.items() if a != 'per_frame'} for k, v in res.items()}))
    dump(out(arm)/'per_frame.json', {k: v['per_frame'] for k, v in res.items()})
    print(arm, {k: (round(v['precision'], 4), round(v['recall'], 4)) for k, v in res.items()}, flush=True)


def verify(n=40):
    """Frozen model + stored repaired root must reproduce the audited GT-free paths and counts."""
    seg = segments(); pred = predictor('frozen'); tau = 0.5559089514906035; bad = 0
    floor = 0.4712897576004438  # min locked tau used in the audited full replay
    for s in seg['full'][::max(1, len(seg['full'])//n)]:
        F = frame(s); r = rollout_frame(s, pred, floor)
        uids, cummin, _ = F['f5r_paths']['control/residual']
        k = sum(1 for z in r['cummin'] if z >= tau)
        same = r['added'] == [int(u) for u in uids] and np.allclose(r['cummin'], cummin)
        cnt = r['prefix_counts'][k] == F['f5r_counts']
        bad += not (same and cnt)
        if not (same and cnt):
            print('MISMATCH', s, same, cnt, r['prefix_counts'][k], F['f5r_counts'], flush=True)
    print('verify mismatches', bad, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'verify':
        verify(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == 'train':
        x, y = training_set(); print(x.shape, int(y.sum()))
    elif cmd == 'fit':
        fit(sys.argv[2])
    elif cmd == 'dev':
        lock(sys.argv[2])
    elif cmd == 'full':
        full(sys.argv[2])
    elif cmd == 'all':
        arm = sys.argv[2]; fit(arm); lock(arm); full(arm)
