"""E7: learned post-detection verifier WITHOUT scene context (reviewer request).

Same box-corrected five-camera pool (confidence floor 0.05), same early training frames (601),
same histogram-gradient-boosting hyper-parameters as the recovery heads. Each detection is
scored independently; duplicates across cameras are merged by ray NMS ordered by the learned
score. Threshold and NMS angle are locked on the 200 development frames, then replayed on the
full sequence and on guard+late. Two feature sets:
  L1  detector/box/class/geometry evidence of the detection itself
  L2  L1 + local multi-view agreement (compatible detections in other cameras near the same ray)
Neither variant sees the accepted scene: there is no reference, no repair and no recovery context.
Also: block-length sensitivity of the paired bootstrap (F5-R vs F5-P, F5-R vs L*).

usage (from scripts/supp/): python3 e7_learned_baseline.py [SUPP_ROOT [OUT_DIR]]
  SUPP_ROOT defaults to ../../ultralytics/experiments/supp_20261101 (needs extract/run01/frames from extract.py)
  OUT_DIR   defaults to SUPP_ROOT/e7_learned_baseline/run01"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import supp_lib as L

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / 'ultralytics/experiments/supp_20261101'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / 'e7_learned_baseline/run01'
OUT.mkdir(parents=True, exist_ok=True)
L.EXTRACT = DATA / 'extract/run01/frames'
FRAMES = sorted(p.stem for p in L.EXTRACT.glob('*.json'))
def seg_of(s):
    i = int(s)
    return 'train' if i <= 600 else 'development' if i <= 800 else 'guard' if i <= 1099 else 'late'
SEG = {k: [s for s in FRAMES if seg_of(s) == k] for k in ('train', 'development', 'guard', 'late')}
SEG['guard_late'] = SEG['guard'] + SEG['late']; SEG['full'] = FRAMES
PARAMS = dict(max_iter=180, learning_rate=0.05, max_depth=5, max_leaf_nodes=24, min_samples_leaf=100,
              l2_regularization=10.0, max_bins=127, early_stopping=False, random_state=0)
COMPAT = {0: 'p', 1: 'b', 3: 'b', 2: 'v', 5: 'v', 7: 'v'}
GROUPS = ['p', 'b', 'v']
W, H = 2464.0, 2056.0
THETAS = [1.0, 2.0, 3.0, 5.0]
TAUS = [round(x, 3) for x in np.arange(0.02, 0.981, 0.01)]


def angle(a, b):
    ra, rb = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12), -1, 1))))


def own_features(d):
    x1, y1, x2, y2 = d['cbox']; w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cam = [float(d['src'] == c) for c in (1, 2, 3, 4, 5)]
    grp = [float(COMPAT.get(d['ccls']) == g) for g in GROUPS]
    crop = d.get('crop') or [0, 0, 0, 0]; flags = [float(v) for v in (d.get('flags') or [0, 0, 0, 0])]
    ray = d.get('ray') or [0.0, 0.0, 0.0]
    return cam + grp + [d['cconf'], d['conf'], float(d['ccls']), x1 / W, y1 / H, x2 / W, y2 / H, w / W, h / H,
                        (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, np.log(w * h), w / h,
                        crop[0] / W, crop[1] / H, (crop[2] - crop[0]) / W if crop[2] else 0., (crop[3] - crop[1]) / H if crop[3] else 0.,
                        *flags, float(d.get('angular_h') or 0.), float(bool(d.get('geometry_valid'))), *map(float, ray)]


def agreement_features(d, dets):
    """Local multi-view agreement only: other-camera compatible detections near the same ray."""
    if not d.get('ray'):
        return [0.] * 9
    near = {1.0: [], 3.0: [], 5.0: []}
    for e in dets:
        if e['src'] == d['src'] or COMPAT.get(e['ccls']) != COMPAT.get(d['ccls']) or not e.get('ray'):
            continue
        a = angle(d['ray'], e['ray'])
        for th in near:
            if a <= th:
                near[th].append(e)
    out = []
    for th in (1.0, 3.0, 5.0):
        es = near[th]
        out += [float(len({e['src'] for e in es})), max([e['cconf'] for e in es], default=0.), float(np.mean([e['cconf'] for e in es])) if es else 0.]
    return out


def labels(F):
    """Detection is positive if it matches an eligible union GT box in its own camera (compatible class, IoU>=0.5)."""
    union = L.build_union(F['gt'], L.MIN_H)
    y = {}
    for u, d in F['dets'].items():
        ok = False
        for iid, g in union.items():
            b = g['boxes'].get(d['src'])
            if b is not None and L.cls_match(g['label'], d['ccls']) and L.iou(d['cbox'], b) >= 0.5:
                ok = True; break
        y[u] = int(ok)
    return y


def frame_arrays(s):
    F = L.frame(s); dets = list(F['dets'].values()); uids = [d['uid'] for d in dets]
    x1 = np.array([own_features(d) for d in dets], float).reshape(len(dets), N_OWN)
    x2 = np.c_[x1, np.array([agreement_features(d, dets) for d in dets], float).reshape(len(dets), 9)]
    return F, uids, x1, x2


N_OWN = 34


def ray_nms_scored(F, score, theta, tau):
    ds = sorted([F['dets'][u] for u in score if score[u] >= tau], key=lambda d: (-score[d['uid']], d['uid']))
    used = set(); groups = []
    for i, d in enumerate(ds):
        if i in used:
            continue
        used.add(i); members = {d['src']: d}
        for j in range(i + 1, len(ds)):
            e = ds[j]
            if j in used or COMPAT.get(e['ccls']) != COMPAT.get(d['ccls']):
                continue
            if e['src'] == d['src']:
                if L.iou(d['cbox'], e['cbox']) >= .5:
                    used.add(j)
            elif e['src'] not in members and d.get('ray') and e.get('ray') and angle(d['ray'], e['ray']) <= theta:
                members[e['src']] = e; used.add(j)
        groups.append([m['uid'] for m in members.values()])
    return groups


t0 = time.time()
print({k: len(v) for k, v in SEG.items()}, flush=True)
cache = {s: frame_arrays(s) for s in FRAMES}
print(f'features built for {len(FRAMES)} frames in {time.time() - t0:.0f}s', flush=True)
models = {}
for name, k in (('L1', 2), ('L2', 3)):
    X = np.vstack([cache[s][k] for s in SEG['train']])
    Y = np.concatenate([[labels(cache[s][0])[u] for u in cache[s][1]] for s in SEG['train']])
    models[name] = HistGradientBoostingClassifier(**PARAMS).fit(X, Y)
    print(name, 'train examples', len(Y), 'positives', int(Y.sum()), 'features', X.shape[1], flush=True)
scores = {name: {s: (dict(zip(cache[s][1], models[name].predict_proba(cache[s][k])[:, 1])) if len(cache[s][1]) else {}) for s in FRAMES}
          for name, k in (('L1', 2), ('L2', 3))}

def sweep(name, frames):
    tot = {}
    for s in frames:
        F = cache[s][0]
        for th in THETAS:
            for t in TAUS:
                c = L.evaluate(L.as_objects(ray_nms_scored(F, scores[name][s], th, t), F), F['gt'])
                tot[(th, t)] = tot.get((th, t), np.zeros(3, int)) + c
    return tot

def per_frame(name, th, t, frames):
    return [L.evaluate(L.as_objects(ray_nms_scored(cache[s][0], scores[name][s], th, t), cache[s][0]), cache[s][0]['gt']).tolist() for s in frames]

f5r = {s: cache[s][0]['f5r_counts'] for s in FRAMES}; f5p = {s: cache[s][0]['f5p_counts'] for s in FRAMES}
res = dict(protocol=dict(params=PARAMS, thetas=THETAS, taus=[TAUS[0], TAUS[-1], 0.01], segments={k: len(v) for k, v in SEG.items()},
                         label='detection matches eligible union GT in own camera (compatible class, IoU>=0.5)',
                         selection='development only: max recall s.t. P>=target (0.90, 0.91); full/guard+late replay after lock',
                         no_scene_context=True), locks={}, results={}, bootstrap={}, block_sensitivity={})
for name in ('L1', 'L2'):
    dev = sweep(name, SEG['development'])
    for target in (0.90, 0.91):
        feas = [(p, L.prf(c)) for p, c in dev.items() if L.prf(c)['precision'] >= target]
        if not feas:
            res['locks'][f'{name}/P{target}'] = None; continue
        (th, t), m = max(feas, key=lambda x: (x[1]['recall'], x[1]['precision']))
        key = f'{name}/P{target}'
        res['locks'][key] = dict(theta=th, tau=t, dev=m)
        rows = per_frame(name, th, t, FRAMES); row = dict(zip(FRAMES, rows))
        res['results'][key] = {d: L.prf(np.sum([row[s] for s in SEG[d]], axis=0)) for d in ('full', 'train', 'development', 'guard', 'late', 'guard_late')}
        for d in ('full', 'guard_late'):
            res['bootstrap'][f'F5R-minus-{key}/{d}'] = L.block_bootstrap([row[s] for s in SEG[d]], [f5r[s] for s in SEG[d]])
        print(key, res['locks'][key]['theta'], res['locks'][key]['tau'], {d: (round(v['precision'] * 100, 2), round(v['recall'] * 100, 2)) for d, v in res['results'][key].items() if d in ('full', 'guard_late')}, flush=True)
for ref in ('F5R', 'F5P'):
    res['results'][ref] = {d: L.prf(np.sum([(f5r if ref == 'F5R' else f5p)[s] for s in SEG[d]], axis=0)) for d in ('full', 'train', 'development', 'guard', 'late', 'guard_late')}
for b in (10, 25, 50, 100):
    res['block_sensitivity'][f'F5R-minus-F5P/guard_late/block{b}'] = L.block_bootstrap([f5p[s] for s in SEG['guard_late']], [f5r[s] for s in SEG['guard_late']], block=b)
    res['block_sensitivity'][f'F5R-minus-F5P/full/block{b}'] = L.block_bootstrap([f5p[s] for s in SEG['full']], [f5r[s] for s in SEG['full']], block=b)
L.dump(OUT / 'results.json', res)
print('done in', round(time.time() - t0), 's', flush=True)
