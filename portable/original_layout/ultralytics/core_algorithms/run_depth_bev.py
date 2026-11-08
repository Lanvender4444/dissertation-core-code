"""Driver for the depth-BEV fusion experiments (ideas 1-7).  Resumable stages:

  cand  build candidate objects + features per frame -> experiments/depth_bev/<detset>/<cfg>/<stem>.pkl
  fit   fit the evidence model on DEVELOPMENT frames only (5-fold frame-blocked CV)
        -> algorithm/depth_bev_params/<detset>_<cfg>.json
  eval  apply frozen params to regression180 / regression60 -> results json

Development = development30 + 20 early frames (stems < 00001000); their
preceding frames (3 files earlier) are used only as temporal history.
"""
import argparse
import copy
import itertools
import json
import pickle
import time
import numpy as np
from multiview_geometry import ROOT, load_cameras
from validate_multiview import truth, ev, evaluate, metrics
from stereo_depth import StereoDepth
import depth_bev_fusion as F
from guided_redetect import ReDetector
import soft_tracking as T

BASE = ROOT / 'ultralytics'
DETSETS = {'960': BASE / 'experiments/detection_cache/0ebbc80d4a76_960_conf010',
           'HR': BASE / 'experiments/detection_cache/0ebbc80d4a76_1920_t2x2@1600_conf010',
           'BAND': BASE / 'experiments/detection_cache/0ebbc80d4a76_960_band850-1250@1600_conf010'}   # E4
VER = 'r3'
VER_OF = {'r': 'r3b'}   # 'r' redefined after a partial run; separate cache dir   # bump when candidate-generation code changes (old caches are kept, not deleted)
OUT = BASE / f'experiments/depth_bev_{VER}'
PARAMS = BASE / 'algorithm/depth_bev_params'
PROBE20 = ['00000004', '00000055', '00000108', '00000161', '00000213', '00000266', '00000319', '00000371',
           '00000424', '00000477', '00000528', '00000581', '00000634', '00000686', '00000739', '00000792',
           '00000843', '00000894', '00000945', '00000996']

# E3: 60 additional early-sequence development frames (stems < 00000990, disjoint from all test frames)
EXTRA_DEV60 = ['00000012', '00000027', '00000044', '00000061', '00000077', '00000094', '00000110', '00000127', '00000143', '00000159', '00000177', '00000192', '00000209', '00000226', '00000242', '00000259', '00000275', '00000292', '00000307', '00000325', '00000342', '00000357', '00000375', '00000391', '00000407', '00000425', '00000440', '00000457', '00000473', '00000490', '00000507', '00000522', '00000540', '00000556', '00000572', '00000590', '00000605', '00000622', '00000639', '00000655', '00000671', '00000689', '00000705', '00000721', '00000737', '00000754', '00000771', '00000787', '00000804', '00000820', '00000836', '00000853', '00000868', '00000884', '00000900', '00000916', '00000932', '00000948', '00000964', '00000980']

S = lambda **k: dict(F.DEFAULT, **k)
CONFIGS = {
    # name: (switches, parent used for redetect selection / temporal history)
    'base':      (S(), None),                                            # old grouping + evidence model
    'd':         (S(depth=True), None),                                  # +1
    'b':         (S(bev=True), None),                                    # only 2 (size-prior depth)
    'o':         (S(occl=True), None),                                   # only 3
    'm':         (S(milp=True), None),                                   # only 5
    'e':         (S(emb=True), None),                                    # only 6
    'r':         (S(redetect=True, redet_conf=.2, redet_iou=.35), 'base'),  # only 4 (dev-selected strict gate)
    't':         (S(track=True), 'base'),                                # only 7
    'dr':        (S(depth=True, redetect=True, redet_conf=.2, redet_iou=.35), 'd'),   # 1+4
    'db':        (S(depth=True, bev=True), None),                        # 1+2 (pure BEV grouping)
    'df':        (S(depth=True, bevf=True), None),                       # 1+2 (BEV field features on old groups)
    'dfo':       (S(depth=True, bevf=True, occl=True), None),                     # 1+2+3
    'dfor':      (S(depth=True, bevf=True, occl=True, redetect=True), 'dfo'),     # +4
    # ---- round 3
    'dfo_E1':    (S(depth=True, bevf=True, occl=True, dedup3d=True), None),
    'dr_E1':     (S(depth=True, redetect=True, redet_conf=.2, redet_iou=.35, dedup3d=True), 'd'),
    'd_E1':      (S(depth=True, dedup3d=True), None),
    'd_E13':     (S(depth=True, dedup3d=True, model='gbdt'), None),
    'dfo_E13':   (S(depth=True, bevf=True, occl=True, dedup3d=True, model='gbdt'), None),
    'd_E1x':     (S(depth=True, dedup3d=True, dev110=True), None),
    'd_E13x':    (S(depth=True, dedup3d=True, model='gbdt', dev110=True), None),
    'dfo_E1x':   (S(depth=True, bevf=True, occl=True, dedup3d=True, dev110=True), None),
    'dfo_E13x':  (S(depth=True, bevf=True, occl=True, dedup3d=True, model='gbdt', dev110=True), None),
    'dr_E1x':    (S(depth=True, redetect=True, redet_conf=.2, redet_iou=.35, dedup3d=True, dev110=True), 'd'),
    'dr_E13x':   (S(depth=True, redetect=True, redet_conf=.2, redet_iou=.35, dedup3d=True, model='gbdt', dev110=True), 'd'),
    'dr_E13':    (S(depth=True, redetect=True, redet_conf=.2, redet_iou=.35, dedup3d=True, model='gbdt'), 'd'),
    'dfor2':     (S(depth=True, bevf=True, occl=True, redetect=True, redet_conf=.2, redet_iou=.35), 'dfo'),
    'dmo':       (S(depth=True, milp=True, bevf=True, occl=True), None),          # 1+2+3+5 (global association)
    'dmor':      (S(depth=True, milp=True, bevf=True, occl=True, redetect=True), 'dmo'),
    'dfoe':      (S(depth=True, bevf=True, occl=True, emb=True), None),                            # dev-selected path
    'dfoe2':     (S(depth=True, bevf=True, occl=True, emb=True, emb_split=.7), None),
    'e2':        (S(emb=True, emb_split=.7), None),
    'dfoet':     (S(depth=True, bevf=True, occl=True, emb=True, track=True), 'dfoe'),
    'dfore':     (S(depth=True, bevf=True, occl=True, redetect=True, redet_conf=.2, redet_iou=.35, emb=True), 'dfo'),
    'dmore':     (S(depth=True, milp=True, bevf=True, occl=True, redetect=True, emb=True), 'dmo'),
    'dforet':    (S(depth=True, bevf=True, occl=True, redetect=True, redet_conf=.2, redet_iou=.35, emb=True, track=True), 'dfore'),
    'dmoret':    (S(depth=True, milp=True, bevf=True, occl=True, redetect=True, emb=True, track=True), 'dmore'),
}


def splits():
    src = json.loads((BASE / 'algorithm/evaluation_reports/final_verification_results.json').read_text())
    fresh = json.loads((BASE / 'experiments/staged_native_fresh_manifest.json').read_text())
    import os
    names = sorted(p.stem for p in (BASE / 'mycar/images/3').glob('*.png')); idx = {n: i for i, n in enumerate(names)}
    dev = PROBE20 + src['development_frames']
    prev = {d: names[idx[d] - 3] for d in dev if idx[d] >= 3}
    t180, t60 = src['test_frames'], fresh['test_frames']
    for seq in (t180, t60):
        for a, b in zip(seq[:-1], seq[1:]): prev[b] = a
    return dict(dev=dev, dev110=dev + EXTRA_DEV60, extra60=EXTRA_DEV60, devprev=sorted(set(prev[d] for d in dev if d in prev)), regression180=t180, regression60=t60), prev


class Frame:
    def __init__(self, stem, detset, stereo):
        self.stem = stem
        self.dets = []
        for c in (2, 3, 4):
            for k, d in enumerate(json.loads((DETSETS[detset] / f'{c}_{stem}.json').read_text())['dets']):
                d = dict(d); d['_k'] = k; d.pop('feature', None); self.dets.append(d)
        ep = BASE / f'experiments/embed_cache/{detset}/{stem}.npz'
        self.emb = None
        if ep.exists():
            z = np.load(ep); self.emb = {c: z[f'c{c}'] for c in (2, 3, 4)}
        self.maps = stereo.load_or_compute(stem, None)
        op = BASE / f'experiments/orb_cache/{stem}.npz'; self.orb = np.load(op) if op.exists() else None


def slim(objs):
    keep = ('src', 'box', 'cls', 'conf', 'uid', 'P', 'Z', 'sigZ', 'tb')
    return [dict(src=o['src'], cls=o['cls'], x=o['x'], dets=[{k: d[k] for k in keep} for d in o['dets']]) for o in objs]


def params_of(detset, name):
    p = PARAMS / f'{detset}_{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


CAND_OF = {'dfo_E1': 'dfo', 'dr_E1': 'dr', 'd_E1': 'd', 'd_E13': 'd', 'dfo_E13': 'dfo', 'dr_E13': 'dr', 'd_E1x': 'd', 'd_E13x': 'd', 'dfo_E1x': 'dfo', 'dfo_E13x': 'dfo', 'dr_E1x': 'dr', 'dr_E13x': 'dr'}   # post-processing-only variants reuse another config's candidates


def pkl(detset, name, stem):
    name = CAND_OF.get(name, name)
    if name in VER_OF: return BASE / f'experiments/depth_bev_{VER_OF[name]}' / detset / name / f'{stem}.pkl'
    return OUT / detset / name / f'{stem}.pkl'


def build(frame, cfg, cams, stereo, rd, parent_cfg, parent_params, previous, T_ego):
    objs, kept = F.candidates(copy.deepcopy(frame.dets), cams, cfg, stereo, frame.maps, frame.emb)
    if cfg['redetect']:
        w = np.asarray(parent_params['weights']); by_key = {}
        for o in objs:
            if not o['missing']: continue
            if 1 / (1 + np.exp(-F.object_features(o, parent_cfg) @ w)) < .1: continue
            added = failed = 0
            for c in list(o['missing']):
                pb = F.predicted_box(o, c, cams)
                if pb is None: continue
                d = rd.verify(frame.stem, o, c, pb, F.prepare.rect, cfg.get('redet_conf', .05), cfg.get('redet_iou', .25))
                if d is None: failed += 1; continue
                same = [k for k in kept if k['src'] == c and F.compatible(k['cls'], d['cls']) and F.box_iou(k['box'], d['box']) >= .7]
                if same: d = same[0]
                else:
                    key = (c, tuple(np.round(d['box'], -1)))
                    if key not in by_key:
                        nd = F.prepare([d], cams, stereo if cfg['depth'] else None, frame.maps, None)[0]
                        nd['uid'] = 100000 + len(by_key); nd['emb'] = None; by_key[key] = nd
                    d = by_key[key]
                o['dets'] = o['dets'] + [d]; o['src'] = sorted(o['src'] + [c]); o['missing'].remove(c); added += 1
            o['redet_added'], o['redet_failed'] = added, failed
    if cfg['track']:
        T.annotate(objs, previous, T_ego, cams)
    for o in objs: o['x'] = F.object_features(o, cfg)
    return objs, kept


def stage_cand(detset, name, lists, prev, limit_s):
    cfg, parent = CONFIGS[name]; cams = load_cameras(); stereo = StereoDepth(cams); rd = ReDetector()
    parent_cfg = CONFIGS[parent][0] if parent else None
    parent_params = params_of(detset, parent) if parent else None
    if parent and parent_params is None: raise SystemExit(f'fit parent {parent} first')
    t0 = time.time(); n = 0
    order = lists['devprev'] + lists['dev'] + lists['regression180'] + lists['regression60'] + lists['extra60']
    for stem in order:
        p = pkl(detset, name, stem)
        if p.exists(): continue
        if not (DETSETS[detset] / f'2_{stem}.json').exists(): continue   # detections not computed for this detset
        if time.time() - t0 > limit_s: break
        frame = Frame(stem, detset, stereo)
        previous = T_ego = None
        if cfg['track'] and stem in prev:
            pp = pkl(detset, parent, prev[stem])
            if pp.exists():
                pobjs = pickle.loads(pp.read_bytes())['objs']
                adm = admit_slim(pobjs, parent_cfg, parent_params)
                previous = T.summarise_slim(adm)
                pf = BASE / f'experiments/orb_cache/{prev[stem]}.npz'
                T_ego = T.ego_motion(np.load(pf) if pf.exists() else None, frame.orb, cams)
        objs, kept = build(frame, cfg, cams, stereo, rd, parent_cfg, parent_params, previous, T_ego)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(dict(objs=slim(objs), kept=[{k: d[k] for k in ('src', 'box', 'cls', 'conf', 'uid')} for d in kept])))
        n += 1
    rd.flush()
    left = sum(not pkl(detset, name, s).exists() and (DETSETS[detset] / f'2_{s}.json').exists() for s in order)
    print(f'cand {detset}/{name}: +{n} frames, {left} remaining', flush=True)
    return left


def _zs(o):
    ds = o['dets']; w = np.array([1 / max(d['sigZ'], 1e-3) ** 2 for d in ds])
    return float(np.sum([d['Z'] * wi for d, wi in zip(ds, w)]) / w.sum()), float(1 / np.sqrt(w.sum()))


def _ios(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0])); iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    sa = (a[2] - a[0]) * (a[3] - a[1]); sb = (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / max(min(sa, sb), 1e-9)


def dedup3d(objs, ios_thr, zfrac=.12):
    """E1 - 3-D aware containment suppression (development-set statistics in
    FUSION_ROUND3.md chose the rules below).
    A lower-ranked object is a duplicate of a kept one if
      (a) both contain the SAME detection (uid) - e.g. guided re-detection re-used a
          box that already belonged to another object; or
      (b) in EVERY shared camera one box lies mostly inside the other
          (intersection over the smaller box >= ios_thr) AND their depths agree.
    'Every camera' matters: adjacent real objects overlap in one view but not in all
    three (parallax), while a duplicate overlaps consistently."""
    order = sorted(objs, key=lambda o: (len(o['src']), o['conf']), reverse=True); kept = []
    for o in order:
        zo, so = _zs(o); dup = False
        bo = {d['src']: d for d in o['dets']}
        for k in kept:
            bk = {d['src']: d for d in k['dets']}; shared = set(bo) & set(bk)
            if not shared: continue
            if any(bo[c]['uid'] == bk[c]['uid'] for c in shared): dup = True; break
            if not F.compatible(o['cls'], k['cls']): continue
            zk, sk = _zs(k)
            if abs(zo - zk) > max(2.5 * np.hypot(so, sk), 1.5, zfrac * min(zo, zk)): continue
            if min(_ios(bo[c]['box'], bk[c]['box']) for c in shared) >= ios_thr: dup = True; break
        if not dup: kept.append(o)
    return kept


_MODELS = {}


def predict(objs, params):
    """Evidence model: logistic regression (weights) or E3 gradient-boosted trees."""
    if not objs: return np.zeros(0)
    X = np.array([o['x'] for o in objs])
    if params.get('model_path') or params.get('_model') is not None:
        m = params.get('_model')
        if m is None:
            key = params['model_path']
            if key not in _MODELS: _MODELS[key] = pickle.loads((BASE / key).read_bytes())
            m = _MODELS[key]
        return m.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-X @ np.asarray(params['weights'])))


def admit_slim(objs, cfg, params):
    pars = dict(params)
    objs = [dict(o) for o in objs]
    for o, p in zip(objs, predict(objs, params)): o['p'] = float(p)
    objs = [o for o in objs if np.median([d['box'][3] - d['box'][1] for d in o['dets']]) >= pars['min_height']]
    if cfg['milp']:
        chosen = F.solve_packing(objs, [F.logit(o['p']) - F.logit(pars['tau']) for o in objs]); out = [objs[i] for i in chosen]
    else:
        out = [o for o in objs if o['p'] >= pars['tau']]
    for o in out: o['conf'] = o['p']
    out = F.deduplicate(out, iou_threshold=.4, min_shared=2)
    multi = [o for o in out if len(o['src']) > 1]; res = []
    for o in out:
        if len(o['src']) == 1:
            d = o['dets'][0]
            if any(m['src'] == d['src'] and F.compatible(m['cls'], d['cls']) and F.box_iou(m['box'], d['box']) >= pars['residual_iou']
                   for k in multi for m in k['dets']): continue
        res.append(o)
    if params.get('ios', 9.) <= 1.5: res = dedup3d(res, params['ios'])   # 1.5 = shared-uid rule only
    return res


def labels(objs, kept, gt, milp):
    U = ev.build_union(gt, 25); ign = ev.build_union(gt, 0)
    def hits(d, UU):
        return {iid for iid, u in UU.items() if d['src'] in u['boxes'] and ev.cls_match(u['label'], d['cls']) and ev.iou(d['box'], u['boxes'][d['src']]) >= .5}
    y, w = [], []
    for o in objs:
        hs = [hits(d, U) for d in o['dets']]
        if milp:
            common = set.intersection(*hs) if hs else set()
            ok = False
            for iid in common:
                outside = any(iid in hits(k, U) for k in kept if k['src'] not in o['src'])
                if not outside: ok = True
            y.append(float(ok))
        else:
            y.append(float(any(hs)))
        small = (not y[-1]) and any(hits(d, ign) for d in o['dets']) and not any(hs)
        w.append(0. if small else 1.)
    return np.array(y), np.array(w)


def stage_fit(detset, name, lists):
    cfg, _ = CONFIGS[name]; frames = []
    for stem in lists['dev110' if cfg.get('dev110') else 'dev']:
        z = pickle.loads(pkl(detset, name, stem).read_bytes()); gt = truth(stem, (2, 3, 4))
        y, w = labels(z['objs'], z['kept'], gt, cfg['milp'])
        X = np.array([o['x'] for o in z['objs']]) if z['objs'] else np.zeros((0, len(F.FEATURES)))
        frames.append(dict(objs=z['objs'], gt=gt, X=X, y=y, w=w))
    active = np.ones(len(F.FEATURES), bool)
    Xall = np.concatenate([f['X'] for f in frames]); active &= Xall.std(0) > 0; active[0] = True
    from yolo_aware_fusion import fit_logistic
    def fit(fs, l2):
        m = np.concatenate([f['w'] for f in fs]) > 0
        if cfg.get('model') == 'gbdt':
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter={.3: 100, 1.: 200, 3.: 400}[l2],
                                                 min_samples_leaf=20, l2_regularization=1., random_state=0)
            clf.fit(np.concatenate([f['X'] for f in fs])[m], np.concatenate([f['y'] for f in fs])[m]); return clf
        X = np.concatenate([f['X'] for f in fs])[m][:, active]; y = np.concatenate([f['y'] for f in fs])[m]
        wa = fit_logistic(X, y, l2); w = np.zeros(len(F.FEATURES)); w[active] = wa; return w
    folds = [frames[i::5] for i in range(5)]
    grid = list(itertools.product((.3, 1., 3.), (.1, .15, .2, .3, .4, .5, .6), (0, 20), (.5, 1.01),
                                  (.6, .75, .9, 1.5) if cfg.get('dedup3d') else (9.,)))
    tot = {g: np.zeros(5, int) for g in grid}
    for k in range(5):
        train = [f for j in range(5) if j != k for f in folds[j]]
        for l2 in (.3, 1., 3.):
            w = fit(train, l2)
            for g in grid:
                if g[0] != l2: continue
                pars = dict(tau=g[1], min_height=g[2], residual_iou=g[3], ios=g[4])
                if cfg.get('model') == 'gbdt': pars['_model'] = w
                else: pars['weights'] = w.tolist()
                for f in folds[k]: tot[g] += evaluate(admit_slim(f['objs'], cfg, pars), f['gt'])
    best = max(grid, key=lambda g: metrics(tot[g])['f1'])
    w = fit(frames, best[0])
    extra = {}
    if cfg.get('model') == 'gbdt':
        rel = f'algorithm/depth_bev_params/{detset}_{name}.gbdt.pkl'; PARAMS.mkdir(parents=True, exist_ok=True)
        (BASE / rel).write_bytes(pickle.dumps(w)); extra = dict(model_path=rel, weights=None)
    else: extra = dict(weights=w.tolist())
    out = dict(config=name, switches=cfg, detset=detset, **extra, features=F.FEATURES, l2=best[0], tau=best[1],
               min_height=best[2], residual_iou=best[3], ios=best[4], cv_development=metrics(tot[best]), development_frames=lists['dev110' if cfg.get('dev110') else 'dev'])
    PARAMS.mkdir(parents=True, exist_ok=True)
    (PARAMS / f'{detset}_{name}.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    m = out['cv_development']; print(f"fit {detset}/{name}: CV P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} tau {best[1]} l2 {best[0]} hmin {best[2]} res {best[3]} ios {best[4]}", flush=True)


def stage_eval(detset, name, lists):
    cfg, _ = CONFIGS[name]; pars = params_of(detset, name); res = {}
    for split in ('regression180', 'regression60'):
        tot = np.zeros(5, int)
        for stem in lists[split]:
            z = pickle.loads(pkl(detset, name, stem).read_bytes())
            tot += evaluate(admit_slim(z['objs'], cfg, pars), truth(stem, (2, 3, 4)))
        res[split] = metrics(tot)
    res['cv_development'] = pars['cv_development']
    p = BASE / 'algorithm/evaluation_reports/depth_bev_results.json'
    allr = json.loads(p.read_text()) if p.exists() else {}
    allr[f'{detset}/{name}'] = res; p.write_text(json.dumps(allr, indent=1), encoding='utf-8')
    for s in ('regression180', 'regression60'):
        m = res[s]; print(f"eval {detset}/{name} {s}: TP {m['tp']} FP {m['fp']} FN {m['fn']} P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('detset'); ap.add_argument('names', nargs='+')
    ap.add_argument('--limit', type=float, default=150); a = ap.parse_args()
    import cv2; cv2.setNumThreads(2)
    lists, prev = splits(); t0 = time.time()
    for name in a.names:
        if a.stage == 'cand':
            if stage_cand(a.detset, name, lists, prev, a.limit - (time.time() - t0)) > 0: break
        elif a.stage == 'fit': stage_fit(a.detset, name, lists)
        elif a.stage == 'eval': stage_eval(a.detset, name, lists)
        elif a.stage == 'all':
            if stage_cand(a.detset, name, lists, prev, a.limit - (time.time() - t0)) > 0: break
            stage_fit(a.detset, name, lists); stage_eval(a.detset, name, lists)


if __name__ == '__main__':
    main()
