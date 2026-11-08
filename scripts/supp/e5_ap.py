"""E5: average precision.
(a) Standard single-camera detector AP (COCO-style 101-point interpolation, greedy score-ordered
    matching per category, GT shorter than 25 px treated as ignore regions) on each camera's own GT,
    using the frozen .05 pool (native boxes). Two category definitions: strict 6 classes, and
    vehicle-merged (car/bus/truck -> vehicle; bicycle/motorcycle/rider -> two-wheeler).
(b) Group AP of fused outputs under the dissertation protocol (any-member, compatible class,
    cardinality-first one-to-one), ranking objects by maximum member detector confidence,
    at IoU 0.50:0.05:0.95, full and guard+late; per class-group AP50 and their mean.
"""
from collections import defaultdict
from supp_lib import *
from e2_simple_baselines import ray_nms

seg = segments()
THRS = [round(0.5+0.05*k, 2) for k in range(10)]
OUT = new_run('e5_ap', dict(
    purpose='Plan objective 4 (mAP): standard single-camera AP and protocol-consistent group AP of fused outputs',
    single_camera=dict(detections='frozen .05 crop pool, native boxes', gt='own camera XML', ignore='GT height<25 px',
                       interpolation='COCO 101-point', categories=['strict', 'merged']),
    group_ap=dict(ranking='max member detector confidence (box-corrected)', protocol='dissertation any-member one-to-one',
                  iou=THRS, configs=['F5P', 'F5R', 'A', 'B1 (all .05)', 'B2 theta5 (all .05)']),
    caveat='.05 detector floor truncates the low-score tail; group AP is not comparable with COCO detector AP'))

STRICT = {'person': 'person', 'rider': 'person', 'bicycle': 'bicycle', 'motorcycle': 'motorcycle',
          'car': 'car', 'bus': 'bus', 'truck': 'truck'}
YSTRICT = {0: 'person', 1: 'bicycle', 3: 'motorcycle', 2: 'car', 5: 'bus', 7: 'truck'}


def coco_ap(scored, npos):
    """scored: list of (score, is_tp); npos: positives. COCO 101-point interpolated AP"""
    if npos == 0:
        return None
    scored.sort(key=lambda x: -x[0]); tp = np.cumsum([t for _, t in scored]); fp = np.cumsum([1-t for _, t in scored])
    rec = tp/npos; prec = tp/np.maximum(tp+fp, 1e-9)
    for i in range(len(prec)-2, -1, -1):
        prec[i] = max(prec[i], prec[i+1])
    rs = np.linspace(0, 1, 101); q = []
    for r in rs:
        k = np.searchsorted(rec, r, side='left')
        q.append(prec[k] if k < len(prec) else 0.)
    return float(np.mean(q))


def single_camera(frames, mapping, ymap):
    res = {}
    for c in CAMS:
        per_thr = {}
        for thr in (0.5, 0.75) + tuple(THRS):
            if thr in per_thr:
                continue
            scored = defaultdict(list); npos = defaultdict(int)
            for s in frames:
                F = frame(s); gts = F['gt'].get(c, [])
                dets = [d for d in F['dets'].values() if d['src'] == c]
                for cat in set(mapping.values()):
                    G = [(b, b[3]-b[1] >= MIN_H) for _, l, b in gts if mapping[l] == cat]
                    npos[cat] += sum(v for _, v in G)
                    Ds = sorted([d for d in dets if ymap[d['cls']] == cat], key=lambda d: -d['conf'])
                    used = [False]*len(G)
                    for d in Ds:
                        best, bj = thr, -1
                        for j, (b, valid) in enumerate(G):
                            if used[j] and valid:
                                continue
                            v = iou(d['box'], b)
                            if v >= best and (bj < 0 or (valid and not G[bj][1]) or v > best):
                                best, bj = v, j
                        if bj < 0:
                            scored[cat].append((d['conf'], 0))
                        elif not G[bj][1]:
                            pass  # matched an ignore region
                        else:
                            used[bj] = True; scored[cat].append((d['conf'], 1))
            aps = {cat: coco_ap(scored[cat], npos[cat]) for cat in npos}
            per_thr[thr] = dict(per_class={k: v for k, v in aps.items()}, npos=dict(npos),
                                mAP=float(np.mean([v for v in aps.values() if v is not None])))
        res[c] = dict(AP50=per_thr[0.5]['mAP'], AP75=per_thr[0.75]['mAP'],
                      AP50_95=float(np.mean([per_thr[t]['mAP'] for t in THRS])), per_class50=per_thr[0.5]['per_class'],
                      npos=per_thr[0.5]['npos'])
        print('cam', c, round(res[c]['AP50'], 4), round(res[c]['AP50_95'], 4), flush=True)
    return res


def group_events(objs, gt_by_cam):
    """Precompute per-object IoU matrices; return function computing incremental (tp, fp) by score prefix."""
    union = build_union(gt_by_cam, MIN_H); ignored = build_union(gt_by_cam, 0)
    keys = list(union); S = np.zeros((len(objs), len(keys))); Si = np.zeros(len(objs))
    for i, o in enumerate(objs):
        for d in o['dets']:
            for j, iid in enumerate(keys):
                u = union[iid]
                if d['src'] in u['boxes'] and cls_match(u['label'], d['cls']):
                    S[i, j] = max(S[i, j], iou(d['box'], u['boxes'][d['src']]))
            for iid, u in ignored.items():
                if iid in union:
                    continue
                if d['src'] in u['boxes'] and cls_match(u['label'], d['cls']):
                    Si[i] = max(Si[i], iou(d['box'], u['boxes'][d['src']]))
    return S, Si, keys, union


def prefix_counts(S, Si, order, thr):
    """(tp, fp) after each tie-free prefix in 'order' (indices), cardinality-max matching at thr."""
    out = []
    for k in range(1, len(order)+1):
        idx = order[:k]; sub = S[idx]
        if sub.size:
            E = (sub >= thr)
            rr, cc = linear_sum_assignment(-(E*1000+sub*E))
            tp = int(E[rr, cc].sum())
        else:
            tp = 0
        ign = sum(1 for i in idx if S[i].max(initial=0) < thr and Si[i] >= thr)
        out.append((tp, k-ign-tp))
    return out


def group_ap(frames, getter, by_class=False):
    events = {thr: [] for thr in THRS}; npos = 0
    cls_events = defaultdict(list); cls_pos = defaultdict(int)
    for s in frames:
        F = frame(s); objs = getter(s, F)
        S, Si, keys, union = group_events(objs, F['gt'])
        npos += len(keys)
        order = sorted(range(len(objs)), key=lambda i: -objs[i]['conf'])
        for thr in THRS:
            prev = (0, 0)
            for i, (tp, fp) in zip(order, prefix_counts(S, Si, order, thr)):
                events[thr].append((objs[i]['conf'], tp-prev[0], fp-prev[1])); prev = (tp, fp)
        if by_class:
            for cg in ('person', 'two-wheeler', 'vehicle'):
                kk = [j for j, iid in enumerate(keys) if GROUP[union[iid]['label']] == cg]
                oo = [i for i in order if YGROUP.get(objs[i]['cls']) == cg]
                cls_pos[cg] += len(kk)
                if not oo:
                    continue
                sub = S[np.ix_(oo, kk)] if kk else np.zeros((len(oo), 0))
                prev = (0, 0)
                for n, i in enumerate(oo, 1):
                    m = sub[:n]
                    if m.size:
                        E = m >= .5; rr, cc = linear_sum_assignment(-(E*1000+m*E)); tp = int(E[rr, cc].sum())
                    else:
                        tp = 0
                    ign = sum(1 for x in oo[:n] if S[x].max(initial=0) < .5 and Si[x] >= .5)
                    fp = n-ign-tp
                    cls_events[cg].append((objs[i]['conf'], tp-prev[0], fp-prev[1])); prev = (tp, fp)
    def ap_from(ev, pos):
        ev = sorted(ev, key=lambda e: -e[0]); tp = np.cumsum([e[1] for e in ev]); fp = np.cumsum([e[2] for e in ev])
        rec = tp/pos; prec = tp/np.maximum(tp+fp, 1e-9)
        env = np.maximum.accumulate(prec[::-1])[::-1]
        ap_all = float(np.sum(np.diff(np.r_[0, rec])*env))
        curve = [dict(score=float(ev[i][0]), recall=float(rec[i]), precision=float(prec[i]))
                 for i in range(0, len(ev), max(1, len(ev)//200))]
        return ap_all, curve
    res = {}
    for thr in THRS:
        ap, curve = ap_from(events[thr], npos)
        res[f'AP{int(round(thr*100))}'] = ap
        if thr == 0.5:
            res['curve50'] = curve
    res['AP50_95'] = float(np.mean([res[f'AP{int(round(t*100))}'] for t in THRS]))
    if by_class:
        res['per_class_AP50'] = {cg: ap_from(cls_events[cg], cls_pos[cg])[0] for cg in cls_events}
        res['mAP50_class_groups'] = float(np.mean(list(res['per_class_AP50'].values())))
    return res


GETTERS = {
    'F5P': lambda s, F: as_objects(configs(s)['F5P'], F),
    'F5R': lambda s, F: as_objects(configs(s)['F5R'], F),
    'A': lambda s, F: as_objects(configs(s)['A'], F),
    'B2': lambda s, F: as_objects(ray_nms(F, 5.0, 0.05), F),
    'B1': lambda s, F: as_objects([[u] for u in F['dets']], F),
}

if __name__ == '__main__':
    import sys
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    res = {}
    if part in ('all', 'single'):
        res['single_camera'] = {'strict': single_camera(seg['full'], STRICT, YSTRICT),
                                'merged': single_camera(seg['full'], GROUP, YGROUP)}
        dump(OUT/'single_camera.json', res['single_camera'])
    if part in ('all', 'group'):
        g = {}
        for name in ('F5P', 'F5R', 'A', 'B2', 'B1'):
            g[name] = {}
            for segname in ('full', 'guard_late'):
                g[name][segname] = group_ap(seg[segname], GETTERS[name], by_class=(segname == 'full'))
                print(name, segname, {k: round(v, 4) for k, v in g[name][segname].items() if k.startswith('AP') or k.startswith('mAP')},
                      g[name][segname].get('per_class_AP50'), flush=True)
            dump(OUT/'group_ap.json', g)
