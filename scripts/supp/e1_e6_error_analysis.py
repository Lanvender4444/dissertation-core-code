"""E1 (baseline detection behaviour / cross-view inconsistency) and E6 (per-class/size breakdown).
Read-only on the frozen snapshot; no detector inference, fitting or GT edits."""
import re
from collections import defaultdict, Counter
from supp_lib import *

seg = segments()
OUT = new_run('e1_e6_error_analysis', dict(
    purpose='Plan objective 1: per-camera behaviour, fisheye radial effect, crop coverage ceiling, cross-view inconsistency; '
            'E6: recall by size/class/visible-camera set for Raw/A/F5-P/F5-R',
    detector_input='frozen five-camera crop-mosaic pool, YOLO11n, conf>=0.05; native (uncorrected) boxes for per-camera analysis',
    thresholds=[0.05, 0.25], matching='per-camera one-to-one, compatible class, IoU>=0.5, GT height>=25 px',
    size_proxy='GT box height in native pixels (no metric distance ground truth)',
    radial='|centre - principal point| / |corner(0,0) - principal point|', segments='full and guard+late'))

txt = (ROOT/'Intrinsics_cam.json').read_text()
PP = {int(a): (float(c), float(d)) for a, _, c, d in re.findall(
    r'"sensorId":\s*(\d+).*?"projection_model":\s*"(\w+)".*?"cu":\s*([\d.]+).*?"cv":\s*([\d.]+)', txt, re.S)}
W, H = 2464, 2056
R_BINS = [0, .2, .4, .6, .8, 9]
R_LABELS = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '>=0.8']


def match_camera(dets, gts, thr=0.5):
    big = [g for g in gts if g[2][3]-g[2][1] >= MIN_H]
    S = np.zeros((len(dets), len(big)))
    for i, d in enumerate(dets):
        for j, (iid, lab, box) in enumerate(big):
            if cls_match(lab, d['cls']):
                S[i, j] = iou(d['box'], box)
    hit = {}; tp = [False]*len(dets); hiou = {}
    if S.size:
        rr, cc = linear_sum_assignment(-((S >= thr)*1000+S))
        for r, c in zip(rr, cc):
            if S[r, c] >= thr:
                hit[big[c][0]] = r; tp[r] = True; hiou[big[c][0]] = S[r, c]
    small = [g for g in gts if g[2][3]-g[2][1] < MIN_H]
    ign = [(not tp[i]) and any(cls_match(l, d['cls']) and iou(d['box'], b) >= thr for _, l, b in small)
           for i, d in enumerate(dets)]
    return hit, tp, ign, hiou


def rnorm(c, box):
    cu, cv = PP[c]; x = (box[0]+box[2])/2; y = (box[1]+box[3])/2
    return np.hypot(x-cu, y-cv)/np.hypot(cu, cv)


def rbin(r):
    for k in range(5):
        if R_BINS[k] <= r < R_BINS[k+1]:
            return k


# crop windows per camera from extracted detection provenance
tiles = defaultdict(set)
for s in seg['full'][::25]:
    for d in frame(s)['dets'].values():
        tiles[d['src']].add(tuple(d['crop']))
TILES = {c: sorted(v) for c, v in tiles.items()}


def crop_cover(c, box):
    """fraction of GT box area inside the union of the camera's two input tiles"""
    x1, y1, x2, y2 = box; area = max(x2-x1, 1e-6)*max(y2-y1, 1e-6)
    # tiles share the same y-band; union in x
    ys = [(t[1], t[3]) for t in TILES[c]]; xs = sorted((t[0], t[2]) for t in TILES[c])
    y0, y1b = min(a for a, _ in ys), max(b for _, b in ys)
    iy = max(0, min(y2, y1b)-max(y1, y0))
    merged = []
    for a, b in xs:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    ix = sum(max(0, min(x2, b)-max(x1, a)) for a, b in merged)
    return ix*iy/area


def run(frames, tau):
    # per-camera own-view
    cam_cnt = defaultdict(lambda: np.zeros(3, int))         # c -> tp, fp, fn (own-view, ignore rule)
    size = defaultdict(lambda: np.zeros(2, int))            # (c, hbin) -> hit, n
    klass = defaultdict(lambda: np.zeros(2, int))           # (c, group) -> hit, n
    kprec = defaultdict(lambda: np.zeros(2, int))           # (c, group) -> tp, counted dets
    radial = defaultdict(lambda: [0, 0, [], []])            # (c, rbin) -> hit, n, heights, ious
    radial_h = defaultdict(lambda: np.zeros(2, int))        # (fish/pin, rbin, hbin) -> hit, n
    conf = defaultdict(list)                                # (c, 'tp'/'fp') -> confidences
    crop_out = Counter(); crop_n = Counter(); union_out = 0; union_n = 0
    AD = Counter(); cond = np.zeros((6, 6, 2), int)         # cond[i,j] = [miss_j, n] | annotated i&j, detected i
    fish_only = Counter()
    for s in frames:
        F = frame(s); g = F['gt']
        per_hit = {}
        for c in CAMS:
            dets = [det(d, corrected=False) for d in F['dets'].values() if d['src'] == c and d['conf'] >= tau]
            gts = g.get(c, [])
            hit, tp, ign, hiou = match_camera(dets, gts)
            per_hit[c] = hit
            big = [x for x in gts if x[2][3]-x[2][1] >= MIN_H]
            n_tp = sum(tp); n_fp = sum(1 for t, i in zip(tp, ign) if not t and not i)
            cam_cnt[c] += [n_tp, n_fp, len(big)-n_tp]
            for d, t, i in zip(dets, tp, ign):
                if not i:
                    conf[(c, 'tp' if t else 'fp')].append(d['conf'])
                    kprec[(c, YGROUP.get(d['cls'], 'other'))] += [int(t), 1]
            for iid, lab, box in big:
                h = box[3]-box[1]; hb = h_bin(h); hit_ = iid in hit
                size[(c, hb)] += [hit_, 1]; klass[(c, GROUP[lab])] += [hit_, 1]
                rb = rbin(rnorm(c, box)); R = radial[(c, rb)]
                R[0] += hit_; R[1] += 1; R[2].append(h)
                if hit_: R[3].append(hiou[iid])
                radial_h[('fisheye' if c in (1, 5) else 'pinhole', rb, hb)] += [hit_, 1]
                cov = crop_cover(c, box); crop_n[c] += 1; crop_out[c] += cov < .5
        # union-level
        union = build_union(g, MIN_H)
        for iid, u in union.items():
            A = {c for c, b in u['boxes'].items() if b[3]-b[1] >= MIN_H}
            if not A:
                continue
            D = {c for c in A if iid in per_hit[c]}
            AD[(len(A), len(D))] += 1
            for i in A:
                for j in A:
                    if i != j and i in D:
                        cond[i, j] += [j not in D, 1]
            if A & {2, 3, 4} and D and D <= {1, 5}:
                fish_only['forward_annotated_only_fisheye_detected'] += 1
            if A <= {1, 5}:
                fish_only['lateral_only_annotated'] += 1
                fish_only['lateral_only_detected'] += bool(D)
            union_n += 1
            union_out += all(crop_cover(c, u['boxes'][c]) < .5 for c in A)
    res = dict(
        camera=[dict(camera=c, **prf(cam_cnt[c])) for c in CAMS],
        size=[dict(camera=c, bin=H_LABELS[b], hits=int(size[(c, b)][0]), n=int(size[(c, b)][1]),
                   recall=size[(c, b)][0]/max(size[(c, b)][1], 1)) for c in CAMS for b in range(4)],
        klass=[dict(camera=c, group=k, hits=int(klass[(c, k)][0]), n=int(klass[(c, k)][1]),
                    recall=klass[(c, k)][0]/max(klass[(c, k)][1], 1),
                    precision=kprec[(c, k)][0]/max(kprec[(c, k)][1], 1), dets=int(kprec[(c, k)][1]))
               for c in CAMS for k in ('person', 'two-wheeler', 'vehicle')],
        radial=[dict(camera=c, bin=R_LABELS[b], hits=radial[(c, b)][0], n=radial[(c, b)][1],
                     recall=radial[(c, b)][0]/max(radial[(c, b)][1], 1),
                     median_height=float(np.median(radial[(c, b)][2])) if radial[(c, b)][2] else None,
                     mean_hit_iou=float(np.mean(radial[(c, b)][3])) if radial[(c, b)][3] else None)
                for c in CAMS for b in range(5)],
        radial_by_height=[dict(kind=k, rbin=R_LABELS[r], hbin=H_LABELS[h], hits=int(v[0]), n=int(v[1]),
                               recall=v[0]/max(v[1], 1)) for (k, r, h), v in sorted(radial_h.items())],
        crop=dict(per_camera=[dict(camera=c, outside=crop_out[c], n=crop_n[c], fraction=crop_out[c]/max(crop_n[c], 1))
                              for c in CAMS],
                  union_all_views_outside=union_out, union_n=union_n, ceiling_recall=1-union_out/max(union_n, 1),
                  tiles={c: TILES[c] for c in CAMS}),
        annotated_vs_detected=[dict(annotated=a, detected=d, n=n) for (a, d), n in sorted(AD.items())],
        conditional_miss=[dict(i=i, j=j, miss=int(cond[i, j, 0]), n=int(cond[i, j, 1]),
                               p_miss=cond[i, j, 0]/max(cond[i, j, 1], 1)) for i in CAMS for j in CAMS if i != j],
        fisheye=dict(fish_only),
        confidence={f'{c}_{k}': np.percentile(conf[(c, k)], [10, 25, 50, 75, 90]).tolist() if conf[(c, k)] else None
                    for c in CAMS for k in ('tp', 'fp')},
        confidence_hist={f'{c}_{k}': np.histogram(conf[(c, k)], bins=np.linspace(0.05, 1, 20))[0].tolist()
                         for c in CAMS for k in ('tp', 'fp')})
    return res


def e6(frames):
    """Recall by size / class group / visible camera set for Raw/A/F5-P/F5-R on the five-view union."""
    names = ['raw', 'A', 'F5P', 'F5R']
    rec = {n: defaultdict(lambda: np.zeros(2, int)) for n in names}
    prec = {n: defaultdict(lambda: np.zeros(2, int)) for n in names}
    for s in frames:
        F = frame(s); C = configs(s); g = F['gt']
        for n in names:
            objs = as_objects(C[n], F)
            cnt, pairs, union, ign = evaluate(objs, g, return_assignment=True)
            hit = {iid for _, iid in pairs}; tpobj = {i for i, _ in pairs}
            for iid, u in union.items():
                h = max(b[3]-b[1] for b in u['boxes'].values()); cams = set(u['boxes'])
                vis = 'forward only' if cams <= {2, 3, 4} else ('lateral only' if cams <= {1, 5} else 'forward + lateral')
                for key in (('size', H_LABELS[h_bin(h)]), ('class', GROUP[u['label']]), ('visible', vis), ('all', 'all')):
                    rec[n][key] += [iid in hit, 1]
            for i, o in enumerate(objs):
                if i in ign:
                    continue
                prec[n][('class', YGROUP.get(o['cls'], 'other'))] += [i in tpobj, 1]
                prec[n][('sources', 'lateral only' if set(o['src']) <= {1, 5} else
                         ('forward only' if set(o['src']) <= {2, 3, 4} else 'forward + lateral'))] += [i in tpobj, 1]
    out = []
    for n in names:
        for k, v in rec[n].items():
            out.append(dict(config=n, kind=k[0], bin=k[1], hits=int(v[0]), n=int(v[1]), recall=v[0]/max(v[1], 1)))
        for k, v in prec[n].items():
            out.append(dict(config=n, kind='precision_'+k[0], bin=k[1], tp=int(v[0]), n=int(v[1]), precision=v[0]/max(v[1], 1)))
    return out


if __name__ == '__main__':
    results = {}
    for name in ('full', 'guard_late'):
        results[name] = {f'tau{t}': run(seg[name], t) for t in (0.05, 0.25)}
        results[name]['e6'] = e6(seg[name])
        print(name, 'done', flush=True)
    dump(OUT/'results.json', results)
    r = results['full']['tau0.25']
    for x in r['camera']:
        print('cam', x['camera'], round(x['precision'], 3), round(x['recall'], 3))
    print('crop', r['crop']['per_camera'], r['crop']['ceiling_recall'])
    print('AD', r['annotated_vs_detected'])
    print('fish', r['fisheye'])
