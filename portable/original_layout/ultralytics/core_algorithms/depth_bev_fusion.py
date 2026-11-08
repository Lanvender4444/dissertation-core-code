"""Depth-aware BEV evidence fusion of 2-D YOLO detections (v3).  GT never read.

Seven switchable components (see algorithm/DEPTH_BEV_FUSION.md):
  depth    idea 1  stereo box depth -> 3-D point + sigma per detection
  bev      idea 2  Fast-BEV-style evidence grid: detections splatted into BEV,
                   per-camera max, cross-camera sum, peaks = objects
  occl     idea 3  occlusion-aware "should have been seen" via depth maps
  redetect idea 4  guided YOLO re-detection in cameras that should see it
  milp     idea 5  global set-packing association (instead of greedy)
  emb      idea 6  YOLO neck-feature appearance similarity
  track    idea 7  soft temporal evidence from ego-motion-compensated tracks
With all switches off the grouping is the old tri-compare.
"""
import itertools
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import milp, LinearConstraint, Bounds
from final_pipeline import filter_ego, core
from legacy_adapter import legacy
from multiview_dedup import deduplicate, box_iou
from multiview_fusion import compatible
from yolo_hr_detect import merge_nms

CLASS_H, CLASS_W = core.CLASS_HEIGHT, core.CLASS_WIDTH
GROUP = {0: 'person', 1: 'cycle', 3: 'cycle', 2: 'veh', 5: 'veh', 7: 'veh'}
W_IMG, H_IMG = 2464, 2056
CHI2_2D = 9.21                       # 99% gate, 2 dof
GRID = dict(x0=0., x1=200., y0=-60., y1=60., res=0.4)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(np.log(p / (1 - p)))


def weight(conf):
    """Evidence weight of one YOLO box: log-odds above a 5% floor (>=0)."""
    return max(logit(conf) - logit(.05), 0.)


# ---------------------------------------------------------------- per-detection geometry
def prepare(dets, cameras, stereo=None, maps=None, emb=None):
    """Attach uid, rectified box, truncation flags, 3-D point/cov, embedding."""
    rect = prepare.rect
    for i, d in enumerate(dets):
        d['uid'] = i
        d['rbox'] = rect.rect_box(d['box'], d['src'])
        x1, y1, x2, y2 = d['box']
        d['tl'], d['tr'], d['tb'] = x1 < core.TRUNC_PX, x2 > W_IMG - core.TRUNC_PX, y2 > H_IMG - core.TRUNC_PX
        cam = cameras[d['src']]
        r = cam.rays([[(x1 + x2) / 2, (y1 + y2) / 2]])[0]; d['ray'] = r
        rz = float((cam.R.T @ r)[2])
        g = stereo.box_depth(d['src'], maps[d['src']], d['box'], d['cls']) if stereo is not None else None
        if g is not None and g['Z'] < 250:
            Z, sZ, P = g['Z'], g['sigma'], g['point']; d['depth_src'] = 'stereo'
        else:
            h = y2 - y1; w = x2 - x1
            Z = cam.K[0, 0] * CLASS_W.get(d['cls'], 1.6) / max(w, 1) if d['tb'] else cam.K[1, 1] * CLASS_H.get(d['cls'], 1.6) / max(h, 1)
            Z = float(np.clip(Z, 1., 250.)); sZ = .35 * Z; P = cam.t + Z / max(rz, 1e-3) * r
            d['depth_src'] = 'size_prior'
        d['Z'], d['sigZ'], d['P'] = float(Z), float(sZ), np.asarray(P, float)
        u = r[:2] / (np.linalg.norm(r[:2]) + 1e-9); v = np.array([-u[1], u[0]])
        s_along = np.hypot(sZ, .5 * CLASS_W.get(d['cls'], 1.))
        s_cross = max(.35, .5 * CLASS_W.get(d['cls'], 1.)) + .003 * Z
        d['cov'] = s_along ** 2 * np.outer(u, u) + s_cross ** 2 * np.outer(v, v)
        d['emb'] = emb[d['src']][d['_k']] if emb is not None else None
    return dets


prepare.rect = core.TriRectifier(core.camera_params)


def maha(a, b):
    delta = a['P'][:2] - b['P'][:2]
    return float(delta @ np.linalg.solve(a['cov'] + b['cov'], delta))


def cosine(a, b):
    if a.get('emb') is None or b.get('emb') is None: return 1.
    return float(np.dot(a['emb'].astype(np.float32), b['emb'].astype(np.float32)))


def pair_ok(a, b, use_depth, use_emb, emb_min=.55):
    """Pairwise association gate.  Old: rectified-row + height ratio (tri-compare).
    +depth: 3-D Mahalanobis on BEV positions.  +emb: YOLO feature cosine."""
    if a['src'] == b['src'] or not core.gate(a, b): return None
    m = maha(a, b)
    if use_depth and m > CHI2_2D: return None
    c = cosine(a, b)
    if use_emb and c < emb_min: return None
    return m, c


# ---------------------------------------------------------------- idea 2: BEV evidence grid
def splat(ds, g=GRID):
    """Fast-BEV-style view transform for BOXES: each detection's Gaussian footprint
    (along-ray sigma from depth, cross-ray from object width) is written into a
    per-camera layer with max(); cameras are then summed (independent evidence)."""
    nx = int((g['x1'] - g['x0']) / g['res']); ny = int((g['y1'] - g['y0']) / g['res'])
    layers = {c: np.zeros((nx, ny), np.float32) for c in (2, 3, 4)}
    for d in ds:
        ev = np.linalg.eigvalsh(d['cov']); r = 3 * np.sqrt(ev.max())
        cx, cy = d['P'][:2]
        ix0, ix1 = max(0, int((cx - r - g['x0']) / g['res'])), min(nx, int((cx + r - g['x0']) / g['res']) + 1)
        iy0, iy1 = max(0, int((cy - r - g['y0']) / g['res'])), min(ny, int((cy + r - g['y0']) / g['res']) + 1)
        if ix0 >= ix1 or iy0 >= iy1: continue
        xs = g['x0'] + (np.arange(ix0, ix1) + .5) * g['res']; ys = g['y0'] + (np.arange(iy0, iy1) + .5) * g['res']
        X, Y = np.meshgrid(xs - cx, ys - cy, indexing='ij'); Q = np.stack([X, Y], -1)
        inv = np.linalg.inv(d['cov']); dens = np.exp(-.5 * np.einsum('...i,ij,...j', Q, inv, Q))
        L = layers[d['src']][ix0:ix1, iy0:iy1]; np.maximum(L, weight(d['conf']) * dens, out=L)
    return sum(layers.values())


def field_value(S, pt, g=GRID):
    ix = int((pt[0] - g['x0']) / g['res']); iy = int((pt[1] - g['y0']) / g['res'])
    if 0 <= ix < S.shape[0] and 0 <= iy < S.shape[1]: return float(S[ix, iy])
    return 0.


def bev_field_features(objs, dets):
    """BEV evidence for existing groups: total field at the object's BEV point and the
    part NOT explained by its own members (support from unassigned/duplicate boxes)."""
    fields = {grp: splat([d for d in dets if GROUP.get(d['cls']) == grp]) for grp in ('veh', 'person', 'cycle')}
    for o in objs:
        P, Z, s = object_point(o); S = fields[GROUP.get(o['cls'], 'veh')]
        total = field_value(S, P[:2]); own = 0.
        for c in set(d['src'] for d in o['dets']):
            vals = []
            for d in o['dets']:
                if d['src'] != c: continue
                q = P[:2] - d['P'][:2]; vals.append(weight(d['conf']) * np.exp(-.5 * float(q @ np.linalg.solve(d['cov'], q))))
            own += max(vals)
        o['bev_evidence'] = total; o['bev_other'] = max(total - own, 0.)


def bev_grouping(dets, s_min=.8):
    g = GRID; objs = []; used = set()
    for grp in ('veh', 'person', 'cycle'):
        ds = [d for d in dets if GROUP.get(d['cls']) == grp]
        if not ds: continue
        S = splat(ds)
        # small NMS window (3 cells = 1.2 m): adjacent pedestrians keep separate peaks
        peaks = [tuple(p) for p in np.argwhere((S == maximum_filter(S, size=3)) & (S >= s_min))]
        pts = {p: np.array([g['x0'] + (p[0] + .5) * g['res'], g['y0'] + (p[1] + .5) * g['res']]) for p in peaks}

        def support(pt):
            """current evidence at pt from UNUSED detections: per-camera max, summed"""
            members, total = [], 0.
            for c in (2, 3, 4):
                best = None
                for d in ds:
                    if d['src'] != c or d['uid'] in used: continue
                    q = pt - d['P'][:2]; m2 = float(q @ np.linalg.solve(d['cov'], q))
                    if m2 > CHI2_2D: continue
                    val = weight(d['conf']) * np.exp(-.5 * m2)
                    if best is None or val > best[0]: best = (val, d)
                if best: members.append(best[1]); total += best[0]
            return total, members

        import heapq
        heap = [(-float(S[p]), p) for p in peaks]; heapq.heapify(heap)
        while heap:
            neg, p = heapq.heappop(heap)
            total, members = support(pts[p])
            if total < s_min or not members: continue
            if heap and total < -heap[0][0] - 1e-9:          # stale: re-rank lazily
                heapq.heappush(heap, (-total, p)); continue
            kept = []
            for d in sorted(members, key=lambda d: -d['conf']):
                if all(core.gate(d, k) for k in kept): kept.append(d)
            for d in kept: used.add(d['uid'])
            objs.append(dict(dets=kept, src=sorted(d['src'] for d in kept), bev_evidence=float(total), point=pts[p]))
    for d in dets:
        if d['uid'] not in used:
            objs.append(dict(dets=[d], src=[d['src']], bev_evidence=weight(d['conf']), point=d['P'][:2]))
    return objs


# ---------------------------------------------------------------- old grouping, re-used
def legacy_grouping(dets):
    _, raw = legacy(dets, prepare.rect)
    by = {d['uid']: d for d in dets}
    out = []
    for o in raw:
        members = [by[m['uid']] for m in o['dets']]
        out.append(dict(dets=members, src=sorted(o['src']), legacy_label=o['label'],
                        bev_evidence=None, point=None))
    return out


def split_by_appearance(objs, thr):
    """Idea 6 in the grouping: detach a member whose YOLO-feature cosine to every
    other member is below thr (threshold from development statistics: keeps 97%
    of same-object pairs, rejects 33% of different-object pairs at 0.7)."""
    out = []
    for o in objs:
        ds = o['dets']
        if len(ds) < 2: out.append(o); continue
        bad = [d for d in ds if max(cosine(d, e) for e in ds if e is not d) < thr]
        if len(bad) == len(ds) or not bad: out.append(o); continue
        good = [d for d in ds if d not in bad]
        out.append(dict(o, dets=good, src=sorted(d['src'] for d in good)))
        out += [dict(dets=[d], src=[d['src']], bev_evidence=None, point=None, legacy_label='SPLIT') for d in bad]
    return out


# ---------------------------------------------------------------- idea 5: global hypotheses
def hypotheses(dets, use_depth, use_emb, max_hyp=4000):
    by = {c: [d for d in dets if d['src'] == c] for c in (2, 3, 4)}
    edges = {}
    for ca, cb in ((2, 3), (3, 4), (2, 4)):
        for a in by[ca]:
            for b in by[cb]:
                e = pair_ok(a, b, use_depth, use_emb)
                if e is not None: edges[a['uid'], b['uid']] = edges[b['uid'], a['uid']] = e
    H = [[d] for d in dets]
    H += [[a, b] for ca, cb in ((2, 3), (3, 4), (2, 4)) for a in by[ca] for b in by[cb] if (a['uid'], b['uid']) in edges]
    for a in by[2]:
        for b in by[3]:
            if (a['uid'], b['uid']) not in edges: continue
            for c in by[4]:
                if (a['uid'], c['uid']) in edges and (b['uid'], c['uid']) in edges: H.append([a, b, c])
    hyps = [dict(dets=h, src=sorted(d['src'] for d in h), bev_evidence=None, point=None) for h in H[:max_hyp]]
    for o in hyps: o['outside_support'] = outside_support(o, dets, edges)
    return hyps


def solve_packing(hyps, scores):
    """max sum(score*x) s.t. every detection used at most once (HiGHS MILP)."""
    keep = [i for i, s in enumerate(scores) if s > 0]
    if not keep: return []
    uids = sorted({d['uid'] for i in keep for d in hyps[i]['dets']}); col = {u: j for j, u in enumerate(uids)}
    A = np.zeros((len(uids), len(keep)))
    for j, i in enumerate(keep):
        for d in hyps[i]['dets']: A[col[d['uid']], j] = 1
    res = milp(c=-np.array([scores[i] for i in keep]), constraints=LinearConstraint(A, 0, 1),
               integrality=np.ones(len(keep)), bounds=Bounds(0, 1))
    if res.x is None: return []
    return [keep[j] for j in np.where(res.x > .5)[0]]


# ---------------------------------------------------------------- idea 3: visibility / occlusion
def object_point(o):
    ds = o['dets']; w = np.array([1 / d['sigZ'] ** 2 for d in ds])
    P = (np.array([d['P'] for d in ds]) * w[:, None]).sum(0) / w.sum()
    Z = float(np.sum([d['Z'] * wi for d, wi in zip(ds, w)]) / w.sum()); s = float(1 / np.sqrt(w.sum()))
    return P, Z, s


def predicted_box(o, cid, cameras):
    """Project the object into camera cid, size taken from the nearest member box."""
    P, Z, _ = object_point(o)
    uv, valid = cameras[cid].project([P])
    if not valid[0]: return None
    ref = min(o['dets'], key=lambda d: abs(d['src'] - cid))
    w, h = ref['box'][2] - ref['box'][0], ref['box'][3] - ref['box'][1]
    cx, cy = uv[0]
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def visibility(o, cameras, stereo=None, maps=None, use_occl=False):
    """Cameras (2/3/4) not supporting o that SHOULD see it.
    Geometric: predicted box centre inside the image (margin 15 px).
    +occl: and the depth map at the predicted box is not clearly nearer
    (a nearer surface means an occluder, so absence is not evidence)."""
    missing, occluded = [], []
    P, Z, s = object_point(o)
    for c in (2, 3, 4):
        if c in o['src']: continue
        b = predicted_box(o, c, cameras)
        if b is None: continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not (15 < cx < W_IMG - 15 and 15 < cy < H_IMG - 15): continue
        if use_occl and stereo is not None:
            xs = np.linspace(b[0] + .3 * (b[2] - b[0]), b[2] - .3 * (b[2] - b[0]), 5)
            ys = np.linspace(b[1] + .25 * (b[3] - b[1]), b[3] - .2 * (b[3] - b[1]), 5)
            zd = stereo.depth_at_pixels(c, maps[c], [(x, y) for y in ys for x in xs])
            zd = zd[np.isfinite(zd)]
            if len(zd) >= 6 and np.median(zd) < Z - max(2.5 * s, 1.5, .12 * Z):
                occluded.append(c); continue
        missing.append(c)
    return missing, occluded


# ---------------------------------------------------------------- features for the evidence model
FEATURES = ['bias', 'v1', 'v2', 'v3', 'lmax', 'lmin', 'lmean', 'logh', 'person', 'trunc',
            'n_missing', 'n_occluded', 'depth_spread', 'size_resid', 'stereo_frac', 'logZ',
            'bev_evidence', 'bev_other', 'emb_min', 'redet_added', 'redet_failed', 'track', 'outside_support']


def object_features(o, cfg):
    ds = o['dets']; cs = [d['conf'] for d in ds]; n = len(ds)
    hs = [d['box'][3] - d['box'][1] for d in ds]
    trunc = any(d['tl'] or d['tr'] or d['tb'] for d in ds)
    P, Z, s = object_point(o)
    spread = max([abs(a['Z'] - b['Z']) / np.hypot(a['sigZ'], b['sigZ']) for a, b in itertools.combinations(ds, 2)], default=0.)
    cls = max(ds, key=lambda d: d['conf'])['cls']
    # physical height implied by depth vs class prior (only for untruncated boxes)
    res = [abs(np.log(max((d['box'][3] - d['box'][1]) * d['Z'] / 1445., 1e-3) / CLASS_H.get(cls, 1.6)))
           for d in ds if not d['tb']]
    embs = [cosine(a, b) for a, b in itertools.combinations(ds, 2)]
    f = dict(bias=1, v1=n == 1, v2=n == 2, v3=n >= 3, lmax=logit(max(cs)), lmin=logit(min(cs)),
             lmean=np.mean([logit(c) for c in cs]), logh=np.log(max(np.median(hs), 1)), person=cls == 0, trunc=trunc,
             n_missing=len(o.get('missing', [])), n_occluded=len(o.get('occluded', [])),
             depth_spread=min(spread, 5.) if cfg['depth'] else 0., size_resid=min(np.mean(res), 3.) if (cfg['depth'] and res) else 0.,
             stereo_frac=np.mean([d['depth_src'] == 'stereo' for d in ds]) if cfg['depth'] else 0.,
             logZ=np.log(Z) if cfg['depth'] else 0., bev_evidence=(o.get('bev_evidence') or 0.) if (cfg['bev'] or cfg.get('bevf')) else 0.,
             bev_other=o.get('bev_other', 0.) if cfg.get('bevf') else 0.,
             emb_min=min(embs) if (cfg['emb'] and embs) else (1. if cfg['emb'] else 0.),
             redet_added=o.get('redet_added', 0), redet_failed=o.get('redet_failed', 0),
             track=o.get('track', 0.), outside_support=o.get('outside_support', 0.))
    return np.array([float(f[k]) for k in FEATURES])


def outside_support(o, dets, edges):
    """Strongest detection in a camera OUTSIDE the hypothesis that is pairwise
    consistent with every member: a hypothesis that leaves such support out is
    probably incomplete (MILP completeness cue)."""
    ids = {d['uid'] for d in o['dets']}; best = 0.
    for d in dets:
        if d['uid'] in ids or d['src'] in o['src']: continue
        if all((d['uid'], m['uid']) in edges for m in o['dets']):
            best = max(best, weight(d['conf']))
    return best


# ---------------------------------------------------------------- main entry
DEFAULT = dict(depth=False, bev=False, bevf=False, occl=False, redetect=False, milp=False, emb=False, track=False)


def candidates(dets, cameras, cfg, stereo=None, maps=None, emb=None):
    """Ego gate -> vehicle NMS -> geometry -> grouping.  Returns (objects, kept dets)."""
    kept, removed = filter_ego([d for d in dets if d['src'] in (2, 3, 4)], cameras, True)
    kept = [dict(d) for c in (2, 3, 4) for d in merge_nms([d for d in kept if d['src'] == c], iou_thr=.7, ios_thr=2.)]
    prepare(kept, cameras, stereo if cfg['depth'] else None, maps, emb if cfg['emb'] else None)
    if cfg['milp']:
        objs = hypotheses(kept, cfg['depth'], cfg['emb'])
    elif cfg['bev']:
        objs = bev_grouping(kept)
    else:
        objs = legacy_grouping(kept)
    if cfg['emb'] and cfg.get('emb_split') and not cfg['milp']:
        objs = split_by_appearance(objs, cfg['emb_split'])
    for o in objs:
        o['missing'], o['occluded'] = visibility(o, cameras, stereo, maps, cfg['occl'])
        o['cls'] = max(o['dets'], key=lambda d: d['conf'])['cls']
    if cfg.get('bevf'): bev_field_features(objs, kept)
    return objs, kept


def score(objs, cfg, w):
    for o in objs:
        o['x'] = object_features(o, cfg); o['p'] = float(1 / (1 + np.exp(-o['x'] @ w)))
    return objs


def admit(objs, cfg, params, w):
    score(objs, cfg, w)
    objs = [o for o in objs if np.median([d['box'][3] - d['box'][1] for d in o['dets']]) >= params['min_height']]
    if cfg['milp']:
        tau = params['tau']
        chosen = solve_packing(objs, [logit(o['p']) - logit(tau) for o in objs])
        out = [objs[i] for i in chosen]
    else:
        out = [o for o in objs if o['p'] >= params['tau']]
    for o in out: o['conf'] = o['p']
    out = deduplicate(out, iou_threshold=params.get('dedup_iou', .4), min_shared=2)
    multi = [o for o in out if len(o['src']) > 1]
    res = []
    for o in out:
        if len(o['src']) == 1:
            d = o['dets'][0]
            if any(m['src'] == d['src'] and compatible(m['cls'], d['cls']) and box_iou(m['box'], d['box']) >= params['residual_iou']
                   for k in multi for m in k['dets']):
                continue
        res.append(o)
    return res
