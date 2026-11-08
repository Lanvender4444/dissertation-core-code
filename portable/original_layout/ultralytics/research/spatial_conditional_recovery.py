"""Sparse leave-view-out spatial evidence and factored residual admission.

Camera-centre triangulation is uncertain object evidence, not ground contact.
The queried view never participates in generating its supporting hypotheses.
Unknown evidence stays explicit; this module has no GT or output deletion.
"""
import itertools
import numpy as np
from multiview_geometry import triangulate
from multiview_fusion import compatible

DIM = 18


def anchors(pool, cameras, probabilities, maximum=96):
    result = []
    edges = sorted(probabilities.items(), key=lambda x: (-x[1], x[0]))
    for (u, v), score in edges:
        a, b = pool[u], pool[v]
        if score < .75:
            break
        if a['src'] == b['src'] or not compatible(a['cls'], b['cls']):
            continue
        if not (a['geometry_valid'] and b['geometry_valid']):
            continue
        p, gap, ranges, angle = triangulate(cameras[a['src']], a['ray'], cameras[b['src']], b['ray'])
        if not np.isfinite(p).all() or min(ranges) < 1 or max(ranges) > 150 or angle < .01:
            continue
        if gap > max(.5, .1 * min(ranges)):
            continue
        result.append(dict(uids=(u, v), cameras=(a['src'], b['src']), point=p,
                           score=float(score), confidence=float(np.sqrt(a['conf'] * b['conf'])),
                           gap=gap/max(min(ranges), 1), angle=angle))
        if len(result) == maximum:
            break
    return result


def evidence(uid, pool, cameras, hypotheses, reference):
    d = pool[uid]
    used = {u for g in reference for u in g}
    rows = []
    box = np.asarray(d['box'], float)
    centre = (box[:2]+box[2:])/2
    size = np.maximum(box[2:]-box[:2], 4)
    for h in hypotheses:
        # Whole view exclusion, stronger than excluding just the queried UID.
        if d['src'] in h['cameras']:
            continue
        if any(not compatible(d['cls'], pool[u]['cls']) for u in h['uids']):
            continue
        uv, valid = cameras[d['src']].project([h['point']])
        if not valid[0] or not np.isfinite(uv).all():
            continue
        error = float(np.linalg.norm((uv[0]-centre)/size))
        distance = float(np.linalg.norm(h['point']-cameras[d['src']].t))
        inside = bool(np.all(uv[0] >= box[:2]) and np.all(uv[0] <= box[2:]))
        quality = h['score']*h['confidence']*np.exp(-min(error, 20)**2/2)
        rows.append((error, distance, inside, quality, h, bool(set(h['uids']) & used)))
    x = np.zeros(DIM)
    x[0] = bool(d['geometry_valid'])
    if not rows:
        return x
    rows.sort(key=lambda r: (r[0], -r[3], r[4]['uids']))
    best = rows[0]
    close = [r for r in rows if r[0] <= 1]
    x[1:] = [1, min(len(rows),96)/96, min(best[0],10)/10,
             sum(r[0] <= .5 for r in rows)/len(rows), len(close)/len(rows),
             best[3], max(r[3] for r in rows), np.log1p(best[1])/6,
             best[4]['gap'], best[4]['angle'], best[4]['score'], best[2], best[5],
             len(set(c for r in close for c in r[4]['cameras']))/5,
             np.std([np.log1p(r[1]) for r in close]) if close else 0,
             min(rows[1][0]-best[0],10)/10 if len(rows)>1 else 1,
             sum(r[5] for r in close)/max(len(close),1)]
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite sparse spatial evidence')
    return x


def factored(validity, novelty):
    v, n = np.asarray(validity, float), np.asarray(novelty, float)
    if v.shape != n.shape or not np.isfinite(v).all() or not np.isfinite(n).all():
        raise ValueError('Invalid conditional probabilities')
    if np.any((v<0)|(v>1)|(n<0)|(n>1)):
        raise ValueError('Probability out of range')
    return v*n
