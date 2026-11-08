"""Frozen, GT-free alternatives to unsafe single-view containment suppression.

These are experiments, not replacements for final_pipeline. Original boxes and
source identities are retained. All numeric gates are engineering hypotheses.
"""
import copy
import numpy as np
from multiview_dedup import box_iou
from multiview_fusion import compatible


def ios(a, b):
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    area = lambda q: np.prod(np.maximum(0, np.asarray(q[2:]) - q[:2]))
    return float(np.prod(wh) / max(min(area(a), area(b)), 1e-9))


def reliable_depth(d):
    return (d.get('depth_src') == 'stereo' and np.isfinite(d.get('Z', np.nan))
            and np.isfinite(d.get('sigZ', np.nan)) and d['Z'] > 0
            and 0 < d['sigZ'] / d['Z'] <= .15)


def duplicate(a, b, threshold=.75):
    """Exact observation exclusivity OR multi-view containment with no reliable veto.

    Unlike E1, one shared view alone cannot establish duplication. A precise
    inconsistent depth vetoes containment; missing depth is NOT precise depth.
    Exact same observation must never be assigned to two output identities.
    """
    aa = {d['src']: d for d in a['dets']}
    bb = {d['src']: d for d in b['dets']}
    shared = set(aa) & set(bb)
    if any(tuple(aa[c]['box']) == tuple(bb[c]['box']) and aa[c]['cls'] == bb[c]['cls'] for c in shared):
        return True
    if len(shared) < 2 or not compatible(a['cls'], b['cls']):
        return False
    for c in shared:
        x, y = aa[c], bb[c]
        if not compatible(x['cls'], y['cls']) or ios(x['box'], y['box']) < threshold:
            return False
        if reliable_depth(x) and reliable_depth(y):
            if abs(x['Z']-y['Z']) > max(1.5, 2.5*np.hypot(x['sigZ'], y['sigZ'])):
                return False
    return True


def conservative_dedup(objects):
    """Complete-link clusters; no averaging or synthetic camera observations."""
    groups = []
    for o in sorted(objects, key=lambda x: (len(x['src']), x['conf']), reverse=True):
        for group in groups:
            if all(duplicate(o, k) for k in group):
                group.append(o)
                break
        else:
            groups.append([o])
    return [g[0] for g in groups]


def rescue_unexplained(objects, legacy_objects):
    """Dual decision path: rescue strong baseline objects unclaimed in ANY view.

    Both decisions see the same detector, so they are NOT independent evidence.
    Fixed >=.60 max confidence, >=2 supporting cameras and <=.10 compatible IoU
    against every accepted member. This tests recall preservation, not truth.
    """
    output = copy.deepcopy(objects)
    for o in sorted(legacy_objects, key=lambda x: -x['conf']):
        if len(o['src']) < 2 or o['conf'] < .6:
            continue
        explained = any(d['src'] == e['src'] and compatible(d['cls'], e['cls'])
                        and box_iou(d['box'], e['box']) > .1
                        for d in o['dets'] for k in output for e in k['dets'])
        if not explained:
            output.append(copy.deepcopy(o))
    return output


def needs_band(dets):
    """Cheap acquisition policy based ONLY on 960 detections, not GT or BAND.

    Zoom for a small candidate (<80px) or a weak candidate (<.45) whose centre
    lies in the fixed horizon band. This may miss completely unproposed objects;
    full-sequence experiments must quantify that cost instead of hiding it.
    """
    return any(850 <= (d['box'][1]+d['box'][3])/2 <= 1250 and
               (d['box'][3]-d['box'][1] < 80 or d['conf'] < .45) for d in dets)
