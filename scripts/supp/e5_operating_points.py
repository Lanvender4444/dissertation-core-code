"""E5 operating points: recall at precision >= 0.90 and precision at the F5-R recall (44.85%), read from the
full ranked group-AP curves at IoU 0.50 (dissertation matching protocol, objects ranked by maximum member
confidence, ties in confidence forming one threshold). e5_ap.py keeps only ~200 points of each curve for
plotting, so the operating points are computed here from every event.

Usage: python e5_operating_points.py [CONFIG ...]   (default: all configs; existing entries of other configs are kept)
Writes ultralytics/experiments/supp_20261101/e5_ap/run01/operating_points.json."""
import ast, json, sys, time
from pathlib import Path
from supp_lib import *

HERE = Path(__file__).resolve().parent
ns = dict(globals())


def pull(fname, names):
    """Load selected definitions from a module without importing it (importing creates a new run directory)."""
    tree = ast.parse((HERE/fname).read_text(encoding='utf-8'))
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef) and node.name in names) or \
           (isinstance(node, ast.Assign) and any(getattr(t, 'id', None) in names for t in node.targets)):
            exec(compile(ast.Module([node], []), fname, 'exec'), ns)


pull('e2_simple_baselines.py', {'angle', 'ray_nms', 'COMPAT'})
pull('e5_ap.py', {'group_events', 'prefix_counts'})
GETTERS = {
    'F5P': lambda s, F: as_objects(configs(s)['F5P'], F),
    'F5R': lambda s, F: as_objects(configs(s)['F5R'], F),
    'A': lambda s, F: as_objects(configs(s)['A'], F),
    'B2': lambda s, F: as_objects(ns['ray_nms'](F, 5.0, 0.05), F),
    'B1': lambda s, F: as_objects([[u] for u in F['dets']], F),
}
P_TARGET, R_TARGET = 0.90, 0.4485


def operating_points(frames, getter):
    ev = []; npos = 0
    for s in frames:
        F = frame(s); objs = getter(s, F)
        S, Si, keys, union = ns['group_events'](objs, F['gt']); npos += len(keys)
        order = sorted(range(len(objs)), key=lambda i: -objs[i]['conf']); prev = (0, 0)
        for i, (tp, fp) in zip(order, ns['prefix_counts'](S, Si, order, 0.5)):
            ev.append((objs[i]['conf'], tp-prev[0], fp-prev[1])); prev = (tp, fp)
    ev.sort(key=lambda e: -e[0])
    pts = []; TP = FP = 0; k = 0
    while k < len(ev):
        c = ev[k][0]
        while k < len(ev) and ev[k][0] == c:
            TP += ev[k][1]; FP += ev[k][2]; k += 1
        pts.append((TP, FP))
    rec = lambda tp: tp/npos
    prec = lambda tp, fp: tp/max(tp+fp, 1e-9)
    return dict(npos=npos, thresholds=len(pts),
                R_at_P90=max((rec(tp) for tp, fp in pts if prec(tp, fp) >= P_TARGET), default=0.0),
                P_at_R4485=max((prec(tp, fp) for tp, fp in pts if rec(tp) >= R_TARGET), default=None),
                end_point=dict(tp=pts[-1][0], fp=pts[-1][1], recall=rec(pts[-1][0]), precision=prec(*pts[-1])))


if __name__ == '__main__':
    out_path = OUT_ROOT/'e5_ap/run01/operating_points.json'
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    seg = segments()
    for name in (sys.argv[1:] or list(GETTERS)):
        res[name] = {}
        for segname in ('full', 'guard_late'):
            t0 = time.time()
            res[name][segname] = operating_points(seg[segname], GETTERS[name])
            print(name, segname, {k: v for k, v in res[name][segname].items() if k != 'end_point'}, f'{time.time()-t0:.0f}s', flush=True)
    res['protocol'] = dict(iou=0.5, ranking='maximum member detector confidence (box-corrected)', ties='one threshold per confidence value',
                           R_at_P90=f'maximum recall over thresholds with precision >= {P_TARGET}',
                           P_at_R4485=f'maximum precision over thresholds with recall >= {R_TARGET}')
    dump(out_path, res)
