"""Resumable extraction of a compact frozen per-frame dataset (read-only) with count verification.
Usage: python3 extract.py [seconds_budget]"""
import sys, time
from common import *
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150
seg = segments(); tau = f5r_tau()
out = new_run('extract', dict(purpose='compact per-frame snapshot of frozen inputs/outputs for supplementary analysis',
    f5p_log=str(F5P_LOG.relative_to(ROOT)), repair=str(REPAIR.relative_to(ROOT)), tau=tau,
    fields='raw pool (conf>=.05) native+box-corrected boxes, rays, crop windows, edge flags; GT all cams; F5-P groups; F5-R root+paths'), exist_ok=True)
fdir = out/'frames'; fdir.mkdir(exist_ok=True)
f5p = load_f5p(); t0 = time.time(); done = 0
for s in seg['full']:
    p = fdir/f'{s}.json'
    if p.exists(): continue
    if time.time()-t0 > budget: break
    pool, changed, descriptor, flags, crops, _ = inputs().load(s)
    dets = {}
    for u, d in pool.items():
        dets[str(u)] = dict(src=int(d['src']), cls=int(d['cls']), conf=float(d['conf']), box=[float(x) for x in d['box']],
            cbox=[float(x) for x in changed[u]['box']], ccls=int(changed[u]['cls']), cconf=float(changed[u]['conf']),
            ray=[float(x) for x in np.asarray(d['ray']).ravel()] if d.get('geometry_valid') else None,
            geometry_valid=bool(d.get('geometry_valid')), angular_h=float(d.get('angular_h') or 0.),
            crop=[float(x) for x in np.asarray(crops[u]).ravel()], flags=[float(x) for x in np.asarray(flags[u]).ravel()])
    g = gt(s); gtall = {c: [[int(i), l, [float(x) for x in b]] for i, l, b in g[c]] for c in g}
    rec = f5r_record(s)
    objs_p = f5p[s]['objects']
    gp = [[int(m['uid']) for m in o['dets']] for o in objs_p]
    rp = [bool(o['recovered']) for o in objs_p]
    # verification against frozen counts
    cp = evaluate(objs_p, g)[:3].tolist()
    root, add = f5r_groups(rec, tau)
    chg = {u: dict(changed[u], uid=u) for u in changed}
    cr = evaluate(as_objects(root+add, chg), g)[:3].tolist()
    ok = cp == f5p[s]['counts'][:3] and cr == rec['counts']['residual/P0.91'] and cp == rec['counts']['baseline']
    # do F5-P members use corrected boxes?
    mbox = all(np.allclose(m['box'], changed[m['uid']]['box']) for o in objs_p for m in o['dets'])
    dump(p, dict(frame=s, dets=dets, gt=gtall, f5p_groups=gp, f5p_recovered=rp, f5p_member_box_is_corrected=mbox,
                 f5r_root=root, f5r_paths=rec['paths'], counts=rec['counts'], f5p_counts=cp, f5r_counts=cr, verified=ok))
    if not ok: print('MISMATCH', s, cp, cr, rec['counts'], flush=True)
    done += 1
n = len(list(fdir.glob('*.json')))
print(f'done_this_call={done} total={n}/1655 elapsed={time.time()-t0:.0f}s', flush=True)
