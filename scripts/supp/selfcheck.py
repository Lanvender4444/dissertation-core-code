"""Reproduce frozen F5-P and F5-R counts frame by frame and cache per-frame objects for later experiments."""
import pickle, time
from common import *
seg = segments(); f5p = load_f5p(); tau = f5r_tau()
out = new_run('selfcheck', dict(purpose='reproduce F5-P 10606/1120/14239 and F5-R 11143/1124/13702', tau=tau), exist_ok=True)
tot = {k: np.zeros(3, int) for k in ('f5p', 'f5r', 'a', 'repair')}; bad = []; cache = {}; t = time.time()
for k, s in enumerate(seg['full']):
    g = gt(s); changed = raw_pool(s, corrected=True)
    p_objs = f5p[s]['objects']; cp = evaluate(p_objs, g)[:3]
    if cp.tolist() != f5p[s]['counts'][:3]: bad.append(('f5p', s))
    r_objs, counts = f5r_objects(s, tau, changed); cr = evaluate(r_objs, g)[:3]
    if cr.tolist() != counts['residual/P0.91']: bad.append(('f5r', s, cr.tolist(), counts['residual/P0.91']))
    if cp.tolist() != counts['baseline']: bad.append(('base', s))
    a_objs = [o for o in p_objs if not o['recovered']]; ca = evaluate(a_objs, g)[:3]
    rep = [o for o in r_objs if not o['recovered']]; crep = evaluate(rep, g)[:3]
    if crep.tolist() != counts['repair_only']: bad.append(('repair', s))
    tot['f5p'] += cp; tot['f5r'] += cr; tot['a'] += ca; tot['repair'] += crep
    cache[s] = dict(f5p=p_objs, f5r=r_objs, counts=dict(f5p=cp.tolist(), f5r=cr.tolist(), a=ca.tolist(), repair=crep.tolist()))
    if k % 200 == 0: print(k, {a: b.tolist() for a, b in tot.items()}, len(bad), round(time.time()-t), flush=True)
with open(out/'objects_cache.pkl', 'wb') as f: pickle.dump(cache, f)
res = dict(totals={a: prf(b) for a, b in tot.items()}, mismatches=bad[:50], n_mismatch=len(bad), tau=tau)
dump(out/'results.json', res); print(json.dumps(res['totals'], indent=1), len(bad))
