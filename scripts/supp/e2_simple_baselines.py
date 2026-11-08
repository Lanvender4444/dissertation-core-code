"""E2: simple fusion baselines on the identical box-corrected five-camera pool.
Selection rule (development 200 frames only): max recall subject to precision>=0.90;
if infeasible, record null and also lock the best-F1 point. Then replay full and guard+late."""
import itertools
from supp_lib import *

seg = segments()
TAUS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
THETAS = [0.5, 1.0, 2.0, 3.0, 5.0]
TAU_S = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
COMPAT = {0: 'p', 1: 'b', 3: 'b', 2: 'v', 5: 'v', 7: 'v'}
OUT = new_run('e2_simple_baselines', dict(
    purpose='Compare F5-P/F5-R with simple post-detection fusion on identical input (box-corrected .05 crop pool)',
    baselines=dict(B1='each detection an object, conf>=tau',
                   B2='greedy confidence-ordered cross-camera ray NMS: same-camera compatible IoU>=0.5 suppressed; '
                      'other-camera compatible detection with centre-ray angle<=theta joins as member (one per camera)',
                   B3='B2 groups kept only with >=2 supporting cameras',
                   B3p='B2 groups kept if >=2 cameras OR single-view conf>=tau_s'),
    grids=dict(tau=TAUS, theta=THETAS, tau_s=TAU_S),
    selection='development only: max recall s.t. P>=0.90; else null + best-F1 point; locked before full/guard+late replay',
    rays='native-box centre rays in the shared IMU frame (checked: matched F5-P members median 0.5-1.5 deg)'))


def angle(a, b):
    ra, rb = np.asarray(a, float), np.asarray(b, float)
    return np.degrees(np.arccos(np.clip(ra@rb/(np.linalg.norm(ra)*np.linalg.norm(rb)+1e-12), -1, 1)))


def ray_nms(F, theta, tau):
    ds = sorted([d for d in F['dets'].values() if d['cconf'] >= tau], key=lambda d: (-d['cconf'], d['uid']))
    used = set(); groups = []
    for i, d in enumerate(ds):
        if i in used:
            continue
        used.add(i); members = {d['src']: d}
        for j in range(i+1, len(ds)):
            e = ds[j]
            if j in used or COMPAT.get(e['ccls']) != COMPAT.get(d['ccls']):
                continue
            if e['src'] == d['src']:
                if iou(d['cbox'], e['cbox']) >= .5:
                    used.add(j)
            elif e['src'] not in members and d['ray'] and e['ray'] and angle(d['ray'], e['ray']) <= theta:
                members[e['src']] = e; used.add(j)
        groups.append([m['uid'] for m in members.values()])
    return groups


def variants(F):
    """yield (name, params, groups) for all grid points of one frame"""
    for t in TAUS:
        yield 'B1', (t,), [[u] for u, d in F['dets'].items() if d['cconf'] >= t]
    for th in THETAS:
        for t in TAUS:
            gs = ray_nms(F, th, t)
            yield 'B2', (th, t), gs
            multi = [g for g in gs if len(g) >= 2]
            yield 'B3', (th, t), multi
            for ts in TAU_S:
                if ts > t:
                    yield 'B3p', (th, t, ts), [g for g in gs if len(g) >= 2 or F['dets'][g[0]]['cconf'] >= ts]


def sweep(frames):
    tot = {}; per = {}
    for s in frames:
        F = frame(s)
        for name, p, gs in variants(F):
            c = evaluate(as_objects(gs, F), F['gt'])
            tot.setdefault((name, p), np.zeros(3, int))
            tot[(name, p)] += c
            per.setdefault((name, p), []).append(c.tolist())
    return tot, per


def select(tot):
    lock = {}
    for name in ('B1', 'B2', 'B3', 'B3p'):
        cand = [(p, prf(c)) for (n, p), c in tot.items() if n == name]
        feas = [x for x in cand if x[1]['precision'] >= .90]
        lock[name] = dict(p90=max(feas, key=lambda x: (x[1]['recall'], x[1]['precision']))[0] if feas else None,
                          best_f1=max(cand, key=lambda x: x[1]['f1'])[0])
    return lock


if __name__ == '__main__':
    dev_tot, _ = sweep(seg['development'])
    lock = select(dev_tot)
    dump(OUT/'development_lock.json', dict(lock={k: dict(p90=v['p90'], best_f1=v['best_f1'],
        p90_dev=prf(dev_tot[(k, v['p90'])]) if v['p90'] else None, best_f1_dev=prf(dev_tot[(k, v['best_f1'])]))
        for k, v in lock.items()}, grid=[dict(name=n, params=p, **prf(c)) for (n, p), c in dev_tot.items()]))
    print(json.dumps({k: v for k, v in lock.items()}), flush=True)
    wanted = {(k, v[r]) for k, v in lock.items() for r in ('p90', 'best_f1') if v[r] is not None}
    results = {}
    for segname in ('full', 'guard_late', 'development'):
        rows = {}; perframe = {}
        for s in seg[segname]:
            F = frame(s); C = configs(s)
            for n in ('raw', 'A', 'F5P', 'F5R'):
                c = evaluate(as_objects(C[n], F), F['gt'])
                rows.setdefault((n, ()), np.zeros(3, int)); rows[(n, ())] += c
                perframe.setdefault(f'{n}', []).append(c.tolist())
            for name, p in wanted:
                if name == 'B1':
                    gs = [[u] for u, d in F['dets'].items() if d['cconf'] >= p[0]]
                else:
                    gs = ray_nms(F, p[0], p[1])
                    if name == 'B3':
                        gs = [g for g in gs if len(g) >= 2]
                    if name == 'B3p':
                        gs = [g for g in gs if len(g) >= 2 or F['dets'][g[0]]['cconf'] >= p[2]]
                c = evaluate(as_objects(gs, F), F['gt'])
                rows.setdefault((name, p), np.zeros(3, int)); rows[(name, p)] += c
                perframe.setdefault(f'{name}{list(p)}', []).append(c.tolist())
        boot = {}
        for k, v in perframe.items():
            if k not in ('F5R', 'F5P'):
                boot[k] = dict(F5R_minus_this=block_bootstrap(v, perframe['F5R']))
        results[segname] = dict(rows=[dict(method=n, params=list(p), **prf(c)) for (n, p), c in rows.items()],
                                bootstrap=boot)
        print(segname, flush=True)
        for r in results[segname]['rows']:
            print(f"  {r['method']:5s} {r['params']} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}", flush=True)
    dump(OUT/'results.json', dict(lock={k: v for k, v in lock.items()}, segments=results))

    # PR curves (full sequence) over tau for B1 and best-theta B2, development-locked theta
    curves = {}
    th = lock['B2']['p90'][0] if lock['B2']['p90'] else lock['B2']['best_f1'][0]
    for t in TAUS:
        tb1 = np.zeros(3, int); tb2 = np.zeros(3, int)
        for s in seg['full']:
            F = frame(s)
            tb1 += evaluate(as_objects([[u] for u, d in F['dets'].items() if d['cconf'] >= t], F), F['gt'])
            tb2 += evaluate(as_objects(ray_nms(F, th, t), F), F['gt'])
        curves.setdefault('B1', []).append(dict(tau=t, **prf(tb1)))
        curves.setdefault(f'B2_theta{th}', []).append(dict(tau=t, **prf(tb2)))
    dump(OUT/'pr_curves.json', curves)
    print('curves done')
