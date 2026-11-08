"""E3-b: output-level contribution of the lateral fisheye cameras (no re-fitting).
Remove every Cam1/Cam5 member from frozen outputs; drop objects left empty; evaluate on the
five-view union and on the forward Cam234 union. Also report fisheye-only objects."""
from supp_lib import *
seg = segments()
OUT = new_run('e3b_fisheye_output', dict(purpose='fisheye contribution by removing Cam1/Cam5 members from frozen F5-P/F5-R outputs',
    note='output-level removal, not a system re-trained without fisheye cameras'))
res = {}
for segname in ('full', 'guard_late'):
    r = {}
    for name in ('F5P', 'F5R'):
        t = {k: np.zeros(3, int) for k in ('five_all', 'five_nofish', 'fwd_all', 'fwd_nofish')}; fish = np.zeros(2, int)
        for s in seg[segname]:
            F = frame(s); G = configs(s)[name]
            nof = [[u for u in g if F['dets'][u]['src'] not in (1, 5)] for g in G]; nof = [g for g in nof if g]
            g5 = F['gt']; g3 = gt_subset(g5, (2, 3, 4))
            t['five_all'] += evaluate(as_objects(G, F), g5); t['five_nofish'] += evaluate(as_objects(nof, F), g5)
            t['fwd_all'] += evaluate(as_objects(G, F), g3); t['fwd_nofish'] += evaluate(as_objects(nof, F), g3)
            fo = [g for g in G if all(F['dets'][u]['src'] in (1, 5) for u in g)]
            c, pairs, union, ign = evaluate(as_objects(G, F), g5, return_assignment=True)
            idx = {i for i, g in enumerate(G) if all(F['dets'][u]['src'] in (1, 5) for u in g)}
            tpi = {i for i, _ in pairs}
            fish += [len(idx & tpi), len([i for i in idx if i not in ign])]
        r[name] = {k: prf(v) for k, v in t.items()}
        r[name]['fisheye_only_objects'] = dict(tp=int(fish[0]), counted=int(fish[1]), precision=fish[0]/max(fish[1], 1))
        print(segname, name, {k: (round(v['precision'], 4), round(v['recall'], 4)) for k, v in r[name].items() if k != 'fisheye_only_objects'}, r[name]['fisheye_only_objects'], flush=True)
    res[segname] = r
dump(OUT/'results.json', res)
