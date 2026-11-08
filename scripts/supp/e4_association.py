"""E4: cross-view association accuracy using annotation instance IDs as an identity proxy.
Members of every output group are matched one-to-one, per camera, to that camera's GT boxes
(compatible class, IoU>=0.5, all heights). Pairwise metrics then follow the usual definition:
predicted pair (same group, different cameras) correct iff both members carry the same ID."""
import itertools
from collections import defaultdict
from supp_lib import *
from e2_simple_baselines import ray_nms

seg = segments()
OUT = new_run('e4_association', dict(
    purpose='Topic 3 Q2: is the same object identified across views? pairwise association precision/recall',
    identity_proxy='supplied XML instanceId (cross-camera semantics unconfirmed by provider)',
    member_labelling='per-camera one-to-one Hungarian over all output members vs own-camera GT, compatible class, IoU>=0.5, all heights',
    pair_definitions=dict(tp='same group, different cameras, same ID', fp='same group, different cameras, both labelled, different ID',
                          unknown='same group, at least one member unlabelled', fn='different groups, different cameras, same ID'),
    configs=['A', 'F5P', 'F5R', 'B2 best-F1 (theta5,tau.2)', 'B3p dev-P90 (theta5,tau.4,tau_s.9)']))


def label_members(groups, F):
    """uid -> iid via per-camera one-to-one matching of all members"""
    lab = {}
    by_cam = defaultdict(list)
    for g in groups:
        for u in g:
            by_cam[F['dets'][u]['src']].append(u)
    for c, us in by_cam.items():
        gts = F['gt'].get(c, [])
        if not gts:
            continue
        S = np.zeros((len(us), len(gts)))
        for i, u in enumerate(us):
            d = F['dets'][u]
            for j, (iid, l, b) in enumerate(gts):
                if cls_match(l, d['ccls']):
                    S[i, j] = iou(d['cbox'], b)
        rr, cc = linear_sum_assignment(-((S >= .5)*1000+S))
        for r, cc_ in zip(rr, cc):
            if S[r, cc_] >= .5:
                lab[us[r]] = gts[cc_][0]
    return lab


def pair_stats(groups, F):
    lab = label_members(groups, F); cam = lambda u: F['dets'][u]['src']
    st = defaultdict(lambda: np.zeros(4, int))  # tp, fp, unk, fn per camera pair
    grp = np.zeros(4, int)                       # multi-view groups: n, all-labelled, pure, impure
    gid = {u: k for k, g in enumerate(groups) for u in g}
    for g in groups:
        if len({cam(u) for u in g}) < 2:
            continue
        grp[0] += 1; ids = [lab.get(u) for u in g]
        if all(i is not None for i in ids):
            grp[1] += 1; grp[2] += len(set(ids)) == 1; grp[3] += len(set(ids)) > 1
        for a, b in itertools.combinations(g, 2):
            if cam(a) == cam(b):
                continue
            key = tuple(sorted((cam(a), cam(b))))
            la, lb = lab.get(a), lab.get(b)
            if la is None or lb is None:
                st[key][2] += 1
            elif la == lb:
                st[key][0] += 1
            else:
                st[key][1] += 1
    labelled = [u for u in lab]
    for a, b in itertools.combinations(labelled, 2):
        if lab[a] == lab[b] and cam(a) != cam(b) and gid[a] != gid[b]:
            st[tuple(sorted((cam(a), cam(b))))][3] += 1
    return st, grp


def config_groups(s, name):
    F = frame(s); C = configs(s)
    if name in C:
        return C[name], F
    if name == 'B2':
        return ray_nms(F, 5.0, 0.2), F
    if name == 'B3p':
        gs = ray_nms(F, 5.0, 0.4)
        return [g for g in gs if len(g) >= 2 or F['dets'][g[0]]['cconf'] >= 0.9], F


def summarise(st, grp):
    tot = sum(st.values(), np.zeros(4, int)); tp, fp, unk, fn = (int(x) for x in tot)
    P = tp/max(tp+fp, 1); R = tp/max(tp+fn, 1)
    pairs = {f'{a}-{b}': dict(tp=int(v[0]), fp=int(v[1]), unknown=int(v[2]), fn=int(v[3]),
                              precision=v[0]/max(v[0]+v[1], 1), recall=v[0]/max(v[0]+v[3], 1))
             for (a, b), v in sorted(st.items())}
    fish = sum((v for k, v in st.items() if (k[0] in (1, 5)) != (k[1] in (1, 5))), np.zeros(4, int))
    fwd = sum((v for k, v in st.items() if k[0] in (2, 3, 4) and k[1] in (2, 3, 4)), np.zeros(4, int))
    agg = lambda v: dict(tp=int(v[0]), fp=int(v[1]), unknown=int(v[2]), fn=int(v[3]),
                         precision=v[0]/max(v[0]+v[1], 1), recall=v[0]/max(v[0]+v[3], 1))
    return dict(tp=tp, fp=fp, unknown=unk, fn=fn, precision=P, recall=R, f1=2*P*R/max(P+R, 1e-12),
                evaluable_fraction=(tp+fp)/max(tp+fp+unk, 1),
                groups=dict(multiview=int(grp[0]), all_labelled=int(grp[1]), pure=int(grp[2]), impure=int(grp[3]),
                            purity=grp[2]/max(grp[1], 1)),
                forward_pairs=agg(fwd), pinhole_fisheye_pairs=agg(fish), by_camera_pair=pairs)


if __name__ == '__main__':
    res = {}
    for segname in ('full', 'guard_late'):
        res[segname] = {}
        for name in ('A', 'F5P', 'F5R', 'B2', 'B3p'):
            st = defaultdict(lambda: np.zeros(4, int)); grp = np.zeros(4, int)
            for s in seg[segname]:
                g, F = config_groups(s, name); a, b = pair_stats(g, F)
                for k, v in a.items():
                    st[k] += v
                grp += b
            res[segname][name] = summarise(st, grp)
            r = res[segname][name]
            print(segname, name, f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} eval={r['evaluable_fraction']:.3f}",
                  'fwd', round(r['forward_pairs']['precision'], 4), round(r['forward_pairs']['recall'], 4),
                  'pin-fish', r['pinhole_fisheye_pairs']['tp'], r['pinhole_fisheye_pairs']['fp'], r['pinhole_fisheye_pairs']['fn'],
                  'purity', round(r['groups']['purity'], 4), r['groups'], flush=True)
    dump(OUT/'results.json', res)
