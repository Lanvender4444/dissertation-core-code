"""E4-b: stratified blind sample of F5-R multi-view groups (guard+late) for visual identity audit."""
import random
from supp_lib import *
from e4_association import label_members
seg = segments(); rng = random.Random(20261101)
pure, impure, allmv = [], [], []
for s in seg['guard_late']:
    F = frame(s); G = configs(s)['F5R']; lab = label_members(G, F)
    for k, g in enumerate(G):
        if len({F['dets'][u]['src'] for u in g}) < 2: continue
        ids = [lab.get(u) for u in g]
        rec = dict(frame=s, group=k, members=[dict(uid=u, src=F['dets'][u]['src'], box=F['dets'][u]['cbox'], cls=F['dets'][u]['ccls'], gt_id=lab.get(u)) for u in g])
        allmv.append(rec)
        if all(i is not None for i in ids): (pure if len(set(ids)) == 1 else impure).append(rec)
S1 = rng.sample(pure, 40); S2 = impure[:]; S3 = rng.sample(allmv, 50)
cases = [dict(stratum=n, **r) for n, L in (('S1_pure', S1), ('S2_impure', S2), ('S3_random', S3)) for r in L]
rng.shuffle(cases)
for i, c in enumerate(cases): c['case'] = f'c{i+1:03d}'
out = new_run('e4_identity_audit', dict(purpose='blind visual identity audit of F5-R multi-view groups (guard+late)',
    strata=dict(S1_pure=f'40 of {len(pure)} annotation-pure groups', S2_impure=f'all {len(impure)} annotation-impure groups',
                S3_random=f'50 of {len(allmv)} multi-view groups'), seed=20261101,
    rendering='member crops padded 30%, no IDs/strata on sheets', labels='same / different / uncertain'))
dump(out/'cases.json', cases)
print(len(pure), len(impure), len(allmv), len(cases))
