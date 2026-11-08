"""Residual existence/novelty features; no truth and no forced identity fusion."""
import numpy as np

import appearance_quality as A
import cam15_scoring as Q
import residual_recovery as R


def features(d,seed_objects,unknown,cameras,offsets):
    base = Q.object_features(dict(src=[d['src']],cls=d['cls'],dets=[d]),[],[],cameras)['native']
    appearance = A.features(base,[d],offsets,conditional=True)
    context = R.features(d,seed_objects,unknown,cameras)[41:]
    return np.r_[appearance,context]  # 285; observations only


def select(seed_groups,group_scores,orphans,residual_scores,group_tau,residual_tau):
    if not (0<=group_tau<=1 and 0<=residual_tau<=1):
        raise ValueError('Invalid threshold')
    if len(seed_groups)!=len(group_scores) or len(orphans)!=len(residual_scores):
        raise ValueError('Score count mismatch')
    if not np.isfinite(group_scores).all() or not np.isfinite(residual_scores).all():
        raise ValueError('Nonfinite score')
    all_seed = [u for g in seed_groups for u in g]
    if len(all_seed)!=len(set(all_seed)) or len(orphans)!=len(set(orphans)) or set(all_seed)&set(orphans):
        raise ValueError('Source reused across hypotheses')
    # Rejected groups are NOT silently flattened into alternative singleton boxes.
    selected = [list(g) for g,p in zip(seed_groups,group_scores) if p>=group_tau]
    selected.extend([u] for u,p in zip(orphans,residual_scores) if p>=residual_tau)
    return selected
