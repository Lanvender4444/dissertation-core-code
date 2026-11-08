"""GT-free reference assembly; protect outputs, not claimed physical truth."""
import numpy as np
import retained_reference as R
import residual_recovery as C
import nonlinear_admission as N

DIM=672


def objects(groups,pool):
    return [dict(src=sorted({pool[u]['src'] for u in g}),cls=pool[g[0]]['cls'],dets=[pool[u] for u in g]) for g in groups]


def validate(groups,pool):
    flat=[u for g in groups for u in g]
    if any(not g for g in groups) or len(flat)!=len(set(flat)) or not set(flat)<=set(pool):
        raise ValueError('Empty/reused/unknown source')
    if any(len(g)!=len({pool[u]['src'] for u in g}) for g in groups):
        raise ValueError('Each object must have one active bbox per view')
    return set(flat)


def contextualize(uids,base_x,candidates,kept,pool,unknown,cameras):
    """Only seven emitted-context features change; original 665 kept EXACT.

    No GT arguments or fitted scores. The baseline feature schema is frozen.
    """
    uids=list(uids);x=np.asarray(base_x,float)
    if x.shape!=(len(uids),DIM) or not np.isfinite(x).all() or len(uids)!=len(set(uids)):
        raise ValueError('Invalid original node features')
    used=validate(kept,pool)
    if set(candidates)&used or not set(candidates)<=set(uids) or len(candidates)!=len(set(candidates)):
        raise ValueError('Already emitted/reused/unknown candidate')
    rows={u:i for i,u in enumerate(uids)}
    out=x[[rows[u] for u in candidates]].copy();refs=objects(kept,pool)
    for i,u in enumerate(candidates):out[i,278:285]=C.features(pool[u],refs,unknown,cameras)[41:]
    return out.reshape(len(candidates),DIM)


def candidate_features(seed,group_scores,uids,base_x,pool,unknown,cameras,group_tau=.7):
    kept,candidates,roles=R.partition(seed,group_scores,pool,group_tau,recycle=True)
    x=contextualize(uids,base_x,candidates,kept,pool,unknown,cameras)
    if len(candidates) and (not np.all(x[:,670]==group_tau) or any(x[i,671]!=float(roles[u]=='recycled_rejected_group') for i,u in enumerate(candidates))):
        raise ValueError('Changed baseline threshold/role schema')
    return kept,candidates,x


def assemble(seed,group_scores,uids,base_x,pool,unknown,cameras,residual_model,group_tau=.7,residual_tau=.55):
    if not 0<=residual_tau<=1:raise ValueError('Invalid residual threshold')
    kept,candidates,x=candidate_features(seed,group_scores,uids,base_x,pool,unknown,cameras,group_tau)
    scores=N.predict(x,residual_model).tolist()
    reference=kept+[[u] for u,p in zip(candidates,scores) if p>=residual_tau]
    validate(reference,pool)
    return dict(reference=reference,kept=kept,candidates=candidates,scores=scores)


def append(reference,extra,pool):
    """Preserve all old group memberships/boxes; no silently hidden additions."""
    used=validate(reference,pool);new=validate(extra,pool)
    if used&new:raise ValueError('Observation already protected')
    output=[list(g) for g in reference]+[list(g) for g in extra]
    validate(output,pool)
    return output
