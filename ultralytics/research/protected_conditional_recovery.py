"""Source-exclusive recovery conditioned on an immutable accepted scene.

No GT inputs. Unknown location is not an automatic rejection. Standard pair
scores/finite cuboid features are not physical identity or metric certificates.
"""
import itertools
import numpy as np
import joint_identity_admission as J
import identity_measurement_policy as H
import protected_reference as P
import residual_recovery as C

CONTEXT_DIM=24
BASE_DIM=J.BASE_DIM+CONTEXT_DIM
DIM=BASE_DIM+8


def partition(reference,pool,probabilities,duplicate_iou=.7,tau=.75,margin=.1,duplicate_tau=.9):
    used=P.validate(reference,pool)
    remaining={u:d for u,d in pool.items() if u not in used}
    edges={k:v for k,v in probabilities.items() if all(u in remaining for u in k)}
    groups,trace=H.partition(remaining,edges,duplicate_iou,tau,margin,duplicate_tau)
    if P.validate(groups,pool)&used:raise ValueError('Recovery altered protected sources')
    return groups,trace


def context(group,reference,pool,unknown,cameras,probabilities):
    used=P.validate(reference,pool)
    P.validate([group],pool)
    if set(group)&used:raise ValueError('Own observation cannot validate its recovery')
    refs=P.objects(reference,pool)
    native=np.asarray([C.features(pool[u],refs,unknown,cameras)[41:] for u in group],float)
    pair=lambda a,b:probabilities.get(tuple(sorted((a,b))),0.)
    best=np.asarray([max((pair(u,v) for v in used),default=0.) for u in group])
    anchor=[min(max((pair(u,v) for v in g),default=0.) for u in group) for g in reference]
    ordered=sorted(anchor,reverse=True)
    possible=len(group)*len(used)
    available=sum(tuple(sorted((u,v))) in probabilities for u in group for v in used)
    extra=[max(best),np.mean(best),min(best),np.std(best),np.mean(best>=.75),np.mean(best>=.95),
        available/max(possible,1),max(anchor,default=0.),sum(v>=.75 for v in anchor)/30,
        ordered[0]-(ordered[1] if len(ordered)>1 else 0.) if ordered else 0.]
    x=np.r_[native.mean(axis=0),native.max(axis=0),extra]
    if x.shape!=(CONTEXT_DIM,) or not np.isfinite(x).all():raise ValueError('Invalid protected context')
    return x


def features(groups,reference,pool,cameras,offsets,descriptors,flags,probabilities,unknown):
    used=P.validate(reference,pool);new=P.validate(groups,pool)
    if used&new:raise ValueError('Nonexclusive additional evidence')
    rows=[]
    for g in groups:
        x=J.features(g,pool,cameras,offsets,descriptors,flags,probabilities)
        rows.append(np.r_[x[:J.BASE_DIM],context(g,reference,pool,unknown,cameras,probabilities),x[J.BASE_DIM:]])
    return np.asarray(rows,float).reshape(len(groups),DIM)


def conflicts(a,b,pool,probabilities):
    """Conservative batch duplicate hypothesis, not a guaranteed identity.

    A shared camera requires high overlap AND every available endpoint pair to
    support duplication; no shared view needs an even stricter score. Unknown
    pairs give zero, so they cannot cause retirement. No centre-depth veto.
    """
    by_a={pool[u]['src']:u for u in a};by_b={pool[u]['src']:u for u in b}
    common=set(by_a)&set(by_b)
    if common and any(J.overlap(pool[by_a[c]]['box'],pool[by_b[c]]['box'])<.7 for c in common):return False
    threshold=.95 if common else .99
    return all(probabilities.get(tuple(sorted((u,v))),0.)>=threshold for u,v in itertools.product(a,b))


def select(reference,groups,scores,tau,pool,probabilities,suppress=False):
    if not 0<=tau<=1 or len(scores)!=len(groups) or not np.isfinite(scores).all() or any(not 0<=p<=1 for p in scores):raise ValueError('Invalid selection')
    P.validate(groups,pool);used=P.validate(reference,pool)
    if used&{u for g in groups for u in g}:raise ValueError('Recovery reuses protected source')
    ranked=sorted(range(len(groups)),key=lambda i:(-float(scores[i]),groups[i][0]))
    accepted=[];retired=[]
    for i in ranked:
        if scores[i]<tau:continue
        if suppress and any(conflicts(groups[i],g,pool,probabilities) for g in accepted):retired.append(i)
        else:accepted.append(groups[i])
    return P.append(reference,accepted,pool),dict(added=accepted,batch_retired=retired)
