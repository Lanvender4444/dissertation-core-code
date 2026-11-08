"""Independent measurement-policy counterfactual; old frozen module unchanged.

Only canonical same-camera overlap gate changes. No scorer, GT or bbox choice
oracle. Refusing duplicate retirement keeps two distinct output hypotheses,
never two alternative boxes for one camera inside an object.
"""
import itertools
import numpy as np
from joint_identity_admission import overlap


def partition(pool,probabilities,duplicate_iou=.25,tau=.75,margin=.1,duplicate_tau=.9):
    if duplicate_iou is not None and not 0<=duplicate_iou<=1:raise ValueError('Invalid overlap gate')
    if not 0<=tau<=1 or not 0<=margin<=1 or not 0<=duplicate_tau<=1:raise ValueError('Invalid thresholds')
    if any(a not in pool or b not in pool for a,b in probabilities):raise ValueError('Unknown source')
    if not np.isfinite(list(probabilities.values())).all():raise ValueError('Nonfinite scores')
    score=lambda a,b:probabilities.get(tuple(sorted((a,b))),0.)
    order=sorted(pool,key=lambda u:(-pool[u]['conf'],u));active=[];retired={}
    for u in order:
        v=next((v for v in active if duplicate_iou is not None and pool[v]['src']==pool[u]['src'] and
            overlap(pool[v]['box'],pool[u]['box'])>=duplicate_iou and score(u,v)>=duplicate_tau),None)
        if v is None:active.append(u)
        else:retired[u]=v
    clusters={u:[u] for u in active};owner={u:u for u in active}
    edges=sorted(((p,a,b) for (a,b),p in probabilities.items() if a in owner and b in owner and
        pool[a]['src']!=pool[b]['src'] and p>=tau),key=lambda v:(-v[0],v[1],v[2]))
    rejected=dict(camera_collision=0,complete_link=0,competition=0)
    for _,a,b in edges:
        ia,ib=owner[a],owner[b]
        if ia==ib:continue
        ga,gb=clusters[ia],clusters[ib]
        if {pool[u]['src'] for u in ga}&{pool[u]['src'] for u in gb}:
            rejected['camera_collision']+=1;continue
        if any(score(u,v)<tau for u in ga for v in gb):rejected['complete_link']+=1;continue
        union=set(ga+gb);ambiguous=False
        for u,v in itertools.product(ga,gb):
            for x,y in ((u,v),(v,u)):
                competitors=[z for z in active if z not in union and pool[z]['src']==pool[y]['src']]
                if competitors and max(score(x,z) for z in competitors)>score(x,y)-margin:
                    ambiguous=True;break
            if ambiguous:break
        if ambiguous:rejected['competition']+=1;continue
        root=min(ia,ib);merged=sorted(ga+gb)
        del clusters[ia];del clusters[ib];clusters[root]=merged
        for u in merged:owner[u]=root
    groups=sorted(clusters.values(),key=lambda g:g[0]);flat=[u for g in groups for u in g]
    if len(flat)!=len(set(flat)) or set(flat)|set(retired)!=set(pool):raise ValueError('Source loss/reuse')
    if any(len(g)!=len({pool[u]['src'] for u in g}) for g in groups):raise ValueError('Alternate bbox bag')
    return groups,dict(retired_same_camera=retired,rejected_edges=rejected)
