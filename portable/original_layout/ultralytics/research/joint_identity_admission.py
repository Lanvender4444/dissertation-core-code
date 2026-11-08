"""Complete-link source-exclusive fusion with explicit joint-state features.

No GT at inference, no self-certified metric depth, no alternate bbox bag.
Finite cuboid states are an approximate model, not a continuous guarantee.
"""
import itertools
import numpy as np
import appearance_quality as A
import cam15_scoring as Q
import shared_geometry as S

BASE_DIM=1063
DIM=1071


def pair_map(pairs,probabilities):
    if len(pairs)!=len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid identity scores')
    result={}
    for (a,b),p in zip(pairs,probabilities):
        key=tuple(sorted((int(a),int(b))))
        if key[0]==key[1] or key in result or not 0<=p<=1:raise ValueError('Reused/invalid pair')
        result[key]=float(p)
    return result


def overlap(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    area=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return float(area/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area,1e-9))


def partition(pool,probabilities,tau=.75,margin=.1,duplicate_tau=.9):
    """Canonical same-view measurement + complete-link inter-view groups.

    Ambiguous competing identities stay separate; no forced null attachment.
    All unknown/singletons remain eligible for the subsequent quality head.
    """
    if not 0<=tau<=1 or not 0<=margin<=1 or not 0<=duplicate_tau<=1:raise ValueError('Invalid thresholds')
    if any(a not in pool or b not in pool for a,b in probabilities):raise ValueError('Unknown source')
    score=lambda a,b:probabilities.get(tuple(sorted((a,b))),0.)
    order=sorted(pool,key=lambda u:(-pool[u]['conf'],u));active=[];retired={}
    for u in order:
        v=next((v for v in active if pool[v]['src']==pool[u]['src'] and
            overlap(pool[v]['box'],pool[u]['box'])>=.25 and score(u,v)>=duplicate_tau),None)
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
        if any(score(u,v)<tau for u in ga for v in gb):
            rejected['complete_link']+=1;continue
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
    groups=sorted(clusters.values(),key=lambda g:g[0])
    flat=[u for g in groups for u in g]
    if len(flat)!=len(set(flat)) or set(flat)|set(retired)!=set(pool):raise ValueError('Source partition loss/reuse')
    if any(len(g)!=len({pool[u]['src'] for u in g}) for g in groups):raise ValueError('Alternative bbox bag')
    return groups,dict(retired_same_camera=retired,rejected_edges=rejected)


def shared_state(group,pool,cameras):
    """Compare one common finite cuboid state with independent per-view fits.

    Unknown has its own flag; a failed prior/centre model is not nonexistence.
    This is an observation-model feature, NOT an independent stereo check.
    """
    ds=[pool[u] for u in group];obj=dict(dets=ds,src=[d['src'] for d in ds],cls=max(ds,key=lambda d:(d['conf'],-d['uid']))['cls'])
    usable=[d for d in ds if d['geometry_valid']]
    if len(usable)<2:return np.zeros(8)
    grid=S.state_grid(obj,cameras,depth_samples=4)
    if grid is None:return np.zeros(8)
    costs=[]
    for d in usable:
        boxes,valid=S.predicted_boxes(grid,cameras[d['src']]);costs.append(S.box_cost(boxes,d,valid))
    costs=np.asarray(costs);joint=costs.mean(axis=0);k=int(np.argmin(joint))
    okay=joint<S.BIG/2
    if not okay.any():return np.r_[0.,len(usable)/5,np.zeros(6)]
    best=float(joint[k]);separate=float(np.mean(costs.min(axis=1)))
    feasible=okay&(joint<=best+.15);ranges=np.asarray(grid['ranges'])[feasible]
    return np.asarray([1.,len(usable)/5,min(best,20.),min(separate,20.),
        min(max(best-separate,0.),20.),float(feasible.mean()),
        np.log1p(grid['ranges'][k])/6,np.log(max(ranges)/min(ranges))/6])


def features(group,pool,cameras,offsets,descriptors,crop_flags,probabilities):
    ds=[pool[u] for u in group]
    if not ds or len(ds)!=len({d['src'] for d in ds}):raise ValueError('Need one active bbox per view')
    chosen=max(ds,key=lambda d:(d['conf'],-d['uid']))
    obj=dict(dets=ds,src=[d['src'] for d in ds],cls=chosen['cls'])
    base=A.features(Q.object_features(obj,[],[],cameras)['native'],ds,offsets,conditional=True)
    roi=np.asarray([descriptors[str(u)]['x'] for u in group],float)
    if roi.shape!=(len(ds),384) or not np.isfinite(roi).all():raise ValueError('Invalid ROI evidence')
    ps=[probabilities.get(tuple(sorted((a,b))),0.) for a,b in itertools.combinations(group,2)]
    margins=[]
    for a,b in itertools.combinations(group,2):
        for u,v in ((a,b),(b,a)):
            alternative=[probabilities.get(tuple(sorted((u,z))),0.) for z in pool if z not in group and pool[z]['src']==pool[v]['src']]
            margins.append(probabilities.get(tuple(sorted((u,v))),0.)-max(alternative,default=0.))
    pair=np.asarray([len(ps)/10,np.mean(ps),min(ps),max(ps),np.std(ps),min(margins),np.mean(margins),np.mean(np.asarray(margins)<.1)]) if ps else np.zeros(8)
    flags=np.asarray([crop_flags[u] for u in group],float)
    x=np.r_[base,roi.mean(axis=0),roi.std(axis=0),float(np.mean([descriptors[str(u)]['valid'] for u in group])),
        pair,flags.mean(axis=0),flags.max(axis=0)]
    result=np.r_[x,shared_state(group,pool,cameras)]
    if result.shape!=(DIM,) or not np.isfinite(result).all():raise ValueError('Invalid joint evidence dimension')
    return result
