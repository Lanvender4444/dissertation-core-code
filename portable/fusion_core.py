"""Mechanically extracted frozen recovery/verification functions; see metadata/extractions.json."""

import itertools
from types import SimpleNamespace
import numpy as np
from scipy.special import expit
from scipy.optimize import linear_sum_assignment

def compile_model(model):
    return dict(dim=model['dim'],baseline=model['baseline'],trees=[{k:np.asarray(v,int if k in
        ('feature_idx','left','right','is_leaf') else float) for k,v in tree.items()} for tree in model['trees']])

def predict(xs,model):
    x = np.asarray(xs,float)
    if not len(x):
        return np.empty(0)
    if x.ndim!=2 or x.shape[1]!=model['dim'] or not np.isfinite(x).all():
        raise ValueError('Invalid portable predictor input')
    eta = np.full(len(x),model['baseline'],float)
    for tree in model['trees']:
        position = np.zeros(len(x),int)
        while True:
            active = np.flatnonzero(np.asarray(tree['is_leaf'])[position]==0)
            if not len(active):
                break
            indices = position[active]
            feature = np.asarray(tree['feature_idx'])[indices]
            left = x[active,feature]<=np.asarray(tree['num_threshold'])[indices]
            position[active] = np.where(left,np.asarray(tree['left'])[indices],np.asarray(tree['right'])[indices])
        eta += np.asarray(tree['value'])[position]
    return expit(eta)

def objects(groups,pool):
    return [dict(src=sorted({pool[u]['src'] for u in g}),cls=pool[g[0]]['cls'],dets=[pool[u] for u in g]) for g in groups]

def validate(groups,pool):
    flat=[u for g in groups for u in g]
    if any(not g for g in groups) or len(flat)!=len(set(flat)) or not set(flat)<=set(pool):
        raise ValueError('Empty/reused/unknown source')
    if any(len(g)!=len({pool[u]['src'] for u in g}) for g in groups):
        raise ValueError('Each object must have one active bbox per view')
    return set(flat)

def append(reference,extra,pool):
    """Preserve all old group memberships/boxes; no silently hidden additions."""
    used=validate(reference,pool);new=validate(extra,pool)
    if used&new:raise ValueError('Observation already protected')
    output=[list(g) for g in reference]+[list(g) for g in extra]
    validate(output,pool)
    return output

def overlap(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    area=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return float(area/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area,1e-9))

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

def counts(self, objects):
    scores = np.zeros((len(objects), len(self.keys)))
    ignored = wrong = multi = 0
    for i, o in enumerate(objects):
        if o['dets']:
            scores[i] = np.maximum.reduce([self.scores[d['uid']] for d in o['dets']])
        hits = int(np.sum(scores[i] >= .5))
        if len(o['src']) > 1:
            multi += 1
            wrong += hits > 1
        if not hits and any(d['uid'] in self.small_hits for d in o['dets']):
            ignored += 1
    rr, cc = linear_sum_assignment(-((scores >= .5)*1000+scores*(scores >= .5))) if scores.size else ([], [])
    tp = sum(scores[i, j] >= .5 for i, j in zip(rr, cc))
    return [int(tp), len(objects)-ignored-int(tp), len(self.keys)-int(tp), wrong, multi]

P=SimpleNamespace(validate=validate,append=append)
J=SimpleNamespace(overlap=overlap)
