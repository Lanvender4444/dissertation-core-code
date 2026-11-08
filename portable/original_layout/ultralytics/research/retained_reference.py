"""Residual reasoning conditioned on emitted groups, not unscored candidates.

Partition is GT-free; recycled observations remain unverified singleton
candidates. No rejected multi-view identity is silently asserted as correct.
"""
import numpy as np

import residual_admission as R
import detector_roi_admission as D


def partition(seed,scores,pool,tau,floor=.05,recycle=False):
    if not 0<=tau<=1 or not 0<=floor<=1 or len(seed)!=len(scores):
        raise ValueError('Invalid retained-reference partition')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite group scores')
    all_uids = [u for g in seed for u in g]
    if any(not g for g in seed) or len(all_uids)!=len(set(all_uids)) or not set(all_uids)<=set(pool):
        raise ValueError('Original groups reuse or lack source observations')
    if any(pool[u]['conf']<floor for u in all_uids):
        raise ValueError('Grouped observation below active floor')
    kept = [list(g) for g,p in zip(seed,scores) if p>=tau]
    used = {u for g in kept for u in g}
    original = set(all_uids)
    candidates = sorted(u for u,d in pool.items() if d['conf']>=floor and u not in used and (recycle or u not in original))
    roles = {u:('recycled_rejected_group' if u in original else 'original_orphan') for u in candidates}
    return kept,candidates,roles


def features(d,kept_objects,unknown,cameras,offsets,descriptors,group_tau,role):
    if any(d['uid']==a['uid'] for o in kept_objects for a in o['dets']):
        raise ValueError('Source already emitted')
    if role not in ('recycled_rejected_group','original_orphan') or not 0<=group_tau<=1:
        raise ValueError('Invalid candidate provenance')
    base = R.features(d,kept_objects,unknown,cameras,offsets)
    x = D.augment(base,[d],descriptors,False)
    return np.r_[x,group_tau,float(role=='recycled_rejected_group')]  # 672


def training_folds(names,n=5,gap=5):
    """Contiguous early-only cross-fitting, with a temporal exclusion gap."""
    names = list(names)
    if len(names)<n*2 or n<2 or gap<0 or len(names)!=len(set(names)) or names!=sorted(names):
        raise ValueError('Invalid ordered early folds')
    if any(not 0<=int(s)<=600 for s in names):
        raise ValueError('Cross-fitting must never read development/guard/late')
    out = []
    for held_idx in np.array_split(np.arange(len(names)),n):
        lo,hi = max(0,int(held_idx[0])-gap),min(len(names),int(held_idx[-1])+gap+1)
        held = [names[i] for i in held_idx]
        train = names[:lo]+names[hi:]
        if set(held)&set(train) or not train:
            raise ValueError('Cross-fit leakage/empty train')
        out.append(dict(held=held,train=train,exclusion=names[lo:hi]))
    return out
