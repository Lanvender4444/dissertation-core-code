"""Marginal maximum-matching capacity labels, NOT physical identity truth.

Binary edges mean a prediction can qualify against a GT. GT input belongs only
in training-label construction/evaluation, never in deployment admission.
"""
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


def marginal_labels(seed_edges,candidate_edges):
    seed,candidates = np.asarray(seed_edges),np.asarray(candidate_edges)
    if seed.ndim!=2 or candidates.ndim!=2 or seed.shape[1]!=candidates.shape[1]:
        raise ValueError('Require aligned object-by-GT binary matrices')
    if not np.isin(seed,[False,True]).all() or not np.isin(candidates,[False,True]).all():
        raise ValueError('Binary qualification edges required')
    seed,candidates = seed.astype(bool),candidates.astype(bool)
    n_gt = seed.shape[1]
    matched_gt = np.full(n_gt,-1,int)
    matched_object = np.full(len(seed),-1,int)
    if seed.size:
        rr,cc = linear_sum_assignment(-seed.astype(int))
        for r,c in zip(rr,cc):
            if seed[r,c]:
                matched_gt[c],matched_object[r] = r,c
    # Reverse alternating reachability: a candidate's edge to any reachable
    # GT permits an augmenting path ending at an unmatched GT. Build this ONCE
    # for all candidates; adding one object can increase capacity by at most 1.
    predecessors = [[] for _ in range(n_gt)]
    for i,j in enumerate(matched_object):
        if j>=0:
            for t in np.flatnonzero(seed[i]):
                predecessors[t].append(int(j))
    reachable = matched_gt<0
    queue = deque(np.flatnonzero(reachable).tolist())
    while queue:
        t = queue.popleft()
        for j in predecessors[t]:
            if not reachable[j]:
                reachable[j] = True
                queue.append(j)
    labels = np.any(candidates[:,reachable],axis=1).astype(int)
    return dict(labels=labels,baseline_capacity=int(np.sum(matched_object>=0)),
        reachable_gt=reachable,baseline_matching=matched_object)
