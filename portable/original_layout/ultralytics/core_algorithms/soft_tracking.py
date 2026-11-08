"""Idea 7 - soft temporal evidence (never deletes).

Previous frame's admitted objects (from the NON-temporal model, so there is no
circularity) are moved into the current IMU frame with visual ego-motion
(temporal_bev.motion, ORB + stereo PnP).  A current candidate gets
    track = +1  if an ego-compensated previous object of a compatible class is
               within a Mahalanobis gate (position sigma + 2 m motion allowance)
    track = -1  if motion is known, the object would have been inside a previous
               camera image, and nothing matched (weak negative evidence)
    track =  0  otherwise (initial frame / motion failure / not previously visible)
The value is only a FEATURE of the evidence model; its weight is learned on
development frames, so a missing history can lower but never veto an object.
"""
import numpy as np
from multiview_fusion import compatible
from temporal_bev import motion
from depth_bev_fusion import object_point


def ego_motion(prev_orb, cur_orb, cameras):
    if prev_orb is None or cur_orb is None: return None
    prev = {c: (prev_orb[f'uv{c}'].astype(float), prev_orb[f'desc{c}']) for c in (2, 4)}
    cur = {c: (cur_orb[f'uv{c}'].astype(float), cur_orb[f'desc{c}']) for c in (2, 4)}
    T, info = motion(prev, cur, cameras)
    return T


def annotate(cands, previous, T, cameras):
    if T is None or previous is None:
        for o in cands: o['track'] = 0.
        return
    R, t = T
    prev_pts = []
    for p in previous:
        P, Z, s = p['P'], p['Z'], p['s']
        prev_pts.append((R @ P + t, s, p['cls']))
    for o in cands:
        P, Z, s = object_point(o); best = None
        for Q, sq, cls in prev_pts:
            if not compatible(cls, o['cls']): continue
            e = (P - Q)[:2]; var = s * s + sq * sq + 2. ** 2
            m = float(e @ e / var)
            if m <= 9.21 and (best is None or m < best): best = m
        if best is not None: o['track'] = 1.; continue
        before = R.T @ (P - t)
        seen = any(cameras[c].project([before])[1][0] for c in (2, 3, 4))
        o['track'] = -1. if seen else 0.


def summarise(objs):
    out = []
    for o in objs:
        P, Z, s = object_point(o); out.append(dict(P=P, Z=Z, s=s, cls=o['cls']))
    return out


def summarise_slim(objs):
    """Same as summarise() for slimmed pickled objects (dets carry P/Z/sigZ)."""
    out = []
    for o in objs:
        ds = o['dets']; w = np.array([1 / d['sigZ'] ** 2 for d in ds])
        P = (np.array([d['P'] for d in ds]) * w[:, None]).sum(0) / w.sum()
        out.append(dict(P=P, Z=float(np.sum([d['Z'] * wi for d, wi in zip(ds, w)]) / w.sum()), s=float(1 / np.sqrt(w.sum())), cls=o['cls']))
    return out
