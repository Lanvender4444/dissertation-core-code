"""Selective residual recovery: existence/novelty, not forced identity merging.

GT-free inference. Models score residual *new-object* evidence, not a detector
probability. Bearing-only recovery must not be interpreted as a metric BEV point.
"""
import numpy as np

import cam15_scoring as Q
from multiview_fusion import compatible
from shared_geometry_extensions import ios


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    area = np.prod(np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
    return float(area/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area, 1e-9))


def overlap(d, others):
    peers = [a for a in others if a['src'] == d['src'] and compatible(a['cls'], d['cls'])]
    return (max((iou(d['box'], a['box']) for a in peers), default=0.),
            max((ios(d['box'], a['box']) for a in peers), default=0.))


def features(d, baseline, unknown, cameras):
    """Native-box evidence + context of already accepted scene; no GT/depth."""
    o = dict(src=[d['src']], cls=d['cls'], dets=[d])
    x = Q.object_features(o, [], [], cameras)['native']
    members = [a for obj in baseline for a in obj['dets']]
    v, s = overlap(d, members)
    peers = [a for a in members if a['src'] == d['src'] and compatible(a['cls'], d['cls'])]
    cam = cameras[d['src']]
    centre = (np.asarray(d['box'][:2])+d['box'][2:])/2
    distance = min((np.linalg.norm((centre-(np.asarray(a['box'][:2])+a['box'][2:])/2)
                                  /[cam.width, cam.height]) for a in peers), default=1.)
    extra = [v, s, distance, len(peers)/20, float(d['uid'] in unknown),
             float(d['src'] in (1, 5)), len(baseline)/30]
    return np.r_[x, extra]


def candidates(pool, baseline, allowed, unknown, cameras):
    used = {d['uid'] for o in baseline for d in o['dets']}
    members = [d for o in baseline for d in o['dets']]
    result = []
    for d in pool:
        if d['uid'] not in allowed or d['uid'] in used:
            continue
        v, s = overlap(d, members)
        if v >= .5 or s >= .8:
            continue
        result.append(dict(detection=d, x=features(d, baseline, unknown, cameras),
                           deferred=d['uid'] in unknown))
    return result


def recover(baseline, proposals, probabilities, tau, scope='safe_all'):
    """Never remove baseline objects; no cross-view identity is asserted.

    Safe policies retain geometric-ambiguity deferral. open_all is a diagnostic
    control which relaxes it, and can cause duplicate scene objects.
    """
    if scope not in ('safe_all', 'safe_side', 'open_all'):
        raise ValueError('Unknown recovery scope')
    if len(proposals) != len(probabilities):
        raise ValueError('Score count mismatch')
    out = list(baseline)
    members = [d for o in out for d in o['dets']]
    ranked = sorted(zip(proposals, probabilities), key=lambda pair: (-float(pair[1]), pair[0]['detection']['uid']))
    for proposal, probability in ranked:
        d = proposal['detection']
        if probability < tau or (scope != 'open_all' and proposal['deferred']):
            continue
        if scope == 'safe_side' and d['src'] not in (1, 5):
            continue
        v, s = overlap(d, members)
        if v >= .5 or s >= .8:
            continue
        out.append(dict(src=[d['src']], cls=d['cls'], dets=[d],
                        recovery_score=float(probability), location_state='bearing_only',
                        association_state='unverified_not_forced_merge',
                        existence_source='residual_scene_recovery'))
        members.append(d)
    return out
