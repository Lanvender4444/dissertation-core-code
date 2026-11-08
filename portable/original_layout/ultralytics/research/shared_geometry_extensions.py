"""GT-free selective scene admission and cross-view rescue experiments.

Unknown candidates remain in a diagnostic layer, not relabelled false positives.
"""
import copy

import numpy as np

import shared_geometry as S
from multiview_fusion import compatible


def defer_ambiguous_singletons(objects, front, side, evidence, links, gate):
    linked = {link['detection_uid'] for link in links}
    uncertain = set()
    for d in side:
        if d['uid'] in linked:
            continue
        for rec in evidence:
            if rec is None:
                continue
            entry = rec['sides'][d['src']].get(d['uid'])
            if entry is not None and entry['object'][rec['feasible']].min() < gate:
                uncertain.add(d['uid'])
                break
    confirmed = [o for o in objects if not (len(o['dets']) == 1 and o['dets'][0]['uid'] in uncertain
                                           and o['src'][0] in (1, 5))]
    diagnostic = [dict(uid=d['uid'], source=d['src'], native_box=d['box'],
                       reason='possible_existing_object_identity_unresolved',
                       existence='not_judged_false', location='unknown')
                  for d in side if d['uid'] in uncertain]
    return confirmed, diagnostic


def ios(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
    return float(intersection/max(min(np.prod(a[2:]-a[:2]), np.prod(b[2:]-b[:2])), 1e-9))


def shadow_front(raw_front, front, minimum_confidence=.25):
    """Nonaccepted front observations are proposals, not asserted new objects."""
    accepted = [d for o in front for d in o['dets']]
    uid = {d['uid'] for d in accepted}
    candidates = []
    for d in raw_front:
        if d['uid'] in uid or d['conf'] < minimum_confidence or not d['geometry_valid']:
            continue
        if any(a['src'] == d['src'] and compatible(a['cls'], d['cls']) and
               ios(a['box'], d['box']) >= .7 for a in accepted):
            continue
        if any(a['src'] == d['src'] and compatible(a['cls'], d['cls']) and
               ios(a['box'], d['box']) >= .7 for o in candidates for a in o['dets']):
            continue
        candidates.append(dict(src=[d['src']], cls=d['cls'], dets=[copy.deepcopy(d)],
                               existence_source='shadow_unadmitted_front_candidate'))
    return candidates


def rescue_scene(objects, side, links, shadows, shadow_evidence, shadow_links,
                 front_conf=.5, side_conf=.5, gate=.45):
    """Add a new object only if front + side fit one common sampled state.

    Confidence cutoffs are existence heuristics, NOT independent probabilities.
    Side observations assigned to original front objects are never reused.
    """
    used = {l['detection_uid'] for l in links}
    by_uid = {d['uid']:d for d in side}
    grouped = {}
    for link in shadow_links:
        if link['detection_uid'] not in used:
            grouped.setdefault(link['front'], []).append(link)
    added = []
    consumed = set()
    for k, additions in grouped.items():
        obj, rec = shadows[k], shadow_evidence[k]
        ds = [by_uid[l['detection_uid']] for l in additions]
        if max(d['conf'] for d in obj['dets']) < front_conf or max(d['conf'] for d in ds) < side_conf:
            continue
        costs = np.stack([rec['sides'][d['src']][d['uid']]['object'][rec['feasible']] for d in ds])
        # Each view must be explainable at the SAME state, not independent minima.
        if np.min(np.max(costs, axis=0)) >= gate:
            continue
        group = copy.deepcopy(obj)
        group['dets'] += copy.deepcopy(ds)
        group['src'] = sorted({d['src'] for d in group['dets']})
        group['existence_source'] = 'new_two_view_rescue_heuristic_not_calibrated_probability'
        group['location_state'] = 'sampled_range_region'
        added.append(group)
        consumed.update(d['uid'] for d in ds)
    # Replace an existing side singleton with its new jointly verified group.
    remaining = [o for o in objects if not (len(o['dets']) == 1 and o['dets'][0]['uid'] in consumed)]
    return remaining+added, dict(rescue_objects=len(added), rescued_side_uids=sorted(consumed))
