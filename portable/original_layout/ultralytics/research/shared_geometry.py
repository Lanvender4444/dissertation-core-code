"""GT-free sampled shared-object hypotheses for mixed pinhole/Mei cameras.

This implementation certifies assignments on a FINITE hypothesis set only.
It does not claim a bound over the continuous depth/calibration domain.
"""
import itertools
import copy

import numpy as np
from scipy.optimize import linear_sum_assignment

from multiview_fusion import compatible, pair_cost
from multiview_geometry import triangulate

BIG = 1e6
DIMENSIONS = {0: (.55, .55, 1.70), 1: (1.75, .65, 1.4), 2: (4.4, 1.8, 1.55),
              3: (2.0, .75, 1.5), 5: (10., 2.5, 3.2), 7: (6.5, 2.5, 3.0)}


def observation_key(d):
    return (int(d['src']), int(d['cls']), tuple(d['box']), float(d['conf']))


def safe_rays(camera, pixels, tolerance=.5):
    """Geometry failure must abstain from association, not delete detection."""
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    rays = camera.rays(pixels)
    back, valid = camera.project_rays(rays)
    valid &= np.isfinite(rays).all(axis=1)
    valid &= np.linalg.norm(back-pixels, axis=1) <= tolerance
    rays[~valid] = np.nan
    return rays, valid


def prepare_detection(d, cameras):
    d = copy.deepcopy(d)
    cam = cameras[d['src']]
    x1, y1, x2, y2 = d['box']
    rays, valid = safe_rays(cam, [[(x1+x2)/2, (y1+y2)/2],
                                  [(x1+x2)/2, y1], [(x1+x2)/2, y2]])
    d['ray'] = rays[0]
    d['geometry_valid'] = bool(valid.all())
    d['angular_h'] = (float(np.arccos(np.clip(rays[1] @ rays[2], -1, 1)))
                      if valid.all() else np.nan)
    f = np.asarray(d.get('feature', np.zeros(48)), float)
    d['feature'] = f / max(np.linalg.norm(f), 1e-9)
    return d


def side_ego(d, cameras):
    """Fixed rig heuristic, identical across five-camera variants; not universal."""
    if d['cls'] not in (2, 5, 7):
        return False
    cam = cameras[d['src']]
    x1, y1, x2, y2 = d['box']
    x, y = (x1+x2)/(2*cam.width), (y1+y2)/(2*cam.height)
    # Observed mirror/body triangles. Record excluded detections for audit.
    corner = ((d['src'] == 1 and x >= .77 and y > 1-(x-.77)*(.35/.23)) or
              (d['src'] == 5 and x <= .23 and y > .65+x*(.35/.23)))
    return bool(corner or y > .97)


def project_unbounded(camera, points):
    """Separate mathematical branch validity from clipping to image bounds."""
    shape = np.asarray(points).shape[:-1]
    q = (np.asarray(points).reshape(-1, 3)-camera.t) @ camera.R
    norm = np.linalg.norm(q, axis=1)
    den = q[:, 2] + camera.xi*norm
    valid = np.isfinite(q).all(axis=1) & (den > 1e-8)
    if camera.xi > 1:
        valid &= q[:, 2] / np.maximum(norm, 1e-9) > -1/camera.xi
    xy = q[:, :2] / np.maximum(den[:, None], 1e-8)
    x, y = xy.T
    k1, k2, p1, p2 = camera.D
    r2 = x*x+y*y
    radial = 1+k1*r2+k2*r2*r2
    uv = np.column_stack([camera.K[0, 0]*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+camera.K[0, 2],
                          camera.K[1, 1]*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+camera.K[1, 2]])
    valid &= np.isfinite(uv).all(axis=1)
    return uv.reshape(*shape, 2), valid.reshape(shape)


def state_grid(obj, cameras, depth_samples=12):
    """One physical centre/size/yaw shared by ALL observations and competitors.

    Candidate generation uses front detections ONLY. Box centre is an uncertain
    seed, not a claimed exact point correspondence or ground contact.
    """
    ds = [d for d in obj['dets'] if d['geometry_valid'] and d['src'] in (2, 3, 4)]
    if not ds:
        return None
    ref = max(ds, key=lambda d: (d['src'] == 3, d['conf']))
    cls = obj['cls']
    length, width, height = DIMENSIONS[cls]
    rough = height / max(ref['angular_h'], .008)
    low, high = max(1.5, rough*.35), min(180., rough*3.)
    if low >= high:
        return None
    distances = list(np.geomspace(low, high, depth_samples))
    for a, b in itertools.combinations(ds, 2):
        point, gap, depths, angle = triangulate(cameras[a['src']], a['ray'],
                                               cameras[b['src']], b['ray'])
        if angle > .001 and min(depths) > 0 and gap < max(.5, .15*min(depths)):
            distance = float((point-cameras[ref['src']].t) @ ref['ray'])
            distances.extend([distance*.85, distance, distance*1.15])
    distances = sorted(set(float(z) for z in distances if 1.5 <= z <= 180))
    yaws = (0., np.pi/4, np.pi/2, 3*np.pi/4) if cls in (1, 2, 3, 5, 7) else (0.,)
    states = np.asarray(list(itertools.product(distances, yaws, (.7, 1., 1.3), range(5))), float)
    ranges, yaw, scale = states[:, 0], states[:, 1], states[:, 2]
    cam = cameras[ref['src']]
    centres = cam.t + ranges[:, None]*ref['ray']
    # Bounded uncertain visible-centre offset in the reference image tangent.
    offsets = np.array([[0, 0], [-.25, 0], [.25, 0], [0, -.20], [0, .20]])
    offset = offsets[states[:, 3].astype(int)]
    centres += ((offset[:, :1]*length*scale[:, None])*cam.R[:, 0] +
                (offset[:, 1:]*height*scale[:, None])*cam.R[:, 1])
    dims = scale[:, None]*np.array([length, width, height])
    # Corners + edge midpoints, needed for non-linear fisheye box envelopes.
    template = np.array([p for p in itertools.product((-1., 0., 1.), repeat=3)
                         if sum(x == 0 for x in p) <= 1]) / 2
    vertices = dims[:, None, :]*template
    x = vertices[:, :, 0]*np.cos(yaw[:, None])-vertices[:, :, 1]*np.sin(yaw[:, None])
    y = vertices[:, :, 0]*np.sin(yaw[:, None])+vertices[:, :, 1]*np.cos(yaw[:, None])
    vertices[:, :, 0], vertices[:, :, 1] = x, y
    vertices += centres[:, None, :]
    return dict(centres=centres, vertices=vertices, ranges=ranges, states=states,
                reference_camera=ref['src'], depth_source='front_box_geometry_and_size_prior',
                location_state='sampled_region_not_metric_ground_truth')


def predicted_boxes(grid, camera):
    uv, valid = project_unbounded(camera, grid['vertices'])
    okay = valid.all(axis=1)
    box = np.column_stack([uv.min(axis=1), uv.max(axis=1)])
    # A detector can only observe the clipped portion inside its sensor.
    box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, camera.width-1)
    box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, camera.height-1)
    okay &= (box[:, 2]-box[:, 0] >= 4) & (box[:, 3]-box[:, 1] >= 4)
    return box, okay


def box_cost(boxes, detection, valid, centre_uv=None, expansion=0.):
    obs = np.asarray(detection['box'], float)
    size = np.maximum(boxes[:, 2:]-boxes[:, :2], 4.)
    centre = (boxes[:, 2:]+boxes[:, :2])/2 if centre_uv is None else centre_uv
    obs_size = np.maximum(obs[2:]-obs[:2], 4.)
    error = np.maximum(np.abs(centre-(obs[2:]+obs[:2])/2)-expansion*size, 0)
    # Size/pose affects the geometry of ALL candidates via the same state.
    position = np.sum((error/(.5*size+12.))**2, axis=1)
    ratio = np.abs(np.log(size/obs_size))
    # Partial visibility allows a detector box to occupy >=40% of full extent.
    # This is a bounded observation model, not a freely moving per-view point.
    partial = np.maximum(ratio-np.log(2.5), 0).sum(axis=1)
    cost = position + .25*partial
    cost[~valid | ~np.isfinite(cost)] = BIG
    return cost


def build_evidence(front, side, cameras, depth_samples=12):
    """Compute sampled state losses once; no truth or side-based anchor fitting."""
    records = []
    for obj in front:
        grid = state_grid(obj, cameras, depth_samples)
        if grid is None:
            records.append(None)
            continue
        view_costs = []
        for d in obj['dets']:
            if not d['geometry_valid']:
                continue
            boxes, valid = predicted_boxes(grid, cameras[d['src']])
            view_costs.append(box_cost(boxes, d, valid))
        losses = np.stack(view_costs)
        joint = losses.mean(axis=0)
        masks = joint <= joint.min()+.15
        if not masks.any():
            records.append(None)
            continue
        row = dict(grid=grid, front_losses=losses, front_loss=joint,
                   feasible=masks, front_best=float(joint.min()), sides={})
        for cid in (1, 5):
            boxes, valid = predicted_boxes(grid, cameras[cid])
            centre_uv, centre_valid = project_unbounded(cameras[cid], grid['centres'])
            entries = {}
            for d in side:
                if d['src'] != cid or not d['geometry_valid'] or not compatible(obj['cls'], d['cls']):
                    continue
                app = min(1., 1-float(np.mean([m['feature'] @ d['feature'] for m in obj['dets']])))
                common = .15*max(0., app)
                entries[d['uid']] = dict(
                    object=box_cost(boxes, d, valid)+common,
                    centre=box_cost(boxes, d, valid & centre_valid, centre_uv=centre_uv)+common,
                    expanded=box_cost(boxes, d, valid & centre_valid, centre_uv=centre_uv, expansion=.25)+common)
            row['sides'][cid] = entries
        records.append(row)
    return records


def relative_lower_bounds(costs, chosen, shared=True):
    """Sampled inf(c_b-c_a), versus independent inf(c_b)-sup(c_a).

    The returned matrix is a lower bound over the supplied FINITE states only.
    """
    costs = np.asarray(costs, float)
    current = costs[:, chosen]
    if shared:
        return np.min(costs-current[:, None], axis=0)
    return np.min(costs, axis=0)-np.max(current)


def exclusion_certificate(delta, chosen, row):
    """Minimum lower-bound total cost for an assignment changing this edge.

    Canonical private null columns avoid null permutations masquerading as
    different physical assignments. Sum of row infima is conservative.
    """
    alternative = np.asarray(delta, float).copy()
    alternative[row, chosen[row]] = BIG
    rr, cc = linear_sum_assignment(alternative)
    return float(alternative[rr, cc].sum())


def sampled_assignment(front, side, evidence, cameras, mode='object_joint',
                       gate=.75, margin=0., allowed_uids=None):
    """At most one observation per camera/object, with explicit abstention."""
    assignments = []
    diagnostics = dict(proposed=0, accepted=0, rejected_ambiguity=0, geometry_unavailable=0)
    for cid in (1, 5):
        candidates = [d for d in side if d['src'] == cid and
                      (allowed_uids is None or d['uid'] in allowed_uids)]
        n, m = len(front), len(candidates)
        if n == 0 or m == 0:
            continue
        matrix = np.full((n, m+n), BIG)
        state_costs = {}
        for k, (obj, rec) in enumerate(zip(front, evidence)):
            matrix[k, m+k] = gate
            if rec is None:
                diagnostics['geometry_unavailable'] += 1
                continue
            mask = rec['feasible']
            costs = np.full((int(mask.sum()), m+n), BIG)
            costs[:, m+k] = gate
            kind = mode if mode in ('centre', 'expanded') else 'object'
            for j, d in enumerate(candidates):
                entry = rec['sides'][cid].get(d['uid'])
                if entry is not None:
                    costs[:, j] = entry[kind][mask]
                    if mode == 'best_pair':
                        pairs = [pair_cost(d, member, cameras) for member in obj['dets']]
                        pairs = [e[0] for e in pairs if e is not None]
                        matrix[k, j] = min(pairs) if pairs else BIG
                    elif mode == 'independent_members':
                        separately = []
                        for loss in rec['front_losses']:
                            member_mask = loss <= loss.min()+.15
                            separately.append(float(entry['object'][member_mask].min()))
                        matrix[k, j] = float(np.mean(separately))
                    else:
                        matrix[k, j] = float(costs[:, j].min())
            state_costs[k] = costs
        rr, cc = linear_sum_assignment(matrix)
        chosen = np.full(n, -1, int)
        chosen[rr] = cc
        delta = np.full_like(matrix, BIG)
        for k in range(n):
            if k in state_costs:
                delta[k] = relative_lower_bounds(state_costs[k], chosen[k],
                                                  shared=mode != 'independent_interval')
            else:
                delta[k, m+k] = 0.
            # Private null columns must stay inaccessible to other rows.
            delta[k, m:] = BIG
            delta[k, m+k] = (relative_lower_bounds(state_costs[k], chosen[k],
                              shared=mode != 'independent_interval')[m+k]
                              if k in state_costs else 0.)
            delta[k, chosen[k]] = 0.
        for k, j in zip(rr, cc):
            if j >= m or matrix[k, j] >= gate:
                continue
            diagnostics['proposed'] += 1
            certificate = None
            if mode in ('shared_difference', 'independent_interval'):
                others = np.delete(delta[k], j)
                certificate = float(others.min())
            elif mode == 'shared_global':
                certificate = exclusion_certificate(delta, chosen, k)
            if certificate is not None and certificate <= margin:
                diagnostics['rejected_ambiguity'] += 1
                continue
            diagnostics['accepted'] += 1
            assignments.append(dict(front=k, detection_uid=candidates[j]['uid'], camera=cid,
                                    cost=float(matrix[k, j]), sampled_certificate=certificate,
                                    certificate_domain='finite_front_generated_states'))
    return assignments, diagnostics


def fuse(front, side, links, singleton_threshold=.2, front_veto=None, evidence=None):
    """Preserve front native boxes; never replace its score with YOLO max."""
    kept = [True]*len(front)
    if front_veto is not None:
        for k, rec in enumerate(evidence):
            if rec is not None and len(front[k]['src']) >= 2:
                kept[k] = rec['front_best'] <= front_veto
    out = [copy.deepcopy(o) for o, yes in zip(front, kept) if yes]
    index = {k: sum(kept[:k]) for k in range(len(front)) if kept[k]}
    by_uid = {d['uid']:d for d in side}
    used = set()
    for link in links:
        if link['front'] not in index:
            continue
        o = out[index[link['front']]]
        d = by_uid[link['detection_uid']]
        if d['uid'] in used or d['src'] in o['src']:
            raise ValueError('Duplicate observation/camera in object')
        o['dets'].append(copy.deepcopy(d))
        o['src'] = sorted(o['src']+[d['src']])
        o.setdefault('side_association', []).append(link)
        used.add(d['uid'])
    for d in side:
        if d['uid'] in used or d['conf'] < singleton_threshold:
            continue
        out.append(dict(src=[d['src']], dets=[copy.deepcopy(d)], cls=d['cls'],
                        existence_source='side_yolo_not_front_calibrated_probability',
                        location_state='bearing_only', association_state='unverified'))
    return out
