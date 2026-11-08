"""Detector-fixed post-hoc evidence calibration, with explicit feature ablations."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def object_features(obj, front, evidence, cameras):
    ds = obj['dets']
    confidence = np.clip([d['conf'] for d in ds], 1e-5, 1-1e-5)
    logits = np.log(confidence/(1-confidence))
    classes = [float(obj['cls'] == c) for c in (0, 1, 2, 3, 5, 7)]
    basic = [float(logits.mean()), float(logits.min()), float(logits.max()),
             len(set(obj['src']))/5, float(len(ds) == 1),
             float(any(d['src'] in (1, 5) for d in ds))]+classes
    native = []
    for d in ds:
        c = cameras[d['src']]
        x1, y1, x2, y2 = np.asarray(d['box'], float)
        w, h = max(x2-x1, 1)/c.width, max(y2-y1, 1)/c.height
        native.append([(x1+x2)/(2*c.width), (y1+y2)/(2*c.height),
                       np.log(w), np.log(h), np.log(w/h),
                       float(min(x1, y1, c.width-x2, c.height-y2) < 5),
                       float(d['geometry_valid']), float(d['angular_h']) if d['geometry_valid'] else 0.])
    native = np.asarray(native)
    view = [float(c in obj['src']) for c in (1, 2, 3, 4, 5)]
    region = [*native.mean(axis=0), *native.min(axis=0), *native.max(axis=0), *view]
    best = next((rec for o, rec in zip(front, evidence)
                 if o.get('front_anchor_id') == obj.get('front_anchor_id')), None) if 'front_anchor_id' in obj else None
    joint_cost = min(best['front_best'], 10.) if best is not None else 10.
    feasible_fraction = float(best['feasible'].mean()) if best is not None else 0.
    associations = obj.get('side_association', [])
    costs = [a['cost'] for a in associations]
    certs = [a['sampled_certificate'] for a in associations if a['sampled_certificate'] is not None]
    potential = []
    for d in ds:
        if d['src'] not in (1, 5):
            continue
        for rec in evidence:
            if rec is None:
                continue
            entry = rec['sides'][d['src']].get(d['uid'])
            if entry is not None:
                potential.append(float(entry['object'][rec['feasible']].min()))
    ordered = sorted(potential)
    first = min(ordered[0], 10.) if ordered else 10.
    second = min(ordered[1], 10.) if len(ordered) > 1 else 10.
    shared = [joint_cost, feasible_fraction, float(len(associations)),
              float(np.mean(costs)) if costs else 1.,
              float(min(max(min(certs), -10.), 10.)) if certs else -10.,
              first, second-first, float(len(potential)), float(any(d['geometry_valid'] for d in ds))]
    return dict(basic=np.asarray(basic), native=np.asarray(basic+region),
                shared=np.asarray(basic+region+shared))


def fit(features, labels, regularization=1.):
    x, y = np.asarray(features, float), np.asarray(labels, float)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or len(np.unique(y)) < 2:
        raise ValueError('Invalid calibration data')
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .05)
    z = (x-mean)/scale
    design = np.column_stack([np.ones(len(z)), z])
    def objective(weights):
        eta = design @ weights
        loss = np.logaddexp(0, eta).sum()-y @ eta+regularization/2*np.dot(weights[1:], weights[1:])
        gradient = design.T @ (expit(eta)-y)
        gradient[1:] += regularization*weights[1:]
        return float(loss), gradient
    result = minimize(objective, np.zeros(design.shape[1]), method='L-BFGS-B', jac=True,
                      options=dict(maxiter=1000, ftol=1e-10))
    if not result.success:
        raise ValueError(f'Calibration failed: {result.message}')
    return dict(weights=result.x.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                train_samples=len(y), positive=int(y.sum()), regularization=regularization,
                target='any-member eligible GT IoU hit proxy, NOT full identity correctness')


def predict(features, model):
    x = np.asarray(features, float)
    if not len(x):
        return np.empty(0)
    w = np.asarray(model['weights'])
    return expit(w[0]+((x-np.asarray(model['mean']))/np.asarray(model['scale'])) @ w[1:])
