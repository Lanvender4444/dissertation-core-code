"""NumPy-only admission augmentation from cached same-forward visual features."""
import numpy as np

from appearance_quality import representatives

DIM = 384


def augment(base,detections,cache,group=True):
    x = np.asarray(base,float)
    if x.shape!=((287,) if group else (285,)) or not np.isfinite(x).all() or not detections:
        raise ValueError('Invalid frozen base feature/observation')
    if not group and len(detections)!=1:
        raise ValueError('Residual requires one original observation')
    ds = representatives(detections) if group else detections
    values = []
    for d in ds:
        entry = cache[str(d['uid'])]
        value = np.asarray(entry['x'],float)
        if value.shape!=(DIM,) or not np.isfinite(value).all():
            raise ValueError('Invalid same-forward detector descriptor')
        if entry['valid']:
            values.append(value)
    mean = np.mean(values,axis=0) if values else np.zeros(DIM)
    fraction = len(values)/len(ds)
    if group:
        std = np.std(values,axis=0) if values else np.zeros(DIM)
        return np.r_[x,mean,std,fraction]  # 1056
    return np.r_[x,mean,fraction]  # 670
