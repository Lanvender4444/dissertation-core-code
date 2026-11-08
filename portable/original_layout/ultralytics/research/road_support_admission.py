"""Validity-aware aggregation of independent physical features, no GT."""
import numpy as np

from appearance_quality import representatives
from road_support import DIM


def validity(d,x):
    valid = np.ones(DIM,bool)
    plane,stereo,ground = x[0]>0,x[1]>0,x[11]>0
    valid[1:3] = plane and d['src'] in (2,4)
    valid[3:11] = stereo
    valid[11] = plane
    valid[12:16] = ground
    valid[16] = x[17]>0
    valid[17] = plane and d['src'] in (2,4)
    valid[18:20] = plane
    return valid


def augment(base,detections,features,group=True):
    base = np.asarray(base,float)
    if base.shape!=((1056,) if group else (670,)) or not np.isfinite(base).all() or not detections:
        raise ValueError('Invalid frozen ROI feature')
    if not group and len(detections)!=1:
        raise ValueError('Residual must have one source observation')
    ds = representatives(detections) if group else detections
    values = np.asarray([features[str(d['uid'])] for d in ds],float)
    if values.shape!=(len(ds),DIM) or not np.isfinite(values).all():
        raise ValueError('Invalid physical feature cache')
    if not group:
        return np.r_[base,values[0]]  # 690, missing flags in the descriptors
    mask = np.asarray([validity(d,x) for d,x in zip(ds,values)])
    n = mask.sum(axis=0)
    mean = np.where(mask,values,0).sum(axis=0)/np.maximum(n,1)
    lo = np.where(n>0,np.min(np.where(mask,values,np.inf),axis=0),0)
    hi = np.where(n>0,np.max(np.where(mask,values,-np.inf),axis=0),0)
    return np.r_[base,mean,lo,hi,n/len(ds)]  # 1136; missing is never a fake zero-height surface
