"""Bounded output-box calibration; not detector training or identity evidence.

Always ONE active output per observation. Original box is audit metadata only.
Pixels/geometry used for association remain the original observation.
"""
import numpy as np


def features(d,cameras,offsets=None):
    cam = cameras[d['src']]
    x1,y1,x2,y2 = np.asarray(d['box'],float)
    w,h = max(x2-x1,2),max(y2-y1,2)
    logit = float(np.log(np.clip(d['conf'],1e-5,1-1e-5)/(1-np.clip(d['conf'],1e-5,1-1e-5))))
    lw,lh = np.log(w/cam.width),np.log(h/cam.height)
    cx,cy = (x1+x2)/2,(y1+y2)/2
    radius = ((cx-cam.K[0,2])/cam.K[0,0])**2+((cy-cam.K[1,2])/cam.K[1,1])**2
    top = offsets[str(d['src'])] if offsets is not None else 0
    bottom = top+568 if offsets is not None else cam.height
    cls = np.array([d['cls']==c for c in (0,1,2,3,5,7)],float)
    view = np.array([d['src']==c for c in (1,2,3,4,5)],float)
    return np.r_[cls,view,logit,lw,lh,np.log(w/h),radius,
                 abs(y1-top)<=5,abs(y2-bottom)<=5,
                 np.outer(cls,[logit,lw,lh]).ravel(),np.outer(view,[logit,lw,lh,radius]).ravel()]


def target(observed,truth):
    a,b = np.asarray(observed,float),np.asarray(truth,float)
    sa,sb = np.maximum(a[2:]-a[:2],2),np.maximum(b[2:]-b[:2],2)
    return np.r_[((b[:2]+b[2:])-(a[:2]+a[2:]))/(2*sa),np.log(sb/sa)]


def fit(xs,ys,l2=10.):
    x,y = np.asarray(xs,float),np.asarray(ys,float)
    if x.ndim!=2 or y.shape!=(len(x),4) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid calibration samples')
    mean,scale = x.mean(axis=0),np.maximum(x.std(axis=0),.05)
    z = np.column_stack([np.ones(len(x)),(x-mean)/scale])
    penalty = np.eye(z.shape[1])*l2
    penalty[0,0] = 0.
    weights = np.linalg.solve(z.T@z+penalty,z.T@y)
    return dict(weights=weights.tolist(),mean=mean.tolist(),scale=scale.tolist(),samples=len(x),l2=l2,
        max_centre_fraction=.20,min_scale=.8,max_scale=1.25,
        target='Normalized centre and log-size annotation residual; not exact physical object extent')


def correct(d,model,cameras,offsets=None,strength=1.):
    if not 0<=strength<=1:
        raise ValueError('Invalid correction strength')
    if strength==0:
        return dict(d,raw_observation_box=list(d['box']),box_calibration_strength=0.)
    x = features(d,cameras,offsets)
    design = np.r_[1.,(x-model['mean'])/np.asarray(model['scale'])]
    delta = design@np.asarray(model['weights'])
    delta[:2] = np.clip(delta[:2],-model['max_centre_fraction'],model['max_centre_fraction'])
    delta[2:] = np.clip(delta[2:],np.log(model['min_scale']),np.log(model['max_scale']))
    box = np.asarray(d['box'],float)
    size = np.maximum(box[2:]-box[:2],2)
    centre = (box[:2]+box[2:])/2+strength*delta[:2]*size
    changed_size = size*np.exp(strength*delta[2:])
    changed = np.r_[centre-changed_size/2,centre+changed_size/2]
    cam = cameras[d['src']]
    changed[[0,2]] = np.clip(changed[[0,2]],0,cam.width-1)
    changed[[1,3]] = np.clip(changed[[1,3]],0,cam.height-1)
    if min(changed[2:]-changed[:2])<2 or not np.isfinite(changed).all():
        changed = box
    return dict(d,box=changed.tolist(),raw_observation_box=box.tolist(),box_calibration_strength=float(strength))
