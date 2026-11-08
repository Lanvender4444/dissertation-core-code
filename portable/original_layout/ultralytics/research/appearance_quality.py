"""Cheap cached-appearance and view-conditioned scoring features. No GT access.

Adds evidence, never declares a one-view object false or forces association.
Histogram similarity alone does NOT prove that observations share identity.
"""
import itertools

import numpy as np


def representatives(detections):
    if not detections:
        raise ValueError('Empty object')
    chosen = {}
    for d in detections:
        c = int(d['src'])
        if c not in chosen or (d['conf'], -d.get('uid',0))>(chosen[c]['conf'], -chosen[c].get('uid',0)):
            chosen[c] = d
    return [chosen[c] for c in sorted(chosen)]


def appearance(detections):
    reps = representatives(detections)
    fs = []
    for d in reps:
        f = np.asarray(d['feature'],float)
        if f.shape!=(48,) or not np.isfinite(f).all() or np.any(f<0):
            raise ValueError('Invalid cached HS descriptor')
        norm = np.linalg.norm(f)
        if norm>1e-9:
            fs.append(f/norm)
    if fs:
        probabilities = np.mean(np.square(fs),axis=0)
        saturation = probabilities.reshape(12,4).sum(axis=0)
        positive = probabilities[probabilities>0]
        entropy = -float(np.sum(positive*np.log(positive)))/np.log(48)
    else:
        probabilities,saturation,entropy = np.zeros(48),np.zeros(4),0.
    sims = [float(np.clip(a@b,0,1)) for a,b in itertools.combinations(fs,2)]
    stats = [float(np.mean(sims)), min(sims), float(np.std(sims)), len(sims)/10] if sims else [0.,0.,0.,0.]
    return np.r_[probabilities,saturation,entropy,stats,len(fs)/len(reps)]  # 58


def features(base,detections,offsets,conditional=False):
    x = np.asarray(base,float)
    if x.shape not in ((41,),(50,)) or not np.isfinite(x).all():
        raise ValueError('Expected frozen native/shared feature vector')
    a = appearance(detections)
    if not conditional:
        return np.r_[x,a]
    reps = representatives(detections)
    top = [abs(d['box'][1]-offsets[str(d['src'])])<=5 for d in reps]
    bottom = [abs(d['box'][3]-(offsets[str(d['src'])]+568))<=5 for d in reps]
    boundary = np.asarray(top)|bottom
    crop = [float(boundary.any()),float(boundary.mean()),float(np.mean(top)),float(np.mean(bottom))]
    # 12 base + 24 native region + 5 camera indicators. Do not reorder sources.
    class_context = np.asarray([x[0],x[2],x[14],x[15],x[16],x[19],crop[0]])
    view_context = np.asarray([x[0],x[2],x[15],x[16],x[19],x[18],crop[0]])
    classes = np.outer(x[6:12],class_context).ravel()  # 42
    views = np.outer(x[36:41],view_context).ravel()   # 35
    multi = float(len(reps)>1)
    support = np.outer([multi,1-multi],[x[0],x[2],x[1],x[41] if len(x)==50 else 0.,a[53]]).ravel()  # 10
    # Hue is circular; low-saturation gray pixels should have little hue weight.
    hs = a[:48].reshape(12,4)
    phase = 2*np.pi*(np.arange(12)+.5)/12
    hue_weight = hs@((np.arange(4)+.5)/4)
    hue_real,hue_imag = float(hue_weight@np.cos(phase)),float(hue_weight@np.sin(phase))
    colour_context = np.r_[a[48:53],hue_real,hue_imag,np.hypot(hue_real,hue_imag)]  # 8
    conditional_colour = np.r_[np.outer(x[6:12],colour_context).ravel(),
                               np.outer(x[36:41],colour_context).ravel()]  # 88
    return np.r_[x,a,crop,classes,views,support,conditional_colour]  # 287 shared, 278 native
