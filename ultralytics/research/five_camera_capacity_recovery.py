"""Source-exclusive five-view transfer of conditional matching-capacity recovery.

Compact old observable features, residual witnesses and independent Cam234
predictions. Fisheye cameras NEVER enter the collinear three-camera equation.
No GT parameters in features/rollout, no forced new identity associations.
"""
import itertools
import numpy as np
import reconditioned_scene_recovery as R
import three_camera_predictive_recovery as Q

DIMENSIONS={'capacity':529,'residual':575,'verification':613}


def residual_features(data,indices):
    n=len(data['uids']);cams=data['cameras'];q=data['q'];known=data['q_known'];cos=data['roi_cos']
    extra=np.zeros((n,46));available=np.zeros(n,bool);available[indices]=True;best={};supported=np.zeros((n,5),bool)
    for k,cam in enumerate(range(1,6)):
        js=np.flatnonzero(available&(cams==cam))
        if not len(js):continue
        values=q[:,js].copy();values[cams==cam]=0.;chosen=js[np.argmax(values,axis=1)];top=values.max(axis=1)
        order=np.sort(values,axis=1);second=order[:,-2] if len(js)>1 else np.zeros(n);found=top>0;mutual=np.zeros(n)
        for i in indices:
            if not found[i]:continue
            peers=indices[cams[indices]==cams[i]];others=peers[peers!=i]
            mutual[i]=not len(others) or q[chosen[i],i]>q[chosen[i],others].max()+1e-9
        valid=known[:,js].mean(axis=1);valid[cams==cam]=0.
        extra[:,k*8:(k+1)*8]=np.c_[top,second,top-second,np.where(found,data['conf'][chosen],0.),mutual,
            np.where(found,cos[np.arange(n),chosen],0.),valid,found]
        best[cam]=chosen;supported[:,k]=found
    extra[:,40]=supported.sum(axis=1)/4;extra[:,41]=np.sum(extra[:,[0,8,16,24,32]]>=.5,axis=1)/4
    if len(indices):
        overlap=data['iou'][:,indices].copy()
        for j,i in enumerate(indices):overlap[i,j]=0.
        extra[:,42]=overlap.max(axis=1)
    for i in indices:
        witnesses=[best[c][i] for c in range(1,6) if c!=cams[i] and c in best and supported[i,c-1]]
        triangle=[min(q[i,a],q[i,b],q[a,b]) for a,b in itertools.combinations(witnesses,2) if known[a,b]]
        extra[i,43]=max(triangle,default=0.);extra[i,44]=bool(triangle)
        extra[i,45]=len(witnesses)/4
    return extra


def appearance_features(data,scene,indices):
    n=len(data['uids']);lookup={int(u):i for i,u in enumerate(data['uids'])};active=np.array([lookup[u] for g in scene for u in g],int)
    cos=data['roi_cos'];cams=data['cameras'];extra=np.zeros((n,14))
    for i in indices:
        same=active[cams[active]==cams[i]];other=active[cams[active]!=cams[i]]
        if len(same):extra[i,0]=max(0.,float(cos[i,same].max()))
        if len(other):
            values=np.sort(cos[i,other]);extra[i,1]=max(0.,float(values[-1]));extra[i,3]=float(values[-1]-(values[-2] if len(values)>1 else 0.))
            gated=other[data['q'][i,other]>0]
            if len(gated):extra[i,2]=max(0.,float(cos[i,gated].max()))
        for k,cam in enumerate(range(1,6)):
            if cam==cams[i]:continue
            js=indices[(cams[indices]==cam)&(data['q'][i,indices]>0)]
            if len(js):
                values=np.sort(cos[i,js]);extra[i,4+2*k]=max(0.,float(values[-1]));extra[i,5+2*k]=max(0.,float(values[-2])) if len(values)>1 else 0.
    return extra


def attach(data,q_known,predictive):
    n=len(data['uids']);known=np.asarray(q_known,bool);predictive=np.asarray(predictive,float)
    if known.shape!=(n,n) or not np.array_equal(known,known.T) or np.diag(known).any() or predictive.shape!=(n,24):raise ValueError('Invalid typed five-view evidence')
    if np.any((data['q']>0)&~known) or not np.isfinite(predictive).all():raise ValueError('Unknown pair cannot be positive support')
    if np.any(predictive[~np.isin(data['cameras'],[2,3,4])]):raise ValueError('Fisheye must not receive collinear prediction')
    roi=data['phi'][:,91:475];norm=roi/np.maximum(np.linalg.norm(roi,axis=1,keepdims=True),1e-12)
    return dict(data,q_known=known,roi_cos=np.clip(norm@norm.T,-1,1),predictive=predictive)


def features(data,scene,head):
    if head not in DIMENSIONS:raise ValueError('Unknown five-view recovery head')
    idx,rich=R.features(data,scene,{},0.)
    # Keep original0:475 and859:913 EXACT; remove only384 zero ROI/std columns.
    if np.any(rich[:,475:859]):raise ValueError('Cannot drop nonzero group ROI dispersion')
    base=np.c_[rich[:,:475],rich[:,859:913]]
    if head=='capacity':x=base
    else:
        x=np.c_[base,residual_features(data,idx)[idx]]
        if head=='verification':x=np.c_[x,appearance_features(data,scene,idx)[idx],data['predictive'][idx]]
    if x.shape!=(len(idx),DIMENSIONS[head]) or not np.isfinite(x).all():raise ValueError('Invalid five-view conditional features')
    return idx,x


def rollout(data,scene,head,predict,floor=.02):
    current=[list(g) for g in scene];added=[];raw=[]
    for _ in range(len(data['uids'])):
        idx,x=features(data,current,head)
        if not len(idx):break
        values=np.asarray(predict(x),float)
        if values.shape!=(len(idx),) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid five-view recovery score')
        j=int(np.argmax(values));score=float(values[j])
        if score<floor:break
        u=int(data['uids'][idx[j]]);current.append([u]);added.append(u);raw.append(score)
    return added,np.minimum.accumulate(raw).tolist(),raw
