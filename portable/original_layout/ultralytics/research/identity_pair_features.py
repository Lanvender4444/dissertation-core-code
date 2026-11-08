"""Vectorized mixed-camera pair observations; no GT, no forced identity.

Two-ray distance is a bbox-centre hypothesis, NOT reliable stereo depth.
Missing geometry stays explicitly unknown; feature extraction never vetoes it.
"""
import numpy as np
from multiview_fusion import compatible

CAMERA_PAIRS=tuple((a,b) for a in range(1,6) for b in range(a,6))
DIM=56


def extract(pool,cameras,descriptors,crop_flags):
    uids=sorted(pool);n=len(uids)
    if n<2:return np.empty((0,2),int),np.empty((0,DIM))
    ds=[pool[u] for u in uids]
    if any(d['uid']!=u or d['src'] not in range(1,6) for u,d in zip(uids,ds)):
        raise ValueError('Invalid source UID/camera')
    boxes=np.asarray([d['box'] for d in ds],float);wh=boxes[:,2:]-boxes[:,:2]
    if np.any(wh<=0) or not np.isfinite(boxes).all():raise ValueError('Invalid native boxes')
    roi=np.asarray([descriptors[str(u)]['x'] for u in uids],float)
    if roi.shape!=(n,384) or not np.isfinite(roi).all():raise ValueError('Invalid signed old ROI descriptors')
    for u,d in zip(uids,ds):
        if descriptors[str(u)]['camera']!=d['src']:raise ValueError('ROI camera differs')
    roi/=np.maximum(np.linalg.norm(roi,axis=1,keepdims=True),1e-9)
    hs=np.asarray([d['feature'] for d in ds],float)
    if hs.shape!=(n,48) or not np.isfinite(hs).all():raise ValueError('Invalid colour descriptor')
    hs/=np.maximum(np.linalg.norm(hs,axis=1,keepdims=True),1e-9)
    flags=np.asarray([crop_flags[u] for u in uids],bool)
    if flags.shape!=(n,4):raise ValueError('Missing artificial-censor metadata')
    aa,bb=np.triu_indices(n,1)
    keep=np.asarray([compatible(ds[a]['cls'],ds[b]['cls']) for a,b in zip(aa,bb)],bool)
    a,b=aa[keep],bb[keep]
    if not len(a):return np.empty((0,2),int),np.empty((0,DIM))
    cams=np.asarray([d['src'] for d in ds],int);ca,cb=cams[a],cams[b];same=ca==cb
    cp=np.zeros((len(a),15))
    for k,(c,d) in enumerate(CAMERA_PAIRS):cp[:,k]=(np.minimum(ca,cb)==c)&(np.maximum(ca,cb)==d)
    conf=np.asarray([d['conf'] for d in ds]);classes=np.asarray([d['cls'] for d in ds])
    sizes=np.asarray([[cameras[d['src']].width,cameras[d['src']].height] for d in ds],float)
    norm_wh=wh/sizes;centre=(boxes[:,:2]+boxes[:,2:])/2;normal_centre=centre/sizes
    aspect=np.log(wh[:,0]/wh[:,1]);ratio=np.abs(np.log(wh[a]/wh[b]))
    screen_delta=np.abs(centre[a]-centre[b])/np.maximum((wh[a]+wh[b])/2,1.)
    intersection=np.prod(np.maximum(0.,np.minimum(boxes[a,2:],boxes[b,2:])-np.maximum(boxes[a,:2],boxes[b,:2])),axis=1)
    area=np.prod(wh,axis=1)
    iou=intersection/np.maximum(area[a]+area[b]-intersection,1e-9)*same
    ios=intersection/np.maximum(np.minimum(area[a],area[b]),1e-9)*same
    rays=np.asarray([d['ray'] for d in ds],float)
    valid=np.asarray([d['geometry_valid'] for d in ds])&np.isfinite(rays).all(axis=1)
    rays=np.where(valid[:,None],rays,0.);rays/=np.maximum(np.linalg.norm(rays,axis=1,keepdims=True),1e-9)
    ray_ok=valid[a]&valid[b];ra,rb=rays[a],rays[b]
    translations=np.asarray([cameras[d['src']].t for d in ds],float)
    delta=translations[b]-translations[a];baseline=np.linalg.norm(delta,axis=1)
    dot=np.clip(np.sum(ra*rb,axis=1),-1,1);sin=np.sqrt(np.maximum(0,1-dot**2))
    cross=np.cross(ra,rb);epipolar=np.abs(np.sum(delta*cross,axis=1))/np.maximum(baseline,1e-9)
    epi_ok=ray_ok&(baseline>.01)
    # Numerically near-parallel centre rays do NOT provide a precise range.
    tri_ok=epi_ok&(sin>.003)
    da,db=np.sum(ra*delta,axis=1),np.sum(rb*delta,axis=1)
    sa=(da-dot*db)/np.maximum(1-dot**2,1e-9);sb=(dot*da-db)/np.maximum(1-dot**2,1e-9)
    gap=np.linalg.norm(translations[a]+sa[:,None]*ra-(translations[b]+sb[:,None]*rb),axis=1)
    positive=tri_ok&(sa>.5)&(sb>.5)&(sa<200)&(sb<200)
    angular=np.asarray([d['angular_h'] if d['geometry_valid'] else 0. for d in ds],float)
    if not np.isfinite(angular).all():raise ValueError('Invalid angular observation')
    geometry=np.c_[ray_ok,tri_ok,positive,np.where(ray_ok,sin,0.),np.where(ray_ok,dot,0.),
        np.where(epi_ok,epipolar,0.),np.where(tri_ok,np.clip(gap/np.maximum(baseline,.01),0,10),0.),
        np.where(positive,np.log1p(np.maximum(0,np.minimum(sa,sb)))/6,0.),
        np.where(positive,np.log1p(np.maximum(0,np.maximum(sa,sb)))/6,0.)]
    # einsum avoids launching a large BLAS threadpool for a tiny per-frame matrix.
    similarity=np.c_[np.einsum('ij,ij->i',hs[a],hs[b]),np.einsum('ij,ij->i',roi[a],roi[b])]
    x=np.c_[cp,same,classes[a]==classes[b],np.minimum(conf[a],conf[b]),np.maximum(conf[a],conf[b]),
        np.minimum(norm_wh[a],norm_wh[b]),np.maximum(norm_wh[a],norm_wh[b]),
        np.minimum(aspect[a],aspect[b]),np.maximum(aspect[a],aspect[b]),np.clip(ratio,0,5),
        np.clip(screen_delta*same[:,None],0,5),np.minimum(normal_centre[a],normal_centre[b]),np.maximum(normal_centre[a],normal_centre[b]),
        iou,ios,np.minimum(angular[a],angular[b]),np.maximum(angular[a],angular[b]),geometry,
        similarity,(flags[a]|flags[b]).astype(float),(flags[a]&flags[b]).astype(float)]
    if x.shape!=(len(a),DIM) or not np.isfinite(x).all():raise ValueError('Nonfinite pair evidence')
    return np.c_[np.asarray(uids)[a],np.asarray(uids)[b]],x
