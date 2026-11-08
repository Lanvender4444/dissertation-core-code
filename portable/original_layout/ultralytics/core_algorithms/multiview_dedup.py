"""Conservative object-level NMS using common-view agreement and depth veto.

All thresholds are engineering gates, not calibrated statistical guarantees.
Ground truth is never read by this module. Single-camera evidence is allowed
only in the explicit baseline mode; multi-view mode requires >=2 shared views.
"""
import itertools
import numpy as np
from multiview_geometry import triangulate
from multiview_fusion import compatible

def box_iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    area_a=np.prod(np.maximum(0,a[2:]-a[:2]));area_b=np.prod(np.maximum(0,b[2:]-b[:2]))
    return float(inter/max(area_a+area_b-inter,1e-12))

def position_uncertainty(obj,cameras):
    """First-order pixel-noise propagation; returns None if ill-conditioned.

    Sigma uses 2px + 5% box dimensions to model uncertain detector centres.
    Calibration uncertainty is not modeled, so depth is a veto, never proof.
    """
    best=None
    for a,b in itertools.combinations(obj['dets'],2):
        ca,cb=cameras[a['src']],cameras[b['src']]
        ba,bb=np.asarray(a['box']),np.asarray(b['box'])
        uv=np.r_[(ba[:2]+ba[2:])/2,(bb[:2]+bb[2:])/2]
        def solve(x): return triangulate(ca,ca.rays([x[:2]])[0],cb,cb.rays([x[2:]])[0])
        p,gap,depths,angle=solve(uv)
        if not np.isfinite(p).all() or min(depths)<.5 or max(depths)>300 or angle<.005: continue
        if gap>max(.5,.1*min(depths)): continue
        if best is not None and angle<=best[0]: continue
        jacobian=np.zeros((3,4))
        for k in range(4):
            delta=np.zeros(4);delta[k]=1.
            jacobian[:,k]=(solve(uv+delta)[0]-solve(uv-delta)[0])/2
        sigma=2+.05*np.r_[ba[2:]-ba[:2],bb[2:]-bb[:2]]
        covariance=(jacobian*sigma)@(jacobian*sigma).T+np.eye(3)*.05**2
        if not np.isfinite(covariance).all(): continue
        relative_std=np.sqrt(np.linalg.eigvalsh(covariance).max())/min(depths)
        if relative_std>.3: continue
        best=(angle,p,covariance)
    return None if best is None else best[1:]

def duplicate_evidence(a,b,iou_threshold,min_shared=2,geometry_a=None,geometry_b=None):
    aa={d['src']:d for d in a['dets']};bb={d['src']:d for d in b['dets']}
    shared=sorted(set(aa)&set(bb))
    if len(shared)<min_shared: return False
    for cid in shared:
        if not compatible(aa[cid]['cls'],bb[cid]['cls']): return False
        if box_iou(aa[cid]['box'],bb[cid]['box'])<iou_threshold: return False
    if geometry_a is not None and geometry_b is not None:
        pa,sa=geometry_a;pb,sb=geometry_b;delta=pa-pb
        if float(delta@np.linalg.solve(sa+sb,delta))>9.: return False
    return True

def deduplicate(objects,iou_threshold=.6,cameras=None,depth_veto=False,min_shared=2):
    """Complete-link greedy suppression, retaining original representative boxes."""
    if min_shared<1 or not 0<iou_threshold<=1: raise ValueError('Invalid duplicate gates')
    geometry=[position_uncertainty(o,cameras) if depth_veto and cameras else None for o in objects]
    order=sorted(range(len(objects)),key=lambda i:(objects[i]['conf'],len(objects[i]['src'])) ,reverse=True)
    clusters=[]
    for i in order:
        for cluster in clusters:
            if all(duplicate_evidence(objects[i],objects[j],iou_threshold,min_shared,geometry[i],geometry[j]) for j in cluster):
                cluster.append(i);break
        else: clusters.append([i])
    out=[]
    for cluster in clusters:
        o=dict(objects[cluster[0]])
        o['suppressed_group_indices']=cluster[1:]
        out.append(o)
    return out
