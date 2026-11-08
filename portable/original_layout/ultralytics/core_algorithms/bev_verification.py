"""Sparse BEV evidence and leave-one-camera-out residual verification.

Experimental, training-free. Never uses GT. Geometry gates are engineering
priors, not calibrated probabilities. Unknown geometry must never delete a box.
"""
import numpy as np
from multiview_dedup import box_iou,position_uncertainty
from multiview_fusion import compatible

DEFAULT_GATE=dict(iou=.9,appearance=.9,mahalanobis2=4.,max_center_fraction=.25)

def ray_plane(origin,ray,normal,offset,min_incidence=.02,max_range=150.):
    """Intersection with EXPLICIT n.P+d=0; no implicit IMU z=0 assumption."""
    origin=np.asarray(origin,float);ray=np.asarray(ray,float);normal=np.asarray(normal,float)
    norm=np.linalg.norm(normal);rnorm=np.linalg.norm(ray)
    if not np.isfinite(np.r_[origin,ray,normal,offset]).all() or norm<1e-12 or rnorm<1e-12:return None
    normal=normal/norm;offset=offset/norm;ray=ray/rnorm;den=normal@ray
    if abs(den)<min_incidence:return None
    distance=-(normal@origin+offset)/den
    if not 0<distance<=max_range:return None
    return origin+distance*ray

def project_distribution(camera,geometry):
    if geometry is None:return None
    point,cov=geometry;uv,valid=camera.project([point])
    if not valid[0]:return None
    jac=np.zeros((2,3));step=.001
    for k in range(3):
        delta=np.zeros(3);delta[k]=step
        plus=camera.project([point+delta])[0][0];minus=camera.project([point-delta])[0][0]
        jac[:,k]=(plus-minus)/(2*step)
    image_cov=jac@cov@jac.T+np.eye(2)*4
    if not np.isfinite(image_cov).all():return None
    return uv[0],image_cov

def appearance_similarity(a,b):
    x=np.asarray(a.get('feature',[]),float);y=np.asarray(b.get('feature',[]),float)
    if x.shape!=y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():return None
    norm=np.linalg.norm(x)*np.linalg.norm(y)
    return None if norm<1e-12 else float(np.clip(x@y/norm,-1,1))

def residual_evidence(single,anchor,cameras,geometry,gate):
    """Anchor geometry must exclude the singleton's camera."""
    d=single['dets'][0];cid=d['src'];other=next((a for a in anchor['dets'] if a['src']==cid),None)
    if other is None or not compatible(d['cls'],other['cls']):return None
    overlap=box_iou(d['box'],other['box'])
    if overlap<gate['iou']:return None
    appearance=appearance_similarity(d,other)
    if appearance is None or appearance<gate['appearance']:return None
    projected=project_distribution(cameras[cid],geometry)
    if projected is None:return None
    uv,cov=projected
    residuals=[]
    # Both the retained anchor member and residual must agree with the excluded-view prediction.
    for member in (d,other):
        box=np.asarray(member['box'],float);size=box[2:]-box[:2]
        if np.any(size<=0):return None
        center=(box[:2]+box[2:])/2;error=center-uv
        if np.max(np.abs(error)/size)>gate['max_center_fraction']:return None
        total_cov=cov+np.diag((2+.05*size)**2)
        distance=float(error@np.linalg.solve(total_cov,error))
        if distance>gate['mahalanobis2']:return None
        residuals.append(distance)
    return dict(shared_camera=cid,iou=overlap,appearance=appearance,mahalanobis2=residuals,projected_center=uv.tolist(),
        geometry_source_cameras=[c for c in anchor['src'] if c!=cid])

def bev_state(obj,cameras,geometry):
    if geometry is None:return dict(status='unknown',reason='insufficient_or_ill_conditioned_geometry')
    point,cov=geometry
    projections={}
    for cid,cam in cameras.items():
        uv,valid=cam.project([point])
        projections[cid]=dict(projected_center=uv[0].tolist() if np.isfinite(uv[0]).all() else None,
            center_in_image=bool(valid[0]),has_associated_detection=cid in obj['src'])
    return dict(status='estimated',point_imu=point.tolist(),xy_imu=point[:2].tolist(),covariance_3d=cov.tolist(),
        covariance_xy=cov[:2,:2].tolist(),kind='box_center_3d_xy_not_ground_contact',projections=projections)

def verify(objects,cameras,gate=None,mode='audit'):
    """audit adds evidence only; suppress removes uniquely supported residual singletons.

    Never joins two multi-camera groups, never creates members or synthetic detections.
    Suppression uses only geometry from OTHER cameras, with an unambiguous anchor.
    """
    if mode not in ('audit','suppress'):raise ValueError('mode must be audit or suppress')
    gate=dict(DEFAULT_GATE if gate is None else gate)
    if not 0<gate['iou']<=1 or not -1<=gate['appearance']<=1 or gate['mahalanobis2']<=0 or gate['max_center_fraction']<=0:raise ValueError('Invalid gates')
    anchors=[i for i,o in enumerate(objects) if len(set(o['src']))>=3]
    output=[];audit=[];geometry_cache={}
    def geometry(i,excluded=None):
        key=(i,excluded)
        if key not in geometry_cache:
            members=[d for d in objects[i]['dets'] if d['src']!=excluded]
            geometry_cache[key]=position_uncertainty(dict(dets=members),cameras) if len(members)>=2 else None
        return geometry_cache[key]
    for i,o in enumerate(objects):
        candidates=[]
        if len(o['src'])==1:
            cid=o['src'][0]
            for j in anchors:
                # Cheap image gates BEFORE computing expensive uncertainty.
                member=next((d for d in objects[j]['dets'] if d['src']==cid),None)
                if member is None or box_iou(o['dets'][0]['box'],member['box'])<gate['iou']:continue
                ev=residual_evidence(o,objects[j],cameras,geometry(j,cid),gate)
                if ev is not None:candidates.append(dict(anchor_index=j,**ev))
        action='suppress_residual' if mode=='suppress' and len(candidates)==1 else 'keep'
        audit.append(dict(index=i,id=o.get('id'),action=action,candidates=candidates,
            reason='unique_other_camera_geometry_support' if len(candidates)==1 else 'ambiguous_anchors' if len(candidates)>1 else 'no_sufficient_evidence'))
        if action=='keep':
            copy=dict(o);copy['bev']=bev_state(o,cameras,geometry(i));output.append(copy)
    return output,audit
