"""Experimental stereo visual ego-motion and conservative temporal confirmation.

No timestamp/INS association is guessed. Transforms are estimated between the
actual sampled images. Motion failure -> no filtering. GT is never used.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiview_geometry import triangulate
from multiview_fusion import compatible

def features(image,dets,width=1232):
    scale=width/image.shape[1];gray=cv2.resize(image,(width,round(image.shape[0]*scale)))
    mask=np.full(gray.shape,255,np.uint8);mask[int(.82*len(mask)):]=0
    for d in dets:
        x1,y1,x2,y2=(np.asarray(d['box'])*scale).astype(int)
        cv2.rectangle(mask,(max(0,x1-8),max(0,y1-8)),(x2+8,y2+8),0,-1)
    key,desc=cv2.ORB_create(nfeatures=2200).detectAndCompute(gray,mask)
    uv=np.array([k.pt for k in key],float).reshape(-1,2)/scale
    return uv,desc

def matches(a,b):
    if a is None or b is None or len(a)<2 or len(b)<2:return {}
    forward=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a,b,k=2)
    reverse=cv2.BFMatcher(cv2.NORM_HAMMING).match(b,a);back={m.queryIdx:m.trainIdx for m in reverse}
    return {m.queryIdx:m.trainIdx for pair in forward if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance and back.get(m.trainIdx)==m.queryIdx}

def motion(previous,current,cameras):
    """previous/current map cid -> (pixel coordinates, descriptors)."""
    left,dl=previous[2];right,dr=previous[4];now,dn=current[2]
    stereo=matches(dl,dr);temporal=matches(dl,dn);points=[];pixels=[]
    for i,j in stereo.items():
        if i not in temporal:continue
        ca,cb=cameras[2],cameras[4]
        p,gap,depths,angle=triangulate(ca,ca.rays([left[i]])[0],cb,cb.rays([right[j]])[0])
        if not np.isfinite(p).all() or min(depths)<2 or max(depths)>80 or angle<.003 or gap>max(.08,.01*min(depths)):continue
        points.append(p);pixels.append(now[temporal[i]])
    info=dict(status='unknown',stereo_matches=len(stereo),temporal_matches=len(temporal),correspondences=len(points))
    if len(points)<12:return None,info
    cam=cameras[2];xyz=np.asarray(points);uv=np.asarray(pixels)
    cv2.setRNGSeed(20260923)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv,cam.K,cam.D,iterationsCount=200,reprojectionError=5.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None:return None,info
    indices=inliers.ravel();info['inliers']=len(indices)
    if len(indices)<12 or len(indices)/len(points)<.5:return None,info
    rvec,tvec=cv2.solvePnPRefineLM(xyz[indices],uv[indices],cam.K,cam.D,rvec,tvec)
    projected=cv2.projectPoints(xyz[indices],rvec,tvec,cam.K,cam.D)[0].reshape(-1,2)
    error=float(np.median(np.linalg.norm(projected-uv[indices],axis=1)))
    R=cam.R@cv2.Rodrigues(rvec)[0];t=cam.R@tvec.ravel()+cam.t
    angle=np.arccos(np.clip((np.trace(R)-1)/2,-1,1));distance=np.linalg.norm(t)
    info.update(median_reprojection_px=error,translation_m=float(distance),rotation_deg=float(np.rad2deg(angle)))
    if error>3 or distance>15 or angle>np.deg2rad(20):return None,info
    info.update(status='estimated',R=R.tolist(),t=t.tolist())
    return (R,t),info

def temporal_filter(previous,current,previous_geometry,current_geometry,transform,cameras,low_conf=.45):
    if transform is None:return list(current),dict(eligible=0,supported=0,removed=0,reason='motion_unknown_or_initial')
    R,t=transform;cost=np.full((len(current),len(previous)+len(current)),10.)
    eligible=[]
    for i,(o,g) in enumerate(zip(current,current_geometry)):
        if g is None:continue
        p,s=g
        # Object centre would have been in at least two previous camera images.
        before=R.T@(p-t)
        visible=sum(bool(c.project([before])[1][0]) for c in cameras.values())
        if o['conf']<low_conf and visible>=2:eligible.append(i)
        for j,(old,oldg) in enumerate(zip(previous,previous_geometry)):
            if oldg is None or not compatible(o['cls'],old['cls']):continue
            pp,ss=oldg;pred=R@pp+t;error=(p-pred)[:2]
            # Fixed displacement allowance, NOT a velocity model (timestamps absent).
            cov=(s+R@ss@R.T)[:2,:2]+np.eye(2)*4.
            d=float(error@np.linalg.solve(cov,error))
            if d<=9 and np.linalg.norm(error)<=max(3.,.1*np.linalg.norm(p)):cost[i,j]=d/9
    rr,cc=linear_sum_assignment(cost);supported={int(i) for i,j in zip(rr,cc) if j<len(previous) and cost[i,j]<=1}
    removed=set(eligible)-supported
    return [o for i,o in enumerate(current) if i not in removed],dict(eligible=len(eligible),supported=len(supported),removed=len(removed),removed_indices=sorted(removed),reason='low_conf_temporal_confirmation')
