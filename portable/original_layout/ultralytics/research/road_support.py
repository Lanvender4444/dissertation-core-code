"""Independent stereo surface/road support. No GT, no z=0 ground assumption.

Plane covariance is a local statistical approximation, not a safety bound.
"""
import numpy as np

from shared_geometry import safe_rays

HEIGHTS = {0:1.7,1:1.4,2:1.55,3:1.5,5:3.2,7:3.0}
NAMES = ('plane_valid','stereo_valid','lr_fraction','depth_log','depth_relative_spread',
         'surface_height_low','surface_height_median','surface_height_high','surface_height_span',
         'same_depth_box_height','same_depth_bottom_height','ground_ray_valid','ground_range_log',
         'ground_ray_relative_sigma','ground_inferred_height','ground_vertical_fit_gap',
         'stereo_ground_range_logratio','range_comparison_valid','plane_mad','plane_inlier_fraction')
DIM = len(NAMES)


def fit_plane(points, seed=20261010, iterations=128):
    p = np.asarray(points,float).reshape(-1,3)
    p = p[np.isfinite(p).all(axis=1)]
    p = p[(p[:,0]>3)&(p[:,0]<50)&(np.abs(p[:,1])<8)&(np.abs(p[:,2])<8)]
    if len(p)<80:
        return dict(valid=False,reason='insufficient_road_points',points=len(p))
    design = np.c_[p[:,:2],np.ones(len(p))]
    tolerance = .10+.002*p[:,0]
    # Spatial balancing prevents dense near points dominating the plane fit.
    bins = np.floor((p[:,:2]-p[:,:2].min(axis=0))/[3.,1.]).astype(int)
    _,inverse,counts = np.unique(bins,axis=0,return_inverse=True,return_counts=True)
    weights = 1/counts[inverse]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        ids = rng.choice(len(p),3,replace=False)
        if abs(np.linalg.det(design[ids]))<.04:
            continue
        beta = np.linalg.solve(design[ids],p[ids,2])
        if np.linalg.norm(beta[:2])>np.tan(np.deg2rad(20)):
            continue
        residual = np.abs(design@beta-p[:,2])/np.sqrt(1+beta[:2]@beta[:2])
        mask = residual<tolerance
        score = float(weights[mask].sum())
        if best is None or score>best[0]:
            best = score,mask
    if best is None:
        return dict(valid=False,reason='no_road_or_degenerate_samples',points=len(p))
    mask = best[1]
    if mask.sum()<80 or mask.mean()<.55:
        return dict(valid=False,reason='insufficient_consensus',points=len(p),inliers=int(mask.sum()))
    q = p[mask]
    if np.ptp(q[:,0])<5 or np.ptp(q[:,1])<1.3:
        return dict(valid=False,reason='insufficient_spatial_support',points=len(p),inliers=len(q))
    x,y,w = design[mask],q[:,2],weights[mask]
    for _ in range(4):
        gram = (x*w[:,None]).T@x
        if np.linalg.cond(gram)>1e7:
            return dict(valid=False,reason='ill_conditioned_plane',points=len(p))
        beta = np.linalg.solve(gram,(x*w[:,None]).T@y)
        residual = y-x@beta
        sigma = max(.025,1.4826*np.median(np.abs(residual-np.median(residual))))
        w = weights[mask]*np.minimum(1.,1.5*sigma/np.maximum(np.abs(residual),1e-9))
    mad = float(np.median(np.abs(residual)))
    if mad>.12 or np.linalg.norm(beta[:2])>np.tan(np.deg2rad(20)):
        return dict(valid=False,reason='nonplanar_or_excessive_tilt',points=len(p),mad=mad)
    # Weight normalization avoids an arbitrary effective-sample reduction.
    wn = w*len(w)/w.sum()
    covariance = sigma**2*np.linalg.inv((x*wn[:,None]).T@x)
    normal = np.r_[-beta[:2],1.]
    norm = np.linalg.norm(normal)
    return dict(valid=True,beta=beta.tolist(),covariance=covariance.tolist(),normal=(normal/norm).tolist(),
                offset=float(-beta[2]/norm),sigma=float(sigma),mad=mad,points=len(p),inliers=len(q),
                inlier_fraction=float(mask.mean()),span_x=float(np.ptp(q[:,0])),span_y=float(np.ptp(q[:,1])))


def stereo_samples(stereo,maps,cid,pixels):
    """Explicit positive-disparity LR consistency on the wide 2/4 pair."""
    if cid not in (2,4):
        raise ValueError('Strong LR evidence only available on existing 2/4 maps')
    sp = stereo.pairs[2,4]
    side,other = ('L',4) if cid==2 else ('R',2)
    uv,disp = stereo.disparity_at(cid,maps[cid],pixels)
    opposite = uv.copy()
    opposite[:,0] += (-disp if cid==2 else disp)
    xy = np.rint(np.nan_to_num(opposite,nan=-1e6)).astype(int)
    valid = np.isfinite(disp)&(disp>=.5)&(disp<191)&(xy[:,0]>=0)&(xy[:,0]<maps[other].shape[1])&(xy[:,1]>=0)&(xy[:,1]<maps[other].shape[0])
    reverse = np.full(len(uv),np.nan)
    reverse[valid] = maps[other][xy[valid,1],xy[valid,0]]/16.
    valid &= np.isfinite(reverse)&(reverse>0)&(np.abs(disp-reverse)<=1.)
    depth = sp.f*sp.B/np.maximum(np.nan_to_num(disp,nan=1),.25)
    points = sp.to_imu(side,uv,depth)
    valid &= np.isfinite(points).all(axis=1)
    return points,valid,disp


def road_pixels(camera,detections):
    x = np.linspace(.30*camera.width,.70*camera.width,40)
    y = np.linspace(.54*camera.height,.79*camera.height,30)
    pts = np.array([(xx,yy) for yy in y for xx in x])
    valid = np.ones(len(pts),bool)
    for d in detections:
        if d['src']!=2 or d['conf']<.05:
            continue
        x1,y1,x2,y2 = d['box']
        valid &= ~((pts[:,0]>=x1-10)&(pts[:,0]<=x2+10)&(pts[:,1]>=y1-10)&(pts[:,1]<=y2+10))
    return pts[valid]


def estimate_road(stereo,maps,detections,cameras):
    pixels = road_pixels(cameras[2],detections)
    points,valid,_ = stereo_samples(stereo,maps,2,pixels)
    plane = fit_plane(points[valid])
    plane.update(road_pixels=len(pixels),lr_road_points=int(valid.sum()))
    if plane['valid']:
        height = np.dot(plane['normal'],cameras[2].t)+plane['offset']
        if not .2<height<4.:
            return dict(valid=False,reason='plane_not_below_camera_or_unreasonable_support',camera_height=float(height),
                        road_pixels=len(pixels),lr_road_points=int(valid.sum()))
        plane['camera_height'] = float(height)
    return plane


def ground_ray(camera,box,plane):
    x1,y1,x2,y2 = box
    rays,valid = safe_rays(camera,[[(x1+x2)/2,y2],[(x1+x2)/2,y1]])
    if not valid.all() or not plane['valid']:
        return None
    n = np.asarray(plane['normal'])
    den = float(n@rays[0])
    if den>=-1e-4:
        return None
    distance = -float(n@camera.t+plane['offset'])/den
    if not .5<distance<180:
        return None
    bottom = camera.t+distance*rays[0]
    design = np.r_[bottom[:2],1.]
    sigma = np.sqrt(plane['sigma']**2+design@np.asarray(plane['covariance'])@design)
    height_ray = np.column_stack([rays[1],-n])
    top_range,height = np.linalg.lstsq(height_ray,bottom-camera.t,rcond=None)[0]
    gap = np.linalg.norm(camera.t+top_range*rays[1]-(bottom+height*n))
    return dict(range=float(distance),relative_sigma=float(sigma/(abs(den)*distance)),
                height=float(height),gap=float(gap),bottom=bottom)


def descriptor(d,plane,stereo,maps,cameras):
    x = np.zeros(DIM)
    if not plane['valid']:
        return x
    x[0],x[18],x[19] = 1.,plane['mad'],plane['inlier_fraction']
    camera,expected = cameras[d['src']],HEIGHTS[d['cls']]
    foot = ground_ray(camera,d['box'],plane)
    if foot is not None:
        x[11:16] = [1.,np.log1p(foot['range'])/6,min(foot['relative_sigma'],5),
                    np.clip(foot['height']/expected,-10,10),min(foot['gap']/expected,10)]
    if d['src'] not in (2,4):
        return x  # Missing stereo is NOT a fabricated precise depth.
    x1,y1,x2,y2 = d['box']
    pixels = np.array([(xx,yy) for yy in np.linspace(y1+.1*(y2-y1),y2-.05*(y2-y1),9)
                       for xx in np.linspace(x1+.1*(x2-x1),x2-.1*(x2-x1),9)])
    points,valid,disp = stereo_samples(stereo,maps,d['src'],pixels)
    x[2] = valid.mean()
    if valid.sum()<12:
        return x
    q,dv = points[valid],disp[valid]
    near = float(np.percentile(dv,65))
    # Foreground-mode restriction; does not prove semantic foreground identity.
    keep = np.abs(dv-near)<=max(.75,.5*np.subtract(*np.percentile(dv,[80,35])))
    if keep.sum()<12:
        return x
    q,dv = q[keep],dv[keep]
    distances = np.linalg.norm(q-camera.t,axis=1)
    distance = float(np.median(distances))
    spread = float(np.subtract(*np.percentile(dv,[80,35])))
    x[1],x[3],x[4] = 1.,np.log1p(distance)/6,min(max(.35,.5*spread)/near,5)
    heights = q@np.asarray(plane['normal'])+plane['offset']
    low,mid,high = np.percentile(heights,[10,50,90])
    x[5:9] = np.clip(np.array([low,mid,high,high-low])/expected,-10,10)
    rays,okay = safe_rays(camera,[[(x1+x2)/2,y1],[(x1+x2)/2,y2]])
    if okay.all():
        end = camera.t+distance*rays
        x[9:11] = np.clip([(end[0]-end[1])@plane['normal']/expected,
                          (end[1]@plane['normal']+plane['offset'])/expected],-10,10)
    if foot is not None:
        x[16],x[17] = np.clip(np.log(distance/foot['range']),-5,5),1.
    if not np.isfinite(x).all():
        raise ValueError('Invalid physical evidence')
    return x
