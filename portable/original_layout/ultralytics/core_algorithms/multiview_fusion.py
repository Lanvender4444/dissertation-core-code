"""Lightweight, detector-independent association; GT never enters this module."""
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from multiview_geometry import triangulate

VEHICLES={2,5,7}
HEIGHT={0:1.7,1:1.5,2:1.55,3:1.6,5:3.2,7:3.5}

def compatible(a,b):
    return a==b or {a,b}<=VEHICLES

def feature(image,box):
    x1,y1,x2,y2=np.asarray(box,int)
    crop=image[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
    if crop.size==0: return np.zeros(48)
    hsv=cv2.cvtColor(cv2.resize(crop,(24,32)),cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[12,4],[0,180,0,256]).flatten()
    return np.sqrt(hist/(hist.sum()+1e-9))

def prepare(dets,cameras):
    for d in dets:
        c=cameras[d['src']]; x1,y1,x2,y2=d['box']
        rays=c.rays([[(x1+x2)/2,(y1+y2)/2],[(x1+x2)/2,y1],[(x1+x2)/2,y2]])
        d['ray']=rays[0]
        d['angular_h']=float(np.arccos(np.clip(rays[1]@rays[2],-1,1)))

def pair_cost(a,b,cameras,appearance=True):
    if not compatible(a['cls'],b['cls']): return None
    ra,rb=a['ray'],b['ray']; ca,cb=cameras[a['src']],cameras[b['src']]
    if not np.isfinite([*ra,*rb]).all(): return None
    normal=np.cross(cb.t-ca.t,ra)
    epi=abs(normal@rb)/(np.linalg.norm(normal)+1e-9)
    h=max(min(a['angular_h'],b['angular_h']),.01)
    if epi>max(.008,.30*h): return None
    ratio=(a['angular_h']+1e-6)/(b['angular_h']+1e-6)
    if not .33<ratio<3: return None
    P,gap,depths,angle=triangulate(ca,ra,cb,rb)
    if min(depths)<.5 or max(depths)>300 or angle<.0005: return None
    if gap>max(.5,.12*min(depths)): return None
    physical_h=min(depths)*h
    if not .25*HEIGHT[a['cls']] < physical_h < 3.5*HEIGHT[a['cls']]: return None
    app=1-float(np.dot(a['feature'],b['feature']))
    if appearance and app>.8: return None
    cost=epi/h+.2*abs(np.log(ratio))+(0.35*app if appearance else 0)
    if cost>.65: return None
    return float(cost),P,angle

def angular_overlap(a,b,cameras):
    """Rotation-only box overlap for front cameras; NOT a depth measurement."""
    boxes=[]
    for d in (a,b):
        x1,y1,x2,y2=d['box']
        rays=cameras[d['src']].rays([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])@cameras[3].R
        if not np.isfinite(rays).all() or (rays[:,2]<=0).any(): return 0.
        xy=rays[:,:2]/rays[:,2,None]
        boxes.append(np.r_[xy.min(axis=0),xy.max(axis=0)])
    aa,bb=boxes
    wh=np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2]))
    inter=np.prod(wh)
    return float(inter/(np.prod(aa[2:]-aa[:2])+np.prod(bb[2:]-bb[:2])-inter+1e-12))


def fuse(dets,cameras,appearance=True,overlap_fallback=None):
    prepare(dets,cameras)
    by={c:[i for i,d in enumerate(dets) if d['src']==c] for c in cameras}
    edges={}; selected=[]
    ids=list(cameras)
    for ci,ca in enumerate(ids):
        for cb in ids[ci+1:]:
            aa,bb=by[ca],by[cb]
            if not aa or not bb: continue
            # Dummy columns make unmatched assignments explicit.
            cost=np.full((len(aa),len(bb)+len(aa)),.7)
            for i,a in enumerate(aa):
                for j,b in enumerate(bb):
                    e=pair_cost(dets[a],dets[b],cameras,appearance)
                    if e is None and overlap_fallback is not None and ca in (2,3,4) and cb in (2,3,4):
                        da,db=dets[a],dets[b]
                        if compatible(da['cls'],db['cls']):
                            overlap=angular_overlap(da,db,cameras)
                            app=1-float(np.dot(da['feature'],db['feature']))
                            if overlap>=overlap_fallback and app<.5:
                                # Similar rays do not establish positive metric depth.
                                e=(.4+.2*(1-overlap)+.05*app,None,0.)
                    cost[i,j]=e[0] if e else 1e3
                    if e: edges[tuple(sorted((a,b)))]=e
            rows,cols=linear_sum_assignment(cost)
            for i,j in zip(rows,cols):
                if j<len(bb) and cost[i,j]<.7: selected.append((cost[i,j],aa[i],bb[j]))
    groups=[{i} for i in range(len(dets))]
    for _,a,b in sorted(selected):
        ga=next(g for g in groups if a in g); gb=next(g for g in groups if b in g)
        if ga is gb: continue
        if {dets[i]['src'] for i in ga}&{dets[i]['src'] for i in gb}: continue
        # Complete-link consistency prevents chains merging two different cars.
        if not all(tuple(sorted((i,j))) in edges for i in ga for j in gb): continue
        ga.update(gb); groups.remove(gb)
    out=[]
    for g in groups:
        ds=[dets[i] for i in sorted(g)]; src=sorted(d['src'] for d in ds)
        es=[e for (a,b),e in edges.items() if a in g and b in g]
        P=max(es,key=lambda e:e[2])[1] if es else None
        visible=[]
        if P is not None:
            visible=[c for c,cam in cameras.items() if cam.project([P])[1][0]]
        else:
            # Visibility over a broad depth interval; absence is not falsehood.
            d=ds[0]; c=cameras[d['src']]
            for cid,other in cameras.items():
                if any(other.project([c.t+z*d['ray']])[1][0] for z in (5,15,40,100)):
                    visible.append(cid)
        label='FULL' if {2,3,4}<=set(src) else 'HALF' if len(src)>=2 else 'UNIQUE' if len(visible)==1 else 'UNVERIFIED'
        out.append({'dets':ds,'src':src,'cls':max(ds,key=lambda d:d['conf'])['cls'],
                    'conf':max(d['conf'] for d in ds),'label':label,'point':P,
                    'potentially_visible':visible,'missing_detection':[c for c in visible if c not in src],
                    'geometry_cost':float(np.mean([e[0] for e in es])) if es else None})
    return out

def select(objects,thresholds):
    # three-view / two-view / single-view thresholds selected on development only
    return [o for o in objects if o['conf']>=thresholds[0 if len(o['src'])>=3 else 1 if len(o['src'])==2 else 2]]


def staged_fusion(front,dets,cameras,front_thresholds,side_threshold):
    """Freeze accepted front objects before extending with side detections.

    There is no second global confidence filter: adding side cameras must not
    silently alter the first-stage front admission thresholds.
    """
    accepted=select(front,front_thresholds)
    side=[d for d in dets if d['src'] in (1,5) and
          d['conf']>=(side_threshold[d['src']] if isinstance(side_threshold,dict) else side_threshold)]
    return extend_front(accepted,side,cameras)


def preserve_single_camera(dets,cameras,thresholds,method='graph',overlap_fallback=None):
    """Associate accepted single-camera observations without deleting any.

    Per-camera thresholds must be selected on development data, not this frame's
    ground truth. Group membership never replaces source boxes. Projection-only
    display is not a new detector measurement. This preserves native-view recall
    by construction, not global identity correctness (which still needs testing).
    """
    accepted=[d for d in dets if d['conf']>=thresholds[d['src']]]
    if method=='hybrid':
        from legacy_adapter import core,legacy
        rect=core.TriRectifier(core.camera_params)
        _,front=legacy([d for d in accepted if d['src'] in (2,3,4)],rect)
        return extend_front(front,accepted,cameras)
    return fuse(accepted,cameras,overlap_fallback=overlap_fallback)

def extend_front(front,dets,cameras,reprojection_gate=None):
    """Preserve established front triplets, then associate each fisheye once."""
    prepare(dets,cameras)
    for o in front:
        prepare(o['dets'],cameras)
        o['point']=None
        ds=o['dets']
        options=[pair_cost(a,b,cameras) for i,a in enumerate(ds) for b in ds[i+1:]]
        options=[e for e in options if e]
        if options: o['point']=max(options,key=lambda e:e[2])[1]
        o['label']='FULL' if {2,3,4}<=set(o['src']) else 'HALF' if len(o['src'])>1 else 'UNVERIFIED'
        o['potentially_visible']=list(o['src']); o['missing_detection']=[]
    for cid in (1,5):
        side=[d for d in dets if d['src']==cid]
        if not side: continue
        n=len(front); cost=np.full((len(side),n+len(side)),.5); evidence={}
        for i,d in enumerate(side):
            for j,o in enumerate(front):
                if reprojection_gate is not None and o['point'] is not None:
                    uv,valid=cameras[cid].project([o['point']])
                    x1,y1,x2,y2=d['box'];center=np.array([(x1+x2)/2,(y1+y2)/2])
                    residual=np.max(np.abs(uv[0]-center)/np.maximum([x2-x1,y2-y1],1))
                    if not valid[0] or residual>reprojection_gate:
                        cost[i,j]=1e3;continue
                options=[pair_cost(d,member,cameras) for member in o['dets']]
                options=[e for e in options if e]
                if not options: cost[i,j]=1e3; continue
                e=min(options,key=lambda e:e[0]); cost[i,j]=e[0]; evidence[i,j]=e
        rows,cols=linear_sum_assignment(cost); matched=set()
        for i,j in zip(rows,cols):
            if j>=n or cost[i,j]>=.5: continue
            o=front[j]; o['dets'].append(side[i]); o['src']=sorted(o['src']+[cid])
            o['conf']=max(o['conf'],side[i]['conf'])
            if o['point'] is None: o['point']=evidence[i,j][1]
            if o['label']!='FULL': o['label']='HALF'
            matched.add(i)
        for i,d in enumerate(side):
            if i not in matched:
                front.extend(fuse([d],cameras))
    for o in front:
        o['cls']=max(o['dets'],key=lambda d:d['conf'])['cls']
        o.setdefault('geometry_cost',None)
        if o['point'] is not None:
            visible=[c for c,cam in cameras.items() if cam.project([o['point']])[1][0]]
        else:
            visible=sorted({c for d in o['dets'] for c,cam in cameras.items()
                            if any(cam.project([cameras[d['src']].t+z*d['ray']])[1][0] for z in (5,15,40,100))})
        o['potentially_visible']=visible
        o['missing_detection']=[c for c in visible if c not in o['src']]
        if len(o['src'])==1: o['label']='UNIQUE' if len(visible)==1 else 'UNVERIFIED'
    return front
