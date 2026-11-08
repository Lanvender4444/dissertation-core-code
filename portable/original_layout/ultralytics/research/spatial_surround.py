"""Low-cost geometric display: bowl surround + textured planar BEV.

No inference, association, footprint/depth estimation or generated pixels.
The surface is a DISPLAY assumption. Metric object states remain unchanged.
"""
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from five_camera_display import colour, text, bearing, CLASS_NAMES, CAM_ORDER
from spatial_observation_overlay import BowlObservations

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_plane(ground_z=None, reference='00000000'):
    """Use a signed GT-free stereo fit once; never silently assume z=0."""
    if ground_z is not None:
        if not np.isfinite(ground_z) or not -4<ground_z<1:
            raise ValueError('Explicit ground-z must be finite in (-4,1) IMU metres')
        return np.array([0.,0.,ground_z]),dict(kind='user_assumed_plane',beta=[0.,0.,ground_z])
    report_path=ROOT/'ultralytics/algorithm/evaluation_reports/road_support_admission_results_20261010.json'
    report=json.loads(report_path.read_text())
    matches=[(p,h) for p,h in report['physical_sha256'].items() if Path(p).stem==reference]
    if not report['complete'] or len(matches)!=1:
        raise ValueError('Missing signed display plane; supply --ground-z explicitly')
    path,expected=matches[0];path=ROOT/path
    if digest(path)!=expected:raise ValueError('Road reference checksum mismatch')
    record=json.loads(path.read_text());plane=record['plane']
    cache=ROOT/'ultralytics/experiments/road_support_20261010/full01/protocol.json'
    if record['frame']!=reference or record['protocol_sha256']!=digest(cache):
        raise ValueError('Road reference protocol/frame mismatch')
    cfg=json.loads(cache.read_text())
    for p,h in {**cfg['source_sha256'],**cfg['calibration_sha256'],**record['input_sha256']}.items():
        if digest(ROOT/p)!=h:raise ValueError(f'Changed road reference input: {p}')
    beta=np.asarray(plane.get('beta',[]),float)
    if not plane['valid'] or beta.shape!=(3,) or not np.isfinite(beta).all():
        raise ValueError('Unreliable display plane; supply --ground-z explicitly')
    return beta,dict(kind='fixed_cached_stereo_plane',reference=reference,beta=beta.tolist(),
        record=str(path.relative_to(ROOT)),record_sha256=expected,protocol_sha256=digest(cache),
        mad=plane['mad'],span_x=plane['span_x'],span_y=plane['span_y'],
        limitation='Local fit frozen for display; not a current-frame or globally measured road surface')


class SpatialSurround:
    def __init__(self,cameras,beta,width=960,height=540,bev_width=540):
        started=time.perf_counter()
        self.cameras=cameras;self.beta=np.asarray(beta,float)
        if self.beta.shape!=(3,) or not np.isfinite(self.beta).all():raise ValueError('Explicit finite plane required')
        self.width=width;self.height=height;self.bev_width=bev_width
        self.thumb=(616,514);self.diagnostics={}
        self.eye=np.array([-9.,0.,self.ground([-9.,0.])+8.])
        target=np.array([12.,0.,self.ground([12.,0.])])
        forward=target-self.eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
        up=np.cross(right,forward);self.basis=np.column_stack((right,up,forward))
        self.focal=(width/2)/np.tan(np.deg2rad(98)/2)
        x,y=np.meshgrid(np.arange(width),np.arange(height))
        local=np.stack(((x-(width-1)/2)/self.focal,-(y-(height-1)/2)/self.focal,np.ones_like(x)),axis=-1)
        rays=local.reshape(-1,3)@self.basis.T;rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        points,okay=self.intersect(rays)
        self.bowl=self.lookup(points,okay,height,width)
        x,y=np.meshgrid(np.linspace(25,-25,bev_width),np.linspace(70,-8,height))
        points=np.column_stack((y.ravel(),x.ravel(),self.ground(np.column_stack((y.ravel(),x.ravel())))))
        # Limit texture extrapolation; farther object points can still use grid.
        near=(points[:,0]<35)&(np.abs(points[:,1])<18)
        self.bev=self.lookup(points,near,height,bev_width)
        self.bowl_base=self.background(height,width);self.bev_base=self.background(height,bev_width)
        # This pattern marks withheld OBJECT texture, not observed road colour.
        iy,ix=np.indices((height,bev_width))
        self.withheld_base=np.empty_like(self.bev_base)
        self.withheld_base[:]=(92,76,54)
        self.withheld_base[(ix+iy)%12<3]=(132,111,81)
        self.observations=BowlObservations(self)
        self.init_s=time.perf_counter()-started

    def ground(self,xy):
        p=np.asarray(xy);return p[...,0]*self.beta[0]+p[...,1]*self.beta[1]+self.beta[2]

    def surface(self,xy):
        p=np.asarray(xy);r=np.linalg.norm(p-[2.,0.],axis=-1)
        return self.ground(p)+.045*np.maximum(r-12.,0)**2

    def intersect(self,rays):
        """First forward hit on bowl; LUT construction only, no per-frame raycast."""
        rays=np.asarray(rays);n=len(rays);lo=np.zeros(n);hi=np.zeros(n);hit=np.zeros(n,bool)
        for distance in np.linspace(0,90,91)[1:]:
            p=self.eye+distance*rays
            f=p[:,2]-self.surface(p[:,:2]);new=(~hit)&(f<=0)
            hi[new]=distance;lo[new]=distance-1.;hit|=new
        for _ in range(16):
            mid=(lo+hi)/2;p=self.eye+mid[:,None]*rays
            above=p[:,2]>self.surface(p[:,:2]);lo=np.where(above,mid,lo);hi=np.where(above,hi,mid)
        p=self.eye+((lo+hi)/2)[:,None]*rays
        hit&=np.linalg.norm(p[:,:2]-[2.,0.],axis=1)<38
        return p,hit

    def lookup(self,points,okay,h,w):
        scores=[];uvs=[]
        body=(points[:,0]>-1.2)&(points[:,0]<4.)&(np.abs(points[:,1])<1.15)
        for cid in CAM_ORDER:
            cam=self.cameras[cid];uv,v=cam.project(points)
            direction=points-cam.t;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
            facing=(direction@cam.R)[:,2]
            # Conservative front-cone and inward pixels avoid Mei folded edges.
            v&=okay&~body&(facing>.15)&(uv[:,0]>3)&(uv[:,0]<cam.width-4)&(uv[:,1]>3)&(uv[:,1]<cam.height-4)
            score=facing.copy()
            if cid==3:score+=.08
            score[~v]=-np.inf;scores.append(score);uvs.append(uv)
        scores=np.asarray(scores);valid=np.isfinite(scores.max(axis=0));owner=scores.argmax(axis=0)
        packed=np.zeros((len(points),2),np.float32)
        for i,cid in enumerate(CAM_ORDER):
            selected=valid&(owner==i);cam=self.cameras[cid]
            packed[selected]=(uvs[i][selected]+.5)*[self.thumb[0]/cam.width,self.thumb[1]/cam.height]-.5
            packed[selected,1]+=i*self.thumb[1]
        packed[~valid]=-1;packed=packed.reshape(h,w,2)
        maps=cv2.convertMaps(packed,None,cv2.CV_16SC2)
        return dict(maps=maps,valid=valid.reshape(h,w),owner=owner.reshape(h,w),
                    coverage=float(valid.mean()),lut_bytes=sum(m.nbytes for m in maps))

    @staticmethod
    def background(h,w):
        y,x=np.indices((h,w));checker=((x//16+y//16)%2)*5+24
        return np.repeat(checker[:,:,None],3,axis=2).astype(np.uint8)

    def project(self,points):
        q=(np.asarray(points)-self.eye)@self.basis
        safe=np.maximum(q[:,2],1e-8)
        uv=np.column_stack(((self.width-1)/2+self.focal*q[:,0]/safe,
                            (self.height-1)/2-self.focal*q[:,1]/safe))
        good=(q[:,2]>.1)&np.isfinite(uv).all(axis=1)
        return uv,good

    def bev_pixel(self,points):
        p=np.asarray(points);return np.column_stack(((25-p[:,1])/50*(self.bev_width-1),
                                                     (70-p[:,0])/78*(self.height-1)))

    def raster(self,packed,blocked,lut,background,withheld_background=None):
        image=cv2.remap(packed,*lut['maps'],cv2.INTER_LINEAR)
        if blocked is None:
            out=background.copy();out[lut['valid']]=image[lut['valid']]
            return out,0
        withheld=cv2.remap(blocked,*lut['maps'],cv2.INTER_NEAREST)>0
        # Unknown and elevated detections aren't filled with fabricated road.
        usable=lut['valid']&~withheld;out=background.copy();out[usable]=image[usable]
        if withheld_background is not None:
            selected=withheld&lut['valid'];out[selected]=withheld_background[selected]
        return out,int((withheld&lut['valid']).sum())

    @staticmethod
    def object_symbol(bev,uv,o,c):
        """Fixed-size UI icon, never a metric footprint, size or orientation."""
        x,y=map(int,uv)
        cv2.circle(bev,(x,y),9,(245,245,245),-1,cv2.LINE_AA)
        cv2.circle(bev,(x,y),7,c,-1,cv2.LINE_AA)
        if o['cls']==0:  # COCO person: head and shoulders.
            cv2.circle(bev,(x,y-3),2,(245,245,245),-1)
            cv2.line(bev,(x,y),(x,y+4),(245,245,245),2)
            cv2.line(bev,(x-3,y+1),(x+3,y+1),(245,245,245),1)
        else:
            cv2.rectangle(bev,(x-4,y-3),(x+4,y+3),(245,245,245),1)
        text(bev,f"#{o['id']} {CLASS_NAMES.get(o['cls'],o['cls'])}",(x+12,y-5),c,.35)

    def line3(self,image,a,b,color,thickness=1):
        uv,good=self.project([a,b])
        if good.all() and np.isfinite(uv).all() and np.max(np.abs(uv))<1e5:
            cv2.line(image,tuple(np.rint(uv[0]).astype(int)),tuple(np.rint(uv[1]).astype(int)),color,thickness,cv2.LINE_AA)

    def grids(self,space,bev):
        for x in range(-5,71,5):
            uv=np.rint(self.bev_pixel([[x,-25],[x,25]])).astype(int)
            cv2.line(bev,tuple(uv[0]),tuple(uv[1]),(70,77,79),1)
            text(bev,f'{x}m',(self.bev_width//2+4,uv[0,1]-2),(190,205,215),.3)
        for y in range(-25,26,5):
            uv=np.rint(self.bev_pixel([[-8,y],[70,y]])).astype(int)
            cv2.line(bev,tuple(uv[0]),tuple(uv[1]),(70,77,79),1)
        # Sparse display-model grid only; not a measured ground mesh.
        for x in range(0,36,5):
            a=np.array([x,-15.,self.surface([x,-15.])]);b=np.array([x,15.,self.surface([x,15.])])
            # Sample curved lines rather than chords through the bowl.
            samples=np.array([[x,y,self.surface([x,y])+.02] for y in np.linspace(-15,15,31)])
            for a,b in zip(samples[:-1],samples[1:]):self.line3(space,a,b,(66,87,94))

    def ego(self,space,bev):
        xy=np.array([[-.8,-.95],[3.7,-.95],[3.7,.95],[-.8,.95]])
        uv=np.rint(self.bev_pixel(xy)).astype(int);cv2.fillConvexPoly(bev,uv,(160,173,180))
        cv2.polylines(bev,[uv],True,(245,245,245),2);text(bev,'EGO',uv.mean(axis=0),scale=.33)
        low=np.column_stack((xy,self.ground(xy)+.15));high=low+[0,0,1.35]
        for a,b in zip(low,np.roll(low,-1,axis=0)):self.line3(space,a,b,(190,200,210),2)
        for a,b in zip(high,np.roll(high,-1,axis=0)):self.line3(space,a,b,(240,240,240),2)
        for a,b in zip(low,high):self.line3(space,a,b,(210,220,230),2)

    def render(self,images,objects,states):
        started=time.perf_counter()
        packed=np.vstack([cv2.resize(images[c],self.thumb,interpolation=cv2.INTER_AREA) for c in CAM_ORDER])
        masks=[]
        for cid in CAM_ORDER:
            mask=np.zeros((self.thumb[1],self.thumb[0]),np.uint8);cam=self.cameras[cid]
            for o in objects:
                for d in o['dets']:
                    if d['src']!=cid:continue
                    q=np.rint(np.asarray(d['box'])*([self.thumb[0]/cam.width,self.thumb[1]/cam.height]*2)).astype(int)
                    cv2.rectangle(mask,tuple(q[:2]),tuple(q[2:]),255,-1)
            masks.append(mask)
        blocked=np.vstack(masks)
        # Panorama keeps the real object image. Only the planar BEV withholds
        # elevated pixels: IPM cannot supply their true ground texture.
        space,space_blocked=self.raster(packed,None,self.bowl,self.bowl_base)
        bev,bev_blocked=self.raster(packed,blocked,self.bev,self.bev_base,self.withheld_base)
        self.grids(space,bev);self.ego(space,bev);visible=[];unknown=[];outside=[]
        for o in objects:
            state=states[o['id']];c=colour(o)
            if state['status']!='estimated':
                unknown.append(o['id']);continue
            p=np.asarray(state['point_imu']);uv=np.rint(self.bev_pixel([p]))[0].astype(int)
            if 0<=uv[0]<self.bev_width and 35<=uv[1]<self.height-28:
                cov=np.asarray(state['covariance_xy']);T=np.array([[0,-(self.bev_width-1)/50],[-(self.height-1)/78,0]])
                vals,vec=np.linalg.eigh(T@cov@T.T);major=vec[:,-1]
                axes=tuple(np.clip(2*np.sqrt(np.maximum(vals[::-1],0)),2,200).astype(int))
                cv2.ellipse(bev,tuple(uv),axes,float(np.rad2deg(np.arctan2(major[1],major[0]))),0,360,c,1)
                self.object_symbol(bev,uv,o,c)
            else:outside.append(o['id'])  # Outside map != unknown distance.
        # RGB lives on the assumed bowl, not at the object's metric 3D position.
        # Reusing project(point_imu) here makes overlays float off the texture.
        anchors,unshown=self.observations.draw(space,objects)
        visible=[item['id'] for item in anchors]
        # Bearing-only indicators are in a separate, explicitly non-metric strip.
        cv2.rectangle(bev,(0,self.height-54),(self.bev_width,self.height),(25,25,25),-1)
        text(bev,'Bearing strip: ? unknown depth | out = beyond metric map',(8,self.height-37),scale=.30)
        for o in objects:
            if o['id'] not in unknown+outside:continue
            ray=bearing(o,self.cameras)
            if ray is None:continue
            yaw=np.rad2deg(np.arctan2(ray[1],ray[0]))
            x=int(np.clip((120-yaw)/240*(self.bev_width-70),4,self.bev_width-70))
            suffix='?' if o['id'] in unknown else 'out'
            # Horizontal position encodes angle only; no distance on this strip.
            cv2.arrowedLine(bev,(x+15,self.height-8),(x+15,self.height-20),colour(o),1,tipLength=.5)
            text(bev,f"#{o['id']} {suffix}",(x,self.height-22),colour(o),.29)
        for im,title in ((space,'3D-LIKE SURROUND | fixed bowl display, NOT reconstruction'),
                         (bev,'BEV | road texture + object symbols')):
            cv2.rectangle(im,(0,0),(im.shape[1],32),(22,22,22),-1);text(im,title,(10,21),scale=.4)
        cv2.rectangle(bev,(0,32),(self.bev_width,51),(22,22,22),-1)
        text(bev,'Hatched: object texture withheld | dark: no road texture',(8,45),scale=.29)
        cv2.rectangle(space,(0,self.height-25),(self.width,self.height),(22,22,22),-1)
        text(space,'Contours = same-texture detections | grid = display assumption | checker = unseen',
             (9,self.height-10),scale=.32)
        self.diagnostics=dict(render_s=time.perf_counter()-started,init_s=self.init_s,
            plane_beta=self.beta.tolist(),spatial_visible_ids=visible,nonmetric_list_ids=unknown,outside_map_ids=outside,
            bowl_coverage=self.bowl['coverage'],bev_coverage=self.bev['coverage'],
            withheld_bowl_pixels=space_blocked,withheld_bev_pixels=bev_blocked,
            object_display='Bowl contours follow original image sampling; BEV icons use unchanged metric estimates, NOT footprints',
            image_anchors=anchors,unshown_image_ids=unshown,
            overlay_index_bytes=self.observations.index_bytes,
            lut_bytes=self.bowl['lut_bytes']+self.bev['lut_bytes'],extra_network_inference=0,
            measurements_unchanged=True,surface_is_display_assumption=True)
        self.last_space=space;self.last_bev=bev
        return space,bev


def compose_spatial(images,objects,renderer,states,frame):
    space,bev=renderer.render(images,objects,states);top=np.hstack((space,bev));w=top.shape[1]
    tiles=[];tw=w//5;th=round(tw*2056/2464)
    for cid in CAM_ORDER:
        im=cv2.resize(images[cid],(tw,th))
        for o in objects:
            for d in o['dets']:
                if d['src']!=cid:continue
                cam=renderer.cameras[cid];q=np.rint(np.asarray(d['box'])*([tw/cam.width,th/cam.height]*2)).astype(int)
                cv2.rectangle(im,tuple(q[:2]),tuple(q[2:]),colour(o),1)
                text(im,f"#{o['id']}",(q[0],max(q[1]-3,35)),colour(o),.33)
        cv2.rectangle(im,(0,0),(tw,27),(22,22,22),-1);text(im,f'Cam{cid}',(8,19));tiles.append(im)
    native=np.hstack(tiles);native=cv2.copyMakeBorder(native,0,0,0,w-native.shape[1],cv2.BORDER_CONSTANT)
    header=np.full((72,w,3),22,np.uint8)
    text(header,f'{frame} | FROZEN FIVE-CAMERA FUSION .55 | objects={len(objects)} | V: original panorama',(10,23),scale=.47)
    text(header,'No new network | plane/bowl textures are assumptions | IDs frame-local | N/B next/back, Space auto',(10,46),scale=.4)
    listing=np.full((95,w,3),22,np.uint8)
    for i,o in enumerate(objects[:20]):
        text(listing,f"#{o['id']} {CLASS_NAMES.get(o['cls'],o['cls'])} C{''.join(map(str,o['src']))} {states[o['id']]['status']}",
             (8+i//4*(w//5),19+i%4*21),colour(o),.34)
    return np.vstack((header,top,native,listing))
