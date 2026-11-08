"""360-degree LOOK-AROUND display, not a bowl / scene reconstruction.

Observed calibrated directions only; camera gaps explicitly remain unknown.
Rotation-only: camera translations / near parallax are not reconstructed.
"""
import time
import cv2
import numpy as np
from five_camera_display import CAM_ORDER,colour,text,CLASS_NAMES,safe_rays


class ImmersiveSphere:
    def __init__(self,cameras,width=1536,height=768):
        start=time.perf_counter();self.cameras=cameras;self.width=width;self.height=height
        self.thumb=(1232,1028)
        yaw=np.pi-(np.arange(width)+.5)*2*np.pi/width
        el=np.pi/2-(np.arange(height)+.5)*np.pi/height
        yy,ee=np.meshgrid(yaw,el)
        rays=np.stack((np.cos(ee)*np.cos(yy),np.cos(ee)*np.sin(yy),np.sin(ee)),axis=-1).reshape(-1,3)
        uv_maps=[];scores=[]
        for cid in CAM_ORDER:
            cam=cameras[cid];uv,v=cam.project_rays(rays);facing=(rays@cam.R)[:,2]
            if cam.xi:v&=facing>.15  # Trusted front cone, no folded rear Mei branch.
            v&=(uv[:,0]>2)&(uv[:,0]<cam.width-3)&(uv[:,1]>2)&(uv[:,1]<cam.height-3)
            score=facing.copy();score[~v]=-np.inf;uv_maps.append(uv);scores.append(score)
        scores=np.asarray(scores);owner=scores.argmax(axis=0);valid=np.isfinite(scores.max(axis=0))
        # Keep Cam3 wherever valid; other front cameras only fill its gaps.
        owner[np.isfinite(scores[2])]=2
        packed=np.zeros((len(rays),2),np.float32)
        for i,cid in enumerate(CAM_ORDER):
            take=valid&(owner==i);cam=cameras[cid]
            packed[take]=(uv_maps[i][take]+.5)*[self.thumb[0]/cam.width,self.thumb[1]/cam.height]-.5
            packed[take,1]+=i*self.thumb[1]
        packed[~valid]=-1
        self.packed_uv=packed.reshape(height,width,2)
        self.maps=cv2.convertMaps(self.packed_uv,None,cv2.CV_16SC2)
        self.owner=owner.reshape(height,width);self.valid=valid.reshape(height,width)
        edge=(self.owner!=np.roll(self.owner,1,axis=1))|(self.valid!=np.roll(self.valid,1,axis=1))
        edge|=(self.owner!=np.roll(self.owner,1,axis=0))|(self.valid!=np.roll(self.valid,1,axis=0))
        self.seam_mask=cv2.dilate(edge.astype(np.uint8),np.ones((3,3),np.uint8))
        y,x=np.indices((height,width));gray=22+((x//24+y//24)%2)*8
        self.background=np.repeat(gray[:,:,None],3,axis=2).astype(np.uint8)
        # Inspect the calibrated horizon, not whether all 360 degrees were filmed.
        horizon=self.valid[height//2];self.coverage=dict(sphere_fraction=float(self.valid.mean()),
            horizon_fraction=float(horizon.mean()),horizon_supported_degrees=float(horizon.mean()*360),
            rear_centre_observed=bool(horizon[0]),angular_canvas_degrees=360,
            reason='Conservative calibrated projection support, NOT true physical visibility or full coverage')
        self.init_s=time.perf_counter()-start;self.diagnostics={}

    def pixels(self,rays):
        r=np.asarray(rays);yaw=np.arctan2(r[:,1],r[:,0]);el=np.arctan2(r[:,2],np.hypot(r[:,0],r[:,1]))
        return np.column_stack((((np.pi-yaw)*self.width/(2*np.pi)-.5)%self.width,
                                (np.pi/2-el)*self.height/np.pi-.5))

    def render(self,images,objects):
        start=time.perf_counter()
        packed=np.vstack([cv2.resize(images[c],self.thumb,interpolation=cv2.INTER_LINEAR) for c in CAM_ORDER])
        self.packed_image=packed;self.annotations=[]
        image=cv2.remap(packed,*self.maps,cv2.INTER_LINEAR);image[~self.valid]=self.background[~self.valid]
        visible=[]
        for o in objects:
            ranked=[]
            for d in o['dets']:
                b=np.asarray(d['box']);ray,good=safe_rays(self.cameras[d['src']],[(b[:2]+b[2:])/2])
                if not good[0]:continue
                p=self.pixels(ray)[0];x,y=np.rint(p).astype(int);x%=self.width
                if not 0<=y<self.height or not self.valid[y,x] or CAM_ORDER[self.owner[y,x]]!=d['src']:continue
                ranked.append((d['src']==3,d['conf'],-d['uid'],d))
            if not ranked:continue
            d=max(ranked,key=lambda a:a[:3])[3];x1,y1,x2,y2=d['box']
            self.annotations.append((o['id'],o['cls'],colour(o),d))
            corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            samples=np.concatenate([a+(b-a)*np.linspace(0,1,8,endpoint=False)[:,None] for a,b in zip(corners,np.roll(corners,-1,axis=0))])
            rays,v=safe_rays(self.cameras[d['src']],samples);uv=self.pixels(rays)
            v&=np.isfinite(uv).all(axis=1)
            for j in range(len(uv)):
                k=(j+1)%len(uv)
                if v[j] and v[k] and np.linalg.norm(uv[j]-uv[k])<self.width/3:
                    cv2.line(image,tuple(np.rint(uv[j]).astype(int)),tuple(np.rint(uv[k]).astype(int)),colour(o),2)
            if v.any():text(image,f"#{o['id']} {CLASS_NAMES.get(o['cls'],o['cls'])}",uv[np.flatnonzero(v)[0]]+[0,-5],colour(o),.38)
            visible.append(o['id'])
        # Detection annotation must not create apparent observations in gaps.
        image[~self.valid]=self.background[~self.valid]
        self.diagnostics=dict(render_s=time.perf_counter()-start,visible_ids=visible,coverage=self.coverage,
            init_s=self.init_s,lut_bytes=sum(m.nbytes for m in self.maps),extra_network_inference=0,
            projection='rotation-only observed-ray equirectangular; no recovered scene depth')
        return image


class LookController:
    """GUI-independent drag / wheel state, testable without moving the desktop."""
    def __init__(self,width=1000,height=560):
        self.width=width;self.height=height;self.yaw=0.;self.pitch=0.;self.fov=90.;self.drag=None

    def turn(self,yaw=0,pitch=0):
        self.yaw=(self.yaw+yaw+180)%360-180;self.pitch=float(np.clip(self.pitch+pitch,-75,75))

    def reset(self):self.yaw=0.;self.pitch=0.;self.fov=90.;self.drag=None

    def mouse(self,event,x,y,flags=0):
        inside=0<=x<self.width and 0<=y<self.height
        if event==cv2.EVENT_LBUTTONDOWN and inside:self.drag=(x,y)
        elif event==cv2.EVENT_LBUTTONUP:self.drag=None
        elif event==cv2.EVENT_MOUSEMOVE and self.drag is not None:
            a,b=self.drag;self.turn((x-a)*self.fov/self.width,(y-b)*self.fov/self.width);self.drag=(x,y)
        elif event==cv2.EVENT_MOUSEWHEEL and inside:
            delta=(flags>>16)&65535;delta=delta-65536 if delta>=32768 else delta
            if delta:self.fov=float(np.clip(self.fov+(-6 if delta>0 else 6),40,115))


class LookViewport:
    def __init__(self,sphere,width=1000,height=560):
        self.sphere=sphere;self.width=width;self.height=height;self.key=None;self.map_builds=0
        x,y=np.meshgrid(np.arange(width,dtype=np.float32),np.arange(height,dtype=np.float32))
        self.x=(x-(width-1)/2)/(width/2);self.y=-((y-(height-1)/2)/(width/2))

    def maps(self,yaw,pitch,fov):
        if not np.isfinite([yaw,pitch,fov]).all() or not -75<=pitch<=75 or not 40<=fov<=115:
            raise ValueError('Finite view pose and FOV40..115, pitch -75..75 required')
        key=(float(yaw)%360,float(pitch),float(fov))
        if key!=self.key:
            y,p=np.deg2rad(key[:2]);forward=np.array([np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),np.sin(p)],np.float32)
            right=np.array([np.sin(y),-np.cos(y),0],np.float32);up=np.cross(right,forward)
            gain=np.tan(np.deg2rad(fov)/2)
            rays=forward+gain*self.x[:,:,None]*right+gain*self.y[:,:,None]*up
            self.basis=np.column_stack((right,up,forward));self.focal=(self.width/2)/gain
            uv=self.sphere.pixels(rays.reshape(-1,3)).reshape(self.height,self.width,2).astype(np.float32)
            uv[:,:,0]+=1  # Periodic padded panorama for bilinear yaw wrap.
            uv[:,:,1]=np.clip(uv[:,:,1],0,self.sphere.height-1)
            self.cached_maps=cv2.convertMaps(uv,None,cv2.CV_16SC2);self.key=key;self.map_builds+=1
            pad=lambda a:np.concatenate((a[:,-1:],a,a[:,:1]),axis=1)
            self.covered=cv2.remap(pad(self.sphere.valid.astype(np.uint8)),*self.cached_maps,cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REPLICATE).astype(bool)
            # Interpolate coordinates, NOT a low-resolution intermediate image.
            source=cv2.remap(pad(self.sphere.packed_uv),*self.cached_maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            seam=cv2.remap(pad(self.sphere.seam_mask),*self.cached_maps,cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)>0
            nearest=cv2.remap(pad(self.sphere.packed_uv),*self.cached_maps,cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
            source[seam]=nearest[seam];source[~self.covered]=-1
            self.source_maps=cv2.convertMaps(source,None,cv2.CV_16SC2)
        return self.cached_maps

    def render(self,texture,control):
        maps=self.maps(control.yaw,control.pitch,control.fov)
        padded=np.concatenate((texture[:,-1:],texture,texture[:,:1]),axis=1)
        out=cv2.remap(padded,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        covered=self.covered
        gap_pixels=out[~covered].copy()
        if hasattr(self.sphere,'packed_image'):
            direct=cv2.remap(self.sphere.packed_image,*self.source_maps,cv2.INTER_LINEAR)
            out[covered]=direct[covered]
            for oid,cls,color,d in self.sphere.annotations:
                x1,y1,x2,y2=d['box'];corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
                samples=np.concatenate([a+(b-a)*np.linspace(0,1,8,endpoint=False)[:,None] for a,b in zip(corners,np.roll(corners,-1,axis=0))])
                ray,good=safe_rays(self.sphere.cameras[d['src']],samples);q=ray@self.basis
                uv=np.column_stack(((self.width-1)/2+self.focal*q[:,0]/np.maximum(q[:,2],1e-8),
                                    (self.height-1)/2-self.focal*q[:,1]/np.maximum(q[:,2],1e-8)))
                good&=(q[:,2]>.05)&np.isfinite(uv).all(axis=1)
                for j in range(len(uv)):
                    k=(j+1)%len(uv)
                    if good[j] and good[k] and np.max(np.abs(uv[[j,k]]))<1e5:
                        cv2.line(out,tuple(np.rint(uv[j]).astype(int)),tuple(np.rint(uv[k]).astype(int)),color,2)
                inside=good&(uv[:,0]>=0)&(uv[:,0]<self.width-100)&(uv[:,1]>45)&(uv[:,1]<self.height)
                if inside.any():
                    text(out,f'#{oid} {CLASS_NAMES.get(cls,cls)}',uv[np.flatnonzero(inside)[0]]+[0,-5],color,.35)
        out[~covered]=gap_pixels
        fraction=float(covered.mean())
        cv2.rectangle(out,(0,0),(self.width,32),(22,22,22),-1)
        text(out,f'LOOK AROUND | yaw {control.yaw:+.0f} pitch {control.pitch:+.0f} FOV {control.fov:.0f} | observed {fraction:.0%}',(10,21),scale=.42)
        if not covered[self.height//2,self.width//2]:
            cv2.rectangle(out,(15,self.height//2-26),(self.width-15,self.height//2+7),(22,22,22),-1)
            text(out,'THIS LOOK DIRECTION IS UNOBSERVED - no fabricated view',(25,self.height//2),scale=.46)
        return out,fraction


def overview(texture,sphere,control,width):
    strip=cv2.resize(texture,(width,90));header=np.full((25,width,3),22,np.uint8)
    text(header,'360-degree overview: REAR | LEFT | FRONT | RIGHT | REAR. Checker = unobserved.',(8,18),scale=.34)
    scale=width/360;centre=((180-control.yaw)%360)*scale
    half=control.fov*scale/2
    for offset in (-width,0,width):
        a=max(0,int(centre-half+offset));b=min(width-1,int(centre+half+offset))
        if a<b:cv2.rectangle(strip,(a,1),(b,88),(0,215,255),2)
    return np.vstack((header,strip))
