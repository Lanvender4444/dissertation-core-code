"""Calibrated pinhole/Mei rays in IMU coordinates; no model inference here."""
from pathlib import Path
import json
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]

def calibration_list(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    # Supplied Calibration-v2 files have an unnamed array after 'version'.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(text[text.index('['):text.rindex(']')+1])
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    return {int(p['sensorId']): p for p in data}

class Camera:
    def __init__(self, intrinsic, extrinsic):
        p = intrinsic
        self.id = int(p['sensorId'])
        self.width, self.height = p['width'], p['height']
        self.K = np.array([[p['fu'],0,p['cu']],[0,p['fv'],p['cv']],[0,0,1.]])
        self.D = np.array([p[k] for k in ('k1','k2','p1','p2')])
        self.xi = p.get('xi', 0.)
        T = np.array(extrinsic['T_imu_cam']).reshape(4,4)
        self.R, self.t = T[:3,:3], T[:3,3]

    def rays(self, pixels):
        xy = cv2.undistortPointsIter(np.asarray(pixels,float).reshape(-1,1,2),
            self.K, self.D, None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)).reshape(-1,2)
        r2 = (xy*xy).sum(axis=1)
        disc = 1+(1-self.xi**2)*r2
        lam = (self.xi + np.sqrt(np.maximum(disc,0)))/(1+r2)
        rays = np.column_stack([lam[:,None]*xy, lam-self.xi])
        rays[disc < 0] = np.nan
        rays /= np.linalg.norm(rays,axis=1,keepdims=True)
        return rays @ self.R.T

    def project_rays(self, rays):
        q = np.asarray(rays,float) @ self.R
        denom = q[:,2]+self.xi*np.linalg.norm(q,axis=1)
        valid = denom > 1e-8
        xy = q[:,:2]/np.maximum(denom[:,None],1e-8)
        x,y = xy.T
        k1,k2,p1,p2 = self.D
        r2=x*x+y*y
        radial=1+k1*r2+k2*r2*r2
        uv=np.column_stack([self.K[0,0]*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+self.K[0,2],
                            self.K[1,1]*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+self.K[1,2]])
        valid &= np.isfinite(uv).all(axis=1)
        valid &= (uv[:,0]>=0)&(uv[:,0]<self.width)&(uv[:,1]>=0)&(uv[:,1]<self.height)
        # Exclude the non-injective outer branch of the Mei projection.
        if self.xi > 1:
            valid &= q[:,2]/np.maximum(np.linalg.norm(q,axis=1),1e-9) > -1/self.xi
        return uv, valid

    def project(self, points):
        return self.project_rays(np.asarray(points)-self.t)

def load_cameras():
    intr=calibration_list(ROOT/'Intrinsics_cam.json')
    extr=calibration_list(ROOT/'T_imu_cam(1).json')
    return {c:Camera(intr[c],extr[c]) for c in intr}

def triangulate(a,ra,b,rb):
    """Closest calibrated rays. Box centres are approximate correspondences."""
    M=np.column_stack([ra,-rb])
    depths=np.linalg.lstsq(M,b.t-a.t,rcond=None)[0]
    pa,pb=a.t+depths[0]*ra,b.t+depths[1]*rb
    angle=np.linalg.norm(np.cross(ra,rb))
    return (pa+pb)/2, float(np.linalg.norm(pa-pb)), depths, float(angle)

class SurroundRenderer:
    """Rotation-only spherical panorama; translation parallax is NOT removed."""
    def __init__(self,cameras,width=1600,height=600):
        self.cameras=cameras
        self.width,self.height=width,height
        yaw=np.linspace(np.pi,-np.pi,width,endpoint=False)
        elevation=np.linspace(np.deg2rad(45),np.deg2rad(-55),height)
        yy,ee=np.meshgrid(yaw,elevation)
        rays=np.stack([np.cos(ee)*np.cos(yy),np.cos(ee)*np.sin(yy),np.sin(ee)],-1).reshape(-1,3)
        scores=[]; self.maps={}
        for cid,c in cameras.items():
            uv,valid=c.project_rays(rays)
            score=(rays@c.R)[:,2]
            score[~valid]=-np.inf
            scores.append(score)
            uv[~valid]=-1
            self.maps[cid]=uv.reshape(height,width,2).astype(np.float32)
        self.ids=list(cameras)
        scores=np.stack(scores)
        self.owner=np.argmax(scores,axis=0).reshape(height,width)
        self.valid=np.isfinite(scores.max(axis=0)).reshape(height,width)

    def pixel(self,ray):
        x,y,z=ray
        yaw=np.arctan2(y,x); el=np.arctan2(z,np.hypot(x,y))
        return (int((np.pi-yaw)/(2*np.pi)*self.width)%self.width,
                int((np.deg2rad(45)-el)/np.deg2rad(100)*self.height))

    def render(self,imgs,objects):
        out=np.zeros((self.height,self.width,3),np.uint8)
        for index,cid in enumerate(self.ids):
            if cid not in imgs: continue
            warped=cv2.remap(imgs[cid],self.maps[cid],None,cv2.INTER_LINEAR)
            mask=(self.owner==index)&self.valid
            out[mask]=warped[mask]
        colors={'FULL':(0,220,0),'HALF':(0,220,255),'UNVERIFIED':(255,190,0),'UNIQUE':(0,140,255)}
        for i,o in enumerate(objects):
            # Draw once using a member whose image actually owns that region.
            ranked=[]
            for d in o['dets']:
                x1,y1,x2,y2=d['box']
                ray=self.cameras[d['src']].rays([[(x1+x2)/2,(y1+y2)/2]])[0]
                if not np.isfinite(ray).all(): continue
                cx,cy=self.pixel(ray)
                owns=0<=cy<self.height and self.ids[self.owner[cy,cx]]==d['src']
                ranked.append((owns,d['conf'],d))
            for _,_,d in sorted(ranked,key=lambda x:(x[0],x[1]),reverse=True)[:1]:
                x1,y1,x2,y2=d['box']
                corners=self.cameras[d['src']].rays([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
                if not np.isfinite(corners).all(): continue
                pts=np.array([self.pixel(r) for r in corners],np.int32)
                if np.ptp(pts[:,0])>self.width/2: continue
                color=colors[o['label']]
                cv2.polylines(out,[pts],True,color,1)
                cv2.putText(out,f"#{i} {o['label']} {''.join(map(str,o['src']))}",tuple(pts[0]),0,.35,color,1)
        cv2.rectangle(out,(0,0),(self.width,32),(20,20,20),-1)
        cv2.putText(out,'Five-camera spherical view | rotation-only; near-object seams retain parallax',(12,22),0,.5,(255,255,255),1)
        return out


class GroundPlaneRenderer:
    """Optional five-camera image mosaic assuming IMU z=0 is flat ground.

    Ground pixels align under this assumption; raised objects stretch/ghost.
    This visualization must not be used as object-association evidence.
    """
    def __init__(self,cameras,size=700):
        self.size=size; self.ids=list(cameras); self.maps={}
        u,v=np.meshgrid(np.arange(size),np.arange(size))
        points=np.stack([(size*.85-v)/(size/100),(size/2-u)/(size/100),np.zeros_like(u)],-1).reshape(-1,3)
        scores=[]
        for cid,cam in cameras.items():
            uv,valid=cam.project(points)
            ray=points-cam.t
            score=(ray@cam.R)[:,2]/np.maximum(np.linalg.norm(ray,axis=1),1e-9)
            score[~valid]=-np.inf; scores.append(score)
            uv[~valid]=-1; self.maps[cid]=uv.reshape(size,size,2).astype(np.float32)
        scores=np.stack(scores)
        self.owner=np.argmax(scores,axis=0).reshape(size,size)
        self.valid=np.isfinite(scores.max(axis=0)).reshape(size,size)

    def render(self,imgs):
        out=np.zeros((self.size,self.size,3),np.uint8)
        for i,cid in enumerate(self.ids):
            warped=cv2.remap(imgs[cid],self.maps[cid],None,cv2.INTER_LINEAR)
            mask=(self.owner==i)&self.valid; out[mask]=warped[mask]
        return out
