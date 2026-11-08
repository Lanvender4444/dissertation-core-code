"""Shared object IDs on a calibrated panorama, native images and sparse BEV.

Display never alters object membership or acceptance. No implicit ground plane.
BEV markers are triangulated detector-centre estimates, not vehicle footprints.
"""
import itertools
import cv2
import numpy as np
from multiview_dedup import position_uncertainty
from shared_geometry import safe_rays

CLASS_NAMES={0:'Person',1:'Bicycle',2:'Car',3:'Motorcycle',5:'Bus',7:'Truck'}
CAM_ORDER=(1,2,3,4,5)


def colour(o):
    if o.get('recovered'):return (30,155,255)
    return (70,220,80) if len(o['src'])>=3 else (40,220,250) if len(o['src'])==2 else (240,170,70)


def text(image,value,xy,color=(230,230,230),scale=.42):
    cv2.putText(image,value,tuple(map(int,xy)),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


class Panorama:
    def __init__(self,cameras,width=1320,height=430):
        self.cameras=cameras;self.width=width;self.height=height
        self.yaw_max=np.deg2rad(120);self.yaw_min=-self.yaw_max
        self.el_max=np.deg2rad(35);self.el_min=np.deg2rad(-50)
        yaw=np.linspace(self.yaw_max,self.yaw_min,width)
        elevation=np.linspace(self.el_max,self.el_min,height)
        y,e=np.meshgrid(yaw,elevation)
        rays=np.stack((np.cos(e)*np.cos(y),np.cos(e)*np.sin(y),np.sin(e)),axis=-1).reshape(-1,3)
        self.ids=list(CAM_ORDER);self.maps={};scores=[]
        for cid in self.ids:
            cam=cameras[cid];uv,valid=cam.project_rays(rays)
            score=(rays@cam.R)[:,2]
            score[~valid]=-np.inf;scores.append(score)
            uv[~valid]=-1;self.maps[cid]=uv.reshape(height,width,2).astype(np.float32)
        scores=np.asarray(scores);self.valid=np.isfinite(scores.max(axis=0)).reshape(height,width)
        self.owner=np.argmax(scores,axis=0).reshape(height,width)

    def pixels(self,rays):
        r=np.asarray(rays,float);yaw=np.arctan2(r[:,1],r[:,0]);el=np.arctan2(r[:,2],np.hypot(r[:,0],r[:,1]))
        uv=np.column_stack(((self.yaw_max-yaw)/(self.yaw_max-self.yaw_min)*(self.width-1),
                            (self.el_max-el)/(self.el_max-self.el_min)*(self.height-1)))
        okay=np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<self.width)&(uv[:,1]>=0)&(uv[:,1]<self.height)
        return uv,okay

    def render(self,images,objects):
        out=np.full((self.height,self.width,3),12,np.uint8)
        for i,cid in enumerate(self.ids):
            warped=cv2.remap(images[cid],self.maps[cid],None,cv2.INTER_LINEAR)
            mask=self.valid&(self.owner==i);out[mask]=warped[mask]
        visible=set()
        for o in objects:
            ranked=[]
            for d in o['dets']:
                box=np.asarray(d['box']);centre=(box[:2]+box[2:])/2
                rays,valid=safe_rays(self.cameras[d['src']],[centre]);uv,okay=self.pixels(rays)
                if not valid[0] or not okay[0]:continue
                x,y=np.rint(uv[0]).astype(int)
                x=min(x,self.width-1);y=min(y,self.height-1)
                owns=self.ids[self.owner[y,x]]==d['src'] and self.valid[y,x]
                ranked.append((owns,d['conf'],d))
            if not ranked:continue
            d=max(ranked,key=lambda v:(v[0],v[1],-v[2]['uid']))[2]
            x1,y1,x2,y2=d['box']
            # Nonlinear fisheye edges are sampled, not just four corners.
            samples=[]
            corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            for a,b in zip(corners,np.roll(corners,-1,axis=0)):
                samples.extend(a+(b-a)*t for t in np.linspace(0,1,9,endpoint=False))
            rays,valid=safe_rays(self.cameras[d['src']],samples);uv,okay=self.pixels(rays)
            valid &= okay
            if valid.sum()<4:continue
            # Draw only valid consecutive pieces; no chord across unseen sectors.
            c=colour(o)
            for j in range(len(uv)):
                k=(j+1)%len(uv)
                if valid[j] and valid[k] and np.linalg.norm(uv[j]-uv[k])<self.width/3:
                    cv2.line(out,tuple(np.rint(uv[j]).astype(int)),tuple(np.rint(uv[k]).astype(int)),c,2)
            point=uv[np.flatnonzero(valid)[0]]
            label=f"#{o['id']} {CLASS_NAMES.get(o['cls'],str(o['cls']))} C{''.join(map(str,o['src']))}"
            text(out,label,(np.clip(point[0],0,self.width-240),np.clip(point[1]-5,50,self.height-5)),c)
            visible.add(o['id'])
        cv2.rectangle(out,(0,0),(self.width,32),(22,22,22),-1)
        text(out,'FIVE-CAMERA PANORAMA | calibrated rotation mapping; near seams retain parallax',(12,21))
        return out,visible


def locate(o,cameras):
    geometry=position_uncertainty(o,cameras)
    if geometry is None:return dict(status='unknown',reason='insufficient_or_unreliable_multiview_depth')
    point,cov=geometry
    # Agreement with all retained members, not just the triangulated pair.
    for d in o['dets']:
        uv,valid=cameras[d['src']].project([point]);box=np.asarray(d['box']);size=np.maximum(box[2:]-box[:2],4)
        if not valid[0] or np.linalg.norm((uv[0]-(box[:2]+box[2:])/2)/size)>1:
            return dict(status='unknown',reason='joint_reprojection_disagreement')
    return dict(status='estimated',point_imu=point.tolist(),covariance_xy=cov[:2,:2].tolist(),
                kind='detector_centre_xy_not_ground_contact',uncertainty='engineering_pixel_noise_not_calibrated')


def bearing(o,cameras):
    d=max(o['dets'],key=lambda d:d['conf']);box=np.asarray(d['box'])
    rays,valid=safe_rays(cameras[d['src']],[(box[:2]+box[2:])/2])
    return rays[0] if valid[0] else None


def radar(objects,cameras,size=430,extent=60):
    out=np.full((size,size,3),(26,22,18),np.uint8)
    centre=np.array([size/2,size*.72]);scale=(size*.64)/extent
    pixel=lambda p:np.rint(centre+np.array([-p[1],-p[0]])*scale).astype(int)
    radius=int(size*.64)
    for metres in (10,20,40,60):
        cv2.circle(out,tuple(centre.astype(int)),round(metres*scale),(65,58,48),1)
        text(out,f'{metres}m',(centre[0]+4,centre[1]-metres*scale),(150,150,150),.32)
    cv2.line(out,(int(centre[0]),34),(int(centre[0]),size-18),(75,68,55),1)
    cv2.rectangle(out,tuple((centre+[-7,-12]).astype(int)),tuple((centre+[7,12]).astype(int)),(235,235,235),-1)
    states={};known=0
    for o in objects:
        state=locate(o,cameras);states[o['id']]=state;c=colour(o)
        if state['status']=='estimated':
            known+=1;p=np.asarray(state['point_imu']);uv=pixel(p)
            if 35<=uv[1]<size-18 and 0<=uv[0]<size:
                cov=np.asarray(state['covariance_xy']);transform=np.array([[0,-scale],[-scale,0]])
                values,vectors=np.linalg.eigh(transform@cov@transform.T)
                axes=tuple(np.clip(2*np.sqrt(np.maximum(values[::-1],0)),2,size).astype(int))
                major=vectors[:,-1];angle=np.degrees(np.arctan2(major[1],major[0]))
                cv2.ellipse(out,tuple(uv),axes,float(angle),0,360,c,1)
                cv2.circle(out,tuple(uv),3,c,-1);text(out,f"#{o['id']}",uv+[5,-4],c,.35)
            else:state['outside_display_range']=True
        else:
            ray=bearing(o,cameras)
            if ray is None:continue
            direction=np.array([-ray[1],-ray[0]])
            direction/=max(np.linalg.norm(direction),1e-9)
            # Indicator lies on display border, never interpreted as a distance.
            anchor=centre+direction*radius
            anchor=np.clip(anchor,[12,45],[size-18,size-28]).astype(int)
            cv2.arrowedLine(out,tuple(anchor-(direction*12).astype(int)),tuple(anchor),c,1,tipLength=.5)
            text(out,f"#{o['id']} ?",np.clip(anchor+[4,-4],[4,45],[size-60,size-20]),c,.30)
    cv2.rectangle(out,(0,0),(size,32),(22,22,22),-1)
    text(out,f'BEV | located {known}/{len(objects)}',(10,21))
    text(out,'Edge ? = bearing only, distance UNKNOWN',(10,size-8),(200,200,200),.34)
    return out,states


def compose(images,objects,panorama,frame):
    pano,visible=panorama.render(images,objects);bev,states=radar(objects,panorama.cameras,size=panorama.height)
    top=np.hstack((pano,bev));width=top.shape[1]
    panels=[];tile_width=width//5;tile_height=round(tile_width*images[1].shape[0]/images[1].shape[1])
    for cid in CAM_ORDER:
        im=cv2.resize(images[cid],(tile_width,tile_height));sx=tile_width/images[cid].shape[1];sy=tile_height/images[cid].shape[0]
        for o in objects:
            for d in o['dets']:
                if d['src']!=cid:continue
                xy=np.asarray(d['box'])*[sx,sy,sx,sy];xy=np.rint(xy).astype(int);c=colour(o)
                cv2.rectangle(im,tuple(xy[:2]),tuple(xy[2:]),c,1)
                text(im,f"#{o['id']}",(xy[0],max(xy[1]-3,35)),c,.35)
        cv2.rectangle(im,(0,0),(tile_width,27),(22,22,22),-1);text(im,f'Cam{cid}',(8,19));panels.append(im)
    native=np.hstack(panels)
    if native.shape[1]<width:native=cv2.copyMakeBorder(native,0,0,0,width-native.shape[1],cv2.BORDER_CONSTANT)
    # Fixed-height object list: all IDs are logged even when only part fits.
    listing=np.full((150,width,3),22,np.uint8)
    for i,o in enumerate(objects[:30]):
        col=i//6;row=i%6;x=8+col*(width//5);y=19+row*22
        missing=''.join(str(c) for c in CAM_ORDER if c not in o['src']) or '-'
        text(listing,f"#{o['id']} {CLASS_NAMES.get(o['cls'],o['cls'])} sup:{''.join(map(str,o['src']))} no-match:{missing}",(x,y),colour(o),.34)
    header=np.full((65,width,3),22,np.uint8)
    text(header,f'{frame} | FIVE-CAMERA CURRENT FUSION + RECOVERY .55 | objects={len(objects)} | CACHE REPLAY',(10,22),scale=.48)
    text(header,'Green:3+ views  Yellow:2  Blue:1  Orange:new recovery | IDs frame-local; no-match means unassociated',(10,46),scale=.42)
    return np.vstack((header,top,native,listing)),states,visible
