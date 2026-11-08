"""Anchor bowl annotations to the actual sampled image, not metric 3D points.

Inverse lookup of the same texture LUT; no new depth or identity estimates.
"""
import cv2
import numpy as np
from five_camera_display import CAM_ORDER,CLASS_NAMES,colour,text


class BowlObservations:
    def __init__(self,renderer):
        self.renderer=renderer;self.indices={}
        lut=renderer.bowl;integer,fraction=lut['maps']
        # OpenCV's fixed-point remap uses 5 fractional bits for each coordinate.
        packed_u=integer[:,:,0].astype(np.float32)+(fraction%32)/32.
        packed_v=integer[:,:,1].astype(np.float32)+(fraction//32)/32.
        yy,xx=np.indices(lut['valid'].shape)
        for i,cid in enumerate(CAM_ORDER):
            mask=lut['valid']&(lut['owner']==i)&(yy>=32)&(yy<renderer.height-25)
            y,x=np.nonzero(mask);cam=renderer.cameras[cid]
            u=(packed_u[y,x]+.5)*cam.width/renderer.thumb[0]-.5
            v=(packed_v[y,x]-i*renderer.thumb[1]+.5)*cam.height/renderer.thumb[1]-.5
            self.indices[cid]=(x,y,u,v)
        self.index_bytes=sum(v.nbytes for values in self.indices.values() for v in values)

    def anchor(self,o):
        candidates=[]
        for d in o['dets']:
            cid=d['src'];x,y,u,v=self.indices[cid];x1,y1,x2,y2=map(float,d['box'])
            chosen=np.flatnonzero((u>=x1)&(u<=x2)&(v>=y1)&(v<=y2))
            if not len(chosen):continue  # Never annotate another camera's texture.
            cx,cy=(x1+x2)/2,(y1+y2)/2
            error=((u[chosen]-cx)/max(x2-x1,1))**2+((v[chosen]-cy)/max(y2-y1,1))**2
            best=int(np.argmin(error));at=chosen[best]
            point=(int(x[at]),int(y[at]))
            rank=(error[best]<=.0625,cid==3,float(d['conf']),-float(error[best]),len(chosen))
            info=dict(id=o['id'],camera=cid,pixel=list(point),
                sampled_native_uv=[float(u[at]),float(v[at])],box=list(d['box']),
                centre_residual_native_px=float(np.hypot(u[at]-cx,v[at]-cy)),
                basis='same_owner_and_texture_LUT_not_metric_3D')
            candidates.append((rank,info,(x[chosen],y[chosen])))
        if not candidates:return None
        _,info,pixels=max(candidates,key=lambda item:item[0])
        return info,pixels

    def draw(self,image,objects):
        occupied=[];anchors=[];omitted=[];height,width=image.shape[:2]
        for o in objects:
            found=self.anchor(o)
            if found is None:omitted.append(o['id']);continue
            info,(xs,ys)=found;anchors.append(info);point=tuple(info['pixel']);c=colour(o)
            mask=np.zeros(image.shape[:2],np.uint8);mask[ys,xs]=255
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image,contours,-1,c,1,cv2.LINE_AA)
            cv2.circle(image,point,3,c,-1,cv2.LINE_AA)
            # A displaced label always has a leader to its true image anchor.
            label=f"#{o['id']} {CLASS_NAMES.get(o['cls'],o['cls'])} C{info['camera']}"
            (tw,th),_=cv2.getTextSize(label,0,.35,1)
            for dx,dy in ((7,-8),(7,-28),(7,-48),(-tw-7,-8),(7,24),(-tw-7,24)):
                x=int(np.clip(point[0]+dx,2,max(2,width-tw-4)))
                y=int(np.clip(point[1]+dy,48,height-30));rect=(x-2,y-th-3,x+tw+3,y+3)
                if any(rect[0]<r[2] and rect[2]>r[0] and rect[1]<r[3] and rect[3]>r[1] for r in occupied):continue
                occupied.append(rect)
                end=(int(np.clip(point[0],rect[0],rect[2])),int(np.clip(point[1],rect[1],rect[3])))
                cv2.line(image,point,end,c,1,cv2.LINE_AA)
                cv2.rectangle(image,(rect[0],rect[1]),(rect[2],rect[3]),(25,25,25),-1)
                text(image,label,(x,y),c,.35)
                break
            else:  # Keep anchor; full ID and evidence still in Objects page.
                text(image,f"#{o['id']}",(point[0]+4,point[1]+4),c,.28)
        return anchors,omitted
