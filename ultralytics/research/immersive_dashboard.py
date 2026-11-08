"""Surround-first pages. Only presentation changes, never object coordinates."""
import cv2
import numpy as np
from immersive_surround import LookViewport, overview
from spatial_dashboard import fit_image, clipped_text, BG, FG, MUTED, ACCENT
from five_camera_display import CAM_ORDER, colour


class SurroundDashboard:
    def __init__(self, sphere):
        self.sphere=sphere;self.viewport=None;self.look_rect=None;self.buttons=[]

    def hit(self,x,y):
        for (a,b,w,h),key in self.buttons:
            if a<=x<a+w and b<=y<b+h:return key
        return None

    def mouse(self,control,event,x,y,flags=0):
        if self.look_rect is None:
            control.drag=None;return
        a,b,w,h=self.look_rect
        if self.inset is not None:
            ix,iy,iw,ih=self.inset
            if ix<=x<ix+iw and iy<=y<iy+ih:
                control.drag=None;return
        # A drag must start and remain on the viewport, not on another panel.
        if not a<=x<a+w or not b<=y<b+h:
            control.drag=None;return
        control.mouse(event,x-a,y-b,flags)

    def render(self,texture,control,images,objects,bev,frame,metrics,visited,
               width=1280,height=720,page='look',context=False):
        if min(width,height)<320:raise ValueError('Window is too small')
        w,h=int(width),int(height);out=np.full((h,w,3),BG,np.uint8)
        self.look_rect=None;self.buttons=[]
        clipped_text(out,'SURROUND / LOOK AROUND',(12,25),w-24,FG,.56)
        clipped_text(out,f"Frame {frame} | {len(objects)} objects | P {metrics['precision']:.2%}  R {metrics['recall']:.2%}  F1 {metrics['f1']:.2%} | {visited} visited",
                     (12,46),w-24,MUTED,.36)
        bw=min(150,(w-24)//4)
        for i,(name,label,key) in enumerate((('look','M  Surround','m'),('sphere','P  Sphere 2:1','p'),('cameras','C  Cameras','c'),('bev','G  BEV','g'))):
            rect=(12+i*bw,54,bw-6,26);a,b,ww,hh=rect
            cv2.rectangle(out,(a,b),(a+ww,b+hh),ACCENT if page==name else (61,51,42),-1)
            clipped_text(out,label,(a+6,b+18),ww-12,BG if page==name else FG,.36)
            self.buttons.append((rect,ord(key)))
        body=(0,88,w,h-120);x,y,cw,ch=body
        if page=='look':
            if self.viewport is None or (self.viewport.width,self.viewport.height)!=(cw,ch):
                self.viewport=LookViewport(self.sphere,cw,ch)
                control.width=cw;control.height=ch;control.drag=None
            image,_=self.viewport.render(texture,control)
            out[y:y+ch,x:x+cw]=image;self.look_rect=body
            if context:
                side=min(ch//2,cw//4);rect=(w-side-12,y+ch-side-12,side,side)
                fit_image(out,bev,rect)
                # An inset is not a draggable view region (handled below).
                self.inset=rect
            else:self.inset=None
        elif page=='sphere':
            # Border/header is fitted along with the image; texture stays 2:1.
            fit_image(out,overview(texture,self.sphere,control,self.sphere.width),body)
        elif page=='bev':fit_image(out,bev,body)
        elif page=='cameras':
            gap=8;tw=(cw-4*gap)//3;th=(ch-3*gap)//2
            for i,cid in enumerate(CAM_ORDER):
                tx=gap+(i%3)*(tw+gap);ty=y+gap+(i//3)*(th+gap)
                clipped_text(out,f'CAM {cid} / fusion members',(tx,ty+16),tw,FG,.36)
                px,py,sx,sy=fit_image(out,images[cid],(tx,ty+22,tw,th-22))
                for obj in objects:
                    for det in obj['dets']:
                        if det['src']!=cid:continue
                        q=np.rint(np.asarray(det['box'])*[sx,sy,sx,sy]+[px,py,px,py]).astype(int)
                        q[[0,2]]=np.clip(q[[0,2]],px,px+round(images[cid].shape[1]*sx)-1)
                        q[[1,3]]=np.clip(q[[1,3]],py,py+round(images[cid].shape[0]*sy)-1)
                        cv2.rectangle(out,tuple(q[:2]),tuple(q[2:]),colour(obj),1)
                        clipped_text(out,f"#{obj['id']}",(q[0],max(py+12,q[1]-3)),tw,colour(obj),.32)
        else:raise ValueError(page)
        clipped_text(out,'Drag / wheel | 1 front 2 left 3 rear 4 right | T inset | F fullscreen | Space play | N/B frames | Q exit',
                     (12,h-12),w-24,MUTED,.34)
        return out
