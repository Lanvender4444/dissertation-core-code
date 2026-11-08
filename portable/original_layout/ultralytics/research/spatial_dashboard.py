"""Small-screen spatial viewer UI. Display only; never edits perception data."""
import cv2
import numpy as np
from five_camera_display import colour, text, CLASS_NAMES, CAM_ORDER

BG=(22,18,15);CARD=(34,29,25);EDGE=(61,51,42)
FG=(239,233,223);MUTED=(165,152,135);ACCENT=(235,186,82)


def fit_work_area(max_width,max_height,work=None):
    """Physical-pixel work area (taskbar excluded); headless callers can omit it."""
    width,height=max_width,max_height
    if work is not None:
        width=min(width,work[2]-work[0]-64);height=min(height,work[3]-work[1]-100)
    return max(1,int(width)),max(1,int(height))


def desktop_work_area():
    """Best effort Windows DPI awareness before creating any OpenCV windows."""
    try:
        import ctypes
        from ctypes import wintypes
        user=ctypes.windll.user32
        try:user.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError,OSError):pass
        # Use the monitor under the pointer, rather than assuming primary screen.
        pt=wintypes.POINT();user.GetCursorPos(ctypes.byref(pt))
        user.MonitorFromPoint.argtypes=[wintypes.POINT,wintypes.DWORD]
        user.MonitorFromPoint.restype=wintypes.HANDLE
        monitor=user.MonitorFromPoint(pt,2)
        class INFO(ctypes.Structure):
            _fields_=[('cbSize',wintypes.DWORD),('monitor',wintypes.RECT),('work',wintypes.RECT),('flags',wintypes.DWORD)]
        info=INFO();info.cbSize=ctypes.sizeof(info)
        user.GetMonitorInfoW.argtypes=[wintypes.HANDLE,ctypes.POINTER(INFO)]
        if user.GetMonitorInfoW(monitor,ctypes.byref(info)):
            r=info.work;return (r.left,r.top,r.right,r.bottom)
    except (AttributeError,OSError):pass
    return None


def clipped_text(image,value,xy,max_width,color=FG,scale=.43):
    value=str(value)
    if cv2.getTextSize(value,0,scale,1)[0][0]>max_width:
        while value and cv2.getTextSize(value+'...',0,scale,1)[0][0]>max_width:value=value[:-1]
        value+='...'
    text(image,value,xy,color,scale)


def fit_image(canvas,image,rect):
    """Letterbox, no stretching or removal of image content."""
    x,y,w,h=map(int,rect);factor=min(w/image.shape[1],h/image.shape[0])
    nw=max(1,round(image.shape[1]*factor));nh=max(1,round(image.shape[0]*factor))
    px=x+(w-nw)//2;py=y+(h-nh)//2
    canvas[py:py+nh,px:px+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
    return px,py,nw/image.shape[1],nh/image.shape[0]


class SpatialDashboard:
    def __init__(self,width=1280,height=720):
        if width<640 or height<440:raise ValueError('Compact UI needs at least 640x440 pixels')
        self.width=width;self.height=height;self.buttons=[]

    def button(self,image,label,rect,key,selected=False):
        x,y,w,h=rect
        cv2.rectangle(image,(x,y),(x+w,y+h),ACCENT if selected else EDGE,-1)
        clipped_text(image,label,(x+10,y+h//2+5),w-18,BG if selected else FG,.4)
        self.buttons.append((rect,key))

    def hit(self,x,y):
        for (bx,by,bw,bh),key in self.buttons:
            if bx<=x<bx+bw and by<=y<by+bh:return key
        return None

    @property
    def body_rect(self):
        width,height=self.layout_size
        top=106 if width>=1000 else 126
        return (18,top,width-36,height-top-56)

    @property
    def layout_size(self):
        # Keep text/buttons legible on maximized high-DPI displays, rather than
        # adding vast empty cards around a tiny fixed-font toolbar.
        scale=max(1.,min(self.width/1280,self.height/720))
        return round(self.width/scale),round(self.height/scale)

    @property
    def object_rows(self):return max(1,(self.body_rect[3]-100)//27)

    @staticmethod
    def card(canvas,rect,title,subtitle):
        x,y,w,h=rect
        cv2.rectangle(canvas,(x,y),(x+w-1,y+h-1),CARD,-1)
        cv2.rectangle(canvas,(x,y),(x+w-1,y+h-1),EDGE,1)
        clipped_text(canvas,title,(x+14,y+24),w-28,FG,.49)
        clipped_text(canvas,subtitle,(x+14,y+44),w-28,MUTED,.34)

    def render(self,*args,**kwargs):
        width,height=self.layout_size
        if (width,height)==(self.width,self.height):return self._render(*args,**kwargs)
        layout=SpatialDashboard(width,height)
        image=layout._render(*args,**kwargs)
        sx,sy=self.width/width,self.height/height
        self.buttons=[((round(x*sx),round(y*sy),round(w*sx),round(h*sy)),key)
                      for (x,y,w,h),key in layout.buttons]
        return cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_LINEAR)

    def _render(self,images,objects,states,space,bev,frame,metrics,visited,
               auto=False,page='overview',view='spatial',panorama=None,object_page=0,fullscreen=False):
        w,h=self.width,self.height;self.buttons=[]
        canvas=np.full((h,w,3),BG,np.uint8)
        text(canvas,'SPATIAL SURROUND',(18,29),FG,.62 if w>=1000 else .5)
        text(canvas,f'Frame {frame}  /  {len(objects)} objects  /  frozen 5-camera replay',(18,49),MUTED,.35)
        # Metrics get their own row on narrow windows; never run over the title.
        if w>=1000:
            for i,(label,key) in enumerate((('Precision','precision'),('Recall','recall'),('F1','f1'))):
                x=w-350+i*110;text(canvas,f'{metrics[key]:.2%}',(x,27),ACCENT,.54)
                text(canvas,label,(x,46),MUTED,.33)
            metric_footer=f'Five-view GT  /  {visited} visited frames'
        else:
            text(canvas,f"P {metrics['precision']:.2%}    R {metrics['recall']:.2%}    F1 {metrics['f1']:.2%}",(18,113),ACCENT,.42)
            metric_footer=f'5-view GT / {visited} frames'
        for i,(name,label) in enumerate((('overview','1  Overview'),('cameras','2  Cameras'),('objects','3  Objects'))):
            self.button(canvas,label,(18+i*128,62,118,30),ord(str(i+1)),page==name)
        if w>=1000:clipped_text(canvas,'Display only / local IDs',(w-350,82),182,MUTED,.34)
        self.button(canvas,'F  Restore' if fullscreen else 'F  Fullscreen',(w-156,62,138,30),ord('f'),fullscreen)
        body=self.body_rect
        if page=='overview':
            x,y,bw,bh=body;left=round((bw-12)*.66);right=bw-left-12
            self.card(canvas,(x,y,left,bh),'SURROUND' if view=='spatial' else 'NATURAL PANORAMA',
                      'Image-aligned contours / bowl is display only' if view=='spatial' else 'Seam-selected display / not metric')
            # Remove only old display title/footer bands, not scene pixels.
            source=space[32:-25] if view=='spatial' else panorama
            if source is not None:fit_image(canvas,source,(x+8,y+56,left-16,bh-90))
            clipped_text(canvas,'V: switch surround / panorama',(x+14,y+bh-13),left-28,MUTED,.34)
            rx=x+left+12
            self.card(canvas,(rx,y,right,bh),'BEV / POSITION ESTIMATES','Ground-plane texture / uncertainty ellipses')
            fit_image(canvas,bev[51:],(rx+8,y+56,right-16,bh-106))
            clipped_text(canvas,'Hatched = withheld object texture',(rx+12,y+bh-31),right-24,MUTED,.32)
            clipped_text(canvas,'Dark = no road texture / ? = unknown depth',(rx+12,y+bh-13),right-24,MUTED,.32)
        elif page=='cameras':
            x,y,bw,bh=body;gap=12;cw=(bw-2*gap)//3;ch=(bh-gap)//2
            for i,cid in enumerate(CAM_ORDER):
                cx=x+(i%3)*(cw+gap);cy=y+(i//3)*(ch+gap)
                self.card(canvas,(cx,cy,cw,ch),f'CAM {cid}','Original image / fusion member boxes')
                px,py,sx,sy=fit_image(canvas,images[cid],(cx+8,cy+51,cw-16,ch-59))
                for o in objects:
                    for d in o['dets']:
                        if d['src']!=cid:continue
                        q=np.rint(np.asarray(d['box'])*[sx,sy,sx,sy]+[px,py,px,py]).astype(int)
                        cv2.rectangle(canvas,tuple(q[:2]),tuple(q[2:]),colour(o),1)
                        clipped_text(canvas,f"#{o['id']}",(q[0],max(py+12,q[1]-4)),cx+cw-q[0]-8,colour(o),.35)
            cx=x+2*(cw+gap);cy=y+ch+gap
            self.card(canvas,(cx,cy,cw,ch),'VIEW SUPPORT','Not a physical truth / false-positive label')
            for j,(label,color) in enumerate((('Green: 3+ views',(70,220,80)),('Yellow: 2 views',(40,220,250)),
                                             ('Blue: 1 view',(240,170,70)),('Orange: recovered',(30,155,255)))):
                if 72+j*23>ch-8:break
                clipped_text(canvas,label,(cx+14,cy+72+j*23),cw-28,color,.38)
        elif page=='objects':
            x,y,bw,bh=body;rows=self.object_rows;pages=max(1,(len(objects)+rows-1)//rows)
            current=int(np.clip(object_page,0,pages-1))
            self.card(canvas,body,'OBJECT EVIDENCE',f'Page {current+1}/{pages}  /  [ and ]: page  /  unmatched does not prove missed detection')
            fractions=(.015,.09,.26,.44,.66);labels=('ID','Class','Views','Unmatched','Position estimate (m)')
            for frac,label in zip(fractions,labels):clipped_text(canvas,label,(x+int(bw*frac),y+73),int(bw*.18),MUTED,.37)
            for j,o in enumerate(objects[current*rows:(current+1)*rows]):
                yy=y+100+j*27
                if j%2==0:cv2.rectangle(canvas,(x+8,yy-18),(x+bw-8,yy+5),(40,34,29),-1)
                state=states[o['id']];missing=','.join(str(c) for c in CAM_ORDER if c not in o['src']) or '-'
                position='UNKNOWN / bearing only'
                if state['status']=='estimated':position=f"x {state['point_imu'][0]:.1f}, y {state['point_imu'][1]:.1f}"
                values=(f"#{o['id']}",CLASS_NAMES.get(o['cls'],str(o['cls'])),','.join(map(str,o['src'])),missing,position)
                for i,(frac,value) in enumerate(zip(fractions,values)):
                    next_frac=fractions[i+1] if i+1<len(fractions) else .995
                    clipped_text(canvas,value,(x+int(bw*frac),yy),int(bw*(next_frac-frac))-8,colour(o) if i==0 else FG,.38)
        else:raise ValueError(f'Unknown page {page}')
        for label,x,bw,key in (('B  Back',18,94,ord('b')),('Pause' if auto else 'Play',122,84,32),('N  Next',216,94,ord('n'))):
            self.button(canvas,label,(x,h-40,bw,28),key,selected=key==32)
        clipped_text(canvas,metric_footer,(328,h-20),w-440,MUTED,.34)
        self.button(canvas,'Q  Exit',(w-100,h-40,82,28),ord('q'))
        return canvas
