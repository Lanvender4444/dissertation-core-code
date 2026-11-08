"""Display-only seam-local registration; never an input to fusion or BEV.

Inspired by parallax-tolerant/seam-driven stitching, NOT an APAP reproduction.
Cam3 anchors the forward scene. Observed pixels only; no generative completion.
Local warp and cropped elevation apply to the panorama, not native detections.
"""
import time
import cv2
import numpy as np
from five_camera_display import Panorama, safe_rays, colour, text, CLASS_NAMES


def minimum_seam(cost, max_step=2):
    """Connected top-to-bottom DP seam, deterministic, bounded row slope."""
    cost=np.asarray(cost,np.float32)
    if cost.ndim!=2 or not np.isfinite(cost).all() or not cost.size:
        raise ValueError('Finite nonempty seam cost required')
    h,w=cost.shape;prev=cost[0].copy();parents=np.zeros((h,w),np.int16)
    x=np.arange(w)
    for y in range(1,h):
        candidates=[]
        for step in range(-max_step,max_step+1):
            q=x+step;row=np.full(w,np.inf,np.float32);okay=(q>=0)&(q<w)
            row[okay]=prev[q[okay]]+.08*abs(step);candidates.append(row)
        stack=np.stack(candidates);which=stack.argmin(axis=0)
        parents[y]=which-max_step;prev=cost[y]+stack[which,x]
    seam=np.zeros(h,np.int32);seam[-1]=prev.argmin()
    for y in range(h-1,0,-1):seam[y-1]=seam[y]+parents[y,seam[y]]
    return seam


def sample_field(field,points):
    points=np.asarray(points,np.float32).reshape(-1,2)
    return cv2.remap(field,points[:,0,None],points[:,1,None],cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).reshape(-1,2)


def forward_points(points,inverse_field):
    """Invert q=p+d(p) for annotations using the SAME displayed pixel warp."""
    q=np.asarray(points,np.float32);p=q.copy()
    for _ in range(8):p=q-sample_field(inverse_field,p)
    error=np.linalg.norm(p+sample_field(inverse_field,p)-q,axis=1)
    return p,np.isfinite(p).all(axis=1)&(error<.8)


def robust_gain(reference,source,mask):
    """Bounded gains, ignore saturated/black samples; do not invent detail."""
    a=reference[mask].astype(float);b=source[mask].astype(float)
    if len(a)<100:return np.ones(3,np.float32)
    okay=(a>25)&(a<235)&(b>25)&(b<235)
    gains=[]
    for c in range(3):
        use=okay[:,c]
        gain=np.median((a[use,c]+1)/(b[use,c]+1)) if use.sum()>=100 else 1.
        gains.append(np.clip(gain,.75,1.30))
    return np.asarray(gains,np.float32)


def multiband(images,masks,coverage,levels=3):
    """CPU numpy/Mat pyramids, avoiding OpenCL allocation stalls in detail."""
    sizes=[(images[0].shape[1],images[0].shape[0])]
    for _ in range(levels):sizes.append(((sizes[-1][0]+1)//2,(sizes[-1][1]+1)//2))
    numerator=[np.zeros((h,w,3),np.float32) for w,h in sizes]
    denominator=[np.zeros((h,w,1),np.float32) for w,h in sizes]
    for image,mask,valid in zip(images,masks,coverage):
        if not mask.any():continue
        gp=[image.astype(np.float32)];weights=[mask.astype(np.float32)]
        support=[valid.astype(np.float32)]
        premultiplied=[gp[0]*support[0][:,:,None]]
        for _ in range(levels):
            support.append(cv2.pyrDown(support[-1]))
            premultiplied.append(cv2.pyrDown(premultiplied[-1]))
            gp.append(premultiplied[-1]/np.maximum(support[-1][:,:,None],1e-6))
            weights.append(cv2.pyrDown(weights[-1]))
        for i in range(levels+1):
            lap=gp[i] if i==levels else gp[i]-cv2.pyrUp(gp[i+1],dstsize=sizes[i])
            weight=weights[i][:,:,None]
            numerator[i]+=lap*weight;denominator[i]+=weight
    pyramid=[a/np.maximum(b,1e-6) for a,b in zip(numerator,denominator)]
    out=pyramid[-1]
    for i in range(levels-1,-1,-1):out=cv2.pyrUp(out,dstsize=sizes[i])+pyramid[i]
    return np.clip(out,0,255).astype(np.uint8)


class NaturalPanorama(Panorama):
    def __init__(self,cameras,width=1320,height=430,local_align=False):
        # Preserve spherical yaw coverage. Restrict ONLY cosmetic near-body
        # elevation; all five native panels/outputs/metrics remain uncropped.
        super().__init__(cameras,width,height)
        self.el_max=np.deg2rad(30);self.el_min=np.deg2rad(-28)
        y,e=np.meshgrid(np.linspace(self.yaw_max,self.yaw_min,width),
                         np.linspace(self.el_max,self.el_min,height))
        rays=np.stack((np.cos(e)*np.cos(y),np.cos(e)*np.sin(y),np.sin(e)),axis=-1).reshape(-1,3)
        self.base_valid={}
        for cid in self.ids:
            uv,v=cameras[cid].project_rays(rays);uv[~v]=-1
            if cameras[cid].xi:
                # The calibrated Brown-Mei polynomial can fold in the far
                # outer/rear branch. Those samples are not trusted display
                # support; side cameras still cover their full primary sector.
                v &= (rays@cameras[cid].R)[:,2]>.15
                uv[~v]=-1
            self.maps[cid]=uv.reshape(height,width,2).astype(np.float32)
            self.base_valid[cid]=v.reshape(height,width)
        self.grid=np.dstack(np.meshgrid(np.arange(width,dtype=np.float32),np.arange(height,dtype=np.float32)))
        self.fields={cid:np.zeros((height,width,2),np.float32) for cid in self.ids}
        self.valid=np.logical_or.reduce(list(self.base_valid.values()))
        self.local_align=local_align;self.diagnostics={};self.mapped_valid={}
        self.surface_maps={};self.surface_valid={};self.surface_radius={1:None,5:None}
        for cid in (1,5):
            for radius in (4.,8.,16.,None):
                if radius is None:uv=self.maps[cid].copy();v=self.base_valid[cid].copy()
                else:
                    points=cameras[3].t+radius*rays
                    uv,v=cameras[cid].project(points)
                    direction=points-cameras[cid].t
                    direction/=np.linalg.norm(direction,axis=1,keepdims=True)
                    v &= (direction@cameras[cid].R)[:,2]>.15
                    uv[~v]=-1;uv=uv.reshape(height,width,2).astype(np.float32);v=v.reshape(height,width)
                self.surface_maps[cid,radius]=uv;self.surface_valid[cid,radius]=v

    def surface_points(self,cid,rays):
        """Intersect source rays with the selected DISPLAY sphere, not BEV."""
        radius=self.surface_radius.get(cid)
        if radius is None:return np.asarray(rays)
        offset=self.cameras[cid].t-self.cameras[3].t;rays=np.asarray(rays)
        dot=rays@offset;disc=dot*dot+radius*radius-offset@offset
        length=-dot+np.sqrt(np.maximum(disc,0))
        return offset+length[:,None]*rays

    def choose_surface(self,image,anchor,front_valid,cid):
        """Seam-driven choice among four display surfaces on quarter images.

        Radius is a cosmetic registration hypothesis, never measured distance.
        No labels choose it. No image content is fabricated.
        """
        stride=4;a=anchor[::stride,::stride];av=front_valid[::stride,::stride]
        candidates=[];left=cid==1
        for radius in (4.,8.,16.,None):
            m=self.surface_maps[cid,radius][::stride,::stride]
            b=cv2.remap(image,m,None,cv2.INTER_LINEAR);bv=self.surface_valid[cid,radius][::stride,::stride]
            overlap=av&bv;nominal=[]
            for y in range(len(a)):
                xs=np.flatnonzero(av[y]);nominal.append(xs[0]+6 if left and len(xs) else xs[-1]-6 if len(xs) else a.shape[1]//2)
            nominal=np.asarray(nominal)
            x0=max(0,int(nominal.min())-14);x1=min(a.shape[1],int(nominal.max())+15)
            ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(float);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(float)
            cost=np.abs(a.astype(float)-b.astype(float)).mean(axis=2)/30.
            cost+=.012*np.abs(cv2.Sobel(ga,cv2.CV_64F,0,1)-cv2.Sobel(gb,cv2.CV_64F,0,1))
            cost+=.012*np.abs(cv2.Sobel(ga,cv2.CV_64F,1,0)-cv2.Sobel(gb,cv2.CV_64F,1,0))
            cost=cost[:,x0:x1]+.015*(np.arange(x0,x1)[None,:]-nominal[:,None])**2
            cost[~overlap[:,x0:x1]]+=10000
            path=minimum_seam(cost,max_step=1);yy=np.arange(len(a));okay=overlap[yy,path+x0]&(yy>8)
            score=float(np.mean(cost[yy[okay],path[okay]])) if okay.sum()>len(a)*.65 else 1e6
            candidates.append((score,radius))
        score,radius=min(candidates,key=lambda a:a[0])
        self.surface_radius[cid]=radius
        return self.surface_maps[cid,radius],self.surface_valid[cid,radius],dict(
            selected_display_radius=radius,seam_cost=score,
            candidate_costs=[dict(radius=r,cost=c) for c,r in candidates])

    def alignment(self,anchor,side,mask,cid):
        """Sparse FB-checked LK, bounded smooth row displacement near overlap.

        This is a 2D cosmetic observation warp, NOT a metric depth estimate.
        Lack of evidence falls back to calibration; no extrapolated GT shift.
        """
        empty=np.zeros((self.height,self.width,2),np.float32)
        info=dict(accepted=False,matches=0,reason='disabled' if not self.local_align else 'insufficient_texture')
        cols=np.flatnonzero(mask.any(axis=0))
        if not len(cols) or not self.local_align:return empty,info
        x0=max(0,int(cols[0])-90);x1=min(self.width,int(cols[-1])+91)
        a=cv2.cvtColor(anchor[:,x0:x1],cv2.COLOR_BGR2GRAY)
        b=cv2.cvtColor(side[:,x0:x1],cv2.COLOR_BGR2GRAY)
        m=mask[:,x0:x1].astype(np.uint8)*255;m[:35]=0;m[-12:]=0
        m=cv2.erode(m,np.ones((9,9),np.uint8))
        pts=cv2.goodFeaturesToTrack(a,maxCorners=180,qualityLevel=.012,minDistance=6,mask=m,blockSize=7)
        if pts is None or len(pts)<8:return empty,info
        opts=dict(winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,25,.02))
        # A finite-distance CALIBRATION hypothesis only initializes LK. It is
        # never exported as measured depth; actual correspondences must pass
        # forward/backward, NCC and positive-depth epipolar checks below.
        def rays_for(pixel):
            yaw=self.yaw_max-pixel[:,0]/(self.width-1)*(self.yaw_max-self.yaw_min)
            el=self.el_max-pixel[:,1]/(self.height-1)*(self.el_max-self.el_min)
            return np.column_stack((np.cos(el)*np.cos(yaw),np.cos(el)*np.sin(yaw),np.sin(el)))
        full=pts.reshape(-1,2)+[x0,0];reference_rays=rays_for(full)
        world=self.cameras[3].t+8*reference_rays
        guess,_=self.pixels(world-self.cameras[cid].t)
        guess=(guess-[x0,0]).astype(np.float32).reshape(-1,1,2)
        target,ok,_=cv2.calcOpticalFlowPyrLK(a,b,pts,guess,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**opts)
        if target is None:return empty,info
        back,back_ok,_=cv2.calcOpticalFlowPyrLK(b,a,target,None,**opts)
        if back is None:return empty,info
        delta=(target-pts).reshape(-1,2);points=pts.reshape(-1,2)
        good=ok.ravel().astype(bool)&back_ok.ravel().astype(bool)
        good &= np.isfinite(delta).all(axis=1)&(np.linalg.norm((back-pts).reshape(-1,2),axis=1)<1.2)
        good &= (np.abs(delta[:,0])<=80)&(np.abs(delta[:,1])<=14)
        observed_rays=rays_for(target.reshape(-1,2)+[x0,0])
        baseline=self.cameras[cid].t-self.cameras[3].t
        for i in np.flatnonzero(good):
            matrix=np.column_stack((reference_rays[i],-observed_rays[i]))
            depth=np.linalg.lstsq(matrix,baseline,rcond=None)[0]
            gap=np.linalg.norm(matrix@depth-baseline)
            if min(depth)<.5 or max(depth)>200 or gap>max(.04,.005*min(depth)):
                good[i]=False
        # Require textured, NCC-agreeing observed patches and source support.
        for i in np.flatnonzero(good):
            q=target[i,0];px,py=np.rint(q).astype(int)
            global_x=px+x0
            if not(6<=px<b.shape[1]-6 and 6<=py<b.shape[0]-6) or not self.base_valid[cid][py,global_x]:
                good[i]=False;continue
            pa=cv2.getRectSubPix(a,(11,11),tuple(points[i])).astype(float)
            pb=cv2.getRectSubPix(b,(11,11),tuple(q)).astype(float)
            pa-=pa.mean();pb-=pb.mean()
            denom=np.linalg.norm(pa)*np.linalg.norm(pb)
            if denom<200 or (pa*pb).sum()/max(denom,1)<.65:good[i]=False
        points=points[good];delta=delta[good]
        info['matches']=len(points)
        gains=[]
        for point,d in zip(points,delta):
            mean_a=cv2.getRectSubPix(anchor,(11,11),tuple(point+[x0,0])).mean(axis=(0,1))
            mean_b=cv2.getRectSubPix(side,(11,11),tuple(point+d+[x0,0])).mean(axis=(0,1))
            okay=(mean_a>30)&(mean_a<235)&(mean_b>30)&(mean_b<235)
            gains.append(np.where(okay,(mean_a+1)/(mean_b+1),np.nan))
        gain=np.nanmedian(gains,axis=0) if len(gains)>=6 else np.ones(3)
        info['verified_gain_bgr']=np.clip(np.nan_to_num(gain,nan=1),.75,1.3).tolist()
        if len(points)<8:return empty,info
        nodes=np.linspace(0,self.height-1,12);shifts=[];known=[]
        for y in nodes:
            use=np.abs(points[:,1]-y)<55
            if use.sum()<3:shifts.append([np.nan,np.nan]);known.append(False);continue
            d=delta[use];median=np.median(d,axis=0)
            inlier=np.linalg.norm(d-median,axis=1)<max(6.,2.5*np.median(np.linalg.norm(d-median,axis=1)))
            shifts.append(np.median(d[inlier],axis=0).tolist());known.append(True)
        shifts=np.asarray(shifts);known=np.asarray(known)
        if known.sum()<3:info['reason']='insufficient_vertical_support';return empty,info
        row=np.column_stack([np.interp(np.arange(self.height),nodes[known],shifts[known,c]) for c in (0,1)]).astype(np.float32)
        row=cv2.GaussianBlur(row[:,None,:],(1,0),sigmaX=0,sigmaY=9).reshape(self.height,2)
        # Reject stretching/folding instead of freely moving image content.
        if np.max(np.abs(np.diff(row,axis=0)))>.55:
            info['reason']='excessive_warp_gradient';return empty,info
        centre=(cols[0]+cols[-1])/2
        xx=np.arange(self.width);distance=np.maximum(np.abs(xx-centre)-(cols[-1]-cols[0])/2,0)
        taper=np.exp(-(distance/160.)**2).astype(np.float32)
        field=row[:,None,:]*taper[None,:,None]
        info.update(accepted=True,reason='fb_ncc_checked_row_warp',median_dx=float(np.median(delta[:,0])),
                    median_dy=float(np.median(delta[:,1])),maximum_shift=float(np.max(np.linalg.norm(field,axis=2))))
        return field,info

    def project_detection(self,d):
        x1,y1,x2,y2=d['box'];corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        samples=np.concatenate([a+(b-a)*np.linspace(0,1,9,endpoint=False)[:,None]
                                for a,b in zip(corners,np.roll(corners,-1,axis=0))])
        rays,v=safe_rays(self.cameras[d['src']],samples)
        uv,k=self.pixels(self.surface_points(d['src'],rays))
        # Non-finite points are replaced for safe sampling and stay invalid.
        uv=np.nan_to_num(uv,nan=-10000,posinf=-10000,neginf=-10000)
        uv,g=forward_points(uv,self.fields[d['src']]);return uv,v&k&g

    def target_mask(self,objects):
        mask=np.zeros((self.height,self.width),np.uint8)
        for o in objects:
            for d in o['dets']:
                uv,v=self.project_detection(d)
                if v.sum()<4:continue
                lo=np.maximum(np.floor(uv[v].min(axis=0)).astype(int)-6,0)
                hi=np.minimum(np.ceil(uv[v].max(axis=0)).astype(int)+6,[self.width-1,self.height-1])
                if (hi>lo).all():cv2.rectangle(mask,tuple(lo),tuple(hi),255,-1)
        return mask.astype(bool)

    def seam(self,anchor,side,front_valid,side_valid,left,protect):
        overlap=front_valid&side_valid;cols=np.flatnonzero(overlap.any(axis=0))
        if not len(cols):return np.full(self.height,0 if left else self.width-1,np.int32)
        # Keep the front scene with Cam3; side cameras supply exterior coverage.
        boundary=[]
        for y in range(self.height):
            xs=np.flatnonzero(front_valid[y])
            boundary.append(xs[0]+26 if left and len(xs) else xs[-1]-26 if len(xs) else self.width//2)
        nominal=np.asarray(boundary)
        x0=max(int(cols[0]),int(nominal.min())-45);x1=min(int(cols[-1])+1,int(nominal.max())+46)
        if x1<=x0:x0,x1=int(cols[0]),int(cols[-1])+1
        diff=np.abs(anchor[:,x0:x1].astype(float)-side[:,x0:x1].astype(float)).mean(axis=2)/30.
        ga=cv2.cvtColor(anchor,cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb=cv2.cvtColor(side,cv2.COLOR_BGR2GRAY).astype(np.float32)
        edge=np.abs(cv2.Sobel(ga,cv2.CV_32F,1,0))+np.abs(cv2.Sobel(ga,cv2.CV_32F,0,1))
        gradient=np.abs(cv2.Sobel(ga,cv2.CV_32F,1,0)-cv2.Sobel(gb,cv2.CV_32F,1,0))
        xx=np.arange(x0,x1)[None,:]
        cost=diff+.008*gradient[:,x0:x1]+.002*edge[:,x0:x1]
        cost+=.0015*(xx-nominal[:,None])**2
        cost+=protect[:,x0:x1]*18.
        cost[~overlap[:,x0:x1]]+=10000
        # Geometry remains native resolution; optimize only the seam search
        # on a coarse cost grid, then interpolate its connected path.
        small=cv2.resize(cost,(max(2,(cost.shape[1]+2)//3),max(2,(self.height+2)//3)),interpolation=cv2.INTER_AREA)
        path=minimum_seam(small,max_step=1)
        seam=np.interp(np.linspace(0,len(path)-1,self.height),np.arange(len(path)),
                       path*(cost.shape[1]-1)/(small.shape[1]-1))
        return np.rint(seam).astype(np.int32)+x0

    def render(self,images,objects):
        started=time.perf_counter()
        warped={cid:cv2.remap(images[cid],self.maps[cid],None,cv2.INTER_LINEAR) for cid in self.ids}
        anchor=warped[3].copy();front_valid=self.base_valid[3].copy()
        front_owner=np.full((self.height,self.width),self.ids.index(3),np.int32)
        for cid in (2,4):
            fill=~front_valid&self.base_valid[cid];anchor[fill]=warped[cid][fill]
            front_owner[fill]=self.ids.index(cid);front_valid|=fill
        masks={cid:v.copy() for cid,v in self.base_valid.items()};diagnostics={}
        for cid in (1,5):
            selected,selected_valid,surface=self.choose_surface(images[cid],anchor,front_valid,cid)
            warped[cid]=cv2.remap(images[cid],selected,None,cv2.INTER_LINEAR)
            masks[cid]=selected_valid.copy()
            self.fields[cid],info=self.alignment(anchor,warped[cid],front_valid&masks[cid],cid)
            info.update(surface)
            coordinates=self.grid+self.fields[cid]
            if info['accepted']:
                warped[cid]=cv2.remap(warped[cid],coordinates,None,cv2.INTER_LINEAR)
                masks[cid]=cv2.remap(masks[cid].astype(np.uint8),coordinates,None,cv2.INTER_NEAREST).astype(bool)
            gain=np.asarray(info.get('verified_gain_bgr',[1,1,1]),np.float32)
            warped[cid]=np.clip(warped[cid].astype(np.float32)*gain,0,255).astype(np.uint8)
            info['gain_bgr']=gain.tolist();diagnostics[cid]=info
        protect=self.target_mask(objects)
        left=self.seam(anchor,warped[1],front_valid,masks[1],True,protect)
        right=self.seam(anchor,warped[5],front_valid,masks[5],False,protect)
        xx=np.arange(self.width)[None,:];owner=front_owner.copy();valid=front_valid.copy()
        for cid,wants in ((1,xx<left[:,None]),(5,xx>right[:,None])):
            use=masks[cid]&(wants|~valid);owner[use]=self.ids.index(cid);valid|=masks[cid]
        # Explicit coverage masks: black observed pixels are not missing data.
        self.owner=owner;self.valid=valid;self.mapped_valid=masks
        out=np.full((self.height,self.width,3),12,np.uint8)
        for cid in self.ids:
            use=valid&(owner==self.ids.index(cid));out[use]=warped[cid][use]
        # Multiband only in narrow seam ROIs, not the whole panorama. This
        # keeps distant pixels untouched and avoids large float pyramids.
        for cid,path in ((1,left),(5,right)):
            x0=max(0,int(path.min())-28);x1=min(self.width,int(path.max())+29)
            side_mask=(owner[:,x0:x1]==self.ids.index(cid)).astype(np.float32)
            front_mask=(~side_mask.astype(bool)&front_valid[:,x0:x1]).astype(np.float32)
            mixed=multiband([warped[cid][:,x0:x1],anchor[:,x0:x1]],
                           [side_mask,front_mask],
                           [masks[cid][:,x0:x1],front_valid[:,x0:x1]])
            use=(np.abs(np.arange(x0,x1)[None,:]-path[:,None])<22)&valid[:,x0:x1]
            out[:,x0:x1][use]=mixed[use]
        out[~valid]=12
        visible=set();label_positions=[]
        for o in objects:
            candidates=[]
            for d in o['dets']:
                centre=(np.array(d['box'][:2])+d['box'][2:])/2
                ray,v=safe_rays(self.cameras[d['src']],[centre])
                uv,k=self.pixels(self.surface_points(d['src'],ray))
                if not v[0] or not k[0]:continue
                uv,g=forward_points(uv,self.fields[d['src']])
                if not g[0]:continue
                x,y=np.rint(uv[0]).astype(int)
                if not(0<=x<self.width and 32<=y<self.height):continue
                # Do not draw a full box on another camera's image content.
                if valid[y,x] and owner[y,x]==self.ids.index(d['src']):
                    candidates.append((d['src']==3,d['conf'],-d['uid'],d))
            if not candidates:continue
            d=max(candidates,key=lambda a:a[:3])[3];uv,v=self.project_detection(d)
            if v.sum()<4:continue
            c=colour(o)
            for j in range(len(uv)):
                k=(j+1)%len(uv)
                if v[j] and v[k] and np.linalg.norm(uv[j]-uv[k])<self.width/3:
                    cv2.line(out,tuple(np.rint(uv[j]).astype(int)),tuple(np.rint(uv[k]).astype(int)),c,2)
            point=uv[np.flatnonzero(v)[0]]
            label=f"#{o['id']} {CLASS_NAMES.get(o['cls'],o['cls'])}"
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.36,1)
            placed=False
            for dx,dy in ((0,-7),(0,-25),(0,-43),(35,-7),(-80,-7),(70,-25),(0,20)):
                x=int(np.clip(point[0]+dx,2,self.width-tw-4))
                y=int(np.clip(point[1]+dy,48,self.height-5))
                rect=(x-2,y-th-3,x+tw+3,y+3)
                if any(rect[0]<r[2] and rect[2]>r[0] and rect[1]<r[3] and rect[3]>r[1] for r in label_positions):
                    continue
                label_positions.append(rect);placed=True;break
            if placed:
                if np.linalg.norm(point-[x,y])>20:cv2.line(out,tuple(np.rint(point).astype(int)),(x,y),c,1)
                cv2.rectangle(out,(rect[0],rect[1]),(rect[2],rect[3]),(25,25,25),-1)
                text(out,label,(x,y),c,.36)
            else:
                # Exact ID is still in native panels/list; don't stack labels.
                text(out,f"#{o['id']}",point,c,.30)
            visible.add(o['id'])
        cv2.rectangle(out,(0,0),(self.width,32),(22,22,22),-1)
        text(out,'NATURAL PANORAMA | Cam3 anchor + seam-selected display surfaces | NOT metric',(12,21))
        self.diagnostics=dict(sides=diagnostics,render_s=time.perf_counter()-started,
            visible_ids=sorted(visible),output_ids=[o['id'] for o in objects],
            elevation_degrees=[-28,30],seam_left_range=[int(left.min()),int(left.max())],
            seam_right_range=[int(right.min()),int(right.max())],
            residual_parallax_possible=True)
        return out,visible
