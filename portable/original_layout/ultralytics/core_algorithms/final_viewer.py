"""Four entry points share this implementation. No GT is used in inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from multiview_geometry import ROOT,load_cameras
from multiview_fusion import HEIGHT,feature
from final_pipeline import run_pipeline,core,SINGLE_THRESHOLDS,FRONT_THRESHOLDS,DEDUP_GATE

COLORS={'FULL':(60,220,60),'PART':(0,220,255),'UNVERIFIED':(255,180,50),'SINGLE':(60,220,60)}

def label(o):
    supported=''.join(map(str,o['src']));missing=''.join(map(str,o['unmatched_cameras'])) or '-'
    return f"#{o['id']} {core.CLASS_NAMES[o['cls']]} {o['conf']:.2f} {o['label']} sup:{supported} no-match:{missing}"

def put_label(img,text,xy,color,scale=.42):
    (w,h),base=cv2.getTextSize(text,0,scale,1)
    x=max(0,min(int(xy[0]),img.shape[1]-w-4));y=max(h+4,min(int(xy[1]),img.shape[0]-base-2))
    cv2.rectangle(img,(x,y-h-3),(min(img.shape[1]-1,x+w+3),y+base),(20,20,20),-1)
    cv2.putText(img,text,(x+1,y),0,scale,color,1,cv2.LINE_AA)

def draw_box_labels(img,items,scale=.42):
    occupied=[]
    for text,anchor,color in items:
        (w,h),base=cv2.getTextSize(text,0,scale,1)
        x=max(0,min(int(anchor[0]),img.shape[1]-w-4));original_y=int(anchor[1]);y=max(h+25,original_y)
        for _ in range(60):
            rect=(x,y-h-3,x+w+3,y+base)
            if not any(rect[0]<r[2] and rect[2]>r[0] and rect[1]<r[3] and rect[3]>r[1] for r in occupied):break
            y-=h+base+5
            if y<h+25:y=min(img.shape[0]-base-3,original_y+len(occupied)*(h+base+5)+25)
        occupied.append(rect)
        cv2.line(img,(x,y),(int(anchor[0]),int(anchor[1])),color,1)
        put_label(img,text,(x,y),color,scale)

class FrontPanorama:
    """Rotation-only front virtual camera; original panels are authoritative."""
    def __init__(self,cameras,width=1440,height=500):
        self.cameras=cameras;self.width=width;self.height=height;self.maps={}
        self.f=cameras[3].K[0,0]*width/cameras[3].width*.72
        u,v=np.meshgrid(np.arange(width),np.arange(height))
        local=np.stack([(u-width/2)/self.f,(v-height/2)/self.f,np.ones_like(u)],-1).reshape(-1,3)
        rays=local@cameras[3].R.T;scores=[]
        for c,cam in cameras.items():
            uv,valid=cam.project_rays(rays)
            dist=((uv[:,0]-cam.K[0,2])/cam.width)**2+((uv[:,1]-cam.K[1,2])/cam.height)**2
            scores.append(np.where(valid,-dist,-np.inf));uv[~valid]=-1
            self.maps[c]=uv.reshape(height,width,2).astype(np.float32)
        scores=np.stack(scores);self.ids=list(cameras);self.owner=np.argmax(scores,axis=0).reshape(height,width)
        self.valid=np.isfinite(scores.max(axis=0)).reshape(height,width)
    def render(self,images,objects):
        canvas=np.zeros((self.height,self.width,3),np.uint8)
        labels=[]
        for i,c in enumerate(self.ids):
            warped=cv2.remap(images[c],self.maps[c],None,cv2.INTER_LINEAR)
            mask=(self.owner==i)&self.valid;canvas[mask]=warped[mask]
        for o in objects:
            d=next((d for d in o['dets'] if d['src']==3),max(o['dets'],key=lambda d:d['conf']))
            a,b,c,e=d['box'];ray=self.cameras[d['src']].rays([[a,b],[c,b],[c,e],[a,e]])@self.cameras[3].R
            if not np.isfinite(ray).all() or np.any(ray[:,2]<=0):continue
            uv=ray[:,:2]/ray[:,2,None]*self.f+[self.width/2,self.height/2]
            pts=np.clip(uv,-100000,100000).astype(np.int32)
            cv2.polylines(canvas,[pts],True,COLORS[o['label']],2)
            labels.append((f"#{o['id']} {core.CLASS_NAMES[o['cls']]} {o['conf']:.2f}",pts[0],COLORS[o['label']]))
        draw_box_labels(canvas,labels)
        put_label(canvas,'FUSED FRONT | rotation-only panorama; near-object seams are not detection errors',(10,20),(255,255,255))
        return canvas

def render(images,objects,removed,panorama,frame,mode):
    panels=[]
    for cid,img in images.items():
        w=480 if len(images)>1 else 1280;scale=w/img.shape[1];h=round(img.shape[0]*scale)
        panel=cv2.resize(img,(w,h))
        labels=[]
        for d in removed:
            if d['src']!=cid:continue
            x1,y1,x2,y2=(np.asarray(d['box'])*scale).astype(int)
            cv2.rectangle(panel,(x1,y1),(x2,y2),(140,140,140),1)
            put_label(panel,'EGO EXCLUDED',(x1,y1),(200,200,200))
        for o in objects:
            for d in o['dets']:
                if d['src']!=cid:continue
                x1,y1,x2,y2=(np.asarray(d['box'])*scale).astype(int);color=COLORS[o['label']]
                cv2.rectangle(panel,(x1,y1),(x2,y2),color,2)
                labels.append((f"#{o['id']} {core.CLASS_NAMES[d['cls']]} {d['conf']:.2f}",(x1,y1),color))
        draw_box_labels(panel,labels,.35 if len(images)>1 else .5)
        put_label(panel,f'Cam{cid} | original-coordinate boxes',(8,20),(255,255,255));panels.append(panel)
    content=np.hstack(panels)
    if panorama is not None:
        height=max(panorama.height,75+len(objects)*42)
        top=np.full((height,1440,3),24,np.uint8);top[:panorama.height,:panorama.width]=panorama.render(images,objects)
        put_label(top,'OBJECT LIST (frame-local IDs)',(1050,22),(255,255,255),.48)
        put_label(top,'no-match: unassociated, NOT proven miss',(1050,43),(220,220,220),.4)
        for i,o in enumerate(objects):
            y=72+i*42;color=COLORS[o['label']]
            put_label(top,f"#{o['id']} {core.CLASS_NAMES[o['cls']]} {o['conf']:.2f} {o['label']}",(1050,y),color,.5)
            put_label(top,f"support:{','.join(map(str,o['src']))}   no-match:{','.join(map(str,o['unmatched_cameras'])) or '-'}",(1050,y+18),color,.43)
        content=np.vstack([top,content])
    header=np.full((64,content.shape[1],3),24,np.uint8)
    put_label(header,f'{frame} | {mode.upper()} | objects={len(objects)} | ego excluded detections={len(removed)}',(10,20),(255,255,255),.48)
    put_label(header,'N/Right: next  B/Left: previous  SPACE: pause/auto  Q/ESC: quit | no-match != proven missed detection',(10,43),(220,220,220),.43)
    return np.vstack([header,content])

def command(key):
    if key in (ord('q'),27):return 'quit'
    if key==32:return 'toggle'
    if key in (ord('b'),2424832,81):return 'previous'
    if key in (ord('n'),ord('d'),13,2555904,83):return 'next'
    return None

def main(cid=None):
    ap=argparse.ArgumentParser(description='Final front fusion viewer' if cid is None else f'Plain YOLO Cam{cid} viewer')
    ap.add_argument('--seq',type=Path,default=ROOT/'ultralytics/mycar/images')
    ap.add_argument('--model',type=Path,default=ROOT/'ultralytics/yolo11n.pt')
    ap.add_argument('--mode',choices=['step','auto'],default='step')
    ap.add_argument('--start',default='',help='First frame stem, e.g. 00001222')
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--fps',type=float,default=5);ap.add_argument('--max-width',type=int,default=1440);ap.add_argument('--max-height',type=int,default=900)
    ap.add_argument('--headless',action='store_true');ap.add_argument('--no-ego',action='store_true')
    ap.add_argument('--cache',action='store_true',help='Read existing validated detection cache; fail if missing/stale')
    ap.add_argument('--bev-mode',choices=['off','audit','suppress'],default='off',help='Experimental sparse BEV verification; baseline unchanged by default')
    ap.add_argument('--bev-config',type=Path,default=ROOT/'ultralytics/experiments/bev_verification/frozen_config.json')
    ap.add_argument('--save',action='store_true',help='Save each visited visualization')
    ap.add_argument('--out',type=Path,default=ROOT/'ultralytics/experiments/final_runs')
    args=ap.parse_args()
    if cid is not None and args.bev_mode!='off':ap.error('BEV verification is for three-camera fusion only; singles remain unchanged')
    if args.limit<0 or args.stride<1 or args.fps<=0 or min(args.max_width,args.max_height)<320:ap.error('Invalid limit/stride/fps/window size')
    if args.headless and args.mode=='step':ap.error('Step mode needs GUI; use --mode auto --headless for batch runs')
    cameras=load_cameras();cameras={c:cameras[c] for c in ((2,3,4) if cid is None else (cid,))}
    dirs={c:args.seq/str(c) for c in cameras}
    if any(not p.is_dir() for p in dirs.values()):ap.error('Missing camera directory')
    names=sorted(set.intersection(*[{p.name for p in path.glob('*.png')} for path in dirs.values()]))
    if args.start and f'{args.start}.png' not in names:ap.error('Start frame is not available in all required cameras')
    names=[n for n in names if n[:-4]>=args.start][::args.stride]
    if args.limit:names=names[:args.limit]
    if not names:ap.error('No synchronized frames')
    model_hash=hashlib.sha256(args.model.read_bytes()).hexdigest()
    expected=json.loads((ROOT/'ultralytics/algorithm/dedup_config.json').read_text())['model_sha256']
    if model_hash!=expected:ap.error('Model differs from validated YOLO11n weights')
    bev_config=None
    if args.bev_mode!='off':
        from bev_verification import verify
        if not args.bev_config.exists():ap.error('Run core_algorithms/experiment_bev_verification.py first')
        bev_config=json.loads(args.bev_config.read_text())
        module_hash=hashlib.sha256(Path(__file__).with_name('bev_verification.py').read_bytes()).hexdigest()
        if bev_config['model_sha256']!=model_hash or bev_config['code_sha256']!=module_hash:ap.error('BEV config model/code mismatch; rerun experiment')
        if bev_config['selected'] is None:ap.error('No BEV candidate passed development selection')
    cache=ROOT/'ultralytics/experiments/detection_cache'/f'{model_hash[:12]}_960_conf010'
    if args.cache:
        missing=[f'{c}/{n}' for n in names for c in cameras if not (cache/f'{c}_{n[:-4]}.json').exists()]
        if missing:ap.error(f'Cache missing for {missing[0]} ({len(missing)} entries). Remove --cache for actual inference.')
    detector=None
    if not args.cache:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4);detector=YOLO(str(args.model))
    rect=core.TriRectifier(core.camera_params) if cid is None else None
    panorama=FrontPanorama(cameras,width=1040) if cid is None else None
    output=args.out/f"{'three_cam' if cid is None else 'cam'+str(cid)}_{time.time_ns()}";output.mkdir(parents=True)
    metadata=dict(cameras=list(cameras),model_sha256=model_hash,imgsz=960,ego_filter=not args.no_ego,
        bev_mode=args.bev_mode,bev_config=bev_config,
        ego_policy='bottom_large_vehicle_v1_rig_specific',single_thresholds=SINGLE_THRESHOLDS,front_thresholds=FRONT_THRESHOLDS,dedup=DEDUP_GATE if cid is None else None,
        source=str(args.seq.resolve()),mode=args.mode,note='IDs are frame-local; no-match means unassociated, not proven detector miss. FULL is support count, not correctness. Single runs have no extra duplicate suppression.')
    (output/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    index=0;auto=args.mode=='auto';visited=set();window='Front fusion' if cid is None else f'Cam{cid}'
    try:
        with (output/'objects.jsonl').open('w',encoding='utf-8') as log:
            while 0<=index<len(names):
                name=names[index];start=time.perf_counter();images={c:cv2.imread(str(dirs[c]/name)) for c in cameras};dets=[]
                for c,img in images.items():
                    cam=cameras[c]
                    if img is None or img.shape[:2]!=(cam.height,cam.width):raise ValueError(f'Image size/decode mismatch: {c}/{name}')
                    if args.cache:
                        entry=json.loads((cache/f'{c}_{name[:-4]}.json').read_text());st=(dirs[c]/name).stat()
                        if entry['image_signature']!=[st.st_size,st.st_mtime_ns]:raise ValueError(f'Stale cache {c}/{name}')
                        dets.extend(entry['dets'])
                    else:
                        for row in detector(img,conf=.1,imgsz=960,device='cpu',verbose=False)[0].boxes.data.cpu().numpy():
                            *box,conf,cls=row
                            if int(cls) in HEIGHT:dets.append(dict(src=c,box=list(map(float,box)),conf=float(conf),cls=int(cls),feature=feature(img,box)))
                objects,removed=run_pipeline(dets,cameras,cid,not args.no_ego,rect)
                bev_audit=[]
                if bev_config is not None:objects,bev_audit=verify(objects,cameras,bev_config['selected'],args.bev_mode)
                serialized=[dict(id=o['id'],label=o['label'],cls=o['cls'],conf=o['conf'],supported_cameras=o['src'],unmatched_cameras=o['unmatched_cameras'],
                    members=[{k:d[k] for k in ('src','box','cls','conf')} for d in o['dets']],bev=o.get('bev'),suppressed_group_indices=o.get('suppressed_group_indices',[])) for o in objects]
                if index not in visited:
                    log.write(json.dumps(dict(frame=name,objects=serialized,bev_audit=bev_audit,ego_removed=[{k:d[k] for k in ('src','box','cls','conf')} for d in removed],processing_s=time.perf_counter()-start))+'\n');log.flush();visited.add(index)
                canvas=render(images,objects,removed,panorama,name,'auto' if auto else 'step')
                factor=min(1,args.max_width/canvas.shape[1],args.max_height/canvas.shape[0]);canvas=cv2.resize(canvas,None,fx=factor,fy=factor)
                if args.save:cv2.imwrite(str(output/name),canvas)
                print(f'{index+1}/{len(names)} {name} objects={len(objects)} ego={len(removed)}',flush=True)
                if args.headless:index+=1;continue
                advance=False
                while not advance:
                    cv2.imshow(window,canvas)
                    key=cv2.waitKeyEx(max(1,round(1000/args.fps)) if auto else 50)
                    if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:return
                    action=command(key)
                    if action=='quit':return
                    if action=='toggle':
                        auto=not auto
                        canvas=render(images,objects,removed,panorama,name,'auto' if auto else 'step')
                        canvas=cv2.resize(canvas,None,fx=factor,fy=factor)
                    elif action=='previous':index=max(0,index-1);advance=True
                    elif action=='next' or auto and key==-1:index+=1;advance=True
    finally:
        cv2.destroyAllWindows();print(f'Results: {output}',flush=True)
