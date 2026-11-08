"""Movie-like 360 look-around: drag/zoom current five-camera scene, cached fusion."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ultralytics/decided')]
from final_five_cam import CachedScene
import experiment_road_support_admission as E
from five_camera_display import radar,colour,text,CAM_ORDER
from immersive_surround import ImmersiveSphere,LookViewport,LookController,overview
from immersive_dashboard import SurroundDashboard
from spatial_dashboard import desktop_work_area,fit_work_area
from validate_multiview import evaluate
import cv2
import numpy as np


def natives(images,objects,width):
    tw=width//5;th=round(tw*2056/2464);tiles=[]
    for cid in CAM_ORDER:
        image=cv2.resize(images[cid],(tw,th))
        for o in objects:
            for d in o['dets']:
                if d['src']!=cid:continue
                q=np.rint(np.asarray(d['box'])*([tw/images[cid].shape[1],th/images[cid].shape[0]]*2)).astype(int)
                cv2.rectangle(image,tuple(q[:2]),tuple(q[2:]),colour(o),1);text(image,f"#{o['id']}",q[:2]+[0,-3],colour(o),.3)
        cv2.rectangle(image,(0,0),(tw,25),(22,22,22),-1);text(image,f'Cam{cid}',(8,18));tiles.append(image)
    result=np.hstack(tiles)
    return cv2.copyMakeBorder(result,0,0,0,width-result.shape[1],cv2.BORDER_CONSTANT)


def run():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('auto','step'),default='step')
    ap.add_argument('--dashboard',action='store_true',help='Optional small BEV inset; T toggles it (camera views are a separate page)')
    ap.add_argument('--page',choices=('look','sphere','cameras','bev'),default='look')
    ap.add_argument('--yaw',type=float,default=0);ap.add_argument('--pitch',type=float,default=0);ap.add_argument('--fov',type=float,default=90)
    ap.add_argument('--headless',action='store_true');ap.add_argument('--save',action='store_true')
    ap.add_argument('--start',default='');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--fps',type=float,default=3);ap.add_argument('--max-width',type=int,default=1750);ap.add_argument('--max-height',type=int,default=1050)
    ap.add_argument('--out',type=Path,default=E.BASE/'experiments/immersive_viewer_20261013')
    args=ap.parse_args()
    if args.limit<0 or args.stride<1 or args.fps<=0 or min(args.max_width,args.max_height)<320:ap.error('Invalid limit/rate/window')
    if args.headless and args.mode=='step':ap.error('Headless requires auto')
    cv2.setNumThreads(1);scene=CachedScene();cs=scene.inputs.cameras;sphere=ImmersiveSphere(cs)
    control=LookController();control.turn(args.yaw,args.pitch);control.fov=args.fov
    ui=SurroundDashboard(sphere)
    names=[s for s in scene.names if s>=args.start][::args.stride]
    if args.limit:names=names[:args.limit]
    if not names:ap.error('No signed frames selected')
    output=args.out/f'run_{time.time_ns()}';output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),ROOT/'ultralytics/research/immersive_surround.py',ROOT/'ultralytics/research/immersive_dashboard.py',ROOT/'ultralytics/decided/final_five_cam.py',ROOT/'ultralytics/research/five_camera_display.py']
    config=dict(method='single/independent/tau0.55',frames=names,cameras=list(CAM_ORDER),no_eval=False,
        prediction_report_sha256=E.sha(E.BASE/'algorithm/evaluation_reports/protected_conditional_full_results_20261010.json'),
        calibration_sha256={p:E.sha(ROOT/p) for p in ('Intrinsics_cam.json','T_imu_cam(1).json')},
        display_source_sha256={str(p.relative_to(ROOT)):E.sha(p) for p in paths},
        cache_replay=True,display_only=True,coverage=sphere.coverage,
        initial_view=dict(yaw=control.yaw,pitch=control.pitch,fov=control.fov,page=args.page),
        projection='Observed-ray equirectangular sphere + freely rotating rectilinear viewport; not full physical 360 coverage')
    E.write_new(output/'config.json',config)
    window='Surround | drag / wheel | M look P sphere C cameras G BEV | F fullscreen'
    width,height=fit_work_area(args.max_width,args.max_height,None if args.headless else desktop_work_area())
    screen=dict(scale=1.);dashboard=args.dashboard;page=args.page;fullscreen=False;pending=[]
    auto=args.mode=='auto';index=0;loaded=None
    visited={};totals=np.zeros(5,int);located=0;objects_total=0;status='stopped';last_frame=time.perf_counter();last_render_key=None

    def mouse(event,x,y,flags,param):
        hit=ui.hit(x,y)
        if event==cv2.EVENT_LBUTTONDOWN and hit is not None:pending.append(hit)
        else:ui.mouse(control,event,x,y,flags)

    if not args.headless:
        cv2.namedWindow(window,cv2.WINDOW_NORMAL|cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(window,width,height);cv2.setMouseCallback(window,mouse)
    try:
        with (output/'objects.jsonl').open('x',encoding='utf-8') as log:
            while 0<=index<len(names):
                s=names[index];new=s!=loaded
                if new:
                    objects,_=scene.frame(s)
                    images={c:cv2.imread(str(E.BASE/f'mycar/images/{c}/{s}.png')) for c in CAM_ORDER}
                    if any(im is None or im.shape[:2]!=(cs[c].height,cs[c].width) for c,im in images.items()):raise ValueError('Bad original images')
                    texture=sphere.render(images,objects);bev,states=radar(objects,cs,size=560)
                    if s not in visited:
                        counts=evaluate(objects,E.truth(s,(1,2,3,4,5))).tolist()
                        if counts!=scene.rows[s]['counts']['single/independent/tau0.55']:raise ValueError('Frozen statistics changed')
                        row=dict(frame=s,counts=counts,display=sphere.diagnostics,
                            objects=[dict(id=o['id'],cls=o['cls'],sources=o['src'],recovered=o['recovered'],
                                members=[{k:d[k] for k in ('uid','src','cls','conf','box')} for d in o['dets']],bev=states[o['id']]) for o in objects])
                        log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();visited[s]=row;totals+=counts
                        located+=sum(st['status']=='estimated' for st in states.values());objects_total+=len(objects)
                    loaded=s;last_frame=time.perf_counter()
                    now=E.metrics(totals)
                    print(f"{s} visited={len(visited)} | P={now['precision']:.2%} R={now['recall']:.2%} F1={now['f1']:.2%} | sphere={sphere.diagnostics['render_s']:.3f}s",flush=True)
                if not args.headless:
                    _,_,ww,hh=cv2.getWindowImageRect(window)
                    if ww>=320 and hh>=320:width,height=ww,hh
                render_key=(s,page,dashboard,width,height,control.yaw,control.pitch,control.fov)
                if render_key!=last_render_key:
                    canvas=ui.render(texture,control,images,objects,bev,s,E.metrics(totals),len(visited),
                                     width,height,page,dashboard);last_render_key=render_key
                if args.save and new:
                    cv2.imwrite(str(output/f'{s}.png'),canvas);cv2.imwrite(str(output/f'{s}_sphere.png'),texture)
                if args.headless:index+=1;continue
                cv2.imshow(window,canvas)
                key=cv2.waitKeyEx(20)
                if pending:key=pending.pop(0)
                if key in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
                if key==32:auto=not auto;last_frame=time.perf_counter()
                elif key==ord('f'):
                    fullscreen=not fullscreen
                    cv2.setWindowProperty(window,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:cv2.resizeWindow(window,args.max_width,args.max_height)
                elif key==ord('t'):dashboard=not dashboard
                elif key in (ord('m'),ord('p'),ord('c'),ord('g')):
                    page={ord('m'):'look',ord('p'):'sphere',ord('c'):'cameras',ord('g'):'bev'}[key];control.drag=None
                elif key==ord('r'):control.reset()
                elif key in (ord('a'),ord('d')):control.turn(15 if key==ord('a') else -15)
                elif key in (ord('w'),ord('s')):control.turn(pitch=10 if key==ord('w') else -10)
                elif key in (ord('1'),ord('2'),ord('3'),ord('4')):
                    control.yaw={ord('1'):0.,ord('2'):90.,ord('3'):-180.,ord('4'):-90.}[key];control.pitch=0.
                elif key in (ord('n'),13,2555904,83):index+=1
                elif key in (ord('b'),2424832,81):index=max(0,index-1)
                if (auto and 0<=index<len(names) and loaded==names[index]
                        and time.perf_counter()-last_frame>=1/args.fps):
                    index+=1
            if index>=len(names):status='selection_completed'
    finally:
        if not args.headless:cv2.destroyAllWindows()
        summary=dict(status=status,visited=len(visited),selected=len(names),complete_selection=len(visited)==len(names),
            metrics=E.metrics(totals),located_objects=located,output_objects=objects_total,coverage=sphere.coverage,
            note='Only visited frames. 360 look direction, not guaranteed full 360 recorded scene. Dragging does not rerun detector or evaluation.')
        E.write_new(output/'metrics.json',summary);print(json.dumps(dict(output=str(output),**summary)),flush=True)


if __name__=='__main__':run()
