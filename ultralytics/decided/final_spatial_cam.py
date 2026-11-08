"""Extra low-cost 3D-like surround / road texture BEV; frozen fusion replay."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ultralytics/decided')]
from final_five_cam import CachedScene
import experiment_road_support_admission as E
from five_camera_display import locate
from spatial_surround import SpatialSurround,display_plane
from spatial_dashboard import SpatialDashboard,desktop_work_area,fit_work_area
from validate_multiview import evaluate
import cv2
import numpy as np


def run():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('auto','step'),default='step')
    ap.add_argument('--view',choices=('spatial','panorama'),default='spatial')
    ap.add_argument('--page',choices=('overview','cameras','objects'),default='overview',help='Compact UI page; keys 1/2/3 switch pages')
    ap.add_argument('--ground-z',type=float,default=None,help='Optional explicit assumed flat plane height in IMU metres')
    ap.add_argument('--plane-ref',default='00000000',help='Signed stereo plane reference, frozen for display')
    ap.add_argument('--headless',action='store_true');ap.add_argument('--save',action='store_true')
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--stride',type=int,default=1);ap.add_argument('--start',default='')
    ap.add_argument('--fps',type=float,default=3)
    ap.add_argument('--max-width',type=int,default=1280,help='Initial/headless canvas width; interactive resizing can exceed it')
    ap.add_argument('--max-height',type=int,default=720,help='Initial/headless canvas height; interactive resizing can exceed it')
    ap.add_argument('--out',type=Path,default=E.BASE/'experiments/spatial_viewer_20261012')
    args=ap.parse_args()
    if args.limit<0 or args.stride<1 or args.fps<=0 or args.max_width<640 or args.max_height<440:ap.error('Invalid rate or dimensions; compact UI needs at least 640x440')
    if args.headless and args.mode=='step':ap.error('Headless requires --mode auto')
    work=None if args.headless else desktop_work_area()
    width,height=fit_work_area(args.max_width,args.max_height,work)
    if width<640 or height<440:ap.error('Available desktop is smaller than 640x440; use headless or a larger display')
    dashboard=SpatialDashboard(width,height)
    cv2.setNumThreads(1);scene=CachedScene();cameras=scene.inputs.cameras
    beta,plane=display_plane(args.ground_z,args.plane_ref)
    renderer=SpatialSurround(cameras,beta);panorama=None
    names=[s for s in scene.names if s>=args.start][::args.stride]
    if args.limit:names=names[:args.limit]
    if not names:ap.error('No signed frames selected')
    output=args.out/f'run_{time.time_ns()}';output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),ROOT/'ultralytics/research/spatial_surround.py',ROOT/'ultralytics/research/spatial_observation_overlay.py',ROOT/'ultralytics/research/spatial_dashboard.py',ROOT/'ultralytics/research/five_camera_display.py',ROOT/'ultralytics/decided/final_five_cam.py']
    config=dict(method='single/independent/tau0.55',frames=names,cameras=[1,2,3,4,5],no_eval=False,
        prediction_report_sha256=E.sha(E.BASE/'algorithm/evaluation_reports/protected_conditional_full_results_20261010.json'),
        calibration_sha256={p:E.sha(ROOT/p) for p in ('Intrinsics_cam.json','T_imu_cam(1).json')},
        cache_replay=True,view=args.view,display_only=True,plane=plane,
        ui=dict(page=args.page,width=width,height=height,desktop_work_area=work,layout='responsive_tabs',fullscreen_key='f'),
        display_source_sha256={str(p.relative_to(ROOT)):E.sha(p) for p in paths},
        bev='Original detector-centre XY estimates over assumed road texture; not ground contacts',
        surface='Fixed flat-centre quadratic bowl, NOT recovered scene shape')
    E.write_new(output/'config.json',config)
    totals=np.zeros(5,int);visited={};located=0;all_objects=0;index=0;auto=args.mode=='auto';view=args.view;status='stopped'
    window='Spatial Surround';page=args.page;object_page=0;loaded=None;dirty=True;pending=[];pano_image=None;fullscreen=False
    deadline=time.monotonic()+1/args.fps
    if not args.headless:
        # Freeratio refers to the outer canvas, not scene images: those remain
        # letterboxed inside their cards. Reflow UI to actual client dimensions.
        cv2.namedWindow(window,cv2.WINDOW_NORMAL|cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(window,width,height)
        if work is not None:cv2.moveWindow(window,work[0]+32,work[1]+32)
        def mouse(event,x,y,flags,param):
            if event==cv2.EVENT_LBUTTONUP:
                key=dashboard.hit(x,y)
                if key is not None:pending.append(key)
        cv2.setMouseCallback(window,mouse)
    try:
        with (output/'objects.jsonl').open('x',encoding='utf-8') as stream:
            while 0<=index<len(names):
                if not args.headless:
                    try:
                        _,_,client_w,client_h=cv2.getWindowImageRect(window)
                        if client_w>0 and client_h>0:
                            size=(max(640,client_w),max(440,client_h))
                            if size!=(dashboard.width,dashboard.height):
                                dashboard=SpatialDashboard(*size);dirty=True
                    except cv2.error:pass  # Some backends lack client-size queries.
                s=names[index];fresh=False
                if loaded!=s:
                    objects,_=scene.frame(s)
                    images={c:cv2.imread(str(E.BASE/f'mycar/images/{c}/{s}.png')) for c in (1,2,3,4,5)}
                    if any(im is None or im.shape[:2]!=(cameras[c].height,cameras[c].width) for c,im in images.items()):raise ValueError('Bad native image')
                    states={o['id']:locate(o,cameras) for o in objects}
                    start=time.perf_counter();space,bev=renderer.render(images,objects,states)
                    raster_s=time.perf_counter()-start
                    loaded=s;pano_image=None;dirty=True;object_page=0;deadline=time.monotonic()+1/args.fps
                if s not in visited:
                    counts=evaluate(objects,E.truth(s,(1,2,3,4,5))).tolist()
                    if counts!=scene.rows[s]['counts']['single/independent/tau0.55']:raise ValueError('Frozen evaluation changed')
                    totals+=counts;fresh=True
                if dirty or fresh:
                    if view=='panorama' and page=='overview' and pano_image is None:
                        if panorama is None:
                            from natural_panorama import NaturalPanorama
                            panorama=NaturalPanorama(cameras)
                        pano_image,_=panorama.render(images,objects)
                    start=time.perf_counter()
                    canvas=dashboard.render(images,objects,states,space,bev,s,E.metrics(totals),len(visited)+int(fresh),
                        auto=auto,page=page,view=view,panorama=pano_image,object_page=object_page,fullscreen=fullscreen)
                    elapsed=raster_s+time.perf_counter()-start
                    dirty=False
                    if not args.headless:cv2.imshow(window,canvas)
                if fresh:
                    row=dict(frame=s,counts=counts,display=renderer.diagnostics,compose_spatial_s=elapsed,
                        objects=[dict(id=o['id'],cls=o['cls'],sources=o['src'],recovered=o['recovered'],
                            members=[{k:d[k] for k in ('uid','src','cls','conf','box')} for d in o['dets']],bev=states[o['id']]) for o in objects])
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();visited[s]=row
                    located+=sum(st['status']=='estimated' for st in states.values());all_objects+=len(objects)
                    if args.save:
                        cv2.imwrite(str(output/f'{s}.png'),canvas)
                        # Share the already rendered frame: no second raster pass.
                        cv2.imwrite(str(output/f'{s}_space.png'),renderer.last_space)
                        cv2.imwrite(str(output/f'{s}_bev.png'),renderer.last_bev)
                    cumulative=E.metrics(totals)
                    print(f"{s} {len(visited)} frames | P={cumulative['precision']:.2%} R={cumulative['recall']:.2%} F1={cumulative['f1']:.2%} | spatial display={elapsed:.3f}s",flush=True)
                if args.headless:index+=1;continue
                key=cv2.waitKeyEx(20)
                if pending:key=pending.pop(0)
                if key==ord('q') or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
                if key==27:
                    if not fullscreen:break
                    fullscreen=False;cv2.setWindowProperty(window,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window,width,height);dirty=True
                elif key==ord('f'):
                    fullscreen=not fullscreen
                    cv2.setWindowProperty(window,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:cv2.resizeWindow(window,width,height)
                    dirty=True
                elif key==32:auto=not auto;dirty=True;deadline=time.monotonic()+1/args.fps
                elif key==ord('v'):view='panorama' if view=='spatial' else 'spatial';page='overview';dirty=True
                elif key in (ord('1'),ord('2'),ord('3')):page={ord('1'):'overview',ord('2'):'cameras',ord('3'):'objects'}[key];dirty=True
                elif key in (ord('['),ord(']')):
                    rows=dashboard.object_rows;pages=max(1,(len(objects)+rows-1)//rows)
                    object_page=int(np.clip(object_page+(1 if key==ord(']') else -1),0,pages-1));dirty=True
                elif key in (ord('b'),2424832,81):index=max(0,index-1);deadline=time.monotonic()+1/args.fps
                elif key in (ord('n'),13,2555904,83):index+=1
                elif auto and time.monotonic()>=deadline:index+=1
            if index>=len(names):status='selection_completed'
    finally:
        if not args.headless:cv2.destroyAllWindows()
        summary=dict(status=status,complete_selection=len(visited)==len(names),visited=len(visited),selected=len(names),metrics=E.metrics(totals),
            located_objects=located,output_objects=all_objects,display_init_s=renderer.init_s,
            median_spatial_compose_s=float(np.median([r['compose_spatial_s'] for r in visited.values()])) if visited else None,
            note='Visited-frame metrics only; no new depth/3D GT, fixed reference plane can fail on other roads')
        E.write_new(output/'metrics.json',summary)
        print(json.dumps(dict(output=str(output),**summary)),flush=True)


if __name__=='__main__':run()
