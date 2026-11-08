"""Validated current five-camera cached prediction viewer and fresh GT stats."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'ultralytics/research'),str(ROOT/'ultralytics/core_algorithms')]
import cv2
import numpy as np
import experiment_road_support_admission as E
import protected_conditional_recovery as C
from protected_conditional_inputs import ProtectedInputs
from fit_protected_conditional_recovery import OUT
from five_camera_display import Panorama,compose
from validate_multiview import evaluate


class CachedScene:
    def __init__(self):
        self.report=json.loads((E.BASE/'algorithm/evaluation_reports/protected_conditional_full_results_20261010.json').read_text())
        self.cfg=json.loads((OUT/'protocol.json').read_text());self.inputs=ProtectedInputs()
        if not self.report['complete'] or self.report['protocol_sha256']!=E.sha(OUT/'protocol.json') or self.report['model_sha256']!=E.sha(OUT/'models.json'):
            raise ValueError('Current five-camera report/model/protocol mismatch')
        E.verify_hashes(self.cfg['source_sha256'])
        self.rows={r['frame']:r for r in self.report['rows']}
        self.names=self.cfg['frames']

    def frame(self,s):
        path=OUT/f'evaluation_full/frames/{s}.json'
        state=E.checked(path,self.report['frame_sha256'][str(path.relative_to(ROOT))])
        E.verify_hashes(state['source_sha256'])
        # Native images must still be the ones that produced the frozen cache.
        descriptor=json.loads(self.inputs.descriptors[s].read_text())
        E.verify_hashes(descriptor['input_sha256'])
        pool,changed,_,_,_,_=self.inputs.load(s)
        pair={tuple(k):v for k,v in zip(state['pairs'],state['pair_scores'])}
        groups,trace=C.select(state['reference'],state['groups']['single'],state['scores']['single'],.55,pool,pair,False)
        objects=E.objects(groups,changed);count=len(state['reference'])
        for i,o in enumerate(objects):
            o.update(id=i+1,conf=max(d['conf'] for d in o['dets']),recovered=i>=count)
        return objects,trace


def run():
    ap=argparse.ArgumentParser(description='Current five-camera panorama + BEV + native views; validated cache replay')
    ap.add_argument('--mode',choices=('step','auto'),default='step');ap.add_argument('--headless',action='store_true')
    ap.add_argument('--start',default='');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--fps',type=float,default=3);ap.add_argument('--save',action='store_true');ap.add_argument('--no-eval',action='store_true')
    ap.add_argument('--panorama',choices=('rotation','natural'),default='rotation',
                    help='Natural: display-only seam-selected sphere + Cam3 anchor; original statistics unchanged')
    ap.add_argument('--max-width',type=int,default=1750);ap.add_argument('--max-height',type=int,default=1000)
    ap.add_argument('--out',type=Path,default=E.BASE/'experiments/five_camera_viewer_20261012')
    args=ap.parse_args()
    if args.limit<0 or args.stride<1 or args.fps<=0 or min(args.max_width,args.max_height)<320:
        ap.error('Invalid limit/stride/fps/window size')
    if args.headless and args.mode=='step':ap.error('Use --mode auto with --headless')
    cv2.setNumThreads(1);scene=CachedScene();cameras=scene.inputs.cameras
    names=[s for s in scene.names if s>=args.start][::args.stride]
    if args.limit:names=names[:args.limit]
    if not names:ap.error('No frames in current signed prediction cache')
    output=args.out/f'run_{time.time_ns()}';output.mkdir(parents=True,exist_ok=False)
    E.write_new(output/'config.json',dict(method='single/independent/tau0.55',frames=names,cameras=[1,2,3,4,5],
        prediction_report_sha256=E.sha(E.BASE/'algorithm/evaluation_reports/protected_conditional_full_results_20261010.json'),
        calibration_sha256={p:E.sha(ROOT/p) for p in ('Intrinsics_cam.json','T_imu_cam(1).json')},
        cache_replay=True,no_eval=args.no_eval,panorama=args.panorama,
        projection='rotation spherical LUT' if args.panorama=='rotation' else 'seam-selected display spheres + anchored panorama',
        display_only=True,
        display_source_sha256={str(p.relative_to(ROOT)):E.sha(p) for p in (
            Path(__file__),ROOT/'ultralytics/research/five_camera_display.py',
            ROOT/'ultralytics/research/natural_panorama.py')},
        bev='detector centre triangulation with uncertainty; unknown direction markers are not metric points'))
    if args.panorama=='natural':
        from natural_panorama import NaturalPanorama
        panorama=NaturalPanorama(cameras)
    else:panorama=Panorama(cameras)
    visited={};totals=np.zeros(5,int);located=0;all_objects=0
    index=0;auto=args.mode=='auto';status='stopped';window='Five cameras | panorama + BEV'
    try:
        with (output/'objects.jsonl').open('x',encoding='utf-8') as log:
            while 0<=index<len(names):
                s=names[index];objects,trace=scene.frame(s)
                images={c:cv2.imread(str(E.BASE/f'mycar/images/{c}/{s}.png')) for c in (1,2,3,4,5)}
                for c,im in images.items():
                    if im is None or im.shape[:2]!=(cameras[c].height,cameras[c].width):raise ValueError(f'Invalid source image {c}/{s}')
                canvas,states,visible=compose(images,objects,panorama,s)
                if s not in visited:
                    row=dict(frame=s,objects=[dict(id=o['id'],cls=o['cls'],sources=o['src'],recovered=o['recovered'],
                        members=[{k:d[k] for k in ('uid','src','cls','conf','box')} for d in o['dets']],
                        bev=states[o['id']],panorama_visible=o['id'] in visible) for o in objects])
                    if args.panorama=='natural':row['display']=panorama.diagnostics
                    if not args.no_eval:
                        gt=E.truth(s,(1,2,3,4,5));counts=evaluate(objects,gt).tolist()
                        expected=scene.rows[s]['counts']['single/independent/tau0.55']
                        if counts[:3]!=expected[:3]:raise ValueError('Viewer independent evaluation differs from validated method')
                        row['counts']=counts;totals+=counts
                    visited[s]=row;log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                    located+=sum(st['status']=='estimated' for st in states.values());all_objects+=len(objects)
                    if args.save:cv2.imwrite(str(output/f'{s}.png'),canvas)
                if not args.no_eval:
                    now=E.metrics(visited[s]['counts']);cum=E.metrics(totals)
                    print(f"{s} frame P={now['precision']:.2%} R={now['recall']:.2%} | cumulative {len(visited)} P={cum['precision']:.2%} R={cum['recall']:.2%} F1={cum['f1']:.2%}",flush=True)
                    cv2.putText(canvas,f"Five-view GT | visited {len(visited)} | P {cum['precision']:.2%} R {cum['recall']:.2%} F1 {cum['f1']:.2%}",(10,62),0,.38,(230,230,230),1)
                if args.headless:index+=1;continue
                factor=min(1,args.max_width/canvas.shape[1],args.max_height/canvas.shape[0]);show=cv2.resize(canvas,None,fx=factor,fy=factor)
                cv2.imshow(window,show);key=cv2.waitKeyEx(max(1,round(1000/args.fps)) if auto else 0)
                if key in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
                if key==32:auto=not auto
                elif key in (ord('b'),2424832,81):index=max(index-1,0)
                elif key in (ord('n'),13,2555904,83):index+=1
                elif auto:index+=1
            if index>=len(names):status='selection_completed'
    finally:
        if not args.headless:cv2.destroyAllWindows()
        summary=dict(status=status,visited=len(visited),selected=len(names),complete_selection=len(visited)==len(names),
                     metrics=E.metrics(totals) if not args.no_eval else None,located_objects=located,output_objects=all_objects,
                     note='Counts are only visited frames; global five-view GT. BEV locations are approximate, no 3D truth.')
        E.write_new(output/'metrics.json',summary);print(json.dumps(dict(output=str(output),**summary)),flush=True)


if __name__=='__main__':run()
