"""Append-only real-frame UI previews and frozen-output regression."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ultralytics/decided'))
from final_five_cam import CachedScene
import experiment_road_support_admission as E
from five_camera_display import locate
from spatial_surround import SpatialSurround,display_plane
from spatial_dashboard import SpatialDashboard
from validate_multiview import evaluate


def fingerprint(value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=convert,allow_nan=False).encode()).hexdigest()


def run(args):
    args.out=args.out.resolve()
    if args.out.exists():raise FileExistsError('Append-only display experiment')
    cv2.setNumThreads(1);scene=CachedScene();cs=scene.inputs.cameras
    beta,plane=display_plane();renderer=SpatialSurround(cs,beta);totals=np.zeros(5,int);rows=[]
    paths=[Path(__file__),ROOT/'ultralytics/research/spatial_dashboard.py',ROOT/'ultralytics/research/spatial_surround.py',ROOT/'ultralytics/research/spatial_observation_overlay.py',ROOT/'ultralytics/decided/final_spatial_cam.py']
    config=dict(frames=args.frames,plane=plane,source_sha256={str(p.relative_to(ROOT)):E.sha(p) for p in paths},
        sizes=args.sizes,pages=['overview','cameras','objects'],display_only=True)
    E.write_new(args.out/'protocol.json',config)
    for index,s in enumerate(args.frames,1):
        objects,_=scene.frame(s);before=fingerprint(objects)
        images={c:cv2.imread(str(E.BASE/f'mycar/images/{c}/{s}.png')) for c in cs}
        states={o['id']:locate(o,cs) for o in objects};original=copy.deepcopy(states)
        counts=evaluate(objects,E.truth(s,(1,2,3,4,5))).tolist()
        if counts!=scene.rows[s]['counts']['single/independent/tau0.55']:raise ValueError('Frozen counts drift')
        totals+=counts;start=time.perf_counter();space,bev=renderer.render(images,objects,states);raster_s=time.perf_counter()-start
        previews=[]
        for w,h in config['sizes']:
            ui=SpatialDashboard(w,h)
            for page in config['pages']:
                start=time.perf_counter()
                canvas=ui.render(images,objects,states,space,bev,s,E.metrics(totals),index,page=page)
                compose_s=time.perf_counter()-start
                if canvas.shape!=(h,w,3):raise ValueError('UI dimension drift')
                path=args.out/s/f'{w}x{h}_{page}.png';path.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(path),canvas):raise OSError('Preview write failed')
                previews.append(dict(size=[w,h],page=page,ui_s=compose_s,path=str(path.relative_to(ROOT))))
        if fingerprint(objects)!=before or fingerprint(states)!=fingerprint(original):raise ValueError('UI mutated original objects or BEV')
        rows.append(dict(frame=s,counts=counts,object_sha256=before,object_and_bev_unchanged=True,raster_s=raster_s,previews=previews))
        print(f'Compact UI {s}: counts {counts[:3]} / {len(previews)} previews / unchanged',flush=True)
    if any(E.sha(p)!=config['source_sha256'][str(p.relative_to(ROOT))] for p in paths):raise ValueError('Source changed during validation')
    E.write_new(args.out/'results.json',dict(complete=True,rows=rows,counts=totals.tolist(),metrics=E.metrics(totals),
        protocol_sha256=E.sha(args.out/'protocol.json'),new_network_inference=0,
        note='Headless layout checks, not interactive window/monitor testing; no claim of P/R gain or measured speed improvement'))
    print(json.dumps(dict(output=str(args.out),previews=len(rows)*len(config['sizes'])*len(config['pages']),metrics=E.metrics(totals))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--frames',nargs='+',default=['00000010','00000650','00001200'])
    ap.add_argument('--sizes',nargs='+',default=['1280x720','1024x640','800x540'])
    ap.add_argument('--out',type=Path,default=ROOT/'ultralytics/experiments/spatial_compact_ui_20261013'/f'validation_{time.time_ns()}')
    args=ap.parse_args()
    try:
        args.sizes=[list(map(int,value.lower().split('x'))) for value in args.sizes]
        if any(len(size)!=2 or size[0]<640 or size[1]<440 for size in args.sizes):raise ValueError('Minimum 640x440')
    except ValueError:ap.error('Sizes must be WIDTHxHEIGHT, at least 640x440')
    run(args)
