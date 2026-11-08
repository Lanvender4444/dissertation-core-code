"""Export actual project render buffers, not fabricated desktop screenshots.

Uses immutable signed predictions. Does not run YOLO, fit models or modify GT.
"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ultralytics/decided'), str(ROOT/'scripts')]
from final_five_cam import CachedScene
import experiment_road_support_admission as E
from five_camera_display import locate
from spatial_surround import SpatialSurround, display_plane
from spatial_dashboard import SpatialDashboard
from natural_panorama import NaturalPanorama
from validate_spatial_dashboard import fingerprint
from validate_multiview import evaluate


def run(args):
    args.out=args.out.resolve()
    if args.out.exists():
        raise FileExistsError('Append-only screenshot export')
    args.out.mkdir(parents=True)
    cv2.setNumThreads(1)
    scene = CachedScene()
    cameras = scene.inputs.cameras
    beta, plane = display_plane()
    renderer = SpatialSurround(cameras, beta)
    natural = NaturalPanorama(cameras)
    dashboard = SpatialDashboard(1600, 1000)
    rows = []
    for s in ('00000650', '00001200'):
        objects, _ = scene.frame(s)
        signature = fingerprint(objects)
        images = {c: cv2.imread(str(E.BASE/f'mycar/images/{c}/{s}.png')) for c in (1,2,3,4,5)}
        if any(im is None for im in images.values()):
            raise FileNotFoundError('Missing native image')
        image_hashes = {str((E.BASE/f'mycar/images/{c}/{s}.png').relative_to(ROOT)):
                        E.sha(E.BASE/f'mycar/images/{c}/{s}.png') for c in images}
        states = {o['id']: locate(o, cameras) for o in objects}
        state_signature = fingerprint(states)
        counts = evaluate(objects, E.truth(s,(1,2,3,4,5))).tolist()
        if counts != scene.rows[s]['counts']['single/independent/tau0.55']:
            raise ValueError('Frozen counts changed')
        space, bev = renderer.render(images, objects, states)
        panorama, _ = natural.render(images, objects)
        exports = [('cameras','spatial'), ('objects','spatial'), ('overview','panorama')] if s=='00000650' else [('overview','spatial')]
        files = []
        for page, view in exports:
            canvas = dashboard.render(images,objects,states,space,bev,s,E.metrics(counts),1,
                                      page=page,view=view,panorama=panorama)
            path = args.out/f'{s}_{page}_{view}.png'
            if not cv2.imwrite(str(path),canvas):
                raise OSError('Screenshot export failed')
            files.append(dict(path=str(path.relative_to(ROOT)),sha256=E.sha(path),page=page,view=view,size=[1600,1000]))
        if s=='00001200':
            path=args.out/f'{s}_bev_detail.png'
            if not cv2.imwrite(str(path),bev):raise OSError('BEV export failed')
            files.append(dict(path=str(path.relative_to(ROOT)),sha256=E.sha(path),page='bev_detail',size=list(bev.shape[1::-1])))
        if fingerprint(objects)!=signature or fingerprint(states)!=state_signature:
            raise ValueError('Rendering mutated perception')
        rows.append(dict(frame=s,counts=counts,metrics=E.metrics(counts),objects=len(objects),
                         prediction_fingerprint=signature,location_fingerprint=state_signature,
                         image_sha256=image_hashes,files=files,perception_unchanged=True))
        print(f'Exported real render buffers for {s}: {counts[:3]}',flush=True)
    sources=[Path(__file__),ROOT/'ultralytics/decided/final_five_cam.py',
             ROOT/'ultralytics/research/spatial_dashboard.py',ROOT/'ultralytics/research/spatial_surround.py',
             ROOT/'ultralytics/research/spatial_observation_overlay.py',ROOT/'ultralytics/research/natural_panorama.py']
    record=dict(complete=True,rows=rows,plane=plane,configuration='single/independent/tau0.55',
                source_sha256={str(p.relative_to(ROOT)):E.sha(p) for p in sources},
                numerical_source_sha256=E.sha(E.BASE/'algorithm/evaluation_reports/protected_conditional_full_results_20261010.json'),
                new_detector_calls=0,model_fits=0,
                scope='Headless screenshots of actual project render buffers; not OS-window captures. '
                      'Metrics displayed concern one visited frame, not full-sequence performance. '
                      'Selected illustrations, not a random sample, physical-ID benchmark or 3D ground truth.')
    E.write_new(args.out/'manifest.json',record)
    print(json.dumps(dict(complete=True,output=str(args.out),files=sum(len(r['files']) for r in rows))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    run(ap.parse_args())
