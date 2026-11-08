"""Full-frame Cam234 experiment: frozen methods, fresh inference, resumable rows.

No fitting or threshold search. ALL common images and XMLs are enumerated.
Inference, source/calibration/parameter hashes and image/annotation signatures
are recorded. Existing unversioned candidate/depth caches are not reused.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import cv2
import numpy as np
from final_pipeline import run_pipeline, filter_ego, core, SINGLE_THRESHOLDS
from multiview_geometry import ROOT, load_cameras
from multiview_fusion import feature
from validate_multiview import truth, evaluate, metrics
from stereo_depth import StereoDepth
from yolo_hr_detect import detect
import depth_bev_fusion as F
from run_depth_bev import admit_slim, slim, splits
from global_fusion_ideas import conservative_dedup, rescue_unexplained, needs_band

BASE = ROOT / 'ultralytics'
HERE = Path(__file__).resolve().parent
EXPECTED_MODEL = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def manifest(limit, shard_index=0, shard_count=1, threads=4):
    sets = [set(p.stem for p in (BASE/f'mycar/images/{c}').glob('*.png')) for c in (2,3,4)]
    sets += [set(p.stem for p in (BASE/f'mycar/cam_{c}').glob('*.xml')) for c in (2,3,4)]
    frames = sorted(set.intersection(*sets))
    if len(set.union(*sets)) != len(frames):
        raise ValueError('Image/XML coverage differs: explicitly resolve missing data, do not skip')
    all_count = len(frames)
    if limit:
        frames = frames[:limit]
    frames = frames[shard_index::shard_count]
    params = {k: json.loads((BASE/f'algorithm/depth_bev_params/{k}.json').read_text())
              for k in ('960_base','960_d','BAND_base','BAND_d','BAND_d_E1','BAND_dfo_E1')}
    old_splits, _ = splits()
    dev = set(old_splits['dev110']) | {s for p in params.values() for s in p['development_frames']}
    known = set(old_splits['regression180'] + old_splits['regression60'])
    hashes = {p.name: sha(p) for p in HERE.glob('*.py')}
    model = BASE/'yolo11n.pt'
    if sha(model) != EXPECTED_MODEL:
        raise ValueError('Unexpected model weights')
    signatures = {}
    for s in frames:
        signatures[s] = {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns]
                         for c in (2,3,4) for p in
                         (BASE/f'mycar/images/{c}/{s}.png', BASE/f'mycar/cam_{c}/{s}.xml')}
    return dict(frames=frames, available_frames=all_count, complete_scope=len(frames)==all_count,
                frame_signatures=signatures, model_sha256=EXPECTED_MODEL, code_sha256=hashes,
                calibration_sha256={p:sha(ROOT/p) for p in ('Intrinsics_cam.json','T_imu_cam(1).json')},
                params=params, development_frames=sorted(dev), known_regression_frames=sorted(known),
                environment={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','opencv-python')},
                python=platform.python_version(), threads=threads, cv_threads=2,
                shard_index=shard_index, shard_count=shard_count,
                protocol='Cam234 union GT; h>=25px; IoU>=.5; one-to-one; same ego gate; no fitting',
                limitations='All frames are same-route descriptive evaluation, not independent generalization. '
                            'BAND includes per-pass vehicle NMS; 960 and BAND must be compared separately. '
                            'Full HR/MILP/embedding/temporal/redetection are NOT part of this full-frame run.',
                new_ideas=dict(reliable_dedup='>=2 shared views or exact observation; stereo relative sigma<=.15; IoS .75',
                               rescue='legacy >=2 views, conf>=.6, no compatible overlap>.1',
                               adaptive='band only if 960 has <80px or conf<.45 candidate centred in y850..1250'),
                single_thresholds={'960':{str(k):v for k,v in SINGLE_THRESHOLDS.items()},
                                   'BAND':{'2':.25,'3':.2,'4':.25}})


def native_single(ds, cams, c, threshold):
    # Do NOT call run_pipeline(cid) before choosing a new threshold: that would
    # impose the old threshold as an invisible lower bound on the search.
    kept, _ = filter_ego([d for d in ds if d['src']==c], cams)
    return [dict(dets=[d], src=[c], cls=d['cls'], conf=d['conf']) for d in kept if d['conf']>=threshold]


def candidate(ds, cams, sd, maps, cfg):
    t = time.perf_counter()
    os, _ = F.candidates(copy.deepcopy(ds), cams, cfg, sd, maps)
    for o in os:
        o['x'] = F.object_features(o, cfg)
    # Preserve depth provenance omitted from old slim caches.
    out = slim(os)
    for a,b in zip(out, os):
        for d,e in zip(a['dets'], b['dets']):
            d['depth_src'] = e['depth_src']
    return out, (time.perf_counter()-t)*1000


def summary(rows, cfg, elapsed):
    frames = cfg['frames']
    if len(rows) != len(set(r['frame'] for r in rows)):
        raise ValueError('Duplicate frame rows')
    groups = {'all_frames':rows,
              'not_used_for_fitting':[r for r in rows if r['frame'] not in cfg['development_frames']],
              'late_sequence_ge1000':[r for r in rows if int(r['frame'])>=1000],
              'previously_reported240':[r for r in rows if r['frame'] in cfg['known_regression_frames']],
              'additional_late_frames':[r for r in rows if int(r['frame'])>=1000 and r['frame'] not in cfg['known_regression_frames']]}
    # Ordered blocks expose scene-dependent failures instead of just a mean.
    groups.update({f'ordered_block_{i//200:02d}':rows[i:i+200] for i in range(0,len(rows),200)})
    result = dict(complete=cfg['complete_scope'] and len(rows)==len(frames), completed_frames=len(rows),
                  planned_frames=len(frames), available_frames=cfg['available_frames'],
                  current_invocation_s=elapsed, manifest_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                  protocol=cfg['protocol'], limitations=cfg['limitations'], splits={})
    for name, rs in groups.items():
        if not rs:
            continue
        methods = {m:metrics(np.sum([r['counts'][m] for r in rs],axis=0)) for m in rs[0]['counts']}
        result['splits'][name] = dict(frames=len(rs), methods=methods,
            native_singles={m:metrics(np.sum([r['native_counts'][m] for r in rs],axis=0)) for m in rs[0]['native_counts']},
            mean_cost_ms={m:float(np.mean([r['cost_ms'][m] for r in rs])) for m in rs[0]['cost_ms']},
            mean_adaptive_cameras=float(np.mean([r['adaptive_cameras'] for r in rs])))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=BASE/'experiments/global_fusion_20260930')
    ap.add_argument('--limit',type=int,default=0,help='Smoke scope only; never labelled full')
    ap.add_argument('--shard-index',type=int,default=0)
    ap.add_argument('--shard-count',type=int,default=1)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.threads < 1:
        ap.error('Invalid shard or thread settings')
    import torch
    torch.set_num_threads(args.threads);cv2.setNumThreads(2)
    from ultralytics import YOLO
    started=time.perf_counter();cfg=manifest(args.limit,args.shard_index,args.shard_count,args.threads);out=args.out;out.mkdir(parents=True,exist_ok=True)
    mp=out/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text(encoding='utf-8')) != cfg:
            raise ValueError('Source/data/config/environment changed; use a NEW output directory')
    else:
        dump(mp,cfg)
    cams=load_cameras();rect=core.TriRectifier(core.camera_params);sd=StereoDepth(cams)
    model=None;rowdir=out/'frames';rowdir.mkdir(exist_ok=True)
    rows=[]
    for i,stem in enumerate(cfg['frames']):
        rp=rowdir/f'{stem}.json'
        if rp.exists():
            rows.append(json.loads(rp.read_text(encoding='utf-8')))
            continue
        t0=time.perf_counter();images={c:cv2.imread(str(BASE/f'mycar/images/{c}/{stem}.png')) for c in (2,3,4)}
        if any(im is None for im in images.values()):
            raise ValueError(f'Image decode failed: {stem}')
        cost={'decode':(time.perf_counter()-t0)*1000};ds={};cam_times={};used=[]
        for detset in ('960','BAND'):
            ds[detset]=[];cam_times[detset]={}
            for c,im in images.items():
                cp=out/'detections'/detset/f'{c}_{stem}.json'
                if cp.exists():
                    entry=json.loads(cp.read_text(encoding='utf-8'))
                else:
                    if model is None:
                        model=YOLO(str(BASE/'yolo11n.pt'))
                    t=time.perf_counter()
                    dd=detect(model,im,960,tile_imgsz=1600,band=(850,1250) if detset=='BAND' else None)
                    for d in dd:
                        d.update(src=c,feature=feature(im,d['box']).tolist())
                    entry=dict(dets=dd,inference_ms=(time.perf_counter()-t)*1000)
                    dump(cp,entry)
                ds[detset]+=entry['dets'];cam_times[detset][c]=entry['inference_ms']
            cost['detect_'+detset]=sum(cam_times[detset].values())
        for c in (2,3,4):
            if needs_band([d for d in ds['960'] if d['src']==c]):
                used.append(c)
        ds['ADAPTIVE']=[d for c in (2,3,4) for d in ds['BAND' if c in used else '960'] if d['src']==c]
        # Conservative acquisition cost: full 960, then a complete BAND pass
        # for chosen cameras (counts its repeated 960 forward too).
        cost['detect_ADAPTIVE']=cost['detect_960']+sum(cam_times['BAND'][c] for c in used)
        t=time.perf_counter();maps=sd.compute_frame(images);cost['stereo']=(time.perf_counter()-t)*1000
        gt=truth(stem,(2,3,4));variants={};native={}
        for detset in ('960','BAND'):
            for c in (2,3,4):
                name=f'{detset}/cam{c}'
                variants[name]=native_single(ds[detset],cams,c,cfg['single_thresholds'][detset][str(c)])
                native[name]=evaluate(variants[name],{c:gt[c]}).tolist()
            t=time.perf_counter()
            variants[f'{detset}/old_fusion']=run_pipeline(ds[detset],cams,rect=rect)[0]
            cost[f'{detset}/old_fusion']=(time.perf_counter()-t)*1000
            for kind in ('base','d'):
                pars=cfg['params'][f'{detset}_{kind}'];switches=pars['switches']
                cand, ms=candidate(ds[detset],cams,sd,maps,switches)
                t=time.perf_counter();variants[f'{detset}/{kind}']=admit_slim(cand,switches,pars)
                cost[f'{detset}/{kind}']=ms+(time.perf_counter()-t)*1000
                if detset=='BAND' and kind=='d':
                    p=cfg['params']['BAND_d_E1'];t=time.perf_counter()
                    variants['BAND/d_E1']=admit_slim(cand,p['switches'],p)
                    cost['BAND/d_E1']=ms+(time.perf_counter()-t)*1000
                    # Ablate only the E1/residual decision, keep its frozen score.
                    p2=dict(p,ios=9.,residual_iou=1.01)
                    t=time.perf_counter()
                    variants['BAND/reliable_dedup']=conservative_dedup(admit_slim(cand,p['switches'],p2))
                    cost['BAND/reliable_dedup']=ms+(time.perf_counter()-t)*1000
                    t=time.perf_counter()
                    variants['BAND/rescue']=rescue_unexplained(variants['BAND/d_E1'],variants['BAND/old_fusion'])
                    cost['BAND/rescue']=cost['BAND/d_E1']+cost['BAND/old_fusion']+(time.perf_counter()-t)*1000
            if detset=='BAND':
                p=cfg['params']['BAND_dfo_E1'];cand,ms=candidate(ds[detset],cams,sd,maps,p['switches'])
                t=time.perf_counter();variants['BAND/dfo_E1']=admit_slim(cand,p['switches'],p)
                cost['BAND/dfo_E1']=ms+(time.perf_counter()-t)*1000
        t=time.perf_counter();variants['ADAPTIVE/old_fusion']=run_pipeline(ds['ADAPTIVE'],cams,rect=rect)[0]
        cost['ADAPTIVE/old_fusion']=(time.perf_counter()-t)*1000
        # Raw detector recall ceiling after the SAME ego filter, not oracle decisions.
        for detset in ('960','BAND'):
            kept,_=filter_ego(ds[detset],cams)
            variants[f'{detset}/raw_union']=[dict(dets=[d],src=[d['src']]) for d in kept]
        counts={name:evaluate(objects,gt).tolist() for name,objects in variants.items()}
        row=dict(frame=stem,counts=counts,native_counts=native,cost_ms=cost,adaptive_cameras=len(used),
                 observed_wall_s=time.perf_counter()-t0,
                 objects={name:[dict(src=o['src'],dets=[{k:d[k] for k in ('src','box','cls','conf')} for d in o['dets']])
                                for o in objects] for name,objects in variants.items() if name in
                                ('BAND/d_E1','BAND/reliable_dedup','BAND/rescue','960/old_fusion','BAND/old_fusion')})
        dump(rp,row);rows.append(row)
        if (i+1)%25==0 or i==0 or i+1==len(cfg['frames']):
            elapsed=time.perf_counter()-started
            dump(out/'progress.json',dict(done=len(rows),planned=len(cfg['frames']),last_frame=stem,elapsed_s=elapsed))
            print(f'{i+1}/{len(cfg["frames"])} {stem} frame={row["observed_wall_s"]:.2f}s elapsed={elapsed:.1f}s',flush=True)
    result=summary(rows,cfg,time.perf_counter()-started)
    dump(out/'results.json',result)
    print(json.dumps(result['splits']['all_frames']['methods'],indent=2),flush=True)
    print(f'COMPLETE={result["complete"]} {out}',flush=True)


if __name__=='__main__':
    main()
