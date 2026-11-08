"""Fresh YOLO11n@960 five-camera cache at conf=.01, never reads GT.

Threshold sweeps filter this common post-NMS pool. We do not change image size,
NMS IoU, max_det, classes, model weights, or use local enhancement/BAND.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'ultralytics'
sys.path.insert(0, str(BASE/'core_algorithms'))
from multiview_fusion import feature

EXPECTED = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def run(args):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if sha(BASE/'yolo11n.pt') != EXPECTED:
        raise ValueError('Wrong weights')
    sets = [set(p.stem for p in (BASE/f'mycar/images/{c}').glob('*.png')) for c in (1, 2, 3, 4, 5)]
    names = sorted(set.intersection(*sets))
    if any(s != set(names) for s in sets):
        raise ValueError('Five-view image coverage differs')
    if args.limit:
        names = names[:args.limit]
    frames = names[args.shard_index::args.shard_count]
    protocol = dict(model_sha256=EXPECTED, imgsz=960, conf=.01, iou=.7, max_det=300,
        target_classes=[0, 1, 2, 3, 5, 7], class_filter='AFTER native class-aware NMS, like historical producer',
        device='cpu', no_band=True, no_retraining=True, torch_threads=args.threads,
        source_sha256=sha(__file__), feature_source_sha256=sha(BASE/'core_algorithms/multiview_fusion.py'))
    manifest = dict(protocol=protocol, domain=names, frames=frames, shard_index=args.shard_index,
                    shard_count=args.shard_count, limit=args.limit)
    mp = args.out/f'shard{args.shard_index}/manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Source/config changed: use a NEW cache directory')
    dump(mp, manifest)
    model = None
    started = time.perf_counter()
    inferred = 0
    for n, stem in enumerate(frames, 1):
        for cid in (2, 3, 4, 1, 5):
            ip = BASE/f'mycar/images/{cid}/{stem}.png'
            st = ip.stat()
            signature = [st.st_size, st.st_mtime_ns]
            cp = args.out/'detections'/f'{cid}_{stem}.json'
            if cp.exists():
                old = json.loads(cp.read_text())
                if old['protocol'] != protocol or old['image_signature'] != signature:
                    raise ValueError('Changed cache/input')
                continue
            binary = ip.read_bytes()
            image = cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (2056, 2464):
                raise ValueError('Image decode/shape')
            if model is None:
                model = YOLO(str(BASE/'yolo11n.pt'))
            t = time.perf_counter()
            result = model(image, conf=.01, imgsz=960, iou=.7, max_det=300, device='cpu', verbose=False)[0]
            elapsed = time.perf_counter()-t
            data = result.boxes.data.cpu().numpy()
            ds = []
            for x1, y1, x2, y2, confidence, cls in data:
                if int(cls) not in protocol['target_classes']:
                    continue
                box = [float(x1), float(y1), float(x2), float(y2)]
                ds.append(dict(src=cid, cls=int(cls), conf=float(confidence), box=box,
                               feature=feature(image, box).tolist()))
            dump(cp, dict(frame=stem, camera=cid, protocol=protocol,
                image_signature=signature, image_sha256=hashlib.sha256(binary).hexdigest(),
                inference_s=elapsed, all_class_native_count=len(data), max_det_cap_hit=len(data)>=300, dets=ds))
            inferred += 1
        if n == 1 or n % 25 == 0 or n == len(frames):
            progress = dict(done=n, planned=len(frames), inferred_images_this_run=inferred,
                last_frame=stem, elapsed_s=time.perf_counter()-started)
            dump(mp.parent/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    dump(mp.parent/'completion.json', dict(done=len(frames), planned=len(frames)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=BASE/'experiments/low_conf_no_band_20261009/cache')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--shard-count', type=int, default=4)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.threads < 1 or args.limit < 0:
        ap.error('Invalid execution scope')
    run(args)
