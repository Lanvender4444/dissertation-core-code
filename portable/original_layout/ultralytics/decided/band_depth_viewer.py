"""Cam234 BAND/depth/E1 viewer, using the frozen full-sequence method.

Annotations are read only AFTER inference for terminal statistics by default;
no annotation or fitting is used to generate predictions. Rendering and keyboard
semantics are shared with final_viewer; the legacy viewer remains available.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core_algorithms'))
import depth_bev_fusion as F
import final_viewer as V
from final_pipeline import filter_ego
from multiview_fusion import feature
from multiview_geometry import ROOT, load_cameras
from run_depth_bev import admit_slim, slim
from stereo_depth import StereoDepth
from yolo_hr_detect import detect
from viewer_evaluation import PROTOCOL, RunEvaluation

BASE = ROOT / 'ultralytics'
METHOD = 'BAND/d_E1'
PARAMS = BASE / 'algorithm/depth_bev_params/BAND_d_E1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fuse(dets, cameras, stereo, images, params, depth_mode='dense'):
    """Same candidates/features/admission as the full-sequence experiment."""
    if depth_mode not in ('dense', 'roi64'):
        raise ValueError('Unsupported depth mode')
    maps = stereo.compute_frame(images, dets) if depth_mode == 'roi64' else stereo.compute_frame(images)
    candidates, _ = F.candidates(copy.deepcopy(dets), cameras, params['switches'], stereo, maps)
    for o in candidates:
        o['x'] = F.object_features(o, params['switches'])
    compact = slim(candidates)
    for a, b in zip(compact, candidates):
        for d, original in zip(a['dets'], b['dets']):
            d['depth_src'] = original['depth_src']
    objects = admit_slim(compact, params['switches'], params)
    for i, o in enumerate(objects, 1):
        o['id'] = i
        o['label'] = 'FULL' if len(o['src']) == 3 else 'PART' if len(o['src']) == 2 else 'UNVERIFIED'
        o['unmatched_cameras'] = [c for c in (2, 3, 4) if c not in o['src']]
    _, removed = filter_ego(dets, cameras)
    return objects, removed


class FrozenBandCache:
    """Only reuse signed BAND detections from the full-sequence experiment."""
    def __init__(self, parent, model_hash, params, names, dirs):
        self.frames = {}
        for manifest in sorted(Path(parent).glob('shard*/manifest.json')):
            cfg = json.loads(manifest.read_text(encoding='utf-8'))
            if cfg['model_sha256'] != model_hash or cfg['params']['BAND_d_E1'] != params:
                raise ValueError(f'Cache model/parameters differ: {manifest}')
            for name, digest in cfg['calibration_sha256'].items():
                if sha(ROOT / name) != digest:
                    raise ValueError(f'Cache calibration changed: {name}')
            # Only the detection producer is needed to validate detection reuse.
            for name in ('yolo_hr_detect.py', 'multiview_fusion.py'):
                if sha(BASE / 'core_algorithms' / name) != cfg['code_sha256'][name]:
                    raise ValueError(f'Cache detection source changed: {name}')
            for stem in cfg['frames']:
                if stem in self.frames:
                    raise ValueError(f'Duplicate cache frame: {stem}')
                signatures = {k.replace('\\', '/'): v for k, v in cfg['frame_signatures'][stem].items()}
                self.frames[stem] = (manifest.parent, signatures)
        for name in names:
            stem = Path(name).stem
            if stem not in self.frames:
                raise ValueError(f'Missing signed BAND cache: {stem}. Remove --cache to infer.')
            shard, signatures = self.frames[stem]
            for c in (2, 3, 4):
                image = dirs[c] / name
                key = f'ultralytics/mycar/images/{c}/{name}'
                st = image.stat()
                if signatures.get(key) != [st.st_size, st.st_mtime_ns]:
                    raise ValueError(f'Cache image signature differs: {image}')
                if not (shard / 'detections/BAND' / f'{c}_{stem}.json').exists():
                    raise ValueError(f'Missing BAND detections: Cam{c}/{stem}')

    def load(self, stem):
        shard, _ = self.frames[stem]
        return [d for c in (2, 3, 4) for d in json.loads(
            (shard / 'detections/BAND' / f'{c}_{stem}.json').read_text(encoding='utf-8'))['dets']]


def serialize(objects):
    return [dict(id=o['id'], label=o['label'], cls=o['cls'], conf=o['conf'],
                 supported_cameras=o['src'], unmatched_cameras=o['unmatched_cameras'],
                 members=[dict(src=d['src'], box=list(d['box']), cls=d['cls'], conf=d['conf'],
                               depth_source=d['depth_src'], depth_m=d['Z'], depth_sigma_m=d['sigZ'])
                          for d in o['dets']]) for o in objects]


def main():
    ap = argparse.ArgumentParser(description='Cam234: BAND + depth + E1; fused panorama and three original panels')
    ap.add_argument('--seq', type=Path, default=BASE / 'mycar/images')
    ap.add_argument('--gt-root', type=Path, help='XML root containing cam_2/cam_3/cam_4; defaults to mycar for the default sequence')
    ap.add_argument('--no-eval', action='store_true', help='Display/inference only, without reading GT or printing P/R/F1')
    ap.add_argument('--model', type=Path, default=BASE / 'yolo11n.pt')
    ap.add_argument('--mode', choices=['step', 'auto'], default='step')
    ap.add_argument('--start', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--depth-mode', choices=['dense', 'roi64'], default='dense',
                    help='dense: frozen reference; roi64: experimental object-row stereo, not pixel-equivalent')
    ap.add_argument('--fps', type=float, default=5)
    ap.add_argument('--max-width', type=int, default=1440)
    ap.add_argument('--max-height', type=int, default=900)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--cache', action='store_true', help='Reuse signed full-experiment BAND detections; depth still computed')
    ap.add_argument('--cache-root', type=Path, default=BASE / 'experiments/global_fusion_20260930_shards')
    ap.add_argument('--out', type=Path, default=BASE / 'experiments/final_runs')
    ap.add_argument('--legacy', action='store_true', help='Use the old 960 viewer instead (handled by final_three_cam.py)')
    args = ap.parse_args()
    if args.legacy:
        ap.error('Use --legacy with decided/final_three_cam.py')
    if args.limit < 0 or args.stride < 1 or args.threads < 1 or args.fps <= 0 or min(args.max_width, args.max_height) < 320:
        ap.error('Invalid limit/stride/threads/fps/window size')
    if args.headless and args.mode == 'step':
        ap.error('Step mode needs GUI; use --mode auto --headless for batch runs')
    cameras = {c: cam for c, cam in load_cameras().items() if c in (2, 3, 4)}
    dirs = {c: args.seq / str(c) for c in cameras}
    if len(cameras) != 3 or any(not p.is_dir() for p in dirs.values()):
        ap.error('Missing Cam2/3/4 calibration or image directory')
    sets = [{p.name for p in directory.glob('*.png')} for directory in dirs.values()]
    if set.union(*sets) != set.intersection(*sets):
        ap.error('Cam2/3/4 frame coverage differs; resolve missing frames first')
    names = sorted(sets[0])
    if args.start and f'{args.start}.png' not in names:
        ap.error('Start frame is not available in all cameras')
    names = [n for n in names if Path(n).stem >= args.start][::args.stride]
    if args.limit:
        names = names[:args.limit]
    if not names:
        ap.error('No synchronized PNG frames')
    gt_root = args.gt_root
    if not args.no_eval:
        if gt_root is None:
            if args.seq.resolve() != (BASE / 'mycar/images').resolve():
                ap.error('A custom --seq needs its corresponding --gt-root, or --no-eval; never reuse mycar GT silently')
            gt_root = BASE / 'mycar'
        try:
            evaluation_method = METHOD + ('/roi64' if args.depth_mode == 'roi64' else '')
            evaluator = RunEvaluation(gt_root, names, cameras, evaluation_method)
        except FileNotFoundError as exc:
            ap.error(str(exc))
    else:
        evaluator = None
    model_hash = sha(args.model)
    expected = json.loads((BASE / 'algorithm/dedup_config.json').read_text())['model_sha256']
    if model_hash != expected:
        ap.error('Model differs from validated YOLO11n weights')
    params = json.loads(PARAMS.read_text(encoding='utf-8'))
    if params['config'] != 'd_E1' or params['detset'] != 'BAND':
        ap.error('Expected the frozen BAND/d_E1 configuration')
    import torch
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(2)
    cache = FrozenBandCache(args.cache_root, model_hash, params, names, dirs) if args.cache else None
    detector = None
    if cache is None:
        from ultralytics import YOLO
        detector = YOLO(str(args.model))
    method = METHOD
    research_metadata = {}
    if args.depth_mode == 'roi64':
        sys.path.insert(0, str(BASE / 'research'))
        from efficient_fusion import RowRoiStereo
        stereo = RowRoiStereo(cameras, margin=64)
        method += '/roi64'
        research_metadata = dict(roi_margin_half_resolution_rows=64,
                                 research_sha256=sha(BASE / 'research/efficient_fusion.py'))
    else:
        stereo = StereoDepth(cameras)
    panorama = V.FrontPanorama(cameras, width=1040)
    suffix = '_roi64' if args.depth_mode == 'roi64' else ''
    output = args.out / f'three_cam_band_depth_e1{suffix}_{time.time_ns()}'
    output.mkdir(parents=True)
    metadata = dict(method=method, depth_mode=args.depth_mode, **research_metadata,
                    cameras=[2, 3, 4], model_sha256=model_hash, params=params,
                    params_sha256=sha(PARAMS), calibration_sha256={n: sha(ROOT / n) for n in ('Intrinsics_cam.json', 'T_imu_cam(1).json')},
                    source=str(args.seq.resolve()), cache=cache is not None, threads=args.threads,
                    ego_filter=True, imgsz=960, band=[850, 1250], tile_imgsz=1600, mode=args.mode,
                    evaluation_enabled=evaluator is not None,
                    gt_root=str(gt_root.resolve()) if evaluator is not None else None,
                    evaluation_protocol=PROTOCOL if evaluator is not None else None,
                    evaluation_source_sha256=sha(Path(__file__).with_name('viewer_evaluation.py')),
                    note='Frame-local IDs. FULL/PART is support count; no-match is unassociated, not a proven detector miss. '
                         'Panorama uses rotation only; inference uses original-camera coordinates.')
    (output / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    index = 0
    auto = args.mode == 'auto'
    visited = set()
    window = f'Cam234 | BAND + DEPTH + E1 | {args.depth_mode}'
    status = 'stopped'
    print(f'[EVAL] {PROTOCOL}' if evaluator is not None else '[EVAL] Disabled (--no-eval)', flush=True)
    try:
        with (output / 'objects.jsonl').open('w', encoding='utf-8') as log:
            while 0 <= index < len(names):
                name = names[index]
                started = time.perf_counter()
                images = {c: cv2.imread(str(dirs[c] / name)) for c in cameras}
                for c, image in images.items():
                    if image is None or image.shape[:2] != (cameras[c].height, cameras[c].width):
                        raise ValueError(f'Image size/decode mismatch: Cam{c}/{name}')
                if cache is not None:
                    dets = cache.load(Path(name).stem)
                else:
                    dets = []
                    for c, image in images.items():
                        ds = detect(detector, image, 960, tile_imgsz=1600, band=(850, 1250))
                        for d in ds:
                            d.update(src=c, feature=feature(image, d['box']).tolist())
                        dets.extend(ds)
                objects, removed = fuse(dets, cameras, stereo, images, params, depth_mode=args.depth_mode)
                processing_s = time.perf_counter() - started
                revisit = index in visited
                evaluation = evaluator.evaluate_frame(name, dets, objects) if evaluator is not None else None
                if index not in visited:
                    log.write(json.dumps(dict(frame=name, method=method, objects=serialize(objects),
                                              ego_removed=[{k: d[k] for k in ('src', 'box', 'cls', 'conf')} for d in removed],
                                              processing_s=processing_s, evaluation=evaluation)) + '\n')
                    log.flush()
                    visited.add(index)
                mode = 'auto' if auto else 'step'
                title = f'BAND + DEPTH + E1 [{args.depth_mode}] | {name}'
                canvas = V.render(images, objects, removed, panorama, title, mode)
                factor = min(1, args.max_width / canvas.shape[1], args.max_height / canvas.shape[0])
                canvas = cv2.resize(canvas, None, fx=factor, fy=factor)
                if args.save:
                    if not cv2.imwrite(str(output / name), canvas):
                        raise OSError(f'Could not save visualization: {output / name}')
                print(f'{index+1}/{len(names)} {name} {method} objects={len(objects)} ego={len(removed)}', flush=True)
                if evaluator is not None:
                    evaluator.print_progress(evaluation, revisit)
                if args.headless:
                    index += 1
                    continue
                advance = False
                while not advance:
                    cv2.imshow(window, canvas)
                    key = cv2.waitKeyEx(max(1, round(1000 / args.fps)) if auto else 50)
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        return
                    action = V.command(key)
                    if action == 'quit':
                        return
                    if action == 'toggle':
                        auto = not auto
                        canvas = V.render(images, objects, removed, panorama, title, 'auto' if auto else 'step')
                        canvas = cv2.resize(canvas, None, fx=factor, fy=factor)
                    elif action == 'previous':
                        index = max(0, index-1)
                        advance = True
                    elif action == 'next' or (auto and key == -1):
                        index += 1
                        advance = True
            status = 'completed'
    except KeyboardInterrupt:
        status = 'interrupted'
        raise
    except Exception:
        status = 'error'
        raise
    finally:
        cv2.destroyAllWindows()
        if evaluator is not None:
            evaluator.finish(output, status)
        print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()
