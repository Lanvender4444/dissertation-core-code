"""No-BAND four-panel viewer; same prediction variants as validate_no_band.py.

Reuse the existing fusion, rendering and post-inference GT evaluator. No core,
weights, thresholds, calibration, or labels are changed or fitted by this viewer.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core_algorithms'))
import final_viewer as V
from band_depth_viewer import BASE, ROOT, fuse, serialize, sha
from final_pipeline import SINGLE_THRESHOLDS
from multiview_fusion import feature
from multiview_geometry import load_cameras
from stereo_depth import StereoDepth
from viewer_evaluation import PROTOCOL, RunEvaluation
from yolo_hr_detect import detect

PARAM_FILES = ('BAND_d_E1.json', '960_d.json')
METHODS = {'frozen': '960/current_frozen', '960': '960/d_E1_existing_score'}


def profile_params(profile):
    """Strict input ablation vs existing 960 score; never rewrite params files."""
    if profile not in METHODS:
        raise ValueError('Unsupported score profile')
    band, plain = [json.loads((BASE / 'algorithm/depth_bev_params' / name).read_text(encoding='utf-8'))
                   for name in PARAM_FILES]
    if band['config'] != 'd_E1' or band['detset'] != 'BAND' or plain['config'] != 'd' or plain['detset'] != '960':
        raise ValueError('Expected frozen BAND/d_E1 and 960/d configurations')
    candidate_keys = ('depth', 'bev', 'bevf', 'occl', 'redetect', 'milp', 'emb', 'track')
    if band['features'] != plain['features'] or any(band['switches'][k] != plain['switches'][k] for k in candidate_keys):
        raise ValueError('Feature/candidate protocol differs between profiles')
    return copy.deepcopy(band) if profile == 'frozen' else dict(copy.deepcopy(plain), ios=band['ios'])


class Frozen960Cache:
    """Signed full-experiment 960 detections; never load BAND or stale inputs."""
    def __init__(self, parent, model_hash, names, dirs):
        self.frames = {}
        paths = sorted(Path(parent).glob('shard*/manifest.json'))
        if not paths:
            raise ValueError('No signed 960 manifests; remove --cache to infer')
        for manifest in paths:
            cfg = json.loads(manifest.read_text(encoding='utf-8'))
            if cfg['model_sha256'] != model_hash:
                raise ValueError(f'Cache model differs: {manifest}')
            if cfg['single_thresholds']['960'] != {str(c): v for c, v in SINGLE_THRESHOLDS.items()}:
                raise ValueError(f'Cache 960 threshold protocol differs: {manifest}')
            for name, digest in cfg['calibration_sha256'].items():
                if sha(ROOT / name) != digest:
                    raise ValueError(f'Cache calibration changed: {name}')
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
                raise ValueError(f'Missing signed 960 cache: {stem}. Remove --cache to infer.')
            shard, signatures = self.frames[stem]
            for c in (2, 3, 4):
                image = dirs[c] / name
                st = image.stat()
                if signatures.get(f'ultralytics/mycar/images/{c}/{name}') != [st.st_size, st.st_mtime_ns]:
                    raise ValueError(f'Cache image signature differs: {image}')
                if not (shard / 'detections/960' / f'{c}_{stem}.json').is_file():
                    raise ValueError(f'Missing 960 detections: Cam{c}/{stem}')

    def load(self, stem):
        shard, _ = self.frames[stem]
        return [d for c in (2, 3, 4) for d in json.loads(
            (shard / 'detections/960' / f'{c}_{stem}.json').read_text(encoding='utf-8'))['dets']]


def infer_detections(detector, images):
    dets = []
    for c, image in images.items():
        # Exactly the frozen 960 producer: one full-image pass, no tile NMS/BAND.
        ds = detect(detector, image, 960)
        for d in ds:
            d.update(src=c, feature=feature(image, d['box']).tolist())
        dets.extend(ds)
    return dets


def main():
    ap = argparse.ArgumentParser(description='Cam234: NO BAND + dense depth + E1; four panels and live GT statistics')
    ap.add_argument('--score-profile', choices=list(METHODS), default='frozen',
                    help='frozen: retain current BAND score; 960: existing 960 score + same E1, neither refitted')
    ap.add_argument('--seq', type=Path, default=BASE / 'mycar/images')
    ap.add_argument('--gt-root', type=Path)
    ap.add_argument('--no-eval', action='store_true')
    ap.add_argument('--model', type=Path, default=BASE / 'yolo11n.pt')
    ap.add_argument('--mode', choices=['step', 'auto'], default='step')
    ap.add_argument('--start', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--depth-mode', choices=['dense'], default='dense', help='Frozen dense reference; no new ROI ablation')
    ap.add_argument('--fps', type=float, default=5)
    ap.add_argument('--max-width', type=int, default=1440)
    ap.add_argument('--max-height', type=int, default=900)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--cache', action='store_true', help='Read signed 960 detections; dense depth still recomputed')
    ap.add_argument('--cache-root', type=Path, default=BASE / 'experiments/global_fusion_20260930_shards')
    ap.add_argument('--out', type=Path, default=BASE / 'experiments/final_runs')
    args = ap.parse_args()
    if args.limit < 0 or args.stride < 1 or args.threads < 1 or args.fps <= 0 or min(args.max_width, args.max_height) < 320:
        ap.error('Invalid limit/stride/threads/fps/window size')
    if args.headless and args.mode == 'step':
        ap.error('Step mode needs GUI; use --mode auto --headless')
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
    method = METHODS[args.score_profile]
    gt_root = args.gt_root
    if not args.no_eval:
        if gt_root is None:
            if args.seq.resolve() != (BASE / 'mycar/images').resolve():
                ap.error('A custom --seq needs its corresponding --gt-root, or --no-eval')
            gt_root = BASE / 'mycar'
        try:
            evaluator = RunEvaluation(gt_root, names, cameras, method, detset='960')
        except FileNotFoundError as exc:
            ap.error(str(exc))
    else:
        evaluator = None
    model_hash = sha(args.model)
    expected = json.loads((BASE / 'algorithm/dedup_config.json').read_text())['model_sha256']
    if model_hash != expected:
        ap.error('Model differs from validated YOLO11n weights')
    params = profile_params(args.score_profile)
    import torch
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(2)
    cache = Frozen960Cache(args.cache_root, model_hash, names, dirs) if args.cache else None
    detector = None
    if cache is None:
        from ultralytics import YOLO
        detector = YOLO(str(args.model))
    stereo = StereoDepth(cameras)
    panorama = V.FrontPanorama(cameras, width=1040)
    output = args.out / f'three_cam_no_band_{args.score_profile}_depth_e1_{time.time_ns()}'
    output.mkdir(parents=True)
    metadata = dict(method=method, input_detset='960', score_profile=args.score_profile,
                    depth_mode='dense', cameras=[2, 3, 4], model_sha256=model_hash, params=params,
                    parameter_files_sha256={n: sha(BASE / 'algorithm/depth_bev_params' / n) for n in PARAM_FILES},
                    calibration_sha256={n: sha(ROOT / n) for n in ('Intrinsics_cam.json', 'T_imu_cam(1).json')},
                    source=str(args.seq.resolve()), cache=cache is not None, threads=args.threads,
                    ego_filter=True, imgsz=960, band=None, tile_imgsz=None, mode=args.mode,
                    single_thresholds=SINGLE_THRESHOLDS, evaluation_enabled=evaluator is not None,
                    gt_root=str(gt_root.resolve()) if evaluator is not None else None,
                    evaluation_protocol=PROTOCOL if evaluator is not None else None,
                    source_sha256={n: sha(Path(__file__).with_name(n)) for n in
                                   ('no_band_depth_viewer.py', 'band_depth_viewer.py', 'viewer_evaluation.py')},
                    note='NO BAND/tile passes. Frame-local IDs; FULL/PART means support, not correctness. '
                         'No-match is not proven missed detection; panorama is rotation-only.')
    (output / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    index, auto, visited, status = 0, args.mode == 'auto', set(), 'stopped'
    window = f'Cam234 | NO BAND + DEPTH + E1 | score={args.score_profile}'
    print(f'[METHOD] {method} | full image imgsz=960 | NO BAND | dense depth | E1', flush=True)
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
                dets = cache.load(Path(name).stem) if cache is not None else infer_detections(detector, images)
                objects, removed = fuse(dets, cameras, stereo, images, params)
                processing_s = time.perf_counter() - started
                revisit = index in visited
                evaluation = evaluator.evaluate_frame(name, dets, objects) if evaluator is not None else None
                if not revisit:
                    log.write(json.dumps(dict(frame=name, method=method, objects=serialize(objects),
                        ego_removed=[{k: d[k] for k in ('src', 'box', 'cls', 'conf')} for d in removed],
                        processing_s=processing_s, evaluation=evaluation)) + '\n')
                    log.flush()
                    visited.add(index)
                title = f'NO BAND + DEPTH + E1 [score={args.score_profile}] | {name}'
                canvas = V.render(images, objects, removed, panorama, title, 'auto' if auto else 'step')
                factor = min(1, args.max_width / canvas.shape[1], args.max_height / canvas.shape[0])
                canvas = cv2.resize(canvas, None, fx=factor, fy=factor)
                if args.save and not cv2.imwrite(str(output / name), canvas):
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
                        index = max(0, index - 1)
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
