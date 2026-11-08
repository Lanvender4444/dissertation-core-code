"""High-resolution / tiled YOLO detection cache for the YOLO-aware fusion (v2).

Why: YOLO11n letterboxes the 2464x2056 frame to imgsz. At imgsz=960 the image
is shrunk ~2.6x, so a 40 px pedestrian becomes ~15 px, close to the stride-8
P3 grid cell; such objects are mostly never proposed. The fusion stage cannot
recover what the detector never emits, so v2 re-runs the SAME weights
(no retraining) at native-ish resolution and, optionally, on overlapping tiles
(SAHI-style slicing), then merges with class-agnostic-for-vehicles NMS.

Cache format is identical to experiments/detection_cache/<hash>_<size>_conf010
(src, box, conf, cls, feature) plus 'mode' metadata, so existing evaluation
code can read it.  GT is never read here.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from multiview_geometry import ROOT
from multiview_fusion import feature

CLASSES = (0, 1, 2, 3, 5, 7)
VEHICLES = {2, 5, 7}


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(np.clip(x[:, 2:] - x[:, :2], 0, None), axis=1)
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def ios_matrix(a, b):
    """Intersection over the SMALLER box: catches a tile-cut fragment inside a full box."""
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(np.clip(x[:, 2:] - x[:, :2], 0, None), axis=1)
    return inter / np.maximum(np.minimum(area(a)[:, None], area(b)[None]), 1e-9)


def group_key(c):
    return 'veh' if c in VEHICLES else c


def merge_nms(dets, iou_thr=0.55, ios_thr=0.8):
    """Greedy NMS across passes/tiles.

    car/bus/truck share one group because YOLO's default class-aware NMS keeps
    a car box and a truck box on the same vehicle.  A lower-scored box is also
    dropped when it lies mostly inside a kept box of the same group (tile edge
    fragments), measured by intersection-over-smaller.
    """
    dets = sorted(dets, key=lambda d: -d['conf']); kept = []
    for d in dets:
        same = [k for k in kept if group_key(k['cls']) == group_key(d['cls'])]
        if same:
            boxes = [k['box'] for k in same]
            if iou_matrix([d['box']], boxes).max() >= iou_thr: continue
            # fragment only if clearly smaller than the kept box
            ios = ios_matrix([d['box']], boxes)[0]
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
            if any(v >= ios_thr and area(d['box']) < .6 * area(k['box']) for v, k in zip(ios, same)): continue
        kept.append(d)
    return kept


def tiles(w, h, nx=2, ny=2, overlap=0.2):
    tw = int(np.ceil(w / (nx - (nx - 1) * overlap))); th = int(np.ceil(h / (ny - (ny - 1) * overlap)))
    xs = np.linspace(0, w - tw, nx).astype(int); ys = np.linspace(0, h - th, ny).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def band_tiles(w, band, n=2, overlap=0.15):
    """E4: tiles covering only the horizon band [y0,y1] where far/small objects live
    (development statistics: 99.8% of GT boxes < 80 px lie in y 880-1200)."""
    y0, y1 = band; tw = int(np.ceil(w / (n - (n - 1) * overlap)))
    xs = np.linspace(0, w - tw, n).astype(int)
    return [(x, y0, x + tw, y1) for x in xs]


def detect(model, image, imgsz, tile_grid=None, tile_imgsz=960, conf=.10, band=None):
    raw = []
    def run(img, size, ox=0, oy=0, edge=None):
        r = model(img, conf=conf, imgsz=size, device='cpu', verbose=False)[0]
        for x1, y1, x2, y2, c, k in r.boxes.data.cpu().numpy():
            if int(k) not in CLASSES: continue
            b = [float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy)]
            if edge is not None:
                # drop boxes touching an INTERNAL tile border: they are truncated
                # fragments; the full-frame pass or the neighbour tile owns them.
                W, H, tx1, ty1, tx2, ty2 = edge; m = 4
                if (x1 < m and tx1 > 0) or (y1 < m and ty1 > 0) or (x2 > (tx2 - tx1) - m and tx2 < W) or (y2 > (ty2 - ty1) - m and ty2 < H):
                    continue
            raw.append(dict(box=b, conf=float(c), cls=int(k)))
    run(image, imgsz)
    H, W = image.shape[:2]
    boxes = tiles(W, H, *tile_grid) if tile_grid else []
    if band: boxes = band_tiles(W, band)
    for (x1, y1, x2, y2) in boxes:
        run(image[y1:y2, x1:x2], tile_imgsz, x1, y1, (W, H, x1, y1, x2, y2))
    return merge_nms(raw) if boxes else raw


def cache_dir(model_path, mode):
    h = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:12]
    return ROOT / 'ultralytics/experiments/detection_cache' / f'{h}_{mode}_conf010'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-json', type=Path, required=True, help='JSON list of frame stems')
    ap.add_argument('--cams', default='2,3,4')
    ap.add_argument('--imgsz', type=int, default=1920)
    ap.add_argument('--tiles', default='', help='e.g. 2x2 for SAHI-style slicing')
    ap.add_argument('--tile-imgsz', type=int, default=960)
    ap.add_argument('--band', default='', help='E4: y0,y1 of the horizon band tiled at --tile-imgsz')
    ap.add_argument('--model', default=str(ROOT / 'ultralytics/yolo11n.pt'))
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    import torch; torch.set_num_threads(a.threads)
    from ultralytics import YOLO
    model = YOLO(a.model)
    grid = tuple(int(v) for v in a.tiles.split('x')) if a.tiles else None
    band = tuple(int(v) for v in a.band.split(',')) if a.band else None
    mode = f'{a.imgsz}' + (f'_t{a.tiles}' + (f'@{a.tile_imgsz}' if a.tile_imgsz != 960 else '') if grid else '')
    if band: mode = f'{a.imgsz}_band{band[0]}-{band[1]}@{a.tile_imgsz}'
    out = cache_dir(a.model, mode); out.mkdir(parents=True, exist_ok=True)
    stems = json.loads(a.frames_json.read_text())
    for n, stem in enumerate(stems):
        for c in map(int, a.cams.split(',')):
            ip = ROOT / f'ultralytics/mycar/images/{c}/{stem}.png'; cp = out / f'{c}_{stem}.json'
            st = ip.stat(); sig = [st.st_size, st.st_mtime_ns]
            try:
                if cp.exists() and json.loads(cp.read_text())['image_signature'] == sig: continue
            except (ValueError, KeyError):
                pass  # truncated cache file from an interrupted run: recompute
            img = cv2.imread(str(ip)); t = time.perf_counter()
            ds = detect(model, img, a.imgsz, grid, a.tile_imgsz, band=band)
            dt = time.perf_counter() - t
            for d in ds: d.update(src=c, feature=feature(img, d['box']).tolist())
            cp.write_text(json.dumps(dict(image_signature=sig, dets=ds, inference_s=dt, mode=mode)), encoding='utf-8')
        print(f'[{n + 1}/{len(stems)}] {stem}', flush=True)


if __name__ == '__main__':
    main()
