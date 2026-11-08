"""Idea 4 - guided YOLO re-detection in cameras that should have seen an object.

For a candidate object supported by some cameras and 'missing' in camera c
(visible and NOT occluded according to idea 3), the object's 3-D point is
projected into c, a window around the predicted box is cropped, up-sampled so
its long side is 640 px, and the SAME YOLO weights are run on it with a low
threshold (0.05).  A compatible box overlapping the prediction (IoU>=0.25) that
also passes the rectified-row/height gate is added as a new member.  This
spends high-resolution compute only where geometry says an object must be.
Results are cached by (frame, camera, crop) so evaluation is reproducible.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from multiview_geometry import ROOT
from multiview_dedup import box_iou
from multiview_fusion import compatible
from final_pipeline import core

CACHE = ROOT / 'ultralytics/experiments/redetect_cache'
CLASSES = (0, 1, 2, 3, 5, 7)


class ReDetector:
    def __init__(self, weights=str(ROOT / 'ultralytics/yolo11n.pt'), conf=.05, crop_side=640):
        self.weights, self.conf, self.side = weights, conf, crop_side
        self.model = None; self.images = {}; self.cache = {}; self.dirty = set()

    def _load(self, stem):
        if stem not in self.cache:
            p = CACHE / f'{stem}.json'
            self.cache[stem] = json.loads(p.read_text()) if p.exists() else {}
        return self.cache[stem]

    def flush(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        for stem in self.dirty:
            (CACHE / f'{stem}.json').write_text(json.dumps(self.cache[stem]), encoding='utf-8')
        self.dirty.clear(); self.images.clear()

    def crop_dets(self, stem, cid, win):
        c = self._load(stem); key = f'{cid}_{win[0]}_{win[1]}_{win[2]}_{win[3]}'
        if key in c: return c[key]
        if self.model is None:
            import torch; torch.set_num_threads(2)
            from ultralytics import YOLO; self.model = YOLO(self.weights)
        if (stem, cid) not in self.images:
            self.images[stem, cid] = cv2.imread(str(ROOT / f'ultralytics/mycar/images/{cid}/{stem}.png'))
        img = self.images[stem, cid][win[1]:win[3], win[0]:win[2]]
        s = self.side / max(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        r = self.model(img, conf=self.conf, imgsz=self.side, device='cpu', verbose=False)[0]
        out = [dict(box=[float(x1 / s + win[0]), float(y1 / s + win[1]), float(x2 / s + win[0]), float(y2 / s + win[1])],
                    conf=float(cf), cls=int(k)) for x1, y1, x2, y2, cf, k in r.boxes.data.cpu().numpy() if int(k) in CLASSES]
        c[key] = out; self.dirty.add(stem)
        return out

    def window(self, pb, W=2464, H=2056):
        cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
        half = max(1.5 * (pb[2] - pb[0]), 1.5 * (pb[3] - pb[1]), 48)
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half)); x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
        q = 8  # quantise so repeated runs hit the cache
        return [x0 // q * q, y0 // q * q, min(W, -(-x1 // q) * q), min(H, -(-y1 // q) * q)]

    def verify(self, stem, o, cid, pb, rect, min_conf=.05, min_iou=.25):
        win = self.window(pb)
        if win[2] - win[0] < 16 or win[3] - win[1] < 16: return None
        best = None
        for d in self.crop_dets(stem, cid, win):
            if not compatible(d['cls'], o['cls']) or d['conf'] < min_conf: continue
            v = box_iou(d['box'], pb)
            if v < min_iou: continue
            cand = dict(d, src=cid)
            cand['rbox'] = rect.rect_box(cand['box'], cid)
            x1, y1, x2, y2 = cand['box']
            cand['tl'], cand['tr'], cand['tb'] = x1 < core.TRUNC_PX, x2 > 2464 - core.TRUNC_PX, y2 > 2056 - core.TRUNC_PX
            if not all(core.gate(cand, m) for m in o['dets']): continue
            if best is None or v * cand['conf'] > best[0]: best = (v * cand['conf'], cand)
        return None if best is None else best[1]
