"""Experimental input reuse and row-ROI stereo; no GT or fitted thresholds.

Kept outside core_algorithms so the historical frozen experiment's source
signatures remain intact. Cropping can change SGBM, so ROI is NOT an exact cache.
"""
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core_algorithms'))
from final_pipeline import filter_ego
from stereo_depth import StereoDepth, PAIR_OF
from yolo_hr_detect import CLASSES, band_tiles, merge_nms


def append_band(model, image, full_dets, tile_imgsz=1600, band=(850, 1250), conf=.10):
    """Reuse ORIGINAL pre-merge full-image detections; only run two tile forwards.

    Same class, edge and NMS rules as yolo_hr_detect.detect. Metadata on reused
    detections is stripped to the original box/conf/cls schema before merge.
    """
    h, w = image.shape[:2]
    raw = [{k: copy.deepcopy(d[k]) for k in ('box', 'conf', 'cls')} for d in full_dets]
    for x0, y0, xend, yend in band_tiles(w, band):
        result = model(image[y0:yend, x0:xend], conf=conf, imgsz=tile_imgsz, device='cpu', verbose=False)[0]
        for x1, y1, x2, y2, score, cls in result.boxes.data.cpu().numpy():
            if int(cls) not in CLASSES:
                continue
            m = 4
            if ((x1 < m and x0 > 0) or (y1 < m and y0 > 0)
                    or (x2 > xend-x0-m and xend < w) or (y2 > yend-y0-m and yend < h)):
                continue
            raw.append(dict(box=[float(x1+x0), float(y1+y0), float(x2+x0), float(y2+y0)],
                            conf=float(score), cls=int(cls)))
    return merge_nms(raw)


def sample_pixels(box):
    x1, y1, x2, y2 = box
    xs = np.linspace(x1+.25*(x2-x1), x2-.25*(x2-x1), 9)
    ys = np.linspace(y1+.2*(y2-y1), y2-.15*(y2-y1), 9)
    return np.array([(x, y) for y in ys for x in xs])


def merge_intervals(intervals, height):
    """Clip half-open intervals and merge overlaps/touching intervals."""
    result = []
    for start, stop in sorted((max(0, int(a)), min(height, int(b))) for a, b in intervals):
        if stop <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], stop)
        else:
            result.append([start, stop])
    return result


class RowRoiStereo(StereoDepth):
    """Full-width row strips preserve x coordinates and all queried box pixels.

    y padding is an engineering hypothesis, NOT a proof of SGBM equivalence.
    MODE_SGBM_3WAY paths and speckle filtering may depend on omitted rows.
    """
    def __init__(self, cameras, margin=64):
        super().__init__(cameras)
        if margin < 0:
            raise ValueError('Nonnegative half-resolution padding required')
        self.margin = margin
        self.last_info = {}

    def plan(self, dets):
        kept, _ = filter_ego(dets, self.cameras)
        kept = [d for c in (2, 3, 4) for d in merge_nms(
            [d for d in kept if d['src'] == c], iou_thr=.7, ios_thr=2.)]
        intervals = {(2, 4): [], (3, 4): []}
        for d in kept:
            pair, side = PAIR_OF[d['src']]
            sp = self.pairs[pair]
            uv = sp.to_rect(side, sample_pixels(d['box']))
            good = np.isfinite(uv).all(1)
            uv = uv[good]
            xy = np.round(uv).astype(int)
            good = (xy[:, 0] >= 0) & (xy[:, 0] < sp.size[0]) & (xy[:, 1] >= 0) & (xy[:, 1] < sp.size[1])
            ys = xy[good, 1]
            if len(ys):
                intervals[pair].append((ys.min()-self.margin, ys.max()+self.margin+1))
        return {pair: merge_intervals(rows, self.pairs[pair].size[1]) for pair, rows in intervals.items()}

    def compute_frame(self, images, dets):
        plans = self.plan(dets)
        gray = {c: cv2.cvtColor(images[c], cv2.COLOR_BGR2GRAY) for c in (2, 3, 4)}
        output = {}
        details = {}
        for pair, intervals in plans.items():
            left, right = pair
            sp = self.pairs[pair]
            h, w = sp.size[1], sp.size[0]
            dl = np.full((h, w), -16, np.int16)
            dr = np.full((h, w), -16, np.int16) if pair == (2, 4) else None
            if intervals:
                rl, rr = sp.rectify(gray[left], gray[right])
                for start, stop in intervals:
                    l, r = rl[start:stop], rr[start:stop]
                    dl[start:stop] = sp.sgbm.compute(l, r)
                    if dr is not None:
                        dr[start:stop] = cv2.flip(sp.sgbm.compute(cv2.flip(r, 1), cv2.flip(l, 1)), 1)
            output[left] = dl
            if dr is not None:
                output[right] = dr
            details[f'{left}{right}'] = dict(intervals=intervals, computed_rows=sum(b-a for a,b in intervals), height=h)
        total_rows = 2*details['24']['computed_rows']+details['34']['computed_rows']
        self.last_info = dict(pairs=details, weighted_row_fraction=total_rows/(3*details['24']['height']))
        return output
