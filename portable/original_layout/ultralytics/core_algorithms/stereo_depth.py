"""Idea 1 - dense stereo depth for the three front cameras and robust per-box 3D.

The old fusion uses ONE pixel (box centre) per view to get disparity/depth.
Box centres are not the same 3D surface point across views, so depth errors are
large for far/truncated/occluded boxes.  Here each box is given a depth from a
dense SGBM disparity map: many pixels inside the box, robust statistics, and an
explicit uncertainty.  Training-free.

Pairs (IMU frame: x forward, y left):  cam2 (left, y=+0.37) - cam4 (right, y=-0.38)
  wide baseline 0.75 m  -> depth maps for cam2 (left view) and cam4 (right view)
  cam3 - cam4 baseline 0.50 m -> depth map for cam3 (left view)
Disparities are computed at half resolution; right views use the flip trick.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from multiview_geometry import ROOT

SCALE = 0.5
NUM_DISP = 192
PAIR_OF = {2: ((2, 4), 'L'), 4: ((2, 4), 'R'), 3: ((3, 4), 'L')}
CACHE = ROOT / 'ultralytics/experiments/stereo_cache'


class StereoPair:
    def __init__(self, cl, cr):
        self.cl, self.cr = cl, cr
        W, H = cl.width, cl.height
        R = cr.R.T @ cl.R                      # left-cam -> right-cam rotation
        T = cr.R.T @ (cl.t - cr.t)             # left origin expressed in right frame
        self.R1, self.R2, P1, P2, *_ = cv2.stereoRectify(cl.K, cl.D, cr.K, cr.D, (W, H), R.astype(np.float64), T.reshape(3, 1).astype(np.float64),
                                                         flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        S = np.diag([SCALE, SCALE, 1.])
        self.P1, self.P2 = S @ P1, S @ P2
        self.size = (int(W * SCALE), int(H * SCALE))
        self.f = self.P1[0, 0]; self.B = float(np.linalg.norm(T))
        self.maps = [cv2.initUndistortRectifyMap(cl.K, cl.D, self.R1, self.P1, self.size, cv2.CV_16SC2),
                     cv2.initUndistortRectifyMap(cr.K, cr.D, self.R2, self.P2, self.size, cv2.CV_16SC2)]
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=NUM_DISP, blockSize=5,
                                          P1=8 * 25, P2=32 * 25, disp12MaxDiff=2, uniquenessRatio=8,
                                          speckleWindowSize=80, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def rectify(self, gl, gr):
        return (cv2.remap(gl, *self.maps[0], cv2.INTER_LINEAR), cv2.remap(gr, *self.maps[1], cv2.INTER_LINEAR))

    def compute(self, img_l, img_r, right=True):
        gl, gr = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (img_l, img_r)]
        rl, rr = self.rectify(gl, gr)
        dl = self.sgbm.compute(rl, rr)                                   # int16, x16
        dr = cv2.flip(self.sgbm.compute(cv2.flip(rr, 1), cv2.flip(rl, 1)), 1) if right else None
        return dl, dr

    def to_rect(self, side, pixels):
        cam = self.cl if side == 'L' else self.cr
        return cv2.undistortPoints(np.asarray(pixels, np.float64).reshape(-1, 1, 2), cam.K, cam.D,
                                   R=self.R1 if side == 'L' else self.R2,
                                   P=self.P1 if side == 'L' else self.P2).reshape(-1, 2)

    def to_imu(self, side, uv, Z):
        P = self.P1 if side == 'L' else self.P2; R = self.R1 if side == 'L' else self.R2
        cam = self.cl if side == 'L' else self.cr
        xyz = np.column_stack([(uv[:, 0] - P[0, 2]) * Z / P[0, 0], (uv[:, 1] - P[1, 2]) * Z / P[1, 1], Z])
        return (cam.R @ (R.T @ xyz.T)).T + cam.t


class StereoDepth:
    """Per-frame depth maps for cams 2/3/4 and robust box depth queries."""

    def __init__(self, cameras):
        self.pairs = {(2, 4): StereoPair(cameras[2], cameras[4]), (3, 4): StereoPair(cameras[3], cameras[4])}
        self.cameras = cameras

    def compute_frame(self, images):
        d24l, d24r = self.pairs[2, 4].compute(images[2], images[4])
        d34l, _ = self.pairs[3, 4].compute(images[3], images[4], right=False)
        return {2: d24l, 4: d24r, 3: d34l}

    @staticmethod
    def cache_path(stem):
        return CACHE / f'{stem}.npz'

    def load_or_compute(self, stem, loader):
        p = self.cache_path(stem)
        if p.exists():
            try:
                z = np.load(p); return {c: z[f'd{c}'] for c in (2, 3, 4)}
            except Exception:
                pass
        maps = self.compute_frame(loader())
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **{f'd{c}': m for c, m in maps.items()})
        return maps

    def disparity_at(self, cid, dmap, pixels):
        (pair, side) = PAIR_OF[cid]; sp = self.pairs[pair]
        uv = sp.to_rect(side, pixels)
        x = np.round(uv[:, 0]).astype(int); y = np.round(uv[:, 1]).astype(int)
        ok = (x >= 0) & (x < dmap.shape[1]) & (y >= 0) & (y < dmap.shape[0])
        d = np.full(len(uv), np.nan)
        d[ok] = dmap[y[ok], x[ok]] / 16.
        d[d <= 0.25] = np.nan                    # invalid / zero disparity
        return uv, d

    def box_depth(self, cid, dmap, box, cls=2):
        """Robust depth of the object inside box.

        Samples a 9x9 grid in the central region (objects dominate there); the
        foreground is the NEAR mode, so the 65th disparity percentile is used
        (background leaks in at lower disparity).  Returns dict or None.
        """
        x1, y1, x2, y2 = box
        xs = np.linspace(x1 + .25 * (x2 - x1), x2 - .25 * (x2 - x1), 9)
        ys = np.linspace(y1 + .2 * (y2 - y1), y2 - .15 * (y2 - y1), 9)
        pts = np.array([(x, y) for y in ys for x in xs])
        uv, d = self.disparity_at(cid, dmap, pts)
        valid = np.isfinite(d)
        if valid.sum() < 12: return None
        dv = d[valid]
        disp = float(np.percentile(dv, 65)); spread = float(np.subtract(*np.percentile(dv, [80, 35])))
        if disp < .5: return None
        (pair, side) = PAIR_OF[cid]; sp = self.pairs[pair]
        Z = sp.f * sp.B / disp
        sigma_d = max(.35, .5 * spread)
        sigma_Z = Z * Z * sigma_d / (sp.f * sp.B)
        c_uv = sp.to_rect(side, [[(x1 + x2) / 2, (y1 + y2) / 2]])
        P = sp.to_imu(side, c_uv, np.array([Z]))[0]
        return dict(Z=float(Z), sigma=float(sigma_Z), point=P, valid_frac=float(valid.mean()), spread=spread)

    def depth_at_pixels(self, cid, dmap, pixels):
        """Depth (m) at raw pixels of camera cid; nan where invalid (occlusion test)."""
        (pair, side) = PAIR_OF[cid]; sp = self.pairs[pair]
        _, d = self.disparity_at(cid, dmap, pixels)
        return sp.f * sp.B / d
