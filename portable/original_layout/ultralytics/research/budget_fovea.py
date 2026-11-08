"""Two road-region tiles in ONE 960x832 detector input, original box coordinates.

This is explicitly foveated input enhancement, NOT raw/no-enhancement YOLO. It
omits outside regions; temporal input multiplexing must be evaluated separately.
"""
import cv2
import numpy as np

WIDTH, HEIGHT = 960, 832
TILE_W = 1312
TILE_H = 568


def pack(image, y0):
    if image.shape[:2] != (2056, 2464):
        raise ValueError('Unexpected camera image size')
    if not 0 <= y0 <= image.shape[0]-TILE_H:
        raise ValueError('Invalid crop')
    x0s = (0, image.shape[1]-TILE_W)
    tiles = [cv2.resize(image[y0:y0+TILE_H, x:x+TILE_W], (WIDTH, HEIGHT//2)) for x in x0s]
    return np.concatenate(tiles, axis=0), dict(y0=int(y0), x0s=x0s, source_w=TILE_W, source_h=TILE_H)


def unpack(box, metadata):
    x1, y1, x2, y2 = np.asarray(box, float)
    seam = HEIGHT//2
    if y1 < seam < y2:
        return None  # No native scene box can span two noncontiguous input tiles.
    k = int((y1+y2)/2 >= seam)
    x0 = metadata['x0s'][k]
    native = np.array([x0+x1*TILE_W/WIDTH, metadata['y0']+(y1-k*seam)*TILE_H/seam,
                       x0+x2*TILE_W/WIDTH, metadata['y0']+(y2-k*seam)*TILE_H/seam])
    native[[0, 2]] = np.clip(native[[0, 2]], 0, 2463)
    native[[1, 3]] = np.clip(native[[1, 3]], 0, 2055)
    return native.tolist() if min(native[2:]-native[:2]) >= 2 else None
