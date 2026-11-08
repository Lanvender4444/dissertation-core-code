"""Known artificial-input windows and censoring, not physical camera FOV.

No GT in window/edge construction. Clipped-GT comparisons belong to diagnostics
or early training only, never to the official full-box P/R definition.
"""
import numpy as np

from budget_fovea import unpack,WIDTH,HEIGHT,TILE_W,TILE_H


def window(d,entry,offsets):
    packed = np.asarray(entry['packed_box'],float)
    if packed.shape!=(4,) or not np.isfinite(packed).all() or entry['camera']!=d['src']:
        raise ValueError('Invalid packed observation')
    if packed[1]<HEIGHT/2<packed[3]:
        raise ValueError('Observation cannot cross disconnected input tiles')
    tile = int((packed[1]+packed[3])/2>=HEIGHT/2)
    y0 = offsets[str(d['src'])]
    metadata = dict(x0s=(0,2464-TILE_W),y0=y0)
    native = unpack(packed,metadata)
    if native is None or not np.allclose(native,d['box'],rtol=0,atol=1e-6):
        raise ValueError('Packed/native source observation mismatch')
    x0 = metadata['x0s'][tile]
    return np.array([x0,y0,x0+TILE_W,y0+TILE_H],float),tile


def edge_flags(box,crop,pixel_tolerance=3.):
    box,crop = np.asarray(box,float),np.asarray(crop,float)
    if box.shape!=(4,) or crop.shape!=(4,) or not np.isfinite(np.r_[box,crop]).all() or pixel_tolerance<0:
        raise ValueError('Invalid artificial edge measurement')
    # Units are detector-input pixels, not arbitrary native-pixel thresholds.
    distance = np.r_[box[:2]-crop[:2],crop[2:]-box[2:]]
    scales = np.array([WIDTH/TILE_W,(HEIGHT/2)/TILE_H]*2)
    if np.any(distance*scales<-.01):
        raise ValueError('Raw observation is not contained in source crop')
    return distance*scales<=pixel_tolerance  # left,top,right,bottom


def clip(box,crop):
    box,crop = np.asarray(box,float),np.asarray(crop,float)
    a = np.r_[np.maximum(box[:2],crop[:2]),np.minimum(box[2:],crop[2:])]
    return None if np.any(a[2:]<=a[:2]) else a


def bounded_area_iou_ceiling(raw_box,gt_box,maximum_scale=1.25):
    """Necessary bound, not attainable-IoU estimate or a global pool ceiling."""
    a,b = np.asarray(raw_box,float),np.asarray(gt_box,float)
    sa,sb = a[2:]-a[:2],b[2:]-b[:2]
    if np.any(sa<=0) or np.any(sb<=0) or maximum_scale<1:
        raise ValueError('Invalid box area bound')
    return float(min(1.,np.prod(sa)*maximum_scale**2/np.prod(sb)))
