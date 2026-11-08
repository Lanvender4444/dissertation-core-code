"""Shared frozen front-fusion and plain single-camera pipelines.

Ego filtering is a rig-specific bottom-large-vehicle ROI gate, not a universal
semantic ego classifier. It is applied BEFORE fusion and identically to singles.
"""
from legacy_adapter import core,legacy
from multiview_fusion import select
from multiview_dedup import deduplicate

SINGLE_THRESHOLDS={2:.15,3:.2,4:.25}
FRONT_THRESHOLDS=[.2,.2,.45]
DEDUP_GATE=dict(iou_threshold=.4,min_shared=2,depth_veto=False)

def ego_candidate(d,width=2464,height=2056):
    x1,y1,x2,y2=d['box']
    return (d['cls'] in (2,5,7) and y2/height>=2031/2056
            and y1/height>=1300/2056 and (x2-x1)/width>=500/2464)

def filter_ego(dets,cameras,enabled=True):
    kept=[];removed=[]
    for d in dets:
        cam=cameras[d['src']]
        (removed if enabled and ego_candidate(d,cam.width,cam.height) else kept).append(d)
    return kept,removed

def run_pipeline(dets,cameras,cid=None,ego=True,rect=None):
    active=[d for d in dets if d['src'] in ((2,3,4) if cid is None else (cid,))]
    kept,removed=filter_ego(active,cameras,ego)
    if cid is not None:
        objects=[dict(dets=[d],src=[cid],conf=d['conf'],label='SINGLE')
                 for d in kept if d['conf']>=SINGLE_THRESHOLDS[cid]]
    else:
        if rect is None:rect=core.TriRectifier(core.camera_params)
        _,raw=legacy(kept,rect)
        objects=deduplicate(select(raw,FRONT_THRESHOLDS),**DEDUP_GATE)
    for i,o in enumerate(objects):
        o['id']=i+1
        o['label']=('FULL' if len(o['src'])==3 else 'PART' if len(o['src'])==2 else 'UNVERIFIED') if cid is None else 'SINGLE'
        o['unmatched_cameras']=[c for c in (2,3,4) if c not in o['src']] if cid is None else []
        o['cls']=max(o['dets'],key=lambda d:d['conf'])['cls']
    return objects,removed
