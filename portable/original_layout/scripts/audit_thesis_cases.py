"""Verify case crops against raw XML/native images and signed scene replay.

Independent XML parsing and IoU arithmetic; not a physical-identity certificate.
"""
import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'ultralytics/decided'),str(ROOT/'ultralytics/research'),str(ROOT/'ultralytics/core_algorithms')]
from final_five_cam import CachedScene
import experiment_road_support_admission as E
from render_thesis_cases import match


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    area=lambda x:float(np.prod(np.maximum(x[2:]-x[:2],0)))
    intersection=float(np.prod(wh))
    return intersection/max(area(a)+area(b)-intersection,1e-12)


def xml_box(path,identity):
    objects=[o for o in ET.parse(path).getroot().findall('object') if o.findtext('instanceId')==str(identity)]
    if len(objects)!=1:raise ValueError('Ambiguous/absent XML identity in illustrated view')
    pts=np.fromstring(objects[0].findtext('vertices/data'),sep=' ').reshape(-1,2)
    return np.r_[pts.min(axis=0),pts.max(axis=0)].tolist()


def run(args):
    if args.out.exists():raise FileExistsError('Append-only case audit')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if not manifest['complete'] or manifest['configuration']!='single/independent/tau0.55':raise ValueError('Wrong case configuration')
    png=args.manifest.parent/'native_cases.png'
    if digest(png)!=manifest['image_sha256'] or digest(ROOT/'scripts/render_thesis_cases.py')!=manifest['renderer_sha256']:
        raise ValueError('Changed illustration/renderer')
    for rel,sha in manifest['source_sha256'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('Changed case source')
    if Image.open(png).size!=(1040,1400):raise ValueError('Unexpected final illustration dimensions')
    scene=CachedScene();cached={}
    for frame in manifest['declared_frames']:
        objects,_=scene.frame(frame)
        evaluator=E.FrameEvaluator(E.truth(frame,(1,2,3,4,5)),[d for o in objects for d in o['dets']])
        # Repeated UIDs are absent by scene validation; compare exact saved counts.
        counts=evaluator.counts(objects)
        expected=next(r['counts'] for r in manifest['counts'] if r['frame']==frame)
        if counts!=expected:raise ValueError('Changed illustrative frame counts')
        matrix,assigned=match(objects,evaluator)
        cached[frame]=(objects,evaluator,matrix,assigned)
    audit=[]
    for panel in manifest['panels']:
        r=panel['record'];frame=r['frame'];objects,evaluator,matrix,assigned=cached[frame]
        i=r['object_id']-1;o=objects[i]
        if r['uids']!=[d['uid'] for d in o['dets']] or r['sources']!=o['src'] or r['recovered']!=o['recovered']:
            raise ValueError('Illustrated group changed')
        native=ROOT/panel['image']
        if digest(native)!=panel['image_sha256']:raise ValueError('Changed native pixels')
        gt=xml_box(ROOT/f'ultralytics/mycar/cam_{panel["camera"]}/{frame}.xml',r['gt_id'])
        if not np.allclose(gt,panel['gt_box'],rtol=0,atol=1e-9):raise ValueError('Illustrated GT differs from raw XML')
        if panel['alternate']:
            if panel['camera'] in o['src'] or panel['active_boxes']:raise ValueError('Absent-view panel invents observation')
            iou=None
        else:
            d=next(d for d in o['dets'] if d['uid']==r['uid'])
            if not np.allclose(d['box'],r['box'],rtol=0,atol=1e-9):raise ValueError('Altered active prediction')
            iou=overlap(d['box'],gt)
            if abs(iou-r['displayed_member_iou'])>1e-9:raise ValueError('Case IoU differs independently')
            if panel['category']=='recovery':
                a=[x for x in objects if not x['recovered']]
                if not o['recovered'] or evaluator.counts(a+[o])[0]!=evaluator.counts(a)[0]+1:raise ValueError('No marginal recovery gain')
            elif panel['category']=='side':
                if len(o['src'])!=1 or o['src'][0] not in (1,5) or i not in assigned:raise ValueError('Not a matched side singleton')
            elif panel['category']=='partial':
                if len(o['src'])!=2 or i not in assigned:raise ValueError('Not a matched two-view example')
            elif panel['category']=='duplicate':
                best=int(np.argmax(matrix[i]));uids={d['uid'] for k,j in assigned.items() if j==best for d in objects[k]['dets']}
                if i in assigned or matrix[i,best]<.5 or r['competitor']['uid'] not in uids:
                    raise ValueError('Not unmatched assignment competition')
            elif panel['category']=='localisation':
                if i in assigned or not .1<=matrix[i].max()<.5:raise ValueError('Not below the eligible IoU gate')
        audit.append(dict(category=panel['category'],frame=frame,camera=panel['camera'],gt_id=r['gt_id'],independent_iou=iou))
    result=dict(complete=True,frames=len(cached),panels=len(audit),checks=audit,
                manifest_sha256=digest(args.manifest),illustration_sha256=digest(png),
                audit_script_sha256=digest(Path(__file__)),
                scope='Raw XML box/native pixel hashes, active bbox/source identity, six signed frame counts and case predicates; '
                      'no physical identity, occlusion explanation, population error frequency or generalisation certificate')
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    run(ap.parse_args())
