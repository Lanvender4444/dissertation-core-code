"""Illustrative native-image crops from the signed final scene, not new inference.

Selects from declared development frames. GT is read only for offline inspection.
No image synthesis, annotation edits, fitting or change of accepted detections.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ultralytics/decided'),str(ROOT/'scripts'),str(ROOT/'ultralytics/research'),str(ROOT/'ultralytics/core_algorithms')]
from final_five_cam import CachedScene
import experiment_road_support_admission as E
from validate_multiview import ev

FRAMES=('00000687','00000688','00000641','00000640','00000650','00000705')
NAMES={0:'Person',1:'Bicycle',2:'Car',3:'Motorcycle',5:'Bus',7:'Truck'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def match(objects,evaluator):
    matrix=np.asarray([np.maximum.reduce([evaluator.scores[d['uid']] for d in o['dets']]) for o in objects])
    matrix=matrix.reshape(len(objects),len(evaluator.keys))
    rr,cc=linear_sum_assignment(-((matrix>=.5)*1000+matrix*(matrix>=.5))) if matrix.size else ([],[])
    return matrix,{int(r):int(c) for r,c in zip(rr,cc) if matrix[r,c]>=.5}


def cases(scene):
    candidates={k:[] for k in ('recovery','partial','side','duplicate','localisation')}
    counts=[];sources={}
    for s in FRAMES:
        if s not in scene.cfg['development']:raise ValueError('Illustrative pool must be development-only')
        objects,_=scene.frame(s)
        state_path=E.BASE/f'experiments/protected_conditional_20261010/run01/recovery01/evaluation_full/frames/{s}.json'
        state=json.loads(state_path.read_text());sources[str(state_path.relative_to(ROOT))]=sha(state_path)
        all_dets={d['uid']:d for o in objects for d in o['dets']}
        gt=E.truth(s,(1,2,3,4,5))
        for cam in (1,2,3,4,5):
            xml=E.BASE/f'mycar/cam_{cam}/{s}.xml'
            sources[str(xml.relative_to(ROOT))]=sha(xml)
        evaluator=E.FrameEvaluator(gt,list(all_dets.values()))
        actual=evaluator.counts(objects)
        if actual!=scene.rows[s]['counts']['single/independent/tau0.55']:raise ValueError('Qualitative replay changed final counts')
        matrix,assigned=match(objects,evaluator)
        counts.append(dict(frame=s,counts=actual))
        reference=[o for o in objects if not o['recovered']]
        base=evaluator.counts(reference)[0]
        for i,o in enumerate(objects):
            best=int(np.argmax(matrix[i])) if matrix.shape[1] else None
            if best is None:continue
            maximum=float(matrix[i,best]);j=assigned.get(i,best);iid=evaluator.keys[j]
            member=max(o['dets'],key=lambda d:float(evaluator.scores[d['uid']][j]))
            uid=member['uid'];cam=member['src'];qualified=float(evaluator.scores[uid][j])
            record=dict(frame=s,object_id=o['id'],uids=[d['uid'] for d in o['dets']],sources=o['src'],
                        recovered=o['recovered'],camera=cam,uid=uid,box=member['box'],cls=member['cls'],
                        confidence=member['conf'],gt_id=str(iid),gt_box=evaluator.union[iid]['boxes'].get(cam),
                        best_compatible_iou=maximum,displayed_member_iou=qualified,
                        assigned=i in assigned,competitor=None)
            area=float(np.prod(np.asarray(member['box'])[2:]-np.asarray(member['box'])[:2]))
            record['selection_area']=area
            if i in assigned:
                if o['recovered'] and evaluator.counts(reference+[o])[0]>base:
                    candidates['recovery'].append(record)
                if len(o['src'])==1 and cam in (1,5):candidates['side'].append(record)
                if len(o['src'])==2 and set(o['src'])<={2,3,4}:
                    absent=next(c for c in (2,3,4) if c not in o['src'])
                    if absent in evaluator.union[iid]['boxes']:
                        p=dict(record,missing_camera=absent,missing_gt_box=evaluator.union[iid]['boxes'][absent])
                        candidates['partial'].append(p)
            elif maximum>=.5:
                rivals=[objects[k] for k,col in assigned.items() if col==best]
                common=[d for rival in rivals for d in rival['dets'] if d['src']==cam]
                if common:
                    record['competitor']={k:common[0][k] for k in ('uid','src','box','cls','conf')}
                    candidates['duplicate'].append(record)
            elif .1<=maximum<.5 and uid not in evaluator.small_hits:
                candidates['localisation'].append(record)
        print(f'Inspected signed development frame {s}: {actual[:3]}',flush=True)
    chosen={k:max(v,key=lambda r:r['selection_area']) for k,v in candidates.items() if v}
    if not {'recovery','partial','duplicate','localisation'}<=set(chosen):
        raise ValueError(f'Illustrative categories missing: {list(chosen)}')
    return chosen,counts,sources,{k:len(v) for k,v in candidates.items()}


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)


def panel(record,title,alternate=False):
    cam=record['missing_camera'] if alternate else record['camera']
    gt_box=record['missing_gt_box'] if alternate else record['gt_box']
    active=[] if alternate else [('prediction',record['box'],(220,40,50))]
    if not alternate and record['competitor']:
        active.append(('competing prediction',record['competitor']['box'],(225,140,15)))
    native=ROOT/f'ultralytics/mycar/images/{cam}/{record["frame"]}.png'
    im=Image.open(native).convert('RGB');w,h=im.size
    if (w,h)!=(2464,2056):raise ValueError('Unexpected native resolution')
    boxes=[np.asarray(gt_box)]+[np.asarray(b) for _,b,_ in active]
    lo=np.min([b[:2] for b in boxes],axis=0);hi=np.max([b[2:] for b in boxes],axis=0)
    middle=(lo+hi)/2;extent=np.maximum(hi-lo,[150,120])*1.6
    crop_box=np.r_[np.maximum(np.floor(middle-extent/2),0),np.minimum(np.ceil(middle+extent/2),[w,h])].astype(int)
    crop=im.crop(tuple(crop_box));scale=min(480/crop.width,270/crop.height)
    crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(520,420),'white');draw=ImageDraw.Draw(canvas)
    draw.text((14,10),title,font=font(23),fill=(20,30,45))
    if draw.textbbox((14,10),title,font=font(23))[2]>515:raise ValueError('Panel title would be clipped')
    draw.text((14,44),f'{record["frame"]} | Cam{cam} | {NAMES.get(record["cls"],str(record["cls"]))}',font=font(19),fill=(55,65,75))
    ox=(520-crop.width)//2;oy=80+(270-crop.height)//2;canvas.paste(crop,(ox,oy))
    def box(b,color):
        mapped=(np.asarray(b)-np.tile(crop_box[:2],2))*scale+np.tile([ox,oy],2)
        draw.rectangle(tuple(mapped),outline=color,width=4)
    box(gt_box,(0,125,180))
    for _,b,color in active:box(b,color)
    support='/'.join(str(c) for c in record['sources'])
    line='No member of this group in this view' if alternate else f'Object {record["object_id"]} | support {support} | IoU {record["displayed_member_iou"]:.3f}'
    draw.text((14,365),line,font=font(18),fill=(25,35,45))
    draw.text((14,393),f'GT annotation ID {record["gt_id"]}',font=font(17),fill=(0,100,150))
    detail=dict(camera=cam,image=str(native.relative_to(ROOT)),image_sha256=sha(native),
                native_crop=crop_box.tolist(),display_scale=scale,gt_box=gt_box,active_boxes=active)
    return canvas,detail


def run(args):
    if args.out.exists():raise FileExistsError('Append-only qualitative output')
    scene=CachedScene();chosen,counts,sources,available=cases(scene)
    layout=[('recovery','(a) Accepted recovery',False),('side','(b) Side-view singleton',False),
            ('partial','(c) Two-view group',False),('partial','(d) Other view of same annotation',True),
            ('duplicate','(e) Duplicate / assignment competition',False),('localisation','(f) Below the IoU gate',False)]
    if 'side' not in chosen:raise ValueError('No side-view singleton in declared illustrative pool')
    output=Image.new('RGB',(1040,1400),(238,241,245));draw=ImageDraw.Draw(output)
    draw.text((15,8),'Current frozen scene | illustrative native-image crops',font=font(25),fill=(20,30,45))
    draw.text((15,43),'RED: active prediction   BLUE: GT annotation   AMBER: competing prediction',font=font(21),fill=(40,50,60))
    panels=[]
    for i,(key,title,alternate) in enumerate(layout):
        im,detail=panel(chosen[key],title,alternate);output.paste(im,((i%2)*520,80+(i//2)*440))
        panels.append(dict(category=key,alternate=alternate,record=chosen[key],**detail))
    args.out.mkdir(parents=True)
    output.save(args.out/'native_cases.png',dpi=(180,180))
    manifest=dict(complete=True,configuration='single/independent/tau0.55',declared_frames=list(FRAMES),
                  selection='Largest active-box area within each defined category in six declared development frames; '
                            'illustrative selection, not random sampling or an estimate of error-category frequency.',
                  counts=counts,available_candidates=available,panels=panels,
                  report_sha256=sha(E.BASE/'algorithm/evaluation_reports/protected_conditional_full_results_20261010.json'),
                  renderer_sha256=sha(Path(__file__)),source_sha256=sources,
                  image_sha256=sha(args.out/'native_cases.png'),
                  limitations='GT ID and IoU are annotation/evaluation proxies, not independently verified physical identity. '
                              'No member in one view means no association for this group, not proved detector failure or occlusion. '
                              'Crops and offline GT overlays never feed inference; no new YOLO or fitting; output boxes are unchanged.')
    with (args.out/'manifest.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.out),counts=counts,selected={k:dict(frame=v['frame'],id=v['object_id'],camera=v['camera'],iou=v['displayed_member_iou']) for k,v in chosen.items()})),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args())
