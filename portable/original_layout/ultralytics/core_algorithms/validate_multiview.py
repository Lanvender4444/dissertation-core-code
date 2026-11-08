"""Cached CPU benchmark with blocked development/test split and unique GT matching."""
import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
from collections import Counter
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiview_geometry import ROOT,load_cameras,SurroundRenderer
from multiview_fusion import feature,fuse,select,extend_front

HERE=Path(__file__).resolve().parent

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

core=module('legacy_core','tri-compare.py')
ev=module('legacy_eval','tri-eval-gt.py')

def truth(stem,cams):
    return {c:[g for g in ev.parse_xml(ROOT/f'ultralytics/mycar/cam_{c}/{stem}.xml') if g[1] in ev.GT_COMPAT] for c in cams}

def evaluate(objects,gt,min_h=25):
    union=ev.build_union(gt,min_h)
    ignored=ev.build_union(gt,0)
    keys=list(union)
    scores=np.zeros((len(objects),len(keys)))
    ignored_objects=set(); wrong=0; multi=0
    for i,o in enumerate(objects):
        matched=set()
        for d in o['dets']:
            for j,iid in enumerate(keys):
                u=union[iid]
                if d['src'] in u['boxes'] and ev.cls_match(u['label'],d['cls']):
                    v=ev.iou(d['box'],u['boxes'][d['src']])
                    scores[i,j]=max(scores[i,j],v)
                    if v>=.5: matched.add(iid)
        if len(o['src'])>1:
            multi+=1; wrong+=len(matched)>1
        if not len(keys) or scores[i].max()<.5:
            for iid,u in ignored.items():
                if iid in union: continue
                if any(d['src'] in u['boxes'] and ev.cls_match(u['label'],d['cls']) and
                       ev.iou(d['box'],u['boxes'][d['src']])>=.5 for d in o['dets']):
                    ignored_objects.add(i); break
    # Maximise cardinality FIRST, IoU second; a GT identity counts at most once.
    reward=(scores>=.5)*1000+scores*(scores>=.5)
    rr,cc=linear_sum_assignment(-reward) if scores.size else ([],[])
    tp=sum(scores[r,c]>=.5 for r,c in zip(rr,cc))
    return np.array([tp,len(objects)-len(ignored_objects)-tp,len(keys)-tp,wrong,multi],int)

def metrics(counts):
    tp,fp,fn,wrong,multi=map(int,counts)
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(p+r,1e-9),mixed_identity_groups=wrong,multiview_groups=multi)

def legacy(dets,rect):
    ds=copy.deepcopy(dets)
    for d in ds: d['rbox']=rect.rect_box(d['box'],d['src'])
    objs,by=core.compare_and_classify(rect,ds)
    kept,_=core.admission(objs)
    def adapt(o):
        return dict(dets=[by[c][i] for c,i in o['members'].items()],src=sorted(o['src']),conf=o['conf'],label=o['label'])
    return [adapt(o) for o in kept],[adapt(o) for o in objs]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frames',type=int,default=60)
    ap.add_argument('--imgsz',type=int,default=640)
    ap.add_argument('--model',default=str(ROOT/'ultralytics/yolov8n.pt'))
    ap.add_argument('--out',type=Path,default=ROOT/'ultralytics/experiments/multiview_v2')
    ap.add_argument('--render',type=int,default=3)
    ap.add_argument('--frozen-report',type=Path,help='Evaluate new late-sequence frames with thresholds frozen in this report; no tuning')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    cams=load_cameras(); rect=core.TriRectifier(core.camera_params)
    names=sorted((ROOT/'ultralytics/mycar/images/3').glob('*.png'))
    frozen=json.loads(args.frozen_report.read_text()) if args.frozen_report else None
    if frozen:
        used=set(frozen['development_frames']+frozen['guard_frames']+frozen['test_frames'])
        names=[p for p in names if p.stem>=min(frozen['test_frames']) and p.stem not in used]
    if not names: ap.error('No evaluation frames available')
    idx=np.linspace(0,len(names)-1,min(args.frames,len(names)),dtype=int)
    frames=[names[i].stem for i in idx]
    model_hash=hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    if frozen and (model_hash!=frozen['model_sha256'] or args.imgsz!=frozen['imgsz']):
        ap.error('Frozen report model hash and image size must match')
    key=f'{model_hash[:12]}_{args.imgsz}_conf010'
    cache=ROOT/'ultralytics/experiments/detection_cache'/key; cache.mkdir(parents=True,exist_ok=True)
    model=None; records=[]; timing=[]
    for fi,stem in enumerate(frames):
        dets=[]
        for cid in cams:
            ip=ROOT/f'ultralytics/mycar/images/{cid}/{stem}.png'
            cp=cache/f'{cid}_{stem}.json'
            stat=ip.stat(); signature=[stat.st_size,stat.st_mtime_ns]
            cached=json.loads(cp.read_text()) if cp.exists() else None
            if cached is None or cached['image_signature']!=signature:
                if model is None:
                    import torch
                    torch.set_num_threads(4)
                    from ultralytics import YOLO
                    model=YOLO(args.model)
                image=cv2.imread(str(ip)); start=time.perf_counter()
                res=model(image,conf=.10,imgsz=args.imgsz,device='cpu',verbose=False)[0]
                elapsed=time.perf_counter()-start
                ds=[]
                for row in res.boxes.data.cpu().numpy():
                    x1,y1,x2,y2,conf,cls=row
                    if int(cls) not in core.CLASS_NAMES: continue
                    box=[float(x1),float(y1),float(x2),float(y2)]
                    ds.append(dict(src=cid,box=box,conf=float(conf),cls=int(cls),feature=feature(image,box).tolist()))
                cached=dict(image_signature=signature,dets=ds,inference_s=elapsed)
                cp.write_text(json.dumps(cached),encoding='utf-8')
            dets.extend(cached['dets']); timing.append(cached['inference_s'])
        d3=[copy.deepcopy(d) for d in dets if d['src'] in (2,3,4)]
        old,all_old=legacy([d for d in d3 if d['conf']>=.2],rect)
        rec={'stem':stem,'gt3':truth(stem,(2,3,4)),'gt5':truth(stem,cams),'legacy':old,'raw_legacy':all_old}
        for tag,ds,cc,app in [('geometry3',d3,{c:cams[c] for c in (2,3,4)},False),
                              ('appearance3',d3,{c:cams[c] for c in (2,3,4)},True),
                              ('appearance5',dets,cams,True)]:
            start=time.perf_counter(); rec[tag]=fuse(copy.deepcopy(ds),cc,app)
            rec[tag+'_ms']=(time.perf_counter()-start)*1000
        rec['single3']=[dict(dets=[d],src=[3],conf=d['conf']) for d in d3 if d['src']==3]
        start=time.perf_counter()
        _,rec['legacy_tuned']=legacy(d3,rect)
        rec['hybrid5']=extend_front(copy.deepcopy(rec['legacy_tuned']),copy.deepcopy(dets),cams)
        rec['hybrid5_ms']=(time.perf_counter()-start)*1000
        records.append(rec)
        print(f'[{fi+1}/{len(frames)}] {stem} detections={len(dets)}',flush=True)
    # First 50% development, next 10% guard gap, final 40% untouched test.
    n=len(records); dev=records[:max(1,int(n*.5))]; test=records[int(n*.6):]
    if frozen: dev=[]; test=records
    reports={}; configurations={}
    def counts(rs,tag,threshold,gtkey):
        return sum((evaluate(select(r[tag],threshold),r[gtkey]) for r in rs),np.zeros(5,int))
    for tag in ('single3','geometry3','appearance3','appearance5','legacy_tuned','hybrid5'):
        gtkey='gt5' if tag.endswith('5') else 'gt3'
        grid=itertools.product((.2,.35,.5,.65),(.2,.35,.5,.65),(.3,.45,.6,.75)) if tag!='single3' else ((t,t,t) for t in (.2,.3,.4,.5,.6,.7))
        trials=[(metrics(counts(dev,tag,t,gtkey)),t) for t in grid] if not frozen else [(frozen['results'][tag]['development'],frozen['results'][tag]['thresholds'])]
        best=max(trials,key=lambda x:x[0]['f1'])
        th=best[1]; configurations[tag]=th
        reports[tag]={'thresholds':th,'development':metrics(counts(dev,tag,th,gtkey)),
                      'test':metrics(counts(test,tag,th,gtkey))}
        precise=[x for x in trials if x[0]['precision']>=.85]
        if frozen:
            profile=frozen['results'][tag].get('precision_profile')
            precise=[(profile['development'],profile['thresholds'])] if profile else []
        if precise:
            _,pth=max(precise,key=lambda x:x[0]['f1'])
            configurations[tag+'_precision']=pth
            reports[tag]['precision_profile']={'thresholds':pth,'development':metrics(counts(dev,tag,pth,gtkey)),
                                              'test':metrics(counts(test,tag,pth,gtkey))}
        if tag+'_ms' in records[0]: reports[tag]['association_ms_mean']=float(np.mean([r[tag+'_ms'] for r in records]))
    reports['legacy']={split:metrics(sum((evaluate(r['legacy'],r['gt3']) for r in rs),np.zeros(5,int))) for split,rs in [('development',dev),('test',test)]}
    # Comparing 3 and 5 cameras requires the SAME five-camera GT denominator.
    reports['appearance3_on_gt5']={split:metrics(counts(rs,'appearance3',configurations['appearance3'],'gt5')) for split,rs in [('development',dev),('test',test)]}
    gray=[o for r in test for o in r['raw_legacy'] if o['label']=='FP']
    gray_counts=sum((evaluate([o for o in r['raw_legacy'] if o['label']=='FP'],r['gt3']) for r in test),np.zeros(5,int))
    report=dict(model=args.model,model_sha256=model_hash,imgsz=args.imgsz,confidence=.1,
                min_gt_height=25,iou=.5,development_frames=[r['stem'] for r in dev],
                test_frames=[r['stem'] for r in test],guard_frames=[] if frozen else [r['stem'] for r in records[len(dev):int(n*.6)]],
                results=reports,legacy_gray_boxes=len(gray),legacy_gray_evaluation=metrics(gray_counts),
                inference_seconds_per_camera_mean=float(np.mean(timing)),frozen_source=str(args.frozen_report) if frozen else None,
                note='CPU wall time, not measured power. Same-sequence evaluation, not independent-scene generalization. No test-set threshold tuning.')
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    renderer=SurroundRenderer(cams) if args.render else None
    for rec in test[::max(1,len(test)//max(args.render,1))][:args.render]:
        selected=select(rec['appearance5'],configurations['appearance5'])
        imgs={c:cv2.imread(str(ROOT/f"ultralytics/mycar/images/{c}/{rec['stem']}.png")) for c in cams}
        cv2.imwrite(str(args.out/f"surround_{rec['stem']}.jpg"),renderer.render(imgs,selected))
        serial=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in o.items() if k!='dets'} for o in selected]
        (args.out/f"objects_{rec['stem']}.json").write_text(json.dumps(serial,indent=2),encoding='utf-8')
    (args.out/'configuration.json').write_text(json.dumps(configurations,indent=2),encoding='utf-8')

if __name__=='__main__': main()
