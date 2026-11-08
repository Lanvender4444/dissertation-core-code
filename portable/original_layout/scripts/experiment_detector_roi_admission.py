"""Same frozen detections/geometry/capacity targets; add same-forward appearance."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'ultralytics'
sys.path[:0] = [str(BASE/'research'),str(BASE/'core_algorithms'),str(ROOT/'scripts')]
import detector_roi_admission as D
import nonlinear_admission as N
import residual_admission as R
from experiment_capacity_admission import checked,FEATURES
from experiment_residual_admission import verify,load_pool,corrected,SOURCE,BOX,POLICY
from experiment_recall_recovery import objects,write_new
from experiment_cam15_shared import FrameEvaluator
from validate_multiview import truth,metrics
from multiview_geometry import load_cameras
from cache_low_conf_no_band import sha

CAPACITY = BASE/'algorithm/evaluation_reports/capacity_admission_results_20261009.json'
DESCRIPTORS = BASE/'experiments/budget_fovea_20261009/detector_roi01'
LABELS = BASE/'experiments/budget_fovea_20261009/capacity01'


def expand(s,ref,index,geometry,cameras):
    fp = FEATURES/f'frames/{s}.json'
    r = checked(fp,ref['feature_sha256'][str(fp.relative_to(ROOT))])
    for ip,d in r['input_sha256'].items():
        if sha(ROOT/ip)!=d:
            raise ValueError('Changed frozen base input')
    p = index[s]
    descriptor = json.loads(p.read_text())
    if descriptor['protocol_sha256']!=sha(DESCRIPTORS/'protocol.json') or not descriptor['detection_signatures_exact'] or descriptor['detector_calls']!=5:
        raise ValueError('Unverified detector replay')
    for ip,d in descriptor['input_sha256'].items():
        if ip in r['input_sha256'] and r['input_sha256'][ip]!=d:
            raise ValueError('Descriptor/geometry provenance mismatch')
    pool,_ = load_pool(json.loads(geometry[s].read_text()),cameras,prepare=False)
    if set(descriptor['features'])!={str(u) for u,d in pool.items() if d['conf']>=.05}:
        raise ValueError('Incomplete active descriptor cache')
    for g in r['groups']:
        g['x'] = D.augment(g['x'],[pool[u] for u in g['uids']],descriptor['features']).tolist()
    for d in r['residuals']:
        d['x'] = D.augment(d['x'],[pool[d['uid']]],descriptor['features'],False).tolist()
    return r,pool


def run(args):
    ref = json.loads(CAPACITY.read_text())
    if not ref['complete'] or ref['frames']!=1655:
        raise ValueError('Require full frozen capacity baseline')
    verify(ref['protocol']['parent_protocol'])
    dp = json.loads((DESCRIPTORS/'protocol.json').read_text())
    for p,d in dp['source_sha256'].items():
        if sha(ROOT/p)!=d:
            raise ValueError('Changed frozen descriptor implementation')
    if args.stage!='fit':
        for i in range(dp['shard_count']):
            if json.loads((DESCRIPTORS/f'shard{i}/completion.json').read_text())!=dict(protocol_sha256=sha(DESCRIPTORS/'protocol.json')):
                raise ValueError('Descriptor cache incomplete')
    index = {p.stem:p for p in DESCRIPTORS.glob('shard*/frames/*.json')}
    if dp['frames']!=ref['protocol']['frames'] or not set(index)<=set(dp['frames']):
        raise ValueError('Different/incomplete frame domains')
    required = ref['protocol']['train'] if args.stage=='fit' else dp['frames']
    if not set(required)<=set(index):
        raise ValueError('Required training/full features not yet present')
    if args.report.exists() or (args.stage!='eval' and args.out.exists()):
        raise FileExistsError('Append-only feature ablation')
    cfg = dict(frames=dp['frames'],**{p:ref['protocol'][p] for p in ('train','development','guard','late')},
        parent_protocol=ref['protocol']['parent_protocol'],capacity_report_sha256=sha(CAPACITY),
        descriptor_protocol_sha256=sha(DESCRIPTORS/'protocol.json'),descriptor_source_sha256=dp['source_sha256'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),BASE/'research/detector_roi_admission.py',BASE/'research/nonlinear_admission.py')},
        models=ref['protocol']['models'],group_taus=ref['protocol']['group_taus'],residual_taus=ref['protocol']['residual_taus'],
        dimensions=dict(group=1056,residual=670),
        interpretation='Only same-forward YOLO appearance descriptors added. Same input and EXACT detector results, geometry identity groups, '
            'E1, corrected output boxes, capacity targets, model hyperparameters and split. Group labels unchanged. '
            'Only group/residual post-hoc heads fit on early train. No detector retraining or additional online detector call. '
            'GT-free feature/output construction; current GT evaluates only. Same explored route, not blind.',
        selection=ref['protocol']['selection'])
    if args.stage=='eval':
        if json.loads((args.out/'protocol.json').read_text())!=cfg:
            raise ValueError('Changed fitted protocol')
    else:
        write_new(args.out/'protocol.json',cfg)
    geometry = {p.stem:p for p in SOURCE.glob('shard*/frames/*.json')}
    cameras,offsets = load_cameras(),json.loads(POLICY.read_text())['y0']
    mp = args.out/'models.json'
    if args.stage!='eval':
        train = {k:dict(x=[],y=[]) for k in ('group','capacity')}
        training_digests = {}
        for i,s in enumerate(cfg['train'],1):
            training_digests[str(index[s].relative_to(ROOT))] = sha(index[s])
            r,_ = expand(s,ref,index,geometry,cameras)
            lp = LABELS/f'labels/{s}.json'
            labels = checked(lp,ref['label_sha256'][str(lp.relative_to(ROOT))])
            if labels['uids']!=[d['uid'] for d in r['residuals']]:
                raise ValueError('Changed training target ordering')
            for d in r['groups']:
                if d['label']>=0:
                    train['group']['x'].append(d['x'])
                    train['group']['y'].append(d['label'])
            for d,y in zip(r['residuals'],labels['labels']):
                if y>=0:
                    train['capacity']['x'].append(d['x'])
                    train['capacity']['y'].append(y)
            if i%200==0:
                print(f'Detector ROI training features {i}/{len(cfg["train"])}',flush=True)
        models = {}
        for family,params in cfg['models'].items():
            for head,data in train.items():
                models[f'{family}/{head}'] = N.fit(data['x'],data['y'],params)
                print(f'Detector ROI model {family}/{head}, {len(data["y"])} samples',flush=True)
        del train
        for p,d in training_digests.items():
            if sha(ROOT/p)!=d:
                raise ValueError('Changed training descriptor while fitting')
        write_new(mp,dict(models=models,protocol_sha256=sha(args.out/'protocol.json'),training_descriptor_sha256=training_digests))
        if args.stage=='fit':
            print('Early-only detector ROI fit complete; full evaluation not run',flush=True)
            return
    else:
        model_doc = json.loads(mp.read_text())
        if model_doc['protocol_sha256']!=sha(args.out/'protocol.json'):
            raise ValueError('Changed fitted model protocol')
        for p,d in model_doc['training_descriptor_sha256'].items():
            if sha(ROOT/p)!=d:
                raise ValueError('Changed fitted training input')
        models = model_doc['models']
    compiled = {k:N.compile_model(v) for k,v in models.items()}
    box = json.loads(BOX.read_text())['models']['foveated']
    old_rows = {r['frame']:r for r in ref['rows']}
    rows = []
    for i,s in enumerate(cfg['frames'],1):
        r,pool = expand(s,ref,index,geometry,cameras)
        changed = corrected(pool,box,cameras,offsets)
        seed,orphans = [d['uids'] for d in r['groups']],[d['uid'] for d in r['residuals']]
        scores,outputs = {},{}
        for family in cfg['models']:
            gs = N.predict([d['x'] for d in r['groups']],compiled[f'{family}/group'])
            rs = N.predict([d['x'] for d in r['residuals']],compiled[f'{family}/capacity'])
            scores[f'{family}/group'],scores[f'{family}/capacity'] = gs.tolist(),rs.tolist()
            for g in cfg['group_taus']:
                for t in cfg['residual_taus']:
                    outputs[f'{family}/group{g}/capacity{t}'] = objects(R.select(seed,gs,orphans,rs,g,t),changed)
        e = FrameEvaluator(truth(s,(1,2,3,4,5)),list(changed.values()))
        counts = {k:e.counts(v) for k,v in outputs.items()}
        counts.update(r['controls'])
        old_method = ref['selection']['recall_raw']
        counts[f'old/{old_method}'] = old_rows[s]['counts'][old_method]
        for family in cfg['models']:
            if counts[f'{family}/group0.0/capacity0.0']!=old_rows[s]['counts'][f'{family}/group0.0/capacity0.0']:
                raise ValueError('All-observation control changed')
        write_new(args.out/f'scores/{s}.json',dict(frame=s,scores=scores))
        rows.append(dict(frame=s,counts=counts))
        if i==1 or i%250==0:
            print(f'Detector ROI admission evaluation {i}/{len(cfg["frames"])}',flush=True)
    splits = {p:{k:metrics(np.sum([r['counts'][k] for r in rows if p=='all' or r['frame'] in cfg[p]],axis=0)) for k in rows[0]['counts']}
        for p in ('all','train','development','guard','late')}
    dev = splits['development']
    candidates = [k for k in dev if k.startswith(('shallow/','medium/'))]
    eligible = lambda control:[k for k in candidates if dev[k]['recall']>=dev[control]['recall']]
    choose = lambda ks:max(ks,key=lambda k:(dev[k]['precision'],dev[k]['recall'])) if ks else None
    selection = dict(recall_raw=choose(eligible('raw')),recall_corrected_raw=choose(eligible('raw_corrected')),
        precision_floors={str(p):choose([k for k in eligible('raw') if dev[k]['precision']>=p]) for p in (.4,.6,.8,.9)})
    verify(cfg['parent_protocol'])
    for p,d in cfg['source_sha256'].items():
        if sha(ROOT/p)!=d:
            raise ValueError('Changed scoring implementation')
    result = dict(complete=len(rows)==1655,frames=len(rows),protocol=cfg,model_sha256=sha(mp),selection=selection,splits=splits,rows=rows,
        descriptor_sha256={str(index[s].relative_to(ROOT)):sha(index[s]) for s in cfg['frames']},
        score_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((args.out/'scores').glob('*.json'))},
        limitations='Cached detector-specific auxiliary appearance, not independent detector/physical proof. '
            'Grouping fixed; capacity labels proxy, GT semantics unverified, same explored route. No end-to-end speed certification.')
    write_new(args.report,result)
    picked = sorted({selection['recall_raw'],selection['recall_corrected_raw'],*selection['precision_floors'].values()}-{None})
    print(json.dumps(dict(selection=selection,selected={k:splits['all'][k] for k in picked})),flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('fit','eval','fit_eval'),default='fit_eval')
    ap.add_argument('--out',type=Path,default=BASE/'experiments/budget_fovea_20261009/detector_roi_admission01')
    ap.add_argument('--report',type=Path,default=BASE/'algorithm/evaluation_reports/detector_roi_admission_results_20261009.json')
    args = ap.parse_args()
    args.out,args.report = args.out.resolve(),args.report.resolve()
    run(args)
