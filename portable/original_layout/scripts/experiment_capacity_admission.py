"""Change ONLY the early residual novelty target; reuse frozen group scorers."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'ultralytics'
sys.path[:0] = [str(BASE/'research'),str(BASE/'core_algorithms'),str(ROOT/'scripts')]
import marginal_capacity as M
import nonlinear_admission as N
import residual_admission as R
from experiment_nonlinear_admission import configurations,GROUP_TAUS,RESIDUAL_TAUS
from experiment_residual_admission import verify,load_pool,corrected,SOURCE,BOX,POLICY
from experiment_recall_recovery import objects,write_new
from experiment_cam15_shared import FrameEvaluator
from validate_multiview import truth,metrics
from multiview_geometry import load_cameras
from cache_low_conf_no_band import sha

PARENT = BASE/'algorithm/evaluation_reports/nonlinear_admission_results_20261009.json'
FEATURES = BASE/'experiments/budget_fovea_20261009/residual01'
OLD = BASE/'experiments/budget_fovea_20261009/nonlinear01'


def checked(path,digest):
    if sha(path)!=digest:
        raise ValueError(f'Changed frozen input: {path}')
    return json.loads(path.read_text())


def source_paths():
    return (Path(__file__),BASE/'research/marginal_capacity.py',BASE/'research/nonlinear_admission.py',
        ROOT/'scripts/experiment_nonlinear_admission.py')


def target_labels(seed,candidates,existence,old):
    result = M.marginal_labels(seed,candidates)
    labels = result['labels'].copy()
    existence,old = np.asarray(existence,int),np.asarray(old,int)
    if labels.shape!=existence.shape or labels.shape!=old.shape:
        raise ValueError('Mismatched residual ordering')
    if np.any((labels>0)&(existence!=1)) or np.any((old>0)&(labels<1)):
        raise ValueError('Capacity/coverage label invariant failed')
    labels[existence<0] = -1
    return labels,result['baseline_capacity']


def parent():
    ref = json.loads(PARENT.read_text())
    if not ref['complete'] or ref['frames']!=1655:
        raise ValueError('Incomplete reference')
    verify(ref['protocol']['parent_protocol'])
    return ref


def extract(args):
    ref = parent()
    if args.out.exists():
        raise FileExistsError('Append-only labels')
    split = ref['protocol']['parent_protocol']
    names = split['frames'][:args.limit] if args.limit else split['frames']
    cfg = dict(frames=names,**{p:[s for s in split[p] if s in names] for p in ('train','development','guard','late')},
        parent_report_sha256=sha(PARENT),parent_protocol=split,models=configurations(),
        group_taus=GROUP_TAUS,residual_taus=RESIDUAL_TAUS,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths()},
        target='Marginal maximum binary matching capacity of adding one residual observation to original unscored seed',
        interpretation='Only residual training target changes. Same group scorer/geometry/detections/output correction/features. '
            'GT-dependent labels are offline supervision/diagnosis ONLY. Fit exclusively original early train; '
            'no test-frame GT in inference. Individual marginal labels do not eliminate mutual residual competition.',
        selection='Development highest P with R>=uncorrected raw; separately same corrected raw and exploratory P floors. '
            'No full/late best selection. Same already explored route, not independent blind test.')
    write_new(args.out/'protocol.json',cfg)
    cameras,offsets = load_cameras(),json.loads(POLICY.read_text())['y0']
    box = json.loads(BOX.read_text())['models']['foveated']
    geometry = {p.stem:p for p in SOURCE.glob('shard*/frames/*.json')}
    totals = {p:Counter() for p in ('all','train','development','guard','late')}
    for i,s in enumerate(names,1):
        fp = FEATURES/f'frames/{s}.json'
        r = checked(fp,ref['feature_sha256'][str(fp.relative_to(ROOT))])
        for ip,d in r['input_sha256'].items():
            if sha(ROOT/ip)!=d:
                raise ValueError('Changed label source')
        pool,_ = load_pool(json.loads(geometry[s].read_text()),cameras,prepare=False)
        changed = corrected(pool,box,cameras,offsets)
        e = FrameEvaluator(truth(s,(1,2,3,4,5)),list(changed.values()))
        seed = np.array([np.maximum.reduce([e.scores[u] for u in g['uids']])>=.5 for g in r['groups']],bool).reshape(len(r['groups']),len(e.keys))
        candidates = np.array([e.scores[d['uid']]>=.5 for d in r['residuals']],bool).reshape(len(r['residuals']),len(e.keys))
        labels,baseline = target_labels(seed,candidates,[d['existence'] for d in r['residuals']],[d['novelty'] for d in r['residuals']])
        if baseline!=r['controls']['seed_corrected'][0]:
            raise ValueError('Baseline capacity/evaluation disagreement')
        stats = dict(samples=int(np.sum(labels>=0)),old_positive=sum(d['novelty']>0 for d in r['residuals']),
            new_positive=int(np.sum(labels>0)),changed_zero_to_positive=int(sum(d['novelty']==0 and label>0 for d,label in zip(r['residuals'],labels))))
        for p in totals:
            if p=='all' or s in cfg[p]:
                totals[p].update(stats)
                totals[p]['frames'] += 1
        write_new(args.out/f'labels/{s}.json',dict(frame=s,labels=labels.tolist(),uids=[d['uid'] for d in r['residuals']],
            stats=stats,baseline_capacity=baseline,feature_sha256=sha(fp)))
        if i==1 or i%250==0:
            print(f'Capacity label extraction {i}/{len(names)}',flush=True)
    verify(split)
    for p,d in cfg['source_sha256'].items():
        if sha(ROOT/p)!=d:
            raise ValueError('Changed target implementation')
    write_new(args.out/'label_completion.json',dict(frames=len(names),protocol_sha256=sha(args.out/'protocol.json'),totals={k:dict(v) for k,v in totals.items()}))
    print(json.dumps({k:dict(v) for k,v in totals.items()}),flush=True)


def fit_eval(args):
    ref = parent()
    cfg = json.loads((args.out/'protocol.json').read_text())
    if sha(PARENT)!=cfg['parent_report_sha256']:
        raise ValueError('Changed baseline')
    for p,d in cfg['source_sha256'].items():
        if sha(ROOT/p)!=d:
            raise ValueError('Changed frozen experiment')
    done = json.loads((args.out/'label_completion.json').read_text())
    if done['frames']!=len(cfg['frames']) or done['protocol_sha256']!=sha(args.out/'protocol.json'):
        raise ValueError('Incomplete labels')
    if args.report.exists() or (args.out/'models.json').exists():
        raise FileExistsError('Append-only fitted result')
    train_x,train_y = [],[]
    for s in cfg['train']:
        fp = FEATURES/f'frames/{s}.json'
        r = checked(fp,ref['feature_sha256'][str(fp.relative_to(ROOT))])
        label = json.loads((args.out/f'labels/{s}.json').read_text())
        if label['feature_sha256']!=sha(fp) or label['uids']!=[d['uid'] for d in r['residuals']]:
            raise ValueError('Changed supervision ordering')
        for d,y in zip(r['residuals'],label['labels']):
            if y>=0:
                train_x.append(d['x'])
                train_y.append(y)
    models = {}
    for family,params in cfg['models'].items():
        models[family] = N.fit(train_x,train_y,params)
        print(f'Capacity training {family}: {len(train_y)} samples, {sum(train_y)} positive',flush=True)
    mp = args.out/'models.json'
    write_new(mp,dict(models=models,protocol_sha256=sha(args.out/'protocol.json')))
    compiled = {k:N.compile_model(v) for k,v in models.items()}
    old_rows = {r['frame']:r for r in ref['rows']}
    cameras,offsets = load_cameras(),json.loads(POLICY.read_text())['y0']
    box = json.loads(BOX.read_text())['models']['foveated']
    geometry = {p.stem:p for p in SOURCE.glob('shard*/frames/*.json')}
    rows = []
    for i,s in enumerate(cfg['frames'],1):
        fp = FEATURES/f'frames/{s}.json'
        r = checked(fp,ref['feature_sha256'][str(fp.relative_to(ROOT))])
        oldp = OLD/f'scores/{s}.json'
        old_scores = checked(oldp,ref['score_sha256'][str(oldp.relative_to(ROOT))])['scores']
        pool,_ = load_pool(json.loads(geometry[s].read_text()),cameras,prepare=False)
        changed = corrected(pool,box,cameras,offsets)
        seed,orphans = [g['uids'] for g in r['groups']],[d['uid'] for d in r['residuals']]
        scores = {f:N.predict([d['x'] for d in r['residuals']],m) for f,m in compiled.items()}
        outputs = {}
        for family in cfg['models']:
            gs = old_scores[f'{family}/group']
            for g in cfg['group_taus']:
                for t in cfg['residual_taus']:
                    outputs[f'{family}/group{g}/capacity{t}'] = objects(R.select(seed,gs,orphans,scores[family],g,t),changed)
        # Evaluation truth arrives AFTER all GT-free outputs have been built.
        e = FrameEvaluator(truth(s,(1,2,3,4,5)),list(changed.values()))
        counts = {k:e.counts(v) for k,v in outputs.items()}
        counts.update(r['controls'])
        chosen = ref['selection']['recall_raw']
        counts[f'old/{chosen}'] = old_rows[s]['counts'][chosen]
        for family in cfg['models']:
            if counts[f'{family}/group0.0/capacity0.0']!=old_rows[s]['counts'][f'{family}/group0.0/novelty0.0']:
                raise ValueError('Full-pool control drift')
        write_new(args.out/f'scores/{s}.json',dict(frame=s,scores={k:v.tolist() for k,v in scores.items()}))
        rows.append(dict(frame=s,counts=counts))
        if i==1 or i%250==0:
            print(f'Capacity admission evaluation {i}/{len(cfg["frames"])}',flush=True)
    splits = {p:{k:metrics(np.sum([r['counts'][k] for r in rows if p=='all' or r['frame'] in cfg[p]],axis=0)) for k in rows[0]['counts']}
        for p in ('all','train','development','guard','late') if p=='all' or cfg[p]}
    dev = splits.get('development',{})
    candidates = [k for k in dev if k.startswith(('shallow/','medium/'))]
    eligible = lambda control:[k for k in candidates if dev[k]['recall']>=dev[control]['recall']]
    choose = lambda ks:max(ks,key=lambda k:(dev[k]['precision'],dev[k]['recall'])) if ks else None
    selection = dict(recall_raw=choose(eligible('raw')),recall_corrected_raw=choose(eligible('raw_corrected')),
        precision_floors={str(p):choose([k for k in eligible('raw') if dev[k]['precision']>=p]) for p in (.4,.6,.8,.9)})
    verify(cfg['parent_protocol'])
    for p,d in cfg['source_sha256'].items():
        if sha(ROOT/p)!=d:
            raise ValueError('Changed experiment source')
    result = dict(complete=len(rows)==1655,frames=len(rows),protocol=cfg,label_totals=done['totals'],model_sha256=sha(mp),
        selection=selection,splits=splits,rows=rows,feature_sha256={str((FEATURES/f'frames/{s}.json').relative_to(ROOT)):sha(FEATURES/f'frames/{s}.json') for s in cfg['frames']},
        label_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((args.out/'labels').glob('*.json'))},
        score_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((args.out/'scores').glob('*.json'))},
        limitations='Individual marginal capacity supervision, not physical identity/existence. Same explored route. '
            'No retrained detector, new boxes or test-GT admission. No end-to-end latency certification.')
    write_new(args.report,result)
    picked = sorted({selection['recall_raw'],selection['recall_corrected_raw'],*selection['precision_floors'].values()}-{None})
    print(json.dumps(dict(selection=selection,selected={k:splits['all'][k] for k in picked},label_totals=done['totals'])),flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage',choices=('extract','fit_eval'))
    ap.add_argument('--out',type=Path,default=BASE/'experiments/budget_fovea_20261009/capacity01')
    ap.add_argument('--report',type=Path,default=BASE/'algorithm/evaluation_reports/capacity_admission_results_20261009.json')
    ap.add_argument('--limit',type=int,default=0)
    args = ap.parse_args()
    if args.limit<0:
        ap.error('Invalid domain')
    args.out,args.report = args.out.resolve(),args.report.resolve()
    (extract if args.stage=='extract' else fit_eval)(args)
