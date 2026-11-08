"""Fixed five-view detections: early-only group/residual calibration ablation.

Groups retain original geometric evidence; output boxes alone are calibrated.
Separate residual existence and new-object targets. All final predictions count.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'ultralytics'
sys.path[:0] = [str(BASE/'research'),str(BASE/'core_algorithms'),str(ROOT/'scripts')]
import residual_admission as R
import observation_conservation as O
import appearance_quality as A
import box_error_calibration as B
import cam15_scoring as Q
import shared_geometry as S
from experiment_recall_recovery import objects,write_new
from experiment_cam15_shared import FrameEvaluator
from multiview_geometry import load_cameras
from validate_multiview import truth,metrics
from cache_low_conf_no_band import sha

SOURCE = BASE/'experiments/budget_fovea_20261009/fusion01'
CACHE = BASE/'experiments/budget_fovea_20261009/full01/detections'
PARENT = BASE/'algorithm/evaluation_reports/budget_fovea_fusion_results_20261009.json'
BOX = BASE/'algorithm/evaluation_reports/box_error_calibration_results_20261009.json'
CONSERVATION = BASE/'algorithm/evaluation_reports/observation_conservation_results_20261009.json'
POLICY = BASE/'experiments/budget_fovea_20261009/probe01/policy.json'
GROUP_TAUS = (0.,.001,.01,.05,.1,.3)
RESIDUAL_TAUS = (0.,.001,.003,.01,.03,.1,.3)


def sources():
    return (Path(__file__),BASE/'research/residual_admission.py',BASE/'research/appearance_quality.py',
        BASE/'research/residual_recovery.py',BASE/'research/box_error_calibration.py',BASE/'research/shared_geometry.py',
        BASE/'research/cam15_scoring.py',BASE/'research/observation_conservation.py',
        ROOT/'scripts/experiment_cam15_shared.py',BASE/'core_algorithms/validate_multiview.py',BASE/'core_algorithms/tri-eval-gt.py')


def verify(cfg):
    for field in ('source_sha256','calibration_sha256','parent_sha256'):
        for p,d in cfg[field].items():
            if sha(ROOT/p)!=d:
                raise ValueError(f'Changed frozen input: {p}')


def load_pool(record,cameras,prepare=True):
    raw,proof = [],{}
    for c in (2,3,4,1,5):
        p = CACHE/f'{c}_{record["frame"]}.json'
        digest = sha(p)
        if digest!=record['input_sha256'][str(p.relative_to(ROOT))]:
            raise ValueError('Changed detection cache')
        proof[str(p.relative_to(ROOT))] = digest
        raw.extend(json.loads(p.read_text())['dets'])
    pool = {i:S.prepare_detection(dict(d,uid=i),cameras) if prepare else dict(d,uid=i) for i,d in enumerate(raw)}
    return pool,proof


def corrected(pool,model,cameras,offsets):
    return {u:B.correct(d,model,cameras,offsets,1.) for u,d in pool.items()}


def extract(args):
    parent,box,control = [json.loads(p.read_text()) for p in (PARENT,BOX,CONSERVATION)]
    if any(not r['complete'] or r['frames']!=1655 for r in (parent,box,control)):
        raise ValueError('Incomplete parents')
    names = parent['protocol']['frames'][:args.limit] if args.limit else parent['protocol']['frames']
    cfg = dict(frames=names,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources()},
        parent_sha256={str(p.relative_to(ROOT)):sha(p) for p in (PARENT,BOX,CONSERVATION,POLICY)},
        calibration_sha256=parent['protocol']['calibration_sha256'],floor=.05,l2=5.,group_taus=GROUP_TAUS,residual_taus=RESIDUAL_TAUS,
        train=[s for s in names if int(s)<=600],development=[s for s in names if 600<int(s)<=800],
        guard=[s for s in names if 800<int(s)<1100],late=[s for s in names if int(s)>=1100],
        interpretation='Same foveated detections and front0.0 groups. Frozen early-only output bbox correction. '
            'Group conditional appearance model refit to corrected output hit labels. Residual unary/context features GT-free. '
            'Two residual targets: any corrected GT hit, or GT not covered by corrected unscored seed. '
            'SourceUID used once; residuals never forced into seed groups; rejected groups not flattened. '
            'No hard overlap suppression, ego exclusion or unknown deferral in residual path: explicitly exploratory. '
            'All selected residuals are predictions in P/R, not hidden coverage. Same explored route, not blind.',
        selection='Development-only highest Precision among methods with Recall>=same-input uncorrected raw YOLO; '
            'report separate P>=.4/.6/.8/.9 feasibility. No full/late best selection.')
    cfg = json.loads(json.dumps(cfg))
    if args.out.exists():
        raise FileExistsError('Append-only output')
    verify(cfg)
    write_new(args.out/'protocol.json',cfg)
    index = {p.stem:p for p in SOURCE.glob('shard*/frames/*.json')}
    offsets,cameras = json.loads(POLICY.read_text())['y0'],load_cameras()
    model = box['models']['foveated']
    controls = {r['frame']:r for r in control['rows']}
    for n,s in enumerate(names,1):
        p = index[s]
        if sha(p)!=parent['extraction_sha256'][str(p.relative_to(ROOT))]:
            raise ValueError('Changed geometry source')
        record = json.loads(p.read_text())
        pool,proof = load_pool(record,cameras)
        proof[str(p.relative_to(ROOT))] = sha(p)
        seed = record['object_uids']['front0.0/unscored_pool_diagnostic']
        _,orphans = O.restore(seed,pool,.05)
        seed_objects = objects(seed,pool)
        unknown = {r['uid'] for r in record['diagnostics']['front0.0']['unknown']}
        group_x = [A.features(p['features']['shared'],[pool[u] for u in p['uids']],offsets,conditional=True)
            for p in record['proposals']['front0.0']]
        if [p['uids'] for p in record['proposals']['front0.0']]!=seed:
            raise ValueError('Feature/group ordering mismatch')
        residual_x = [R.features(pool[u],seed_objects,unknown,cameras,offsets) for u in orphans]
        changed = corrected(pool,model,cameras,offsets)
        # All features and changed outputs generated before current-frame GT.
        for c in (1,2,3,4,5):
            xp = BASE/f'mycar/cam_{c}/{s}.xml'
            digest = sha(xp)
            if digest!=record['input_sha256'][str(xp.relative_to(ROOT))]:
                raise ValueError('Changed truth')
            proof[str(xp.relative_to(ROOT))] = digest
        evaluator = FrameEvaluator(truth(s,(1,2,3,4,5)),list(changed.values()))
        covered = np.zeros(len(evaluator.keys),bool)
        groups = []
        for g,x in zip(seed,group_x):
            score = np.maximum.reduce([evaluator.scores[u] for u in g])
            covered |= score>=.5
            label = 1 if np.any(score>=.5) else (-1 if any(u in evaluator.small_hits for u in g) else 0)
            groups.append(dict(uids=g,x=x.tolist(),label=label))
        residuals = []
        for u,x in zip(orphans,residual_x):
            hits = evaluator.scores[u]>=.5
            existence = 1 if hits.any() else (-1 if u in evaluator.small_hits else 0)
            novelty = 1 if (hits & ~covered).any() else (-1 if existence<0 else 0)
            residuals.append(dict(uid=u,x=x.tolist(),existence=existence,novelty=novelty))
        raw = [[u] for u,d in pool.items() if d['conf']>=.05]
        zero_pool = FrameEvaluator(evaluator.gt,list(pool.values()))
        checks = dict(raw=zero_pool.counts(objects(raw,pool)),raw_corrected=evaluator.counts(objects(raw,changed)),
            seed_corrected=evaluator.counts(objects(seed,changed)))
        for k,old in [('raw','raw/s0.0'),('raw_corrected','raw/s1.0'),('seed_corrected','geometric_pool/s1.0')]:
            if checks[k]!=controls[s]['counts'][old]:
                raise ValueError('Control drift')
        write_new(args.out/f'frames/{s}.json',dict(frame=s,groups=groups,residuals=residuals,controls=checks,input_sha256=proof))
        if n==1 or n%250==0:
            print(f'Residual feature extraction {n}/{len(names)}',flush=True)
    verify(cfg)
    write_new(args.out/'completion.json',dict(protocol_sha256=sha(args.out/'protocol.json'),frames=len(names)))


def fit_eval(args):
    cfg = json.loads((args.out/'protocol.json').read_text())
    verify(cfg)
    if json.loads((args.out/'completion.json').read_text())!=dict(protocol_sha256=sha(args.out/'protocol.json'),frames=len(cfg['frames'])):
        raise ValueError('Incomplete extraction')
    paths = [args.out/f'frames/{s}.json' for s in cfg['frames']]
    rs = [json.loads(p.read_text()) for p in paths]
    models = {}
    for head,key in [('group','groups'),('existence','residuals'),('novelty','residuals')]:
        target = 'label' if head=='group' else head
        samples = [v for r in rs if r['frame'] in cfg['train'] for v in r[key] if v[target]>=0]
        models[head] = Q.fit([p['x'] for p in samples],[p[target] for p in samples],regularization=cfg['l2'])
        print(f'Residual model {head}, n={len(samples)}, positive={sum(p[target] for p in samples)}',flush=True)
    mp = args.out/'models.json'
    write_new(mp,dict(models=models,protocol_sha256=sha(args.out/'protocol.json')))
    offsets,cameras = json.loads(POLICY.read_text())['y0'],load_cameras()
    box_model = json.loads(BOX.read_text())['models']['foveated']
    geometry = {p.stem:p for p in SOURCE.glob('shard*/frames/*.json')}
    rows = []
    for n,r in enumerate(rs,1):
        for p,d in r['input_sha256'].items():
            if sha(ROOT/p)!=d:
                raise ValueError('Changed extracted input')
        record = json.loads(geometry[r['frame']].read_text())
        pool,_ = load_pool(record,cameras,prepare=False)
        changed = corrected(pool,box_model,cameras,offsets)
        gs = Q.predict([p['x'] for p in r['groups']],models['group'])
        seed = [p['uids'] for p in r['groups']]
        orphans = [p['uid'] for p in r['residuals']]
        residual_scores = {head:Q.predict([p['x'] for p in r['residuals']],models[head]) for head in ('existence','novelty')}
        outputs = {}
        for group_tau in cfg['group_taus']:
            outputs[f'group{group_tau}/no_residual'] = objects(R.select(seed,gs,[],[],group_tau,1.),changed)
            for head,scores in residual_scores.items():
                for tau in cfg['residual_taus']:
                    outputs[f'group{group_tau}/{head}{tau}'] = objects(R.select(seed,gs,orphans,scores,group_tau,tau),changed)
        # Current-frame truth only evaluates already frozen predictions.
        evaluator = FrameEvaluator(truth(r['frame'],(1,2,3,4,5)),list(changed.values()))
        counts = {k:evaluator.counts(v) for k,v in outputs.items()}
        counts.update(r['controls'])
        if counts['group0.0/no_residual']!=r['controls']['seed_corrected']:
            raise ValueError('Zero-score control drift')
        rows.append(dict(frame=r['frame'],counts=counts))
        if n==1 or n%250==0:
            print(f'Residual admission eval {n}/{len(rs)}',flush=True)
    splits = {part:{k:metrics(np.sum([r['counts'][k] for r in rows if part=='all' or r['frame'] in cfg[part]],axis=0)) for k in rows[0]['counts']}
        for part in ('all','train','development','guard','late') if part=='all' or cfg[part]}
    dev = splits.get('development',{})
    candidates = [k for k in dev if k.startswith('group')]
    feasible = [k for k in candidates if dev[k]['recall']>=dev['raw']['recall']]
    choose = lambda ks:max(ks,key=lambda k:(dev[k]['precision'],dev[k]['recall'])) if ks else None
    selection = dict(recall_raw=choose(feasible),precision_floors={str(p):choose([k for k in feasible if dev[k]['precision']>=p]) for p in (.4,.6,.8,.9)})
    result = dict(complete=len(rows)==1655,frames=len(rows),protocol=cfg,splits=splits,selection=selection,rows=rows,
        model_sha256=sha(mp),extraction_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        limitations='Identity groups frozen; labels are corrected any-member GT hits/novelty proxies. No identity or latency guarantee. '
            'Same explored route; no oracle test-frame admission. All-threshold full metrics diagnostic, only dev selects.')
    verify(cfg)
    write_new(args.report,result)
    picked = sorted({selection['recall_raw'],*selection['precision_floors'].values()}-{None})
    print(json.dumps(dict(selection=selection,selected={k:splits['all'][k] for k in picked},raw=splits['all']['raw'])),flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage',choices=('extract','fit_eval'))
    ap.add_argument('--out',type=Path,default=BASE/'experiments/budget_fovea_20261009/residual01')
    ap.add_argument('--report',type=Path,default=BASE/'algorithm/evaluation_reports/residual_admission_results_20261009.json')
    ap.add_argument('--limit',type=int,default=0)
    args = ap.parse_args()
    if args.limit<0:
        ap.error('Invalid scope')
    args.out,args.report = args.out.resolve(),args.report.resolve()
    (extract if args.stage=='extract' else fit_eval)(args)
