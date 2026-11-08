"""Append-only same-input two-head and sparse-BEV recovery experiment."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import experiment_road_support_admission as E
import protected_conditional_recovery as C
import spatial_conditional_recovery as S
from protected_conditional_inputs import ProtectedInputs, REFERENCES
from fit_protected_conditional_recovery import OUT as OLD, labels

OUT = E.BASE/'experiments/spatial_conditional_20261012/run01'
REPORTS = E.BASE/'algorithm/evaluation_reports'
TAUS = [0.,.02,.05,.1,.15,.25,.4,.55,.7,.8,.9,.95,.98,.999]
ARMS = ('factored','spatial_factored')


def prepared(inputs, s, early=False):
    pool, changed, desc, flags, crops, proof, unknown, a = inputs.frame(s, early=early)
    pair, _, _ = inputs.probabilities(pool, desc, flags)
    used = {u for g in a for u in g}
    groups = [[u] for u in sorted(pool) if u not in used]
    x = C.features(groups,a,pool,inputs.cameras,inputs.offsets,desc,flags,pair,unknown)[:,:C.BASE_DIM]
    anchor = S.anchors(pool,inputs.cameras,pair)
    spatial = np.asarray([S.evidence(g[0],pool,inputs.cameras,anchor,a) for g in groups]).reshape(len(groups),S.DIM)
    return pool,changed,proof,a,groups,x,spatial,pair


def fit():
    if OUT.exists():
        raise FileExistsError('Use a new experiment directory; existing results are retained')
    old_cfg = json.loads((OLD/'protocol.json').read_text())
    E.verify_hashes(old_cfg['source_sha256'])
    sources = [Path(__file__), E.BASE/'research/spatial_conditional_recovery.py']
    cfg = dict(date='2026-10-12', **{k:old_cfg[k] for k in ('frames','train','development','guard','late')},
               old_protocol_sha256=E.sha(OLD/'protocol.json'),old_model_sha256=E.sha(OLD/'models.json'),
               reference_completion_sha256=E.sha(REFERENCES/'crossfit_completion.json'),
               source_sha256={str(p.relative_to(E.ROOT)):E.sha(p) for p in sources},
               taus=TAUS,arms=list(ARMS),params=dict(old_cfg['params'],max_iter=90),
               baseline='single/independent/tau0.55',
               selection='Development only: highest R with P>=.90; freeze before full. All thresholds retained.',
               limitations='Same explored scene. Scoring references nested OOF; bbox and pair NOT OOF. '
                 'Spatial anchors omit the queried camera but box-centre triangulation is approximate. '
                 'No new YOLO, GT-free prediction, no speed or physical-ID certificate.')
    E.write_new(OUT/'protocol.json',cfg)
    inputs=ProtectedInputs();xs=[];zs=[];vs=[];ns=[];proofs={};counts=dict(ignored=0,spatial_available=0)
    for i,s in enumerate(cfg['train'],1):
        pool,changed,proof,a,groups,x,z,pair=prepared(inputs,s,early=True)
        # All prediction inputs precede the first current-frame GT access.
        ev=E.FrameEvaluator(E.truth(s,(1,2,3,4,5)),list(changed.values()))
        n=np.asarray(labels(groups,a,ev));v=np.asarray([int(np.any(ev.scores[g[0]]>=.5)) for g in groups])
        keep=n>=0
        if np.any(n[keep]>v[keep]):raise ValueError('Novelty must imply validity')
        xs.extend(x[keep]);zs.extend(z[keep]);vs.extend(v[keep]);ns.extend(n[keep])
        counts['ignored']+=int((~keep).sum());counts['spatial_available']+=int(np.sum(z[keep,1]>0))
        fp=OUT/f'training/{s}.json'
        E.write_new(fp,dict(frame=s,reference=a,groups=groups,validity=v.tolist(),novelty=n.tolist(),
                           spatial=z.tolist(),source_sha256=proof))
        proofs[str(fp.relative_to(E.ROOT))]=E.sha(fp)
        if i==1 or i%50==0:print(f'Spatial conditional early assembly {i}/{len(cfg["train"])}',flush=True)
    x=np.asarray(xs,float);z=np.asarray(zs,float);v=np.asarray(vs,int);n=np.asarray(ns,int)
    models={}
    for arm,features in (('factored',x),('spatial_factored',np.column_stack((x,z)))):
        models[arm+'/validity']=E.N.fit(features,v,cfg['params'])
        # N subset V; second head learns only among valid early candidates.
        models[arm+'/novelty']=E.N.fit(features[v==1],n[v==1],cfg['params'])
        print(f'Fit {arm}: validity n={len(v)}, novel|valid n={sum(v)}, positives={sum(n)}',flush=True)
    E.verify_hashes(cfg['source_sha256']);E.verify_hashes(proofs)
    E.write_new(OUT/'models.json',dict(models=models,protocol_sha256=E.sha(OUT/'protocol.json'),
       training_sha256=proofs,totals=dict(samples=len(v),valid=int(v.sum()),novel=int(n.sum()),**counts)))
    print('Early-only fit complete; no performance claim before development/full evaluation',flush=True)


def choose(dev):
    result={}
    for arm in ARMS:
        ks=[k for k in dev if k.startswith(arm+'/')]
        p90=[k for k in ks if dev[k]['precision']>=.9]
        joint=[k for k in p90 if dev[k]['recall']>=dev['raw']['recall']]
        rank=lambda k:(dev[k]['recall'],dev[k]['precision'],k)
        result[arm]=dict(precision90_selected=max(p90,key=rank) if p90 else None,
                        joint_selected=max(joint,key=rank) if joint else None,
                        f1_diagnostic=max(ks,key=lambda k:(dev[k]['f1'],dev[k]['precision'],k)))
    return result


def evaluate(domain):
    target=REPORTS/f'spatial_conditional_{domain}_results_20261012.json'
    if target.exists() or (OUT/f'evaluation_{domain}').exists():raise FileExistsError('Append-only evaluation')
    cfg=json.loads((OUT/'protocol.json').read_text());doc=json.loads((OUT/'models.json').read_text())
    if doc['protocol_sha256']!=E.sha(OUT/'protocol.json'):raise ValueError('Model protocol mismatch')
    if cfg['old_protocol_sha256']!=E.sha(OLD/'protocol.json') or cfg['old_model_sha256']!=E.sha(OLD/'models.json'):
        raise ValueError('Frozen baseline changed')
    E.verify_hashes(cfg['source_sha256'])
    old=json.loads((REPORTS/'protected_conditional_full_results_20261010.json').read_text())
    controls={r['frame']:r['counts'] for r in old['rows']}
    models={k:E.N.compile_model(m) for k,m in doc['models'].items()}
    baseline=E.N.compile_model(json.loads((OLD/'models.json').read_text())['models']['single'])
    inputs=ProtectedInputs();names=cfg['development'] if domain=='development' else cfg['frames']
    lock=None
    if domain=='full':
        lock=json.loads((REPORTS/'spatial_conditional_development_results_20261012.json').read_text())
        if not lock['complete'] or lock['model_sha256']!=E.sha(OUT/'models.json'):raise ValueError('Development not locked')
    rows=[];proofs={};times=[]
    for i,s in enumerate(names,1):
        start=time.perf_counter();pool,changed,proof,a,groups,x,z,pair=prepared(inputs,s)
        scores={};outputs={'protected':a}
        oldscores=E.N.predict(x,baseline)
        outputs['baseline'],_=C.select(a,groups,oldscores,.55,pool,pair,False)
        for arm,features in (('factored',x),('spatial_factored',np.column_stack((x,z)))):
            v=E.N.predict(features,models[arm+'/validity']);n=E.N.predict(features,models[arm+'/novelty'])
            score=S.factored(v,n);scores[arm]=dict(validity=v.tolist(),novelty=n.tolist(),product=score.tolist())
            for tau in cfg['taus']:
                key=f'{arm}/tau{tau}';outputs[key],_=C.select(a,groups,score,tau,pool,pair,False)
        times.append(time.perf_counter()-start)
        # Output decisions fixed before evaluation truth.
        ev=E.FrameEvaluator(E.truth(s,(1,2,3,4,5)),list(changed.values()))
        counts={k:ev.counts(E.objects(g,changed)) for k,g in outputs.items()}
        counts['raw']=controls[s]['raw'];counts['raw_corrected']=controls[s]['raw_corrected']
        if counts['baseline']!=controls[s][cfg['baseline']] or counts['protected']!=controls[s]['protected']:
            raise ValueError('Old baseline control failed')
        if any(counts[k][0]<counts['protected'][0] for k in outputs):raise ValueError('Lost protected TP')
        fp=OUT/f'evaluation_{domain}/{s}.json'
        E.write_new(fp,dict(frame=s,reference=a,groups=groups,scores=scores,spatial=z.tolist(),source_sha256=proof))
        proofs[str(fp.relative_to(E.ROOT))]=E.sha(fp);rows.append(dict(frame=s,counts=counts))
        if i==1 or i%50==0:print(f'Spatial conditional {domain} {i}/{len(names)}',flush=True)
    parts={k:set(cfg[k]) for k in ('train','development','guard','late')};parts['guard_late']=parts['guard']|parts['late']
    domains=['development'] if domain=='development' else ['all',*parts]
    splits={p:{k:E.metrics(np.sum([r['counts'][k] for r in rows if p=='all' or r['frame'] in parts[p]],axis=0))
               for k in rows[0]['counts']} for p in domains}
    selection=choose(splits['development'])
    if lock and (selection!=lock['selection'] or [r for r in rows if r['frame'] in parts['development']]!=lock['rows']):
        raise ValueError('Development replay changed')
    E.verify_hashes(cfg['source_sha256'])
    E.write_new(target,dict(complete=True,frames=len(rows),domain=domain,splits=splits,selection=selection,rows=rows,
        frame_sha256=proofs,model_sha256=E.sha(OUT/'models.json'),protocol_sha256=E.sha(OUT/'protocol.json'),
        timing=dict(mean_s=float(np.mean(times)),scope='cache loading and all heads; NOT full deployment timing'),
        limitations=cfg['limitations']))
    p='development' if domain=='development' else 'all'
    print(json.dumps(dict(selection=selection,baseline=splits[p]['baseline'],selected={a:splits[p][v['precision90_selected']]
       if v['precision90_selected'] else None for a,v in selection.items()})),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('fit','development','full'),required=True)
    args=ap.parse_args();fit() if args.stage=='fit' else evaluate(args.stage)
