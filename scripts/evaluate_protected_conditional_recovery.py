"""Immutable baseline + extra outputs: development lock, all-counted P/R."""
import argparse
import json
import time
from collections import Counter
import numpy as np
import experiment_road_support_admission as E
import protected_conditional_recovery as C
from protected_conditional_inputs import ProtectedInputs,REFERENCES,PARENT,PAIR
from fit_protected_conditional_recovery import OUT

REPORTS=E.BASE/'algorithm/evaluation_reports'


def select(dev):
    result={}
    for arm in ('single','pair','shared'):
        ks=[k for k in dev if k.startswith(arm+'/')]
        p90=[k for k in ks if dev[k]['precision']>=.9]
        joint=[k for k in p90 if dev[k]['recall']>=dev['raw']['recall']]
        raw=[k for k in ks if dev[k]['recall']>=dev['raw']['recall']]
        best=lambda a:max(a,key=lambda k:(dev[k]['recall'],dev[k]['precision'],k)) if a else None
        result[arm]=dict(joint_candidates=joint,joint_selected=best(joint),precision90_selected=best(p90),
            raw_recall_diagnostic=max(raw,key=lambda k:(dev[k]['precision'],dev[k]['recall'],k)) if raw else None,
            f1_diagnostic=max(ks,key=lambda k:(dev[k]['f1'],dev[k]['precision'],k)))
    return result


def run(domain):
    target=REPORTS/f'protected_conditional_{domain}_results_20261010.json'
    if target.exists() or (OUT/f'evaluation_{domain}').exists():raise FileExistsError('Append-only protected evaluation')
    cfg=json.loads((OUT/'protocol.json').read_text());doc=json.loads((OUT/'models.json').read_text());inputs=ProtectedInputs()
    dependencies={E.OLD/'models.json':'baseline_group_model_sha256',PARENT/'models.json':'baseline_residual_model_sha256',
        PAIR/'models.json':'pair_model_sha256',E.BOX:'output_box_sha256',E.POLICY:'policy_sha256',
        REFERENCES/'protocol.json':'references_protocol_sha256',REFERENCES/'crossfit_completion.json':'references_completion_sha256'}
    if doc['protocol_sha256']!=E.sha(OUT/'protocol.json'):raise ValueError('Changed recovery model/protocol')
    for p,k in dependencies.items():
        if E.sha(p)!=cfg[k]:raise ValueError(f'Changed dependency {k}')
    E.verify_hashes(cfg['source_sha256'])
    if domain=='full':
        lock=json.loads((REPORTS/'protected_conditional_development_results_20261010.json').read_text())
        if not lock['complete'] or lock['model_sha256']!=E.sha(OUT/'models.json'):raise ValueError('Development not locked')
    models={k:E.N.compile_model(v) for k,v in doc['models'].items()};rows=[];proofs={};times=[];trace_totals=Counter()
    names=cfg['development'] if domain=='development' else cfg['frames']
    for i,s in enumerate(names,1):
        begin=time.perf_counter();pool,changed,descriptor,flags,crops,source,unknown,a=inputs.frame(s)
        probabilities,pairs,pvalues=inputs.probabilities(pool,descriptor,flags)
        groups,trace=C.partition(a,pool,probabilities,**cfg['partition']);used={u for g in a for u in g}
        single=[[u] for u in sorted(pool) if u not in used]
        xs={k:C.features(gs,a,pool,inputs.cameras,inputs.offsets,descriptor,flags,probabilities,unknown)
            for k,gs in (('single',single),('pair',groups))}
        scores={k:E.N.predict(xs['single' if k=='single' else 'pair'][:,:m['dim']],m).tolist() for k,m in models.items()}
        outputs={'protected':a};decision={}
        for arm,values in scores.items():
            for batch in cfg['batch_policies']:
                for tau in cfg['taus']:
                    key=f'{arm}/{batch}/tau{tau}'
                    outputs[key],decision[key]=C.select(a,single if arm=='single' else groups,values,tau,pool,probabilities,batch=='conservative')
        outputs['pair_pool_diagnostic'],_=C.select(a,groups,np.ones(len(groups)),0.,pool,probabilities,False)
        elapsed=time.perf_counter()-begin
        # All inferred objects/decisions are fixed before evaluation GT access.
        truth=E.truth(s,(1,2,3,4,5));ev=E.FrameEvaluator(truth,list(changed.values()))
        counts={k:ev.counts(E.objects(g,changed)) for k,g in outputs.items()}
        corrected=ev.counts(E.objects([[u] for u in sorted(changed)],changed))
        evraw=E.FrameEvaluator(truth,list(pool.values()));raw=evraw.counts(E.objects([[u] for u in sorted(pool)],pool))
        old=inputs.old_rows[s]
        if counts['protected']!=old['novelty/tau0.55'] or raw!=old['raw'] or corrected!=old['raw_corrected']:
            raise ValueError('Frozen baseline/raw count control changed')
        if any(c[0]<counts['protected'][0] for c in counts.values()):raise ValueError('Immutable baseline TP protection failed')
        counts.update(raw=raw,raw_corrected=corrected)
        # Identity proxies for predeclared moderate/high points, not physical truth.
        identity={k:ev.identity(E.objects(g,changed)) for k,g in outputs.items()
            if k=='protected' or k=='pair_pool_diagnostic' or k.endswith('/tau0.55') or k.endswith('/tau0.95')}
        fp=OUT/f'evaluation_{domain}/frames/{s}.json';E.write_new(fp,dict(frame=s,reference=a,groups=dict(single=single,pair=groups),
            scores=scores,trace=trace,pairs=pairs.tolist(),pair_scores=pvalues.tolist(),decisions=decision,
            source_sha256=source,inference_s=elapsed,protocol_sha256=E.sha(OUT/'protocol.json')))
        proofs[str(fp.relative_to(E.ROOT))]=E.sha(fp);rows.append(dict(frame=s,counts=counts,identity=identity));times.append(elapsed)
        trace_totals.update(retired_same_camera=len(trace['retired_same_camera']),**trace['rejected_edges'])
        if i==1 or i%100==0:print(f'Protected conditional {domain} {i}/{len(names)}',flush=True)
    parts={p:set(cfg[p]) for p in ('train','development','guard','late')};parts['guard_late']=parts['guard']|parts['late']
    domains=('development',) if domain=='development' else ('all',*parts)
    splits={p:{k:E.metrics(np.sum([r['counts'][k] for r in rows if p=='all' or r['frame'] in parts[p]],axis=0)) for k in rows[0]['counts']} for p in domains}
    selection=select(splits['development'])
    if domain=='full' and (selection!=lock['selection'] or [r for r in rows if r['frame'] in parts['development']]!=lock['rows']):
        raise ValueError('Changed locked development decision')
    identities={p:{k:{f:sum(r['identity'][k].get(f,0) for r in rows if p=='all' or r['frame'] in parts[p])
        for f in set().union(*(r['identity'][k] for r in rows))} for k in rows[0]['identity']} for p in domains}
    E.verify_hashes(cfg['source_sha256'])
    E.write_new(target,dict(complete=len(rows)==len(names),frames=len(rows),domain=domain,selection=selection,splits=splits,
        identity_splits=identities,rows=rows,trace_totals=dict(trace_totals),frame_sha256=proofs,
        model_sha256=E.sha(OUT/'models.json'),protocol_sha256=E.sha(OUT/'protocol.json'),
        baseline_tp_nondecrease_all_frames=True,timing=dict(mean_s=float(np.mean(times)),p95_s=float(np.quantile(times,.95)),
            limitation='All three heads/policies and loader IO, excludes YOLO; NOT comparable deployment timing.'),
        limitations=cfg['interpretation']+' Same extensively explored route; no independent physical GT/scene or speed certificate. '
            'Individual training positives cannot be summed as jointly distinct recovered objects.'))
    ks=sorted({v for a in selection.values() for k,v in a.items() if k!='joint_candidates' and v is not None});p='development' if domain=='development' else 'all'
    print(json.dumps(dict(selection=selection,selected={k:splits[p][k] for k in ks},protected=splits[p]['protected'],
        raw=splits[p]['raw'],pool=splits[p]['pair_pool_diagnostic'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--domain',choices=('development','full'),default='development');run(ap.parse_args().domain)
