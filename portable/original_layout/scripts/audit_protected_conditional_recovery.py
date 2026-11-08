"""Independent counts/protection + all models/sources, early target replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import experiment_road_support_admission as E
import protected_reference as P
import protected_conditional_recovery as C
from protected_conditional_inputs import ProtectedInputs,REFERENCES,PARENT,PAIR
from fit_protected_conditional_recovery import OUT
from evaluate_protected_conditional_recovery import select
from validate_multiview import evaluate

REPORTS=E.BASE/'algorithm/evaluation_reports'


def independent_labels(groups,reference,ev):
    def edges(gs):return np.asarray([np.any([ev.scores[u]>=.5 for u in g],axis=0) for g in gs],bool).reshape(len(gs),len(ev.keys))
    def capacity(a):
        if not a.size:return 0
        r,c=linear_sum_assignment(-a.astype(int));return int(a[r,c].sum())
    base=edges(reference);base_capacity=capacity(base);extra=edges(groups);labels=[]
    for g,row in zip(groups,extra):
        y=capacity(np.vstack((base,row)))-base_capacity
        known=set()
        for u in g:
            hits=np.flatnonzero(ev.scores[u]>=.5)
            if len(hits)==1:known.add(int(hits[0]))
        if len(known)>1:y=0
        if not row.any() and any(u in ev.small_hits for u in g):y=-1
        labels.append(y)
    return labels


def run(domain='full'):
    target=REPORTS/('protected_conditional_independent_audit_20261010.json' if domain=='full' else 'protected_conditional_development_independent_audit_20261010.json')
    if target.exists():raise FileExistsError('Append-only protected recovery audit')
    rp=REPORTS/f'protected_conditional_{domain}_results_20261010.json';report=json.loads(rp.read_text())
    cfg=json.loads((OUT/'protocol.json').read_text());doc=json.loads((OUT/'models.json').read_text());inputs=ProtectedInputs()
    names=cfg['frames'] if domain=='full' else cfg['development']
    if not report['complete'] or report['frames']!=len(names) or report['model_sha256']!=E.sha(OUT/'models.json'):
        raise ValueError('Incomplete/changed protected full report')
    if report['protocol_sha256']!=E.sha(OUT/'protocol.json') or doc['protocol_sha256']!=E.sha(OUT/'protocol.json'):
        raise ValueError('Changed model/evaluation protocol')
    for p,key in {REFERENCES/'protocol.json':'references_protocol_sha256',REFERENCES/'crossfit_completion.json':'references_completion_sha256',
        E.OLD/'models.json':'baseline_group_model_sha256',PARENT/'models.json':'baseline_residual_model_sha256',
        PAIR/'models.json':'pair_model_sha256',E.BOX:'output_box_sha256',E.POLICY:'policy_sha256'}.items():
        if E.sha(p)!=cfg[key]:raise ValueError(f'Changed dependency {key}')
    reference_cfg=json.loads((REFERENCES/'protocol.json').read_text())
    if E.sha(E.REFERENCE)!=reference_cfg['reference_sha256']:raise ValueError('Changed original ROI evaluation source')
    E.verify_hashes(reference_cfg['original_group_fold_sha256']);E.verify_hashes(reference_cfg['source_sha256'])
    if set(cfg['train'])&set(cfg['development']+cfg['guard']+cfg['late']) or len(cfg['train'])!=601:raise ValueError('Early fit overlap')
    E.verify_hashes(cfg['source_sha256']);E.verify_hashes(doc['early_training_sha256'])
    lock=json.loads((REPORTS/'protected_conditional_development_results_20261010.json').read_text())
    if report['selection']!=lock['selection'] or select(report['splits']['development'])!=lock['selection']:raise ValueError('Development gate changed')
    ks=sorted({v for a in lock['selection'].values() for k,v in a.items() if k!='joint_candidates' and v is not None})
    ks=sorted(set(ks)|{f'{arm}/{batch}/tau{t}' for arm in ('single','pair','shared') for batch in cfg['batch_policies'] for t in (.55,.95)})
    models={k:E.N.compile_model(v) for k,v in doc['models'].items()};previous={r['frame']:r for r in report['rows']}
    max_error=0.;rows=[];samples={k:0 for k in ('single','pair')};positives=samples.copy()
    for i,s in enumerate(names,1):
        fp=OUT/f'evaluation_{domain}/frames/{s}.json';state=E.checked(fp,report['frame_sha256'][str(fp.relative_to(E.ROOT))])
        pool,changed,desc,flags,crops,source,unknown,a=inputs.frame(s)
        ps,pairs,pvalues=inputs.probabilities(pool,desc,flags);groups,trace=C.partition(a,pool,ps,**cfg['partition'])
        used=P.validate(a,pool);single=[[u] for u in sorted(pool) if u not in used]
        if a!=state['reference'] or dict(single=single,pair=groups)!=state['groups'] or source!=state['source_sha256']:
            raise ValueError('Reference/additional source replay mismatch')
        if json.loads(json.dumps(trace))!=state['trace'] or pairs.tolist()!=state['pairs']:raise ValueError('Partition/competition replay mismatch')
        error=float(np.max(np.abs(pvalues-state['pair_scores']))) if len(pvalues) else 0.;max_error=max(max_error,error)
        xs={k:C.features(gs,a,pool,inputs.cameras,inputs.offsets,desc,flags,ps,unknown) for k,gs in (('single',single),('pair',groups))}
        for k,m in models.items():
            actual=E.N.predict(xs['single' if k=='single' else 'pair'][:,:m['dim']],m)
            error=float(np.max(np.abs(actual-state['scores'][k]))) if len(actual) else 0.;max_error=max(max_error,error)
        if max_error>1e-12:raise ValueError('Pair/three recovery model prediction mismatch')
        outputs={'protected':a,'pair_pool_diagnostic':P.append(a,groups,pool)}
        for key in ks:
            arm,batch,t=key.split('/');gs=single if arm=='single' else groups
            outputs[key],decision=C.select(a,gs,state['scores'][arm],float(t[3:]),pool,ps,batch=='conservative')
            if decision!=state['decisions'][key]:raise ValueError('Batch-decision trace mismatch')
            if outputs[key][:len(a)]!=a:raise ValueError('Old output group replaced or mutated')
        gt=E.truth(s,(1,2,3,4,5));ev=E.FrameEvaluator(gt,list(changed.values()))
        # Independent matcher is not the producer's FrameEvaluator.counts.
        counts={k:evaluate(E.objects(g,changed),gt).tolist() for k,g in outputs.items()}
        counts['raw']=evaluate(E.objects([[u] for u in sorted(pool)],pool),gt).tolist()
        counts['raw_corrected']=evaluate(E.objects([[u] for u in sorted(pool)],changed),gt).tolist()
        if any(c!=previous[s]['counts'][k] for k,c in counts.items()):raise ValueError('Independent count mismatch')
        if any(c[0]<counts['protected'][0] for k,c in counts.items() if k not in ('raw','raw_corrected')):raise ValueError('Baseline TP lost')
        for k,g in outputs.items():
            P.validate(g,pool)
            if k in previous[s]['identity'] and ev.identity(E.objects(g,changed))!=previous[s]['identity'][k]:raise ValueError('Identity proxy mismatch')
        if s in cfg['train']:
            _,early_changed,early_desc,early_flags,_,proof,early_unknown,early_a=inputs.frame(s,early=True)
            eg,et=C.partition(early_a,pool,ps,**cfg['partition']);eu=P.validate(early_a,pool)
            es=[[u] for u in sorted(pool) if u not in eu]
            tp=OUT/f'training/{s}.json';saved=E.checked(tp,doc['early_training_sha256'][str(tp.relative_to(E.ROOT))])
            if saved['reference']!=early_a or proof!=saved['source_sha256']:raise ValueError('Early OOF reference changed')
            early_ev=E.FrameEvaluator(gt,list(early_changed.values()))
            for k,gs in (('single',es),('pair',eg)):
                y=independent_labels(gs,early_a,early_ev)
                if gs!=saved['groups'][k] or y!=saved['labels'][k]:raise ValueError('Early independent marginal-target arithmetic mismatch')
                samples[k]+=sum(v>=0 for v in y);positives[k]+=sum(v>0 for v in y)
        rows.append(dict(frame=s,counts=counts))
        if i==1 or i%200==0:print(f'Independent protected recovery audit {domain} {i}/{len(names)}',flush=True)
    for k in samples:
        if domain=='full' and (samples[k]!=doc['totals'][k+'_samples'] or positives[k]!=doc['totals'][k+'_positive']):raise ValueError('Early fit counts mismatch')
    E.verify_hashes(cfg['source_sha256'])
    E.write_new(target,dict(complete=True,domain=domain,frames=len(names),configurations=len(rows[0]['counts']),
        totals={k:E.metrics(np.sum([r['counts'][k] for r in rows],axis=0)) for k in rows[0]['counts']},
        maximum_prediction_error=max_error,pair_and_three_models_gt_free_replayed=True,
        immutable_baseline_source_exclusion_and_tp_nondecrease_all_frames=True,development_lock_replayed=True,
        early_targets_independently_recomputed=dict(frames=601 if domain=='full' else 0,samples=samples,positive=positives),
        report_sha256=E.sha(rp),audit_source_sha256=E.sha(Path(__file__)),
        limitations='Physical IDs unverified; scoring-reference DAG OOF only, pair/bbox calibration NOT OOF. '
            'Same explored scene, no new detector replay or end-to-end speed certificate.'))
    print('Independent protected recovery audit complete',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--domain',choices=('development','full'),default='full');run(ap.parse_args().domain)
