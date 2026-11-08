"""Independent selected-output counts, model replay and early target audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import experiment_road_support_admission as E
import spatial_conditional_recovery as S
import protected_conditional_recovery as C
from protected_conditional_inputs import ProtectedInputs
from experiment_spatial_conditional_recovery import OUT,REPORTS,prepared
from validate_multiview import evaluate


def capacity(edges):
    if not edges.size:return 0
    r,c=linear_sum_assignment(-edges.astype(int))
    return int(edges[r,c].sum())


def run(domain):
    rp=REPORTS/f'spatial_conditional_{domain}_results_20261012.json'
    target=REPORTS/f'spatial_conditional_{domain}_independent_audit_20261012.json'
    if target.exists():raise FileExistsError('Append-only independent audit')
    report=json.loads(rp.read_text());cfg=json.loads((OUT/'protocol.json').read_text())
    if not report['complete'] or report['protocol_sha256']!=E.sha(OUT/'protocol.json'):
        raise ValueError('Incomplete production report')
    E.verify_hashes(cfg['source_sha256']);doc=json.loads((OUT/'models.json').read_text())
    models={k:E.N.compile_model(v) for k,v in doc['models'].items()};inputs=ProtectedInputs()
    selected={arm:v['precision90_selected'] for arm,v in report['selection'].items() if v['precision90_selected']}
    totals={k:np.zeros(3,int) for k in selected.values()};maximum=0.;early_frames=0;early_samples=0
    for i,row in enumerate(report['rows'],1):
        s=row['frame'];pool,changed,proof,a,groups,x,z,pair=prepared(inputs,s)
        fp=OUT/f'evaluation_{domain}/{s}.json'
        state=E.checked(fp,report['frame_sha256'][str(fp.relative_to(E.ROOT))])
        if state['reference']!=a or state['groups']!=groups or state['source_sha256']!=proof:
            raise ValueError('Changed inference source/groups')
        maximum=max(maximum,float(np.max(np.abs(np.asarray(state['spatial'])-z))) if len(z) else 0.)
        outputs={}
        for arm,key in selected.items():
            features=x if arm=='factored' else np.column_stack((x,z))
            v=E.N.predict(features,models[arm+'/validity']);n=E.N.predict(features,models[arm+'/novelty'])
            p=S.factored(v,n)
            for name,actual in (('validity',v),('novelty',n),('product',p)):
                maximum=max(maximum,float(np.max(np.abs(actual-state['scores'][arm][name]))) if len(actual) else 0.)
            outputs[key],_=C.select(a,groups,p,float(key.split('tau')[1]),pool,pair,False)
        # All outputs fixed before truth access. Independent evaluator implementation.
        gt=E.truth(s,(1,2,3,4,5))
        for key,g in outputs.items():
            actual=evaluate(E.objects(g,changed),gt)[:3]
            if list(actual)!=row['counts'][key][:3]:raise ValueError('Independent metric mismatch')
            totals[key]+=actual
        if i==1 or i%100==0:print(f'Spatial independent {domain} {i}/{len(report["rows"])}',flush=True)
    if domain=='full':
        for i,s in enumerate(cfg['train'],1):
            pool,changed,proof,a,groups,x,z,pair=prepared(inputs,s,early=True)
            fp=OUT/f'training/{s}.json';state=E.checked(fp,doc['training_sha256'][str(fp.relative_to(E.ROOT))])
            if a!=state['reference'] or groups!=state['groups'] or proof!=state['source_sha256']:
                raise ValueError('Early reference/source mismatch')
            ev=E.FrameEvaluator(E.truth(s,(1,2,3,4,5)),list(changed.values()))
            edges=np.asarray([np.maximum.reduce([ev.scores[u]>=.5 for u in g]) for g in a],bool).reshape(len(a),len(ev.keys))
            base=capacity(edges);v=[];n=[]
            for g in groups:
                hit=ev.scores[g[0]]>=.5;v.append(int(hit.any()))
                value=capacity(np.vstack((edges,hit)))-base
                if not hit.any() and g[0] in ev.small_hits:value=-1
                n.append(value)
            if v!=state['validity'] or n!=state['novelty']:raise ValueError('Independent early target mismatch')
            early_frames+=1;early_samples+=len(groups)
            if i==1 or i%100==0:print(f'Spatial early target audit {i}/{len(cfg["train"])}',flush=True)
    if maximum>1e-12:raise ValueError('Prediction replay mismatch')
    E.write_new(target,dict(complete=True,domain=domain,frames=len(report['rows']),selected=selected,
        independent_counts={k:v.tolist() for k,v in totals.items()},maximum_prediction_error=maximum,
        early_target_frames=early_frames,early_target_candidates=early_samples,
        report_sha256=E.sha(rp),source_sha256=E.sha(Path(__file__)),limitations=cfg['limitations']))
    print(json.dumps(dict(complete=True,frames=len(report['rows']),maximum_error=maximum,early_frames=early_frames)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--domain',choices=('development','full'),required=True);run(ap.parse_args().domain)
