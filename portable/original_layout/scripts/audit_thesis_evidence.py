"""Read frozen results and produce append-only evidence for dissertation writing.

No detector inference, model fitting, annotation editing or threshold selection.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'ultralytics/algorithm/evaluation_reports'
KEY='single/independent/tau0.55'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):return json.loads((REPORTS/name).read_text(encoding='utf-8'))


def arithmetic(value):
    tp,fp,fn=(int(value[k]) for k in ('tp','fp','fn'))
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);f=2*tp/max(2*tp+fp+fn,1)
    for k,expected in (('precision',p),('recall',r),('f1',f)):
        if abs(value[k]-expected)>1e-10:raise ValueError(f'{k} mismatch')
    return dict(tp=tp,fp=fp,fn=fn,gt=tp+fn,precision=p,recall=r,f1=f)


def run(args):
    if args.out.exists():raise FileExistsError('Evidence snapshot exists; use a new output')
    full=read('protected_conditional_full_results_20261010.json')
    independent=read('protected_conditional_independent_audit_20261010.json')
    no_band=read('no_band_results_20261007.json')
    global3=read('global_fusion_results.json')
    original5=read('original_on_five_gt_20261008_retest01.json')
    spatial=read('spatial_conditional_full_results_20261012.json')
    spatial_audit=read('spatial_conditional_full_independent_audit_20261012.json')
    exact=read('spatial_recovery_exact_full_replay_20261012.json')
    intervals=read('spatial_conditional_block_intervals_20261012.json')
    protocol_path=ROOT/'ultralytics/experiments/protected_conditional_20261010/run01/recovery01/protocol.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if sha(protocol_path)!=full['protocol_sha256']:raise ValueError('Changed recovery protocol')
    if sorted(protocol['frames'])!=sorted(r['frame'] for r in full['rows']):raise ValueError('Recorded frame lists differ')
    segment_meta={name:dict(frames=len(protocol[name]),first=protocol[name][0],last=protocol[name][-1])
                  for name in ('train','development','guard','late')}
    viewer_audit=read('five_camera_viewer_run_1791807403777670200_audit_20261012.json')
    viewer_dir=ROOT/viewer_audit['run']
    viewer=json.loads((viewer_dir/'metrics.json').read_text())
    if not viewer_audit['complete'] or not viewer['complete_selection'] or viewer['visited']!=1655:
        raise ValueError('Incomplete position-availability evidence')
    for rel,digest in viewer_audit['run_sha256'].items():
        if sha(viewer_dir/rel)!=digest:raise ValueError('Changed original viewer run')
    if viewer['located_objects']!=viewer_audit['located_objects'] or viewer['output_objects']!=viewer_audit['output_objects']:
        raise ValueError('Viewer position counts differ')
    availability=viewer['located_objects']/viewer['output_objects']
    if abs(availability-viewer_audit['located_fraction'])>1e-12:raise ValueError('Viewer availability arithmetic differs')
    for doc in (spatial,spatial_audit,exact,intervals):
        if not doc['complete']:raise ValueError('Incomplete extension source')
    if any(doc['frames']!=1655 for doc in (spatial,spatial_audit,exact)):raise ValueError('Extension domain differs')
    if spatial_audit['report_sha256']!=sha(REPORTS/'spatial_conditional_full_results_20261012.json'):
        raise ValueError('Spatial independent audit refers to another report')
    extension=dict(coarse={k:arithmetic(spatial['splits']['all'][k]) for k in ('baseline','factored/tau0.55','spatial_factored/tau0.55')},
                   exact={k:arithmetic(exact['splits']['all'][k]) for k in ('existing_single','factored','spatial_factored')},
                   exact_selection=exact['selection'],spatial_recall_interval=intervals['results']['coarse']['all']['spatial_factored/tau0.55'])
    for k,counts in spatial_audit['independent_counts'].items():
        if counts!=[extension['coarse'][k][f] for f in ('tp','fp','fn')]:raise ValueError('Spatial independent totals differ')
    for doc,keys in ((spatial,extension['coarse']), (exact,extension['exact'])):
        for k in keys:
            summed=np.sum([row['counts'][k][:3] for row in doc['rows']],axis=0).tolist()
            if summed!=[arithmetic(doc['splits']['all'][k])[f] for f in ('tp','fp','fn')]:raise ValueError('Extension per-frame closure failed')
    if not full['complete'] or not independent['complete'] or not no_band['complete']:raise ValueError('Incomplete source')
    if full['frames']!=1655 or independent['frames']!=1655:raise ValueError('Wrong full domain')
    five={s:{k:arithmetic(full['splits'][s][k]) for k in ('protected',KEY,'pair/independent/tau0.55','shared/independent/tau0.55','raw','raw_corrected')}
          for s in ('all','train','development','guard','late','guard_late')}
    for k,value in independent['totals'].items():
        if k in full['splits']['all'] and arithmetic(value)!=arithmetic(full['splits']['all'][k]):raise ValueError('Independent audit differs')
    for k in five['all']:
        for field in ('tp','fp','fn'):
            if sum(five[s][k][field] for s in ('train','development','guard','late'))!=five['all'][k][field]:raise ValueError('Split totals do not close')
    row_totals=np.zeros(3,dtype=np.int64);base_totals=np.zeros(3,dtype=np.int64)
    for row in full['rows']:
        a=np.asarray(row['counts']['protected'][:3],int);b=np.asarray(row['counts'][KEY][:3],int)
        if b[0]<a[0]:raise ValueError('Protected TP fell on a frame')
        row_totals+=b;base_totals+=a
    if row_totals.tolist()!=[five['all'][KEY][k] for k in ('tp','fp','fn')]:raise ValueError('Per-frame totals differ')
    if base_totals.tolist()!=[five['all']['protected'][k] for k in ('tp','fp','fn')]:raise ValueError('Protected per-frame totals differ')
    three={k:arithmetic(global3['splits']['all_frames']['methods'][k]) for k in ('960/cam2','960/cam3','960/cam4','BAND/d','BAND/d_E1')}
    three['strict_no_band']=arithmetic(no_band['splits']['all_frames']['methods']['960/current_frozen'])
    three['adapted_960_score']=arithmetic(no_band['splits']['all_frames']['methods']['960/d_E1_existing_score'])
    sources=[REPORTS/n for n in ('protected_conditional_full_results_20261010.json','protected_conditional_independent_audit_20261010.json',
        'no_band_results_20261007.json','global_fusion_results.json','protected_conditional_full_stage_losses_20261010.json','original_on_five_gt_20261008_retest01.json',
        'spatial_conditional_full_results_20261012.json','spatial_conditional_full_independent_audit_20261012.json',
        'spatial_recovery_exact_full_replay_20261012.json','spatial_conditional_block_intervals_20261012.json',
        'five_camera_viewer_run_1791807403777670200_audit_20261012.json')]
    sources.extend([ROOT/'THESIS_DATA_LEDGER.json',ROOT/'dissertation template/main.tex',ROOT/'ultralytics/research/protected_conditional_recovery.py',
        ROOT/'ultralytics/research/nonlinear_admission.py',ROOT/'ultralytics/core_algorithms/validate_multiview.py',protocol_path,
        ROOT/'thesis/recovery_block_intervals_v1.json',ROOT/'scripts/audit_thesis_recovery_intervals.py',
        ROOT/'thesis/case_audit_v1.json',ROOT/'thesis/figures/native_cases_v3/manifest.json',ROOT/'thesis/figures/native_cases_v3/native_cases.png',
        ROOT/'scripts/render_thesis_cases.py',ROOT/'scripts/audit_thesis_cases.py',viewer_dir/'metrics.json',ROOT/'Zhang Chengwei Plan.pdf'])
    delta_tp=five['all'][KEY]['tp']-five['all']['protected']['tp'];delta_fp=five['all'][KEY]['fp']-five['all']['protected']['fp']
    result=dict(complete=True,scope='Arithmetic and saved-result audit, not new detector/identity/3D validation',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),five=five,three=three,
        current_key=KEY,protected_tp_nondecrease_frames=1655,extensions=extension,segment_meta=segment_meta,
        position_availability=dict(located=viewer['located_objects'],displayed=viewer['output_objects'],fraction=availability,
                                   ignored_displayed=viewer['output_objects']-viewer['metrics']['tp']-viewer['metrics']['fp']),
        original_three_on_five=arithmetic(original5['splits']['all_frames']['on_gt5']['three_cam_frozen']),
        controlled_recovery=dict(delta_tp=delta_tp,delta_fp=delta_fp,marginal_precision=delta_tp/(delta_tp+delta_fp)),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(complete=True,output=str(args.out),current=five['all'][KEY],recovery=result['controlled_recovery'])))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'thesis/evidence_snapshot.json')
    run(ap.parse_args())
