"""Package exact thesis source snapshots and finite frozen-feature examples.

Does not copy RGB/XML datasets, weights, .git or virtual environments. The small
standalone replay is explicitly downstream of prepared features/reference A.
"""
import argparse
import ast
import hashlib
import json
import shutil
import sys
import textwrap
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ultralytics/decided'),str(ROOT/'scripts')]
from final_five_cam import CachedScene
import experiment_road_support_admission as E
import protected_conditional_recovery as C

SEEDS=[
 'ultralytics/decided/final_three_cam.py','ultralytics/decided/final_three_cam_no_band.py',
 'ultralytics/decided/final_cam2.py','ultralytics/decided/final_cam3.py','ultralytics/decided/final_cam4.py',
 'ultralytics/decided/final_five_cam.py','ultralytics/decided/final_spatial_cam.py',
 'ultralytics/decided/final_immersive_cam.py','ultralytics/core_algorithms/run_depth_bev.py',
 'scripts/fit_protected_conditional_recovery.py','scripts/evaluate_protected_conditional_recovery.py',
 'scripts/audit_protected_conditional_recovery.py','scripts/crossfit_protected_reference.py',
 'scripts/test_protected_conditional_recovery.py','scripts/test_protected_reference.py',
 'scripts/experiment_spatial_conditional_recovery.py','scripts/audit_spatial_conditional_recovery.py',
 'scripts/render_thesis_screenshots.py','scripts/build_thesis_figures.py',
 'scripts/audit_thesis_evidence.py','scripts/review_thesis_draft.py','scripts/audit_thesis_cases.py',
 'scripts/audit_thesis_recovery_intervals.py','scripts/audit_thesis_illustrations.py',
 'scripts/build_thesis_review.py','scripts/inspect_thesis_pdf.py','scripts/audit_thesis_archive.py']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def json_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def closure():
    folders=[ROOT/'scripts',ROOT/'ultralytics/research',ROOT/'ultralytics/core_algorithms',ROOT/'ultralytics/decided']
    index={}
    for directory in folders:
        for p in directory.glob('*.py'):index.setdefault(p.stem,[]).append(p)
    seen=set();pending=[ROOT/p for p in SEEDS];edges={};external=set()
    while pending:
        p=pending.pop()
        if p in seen:continue
        if not p.exists():raise FileNotFoundError(p)
        seen.add(p);tree=ast.parse(p.read_text(encoding='utf-8-sig'));deps=set()
        for node in ast.walk(tree):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                stem=name.split('.')[0]
                if stem in index:deps.update(index[stem])
                else:external.add(stem)
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py'):
                candidate=p.parent/node.value
                if candidate.is_file():deps.add(candidate)
        edges[str(p.relative_to(ROOT))]=sorted(str(d.relative_to(ROOT)) for d in deps)
        pending.extend(deps-seen)
    return sorted(seen),edges,sorted(external)


def extracted(path,name,method=False):
    source=path.read_text(encoding='utf-8-sig');tree=ast.parse(source)
    nodes=list(ast.walk(tree)) if method else tree.body
    matches=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(matches)!=1:raise ValueError(f'Nonunique source function {path}/{name}')
    n=matches[0];text=textwrap.dedent('\n'.join(source.splitlines()[n.lineno-1:n.end_lineno]))
    return text,dict(source=str(path.relative_to(ROOT)),function=name,start_line=n.lineno,
                    source_sha256=sha(path),extracted_sha256=hashlib.sha256(text.encode()).hexdigest())


def run(args):
    out=args.out.resolve();snapshot=out/'original_layout'
    if snapshot.exists() or (out/'fusion_core.py').exists():raise FileExistsError('Submission snapshot already exists')
    sources,edges,external=closure();records={}
    extras=['Intrinsics_cam.json','T_imu_cam(1).json','ultralytics/pyproject.toml','ultralytics/uv.lock','ultralytics/LICENSE',
            'thesis/evidence_snapshot_v6.json','thesis/recovery_block_intervals_v1.json','thesis/references.bib']
    for p in sources+[ROOT/p for p in extras]:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        if sha(p)!=sha(target):raise ValueError('Snapshot byte mismatch')
        records[str(p.relative_to(ROOT))]=dict(snapshot=str(target.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size)
    json_new(out/'metadata/source_manifest.json',dict(complete=True,seeds=SEEDS,files=records,
        dependency_edges=edges,external_imports=external,python_source_count=len(sources),
        scope='Exact local import closure plus explicit dynamic .py literals. External packages are dependencies, '
              'not copied implementations. Full pipeline still needs original external data/caches/models.'))
    parts=['"""Mechanically extracted frozen recovery/verification functions; see metadata/extractions.json."""',
           'import itertools\nfrom types import SimpleNamespace\nimport numpy as np\nfrom scipy.special import expit\nfrom scipy.optimize import linear_sum_assignment']
    mappings=[]
    for rel,names,method in [
        ('ultralytics/research/nonlinear_admission.py',['compile_model','predict'],False),
        ('ultralytics/research/protected_reference.py',['objects','validate','append'],False),
        ('ultralytics/research/joint_identity_admission.py',['overlap'],False),
        ('ultralytics/research/protected_conditional_recovery.py',['conflicts','select'],False),
        ('scripts/experiment_cam15_shared.py',['counts'],True)]:
        for name in names:
            text,record=extracted(ROOT/rel,name,method);parts.append(text);mappings.append(record)
    parts.append('P=SimpleNamespace(validate=validate,append=append)\nJ=SimpleNamespace(overlap=overlap)')
    (out/'fusion_core.py').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    json_new(out/'metadata/extractions.json',dict(complete=True,functions=mappings,
             adaptation='Import scaffolding and P/J namespaces only. Extracted function bodies are unchanged.'))
    from fit_protected_conditional_recovery import OUT
    doc=json.loads((OUT/'models.json').read_text());model=doc['models']['single'];compiled=E.N.compile_model(model)
    json_new(out/'models/recovery_single.json',model)
    scene=CachedScene();frames=[]
    for s in ('00000641','00000650','00001200'):
        state_path=OUT/f'evaluation_full/frames/{s}.json';state=E.checked(state_path,scene.report['frame_sha256'][str(state_path.relative_to(ROOT))])
        pool,changed,descriptor,flags,_,source,unknown,reference=scene.inputs.frame(s)
        if reference!=state['reference']:raise ValueError('Reference drift')
        probabilities={tuple(k):v for k,v in zip(state['pairs'],state['pair_scores'])}
        x=C.features(state['groups']['single'],reference,pool,scene.inputs.cameras,scene.inputs.offsets,
                     descriptor,flags,probabilities,unknown)[:,:model['dim']]
        scores=E.N.predict(x,compiled)
        error=float(np.max(np.abs(scores-np.asarray(state['scores']['single'])))) if len(scores) else 0.
        if error>1e-12:raise ValueError('Prepared feature replay differs from signed scores')
        final,_=C.select(reference,state['groups']['single'],scores,.55,pool,probabilities,False)
        truth=E.truth(s,(1,2,3,4,5));evaluator=E.FrameEvaluator(truth,list(changed.values()))
        actual=evaluator.counts(E.objects(final,changed));baseline=evaluator.counts(E.objects(reference,changed))
        expected=scene.rows[s]['counts']['single/independent/tau0.55']
        if actual!=expected:raise ValueError('Fixture counts drift')
        uids=sorted(pool);pairs=np.asarray(state['pairs'],int).reshape(-1,2)
        # Only finite numeric features and offline IoU graph; no pixels, XML or pickle.
        data=out/f'examples/{s}.npz';data.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(data,features=x,expected_scores=scores,
            evaluation_iou=np.asarray([evaluator.scores[u] for u in uids]),uids=np.asarray(uids,int),
            pairs=pairs,pair_scores=np.asarray(state['pair_scores']))
        fixture=dict(frame=s,reference=reference,candidates=state['groups']['single'],expected_final=final,
            pool={str(u):{k:pool[u][k] for k in ('uid','src','cls','conf','box')} for u in uids},
            corrected={str(u):{k:changed[u][k] for k in ('uid','src','cls','conf','box')} for u in uids},
            evaluation_keys=[str(k) for k in evaluator.keys],small_hits=sorted(evaluator.small_hits),
            expected_counts=expected,reference_counts=baseline,feature_score_replay_max_abs_error=error,
            source_sha256={**source,str(state_path.relative_to(ROOT)):sha(state_path)},
            npz_sha256=sha(data),scope='Frozen prepared features and protected reference; not raw-image end-to-end inference.')
        json_new(out/f'examples/{s}.json',fixture);frames.append(dict(frame=s,counts=actual,reference_counts=baseline,
                    candidates=len(x),features=model['dim'],maximum_score_error=error))
        print(f'Packaged prepared-feature example {s}: {actual[:3]}, score error {error:.3g}',flush=True)
    json_new(out/'metadata/sample_export.json',dict(complete=True,frames=frames,model_source=str((OUT/'models.json').relative_to(ROOT)),
        original_models_sha256=sha(OUT/'models.json'),original_protocol_sha256=sha(OUT/'protocol.json'),
        exported_single_sha256=sha(out/'models/recovery_single.json'),builder_sha256=sha(Path(__file__)),
        no_new_detector_calls=True,no_model_fit=True,scope='Three existing frames verify portable recovery only; '
              'not a new experiment or full-domain re-evaluation. Review dataset/provider and source licence rights before release.'))
    print(json.dumps(dict(output=str(out),source_files=len(sources),examples=frames)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'code');run(ap.parse_args())
