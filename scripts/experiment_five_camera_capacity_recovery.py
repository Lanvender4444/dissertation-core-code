"""Five-view transfer, fixed final root/input/GT, isolated prefix-state controls.

Old five-view .05 crop cache retained; not the native Cam234 detector pool.
601 fit /200 lock /1655 full /native independent audit, append-only files.
"""
import argparse
from pathlib import Path
import numpy as np
import experiment_reconditioned_scene_recovery as R
import experiment_three_camera_capacity_recovery as C
import five_camera_capacity_recovery as S
import legacy_adapter

X=C.X
OUT=X.BASE/'experiments/five_camera_capacity_recovery_20261017/run01'
SUFFIX='20261017_v1'


def rp(stage):return X.REPORTS/f'five_camera_capacity_recovery_{stage}_{SUFFIX}.json'


def init():
    if OUT.exists():raise FileExistsError('Retain all previous five-view studies')
    c=X.read(R.OUT/'protocol.json');R.E.verify_hashes(c['sources']);R.E.verify_hashes(c['dependencies'])
    full=X.REPORTS/'reconditioned_scene_full_20261016_v1.json';audit=X.REPORTS/'reconditioned_scene_independent_audit_20261016_v1.json'
    a=X.read(audit)
    if not a['complete'] or a['frames']!=1655 or a['report_sha256']!=X.sha(full):raise ValueError('Unaudited five-view parent')
    files=[Path(__file__),X.BASE/'research/five_camera_capacity_recovery.py',X.BASE/'research/reconditioned_scene_recovery.py',
        X.BASE/'research/three_camera_predictive_recovery.py',X.ROOT/'scripts/experiment_reconditioned_scene_recovery.py',
        X.ROOT/'scripts/experiment_three_camera_capacity_recovery.py',X.ROOT/'scripts/protected_conditional_inputs.py',
        X.BASE/'research/marginal_capacity.py',X.BASE/'core_algorithms/tri-compare.py']
    deps=[R.OUT/'protocol.json',R.OUT/'model.json',full,audit,R.ORIGINAL/'models.json',
        X.REPORTS/'protected_conditional_full_results_20261010.json']
    X.write(OUT/'protocol.json',dict(**{k:c[k] for k in ('frames','train','development','guard','late','params')},
        sources={str(p.relative_to(X.ROOT)):X.sha(p) for p in files},dependencies={str(p.relative_to(X.ROOT)):X.sha(p) for p in deps},
        heads=list(S.DIMENSIONS),dimensions=S.DIMENSIONS,floors=[.9,.91,.92,.93],score_floor=.02,prefix_lengths=[0,1,3,6],
        intervention='Fixed original five-view final .55 scene, old corrected native-coordinate output boxes, old .05 foveated detector/ROI pool. '
            'Unlimited source-exclusive singleton recovery, confidence-prefix states0/1/3/6, exact marginal matching-capacity targets. '
            'All unused sources considered, not only old accepted candidates. No new YOLO or five-view weight reuse from Cam234.',
        evidence='Capacity529 preserves old rich913 except384 zero singleton ROI/std columns. Residual575 adds46 five-view '
            'remaining witness/competition features. Verification613 adds14 ROI competition and24 Cam234-only independent '
            'anchor prediction features. Fisheye1/5 use existing pinhole-Mei pair geometry, NEVER collinear translation.',
        selection='Each head same601 samples/targets/180trees. Develop200 locks exact P90/91/92/93 curves, then1655/native audit. '
            'Old baseline and raw corrected pool retained. No full/Guard/Late threshold selection.',
        limitations=c['limitations']+' Transfer of recovery PRINCIPLES, not identical detectors/parameters or a cross-dataset generalisation test. '
            'FiveGT24845 differs from threeGT13473. Upstream references/bbox/pair not newly fully nested OOF. '
            'Shared old crop/ROI/pair calibration already fitted; posthoc heads newly trained, detector frozen. '
            'Box-centre predictive uncertainty heuristic not physical identity/depth proof. E2E performance unverified.'))
    print('Five-camera recovery protocol locked, same five-view final/input/GT',flush=True)


def config():
    c=X.read(OUT/'protocol.json');X.check(c['sources']);X.check(c['dependencies']);return c


def parent(s,kind):
    ip=R.OUT/f'prepared/{kind}/{s}.npz';lp=R.OUT/f'labels/{kind}/{s}.npz';m=X.read(ip.with_suffix('.json'))
    if X.sha(ip)!=m['inference_sha256'] or X.sha(lp)!=m['labels_sha256']:raise ValueError('Changed five-view frozen banks')
    return X.array_file(ip),X.array_file(lp),m


def extra(s,kind,data,meta,inputs,old):
    pool,changed,descriptor,flags,crops,proof=inputs.load(s)
    ids=list(map(int,data['uids']));lookup={u:i for i,u in enumerate(ids)}
    if ids!=sorted(pool):raise ValueError('Five-view source bank changed')
    p=R.ORIGINAL/f'evaluation_full/frames/{s}.json';state=R.E.checked(p,old['frame_sha256'][str(p.relative_to(X.ROOT))])
    known=np.zeros(data['q'].shape,bool)
    for a,b in state['pairs']:
        if a in lookup and b in lookup and a!=b:known[lookup[a],lookup[b]]=known[lookup[b],lookup[a]]=True
    rect=legacy_adapter.core.TriRectifier(legacy_adapter.core.camera_params);n=len(ids);boxes=np.zeros((n,4));trunc=np.zeros((n,4),bool);baseline=np.zeros(n);fu=np.full(n,rect.fu)
    for i,u in enumerate(ids):
        d=pool[u]
        if d['src'] not in (2,3,4):continue
        b=d['box'];boxes[i]=rect.rect_box(b,d['src']);baseline[i]=rect.bx[d['src']]
        trunc[i]=[b[0]<8,b[2]>rect.W-8,b[1]<8,b[3]>rect.H-8]
    geometry=dict(data,rect_boxes=boxes,trunc=trunc,baseline=baseline,fu=fu,classes=np.array([pool[u]['cls'] for u in ids]),
        roi_cos=np.clip((data['phi'][:,91:475]/np.maximum(np.linalg.norm(data['phi'][:,91:475],axis=1,keepdims=True),1e-12))@
            (data['phi'][:,91:475]/np.maximum(np.linalg.norm(data['phi'][:,91:475],axis=1,keepdims=True),1e-12)).T,-1,1))
    front=np.flatnonzero(np.isin(data['cameras'],[2,3,4]));predictive=np.zeros((n,24))
    predictive[front]=S.Q.predictive_features(geometry,meta['scenes']['control'],front)
    S.attach(data,known,predictive)
    proof[str(p.relative_to(X.ROOT))]=X.sha(p)
    return dict(q_known=known,predictive=predictive),proof


def prepare(scope):
    c=config();inputs=R.ProtectedInputs();old=X.read(X.REPORTS/'protected_conditional_full_results_20261010.json');proof={}
    names=c['train']+c['development'] if scope=='early' else c['guard']+c['late']
    for n,s in enumerate(names,1):
        kinds=['train','replay'] if s in c['train'] else ['replay']
        for kind in kinds:
            d,lab,m=parent(s,kind);aux,src=extra(s,kind,d,m,inputs,old);p=OUT/f'prepared/{kind}/{s}.npz';X.save_npz(p,**aux)
            X.write(p.with_suffix('.json'),dict(frame=s,kind=kind,inference_sha256=X.sha(p),source_sha256=src,
                parent_metadata_sha256=X.sha(R.OUT/f'prepared/{kind}/{s}.json')))
            proof[str(p.with_suffix('.json').relative_to(X.ROOT))]=X.sha(p.with_suffix('.json'))
        if n==1 or n%100==0:print(f'Five-view evidence {scope} {n}/{len(names)}',flush=True)
    X.write(OUT/f'prepared_{scope}.json',dict(complete=True,frames=len(names),metadata_sha256=proof))


def load(s,kind):
    d,lab,m=parent(s,kind);p=OUT/f'prepared/{kind}/{s}.npz';a=X.read(p.with_suffix('.json'))
    if X.sha(p)!=a['inference_sha256'] or X.sha(R.OUT/f'prepared/{kind}/{s}.json')!=a['parent_metadata_sha256']:raise ValueError('Changed five-view feature alignment')
    return S.attach(d,**X.array_file(p)),lab,m


def fit():
    c=config();X.check(X.read(OUT/'prepared_early.json')['metadata_sha256']);xs=[];ys=[];stats={}
    for s in c['train']:
        d,lab,m=load(s,'train');root=m['scenes']['control'];idx,unused=S.features(d,root,'capacity');del unused
        ranking=sorted(idx.tolist(),key=lambda i:(-d['conf'][i],int(d['uids'][i])));seen=set();samples=positive=0
        for length in c['prefix_lengths']:
            scene=root+[[int(d['uids'][i])] for i in ranking[:length]];identity=tuple(tuple(g) for g in scene)
            if identity in seen:continue
            seen.add(identity);idx,x=S.features(d,scene,'verification');e=lab['edges'][idx]
            y=X.M.marginal_labels(C.scene_edges(scene,d,lab),e)['labels'];keep=~(~e.any(axis=1)&lab['small'][idx]);xs.append(x[keep]);ys.append(y[keep]);samples+=int(keep.sum());positive+=int(y[keep].sum())
        stats[s]=dict(samples=samples,positive=positive)
    x=np.concatenate(xs);y=np.concatenate(ys);models={};err=0.
    for head in c['heads']:
        doc=X.N.fit(x[:,:S.DIMENSIONS[head]],y,c['params']);err=max(err,X.portable_check(doc,x[:300,:S.DIMENSIONS[head]]));models[head]=doc
        print(f'Five-view fitted {head}: samples={len(y)} positive={int(y.sum())}',flush=True)
    X.write(OUT/'models.json',dict(models=models,samples=len(y),positive=int(y.sum()),per_frame=stats,max_native_error=err,protocol_sha256=X.sha(OUT/'protocol.json')))


def evaluate(domain):
    c=config();doc=X.read(OUT/'models.json');lock=X.read(OUT/'development_lock.json') if domain=='full' else None
    if doc['protocol_sha256']!=X.sha(OUT/'protocol.json'):raise ValueError('Changed five-view model protocol')
    if lock and (lock['protocol_sha256']!=X.sha(OUT/'protocol.json') or lock['model_sha256']!=X.sha(OUT/'models.json')):raise ValueError('Changed five-view lock')
    models={h:X.native_predictor(v) for h,v in doc['models'].items()};names=c['development'] if domain=='development' else c['frames'];frames=[];rows=[];proof={}
    original=X.read(X.REPORTS/'protected_conditional_full_results_20261010.json');oldrows={r['frame']:r for r in original['rows']}
    for n,s in enumerate(names,1):
        d,lab,m=load(s,'replay');root=m['scenes']['control'];paths={f'control/{h}':S.rollout(d,root,h,models[h],c['score_floor']) for h in c['heads']}
        counts=dict(baseline=C.counts(root,d,lab),raw=C.counts([[int(u)] for u in d['uids']],d,lab))
        expected=oldrows[s]['counts']['single/independent/tau0.55']
        if counts['baseline']!=expected[:3]:raise ValueError('Five-view final baseline changed')
        if domain=='development':frames.append(dict(data=d,lab=lab,meta=m,paths=paths))
        else:
            for k,v in lock['selection'].items():
                if v is None:continue
                p=paths[f"control/{v['head']}"];groups=root+[[u] for u,z in zip(p[0],p[1]) if z>=v['tau']];counts[k]=C.counts(groups,d,lab)
                if counts[k][0]<counts['baseline'][0]:raise ValueError('Protected append lost matching capacity')
        p=OUT/f'evaluation_{domain}/{s}.json';X.write(p,dict(frame=s,counts=counts,paths=paths));proof[str(p.relative_to(X.ROOT))]=X.sha(p);rows.append(dict(frame=s,counts=counts))
        if n==1 or n%200==0:print(f'Five-view recovery {domain} {n}/{len(names)}',flush=True)
    if domain=='development':
        selection={};curves={}
        for h in c['heads']:
            selected,points=C.curves(frames,'control',h,c['floors']);curves[h]=points
            for floor,z in selected.items():selection[f'{h}/P{floor}']=dict(head=h,**z) if z else None
        result=dict(selection=selection,curves=curves,model_sha256=X.sha(OUT/'models.json'),protocol_sha256=X.sha(OUT/'protocol.json'))
        X.write(OUT/'development_lock.json',result);print(str(selection),flush=True)
    else:
        parts={p:set(c[p]) for p in ('train','development','guard','late')};parts['guard_late']=parts['guard']|parts['late']
        splits={p:{k:X.metrics(np.sum([r['counts'][k] for r in rows if p=='all' or r['frame'] in parts[p]],axis=0)) for k in rows[0]['counts']} for p in ('all',*parts)}
        for k,v in lock['selection'].items():
            if v is not None and splits['development'][k]!={f:v[f] for f in ('tp','fp','fn','precision','recall','f1')}:raise ValueError('Five-view development curve differs')
        if splits['all']['baseline']['tp']+splits['all']['baseline']['fn']!=24845:raise ValueError('Wrong five-view GT denominator')
        result=dict(selection=lock['selection'],splits=splits,model_sha256=X.sha(OUT/'models.json'),lock_sha256=X.sha(OUT/'development_lock.json'))
    X.write(rp(domain),dict(complete=True,frames=len(names),rows=rows,frame_sha256=proof,**result))


def audit():
    c=config();r=X.read(rp('full'));X.check(r['frame_sha256']);models={h:X.native_predictor(v) for h,v in X.read(OUT/'models.json')['models'].items()};inputs=R.ProtectedInputs();old=X.read(X.REPORTS/'protected_conditional_full_results_20261010.json')
    totals={k:np.zeros(3,int) for k in r['splits']['all']}
    for n,row in enumerate(r['rows'],1):
        s=row['frame'];pool,changed,desc,flags,crops,proof=inputs.load(s);d,unused,m=parent(s,'replay');aux,proof=extra(s,'replay',d,m,inputs,old);d=S.attach(d,**aux)
        # Recompute underlying rich singleton features from signed native inputs.
        original=X.read(R.ORIGINAL/f'evaluation_full/frames/{s}.json');q={tuple(map(int,p)):float(v) for p,v in zip(original['pairs'],original['pair_scores'])}
        rich=R.C.object_features([[int(u)] for u in d['uids']],[],pool,inputs.cameras,desc,flags,q,changed)
        if not np.allclose(rich,d['phi'],rtol=0,atol=1e-12):raise ValueError('Fresh five-view observable evidence differs')
        root=m['scenes']['control'];paths={h:S.rollout(d,root,h,models[h],c['score_floor']) for h in c['heads']};state=X.read(OUT/f'evaluation_full/{s}.json')
        for h,p in paths.items():
            prev=state['paths'][f'control/{h}']
            if p[0]!=prev[0] or not np.allclose(p[1:],prev[1:],rtol=0,atol=1e-12):raise ValueError('Fresh five-view GT-free path differs')
        groups=dict(baseline=root,raw=[[int(u)] for u in d['uids']])
        for k,v in r['selection'].items():
            if v is not None:
                p=paths[v['head']];groups[k]=root+[[u] for u,z in zip(p[0],p[1]) if z>=v['tau']]
        gt=X.truth(s,(1,2,3,4,5));cache={}
        for k,g in groups.items():
            X.P.validate(g,pool)
            if k!='raw' and g[:len(root)]!=root:raise ValueError('Changed five-view protected outputs')
            identity=tuple(tuple(x) for x in g)
            if identity not in cache:cache[identity]=X.evaluate(X.P.objects(g,changed),gt)[:3].tolist()
            if cache[identity]!=row['counts'][k] or cache[identity]!=state['counts'][k]:raise ValueError('Native five-view XML count differs')
            totals[k]+=cache[identity]
        if n==1 or n%200==0:print(f'Five-view independent native audit {n}/1655',flush=True)
    if {k:X.metrics(v) for k,v in totals.items()}!=r['splits']['all']:raise ValueError('Five-view audit aggregate differs')
    X.write(rp('audit'),dict(complete=True,frames=1655,report_sha256=X.sha(rp('full')),fresh_native_observables=True,
        gt_free_paths_replayed=True,native_XML_counts=True,source_exclusive=True,protected_final_unchanged=True,
        audit_source_sha256=X.sha(Path(__file__))))


def report():
    c=config();r=X.read(rp('full'));a=X.read(rp('audit'))
    if not a['complete'] or a['report_sha256']!=X.sha(rp('full')):raise ValueError('Unverified five-view results')
    keys=[k for k,v in r['selection'].items() if v is not None]
    gates={k:all(r['splits'][p][k]['precision']>=.9 for p in ('all','train','development','guard','late','guard_late')) for k in keys}
    lines=['# 三目恢复机制迁移至五目：完整受控实验（2026-10-17）','',
        '五目GT24,845，固定原五目最终输出、原0.05裁剪检测缓存、原校准框；不是三目全图960输入，也不能把三目43%与五目43%直接比较。检测器冻结，新增后处理头在601拟合，200开发锁点后再1655及独立原生XML审核。','',
        'capacity529：当前场景条件新增性与0/1/3/6状态，取消此前最多两次新增限制。residual575另加五相机剩余观测支持/竞争46维。verification613再加ROI竞争14维、仅Cam234的独立锚点预测24维。Cam1/5继续用原针孔—Mei几何配对证据，不套三目平移或共线公式。三头完全相同训练样本/标签/180树，只有证据不同。','',
        '保留原最终组/框，未用源独立输出，不将多个备选框塞进一组。每次追加更新真实当前场景，目标为一对一匹配容量的边际增益，不只是“框附近有没有GT”。全域包含拟合，同路线反复探索，上游不完整嵌套OOF，不宣称物理身份或独立场景保证。','']
    for part,title in (('all','全域1655，含拟合'),('guard','Guard299'),('late','Late555'),('guard_late','后段854')):
        lines += [f'## {title}','','| 方法 | TP | FP | FN | P | R | F1 |','|---|---:|---:|---:|---:|---:|---:|']
        for k in ['baseline','raw']+keys:
            z=r['splits'][part][k];lines.append(f"| {k} | {z['tp']} | {z['fp']} | {z['fn']} | {z['precision']:.4%} | {z['recall']:.4%} | {z['f1']:.4%} |")
        lines.append('')
    later=[row for row in r['rows'] if row['frame'] in c['guard']+c['late']];ci={k:X.paired_bootstrap(later,'counts',k,'baseline') for k in keys}
    lines += ['## P90与论文边界','']+[f'- {k}: 各已测分段P90={ok}。' for k,ok in gates.items()]
    lines += ['','全部开发预锁点保留，包括分段不合格。区间为同路线25帧块探索性bootstrap，不校正反复研究。新增计算尚未端到端验收不降速，未替换默认入口、论文主表或code。可作为扩展实验/消融候选，不能自动宣布最终部署或独创理论。','']
    md=X.BASE/f'algorithm/FIVE_CAMERA_CAPACITY_RECOVERY_{SUFFIX}.md'
    with md.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    X.write(rp('evidence'),dict(complete=True,full_sha256=X.sha(rp('full')),audit_sha256=X.sha(rp('audit')),report_sha256=X.sha(md),
        gates=gates,guard_late_intervals=ci,default_replaced=False))
    print(str(md),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('init','prepare_early','prepare_later','fit','development','full','audit','report','early','later'))
    stage=p.parse_args().stage
    if stage=='early':init();prepare('early');fit();evaluate('development')
    elif stage=='later':prepare('later');evaluate('full');audit();report()
    elif stage.startswith('prepare_'):prepare(stage.split('_')[1])
    elif stage in ('development','full'):evaluate(stage)
    else:globals()[stage]()
