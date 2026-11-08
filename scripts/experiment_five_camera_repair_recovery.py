"""Frozen risk/disagreement repair THEN frozen transferred recovery heads.

Exploratory composition motivated by first transfer's precision loss. No
refitting or test-label gate selection; dev200 locks before full1655. Old
negative study and original final outputs remain untouched.
"""
import argparse
import copy
from pathlib import Path
import numpy as np
import experiment_five_camera_capacity_recovery as P
import experiment_shared_edit_uncertainty as U

X=P.X
OUT=X.BASE/'experiments/five_camera_repair_recovery_20261017/run01'
SUFFIX='20261017_v1'


def rp(stage):return X.REPORTS/f'five_camera_repair_recovery_{stage}_{SUFFIX}.json'


def init():
    if OUT.exists():raise FileExistsError('Retain prior repair/recovery composition')
    c=P.config();full=X.REPORTS/'shared_edit_full_20261016_v1.json';audit=X.REPORTS/'shared_edit_independent_audit_20261016_v1.json';a=X.read(audit)
    if not a['complete'] or a['report_sha256']!=X.sha(full):raise ValueError('Unverified existing risk repair')
    a=X.read(P.rp('audit'))
    if not a['complete'] or a['report_sha256']!=X.sha(P.rp('full')):raise ValueError('Unverified first transfer')
    sources=[Path(__file__),Path(P.__file__),Path(P.S.__file__),Path(U.__file__),X.BASE/'research/shared_edit_uncertainty.py']
    deps=[P.OUT/'protocol.json',P.OUT/'models.json',P.rp('full'),P.rp('audit'),full,audit,U.OUT/'protocol.json',U.OUT/'models.json',U.OUT/'development_lock.json']
    X.write(OUT/'protocol.json',dict(**{k:c[k] for k in ('frames','train','development','guard','late','heads','floors')},path_floor=.3,
        sources={str(p.relative_to(X.ROOT)):X.sha(p) for p in sources},dependencies={str(p.relative_to(X.ROOT)):X.sha(p) for p in deps},
        method='Reuse previously dev-locked robust gain/harm/disagreement repair, lambda1/tau.05/kappa1, at most1 edit/frame. '
            'Then SAME newly fitted capacity529/residual575/verification613 heads from first transfer. No new fitting. '
            'Recondition on repaired scene; refresh independent predictive anchors if a two-forward-camera root is retired. '
            'Retired observations may be reconsidered as separate unused sources, never silently reattached.',
        selection='Exact dev200 curves down to preset score.3, P90/91/92/93; no feasible point stored null. '
            'Full uses identical accepted prefixes, cutoff min locked tau. All same-model no-repair controls retained from audited first study.',
        limitations='Exploratory composition AFTER seeing first transfer full precision failures; NOT blind confirmation. '
            'Original pipeline/upstream calibration and initial repair not newly cross-fitted. Heads trained on original roots, '
            'not falsely described as refit on repaired states. All1655 includes fitting, same repeatedly explored route. '
            'No physical-ID or deployment-speed guarantee. No default/thesis master/code snapshot changes.'))
    print('Five-view repair/recovery composition locked, NO new fitting',flush=True)


def config():
    c=X.read(OUT/'protocol.json');X.check(c['sources']);X.check(c['dependencies']);return c


class Inputs:
    def __init__(self):
        self.c=config();self.inputs=P.R.ProtectedInputs();self.repair=X.read(X.REPORTS/'shared_edit_full_20261016_v1.json')
        self.old=X.read(X.REPORTS/'protected_conditional_full_results_20261010.json');self.setting=self.repair['settings']['robust'];self.rr={r['frame']:r for r in self.repair['rows']}

    def frame(self,s):
        data,lab,m=P.load(s,'replay');op=U.OUT/f'evaluation_full/{s}.json'
        state=P.R.E.checked(op,self.repair['frame_sha256'][str(op.relative_to(X.ROOT))]);mp=U.PARENT/f'prepared/replay/{s}.json';meta=X.read(mp)
        if X.sha(mp)!=state['parent_meta_sha256'] or meta['scene']!=m['scenes']['control']:raise ValueError('Risk repair original root differs')
        n=len(meta['pairs']);gain=np.asarray(state['gain']).reshape(n,5);harm=np.asarray(state['harm']).reshape(n,5)
        index,unused=U.U.choose(gain,harm,**self.setting)
        if index!=state['decisions']['robust']:raise ValueError('Risk repair GT-free decision differs')
        original=m['scenes']['control'];root=[list(g) for g in original];refresh=False
        if index is not None:
            i,j=meta['pairs'][index];u=meta['residual'][j];lookup={int(u):k for k,u in enumerate(data['uids'])}
            refresh=sum(int(data['cameras'][lookup[u]]) in (2,3,4) for u in root[i])>=2
            root=[g for k,g in enumerate(root) if k!=i]+[[u]]
        current=copy.deepcopy(m);current['scenes']['control']=root
        if refresh:
            base,unused,unused2=P.parent(s,'replay');aux,proof=P.extra(s,'replay',base,current,self.inputs,self.old);data=P.S.attach(base,**aux)
        # Offline consistency check only; annotations never choose index/root.
        if P.C.counts(root,data,lab)!=self.rr[s]['counts']['robust'][:3]:raise ValueError('Saved risk repair counts differ')
        return data,lab,current,original,root


def run(domain):
    inp=Inputs();c=inp.c;doc=X.read(P.OUT/'models.json');models={h:X.native_predictor(v) for h,v in doc['models'].items()}
    lock=X.read(OUT/'development_lock.json') if domain=='full' else None
    if lock and (lock['protocol_sha256']!=X.sha(OUT/'protocol.json') or lock['model_sha256']!=X.sha(P.OUT/'models.json')):raise ValueError('Changed composition lock')
    floor=c['path_floor'] if lock is None else min([v['tau'] for v in lock['selection'].values() if v is not None],default=1.0000001)
    names=c['development'] if domain=='development' else c['frames'];frames=[];rows=[];proof={}
    for n,s in enumerate(names,1):
        data,lab,m,original,root=inp.frame(s);paths={f'control/{h}':P.S.rollout(data,root,h,models[h],floor) for h in c['heads']}
        counts=dict(baseline=P.C.counts(original,data,lab),repair_only=P.C.counts(root,data,lab));outputs=dict(baseline=original,repair_only=root)
        if domain=='development':frames.append(dict(data=data,lab=lab,meta=m,paths=paths))
        else:
            for k,v in lock['selection'].items():
                if v is None:continue
                p=paths[f"control/{v['head']}"];g=root+[[u] for u,z in zip(p[0],p[1]) if z>=v['tau']];counts[k]=P.C.counts(g,data,lab);outputs[k]=g
            # Independent native features, repaired-anchor refresh and paths.
            fresh,unused,meta,orig2,root2=inp.frame(s)
            if root2!=root:raise ValueError('Fresh repaired root differs')
            freshpaths={f'control/{h}':P.S.rollout(fresh,root2,h,models[h],floor) for h in c['heads']}
            if freshpaths!=paths:raise ValueError('Fresh composition GT-free path differs')
            pool,changed,desc,flags,crops,src=inp.inputs.load(s);base=inp.inputs
            source=X.read(P.R.ORIGINAL/f'evaluation_full/frames/{s}.json');q={tuple(map(int,a)):float(b) for a,b in zip(source['pairs'],source['pair_scores'])}
            rich=P.R.C.object_features([[int(u)] for u in fresh['uids']],[],pool,base.cameras,desc,flags,q,changed)
            if not np.allclose(rich,fresh['phi'],rtol=0,atol=1e-12):raise ValueError('Native composition evidence differs')
            gt=X.truth(s,(1,2,3,4,5));cache={}
            for k,g in outputs.items():
                X.P.validate(g,pool);identity=tuple(tuple(a) for a in g)
                if identity not in cache:cache[identity]=X.evaluate(X.P.objects(g,changed),gt)[:3].tolist()
                if cache[identity]!=counts[k]:raise ValueError('Native composition XML counts differ')
        fp=OUT/f'evaluation_{domain}/{s}.json';X.write(fp,dict(frame=s,paths=paths,counts=counts,root=root));proof[str(fp.relative_to(X.ROOT))]=X.sha(fp);rows.append(dict(frame=s,counts=counts))
        if n==1 or n%200==0:print(f'Five-view repair+recovery {domain} {n}/{len(names)}',flush=True)
    if domain=='development':
        selection={};curves={}
        for h in c['heads']:
            selected,points=P.C.curves(frames,'control',h,c['floors']);curves[h]=points
            for f,z in selected.items():selection[f'{h}/P{f}']=dict(head=h,**z) if z else None
        X.write(OUT/'development_lock.json',dict(selection=selection,curves=curves,protocol_sha256=X.sha(OUT/'protocol.json'),model_sha256=X.sha(P.OUT/'models.json')))
        X.write(rp('development'),dict(complete=True,frames=200,rows=rows,selection=selection,frame_sha256=proof));print(str(selection),flush=True)
    else:
        parts={p:set(c[p]) for p in ('train','development','guard','late')};parts['guard_late']=parts['guard']|parts['late']
        splits={p:{k:X.metrics(np.sum([r['counts'][k] for r in rows if p=='all' or r['frame'] in parts[p]],axis=0)) for k in rows[0]['counts']} for p in ('all',*parts)}
        for k,v in lock['selection'].items():
            if v is not None and splits['development'][k]!={f:v[f] for f in ('tp','fp','fn','precision','recall','f1')}:raise ValueError('Composition locked development mismatch')
        if splits['all']['baseline']['tp']!=10606 or splits['all']['repair_only']['tp']!=10737:raise ValueError('Wrong five-view roots')
        X.write(rp('full'),dict(complete=True,frames=1655,rows=rows,selection=lock['selection'],splits=splits,frame_sha256=proof,
            model_sha256=X.sha(P.OUT/'models.json'),protocol_sha256=X.sha(OUT/'protocol.json'),lock_sha256=X.sha(OUT/'development_lock.json')))
        X.write(rp('audit'),dict(complete=True,frames=1655,report_sha256=X.sha(rp('full')),fresh_native_features=True,
            gt_free_repaired_roots_and_paths=True,native_XML_counts=True,audit_source_sha256=X.sha(Path(__file__))))


def report():
    c=config();r=X.read(rp('full'));a=X.read(rp('audit'));parent=X.read(P.rp('full'))
    if not a['complete'] or a['report_sha256']!=X.sha(rp('full')):raise ValueError('Unaudited repair/recovery')
    keys=[k for k,v in r['selection'].items() if v is not None];gates={k:all(r['splits'][p][k]['precision']>=.9 for p in ('all','train','development','guard','late','guard_late')) for k in keys}
    lines=['# 五目：固定风险修复后再恢复（2026-10-17）','',
        '第一轮直接追加的三个分支全域P均不足90%，负结果不覆盖。本轮复用早已开发锁定的风险/波动修复，再应用完全相同的三个恢复头；不拟合新模型。修复改变当前场景时重算竞争；删除含至少两个前向源的旧组时刷新独立预测锚点，不拿已退役组继续证明自己。','',
        '五目GT24,845和原.05裁剪/ROI/校准框固定。开发200在预声明路径下界.3以上选择P90/91/92/93点，再全1655及原生独立复核。组合是在看到第一轮精度失败后设计，同路线探索，不是盲测。现有上游并非完整嵌套OOF；恢复头没有重新拟合到修复后的状态，不冒充重新训练。','']
    for part,title in (('all','全域1655，含拟合'),('guard','Guard299'),('late','Late555'),('guard_late','后段854')):
        entries=[(k,r['splits'][part][k]) for k in ('baseline','repair_only')]+[(f'without_repair/{k}',v) for k,v in parent['splits'][part].items() if k not in ('baseline','raw')]+[(k,r['splits'][part][k]) for k in keys]
        lines += [f'## {title}','','| 方法 | TP | FP | FN | P | R | F1 |','|---|---:|---:|---:|---:|---:|---:|']
        for k,z in entries:lines.append(f"| {k} | {z['tp']} | {z['fp']} | {z['fn']} | {z['precision']:.4%} | {z['recall']:.4%} | {z['f1']:.4%} |")
        lines.append('')
    later=[row for row in r['rows'] if row['frame'] in c['guard']+c['late']];ci={k:X.paired_bootstrap(later,'counts',k,'baseline') for k in keys};ci_repair={k:X.paired_bootstrap(later,'counts',k,'repair_only') for k in keys}
    lines += ['## 已测P90检查','']+[f'- {k}: 各已测分段P90={ok}。' for k,ok in gates.items()]
    lines += ['','保留同模型无修复控制与修复独立控制；不能把全部组合增益归于新增恢复或波动项。后段25帧成对区间仅探索性，不校正多轮搜索；没有端到端不降速或物理身份认证。默认入口、LaTeX主表与code快照未替换。','']
    md=X.BASE/'algorithm/FIVE_CAMERA_REPAIR_RECOVERY_20261017_v1.md'
    with md.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    X.write(rp('evidence'),dict(complete=True,full_sha256=X.sha(rp('full')),audit_sha256=X.sha(rp('audit')),report_sha256=X.sha(md),
        gates=gates,guard_late_vs_original=ci,guard_late_vs_repair=ci_repair,default_replaced=False))
    print(str(md),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('init','development','full','report','early','later'));a=ap.parse_args()
    if a.stage=='early':init();run('development')
    elif a.stage=='later':run('full');report()
    elif a.stage in ('development','full'):run(a.stage)
    else:globals()[a.stage]()
