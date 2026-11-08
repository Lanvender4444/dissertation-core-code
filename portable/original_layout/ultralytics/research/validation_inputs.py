"""Read-only signature checks shared by new post-hoc association studies."""
import json
from pathlib import Path

from experiment_global_fusion import sha,EXPECTED_MODEL
from band_depth_viewer import FrozenBandCache

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'ultralytics';REPORTS=BASE/'algorithm/evaluation_reports'


def load_stage1_verified():
    parent=BASE/'experiments/topic3_stage1_20261006_shards';configs=[];index={}
    for i in range(4):
        p=parent/f'shard{i}';cfg=json.loads((p/'manifest.json').read_text());c=json.loads((p/'completion.json').read_text())
        if cfg['limit'] or cfg['shard_index']!=i or cfg['shard_count']!=4 or c['done']!=c['planned'] or c['done']!=len(cfg['frames']):
            raise ValueError('Incomplete first-stage parents')
        if configs and any(cfg[k]!=configs[0][k] for k in ('params','available_frames','source_sha256','baseline_protocol_sha256','development_frames')):
            raise ValueError('Mixed parents')
        for n,h in cfg['source_sha256'].items():
            if sha(ROOT/n)!=h:raise ValueError('Parent code changed')
        for files in cfg['input_signatures'].values():
            for n,sig in files.items():
                st=(ROOT/n).stat()
                if [st.st_size,st.st_mtime_ns]!=sig:raise ValueError('Data changed')
        index.update({s:p/'frames'/f'{s}.json' for s in cfg['frames']});configs.append(cfg)
    cfg=configs[0];names=sorted(index)
    if names!=cfg['available_frames'] or sum(len(c['frames']) for c in configs)!=len(names):raise ValueError('Missing/duplicate domain')
    protocol_path=REPORTS/'global_fusion_protocol.json';protocol=json.loads(protocol_path.read_text())
    if sha(protocol_path)!=cfg['baseline_protocol_sha256']:raise ValueError('Protocol changed')
    for n,h in protocol['code_sha256'].items():
        if sha(BASE/'core_algorithms'/n)!=h:raise ValueError('Frozen core changed')
    for n,h in protocol['calibration_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Calibration changed')
    if sha(BASE/'yolo11n.pt')!=EXPECTED_MODEL:raise ValueError('Weights changed')
    cache=FrozenBandCache(BASE/'experiments/global_fusion_20260930_shards',EXPECTED_MODEL,cfg['params'],
        [s+'.png' for s in names],{c:BASE/f'mycar/images/{c}' for c in (2,3,4)})
    return cfg,index,cache,[sha(parent/f'shard{i}/manifest.json') for i in range(4)]
