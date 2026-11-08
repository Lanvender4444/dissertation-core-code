"""Read-only GT evaluation, separate from the frozen prediction pipeline."""
import hashlib
import json
from pathlib import Path

import numpy as np

from final_pipeline import filter_ego, SINGLE_THRESHOLDS as PLAIN_THRESHOLDS
from validate_multiview import ev, evaluate, metrics

CAMERAS = (2, 3, 4)
# Frozen BAND single-camera thresholds from global_fusion_protocol.json.
SINGLE_THRESHOLDS = {2: .25, 3: .20, 4: .25}
PROTOCOL = ('Cam234 union GT by instanceId; any-view height >=25px; compatible '
            'class IoU >=0.5; Hungarian maximum-cardinality one-to-one; '
            'any fused member may match; small-only GT matches ignored.')


class RunEvaluation:
    def __init__(self, gt_root, names, cameras, method, detset='BAND'):
        if detset not in ('BAND', '960'):
            raise ValueError('Unsupported evaluation detection set')
        self.gt_root = Path(gt_root).resolve()
        self.names = list(names)
        self.cameras = cameras
        self.method = method
        self.detset = detset
        self.single_thresholds = dict(SINGLE_THRESHOLDS if detset == 'BAND' else PLAIN_THRESHOLDS)
        self.single_methods = {c: f'{detset}/cam{c}' for c in CAMERAS}
        self.rows = {}
        self.methods = [self.single_methods[c] for c in CAMERAS] + [method]
        self.totals = {m: np.zeros(5, int) for m in self.methods}
        self.native_totals = {self.single_methods[c]: np.zeros(5, int) for c in CAMERAS}
        # parse_xml returns [] for missing files: forbid that silent empty-GT path.
        for name in self.names:
            for c in CAMERAS:
                path = self.xml_path(name, c)
                if not path.is_file():
                    raise FileNotFoundError(f'Missing GT: {path}. Supply --gt-root or use --no-eval.')

    def xml_path(self, name, camera):
        return self.gt_root / f'cam_{camera}' / f'{Path(name).stem}.xml'

    def evaluate_frame(self, name, detections, objects):
        # Revisiting a frame does not change either its saved statistics or totals.
        if name in self.rows:
            return self.rows[name]
        if name not in self.names:
            raise ValueError(f'Unplanned evaluation frame: {name}')
        gt = {}
        hashes = {}
        for c in CAMERAS:
            path = self.xml_path(name, c)
            if not path.is_file():
                raise FileNotFoundError(f'Missing GT: {path}')
            hashes[str(c)] = hashlib.sha256(path.read_bytes()).hexdigest()
            gt[c] = [g for g in ev.parse_xml(path) if g[1] in ev.GT_COMPAT]
        kept, _ = filter_ego(detections, self.cameras)
        predictions = {self.single_methods[c]: [dict(dets=[d], src=[c], cls=d['cls'], conf=d['conf'])
                       for d in kept if d['src'] == c and d['conf'] >= self.single_thresholds[c]]
                       for c in CAMERAS}
        predictions[self.method] = objects
        counts = {m: evaluate(predictions[m], gt).tolist() for m in self.methods}
        native = {self.single_methods[c]: evaluate(predictions[self.single_methods[c]], {c: gt[c]}).tolist()
                  for c in CAMERAS}
        row = dict(frame=name, gt_sha256=hashes, counts=counts, native_counts=native,
                   methods={m: metrics(v) for m, v in counts.items()},
                   native_singles={m: metrics(v) for m, v in native.items()})
        self.rows[name] = row
        for m, value in counts.items():
            self.totals[m] += value
        for m, value in native.items():
            self.native_totals[m] += value
        return row

    def summary(self, status='stopped'):
        return dict(method=self.method, detset=self.detset, protocol=PROTOCOL, gt_root=str(self.gt_root),
                    single_thresholds=self.single_thresholds, planned_frames=len(self.names),
                    evaluated_frames=len(self.rows), complete_selection=len(self.rows) == len(self.names),
                    status=status, evaluated_frame_names=list(self.rows),
                    methods={m: metrics(v) for m, v in self.totals.items()},
                    native_singles={m: metrics(v) for m, v in self.native_totals.items()},
                    note='Totals cover unique visited frames only, not necessarily the full dataset. '
                         'P/R/F1 measure object hits, not group identity accuracy or mAP. '
                         'Cross-camera GT-ID semantics and synchronization remain assumptions; '
                         'this route has been used in development, not independent blind testing.')

    def print_progress(self, row, revisit=False):
        print(f"[EVAL] {row['frame']} {'REVISIT (not counted again)' if revisit else 'NEW'} "
              f'| cumulative unique frames={len(self.rows)}/{len(self.names)} | Cam234 union GT', flush=True)
        for m in self.methods:
            frame = row['methods'][m]
            total = metrics(self.totals[m])
            print(f"  {m:<22} frame TP={frame['tp']} FP={frame['fp']} FN={frame['fn']} "
                  f"P={frame['precision']:.2%} R={frame['recall']:.2%} F1={frame['f1']:.2%} "
                  f"| cumulative TP={total['tp']} FP={total['fp']} FN={total['fn']} "
                  f"P={total['precision']:.2%} R={total['recall']:.2%} F1={total['f1']:.2%}", flush=True)

    def finish(self, output, status):
        summary = self.summary(status)
        path = Path(output) / 'metrics.json'
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n[EVAL SUMMARY] {status} | {len(self.rows)}/{len(self.names)} unique frames "
              '| Cam234 union GT (not classification Accuracy)', flush=True)
        for m, s in summary['methods'].items():
            print(f"  {m:<22} TP={s['tp']} FP={s['fp']} FN={s['fn']} "
                  f"Precision={s['precision']:.2%} Recall={s['recall']:.2%} F1={s['f1']:.2%}", flush=True)
        print(f'[EVAL] Native-camera results and protocol also saved: {path}', flush=True)
