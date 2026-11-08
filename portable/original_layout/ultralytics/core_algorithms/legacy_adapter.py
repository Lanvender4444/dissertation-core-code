"""Adapter preserving the existing front-camera algorithm for comparisons/runtime."""
import copy
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('front_legacy',Path(__file__).with_name('tri-compare.py'))
core=importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

def legacy(dets,rect):
    ds=copy.deepcopy(dets)
    for d in ds: d['rbox']=rect.rect_box(d['box'],d['src'])
    objs,by=core.compare_and_classify(rect,ds)
    kept,_=core.admission(objs)
    def adapt(o):
        return dict(dets=[by[c][i] for c,i in o['members'].items()],src=sorted(o['src']),conf=o['conf'],label=o['label'])
    return [adapt(o) for o in kept],[adapt(o) for o in objs]
