"""Run Windows-authored frozen protocols under Linux: normalise backslash paths.
Read-only shim; does not change any file or hash."""
import json, pathlib

_orig = pathlib.PurePath._parse_args.__func__
def _pa(cls, args):
    return _orig(cls, [a.replace('\\', '/') if isinstance(a, str) else a for a in args])
pathlib.PurePath._parse_args = classmethod(_pa)

def _fix(o):
    if isinstance(o, dict):
        return {(k.replace('\\', '/') if isinstance(k, str) else k): _fix(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_fix(v) for v in o]
    return o
_loads = json.loads
def loads(s, *a, **k):
    return _fix(_loads(s, *a, **k))
json.loads = loads
_load = json.load
def load(fp, *a, **k):
    return loads(fp.read(), *a, **k)
json.load = load
