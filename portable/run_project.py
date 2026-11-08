"""Launch matching full project entry points when the external dataset/cache exists.

This is explicitly NOT the standalone prepared-feature example. It refuses a
project whose source snapshot no longer matches and never invokes a shell.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from verify import verify

ENTRIES={
 'three-no-band':'ultralytics/decided/final_three_cam_no_band.py',
 'three-band':'ultralytics/decided/final_three_cam.py',
 'cam2':'ultralytics/decided/final_cam2.py','cam3':'ultralytics/decided/final_cam3.py',
 'cam4':'ultralytics/decided/final_cam4.py','five':'ultralytics/decided/final_five_cam.py',
 'spatial':'ultralytics/decided/final_spatial_cam.py','immersive':'ultralytics/decided/final_immersive_cam.py'}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--entry',choices=ENTRIES,default='five')
    ap.add_argument('--check-only',action='store_true');ap.add_argument('arguments',nargs=argparse.REMAINDER)
    args=ap.parse_args();root=args.project_root.resolve();verify(root)
    if args.check_only:print('Snapshot and external project agree; no viewer or detector executed.')
    else:
        if not (root/'ultralytics/mycar/images').is_dir():ap.error('External dataset is missing; use demo.py for bundled prepared features')
        extra=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
        raise SystemExit(subprocess.call([sys.executable,str(root/ENTRIES[args.entry]),*extra],cwd=root))
