"""Three-camera BAND/depth/E1 display + GT statistics; --no-eval disables GT.

Add --no-band for the current no-BAND depth/E1 viewer, or --legacy for the old baseline.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core_algorithms'))
def main():
    if '--no-band' in sys.argv and '--legacy' in sys.argv:
        import argparse
        argparse.ArgumentParser().error('--no-band and --legacy select different pipelines; choose one')
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        from final_viewer import main as run
    elif '--no-band' in sys.argv:
        sys.argv.remove('--no-band')
        from no_band_depth_viewer import main as run
    else:
        from band_depth_viewer import main as run
    run()
if __name__=='__main__':main()
