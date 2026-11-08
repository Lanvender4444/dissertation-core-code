"""Cam4 YOLO only, with the same ego exclusion as fusion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core_algorithms'))
from final_viewer import main
if __name__=='__main__':main(4)
