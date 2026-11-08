"""Render blind audit sheets (10 cases per sheet) from native images. Resumable; Usage: python3 e4_render_audit.py CASES OUTDIR BUDGET_S"""
import sys, json, time, cv2, numpy as np
from pathlib import Path
cases = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
budget = float(sys.argv[3]); t0 = time.time(); ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT/'ultralytics/mycar/images'; H = 170
for k in range(0, len(cases), 10):
    p = out/f'sheet_{k//10+1:02d}.png'
    if p.exists(): continue
    if time.time()-t0 > budget: break
    rows = []
    for c in cases[k:k+10]:
        tiles = []
        for m in c['members']:
            im = cv2.imread(str(IMG/f"{m['src']}/{c['frame']}.png"))
            x1, y1, x2, y2 = m['box']; w, h = x2-x1, y2-y1
            a, b = int(max(0, x1-.3*w)), int(max(0, y1-.3*h)); cc, d = int(min(im.shape[1], x2+.3*w)), int(min(im.shape[0], y2+.3*h))
            crop = im[b:d, a:cc].copy(); cv2.rectangle(crop, (int(x1-a), int(y1-b)), (int(x2-a), int(y2-b)), (0, 0, 255), 2)
            s = H/max(crop.shape[0], 1); crop = cv2.resize(crop, (max(1, int(crop.shape[1]*s)), H))
            crop = crop[:, :min(crop.shape[1], 360)]
            cv2.putText(crop, f"Cam{m['src']}", (4, 16), 0, .5, (0, 255, 255), 1); tiles.append(crop)
            tiles.append(np.full((H, 6, 3), 255, np.uint8))
        label = np.full((H, 80, 3), 40, np.uint8); cv2.putText(label, c['case'], (6, 90), 0, .7, (255, 255, 255), 2)
        row = np.hstack([label]+tiles); rows.append(row)
    W = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.hstack([r, np.full((H, W-r.shape[1], 3), 255, np.uint8)]) for r in rows] )
    cv2.imwrite(str(p), sheet)
print('sheets', len(list(out.glob('sheet_*.png'))), 'of', (len(cases)+9)//10)
