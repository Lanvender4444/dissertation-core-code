"""Vector figures for the supplementary experiments (thesis v1.0). Reads only results JSON."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'ultralytics/experiments/supp_20261101'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else R/'figures'
OUT.mkdir(parents=True, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#1f2328', '#5b636b', '#e3e6ea'
MK = ['o', 's', '^', 'D', 'v']
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'legend.frameon': False,
                     'lines.linewidth': 2, 'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight'); fig.savefig(OUT/f'{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


e1 = json.loads((R/'e1_e6_error_analysis/run01/results.json').read_text())
f = e1['full']['tau0.25']
H = ['25-50', '50-100', '100-200', '>=200']; HL = ['25–50', '50–100', '100–200', '≥200']

# E1-1 recall by height per camera
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for k, cam in enumerate((1, 2, 3, 4, 5)):
    ys = [next(x['recall'] for x in f['size'] if x['camera'] == cam and x['bin'] == b) for b in H]
    ax.plot(range(4), np.array(ys)*100, color=C[k], marker=MK[k], ms=6, ls='--' if cam in (1, 5) else '-',
            label=f'Cam{cam} ({"fisheye" if cam in (1, 5) else "pinhole"})')
ax.set_xticks(range(4), HL); ax.set_xlabel('Annotated box height in native pixels'); ax.set_ylabel('Own-view recall (%)')
ax.set_ylim(0, 85); ax.legend(ncol=2, fontsize=8, loc='upper left')
save(fig, 'e1_recall_by_height')

# E1-2 radial effect, height-stratified
RB = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8']
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
for ax, kind, title in zip(axs, ('pinhole', 'fisheye'), ('Pinhole Cam2–4', 'Fisheye Cam1, Cam5')):
    for k, h in enumerate(H):
        ys = []
        for r in RB:
            x = next((x for x in f['radial_by_height'] if x['kind'] == kind and x['rbin'] == r and x['hbin'] == h), None)
            ys.append(x['recall']*100 if x and x['n'] >= 50 else np.nan)
        ax.plot(range(4), ys, color=C[k], marker=MK[k], ms=5, label=f'height {HL[k]} px')
    ax.set_title(title, fontsize=9, color=INK); ax.set_xticks(range(4), ['0–.2', '.2–.4', '.4–.6', '.6–.8'])
axs[0].set_ylabel('Own-view recall (%)'); axs[0].set_ylim(0, 95)
fig.supxlabel('Normalised radial distance of box centre from principal point', fontsize=9, y=-0.02)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=4, fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, 1.07))
save(fig, 'e1_radial_by_height')

# E1-3 conditional miss heatmap (sequential single hue)
M = np.full((5, 5), np.nan)
for x in f['conditional_miss']:
    if x['n'] >= 100:
        M[x['i']-1, x['j']-1] = x['p_miss']*100
fig, ax = plt.subplots(figsize=(3.6, 3.1))
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('blue', ['#eef4fc', '#2a78d6', '#0d3d75'])
im = ax.imshow(M, cmap=cm, vmin=0, vmax=60)
for i in range(5):
    for j in range(5):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=8, color='white' if M[i, j] > 35 else INK)
ax.set_xticks(range(5), [f'Cam{c}' for c in range(1, 6)]); ax.set_yticks(range(5), [f'Cam{c}' for c in range(1, 6)])
ax.set_xlabel('Camera j (missed?)'); ax.set_ylabel('Camera i (detected)'); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04); cb.set_label('P(miss in j | detected in i) (%)', fontsize=8)
save(fig, 'e1_conditional_miss')

# E6 recall by size for Raw/A/F5P/F5R
e6 = e1['full']['e6']; cfg = [('raw', 'Raw pool (box-corrected)'), ('A', 'Reference A'), ('F5P', 'F5-P'), ('F5R', 'F5-R')]
fig, ax = plt.subplots(figsize=(5.8, 2.9)); w = .2
for k, (c, lab) in enumerate(cfg):
    ys = [next(x['recall'] for x in e6 if x['config'] == c and x['kind'] == 'size' and x['bin'] == b)*100 for b in H]
    ax.bar(np.arange(4)+(k-1.5)*w, ys, w*.9, color=C[k], label=lab)
ax.set_xticks(range(4), HL); ax.set_xlabel('Maximum annotated box height over views (px)'); ax.set_ylabel('Recall (%)')
ax.set_ylim(0, 100); ax.legend(ncol=2, fontsize=8, loc='upper left'); ax.grid(axis='x', visible=False)
save(fig, 'e6_recall_by_size')

# PR curves: baselines + fused outputs (full sequence, five-view GT)
pr = json.loads((R/'e2_simple_baselines/run01/pr_curves.json').read_text())
ap = json.loads((R/'e5_ap/run01/group_ap.json').read_text())
fig, ax = plt.subplots(figsize=(5.6, 3.4))
b1 = pr['B1']; b2 = [v for k, v in pr.items() if k.startswith('B2')][0]
for k, (curve, lab) in enumerate(((ap['B1']['full']['curve50'], 'B1 detections, ranked by confidence'),
                                  (ap['B2']['full']['curve50'], 'B2 ray NMS (θ=5°), ranked'),
                                  (ap['F5P']['full']['curve50'], 'F5-P outputs, ranked'),
                                  (ap['F5R']['full']['curve50'], 'F5-R outputs, ranked'))):
    ax.plot([p['recall']*100 for p in curve], [p['precision']*100 for p in curve], color=C[k], lw=1.6,
            ls=['-', '--', '-', '-'][k], label=lab)
ax.scatter([42.69], [90.45], marker='*', s=140, color=C[2], edgecolor='white', zorder=5)
ax.scatter([44.85], [90.84], marker='*', s=160, color=C[3], edgecolor='white', zorder=5)
ax.annotate('F5-R locked point\nP 90.84, R 44.85', (44.85, 90.84), (50, 74), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
ax.axhline(90, color=MUTED, lw=.8, ls=':'); ax.text(1, 91.2, 'P = 90%', fontsize=7.5, color=MUTED)
ax.set_xlim(0, 60); ax.set_ylim(0, 102); ax.set_xlabel('Recall (%) — five-view GT, 24,845 instances'); ax.set_ylabel('Precision (%)')
ax.legend(fontsize=7.5, loc='lower left')
save(fig, 'e2_e5_pr_curves')

# E3 ablation (if available)
arms = ['none', 'no_appearance', 'no_geometry', 'no_crossview', 'no_camera_id', 'detector_only']
names = {'none': 'Full residual head', 'no_appearance': 'Without appearance (ROI, HSV)', 'no_geometry': 'Without observation-ray geometry',
         'no_crossview': 'Without cross-view support/competition', 'no_camera_id': 'Without camera identity',
         'detector_only': 'Detector/box evidence only'}
rows = []
for a in arms:
    p = R/f'e3_feature_ablation/{a}/run01/results.json'
    if p.exists():
        rows.append((a, json.loads(p.read_text())))
if len(rows) > 1:
    base = rows[0][1]
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    labs = [names[a] for a, _ in rows[1:]][::-1]
    dr = [(r['full']['recall']-base['full']['recall'])*100 for _, r in rows[1:]][::-1]
    dp = [(r['full']['precision']-base['full']['precision'])*100 for _, r in rows[1:]][::-1]
    y = np.arange(len(labs))
    ax.barh(y+.19, dr, .36, color=C[0], label='Δ recall'); ax.barh(y-.19, dp, .36, color=C[1], label='Δ precision')
    for yy, v in zip(y, dr):
        ax.text(0.08, yy+.19, f'{v:+.2f}', va='center', ha='left', fontsize=7.5, color=INK)
    ax.axvline(0, color=MUTED, lw=.8); ax.set_yticks(y, labs); ax.set_xlabel('Change vs. full head, percentage points (full sequence, dev-locked τ)')
    ax.set_xlim(min(min(dr), min(dp))-0.25, max(0.9, max(dp)+0.2))
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, 1.16), ncol=2); ax.grid(axis='y', visible=False)
    save(fig, 'e3_feature_ablation')
print('figures ->', OUT)
