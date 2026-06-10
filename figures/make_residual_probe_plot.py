"""
Re-plots fix1_residual_probe.pdf from the pre-computed CSV (no re-computation).
Layout: 3 panels — top-left (r2_total), top-right (r2_t3_total), bottom full-width (r2_t3_resid).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

df = pd.read_csv(os.path.join(RESULTS, 'fix1_residual_probe_results.csv'))

MODELS = {
    'olmo_3_7b_sft':      {'label': 'OLMo-7B-SFT',  'color': '#7986CB'},
    'olmo_3_7b_dpo':      {'label': 'OLMo-7B-DPO',  'color': '#3F51B5'},
    'olmo_3_7b_instruct': {'label': 'OLMo-7B-IT',   'color': '#1A237E'},
    'tulu_3_8b_sft':      {'label': 'Tulu3-8B-SFT', 'color': '#F3A046'},
    'tulu_3_8b_dpo':      {'label': 'Tulu3-8B-DPO', 'color': '#E67C13'},
    'tulu_3_8b_instruct': {'label': 'Tulu3-8B-IT',  'color': '#C65D00'},
    'llama_3.1_8b_instruct': {'label': 'LLaMA-8B-IT', 'color': '#2E7D32'},
}

# Get question-only baseline from existing data as metadata fallback
r2_question_baseline = 0.666  # known WVS baseline; also stored during original run

fig = plt.figure(figsize=(13, 9))
fig.patch.set_facecolor('white')
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
ax_tl  = fig.add_subplot(gs[0, 0])
ax_tr  = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, :])

top_panels = [
    (ax_tl, 'r2_total',    'Standard probe (all tiers)',  True),
    (ax_tr, 'r2_t3_total', 'Standard T3 probe',           True),
]

for ax, metric, title, show_baseline in top_panels:
    ax.set_facecolor('#FAFAFA')
    for mk, info in MODELS.items():
        sub = df[df['model'] == mk].sort_values('layer')
        if len(sub) == 0:
            continue
        y_vals = sub[metric].values
        valid_l = ~np.isnan(y_vals)
        if valid_l.sum() < 2:
            continue
        ax.plot(sub['layer'].values[valid_l], y_vals[valid_l],
                color=info['color'], lw=1.8, label=info['label'],
                marker='o', ms=3, alpha=0.9)
    ax.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
    if show_baseline:
        ax.axhline(r2_question_baseline, color='#E53935', lw=1.2, ls=':', alpha=0.7,
                   label=f'Q-only baseline ({r2_question_baseline:.3f})')
    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('Probe R²', fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(-0.15, 0.85)
    ax.grid(lw=0.3, alpha=0.4, color='#ccc')

ax_tl.legend(fontsize=6.5, framealpha=0.9, ncol=2)
ax_tr.legend(fontsize=6.5, framealpha=0.9, loc='upper left',
             handles=[plt.Line2D([],[],color='#E53935',ls=':',lw=1.2,
                                  label=f'Q-only baseline ({r2_question_baseline:.3f})')])

# Bottom panel: residualized T3 probe (full width)
ax_bot.set_facecolor('#FAFAFA')
for mk, info in MODELS.items():
    sub = df[df['model'] == mk].sort_values('layer')
    if len(sub) == 0:
        continue
    y_vals = sub['r2_t3_resid'].values
    valid_l = ~np.isnan(y_vals)
    if valid_l.sum() < 2:
        continue
    ax_bot.plot(sub['layer'].values[valid_l], y_vals[valid_l],
                color=info['color'], lw=1.8, label=info['label'],
                marker='o', ms=3, alpha=0.9)
ax_bot.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
ax_bot.axvspan(28, 32, alpha=0.12, color='#E53935', label='Erasure zone (layers 28–32)')
ax_bot.text(30.0, 0.06, 'erasure\nzone', fontsize=7.5, color='#E53935',
            ha='center', va='bottom', alpha=0.85)
ax_bot.set_xlabel('Layer', fontsize=9)
ax_bot.set_ylabel('Probe R²', fontsize=9)
ax_bot.set_title('Residualized T3 probe — pure country-specific cultural encoding\n'
                 '(per-question mean subtracted; erasure zone = layers 28–32)', fontsize=9)
ax_bot.set_ylim(-0.15, 0.85)
ax_bot.grid(lw=0.3, alpha=0.4, color='#ccc')
ax_bot.legend(fontsize=6.5, framealpha=0.9, ncol=4)

fig.suptitle(
    'Probing Confound Control: Standard vs. Residualized\n'
    f'Standard probe R² includes question-level baseline ({r2_question_baseline:.3f}); '
    'residualized probe isolates country-specific cultural signal',
    fontsize=10.5)
out = os.path.join(RESULTS, 'fix1_residual_probe.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'Saved: {out}')
