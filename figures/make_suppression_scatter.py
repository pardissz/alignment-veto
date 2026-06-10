"""
Tier 2 vs Tier 3 WVS agreement scatter plot (identity-line plot).
Uses Exp 3 data at 75% concentration threshold.
Points above the diagonal = model has HIGHER T3 latent agreement than T2
= "hides accurate cultural knowledge on sensitive questions."
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = '/shared/storage-01/users/zahraei2/mena_normal/results'

df = pd.read_csv(os.path.join(RESULTS, 'exp3_suppression.csv'))
df75 = df[df['threshold'] == 0.75].copy()

# Pivot: rows = model, cols = tier
piv = df75.pivot_table(index='model', columns='tier',
                        values=['agreement_rate', 'ci_lo', 'ci_hi', 'n_above'])
piv.columns = ['_'.join(str(c) for c in col) for col in piv.columns]
piv = piv.reset_index()

# Keep only models that have both T2 and T3
piv = piv.dropna(subset=['agreement_rate_2', 'agreement_rate_3'])

# --- clean model labels ---
label_map = {
    'allam_7b_instruct':      'ALLAM-7B',
    'aya_expanse_8b':         'AYA-8B',
    'aya_expanse_32b':        'AYA-32B',
    'fanar_1_9b_instruct':    'FANAR-9B',
    'gemma_3_4b_it':          'Gemma3-4B',
    'gemma_3_12b_it':         'Gemma3-12B',
    'gemma_3_27b_it':         'Gemma3-27B',
    'gpt4o_mini':             'GPT-4o-mini',
    'gpt_4o_mini':            'GPT-4o-mini',
    'gpt_5':                  'GPT-5',
    'jais_2_8b_chat':         'JAIS-8B',
    'llama_3.1_8b_base':      'Llama3.1-8B-Base',
    'llama_3.1_8b_instruct':  'Llama3.1-8B-IT',
    'mistral_7b_instruct':    'Mistral-7B',
    'olmo_3_7b_base':         'OLMo3-7B-Base',
    'olmo_3_7b_sft':          'OLMo3-7B-SFT',
    'olmo_3_7b_dpo':          'OLMo3-7B-DPO',
    'olmo_3_7b_instruct':     'OLMo3-7B-IT',
    'olmo_3_32b_base':        'OLMo3-32B-Base',
    'olmo_3_32b_sft':         'OLMo3-32B-SFT',
    'olmo_3_32b_dpo':         'OLMo3-32B-DPO',
    'olmo_3_32b_instruct':    'OLMo3-32B-IT',
    'qwen2.5_7b_instruct':    'Qwen2.5-7B',
    'qwen3_4b':               'Qwen3-4B',
    'qwen3_30b':              'Qwen3-30B',
    'qwen3_30b_a3b_instruct': 'Qwen3-30B',
    'qwen3_4b_instruct':      'Qwen3-4B',
    'tulu_3_8b_sft':          'Tulu3-8B-SFT',
    'tulu_3_8b_dpo':          'Tulu3-8B-DPO',
    'tulu_3.1_8b':            'Tulu3.1-8B',
    'tulu_3.1_8b_instruct':   'Tulu3.1-8B',
}

piv['label'] = piv['model'].map(label_map).fillna(piv['model'])

# --- colour scheme: model family ---
def family_color(m):
    if 'olmo' in m:     return '#1565C0'   # blue
    if 'tulu' in m:     return '#6A1B9A'   # purple
    if 'llama' in m:    return '#2E7D32'   # green
    if 'gemma' in m:    return '#F57F17'   # amber
    if 'gpt' in m:      return '#B71C1C'   # red
    if 'qwen' in m:     return '#00838F'   # teal
    if 'aya' in m:      return '#AD1457'   # pink
    if 'mistral' in m:  return '#4E342E'   # brown
    if 'allam' in m or 'jais' in m or 'fanar' in m: return '#37474F'  # grey-blue
    return '#757575'

piv['color'] = piv['model'].apply(family_color)

# Error bars: half-width of CI
piv['xerr'] = (piv['ci_hi_2'] - piv['ci_lo_2']) / 2
piv['yerr'] = (piv['ci_hi_3'] - piv['ci_lo_3']) / 2

# ── Main figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))

ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

# Identity line
lim_min = max(0, min(piv['agreement_rate_2'].min(), piv['agreement_rate_3'].min()) - 0.03)
lim_max = min(1, max(piv['agreement_rate_2'].max(), piv['agreement_rate_3'].max()) + 0.06)
ax.plot([lim_min, lim_max], [lim_min, lim_max],
        color='#555555', lw=1.4, ls='--', zorder=1, label='y = x  (T2 = T3 agreement)')

# Shade regions
ax.fill_between([lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max],
                alpha=0.04, color='#1565C0', zorder=0)
ax.fill_between([lim_min, lim_max], [lim_min, lim_min], [lim_min, lim_max],
                alpha=0.04, color='#B71C1C', zorder=0)

# Error bars (thin, behind points)
for _, row in piv.iterrows():
    ax.errorbar(row['agreement_rate_2'], row['agreement_rate_3'],
                xerr=row['xerr'], yerr=row['yerr'],
                fmt='none', ecolor=row['color'], elinewidth=0.8, capsize=2,
                alpha=0.5, zorder=2)

# Scatter
ax.scatter(piv['agreement_rate_2'], piv['agreement_rate_3'],
           c=piv['color'], s=90, zorder=3, linewidths=0.6, edgecolors='white')

# Labels — offset to avoid overlap
offset = {
    'OLMo3-7B-Base':   ( 0.005,  0.012),
    'OLMo3-7B-SFT':    ( 0.005, -0.014),
    'OLMo3-7B-DPO':    ( 0.005,  0.012),
    'OLMo3-7B-IT':     (-0.005,  0.012),
    'OLMo3-32B-Base':  ( 0.005,  0.012),
    'OLMo3-32B-SFT':   ( 0.005, -0.013),
    'OLMo3-32B-DPO':   (-0.005,  0.012),
    'OLMo3-32B-IT':    ( 0.005, -0.013),
    'Tulu3-8B-SFT':    ( 0.005,  0.012),
    'Tulu3-8B-DPO':    ( 0.005, -0.013),
    'Tulu3.1-8B':      (-0.060,  0.010),
    'Llama3.1-8B-Base':( 0.005,  0.012),
    'Llama3.1-8B-IT':  ( 0.005, -0.013),
    'GPT-4o-mini':     ( 0.005,  0.012),
    'GPT-5':           ( 0.005,  0.012),
    'Gemma3-27B':      ( 0.005,  0.012),
    'Gemma3-12B':      ( 0.005, -0.013),
    'Gemma3-4B':       ( 0.005,  0.012),
    'ALLAM-7B':        ( 0.005,  0.012),
    'AYA-8B':          ( 0.005,  0.012),
    'AYA-32B':         ( 0.005, -0.013),
    'FANAR-9B':        ( 0.005,  0.012),
    'JAIS-8B':         ( 0.005,  0.012),
    'Mistral-7B':      ( 0.005,  0.012),
    'Qwen2.5-7B':      ( 0.005, -0.013),
    'Qwen3-4B':        ( 0.005,  0.012),
    'Qwen3-30B':       ( 0.005,  0.012),
}

for _, row in piv.iterrows():
    lbl = row['label']
    dx, dy = offset.get(lbl, (0.005, 0.010))
    ax.text(row['agreement_rate_2'] + dx, row['agreement_rate_3'] + dy,
            lbl, fontsize=7.5, color=row['color'],
            va='bottom' if dy >= 0 else 'top', ha='left', zorder=4)

# Annotation arrows for shaded regions
ax.annotate('Hides accurate\ncultural knowledge\n(T3 > T2)',
            xy=(0.61, 0.64), fontsize=8, color='#1565C0', alpha=0.7,
            ha='center', style='italic')
ax.annotate('Genuine T3\nuncertainty\n(T3 < T2)',
            xy=(0.60, 0.18), fontsize=8, color='#B71C1C', alpha=0.7,
            ha='center', style='italic')

# Legend for model families
legend_items = [
    mpatches.Patch(color='#1565C0', label='OLMo-3 family'),
    mpatches.Patch(color='#6A1B9A', label='Tulu-3 family'),
    mpatches.Patch(color='#2E7D32', label='Llama-3.1'),
    mpatches.Patch(color='#F57F17', label='Gemma-3'),
    mpatches.Patch(color='#B71C1C', label='GPT'),
    mpatches.Patch(color='#00838F', label='Qwen'),
    mpatches.Patch(color='#AD1457', label='AYA'),
    mpatches.Patch(color='#37474F', label='Arabic-spec. (ALLAM/JAIS/FANAR)'),
    mpatches.Patch(color='#4E342E', label='Mistral'),
    plt.Line2D([0],[0], color='#555555', lw=1.4, ls='--', label='y = x  (T2 = T3)'),
]
ax.legend(handles=legend_items, fontsize=7, loc='lower right',
          framealpha=0.9, edgecolor='#cccccc')

ax.set_xlabel('Tier 2 WVS Agreement Rate (75% threshold)', fontsize=11)
ax.set_ylabel('Tier 3 WVS Agreement Rate (75% threshold)', fontsize=11)
ax.set_title('Does the model hide accurate cultural knowledge on sensitive questions?\n'
             'Points above the diagonal: higher latent T3 accuracy than T2 (suppression signature)',
             fontsize=10.5)

ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_aspect('equal')
ax.grid(True, lw=0.4, alpha=0.5, color='#cccccc')

# Fraction above diagonal
above = (piv['agreement_rate_3'] > piv['agreement_rate_2']).sum()
total = len(piv)
ax.text(0.03, 0.97,
        f'{above}/{total} models above diagonal\n(T3 latent agreement > T2)',
        transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#aaa', alpha=0.9))

fig.tight_layout()
out = os.path.join(RESULTS, 'exp3_t2_vs_t3_scatter.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'Saved: {out}')

# ── Also save a cleaner version without labels for compact figure ─────────────
fig2, ax2 = plt.subplots(figsize=(6, 5.5))
ax2.set_facecolor('#FAFAFA')
fig2.patch.set_facecolor('white')
ax2.plot([lim_min, lim_max], [lim_min, lim_max],
         color='#555555', lw=1.4, ls='--', zorder=1)
ax2.fill_between([lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max],
                 alpha=0.05, color='#1565C0', zorder=0)
ax2.fill_between([lim_min, lim_max], [lim_min, lim_min], [lim_min, lim_max],
                 alpha=0.05, color='#B71C1C', zorder=0)
for _, row in piv.iterrows():
    ax2.errorbar(row['agreement_rate_2'], row['agreement_rate_3'],
                 xerr=row['xerr'], yerr=row['yerr'],
                 fmt='none', ecolor=row['color'], elinewidth=0.7, capsize=2,
                 alpha=0.45, zorder=2)
ax2.scatter(piv['agreement_rate_2'], piv['agreement_rate_3'],
            c=piv['color'], s=65, zorder=3, linewidths=0.6, edgecolors='white')

for _, row in piv.iterrows():
    dx, dy = offset.get(row['label'], (0.004, 0.008))
    ax2.text(row['agreement_rate_2'] + dx, row['agreement_rate_3'] + dy,
             row['label'], fontsize=6, color=row['color'],
             va='bottom' if dy >= 0 else 'top', ha='left', zorder=4)

ax2.set_xlabel('Tier 2 WVS Agreement Rate (75% threshold)', fontsize=10)
ax2.set_ylabel('Tier 3 WVS Agreement Rate (75% threshold)', fontsize=10)
ax2.set_title('Suppression signature: T2 vs T3 latent WVS agreement', fontsize=10)
ax2.set_xlim(lim_min, lim_max); ax2.set_ylim(lim_min, lim_max)
ax2.set_aspect('equal')
ax2.grid(True, lw=0.4, alpha=0.4, color='#cccccc')
ax2.legend(handles=legend_items, fontsize=6.5, loc='lower right',
           framealpha=0.9, edgecolor='#cccccc')
ax2.text(0.03, 0.97, f'{above}/{total} above diagonal',
         transform=ax2.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#aaa', alpha=0.9))
fig2.tight_layout()
out2 = os.path.join(RESULTS, 'exp3_t2_vs_t3_scatter_compact.pdf')
fig2.savefig(out2, bbox_inches='tight', dpi=200)
plt.close(fig2)
print(f'Saved: {out2}')

# ── Print summary table ──────────────────────────────────────────────────────
print('\nModel              T2 agree  T3 agree  T3>T2?')
for _, row in piv.sort_values('agreement_rate_3', ascending=False).iterrows():
    flag = '↑ suppression' if row['agreement_rate_3'] > row['agreement_rate_2'] else '↓ genuine gap'
    print(f"  {row['label']:<22} {row['agreement_rate_2']:.3f}     {row['agreement_rate_3']:.3f}   {flag}")
print(f'\n{above}/{total} models above diagonal at 75% threshold.')
