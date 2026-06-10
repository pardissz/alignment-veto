"""
Bar chart: T3 WVS agreement - T2 WVS agreement (at 75% threshold) per model.
Positive = model's latent distribution agrees MORE with WVS on Tier 3 (sensitive)
          than Tier 2 (moderate) during refusals → suppression signature.
Negative = genuine Tier 3 confusion.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = '/shared/storage-01/users/zahraei2/mena_normal/results'

df = pd.read_csv(os.path.join(RESULTS, 'exp3_suppression.csv'))
df75 = df[df['threshold'] == 0.75].copy()

piv = df75.pivot_table(index='model', columns='tier',
                        values=['agreement_rate', 'ci_lo', 'ci_hi']).reset_index()
piv.columns = ['_'.join(str(c) for c in col).strip('_') for col in piv.columns]
piv = piv.dropna(subset=['agreement_rate_2', 'agreement_rate_3'])

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
    'qwen3_4b_instruct':      'Qwen3-4B',
    'qwen3_30b_a3b_instruct': 'Qwen3-30B',
    'tulu_3_8b_sft':          'Tulu3-8B-SFT',
    'tulu_3_8b_dpo':          'Tulu3-8B-DPO',
    'tulu_3.1_8b':            'Tulu3.1-8B',
}

def family_color(m, positive):
    base = {
        'olmo':    ('#1565C0', '#90CAF9'),
        'tulu':    ('#6A1B9A', '#CE93D8'),
        'llama':   ('#2E7D32', '#A5D6A7'),
        'gemma':   ('#E65100', '#FFCC80'),
        'gpt':     ('#B71C1C', '#EF9A9A'),
        'qwen':    ('#00838F', '#80DEEA'),
        'aya':     ('#AD1457', '#F48FB1'),
        'fanar':   ('#37474F', '#B0BEC5'),
        'jais':    ('#37474F', '#B0BEC5'),
        'allam':   ('#37474F', '#B0BEC5'),
        'mistral': ('#4E342E', '#BCAAA4'),
    }
    for key, (dark, light) in base.items():
        if key in m:
            return dark if positive else light
    return ('#555555', '#BBBBBB')[0 if positive else 1]

piv['label'] = piv['model'].map(label_map).fillna(piv['model'])
piv['delta'] = piv['agreement_rate_3'] - piv['agreement_rate_2']

# propagate CI via sqrt(se2^2 + se3^2) (independent bootstrap CIs → approx)
piv['se2'] = (piv['ci_hi_2'] - piv['ci_lo_2']) / (2 * 1.96)
piv['se3'] = (piv['ci_hi_3'] - piv['ci_lo_3']) / (2 * 1.96)
piv['delta_err'] = 1.96 * np.sqrt(piv['se2']**2 + piv['se3']**2)

piv = piv.sort_values('delta', ascending=True).reset_index(drop=True)
piv['color'] = [family_color(m, d > 0) for m, d in zip(piv['model'], piv['delta'])]

n = len(piv)
fig, ax = plt.subplots(figsize=(9, max(6, n * 0.38)))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

y = np.arange(n)
bars = ax.barh(y, piv['delta'], color=piv['color'], height=0.65,
               edgecolor='white', linewidth=0.4, zorder=3)

# Error bars
ax.errorbar(piv['delta'], y,
            xerr=piv['delta_err'], fmt='none',
            ecolor='#333333', elinewidth=0.9, capsize=3, alpha=0.6, zorder=4)

# Zero line
ax.axvline(0, color='#333333', lw=1.2, zorder=5)

# Shade positive region lightly
xlim = max(abs(piv['delta'].min()), abs(piv['delta'].max())) + 0.06
ax.axvspan(0, xlim, alpha=0.04, color='#1565C0', zorder=0)
ax.axvspan(-xlim, 0, alpha=0.04, color='#B71C1C', zorder=0)

# Value labels
for i, (_, row) in enumerate(piv.iterrows()):
    sign = '+' if row['delta'] >= 0 else ''
    x_label = row['delta'] + row['delta_err'] + 0.005 if row['delta'] >= 0 \
               else row['delta'] - row['delta_err'] - 0.005
    ha = 'left' if row['delta'] >= 0 else 'right'
    ax.text(x_label, i, f"{sign}{row['delta']:.3f}",
            va='center', ha=ha, fontsize=7.5,
            color='#1A237E' if row['delta'] > 0 else '#7F0000')

ax.set_yticks(y)
ax.set_yticklabels(piv['label'], fontsize=8.5)
ax.set_xlabel('T3 WVS Agreement − T2 WVS Agreement  (75% threshold)', fontsize=10.5)
ax.set_title('Extra latent WVS agreement on Tier 3 vs Tier 2 during refusals\n'
             'Positive = model hides more accurate cultural knowledge on sensitive topics',
             fontsize=11)

ax.set_xlim(-xlim, xlim)
ax.grid(axis='x', lw=0.4, alpha=0.5, color='#cccccc', zorder=0)

# Annotations
ax.text(xlim * 0.97, n - 0.5, 'Suppression\nsignature\n(knows but hides)',
        ha='right', va='top', fontsize=8, color='#1565C0',
        style='italic', alpha=0.75)
ax.text(-xlim * 0.97, 0.5, 'Genuine Tier 3\nuncertainty',
        ha='left', va='bottom', fontsize=8, color='#B71C1C',
        style='italic', alpha=0.75)

# Count summary
n_pos = (piv['delta'] > 0).sum()
ax.text(0.01, 0.01,
        f'{n_pos}/{n} models: T3 agreement > T2 agreement',
        transform=ax.transAxes, fontsize=8.5, va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#aaa', alpha=0.9))

fig.tight_layout()
out = os.path.join(RESULTS, 'exp3_suppression_delta.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'Saved: {out}')

print('\nModel                    T2      T3    Delta')
for _, row in piv.sort_values('delta', ascending=False).iterrows():
    print(f"  {row['label']:<22}  {row['agreement_rate_2']:.3f}   {row['agreement_rate_3']:.3f}  "
          f"{'+'if row['delta']>0 else ''}{row['delta']:.3f}")
