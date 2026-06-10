"""
Generates 4 figures that were created by a now-deleted script:
  1. gating_vs_erasing_pipeline.pdf
  2. flagship_gating_scatter.pdf
  3. suppression_cost.pdf
  4. size_vs_wvs_lift.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

refusal_df = pd.read_csv(os.path.join(RESULTS, 'exp2a_refusal.csv'))
nvas_df    = pd.read_csv(os.path.join(RESULTS, 'exp2b_nvas.csv'))
supp_idx   = pd.read_csv(os.path.join(RESULTS, 'exp11_suppression_index.csv'))
supp_df    = pd.read_csv(os.path.join(RESULTS, 'exp3_suppression.csv'))

LABEL_MAP = {
    'allam_7b_instruct':      'ALLAM-7B',
    'aya_expanse_8b':         'AYA-8B',
    'aya_expanse_32b':        'AYA-32B',
    'fanar_1_9b_instruct':    'FANAR-9B',
    'gemma_3_4b_it':          'Gemma3-4B',
    'gemma_3_12b_it':         'Gemma3-12B',
    'gemma_3_27b_it':         'Gemma3-27B',
    'gpt4o_mini':             'GPT-4o-mini',
    'gpt_5':                  'GPT-5',
    'jais_2_8b_chat':         'JAIS-8B',
    'llama_3.1_8b_base':      'LLaMA-8B-Base',
    'llama_3.1_8b_instruct':  'LLaMA-8B-IT',
    'olmo_3_7b_base':         'OLMo-7B-Base',
    'olmo_3_7b_sft':          'OLMo-7B-SFT',
    'olmo_3_7b_dpo':          'OLMo-7B-DPO',
    'olmo_3_7b_instruct':     'OLMo-7B-IT',
    'olmo_3_32b_base':        'OLMo-32B-Base',
    'olmo_3_32b_sft':         'OLMo-32B-SFT',
    'olmo_3_32b_dpo':         'OLMo-32B-DPO',
    'olmo_3_32b_instruct':    'OLMo-32B-IT',
    'qwen2.5_7b_instruct':    'Qwen2.5-7B',
    'qwen3_4b_instruct':      'Qwen3-4B',
    'qwen3_30b_a3b_instruct': 'Qwen3-30B',
    'tulu_3_8b_sft':          'Tulu3-SFT',
    'tulu_3_8b_dpo':          'Tulu3-DPO',
    'tulu_3.1_8b':            'Tulu3.1-8B',
}

def fam_color(m):
    if 'olmo' in m:   return '#1565C0'
    if 'tulu' in m:   return '#6A1B9A'
    if 'llama' in m:  return '#2E7D32'
    if 'gemma' in m:  return '#F57F17'
    if 'gpt' in m:    return '#B71C1C'
    if 'qwen' in m:   return '#00838F'
    if 'aya' in m:    return '#AD1457'
    if 'allam' in m or 'jais' in m or 'fanar' in m: return '#37474F'
    return '#757575'

def fam_type(m):
    if m in ('olmo_3_7b_base','olmo_3_32b_base','llama_3.1_8b_base'): return 'base'
    if m in ('olmo_3_7b_sft','olmo_3_7b_dpo','olmo_3_32b_sft','olmo_3_32b_dpo',
             'tulu_3_8b_sft','tulu_3_8b_dpo'): return 'train'
    return 'instruct'

# ─────────────────────────────────────────────────────────────────────────────
# 1. gating_vs_erasing_pipeline.pdf
# ─────────────────────────────────────────────────────────────────────────────
print('=== 1. gating_vs_erasing_pipeline ===')

OLMO7  = ['olmo_3_7b_base',  'olmo_3_7b_sft',  'olmo_3_7b_dpo',  'olmo_3_7b_instruct']
OLMO32 = ['olmo_3_32b_base', 'olmo_3_32b_sft', 'olmo_3_32b_dpo', 'olmo_3_32b_instruct']
TULU   = ['llama_3.1_8b_base', 'tulu_3_8b_sft', 'tulu_3_8b_dpo', 'tulu_3.1_8b']
X_LABELS = ['Base', 'SFT', 'DPO', 'RLVR (IT)']

PIPELINE_SPECS = [
    ('OLMo-7B',  OLMO7,  '#1565C0'),
    ('OLMo-32B', OLMO32, '#0D47A1'),
    ('Tulu-8B',  TULU,   '#6A1B9A'),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('white')

for panel_i, (ax, metric_col, ylabel, title_sfx) in enumerate([
    (axes[0], 'refusal_rate', 'Refusal rate', 'T3 Refusal Rate'),
    (axes[1], 'nvas',         'Accepted NVAS', 'T3 Accepted NVAS'),
]):
    ax.set_facecolor('#FAFAFA')
    src = refusal_df if metric_col == 'refusal_rate' else nvas_df
    for name, stages, color in PIPELINE_SPECS:
        avail = [s for s in stages if s in src['model'].values]
        if not avail:
            continue
        vals, los, his = [], [], []
        for s in avail:
            sub = src[(src['model']==s) & (src['tier']==3)]
            if len(sub):
                vals.append(sub[metric_col].values[0])
                los.append(sub['ci_lo'].values[0])
                his.append(sub['ci_hi'].values[0])
            else:
                vals.append(np.nan); los.append(np.nan); his.append(np.nan)
        xs = range(len(avail))
        ax.plot(xs, vals, marker='o', color=color, lw=2.2, label=name, ms=7)
        ax.fill_between(xs, los, his, alpha=0.15, color=color)
        # Map stage position to X_LABELS
        x_lbls_avail = []
        for s in avail:
            idx = stages.index(s)
            x_lbls_avail.append(X_LABELS[idx])
        ax.set_xticks(range(len(avail)))
        ax.set_xticklabels(x_lbls_avail, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'Training Pipeline: {title_sfx}', fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(lw=0.4, alpha=0.4, color='#ccc')
    ax.legend(fontsize=9, framealpha=0.9)

fig.suptitle('Gating vs Erasing: T3 Refusal Rate and Cultural Accuracy Along Training Pipelines',
             fontsize=12)
fig.tight_layout()
out = os.path.join(RESULTS, 'gating_vs_erasing_pipeline.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'  Saved: {out}')

# ─────────────────────────────────────────────────────────────────────────────
# 2. flagship_gating_scatter.pdf
# ─────────────────────────────────────────────────────────────────────────────
print('=== 2. flagship_gating_scatter ===')

all_models = sorted(refusal_df['model'].unique())
rows = []
for mk in all_models:
    r1 = refusal_df[(refusal_df['model']==mk) & (refusal_df['tier']==1)]
    r3 = refusal_df[(refusal_df['model']==mk) & (refusal_df['tier']==3)]
    n3 = nvas_df[(nvas_df['model']==mk) & (nvas_df['tier']==3)]
    n1 = nvas_df[(nvas_df['model']==mk) & (nvas_df['tier']==1)]
    if len(r1) and len(r3) and len(n3):
        rows.append({
            'model': mk,
            'label': LABEL_MAP.get(mk, mk),
            'safety_tax': float(r3['refusal_rate'].values[0]) - float(r1['refusal_rate'].values[0]),
            'nvas_t3': float(n3['nvas'].values[0]),
            'nvas_t1': float(n1['nvas'].values[0]) if len(n1) else np.nan,
        })
scatter_df = pd.DataFrame(rows)

mean_t1_nvas = scatter_df['nvas_t1'].mean()

FAM_COLORS = {'base': '#90A4AE', 'train': '#7986CB', 'instruct': '#42A5F5',
              'frontier': '#FF6F00', 'mena': '#E53935'}
MARKERS = {'base': 's', 'train': '^', 'instruct': 'o', 'frontier': '*', 'mena': 'D'}

def model_fam_type(mk):
    if mk in ('gpt_5', 'gpt4o_mini'):    return 'frontier'
    if mk in ('aya_expanse_8b', 'aya_expanse_32b', 'allam_7b_instruct',
              'fanar_1_9b_instruct', 'jais_2_8b_chat'): return 'mena'
    return fam_type(mk)

fig, ax = plt.subplots(figsize=(12, 8)); fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

if not np.isnan(mean_t1_nvas):
    ax.axhline(mean_t1_nvas, color='#555', lw=1.2, ls=':', alpha=0.7,
               label=f'Mean T1 accepted NVAS ({mean_t1_nvas:.3f})')
ax.axvline(0, color='#555', lw=0.8, ls='--', alpha=0.4)

# Jitter offsets to avoid label overlap
label_offsets = {
    'OLMo-7B-Base': (-5, 6), 'OLMo-7B-SFT': (5, 4), 'OLMo-7B-DPO': (5, -9),
    'OLMo-7B-IT': (5, 4), 'OLMo-32B-Base': (-5, 6), 'OLMo-32B-IT': (5, 4),
    'GPT-5': (5, 5), 'GPT-4o-mini': (5, -9), 'LLaMA-8B-Base': (-5, 6),
    'Tulu3.1-8B': (5, -9), 'ALLAM-7B': (-5, 6), 'AYA-32B': (5, 4),
}

for _, row in scatter_df.iterrows():
    ft = model_fam_type(row['model'])
    col = FAM_COLORS[ft]
    mk_sym = MARKERS[ft]
    ms = 130 if ft == 'frontier' else 70
    ax.scatter([row['safety_tax']], [row['nvas_t3']], s=ms,
               color=col, marker=mk_sym, zorder=5,
               edgecolors='white', linewidths=0.6)
    dx, dy = label_offsets.get(row['label'], (5, 4))
    ax.annotate(row['label'], (row['safety_tax'], row['nvas_t3']),
                fontsize=6.5, xytext=(dx, dy), textcoords='offset points',
                color='#333', zorder=6)

handles = [mpatches.Patch(color=FAM_COLORS[k], label=k.title())
           for k in FAM_COLORS]
ax.legend(handles=handles, fontsize=8.5, framealpha=0.9, loc='lower right')
ax.set_xlabel('Safety tax (T3 − T1 refusal rate, Persona EN)', fontsize=11)
ax.set_ylabel('T3 accepted NVAS (Persona EN)', fontsize=11)
ax.set_title('Safety Tax vs Cultural Accuracy: All 26 Models', fontsize=12)
ax.grid(lw=0.3, alpha=0.4, color='#ccc')
fig.tight_layout()
out = os.path.join(RESULTS, 'flagship_gating_scatter.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'  Saved: {out}')

# ─────────────────────────────────────────────────────────────────────────────
# 3. suppression_cost.pdf
# ─────────────────────────────────────────────────────────────────────────────
print('=== 3. suppression_cost ===')

# accepted T3 NVAS
acc_nvas = nvas_df[nvas_df['tier']==3][['model','nvas']].rename(columns={'nvas':'acc_nvas'})
# EV-NVAS on refused T3: accepted_nvas + suppression_index
sup_t3 = supp_idx[supp_idx['tier']==3][['model','suppression_index','ci_lo','ci_hi']]
cost_df = acc_nvas.merge(sup_t3, on='model', how='inner')
cost_df['ev_nvas_refused'] = cost_df['acc_nvas'] + cost_df['suppression_index']
cost_df['cost'] = cost_df['suppression_index']  # EV-NVAS(refused) - acc_nvas
cost_df['label'] = cost_df['model'].map(LABEL_MAP).fillna(cost_df['model'])
cost_df = cost_df.sort_values('cost', ascending=False).reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
fig.patch.set_facecolor('white')

# Left panel: scatter accepted NVAS vs EV-NVAS on refused rows
lim_vals = []
for _, r in cost_df.iterrows():
    lim_vals.extend([r['acc_nvas'], r['ev_nvas_refused']])
lim_vals = [v for v in lim_vals if not np.isnan(v)]
lmin = min(lim_vals) - 0.02; lmax = max(lim_vals) + 0.02

ax1.set_facecolor('#FAFAFA')
ax1.plot([lmin, lmax], [lmin, lmax], color='#555', lw=1.2, ls='--', zorder=1,
         label='diagonal (equal)')
for _, r in cost_df.iterrows():
    col = fam_color(r['model'])
    ax1.scatter([r['acc_nvas']], [r['ev_nvas_refused']], s=70,
                color=col, zorder=4, edgecolors='white', linewidths=0.5)
    ax1.annotate(r['label'], (r['acc_nvas'], r['ev_nvas_refused']),
                 fontsize=6, xytext=(4, 3), textcoords='offset points',
                 color='#333', zorder=5)
ax1.set_xlabel('Accepted T3 NVAS', fontsize=10)
ax1.set_ylabel('EV-NVAS on refused T3 rows', fontsize=10)
ax1.set_title('Refused rows carry more accurate\ncultural distributions (above diagonal)', fontsize=10)
ax1.set_xlim(lmin, lmax); ax1.set_ylim(lmin, lmax)
ax1.set_aspect('equal')
ax1.legend(fontsize=8, framealpha=0.9)
ax1.grid(lw=0.3, alpha=0.4, color='#ccc')
n_above = (cost_df['ev_nvas_refused'] > cost_df['acc_nvas']).sum()
ax1.text(0.04, 0.97, f'{n_above}/{len(cost_df)} above diagonal',
         transform=ax1.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#aaa', alpha=0.9))

# Right panel: per-model suppression cost bar chart, sorted descending
ax2.set_facecolor('#FAFAFA')
colors_right = [fam_color(m) for m in cost_df['model']]
ys = range(len(cost_df))
bars = ax2.barh(list(ys), cost_df['cost'].values, color=colors_right, alpha=0.85,
                edgecolor='white', linewidth=0.5)
ax2.errorbar(cost_df['cost'].values, list(ys),
             xerr=[cost_df['cost']-cost_df['ci_lo'], cost_df['ci_hi']-cost_df['cost']],
             fmt='none', ecolor='#555', elinewidth=0.8, capsize=2, alpha=0.6)
ax2.axvline(0, color='#555', lw=0.8, ls='--', alpha=0.6)
ax2.set_yticks(list(ys))
ax2.set_yticklabels(cost_df['label'].values, fontsize=7.5)
ax2.set_xlabel('Suppression cost (EV-NVAS refused − accepted NVAS)', fontsize=9)
ax2.set_title('Per-model suppression cost\n(sorted descending)', fontsize=10)
ax2.grid(axis='x', lw=0.3, alpha=0.4, color='#ccc')
ax2.invert_yaxis()

fig.suptitle('Suppression Cost: Refused T3 rows carry more accurate cultural distributions',
             fontsize=11)
fig.tight_layout()
out = os.path.join(RESULTS, 'suppression_cost.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'  Saved: {out}')

# ─────────────────────────────────────────────────────────────────────────────
# 4. size_vs_wvs_lift.pdf
# ─────────────────────────────────────────────────────────────────────────────
print('=== 4. size_vs_wvs_lift ===')

# Parameter counts in billions
PARAM_B = {
    'llama_3.1_8b_base':      8.0,
    'llama_3.1_8b_instruct':  8.0,
    'olmo_3_7b_base':         7.0,
    'olmo_3_7b_sft':          7.0,
    'olmo_3_7b_dpo':          7.0,
    'olmo_3_7b_instruct':     7.0,
    'olmo_3_32b_base':        32.0,
    'olmo_3_32b_sft':         32.0,
    'olmo_3_32b_dpo':         32.0,
    'olmo_3_32b_instruct':    32.0,
    'tulu_3_8b_sft':          8.0,
    'tulu_3_8b_dpo':          8.0,
    'tulu_3.1_8b':            8.0,
    'gemma_3_4b_it':          4.0,
    'gemma_3_12b_it':         12.0,
    'gemma_3_27b_it':         27.0,
    'gpt4o_mini':             8.0,   # approximate
    'qwen2.5_7b_instruct':    7.0,
    'qwen3_4b_instruct':      4.0,
    'qwen3_30b_a3b_instruct': 30.0,
    'aya_expanse_8b':         8.0,
    'aya_expanse_32b':        32.0,
    'allam_7b_instruct':      7.0,
    'fanar_1_9b_instruct':    9.0,
    'jais_2_8b_chat':         8.0,
    # gpt_5 excluded (unknown size)
}

# WVS mass lift from exp3_suppression at threshold=0.75, tier=3
# lift = pool_mass - 0.25 (uniform baseline for 4-option scale)
wvs_sub = supp_df[(supp_df['tier']==3) & (supp_df['threshold']==0.75)][['model','pool_mass']]
wvs_sub = wvs_sub.copy()
wvs_sub['wvs_lift'] = wvs_sub['pool_mass'] - 0.25

size_rows = []
for mk, params in PARAM_B.items():
    sub = wvs_sub[wvs_sub['model']==mk]
    if len(sub):
        size_rows.append({'model': mk, 'params_b': params,
                          'wvs_lift': float(sub['wvs_lift'].values[0]),
                          'label': LABEL_MAP.get(mk, mk)})
size_df = pd.DataFrame(size_rows)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

for _, r in size_df.iterrows():
    ft = fam_type(r['model'])
    col = fam_color(r['model'])
    mk_sym = '^' if ft == 'base' else 'o'
    ax.scatter([np.log10(r['params_b'])], [r['wvs_lift']], s=80,
               color=col, marker=mk_sym, zorder=4,
               edgecolors='white', linewidths=0.6)
    ax.annotate(r['label'], (np.log10(r['params_b']), r['wvs_lift']),
                fontsize=6.5, xytext=(5, 3), textcoords='offset points',
                color='#333', zorder=5)

ax.axhline(0, color='#555', lw=0.8, ls='--', alpha=0.5,
           label='Uniform null (lift = 0)')

# Spearman correlation
valid = size_df.dropna(subset=['params_b','wvs_lift'])
if len(valid) > 3:
    rho, pval = stats.spearmanr(np.log10(valid['params_b']), valid['wvs_lift'])
    ax.text(0.97, 0.05, f'Spearman r={rho:.3f}, p={pval:.3f}',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#aaa', alpha=0.9))
    print(f'  Spearman r={rho:.3f}, p={pval:.3f}')

xtick_vals = [0, np.log10(4), np.log10(8), np.log10(32)]
xtick_lbl  = ['1B', '4B', '8B', '32B']
ax.set_xticks(xtick_vals)
ax.set_xticklabels(xtick_lbl, fontsize=9)

base_patch   = plt.Line2D([0],[0], marker='^', color='gray', ls='', ms=8, label='Base models')
align_patch  = plt.Line2D([0],[0], marker='o', color='gray', ls='', ms=8, label='Aligned models')
ax.legend(handles=[base_patch, align_patch], fontsize=9, framealpha=0.9)

ax.set_xlabel('Model size (approximate parameters)', fontsize=11)
ax.set_ylabel('WVS-mass lift over uniform null (T3 refused rows)', fontsize=11)
ax.set_title('Model size vs WVS knowledge retained in suppressed distributions\n'
             '(T3 refused rows where WVS majority ≠ 1)', fontsize=11)
ax.grid(lw=0.3, alpha=0.4, color='#ccc')
fig.tight_layout()
out = os.path.join(RESULTS, 'size_vs_wvs_lift.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'  Saved: {out}')

print('\nAll 4 figures done.')
