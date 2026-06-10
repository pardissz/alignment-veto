"""
Regenerates 7 improved paper figures:
  1. gating_vs_erasing_pipeline.pdf  — blue/green/purple line colors
  2. suppression_cost.pdf            — bigger scatter dots
  3. gapB_t3_question_taxonomy.pdf   — pastel quadrant backgrounds
  4. flagship_gating_scatter.pdf     — much bigger dots
  5. pca_7_3_native.pdf              — bigger dots
  6. pca_7_4_persona_neutral.pdf     — bigger dots
  7. fix1_residual_probe.pdf         — fixed axis coverage

All data is loaded from pre-computed CSV/pkl files in results/.
Run: python make_fixed_figures.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

# ── shared constants ─────────────────────────────────────────────────────────
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

COUNTRIES = ['Algeria','Egypt','Iran','Iraq','Jordan','Kuwait','Lebanon',
             'Libya','Mauritania','Morocco','Palestine','Qatar',
             'Saudi Arabia','Sudan','Tunisia','Turkey']
LANG_FAMILY = {
    'Arabic':  ['Algeria','Egypt','Iraq','Jordan','Kuwait','Lebanon',
                'Libya','Mauritania','Morocco','Palestine','Qatar',
                'Saudi Arabia','Sudan','Tunisia'],
    'Persian': ['Iran'],
    'Turkish': ['Turkey'],
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

def save(fig, name):
    out = os.path.join(RESULTS, name)
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved: {out}')


# ══════════════════════════════════════════════════════════════════════════════
# 1. gating_vs_erasing_pipeline.pdf  — blue / green / purple lines
# ══════════════════════════════════════════════════════════════════════════════
print('=== 1. gating_vs_erasing_pipeline ===')

refusal_df = pd.read_csv(os.path.join(RESULTS, 'exp2a_refusal.csv'))
nvas_df    = pd.read_csv(os.path.join(RESULTS, 'exp2b_nvas.csv'))

OLMO7  = ['olmo_3_7b_base',  'olmo_3_7b_sft',  'olmo_3_7b_dpo',  'olmo_3_7b_instruct']
OLMO32 = ['olmo_3_32b_base', 'olmo_3_32b_sft', 'olmo_3_32b_dpo', 'olmo_3_32b_instruct']
TULU   = ['llama_3.1_8b_base', 'tulu_3_8b_sft', 'tulu_3_8b_dpo', 'tulu_3.1_8b']
X_LABELS = ['Base', 'SFT', 'DPO', 'RLVR (IT)']

# ← changed colors to blue / green / purple
PIPELINE_SPECS = [
    ('OLMo-7B',  OLMO7,  '#1565C0'),   # blue
    ('OLMo-32B', OLMO32, '#2E7D32'),   # green
    ('Tulu-8B',  TULU,   '#6A1B9A'),   # purple
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('white')

for ax, metric_col, ylabel, title_sfx in [
    (axes[0], 'refusal_rate', 'Refusal rate',    'T3 Refusal Rate'),
    (axes[1], 'nvas',         'Accepted NVAS',   'T3 Accepted NVAS'),
]:
    ax.set_facecolor('#FAFAFA')
    src = refusal_df if metric_col == 'refusal_rate' else nvas_df
    for name, stages, color in PIPELINE_SPECS:
        avail = [s for s in stages if s in src['model'].values]
        if not avail:
            continue
        vals, los, his = [], [], []
        for s in avail:
            sub = src[(src['model'] == s) & (src['tier'] == 3)]
            if len(sub):
                vals.append(sub[metric_col].values[0])
                los.append(sub['ci_lo'].values[0])
                his.append(sub['ci_hi'].values[0])
            else:
                vals.append(np.nan); los.append(np.nan); his.append(np.nan)
        xs = list(range(len(avail)))
        ax.plot(xs, vals, marker='o', color=color, lw=2.4, label=name, ms=8)
        ax.fill_between(xs, los, his, alpha=0.15, color=color)
        x_lbls = [X_LABELS[stages.index(s)] for s in avail]
        ax.set_xticks(xs)
        ax.set_xticklabels(x_lbls, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'Training Pipeline: {title_sfx}', fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(lw=0.4, alpha=0.4, color='#ccc')
    ax.legend(fontsize=9, framealpha=0.9)

fig.suptitle(
    'Gating vs Erasing: T3 Refusal Rate and Cultural Accuracy Along Training Pipelines',
    fontsize=12)
fig.tight_layout()
save(fig, 'gating_vs_erasing_pipeline.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# 2. suppression_cost.pdf  — bigger scatter dots
# ══════════════════════════════════════════════════════════════════════════════
print('=== 2. suppression_cost ===')

supp_idx = pd.read_csv(os.path.join(RESULTS, 'exp11_suppression_index.csv'))

acc_nvas = nvas_df[nvas_df['tier'] == 3][['model','nvas']].rename(columns={'nvas':'acc_nvas'})
sup_t3   = supp_idx[supp_idx['tier'] == 3][['model','sup_idx','ci_lo','ci_hi']]
cost_df  = acc_nvas.merge(sup_t3, on='model', how='inner')
cost_df['ev_nvas_refused'] = cost_df['acc_nvas'] + cost_df['sup_idx']
cost_df['cost']  = cost_df['sup_idx']
cost_df['label'] = cost_df['model'].map(LABEL_MAP).fillna(cost_df['model'])
cost_df = cost_df.sort_values('cost', ascending=False).reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
fig.patch.set_facecolor('white')

lim_vals = [v for pair in zip(cost_df['acc_nvas'], cost_df['ev_nvas_refused'])
            for v in pair if not np.isnan(v)]
lmin = min(lim_vals) - 0.02
lmax = max(lim_vals) + 0.02

ax1.set_facecolor('#FAFAFA')
ax1.plot([lmin, lmax], [lmin, lmax], color='#555', lw=1.2, ls='--', zorder=1,
         label='diagonal (equal)')
for _, r in cost_df.iterrows():
    col = fam_color(r['model'])
    # ← bigger dots: s=70 → s=160
    ax1.scatter([r['acc_nvas']], [r['ev_nvas_refused']], s=160,
                color=col, zorder=4, edgecolors='white', linewidths=0.8)
    ax1.annotate(r['label'], (r['acc_nvas'], r['ev_nvas_refused']),
                 fontsize=6.5, xytext=(5, 3), textcoords='offset points',
                 color='#333', zorder=5)
ax1.set_xlabel('Accepted T3 NVAS', fontsize=10)
ax1.set_ylabel('EV-NVAS on refused T3 rows', fontsize=10)
ax1.set_title('Refused rows carry more accurate\ncultural distributions (above diagonal)',
              fontsize=10)
ax1.set_xlim(lmin, lmax); ax1.set_ylim(lmin, lmax)
ax1.set_aspect('equal')
ax1.legend(fontsize=8, framealpha=0.9)
ax1.grid(lw=0.3, alpha=0.4, color='#ccc')
n_above = (cost_df['ev_nvas_refused'] > cost_df['acc_nvas']).sum()
ax1.text(0.04, 0.97, f'{n_above}/{len(cost_df)} above diagonal',
         transform=ax1.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#aaa', alpha=0.9))

ax2.set_facecolor('#FAFAFA')
colors_right = [fam_color(m) for m in cost_df['model']]
ys   = list(range(len(cost_df)))
ax2.barh(ys, cost_df['cost'].values, color=colors_right, alpha=0.85,
         edgecolor='white', linewidth=0.5)
ax2.errorbar(cost_df['cost'].values, ys,
             xerr=[cost_df['cost'] - cost_df['ci_lo'],
                   cost_df['ci_hi'] - cost_df['cost']],
             fmt='none', ecolor='#555', elinewidth=0.8, capsize=2, alpha=0.6)
ax2.axvline(0, color='#555', lw=0.8, ls='--', alpha=0.6)
ax2.set_yticks(ys)
ax2.set_yticklabels(cost_df['label'].values, fontsize=7.5)
ax2.set_xlabel('Suppression cost (EV-NVAS refused − accepted NVAS)', fontsize=9)
ax2.set_title('Per-model suppression cost\n(sorted descending)', fontsize=10)
ax2.grid(axis='x', lw=0.3, alpha=0.4, color='#ccc')
ax2.invert_yaxis()

fig.suptitle(
    'Suppression Cost: Refused T3 rows carry more accurate cultural distributions',
    fontsize=11)
fig.tight_layout()
save(fig, 'suppression_cost.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# 3. gapB_t3_question_taxonomy.pdf  — pastel quadrant backgrounds
# ══════════════════════════════════════════════════════════════════════════════
print('=== 3. gapB_t3_question_taxonomy ===')

q_stats = pd.read_csv(os.path.join(RESULTS, 'gapB_t3_question_taxonomy.csv'))

REFUSAL_THRESH = 0.30
NVAS_THRESH    = 0.60

mode_colors = {
    'Suppressed\n(high refusal, OK when answered)':        '#E67C13',
    'Double failure\n(high refusal + wrong when answered)':'#B71C1C',
    'Representational\n(answers but wrong)':               '#7B1FA2',
    'Well-aligned\n(low refusal, high NVAS)':              '#2E7D32',
}

fig, ax = plt.subplots(figsize=(6.5, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

XLIM = (-0.02, 0.65)
YLIM = (0.10, 1.00)

# ← pastel quadrant backgrounds
# top-left  (low refusal, high nvas)  = Well-aligned        → light green
# top-right (high refusal, high nvas) = Suppressed          → light yellow
# bottom-left  (low refusal, low nvas)  = Representational  → light lavender
# bottom-right (high refusal, low nvas) = Double failure    → light salmon
quadrants = [
    (XLIM[0], REFUSAL_THRESH, NVAS_THRESH, YLIM[1],  '#E8F5E9'),   # top-left: light green
    (REFUSAL_THRESH, XLIM[1], NVAS_THRESH, YLIM[1],  '#FFFDE7'),   # top-right: light yellow
    (XLIM[0], REFUSAL_THRESH, YLIM[0],    NVAS_THRESH,'#F3E5F5'),  # bottom-left: light lavender
    (REFUSAL_THRESH, XLIM[1], YLIM[0],    NVAS_THRESH,'#FFEBEE'),  # bottom-right: light pink/salmon
]
for x0, x1, y0, y1, c in quadrants:
    ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=c, zorder=0, alpha=1.0)

for mode, grp in q_stats.groupby('failure_mode'):
    ax.scatter(grp['refusal_rate'], grp['nvas_when_answered'],
               color=mode_colors.get(mode, '#888'), s=80, zorder=5,
               edgecolors='white', linewidths=0.6, label=mode)
    for _, row in grp.iterrows():
        ax.annotate(f'Q{int(row["question_id"])}',
                    (row['refusal_rate'], row['nvas_when_answered']),
                    fontsize=6.5, xytext=(3, 2), textcoords='offset points', color='#333')

ax.axvline(REFUSAL_THRESH, color='#999', lw=1.0, ls='--', alpha=0.8, zorder=3)
ax.axhline(NVAS_THRESH,    color='#999', lw=1.0, ls='--', alpha=0.8, zorder=3)

ax.text(0.55, 0.92, 'Suppressed\n(knows, hides)', transform=ax.transAxes,
        fontsize=8.5, color=mode_colors['Suppressed\n(high refusal, OK when answered)'],
        ha='center', va='top', fontweight='bold', alpha=0.85)
ax.text(0.55, 0.10, 'Double failure', transform=ax.transAxes,
        fontsize=8.5, color=mode_colors['Double failure\n(high refusal + wrong when answered)'],
        ha='center', va='bottom', fontweight='bold', alpha=0.85)
ax.text(0.15, 0.10, 'Representational\nbias', transform=ax.transAxes,
        fontsize=8.5, color=mode_colors['Representational\n(answers but wrong)'],
        ha='center', va='bottom', fontweight='bold', alpha=0.85)
ax.text(0.15, 0.92, 'Well-aligned', transform=ax.transAxes,
        fontsize=8.5, color=mode_colors['Well-aligned\n(low refusal, high NVAS)'],
        ha='center', va='top', fontweight='bold', alpha=0.85)

ax.set_xlabel('Refusal Rate (Persona EN, instruct models)', fontsize=10)
ax.set_ylabel('NVAS when Answered', fontsize=10)
ax.set_title(
    'T3 Question Taxonomy: Two Distinct Failure Modes\n'
    'Third-person framing fixes suppression (right) but not representational bias (left)',
    fontsize=10.5)
ax.legend(fontsize=9, framealpha=0.9, loc='lower right')
ax.grid(lw=0.3, alpha=0.4, color='#bbb', zorder=1)
ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
fig.tight_layout()
save(fig, 'gapB_t3_question_taxonomy.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# 4. flagship_gating_scatter.pdf  — much bigger dots
# ══════════════════════════════════════════════════════════════════════════════
print('=== 4. flagship_gating_scatter ===')

all_models = sorted(refusal_df['model'].unique())
rows = []
for mk in all_models:
    r1 = refusal_df[(refusal_df['model'] == mk) & (refusal_df['tier'] == 1)]
    r3 = refusal_df[(refusal_df['model'] == mk) & (refusal_df['tier'] == 3)]
    n3 = nvas_df[(nvas_df['model'] == mk) & (nvas_df['tier'] == 3)]
    n1 = nvas_df[(nvas_df['model'] == mk) & (nvas_df['tier'] == 1)]
    if len(r1) and len(r3) and len(n3):
        rows.append({
            'model':       mk,
            'label':       LABEL_MAP.get(mk, mk),
            'safety_tax':  float(r3['refusal_rate'].values[0]) - float(r1['refusal_rate'].values[0]),
            'nvas_t3':     float(n3['nvas'].values[0]),
            'nvas_t1':     float(n1['nvas'].values[0]) if len(n1) else np.nan,
        })
scatter_df = pd.DataFrame(rows)
mean_t1_nvas = scatter_df['nvas_t1'].mean()

FAM_COLORS = {'base': '#90A4AE', 'train': '#7986CB', 'instruct': '#42A5F5',
              'frontier': '#FF6F00', 'mena': '#E53935'}
MARKERS    = {'base': 's', 'train': '^', 'instruct': 'o', 'frontier': '*', 'mena': 'D'}

def model_fam_type(mk):
    if mk in ('gpt_5', 'gpt4o_mini'): return 'frontier'
    if mk in ('aya_expanse_8b', 'aya_expanse_32b', 'allam_7b_instruct',
              'fanar_1_9b_instruct', 'jais_2_8b_chat'): return 'mena'
    return fam_type(mk)

label_offsets = {
    'OLMo-7B-Base': (-5, 7), 'OLMo-7B-SFT': (6, 5), 'OLMo-7B-DPO': (6, -10),
    'OLMo-7B-IT': (6, 5), 'OLMo-32B-Base': (-5, 7), 'OLMo-32B-IT': (6, 5),
    'GPT-5': (6, 6), 'GPT-4o-mini': (6, -10), 'LLaMA-8B-Base': (-5, 7),
    'Tulu3.1-8B': (6, -10), 'ALLAM-7B': (-5, 7), 'AYA-32B': (6, 5),
}

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

if not np.isnan(mean_t1_nvas):
    ax.axhline(mean_t1_nvas, color='#555', lw=1.2, ls=':', alpha=0.7,
               label=f'Mean T1 accepted NVAS ({mean_t1_nvas:.3f})')
ax.axvline(0, color='#555', lw=0.8, ls='--', alpha=0.4)

for _, row in scatter_df.iterrows():
    ft  = model_fam_type(row['model'])
    col = FAM_COLORS[ft]
    mk_sym = MARKERS[ft]
    ms = 350 if ft == 'frontier' else 200
    ax.scatter([row['safety_tax']], [row['nvas_t3']], s=ms,
               color=col, marker=mk_sym, zorder=5,
               edgecolors='white', linewidths=0.9)
    dx, dy = label_offsets.get(row['label'], (6, 5))
    ax.annotate(row['label'], (row['safety_tax'], row['nvas_t3']),
                fontsize=7, xytext=(dx, dy), textcoords='offset points',
                color='#333', zorder=6)

handles = [mpatches.Patch(color=FAM_COLORS[k], label=k.title()) for k in FAM_COLORS]
ax.legend(handles=handles, fontsize=13, framealpha=0.9, loc='lower right')
ax.set_xlabel('Safety tax (T3 − T1 refusal rate, Persona EN)', fontsize=11)
ax.set_ylabel('T3 accepted NVAS (Persona EN)', fontsize=11)
ax.set_title('Safety Tax vs Cultural Accuracy: All 26 Models', fontsize=12)
ax.grid(lw=0.3, alpha=0.4, color='#ccc')
fig.tight_layout()
save(fig, 'flagship_gating_scatter.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# 5 & 6. pca_7_3_native.pdf and pca_7_4_persona_neutral.pdf  — bigger dots
# ══════════════════════════════════════════════════════════════════════════════
print('=== 5 & 6. PCA 7.3 and 7.4 (loading master_table.pkl ~206 MB) ===')
MASTER = pd.read_pickle(os.path.join(RESULTS, 'master_table.pkl'))
print('  master_table loaded.')

fam_col = {'Arabic': '#E53935', 'Persian': '#1E88E5', 'Turkish': '#43A047'}

def get_country_matrix(master, model, sheet):
    qids = sorted(master['question_id'].unique())
    mat, valid_c = [], []
    for country in COUNTRIES:
        sub = (master[(master['model'] == model) & (master['sheet'] == sheet) &
                      (master['country'] == country) & ~master['refusal'] &
                      master['extracted'].notna()]
               .set_index('question_id'))
        if len(sub) < 30:
            continue
        vec = [float(sub.loc[q, 'extracted']) if q in sub.index else np.nan for q in qids]
        mat.append(vec); valid_c.append(country)
    if len(mat) < 3:
        return None, None
    mat = np.array(mat, dtype=float)
    valid_cols = ~np.all(np.isnan(mat), axis=0)
    mat = mat[:, valid_cols]
    cm = np.nanmean(mat, axis=0)
    ix = np.where(np.isnan(mat))
    mat[ix] = np.take(cm, ix[1])
    try:
        coords = PCA(2).fit_transform(StandardScaler().fit_transform(mat))
    except Exception:
        return None, None
    return coords, valid_c


# ── PCA 7.3: Native Language Observer ────────────────────────────────────────
focus_73 = ['olmo_3_7b_instruct','llama_3.1_8b_instruct',
            'qwen2.5_7b_instruct','gemma_3_12b_it','aya_expanse_8b']
focus_73 = [m for m in focus_73 if m in MASTER['model'].unique()]

if focus_73:
    fig, axes = plt.subplots(1, len(focus_73), figsize=(5 * len(focus_73), 5))
    if len(focus_73) == 1:
        axes = [axes]
    for ax, model in zip(axes, focus_73):
        coords, valid_c = get_country_matrix(MASTER, model, 'Third_Diff')
        if coords is None:
            ax.axis('off'); ax.set_title(model, fontsize=11); continue
        for i, c in enumerate(valid_c):
            fam = next((f for f, cs in LANG_FAMILY.items() if c in cs), 'Other')
            ax.scatter(coords[i, 0], coords[i, 1],
                       color=fam_col.get(fam, 'gray'), s=140, zorder=5,
                       edgecolors='white', linewidths=0.6)
            ax.annotate(c[:3], (coords[i, 0], coords[i, 1]),
                        fontsize=9, xytext=(4, 4), textcoords='offset points')
        handles = [mpatches.Patch(color=v, label=k) for k, v in fam_col.items()]
        ax.legend(handles=handles, fontsize=11)
        ax.set_title(model.replace('_', ' '), fontsize=11)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(lw=0.3, alpha=0.4, color='#ccc')
    fig.suptitle('PCA 7.3: Native Language Observer — Language-Family Collapse', fontsize=13)
    fig.tight_layout()
    save(fig, 'pca_7_3_native.pdf')


# ── PCA 7.4: Persona + Neutral ────────────────────────────────────────────────
focus_74 = ['olmo_3_7b_instruct','llama_3.1_8b_instruct',
            'qwen2.5_7b_instruct','gemma_3_12b_it']
focus_74 = [m for m in focus_74 if m in MASTER['model'].unique()]

if focus_74:
    fig, axes = plt.subplots(1, len(focus_74), figsize=(5 * len(focus_74), 5))
    if len(focus_74) == 1:
        axes = [axes]
    qids = sorted(MASTER['question_id'].unique())
    for ax, model in zip(axes, focus_74):
        mat, labels, colors = [], [], []
        for country in COUNTRIES:
            sub = (MASTER[(MASTER['model'] == model) &
                          (MASTER['sheet'] == 'Personalization') &
                          (MASTER['country'] == country) &
                          ~MASTER['refusal'] & MASTER['extracted'].notna()]
                   .set_index('question_id'))
            if len(sub) < 30:
                continue
            vec = [float(sub.loc[q, 'extracted']) if q in sub.index else np.nan for q in qids]
            mat.append(vec)
            fam = next((f for f, cs in LANG_FAMILY.items() if country in cs), 'Other')
            colors.append(fam_col.get(fam, 'gray'))
            labels.append(country[:3])

        neut = (MASTER[(MASTER['model'] == model) &
                       (MASTER['sheet'] == 'No Mention') &
                       (MASTER['country'] == 'NEUTRAL') &
                       ~MASTER['refusal'] & MASTER['extracted'].notna()]
                .set_index('question_id'))
        if len(neut) >= 30:
            vec = [float(neut.loc[q, 'extracted']) if q in neut.index else np.nan for q in qids]
            mat.append(vec); colors.append('black'); labels.append('LLM★')

        if len(mat) < 3:
            ax.axis('off'); continue

        mat_np = np.array(mat, dtype=float)
        valid_cols = ~np.all(np.isnan(mat_np), axis=0)
        mat_np = mat_np[:, valid_cols]
        cm = np.nanmean(mat_np, axis=0)
        ix = np.where(np.isnan(mat_np))
        mat_np[ix] = np.take(cm, ix[1])
        try:
            coords = PCA(2).fit_transform(StandardScaler().fit_transform(mat_np))
        except Exception:
            ax.axis('off'); continue

        for i, (lbl, col) in enumerate(zip(labels, colors)):
            ms = 350 if lbl == 'LLM★' else 120
            mk = '*' if lbl == 'LLM★' else 'o'
            ax.scatter(coords[i, 0], coords[i, 1], color=col, s=ms, marker=mk,
                       zorder=5, edgecolors='white', linewidths=0.6)
            ax.annotate(lbl, (coords[i, 0], coords[i, 1]),
                        fontsize=8, xytext=(3, 2), textcoords='offset points')
        ax.set_title(model.replace('_', ' '), fontsize=11)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(lw=0.3, alpha=0.4, color='#ccc')
        # legend for language families
        handles74 = [mpatches.Patch(color=v, label=k) for k, v in fam_col.items()]
        handles74.append(plt.Line2D([0],[0], marker='*', color='black', ls='',
                                    ms=12, label='LLM (neutral)'))
        ax.legend(handles=handles74, fontsize=11, framealpha=0.9)

    fig.suptitle('PCA 7.4: Persona + Neutral (★=LLM) — Cultural Identity Crisis', fontsize=13)
    fig.tight_layout()
    save(fig, 'pca_7_4_persona_neutral.pdf')


# ══════════════════════════════════════════════════════════════════════════════
# 7. fix1_residual_probe.pdf  — fixed axis coverage
# ══════════════════════════════════════════════════════════════════════════════
print('=== 7. fix1_residual_probe ===')

df = pd.read_csv(os.path.join(RESULTS, 'fix1_residual_probe_results.csv'))

PROBE_MODELS = {
    'olmo_3_7b_sft':         {'label': 'OLMo-7B-SFT',  'color': '#7986CB'},
    'olmo_3_7b_dpo':         {'label': 'OLMo-7B-DPO',  'color': '#3F51B5'},
    'olmo_3_7b_instruct':    {'label': 'OLMo-7B-IT',   'color': '#1A237E'},
    'tulu_3_8b_sft':         {'label': 'Tulu3-SFT',    'color': '#F3A046'},
    'tulu_3_8b_dpo':         {'label': 'Tulu3-DPO',    'color': '#E67C13'},
    'tulu_3_8b_instruct':    {'label': 'Tulu3-IT',     'color': '#C65D00'},
    'llama_3.1_8b_instruct': {'label': 'LLaMA-8B-IT',  'color': '#2E7D32'},
}

def sensible_ylim(series, pad=0.05):
    """Return ylim that clips the bottom 5th percentile of extreme negatives."""
    vals = series.dropna().values
    lo = np.percentile(vals, 5)
    hi = np.percentile(vals, 100)
    span = max(hi - lo, 0.1)
    return (lo - pad * span, hi + pad * span)

r2_question_baseline = 0.666

fig = plt.figure(figsize=(13, 9))
fig.patch.set_facecolor('white')
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
ax_tl  = fig.add_subplot(gs[0, 0])
ax_tr  = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, :])

# Pre-compute reasonable ylims per metric
ylim_total     = sensible_ylim(df['r2_total'])
ylim_t3_total  = sensible_ylim(df['r2_t3_total'])
ylim_t3_resid  = sensible_ylim(df['r2_t3_resid'])

top_panels = [
    (ax_tl, 'r2_total',    ylim_total,    'Standard probe R²\n(all tiers, all layers)'),
    (ax_tr, 'r2_t3_total', ylim_t3_total, 'T3 standard probe R²\n(all layers)'),
]

for ax, metric, ylim, title in top_panels:
    ax.set_facecolor('#FAFAFA')
    for mk, info in PROBE_MODELS.items():
        sub = df[df['model'] == mk].sort_values('layer')
        if len(sub) == 0:
            continue
        y_vals = sub[metric].values.copy()
        valid  = ~np.isnan(y_vals)
        if valid.sum() < 2:
            continue
        # Clip to ylim so extreme early-layer spikes don't compress the view
        y_plot = np.clip(y_vals, ylim[0], ylim[1])
        ax.plot(sub['layer'].values[valid], y_plot[valid],
                color=info['color'], lw=1.8, label=info['label'],
                marker='o', ms=3.5, alpha=0.9)
    ax.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(r2_question_baseline, color='#E53935', lw=1.2, ls=':', alpha=0.7,
               label=f'Q-only baseline ({r2_question_baseline:.3f})')
    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('Probe R²', fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(*ylim)
    ax.grid(lw=0.3, alpha=0.4, color='#ccc')

ax_tl.legend(fontsize=6.5, framealpha=0.9, ncol=2)
ax_tr.legend(fontsize=6.5, framealpha=0.9, loc='upper left',
             handles=[plt.Line2D([], [], color='#E53935', ls=':', lw=1.2,
                                 label=f'Q-only baseline ({r2_question_baseline:.3f})')])

# Bottom: residualized T3 probe
ax_bot.set_facecolor('#FAFAFA')
for mk, info in PROBE_MODELS.items():
    sub = df[df['model'] == mk].sort_values('layer')
    if len(sub) == 0:
        continue
    y_vals = sub['r2_t3_resid'].values.copy()
    valid  = ~np.isnan(y_vals)
    if valid.sum() < 2:
        continue
    y_plot = np.clip(y_vals, ylim_t3_resid[0], ylim_t3_resid[1])
    ax_bot.plot(sub['layer'].values[valid], y_plot[valid],
                color=info['color'], lw=1.8, label=info['label'],
                marker='o', ms=3.5, alpha=0.9)
ax_bot.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
ax_bot.axvspan(28, 32, alpha=0.12, color='#E53935', label='Erasure zone (layers 28–32)')
ax_bot.text(30.0, ylim_t3_resid[0] + 0.05 * (ylim_t3_resid[1] - ylim_t3_resid[0]),
            'erasure\nzone', fontsize=7.5, color='#E53935',
            ha='center', va='bottom', alpha=0.85)
ax_bot.set_xlabel('Layer', fontsize=9)
ax_bot.set_ylabel('Probe R²', fontsize=9)
ax_bot.set_title(
    'Residualized T3 probe — pure country-specific cultural encoding\n'
    '(per-question mean subtracted; erasure zone = layers 28–32)', fontsize=9)
ax_bot.set_ylim(*ylim_t3_resid)
ax_bot.grid(lw=0.3, alpha=0.4, color='#ccc')
ax_bot.legend(fontsize=6.5, framealpha=0.9, ncol=4)

fig.suptitle(
    'Probing Confound Control: Standard vs. Residualized\n'
    f'Standard probe R² includes question-level baseline ({r2_question_baseline:.3f}); '
    'residualized probe isolates country-specific cultural signal',
    fontsize=10.5)
save(fig, 'fix1_residual_probe.pdf')

print('\nAll 7 figures done.')
