"""
Gap 1: Third framing as a practical intervention.
Shows refusal reduction AND NVAS improvement across all models.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

BASE = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

mt = pd.read_pickle(os.path.join(RESULTS, 'master_table.pkl'))

def nvas(row):
    if pd.isna(row['extracted']) or pd.isna(row['human_mean']): return np.nan
    d = row['vmax'] - row['vmin']
    if d == 0: return np.nan
    return 1 - abs(row['extracted'] - row['human_mean']) / d

mt_acc = mt[~mt['refusal']].copy()
mt_acc['nvas'] = mt_acc.apply(nvas, axis=1)

INSTRUCT_MODELS = [
    'olmo_3_7b_instruct', 'olmo_3_32b_instruct',
    'tulu_3_8b_dpo', 'tulu_3_8b_sft', 'tulu_3.1_8b',
    'llama_3.1_8b_instruct',
    'gemma_3_4b_it', 'gemma_3_12b_it', 'gemma_3_27b_it',
    'gpt4o_mini', 'gpt_5',
    'qwen2.5_7b_instruct', 'qwen3_4b_instruct', 'qwen3_30b_a3b_instruct',
    'aya_expanse_8b', 'aya_expanse_32b',
    'allam_7b_instruct', 'fanar_1_9b_instruct', 'jais_2_8b_chat',
    'mistral_7b_instruct',
]
LABEL_MAP = {
    'olmo_3_7b_instruct': 'OLMo-7B-IT', 'olmo_3_32b_instruct': 'OLMo-32B-IT',
    'tulu_3_8b_dpo': 'Tulu3-DPO', 'tulu_3_8b_sft': 'Tulu3-SFT', 'tulu_3.1_8b': 'Tulu3.1',
    'llama_3.1_8b_instruct': 'LLaMA-8B-IT',
    'gemma_3_4b_it': 'Gemma3-4B', 'gemma_3_12b_it': 'Gemma3-12B', 'gemma_3_27b_it': 'Gemma3-27B',
    'gpt4o_mini': 'GPT-4o-mini', 'gpt_5': 'GPT-5',
    'qwen2.5_7b_instruct': 'Qwen2.5-7B', 'qwen3_4b_instruct': 'Qwen3-4B',
    'qwen3_30b_a3b_instruct': 'Qwen3-30B',
    'aya_expanse_8b': 'AYA-8B', 'aya_expanse_32b': 'AYA-32B',
    'allam_7b_instruct': 'ALLAM-7B', 'fanar_1_9b_instruct': 'FANAR-9B',
    'jais_2_8b_chat': 'JAIS-8B', 'mistral_7b_instruct': 'Mistral-7B',
}
FAM_COLOR = {'MENA': '#E53935', 'Western': '#1E88E5', 'Frontier': '#FF6F00'}
MODEL_FAM = {
    'OLMo-7B-IT': 'Western', 'OLMo-32B-IT': 'Western',
    'Tulu3-DPO': 'Western', 'Tulu3-SFT': 'Western', 'Tulu3.1': 'Western',
    'LLaMA-8B-IT': 'Western', 'Gemma3-4B': 'Western', 'Gemma3-12B': 'Western',
    'Gemma3-27B': 'Western', 'Mistral-7B': 'Western',
    'Qwen2.5-7B': 'Western', 'Qwen3-4B': 'Western', 'Qwen3-30B': 'Western',
    'AYA-8B': 'MENA', 'AYA-32B': 'MENA', 'ALLAM-7B': 'MENA',
    'FANAR-9B': 'MENA', 'JAIS-8B': 'MENA',
    'GPT-4o-mini': 'Frontier', 'GPT-5': 'Frontier',
}

rows = []
for mk in INSTRUCT_MODELS:
    if mk not in mt['model'].values: continue
    label = LABEL_MAP.get(mk, mk)
    sub = mt[mt['model'] == mk]
    sub_acc = mt_acc[mt_acc['model'] == mk]
    for tier in [1, 2, 3]:
        for framing, sheet in [('Persona', 'Personalization'), ('Third', 'Third')]:
            n_all = len(sub[(sub['tier']==tier)&(sub['sheet']==sheet)])
            n_ref = sub[(sub['tier']==tier)&(sub['sheet']==sheet)]['refusal'].sum()
            nv = sub_acc[(sub_acc['tier']==tier)&(sub_acc['sheet']==sheet)]['nvas'].mean()
            if n_all > 0:
                rows.append({'model': mk, 'label': label, 'tier': tier,
                             'framing': framing, 'refusal': n_ref/n_all,
                             'nvas': nv, 'n': n_all})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, 'gap1_framing_comparison.csv'), index=False)

# Pivot: per model, T3 refusal and NVAS for Persona vs Third
t3 = df[df['tier']==3].copy()
piv_ref  = t3.pivot_table(index='label', columns='framing', values='refusal')
piv_nvas = t3.pivot_table(index='label', columns='framing', values='nvas')
piv_ref['delta_ref']   = piv_ref['Third']  - piv_ref['Persona']   # neg = good
piv_nvas['delta_nvas'] = piv_nvas['Third'] - piv_nvas['Persona']  # pos = good

print('=== Third framing effect on T3 ===')
combined = pd.DataFrame({
    'Persona_ref': piv_ref['Persona'], 'Third_ref': piv_ref['Third'],
    'Δ_ref': piv_ref['delta_ref'],
    'Persona_nvas': piv_nvas['Persona'], 'Third_nvas': piv_nvas['Third'],
    'Δ_nvas': piv_nvas['delta_nvas'],
})
print(combined.round(3).sort_values('Δ_nvas', ascending=False).to_string())
print(f"\nMean Δ_ref={piv_ref['delta_ref'].mean():+.3f}  "
      f"Mean Δ_nvas={piv_nvas['delta_nvas'].mean():+.3f}")

# Paired t-test across models
valid = piv_nvas.dropna()
t, p = stats.ttest_rel(valid['Third'], valid['Persona'])
print(f"Paired t-test Third vs Persona NVAS (T3): t={t:.2f}, p={p:.4f}")

# ── FIGURE 1: Two-panel — refusal reduction + NVAS improvement ─────────────
models_sorted = combined.sort_values('Δ_nvas', ascending=False).index.tolist()
n_m = len(models_sorted)
y   = np.arange(n_m)
cols = [FAM_COLOR.get(MODEL_FAM.get(m, 'Western'), '#555') for m in models_sorted]

fig, axes = plt.subplots(1, 2, figsize=(13, max(6, n_m*0.38)))
fig.patch.set_facecolor('white')

# Left: T3 refusal rate Persona vs Third
ax = axes[0]; ax.set_facecolor('#FAFAFA')
p_vals = [combined.loc[m, 'Persona_ref'] if m in combined.index else np.nan for m in models_sorted]
t_vals = [combined.loc[m, 'Third_ref']   if m in combined.index else np.nan for m in models_sorted]
for i, (pv, tv, col) in enumerate(zip(p_vals, t_vals, cols)):
    if np.isnan(pv) or np.isnan(tv): continue
    ax.plot([pv, tv], [i, i], color=col, lw=1.3, alpha=0.6)
    ax.scatter([pv], [i], s=35, color=col, marker='o', zorder=4)
    ax.scatter([tv], [i], s=35, color=col, marker='D', zorder=4)
ax.set_yticks(y); ax.set_yticklabels(models_sorted, fontsize=7.5)
ax.set_xlabel('T3 refusal rate', fontsize=9.5)
ax.set_title('Refusal rate: Persona (●) → Third (◆)\nThird framing reduces T3 gating', fontsize=9)
ax.axvline(0.1, color='#aaa', lw=0.7, ls='--', alpha=0.5)
ax.grid(axis='x', lw=0.3, alpha=0.4, color='#ccc')
ax.set_xlim(-0.02, 1.0)

# Right: T3 NVAS Persona vs Third
ax2 = axes[1]; ax2.set_facecolor('#FAFAFA')
pn_vals = [combined.loc[m, 'Persona_nvas'] if m in combined.index else np.nan for m in models_sorted]
tn_vals = [combined.loc[m, 'Third_nvas']   if m in combined.index else np.nan for m in models_sorted]
for i, (pv, tv, col) in enumerate(zip(pn_vals, tn_vals, cols)):
    if np.isnan(pv) or np.isnan(tv): continue
    ax2.plot([pv, tv], [i, i], color=col, lw=1.3, alpha=0.6)
    ax2.scatter([pv], [i], s=35, color=col, marker='o', zorder=4)
    ax2.scatter([tv], [i], s=35, color=col, marker='D', zorder=4)
ax2.set_yticks(y); ax2.set_yticklabels([], fontsize=7.5)
ax2.set_xlabel('T3 accepted NVAS', fontsize=9.5)
ax2.set_title(f'Accepted NVAS: Persona (●) → Third (◆)\nMean Δ = {piv_nvas["delta_nvas"].mean():+.3f} (p={p:.3f})', fontsize=9)
ax2.grid(axis='x', lw=0.3, alpha=0.4, color='#ccc')
ax2.set_xlim(0.3, 1.0)

# Legend
handles = [mpatches.Patch(color=c, label=k) for k, c in FAM_COLOR.items()
           if any(MODEL_FAM.get(m,'') == k for m in models_sorted)]
handles += [plt.Line2D([0],[0], marker='o', color='#555', label='Persona', ls='none', ms=6),
            plt.Line2D([0],[0], marker='D', color='#555', label='Third',   ls='none', ms=6)]
axes[0].legend(handles=handles, fontsize=7.5, loc='lower right', framealpha=0.9)

fig.suptitle('Third-Person Framing as a Practical Intervention\n'
             'Reduces T3 refusals AND improves cultural accuracy (accepted NVAS)',
             fontsize=11, y=1.01)
fig.tight_layout()
out = os.path.join(RESULTS, 'gap1_third_framing_intervention.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'Saved: {out}')

# ── FIGURE 2: Scatter — Δ_ref vs Δ_nvas (tradeoff space) ────────────────────
fig2, ax3 = plt.subplots(figsize=(8, 6)); fig2.patch.set_facecolor('white')
ax3.set_facecolor('#FAFAFA')
for m in models_sorted:
    if m not in combined.index: continue
    dr = combined.loc[m, 'Δ_ref']; dn = combined.loc[m, 'Δ_nvas']
    if np.isnan(dr) or np.isnan(dn): continue
    col = FAM_COLOR.get(MODEL_FAM.get(m,'Western'), '#555')
    ax3.scatter([dr], [dn], s=60, color=col, zorder=4, edgecolors='white', linewidths=0.5)
    ax3.annotate(m, (dr, dn), fontsize=6.5, xytext=(3,3), textcoords='offset points', color='#333')
ax3.axhline(0, color='#333', lw=0.8, ls='--', alpha=0.4)
ax3.axvline(0, color='#333', lw=0.8, ls='--', alpha=0.4)
ax3.fill_betweenx([-0.5,0.5], -1, 0, alpha=0.04, color='#4CAF50', label='Ideal quadrant')
ax3.set_xlabel('Δ T3 refusal rate  (Third − Persona; negative = fewer refusals)', fontsize=9.5)
ax3.set_ylabel('Δ T3 accepted NVAS  (Third − Persona; positive = more accurate)', fontsize=9.5)
ax3.set_title('Third framing tradeoff space per model\n'
              'Ideal: bottom-right (fewer refusals AND more accurate)', fontsize=10)
handles2 = [mpatches.Patch(color=c, label=k) for k, c in FAM_COLOR.items()]
ax3.legend(handles=handles2, fontsize=8, framealpha=0.9)
ax3.grid(lw=0.3, alpha=0.4, color='#ccc')
fig2.tight_layout()
out2 = os.path.join(RESULTS, 'gap1_third_framing_scatter.pdf')
fig2.savefig(out2, bbox_inches='tight', dpi=200); plt.close(fig2)
print(f'Saved: {out2}')

# ── FIGURE 3: T3 NVAS across all 3 tiers (Third vs Persona) — T1/T2 stability
fig3, axes3 = plt.subplots(1, 3, figsize=(13, 5)); fig3.patch.set_facecolor('white')
for ti, t in enumerate([1, 2, 3]):
    ax = axes3[ti]; ax.set_facecolor('#FAFAFA')
    sub = df[df['tier']==t]
    piv = sub.pivot_table(index='label', columns='framing', values='nvas').dropna()
    delta = (piv['Third'] - piv['Persona']).sort_values(ascending=False)
    colors_bar = ['#4CAF50' if v>0 else '#F44336' for v in delta.values]
    ax.barh(range(len(delta)), delta.values, color=colors_bar, alpha=0.8, edgecolor='white')
    ax.axvline(0, color='#333', lw=1.0, ls='--', alpha=0.6)
    ax.set_yticks(range(len(delta))); ax.set_yticklabels(delta.index, fontsize=7)
    ax.set_xlabel('Δ NVAS (Third − Persona)', fontsize=8.5)
    ax.set_title(f'Tier {t}: {"T1 (benign)" if t==1 else "T2 (moderate)" if t==2 else "T3 (sensitive)"}\n'
                 f'mean Δ={delta.mean():+.3f}', fontsize=9)
    ax.grid(axis='x', lw=0.3, alpha=0.4, color='#ccc')
fig3.suptitle('Third-person framing NVAS gain per tier\n'
              'T3 benefits most; T1/T2 show negligible/mixed effects (T3-specific intervention)',
              fontsize=10, y=1.01)
fig3.tight_layout()
out3 = os.path.join(RESULTS, 'gap1_third_nvas_by_tier.pdf')
fig3.savefig(out3, bbox_inches='tight', dpi=200); plt.close(fig3)
print(f'Saved: {out3}')
print('\nDone.')
