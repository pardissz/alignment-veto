"""
Gap 3: GPT-5 as the 'solved frontier' — spotlight figure.
Also shows safety tax vs T3 NVAS scatter for all models.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

mt = pd.read_pickle(os.path.join(RESULTS, 'master_table.pkl'))

def nvas(row):
    if pd.isna(row['extracted']) or pd.isna(row['human_mean']): return np.nan
    d = row['vmax'] - row['vmin']
    return np.nan if d == 0 else 1 - abs(row['extracted'] - row['human_mean']) / d

mt_acc = mt[~mt['refusal']].copy()
mt_acc['nvas'] = mt_acc.apply(nvas, axis=1)

# Per-model stats (Persona EN)
rows = []
for mk in mt['model'].unique():
    sub = mt[(mt['model']==mk) & (mt['sheet']=='Personalization')]
    sub_t = mt[(mt['model']==mk) & (mt['sheet']=='Third')]
    if len(sub) == 0: continue
    r_t3_p = sub[sub['tier']==3]['refusal'].mean()
    r_t1_p = sub[sub['tier']==1]['refusal'].mean()
    r_t3_th = sub_t[sub_t['tier']==3]['refusal'].mean() if len(sub_t)>0 else np.nan
    nv_t3_p = mt_acc[(mt_acc['model']==mk)&(mt_acc['sheet']=='Personalization')&(mt_acc['tier']==3)]['nvas'].mean()
    nv_t3_th = mt_acc[(mt_acc['model']==mk)&(mt_acc['sheet']=='Third')&(mt_acc['tier']==3)]['nvas'].mean()
    nv_all   = mt_acc[(mt_acc['model']==mk)&(mt_acc['sheet']=='Personalization')]['nvas'].mean()
    rows.append({'model': mk, 'safety_tax': r_t3_p - r_t1_p,
                 'refusal_t3_persona': r_t3_p, 'refusal_t3_third': r_t3_th,
                 'nvas_t3_persona': nv_t3_p, 'nvas_t3_third': nv_t3_th,
                 'nvas_all': nv_all})
df = pd.DataFrame(rows)

LABEL_MAP = {
    'olmo_3_7b_base':'OLMo-7B-Base','olmo_3_7b_sft':'OLMo-7B-SFT',
    'olmo_3_7b_dpo':'OLMo-7B-DPO','olmo_3_7b_instruct':'OLMo-7B-IT',
    'olmo_3_32b_base':'OLMo-32B-Base','olmo_3_32b_sft':'OLMo-32B-SFT',
    'olmo_3_32b_dpo':'OLMo-32B-DPO','olmo_3_32b_instruct':'OLMo-32B-IT',
    'tulu_3_8b_sft':'Tulu3-SFT','tulu_3_8b_dpo':'Tulu3-DPO','tulu_3.1_8b':'Tulu3.1',
    'llama_3.1_8b_base':'LLaMA-8B-Base','llama_3.1_8b_instruct':'LLaMA-8B-IT',
    'gemma_3_4b_it':'Gemma3-4B','gemma_3_12b_it':'Gemma3-12B','gemma_3_27b_it':'Gemma3-27B',
    'gpt4o_mini':'GPT-4o-mini','gpt_5':'GPT-5',
    'qwen2.5_7b_instruct':'Qwen2.5-7B','qwen3_4b_instruct':'Qwen3-4B',
    'qwen3_30b_a3b_instruct':'Qwen3-30B',
    'aya_expanse_8b':'AYA-8B','aya_expanse_32b':'AYA-32B',
    'allam_7b_instruct':'ALLAM-7B','fanar_1_9b_instruct':'FANAR-9B',
    'jais_2_8b_chat':'JAIS-8B','mistral_7b_instruct':'Mistral-7B',
}
df['label'] = df['model'].map(LABEL_MAP).fillna(df['model'])

FAM = {
    'OLMo-7B-Base':'base','OLMo-7B-SFT':'train','OLMo-7B-DPO':'train','OLMo-7B-IT':'instruct',
    'OLMo-32B-Base':'base','OLMo-32B-SFT':'train','OLMo-32B-DPO':'train','OLMo-32B-IT':'instruct',
    'Tulu3-SFT':'train','Tulu3-DPO':'train','Tulu3.1':'instruct',
    'LLaMA-8B-Base':'base','LLaMA-8B-IT':'instruct',
    'Gemma3-4B':'instruct','Gemma3-12B':'instruct','Gemma3-27B':'instruct',
    'GPT-4o-mini':'frontier','GPT-5':'frontier',
    'Qwen2.5-7B':'instruct','Qwen3-4B':'instruct','Qwen3-30B':'instruct',
    'AYA-8B':'mena','AYA-32B':'mena','ALLAM-7B':'mena','FANAR-9B':'mena','JAIS-8B':'mena',
    'Mistral-7B':'instruct',
}
FAM_COLORS = {'base':'#90A4AE','train':'#7986CB','instruct':'#42A5F5',
              'frontier':'#FF6F00','mena':'#E53935'}
MARKERS = {'base':'s','train':'^','instruct':'o','frontier':'*','mena':'D'}

df['fam'] = df['label'].map(FAM).fillna('instruct')

print(df[['label','safety_tax','nvas_t3_persona','nvas_t3_third','nvas_all']].round(3).sort_values('nvas_t3_persona', ascending=False).to_string())

# ── Figure 1: Safety tax vs T3 NVAS scatter + Third framing arrows ───────────
fig, ax = plt.subplots(figsize=(11, 7.5)); fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

for _, row in df.iterrows():
    fam  = row['fam']
    col  = FAM_COLORS[fam]
    mk   = MARKERS[fam]
    ms   = 120 if fam == 'frontier' else 65
    lbl  = row['label']
    if not np.isnan(row['safety_tax']) and not np.isnan(row['nvas_t3_persona']):
        ax.scatter([row['safety_tax']], [row['nvas_t3_persona']], s=ms,
                   color=col, marker=mk, zorder=5, edgecolors='white', linewidths=0.6)
        offset = {'GPT-5': (4,5), 'GPT-4o-mini': (4,-9),
                  'ALLAM-7B': (-4,5), 'OLMo-7B-DPO': (4,-9),
                  'OLMo-32B-IT': (4,4)}.get(lbl, (4,2))
        ax.annotate(lbl, (row['safety_tax'], row['nvas_t3_persona']),
                    fontsize=6.5, xytext=offset, textcoords='offset points', color='#333')

ax.axvline(0, color='#555', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Safety tax (T3 − T1 refusal rate, Persona EN framing)', fontsize=10)
ax.set_ylabel('T3 accepted NVAS (Persona EN)', fontsize=10)
ax.set_title('Safety Tax vs Cultural Accuracy: All Models', fontsize=10.5)

handles = [mpatches.Patch(color=FAM_COLORS[k], label=k.title()) for k in FAM_COLORS]
ax.legend(handles=handles, fontsize=8, framealpha=0.9, loc='lower right')
ax.grid(lw=0.3, alpha=0.4, color='#ccc')

# Highlight GPT-5
gpt5_row = df[df['label']=='GPT-5']
if len(gpt5_row):
    circ = plt.Circle((gpt5_row['safety_tax'].values[0], gpt5_row['nvas_t3_persona'].values[0]),
                       0.015, color='#FF6F00', fill=False, lw=2, zorder=6)
    ax.add_patch(circ)

fig.tight_layout()
out = os.path.join(RESULTS, 'gap3_frontier_scatter.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'Saved: {out}')

# ── Figure 2: GPT-5 spotlight — per-tier refusal + NVAS ─────────────────────
SPOTLIGHT = ['allam_7b_instruct','olmo_3_32b_instruct','aya_expanse_32b',
             'llama_3.1_8b_instruct','qwen3_30b_a3b_instruct',
             'gemma_3_27b_it','gpt4o_mini','gpt_5']
SLABELS = {mk: LABEL_MAP.get(mk,mk) for mk in SPOTLIGHT}
TIERS = [1,2,3]
TIER_COLS = {1:'#4CAF50',2:'#FF9800',3:'#F44336'}

fig2, axes2 = plt.subplots(2, len(SPOTLIGHT), figsize=(14, 6), sharey='row')
fig2.patch.set_facecolor('white')

for ci, mk in enumerate(SPOTLIGHT):
    for ri, (metric, ylabel, sheet) in enumerate([
        ('refusal', 'Refusal rate', 'Personalization'),
        ('nvas', 'Accepted NVAS', 'Personalization'),
    ]):
        ax = axes2[ri][ci]; ax.set_facecolor('#FAFAFA')
        for t in TIERS:
            if metric == 'refusal':
                val_p = mt[(mt['model']==mk)&(mt['sheet']=='Personalization')&(mt['tier']==t)]['refusal'].mean()
                val_th= mt[(mt['model']==mk)&(mt['sheet']=='Third')&(mt['tier']==t)]['refusal'].mean()
            else:
                val_p = mt_acc[(mt_acc['model']==mk)&(mt_acc['sheet']=='Personalization')&(mt_acc['tier']==t)]['nvas'].mean()
                val_th= mt_acc[(mt_acc['model']==mk)&(mt_acc['sheet']=='Third')&(mt_acc['tier']==t)]['nvas'].mean()
            x_p  = (t-1)*1.0 - 0.15
            x_th = (t-1)*1.0 + 0.15
            ax.bar(x_p,  val_p,  0.28, color=TIER_COLS[t], alpha=0.7, edgecolor='white')
            ax.bar(x_th, val_th, 0.28, color=TIER_COLS[t], alpha=0.35, edgecolor=TIER_COLS[t], linewidth=0.8)
        ax.set_xticks([0,1,2]); ax.set_xticklabels(['T1','T2','T3'], fontsize=7.5)
        if ci == 0: ax.set_ylabel(ylabel, fontsize=8)
        if ri == 0:
            ax.set_title(SLABELS[mk], fontsize=8.5,
                         color='#E65100' if mk in ('gpt4o_mini','gpt_5') else '#333',
                         fontweight='bold' if mk=='gpt_5' else 'normal')
        ax.grid(axis='y', lw=0.3, alpha=0.4, color='#ccc')

fig2.suptitle('Safety Tax and Cultural Accuracy: Spotlight on Selected Models\n'
              'Solid=Persona, Faded=Third framing. GPT-5 achieves near-zero safety tax with highest T3 NVAS.',
              fontsize=10, y=1.01)
fig2.tight_layout()
out2 = os.path.join(RESULTS, 'gap3_frontier_spotlight.pdf')
fig2.savefig(out2, bbox_inches='tight', dpi=200); plt.close(fig2)
print(f'Saved: {out2}')
print('Done.')
