"""
Gap 4: Native language framing effect.
EN vs Native (Diff) on safety tax and T3 NVAS.
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
    return np.nan if d == 0 else 1 - abs(row['extracted'] - row['human_mean']) / d

mt_acc = mt[~mt['refusal']].copy()
mt_acc['nvas'] = mt_acc.apply(nvas, axis=1)

PAIRS = [('Personalization','Personalization_Diff','Persona EN','Persona Native'),
         ('Third','Third_Diff','Third EN','Third Native')]

LABEL_MAP = {
    'olmo_3_7b_instruct':'OLMo-7B-IT','olmo_3_32b_instruct':'OLMo-32B-IT',
    'olmo_3_7b_dpo':'OLMo-7B-DPO','olmo_3_7b_sft':'OLMo-7B-SFT',
    'tulu_3_8b_dpo':'Tulu3-DPO','tulu_3_8b_sft':'Tulu3-SFT','tulu_3.1_8b':'Tulu3.1',
    'llama_3.1_8b_instruct':'LLaMA-8B-IT','llama_3.1_8b_base':'LLaMA-8B-Base',
    'gemma_3_4b_it':'Gemma3-4B','gemma_3_12b_it':'Gemma3-12B','gemma_3_27b_it':'Gemma3-27B',
    'gpt4o_mini':'GPT-4o-mini','gpt_5':'GPT-5',
    'qwen2.5_7b_instruct':'Qwen2.5-7B','qwen3_4b_instruct':'Qwen3-4B',
    'qwen3_30b_a3b_instruct':'Qwen3-30B',
    'aya_expanse_8b':'AYA-8B','aya_expanse_32b':'AYA-32B',
    'allam_7b_instruct':'ALLAM-7B','fanar_1_9b_instruct':'FANAR-9B',
    'jais_2_8b_chat':'JAIS-8B','mistral_7b_instruct':'Mistral-7B',
    'olmo_3_32b_dpo':'OLMo-32B-DPO','olmo_3_32b_sft':'OLMo-32B-SFT',
    'olmo_3_7b_base':'OLMo-7B-Base','olmo_3_32b_base':'OLMo-32B-Base',
}
MENA_MODELS = {'ALLAM-7B','FANAR-9B','JAIS-8B','AYA-8B','AYA-32B'}
FRONTIER = {'GPT-4o-mini','GPT-5'}

rows = []
for mk in sorted(mt['model'].unique()):
    label = LABEL_MAP.get(mk, mk)
    for (sh_en, sh_nat, name_en, name_nat) in PAIRS:
        sub_en  = mt[(mt['model']==mk) & (mt['sheet']==sh_en)]
        sub_nat = mt[(mt['model']==mk) & (mt['sheet']==sh_nat)]
        if len(sub_en)==0 or len(sub_nat)==0: continue
        for tier in [1,2,3]:
            r_en  = sub_en[sub_en['tier']==tier]['refusal'].mean()
            r_nat = sub_nat[sub_nat['tier']==tier]['refusal'].mean()
            nv_en = mt_acc[(mt_acc['model']==mk)&(mt_acc['sheet']==sh_en)&(mt_acc['tier']==tier)]['nvas'].mean()
            nv_nat= mt_acc[(mt_acc['model']==mk)&(mt_acc['sheet']==sh_nat)&(mt_acc['tier']==tier)]['nvas'].mean()
            rows.append({'model':mk,'label':label,'framing_pair':name_en[:6],
                         'tier':tier,'refusal_en':r_en,'refusal_nat':r_nat,
                         'nvas_en':nv_en,'nvas_nat':nv_nat,
                         'delta_ref':r_nat-r_en,'delta_nvas':nv_nat-nv_en})

df = pd.DataFrame(rows)

# Safety tax EN vs Native
print('=== Safety tax (T3-T1 refusal) by framing language ===')
for (sh_en, sh_nat, name_en, name_nat) in PAIRS:
    t3_en = mt[(mt['sheet']==sh_en)&(mt['tier']==3)]['refusal'].mean()
    t1_en = mt[(mt['sheet']==sh_en)&(mt['tier']==1)]['refusal'].mean()
    t3_nat= mt[(mt['sheet']==sh_nat)&(mt['tier']==3)]['refusal'].mean()
    t1_nat= mt[(mt['sheet']==sh_nat)&(mt['tier']==1)]['refusal'].mean()
    print(f'{name_en}: tax={t3_en-t1_en:+.3f}  ({name_nat}): tax={t3_nat-t1_nat:+.3f}  '
          f'Δtax={t3_nat-t1_nat-(t3_en-t1_en):+.3f}')

print('\n=== T3 NVAS shift: Native − EN ===')
t3_sub = df[(df['tier']==3) & (df['framing_pair']=='Perso')]
print(t3_sub[['label','nvas_en','nvas_nat','delta_nvas','delta_ref']].sort_values('delta_nvas', ascending=False).round(3).to_string())

# ── FIGURE: 4-panel per framing-pair (refusal + NVAS for T3) ─────────────────
instruct_models = df['label'].unique().tolist()
# Keep only models with non-trivial safety tax OR interesting behaviour
keep = [m for m in instruct_models if LABEL_MAP.get(
    [k for k,v in LABEL_MAP.items() if v==m][0] if any(v==m for v in LABEL_MAP.values()) else '', m, ) != '']
models_show = sorted(instruct_models)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor('white')

for row_i, (sh_en, sh_nat, name_en, name_nat) in enumerate(PAIRS):
    sub = df[(df['tier']==3) & (df['framing_pair']==name_en[:6])].copy()
    sub = sub.set_index('label')
    delta_ref  = (sub['refusal_nat'] - sub['refusal_en']).dropna().sort_values()
    delta_nvas = (sub['nvas_nat']    - sub['nvas_en']).dropna().sort_values(ascending=False)

    for col_i, (delta, xlabel, title) in enumerate([
        (delta_ref,  f'Δ refusal rate (Native − EN)',  f'{name_en} → {name_nat}\nRefusal change'),
        (delta_nvas, f'Δ accepted NVAS (Native − EN)', f'{name_en} → {name_nat}\nNVAS change'),
    ]):
        ax = axes[row_i][col_i]; ax.set_facecolor('#FAFAFA')
        colors_bar = []
        for m in delta.index:
            if m in MENA_MODELS: colors_bar.append('#E53935')
            elif m in FRONTIER:  colors_bar.append('#FF6F00')
            else:                colors_bar.append('#1E88E5')
        ax.barh(range(len(delta)), delta.values, color=colors_bar, alpha=0.8, edgecolor='white')
        ax.axvline(0, color='#333', lw=1.0, ls='--', alpha=0.6)
        ax.set_yticks(range(len(delta))); ax.set_yticklabels(delta.index, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='x', lw=0.3, alpha=0.4, color='#ccc')
        ax.text(0.98, 0.02, f'mean Δ={delta.mean():+.3f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='#333')

handles = [mpatches.Patch(color='#E53935', label='MENA-specialized'),
           mpatches.Patch(color='#FF6F00', label='Frontier (GPT)'),
           mpatches.Patch(color='#1E88E5', label='Western open-source')]
axes[0][0].legend(handles=handles, fontsize=7.5, framealpha=0.9)

fig.suptitle('Native Language Framing Effect on T3 Safety Tax and Cultural Accuracy\n'
             'Positive Δ NVAS = native language improves accuracy; '
             'Negative Δ refusal = native language reduces suppression',
             fontsize=10.5, y=1.01)
fig.tight_layout()
out = os.path.join(RESULTS, 'gap4_native_framing.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'\nSaved: {out}')

# Key stats
t3_acc_en  = mt_acc[(mt_acc['sheet']=='Personalization')&(mt_acc['tier']==3)]['nvas'].mean()
t3_acc_nat = mt_acc[(mt_acc['sheet']=='Personalization_Diff')&(mt_acc['tier']==3)]['nvas'].mean()
tax_en  = mt[(mt['sheet']=='Personalization')&(mt['tier']==3)]['refusal'].mean() - \
          mt[(mt['sheet']=='Personalization')&(mt['tier']==1)]['refusal'].mean()
tax_nat = mt[(mt['sheet']=='Personalization_Diff')&(mt['tier']==3)]['refusal'].mean() - \
          mt[(mt['sheet']=='Personalization_Diff')&(mt['tier']==1)]['refusal'].mean()
print(f'\nOverall: T3 NVAS EN={t3_acc_en:.3f} vs Native={t3_acc_nat:.3f}  (Δ={t3_acc_nat-t3_acc_en:+.3f})')
print(f'Safety tax EN={tax_en:.3f} vs Native={tax_nat:.3f}  (Δ={tax_nat-tax_en:+.3f})')

# FANAR exception highlight
fanar_t3 = df[(df['model']=='fanar_1_9b_instruct') & (df['tier']==3) & (df['framing_pair']=='Perso')]
if len(fanar_t3):
    print(f'\nFANAR exception: refusal EN={fanar_t3["refusal_en"].values[0]:.3f} → '
          f'Native={fanar_t3["refusal_nat"].values[0]:.3f} (increases with native!)')
print('Done.')
