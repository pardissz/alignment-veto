"""
Comprehensive framing, linguistic, and mechanistic analysis.

Experiments:
  F1 — Framing effects: No Mention vs Persona vs Third
         • Refusal rates by framing × tier × model
         • NVAS (closeness to human data) by framing
         • Jensen-Shannon divergence from human distribution
         • Ordering test: is Persona intermediate between NoMention and Third?
  F2 — Linguistic analysis: English vs native language (_Diff sheets)
         • NVAS_native vs NVAS_english per model / language family
         • JS divergence between English and native distributions
         • Refusal rate change: English → native
  F3 — Tier 3 framing effects on logit distributions
         • Refusal heatmap: framing × tier × model
         • Logit entropy by framing (Tier 3 focus)
         • Digit-token probability mass for Tier 3 refusals
  M1 — Mechanistic: cross-framing logit similarity & entropy
         • Cosine similarity of norm_probs: Persona vs Third
         • Entropy of logit distribution by framing
         • Framing-induced shift direction (does Third push higher or lower?)
  M2 — Mechanistic: representation-level RSA
         • Using cached probe activations (5000×33×4096)
         • RSA comparing country / value-range groupings
  M3 — SAE feature analysis by framing (if sae_codes cached)
"""

import os, ast, json, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial.distance import jensenshannon, cosine as cosine_dist
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')

def plog(msg): print(msg, flush=True)

BASE     = '/shared/storage-01/users/zahraei2/mena_normal'
DATA_DIR = os.path.join(BASE, 'MENA_TRANSLATED_reasoning')
RESULTS  = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

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
FAM_COL = {'Arabic':'#E53935','Persian':'#1E88E5','Turkish':'#43A047'}

# Map _Diff prompt columns to language family
DIFF_LANG_MAP = {
    'Arabic No Mention': 'Arabic', 'Persian No Mention': 'Persian', 'Turkey No Mention': 'Turkish',
}
# Country → language family
COUNTRY_FAM = {}
for fam, clist in LANG_FAMILY.items():
    for c in clist:
        COUNTRY_FAM[c] = fam

EXCLUDED_MODELS = {'mistral_7b_instruct'}
ALL_MODELS = sorted([f.replace('.xlsx','') for f in os.listdir(DATA_DIR)
                     if f.endswith('.xlsx') and not f.startswith('_worker')
                     and f.replace('.xlsx','') not in EXCLUDED_MODELS])

# Some xlsx files use a different column prefix than the filename (e.g. gpt_5.xlsx → gpt5)
XLSX_TO_PREFIX = {'gpt_5': 'gpt5'}

plt.rcParams.update({'font.family':'serif','font.size':10,'axes.titlesize':11,
                     'figure.dpi':150,'savefig.dpi':200,'figure.facecolor':'white'})

def safe_parse_probs(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return {}
    try: return json.loads(str(x).replace("'",'"'))
    except:
        try: return ast.literal_eval(str(x))
        except: return {}

def bootstrap_ci(data, n=1000, alpha=0.05, seed=42):
    data = np.asarray(data,dtype=float); data = data[~np.isnan(data)]
    if len(data)<2: return np.nan,np.nan,np.nan
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(data,len(data),replace=True)) for _ in range(n)]
    return np.mean(data), np.percentile(boots,100*alpha/2), np.percentile(boots,100*(1-alpha/2))

def save_fig(fig, fname):
    fig.savefig(os.path.join(RESULTS,fname),bbox_inches='tight'); plt.close(fig)
    plog(f'  Saved: {fname}')

def save_latex(text, fname):
    with open(os.path.join(RESULTS,fname),'w') as f: f.write(text)
    plog(f'  Saved: {fname}')

def latex_table(df, caption, label, col_format=None):
    if col_format is None: col_format = 'l'+'r'*len(df.columns)
    return '\n'.join([r'\begin{table}[htbp]',r'\centering',r'\small',
        f'\\caption{{{caption}}}',f'\\label{{{label}}}',
        df.to_latex(index=True,escape=True,float_format='%.3f',
                    column_format=col_format,na_rep='—'),r'\end{table}'])

def js_divergence(p_dict, q_dict):
    """JS divergence between two probability dicts. Returns NaN if either is empty."""
    if not p_dict or not q_dict: return np.nan
    keys = sorted(set(p_dict)|set(q_dict))
    p = np.array([float(p_dict.get(k,0)) for k in keys])
    q = np.array([float(q_dict.get(k,0)) for k in keys])
    if p.sum()==0 or q.sum()==0: return np.nan
    p = p/p.sum(); q = q/q.sum()
    try: return float(jensenshannon(p,q))
    except: return np.nan

def entropy(prob_dict):
    if not prob_dict: return np.nan
    vals = np.array(list(prob_dict.values()),dtype=float)
    if vals.sum()==0: return np.nan
    vals = vals/vals.sum()
    vals = vals[vals>0]
    return float(-np.sum(vals*np.log2(vals)))

def digit_mass(prob_dict, vmin, vmax):
    """Fraction of probability mass on valid digit tokens."""
    if not prob_dict: return np.nan
    valid = set(str(i) for i in range(int(vmin),int(vmax)+1))
    total = sum(float(v) for v in prob_dict.values())
    if total==0: return np.nan
    on_digit = sum(float(v) for k,v in prob_dict.items() if str(k) in valid)
    return on_digit/total

def cosine_sim(d1, d2):
    if not d1 or not d2: return np.nan
    keys = sorted(set(d1)|set(d2))
    v1 = np.array([float(d1.get(k,0)) for k in keys])
    v2 = np.array([float(d2.get(k,0)) for k in keys])
    if v1.sum()==0 or v2.sum()==0: return np.nan
    v1 = v1/v1.sum(); v2 = v2/v2.sum()
    n = np.linalg.norm(v1)*np.linalg.norm(v2)
    return float(np.dot(v1,v2)/n) if n>0 else np.nan

# ── Load human data ────────────────────────────────────────────────────────────
plog('Loading human data...')
TIERS = pd.read_csv(os.path.join(BASE,'tiers.csv'))
TIERS = TIERS.rename(columns={'No Mention':'question_text'})
TIERS_IDX = TIERS.set_index('question_id')

def load_human_data():
    df = pd.read_excel(os.path.join(BASE,'new_weights_transposed.xlsx'))
    df = df.rename(columns={'question_number':'question_id'})
    result = {'question_id': df['question_id'].values}
    for c in COUNTRIES:
        if c not in df.columns: continue
        means, dists = [], []
        for v in df[c]:
            if pd.isna(v): means.append(np.nan); dists.append({}); continue
            try:
                tup = ast.literal_eval(str(v))
                means.append(float(tup[0]))
                dists.append({k: float(str(vv).strip('%'))/100 for k,vv in tup[1].items()})
            except: means.append(np.nan); dists.append({})
        result[f'{c}_mean'] = means
        result[f'{c}_dist'] = dists
    return pd.DataFrame(result).set_index('question_id')
HUMAN = load_human_data()
plog(f'  Human data: {len(HUMAN)} questions × {len(COUNTRIES)} countries')

# ── Rebuild / load master table ────────────────────────────────────────────────
plog(f'Building master table for {len(ALL_MODELS)} models...')
t0 = time.time()

def load_no_mention_sheet(sh, model_name, sheet_key):
    col_model = XLSX_TO_PREFIX.get(model_name, model_name)
    prefix = 'No Mention'
    if 'question_id' in sh.columns:
        sh = sh.copy(); sh['_qid'] = sh['question_id'].astype(int)
    else:
        sh = sh.copy(); sh['_qid'] = np.arange(1,len(sh)+1)
    ext_col = f'{prefix}_{col_model}_extracted_number'
    ref_col = f'{prefix}_{col_model}_refusal'
    prb_col = f'{prefix}_{col_model}_normalized_probs'
    if ext_col not in sh.columns: return pd.DataFrame()
    df_out = pd.DataFrame({
        'model': model_name, 'sheet': sheet_key, 'country': 'NEUTRAL',
        'question_id': sh['_qid'].values,
        'extracted': pd.to_numeric(sh[ext_col],errors='coerce').values,
        'refusal': sh[ref_col].notna().values if ref_col in sh.columns else np.zeros(len(sh),dtype=bool),
        'norm_probs': sh[prb_col].apply(safe_parse_probs).values if prb_col in sh.columns else [{}]*len(sh),
        'human_mean': np.nan, 'human_dist': [{}]*len(sh),
    })
    df_out = df_out.merge(
        TIERS_IDX[['Tier','Min','MAX']].rename(columns={'Tier':'tier','Min':'vmin','MAX':'vmax'}),
        left_on='question_id', right_index=True, how='left')
    return df_out

def load_no_mention_diff_sheet(sh, model_name):
    """No Mention Diff: Arabic/Persian/Turkish prompt variants."""
    col_model = XLSX_TO_PREFIX.get(model_name, model_name)
    if 'question_id' in sh.columns:
        sh = sh.copy(); sh['_qid'] = sh['question_id'].astype(int)
    else:
        sh = sh.copy(); sh['_qid'] = np.arange(1,len(sh)+1)
    parts = []
    for lang_col, fam in DIFF_LANG_MAP.items():
        ext_col = f'{lang_col}_{col_model}_extracted_number'
        ref_col = f'{lang_col}_{col_model}_refusal'
        prb_col = f'{lang_col}_{col_model}_normalized_probs'
        if ext_col not in sh.columns: continue
        df_out = pd.DataFrame({
            'model': model_name, 'sheet': 'No Mention Diff',
            'country': f'NEUTRAL_{fam}', 'lang_family': fam,
            'question_id': sh['_qid'].values,
            'extracted': pd.to_numeric(sh[ext_col],errors='coerce').values,
            'refusal': sh[ref_col].notna().values if ref_col in sh.columns else np.zeros(len(sh),dtype=bool),
            'norm_probs': sh[prb_col].apply(safe_parse_probs).values if prb_col in sh.columns else [{}]*len(sh),
            'human_mean': np.nan, 'human_dist': [{}]*len(sh),
        })
        df_out = df_out.merge(
            TIERS_IDX[['Tier','Min','MAX']].rename(columns={'Tier':'tier','Min':'vmin','MAX':'vmax'}),
            left_on='question_id', right_index=True, how='left')
        parts.append(df_out)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def load_country_sheet(sh, model_name, sheet_key):
    if 'question_id' not in sh.columns: return pd.DataFrame()
    col_model = XLSX_TO_PREFIX.get(model_name, model_name)
    sh = sh.copy(); sh['question_id'] = sh['question_id'].astype(int)
    rows = []
    for country in COUNTRIES:
        prefix = f'{country}_{col_model}'
        ext_col = f'{prefix}_extracted_number'
        ref_col = f'{prefix}_refusal'
        prb_col = f'{prefix}_normalized_probs'
        if ext_col not in sh.columns: continue
        h_means = HUMAN[f'{country}_mean'].reindex(sh['question_id']).values if f'{country}_mean' in HUMAN.columns else np.full(len(sh),np.nan)
        h_dists = HUMAN[f'{country}_dist'].reindex(sh['question_id']).values if f'{country}_dist' in HUMAN.columns else np.array([{}]*len(sh))
        c_df = pd.DataFrame({
            'model': model_name, 'sheet': sheet_key, 'country': country,
            'lang_family': COUNTRY_FAM.get(country,'Unknown'),
            'question_id': sh['question_id'].values,
            'extracted': pd.to_numeric(sh[ext_col],errors='coerce').values,
            'refusal': sh[ref_col].notna().values if ref_col in sh.columns else np.zeros(len(sh),dtype=bool),
            'norm_probs': sh[prb_col].apply(safe_parse_probs).values if prb_col in sh.columns else [{}]*len(sh),
            'human_mean': h_means, 'human_dist': h_dists,
        })
        rows.append(c_df)
    if not rows: return pd.DataFrame()
    out = pd.concat(rows,ignore_index=True)
    out = out.merge(
        TIERS_IDX[['Tier','Min','MAX']].rename(columns={'Tier':'tier','Min':'vmin','MAX':'vmax'}),
        left_on='question_id', right_index=True, how='left')
    return out

all_parts = []
for i, model_name in enumerate(ALL_MODELS):
    path = os.path.join(DATA_DIR, f'{model_name}.xlsx')
    if not os.path.exists(path): plog(f'  [{i+1}] MISSING: {model_name}'); continue
    plog(f'  [{i+1}/{len(ALL_MODELS)}] {model_name}')
    xl = pd.ExcelFile(path)
    for sh_name in xl.sheet_names:
        df_sh = pd.read_excel(xl, sheet_name=sh_name)
        if sh_name == 'No Mention':
            part = load_no_mention_sheet(df_sh, model_name, 'No Mention')
        elif sh_name == 'No Mention Diff':
            part = load_no_mention_diff_sheet(df_sh, model_name)
        else:
            part = load_country_sheet(df_sh, model_name, sh_name)
        if len(part)>0: all_parts.append(part)

MASTER = pd.concat(all_parts, ignore_index=True)
MASTER['tier'] = pd.to_numeric(MASTER['tier'],errors='coerce').astype('Int64')
MASTER['refusal'] = MASTER['refusal'].astype(bool)
MASTER['extracted'] = pd.to_numeric(MASTER['extracted'],errors='coerce')
MASTER['vmin'] = pd.to_numeric(MASTER['vmin'],errors='coerce')
MASTER['vmax'] = pd.to_numeric(MASTER['vmax'],errors='coerce')
if 'lang_family' not in MASTER.columns:
    MASTER['lang_family'] = MASTER['country'].map(COUNTRY_FAM)

plog(f'Master table: {len(MASTER):,} rows, {MASTER.model.nunique()} models, {MASTER.sheet.unique().tolist()}')
MASTER.to_pickle(os.path.join(RESULTS,'master_table.pkl'))
plog('  Saved master_table.pkl')

# ── NVAS helper ───────────────────────────────────────────────────────────────
def compute_nvas(df):
    rng = (df['vmax']-df['vmin']).replace(0,np.nan)
    return 1.0 - (df['extracted']-df['human_mean']).abs()/rng

# ─────────────────────────────────────────────────────────────────────────────
# EXP F1: Framing Effects
# ─────────────────────────────────────────────────────────────────────────────
plog('\n=== Exp F1: Framing Effects ===')

FRAMING_SHEETS = {
    'No Mention':       ('EN', 'neutral'),
    'Personalization':  ('EN', 'persona'),
    'Third':            ('EN', 'third'),
}
FRAMING_DIFF_SHEETS = {
    'Personalization_Diff': ('Native', 'persona'),
    'Third_Diff':           ('Native', 'third'),
}

# F1a: Refusal rate by framing × tier × model
rows_f1 = []
for model, m_df in MASTER.groupby('model'):
    for sheet, (lang, framing) in {**FRAMING_SHEETS, **FRAMING_DIFF_SHEETS}.items():
        sub = m_df[m_df['sheet']==sheet]
        for tier in [1,2,3]:
            t_df = sub[sub['tier']==tier]
            if len(t_df)==0: continue
            obs,lo,hi = bootstrap_ci(t_df['refusal'].astype(float).values)
            rows_f1.append({'model':model,'sheet':sheet,'framing':framing,'lang':lang,
                            'tier':tier,'refusal_rate':obs,'ci_lo':lo,'ci_hi':hi,'n':len(t_df)})
f1_df = pd.DataFrame(rows_f1)
f1_df.to_csv(os.path.join(RESULTS,'f1_refusal_by_framing.csv'),index=False)

# Plot: refusal rate heatmap (model × framing) for Tier 3
fig, axes = plt.subplots(1,3,figsize=(18,8))
tiers = [1,2,3]
for ax_idx, tier in enumerate(tiers):
    sub = f1_df[f1_df['tier']==tier]
    piv = sub.pivot_table(index='model',columns='sheet',values='refusal_rate')
    sheet_order = ['No Mention','Personalization','Personalization_Diff','Third','Third_Diff']
    piv = piv.reindex(columns=[s for s in sheet_order if s in piv.columns])
    if piv.empty: continue
    ax = axes[ax_idx]
    sns.heatmap(piv,ax=ax,cmap='Reds',vmin=0,vmax=1,annot=True,fmt='.2f',
                linewidths=0.5,cbar_kws={'label':'Refusal Rate'},annot_kws={'size':6})
    ax.set_title(f'Tier {tier} Refusal Rate by Framing')
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(axis='x',rotation=45,labelsize=7)
    ax.tick_params(axis='y',labelsize=6)
fig.suptitle('Refusal Rate by Framing, Tier, and Model',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
save_fig(fig,'f1a_refusal_framing_heatmap.pdf')

# F1b: NVAS by framing for country-specific sheets
rows_f1b = []
for model, m_df in MASTER.groupby('model'):
    for sheet, (lang, framing) in {**FRAMING_SHEETS, **FRAMING_DIFF_SHEETS}.items():
        sub = m_df[(m_df['sheet']==sheet) & ~m_df['refusal'] &
                   m_df['human_mean'].notna() & m_df['extracted'].notna() &
                   m_df['vmin'].notna() & m_df['vmax'].notna()]
        if len(sub)==0: continue
        nvas_vals = compute_nvas(sub).values
        for tier in [1,2,3]:
            t_sub = sub[sub['tier']==tier]
            if len(t_sub)==0: continue
            obs,lo,hi = bootstrap_ci(compute_nvas(t_sub).values)
            rows_f1b.append({'model':model,'sheet':sheet,'framing':framing,'lang':lang,
                             'tier':tier,'nvas':obs,'ci_lo':lo,'ci_hi':hi,'n':len(t_sub)})
f1b_df = pd.DataFrame(rows_f1b)
f1b_df.to_csv(os.path.join(RESULTS,'f1b_nvas_by_framing.csv'),index=False)

# Plot: NVAS by framing per tier, averaged over models
fig, axes = plt.subplots(1,3,figsize=(18,6))
FCOLORS = {'No Mention':'#607D8B','Personalization':'#1976D2','Third':'#388E3C',
           'Personalization_Diff':'#7986CB','Third_Diff':'#81C784'}
for ax_idx, tier in enumerate([1,2,3]):
    ax = axes[ax_idx]
    sub = f1b_df[f1b_df['tier']==tier]
    if sub.empty: continue
    avg = sub.groupby('sheet')['nvas'].mean().reindex(
        ['No Mention','Personalization','Personalization_Diff','Third','Third_Diff']).dropna()
    bars = ax.bar(range(len(avg)), avg.values, color=[FCOLORS.get(s,'gray') for s in avg.index],alpha=0.8)
    ax.set_xticks(range(len(avg)))
    ax.set_xticklabels([s.replace('_',' ') for s in avg.index],rotation=30,ha='right',fontsize=8)
    ax.set_ylabel('Mean NVAS'); ax.set_ylim(0, 0.8)
    ax.set_title(f'Tier {tier}: NVAS by Framing')
    ax.axhline(avg.get('No Mention',np.nan),ls='--',lw=1,color='#607D8B',alpha=0.5)
fig.suptitle('NVAS by Framing (averaged across models & countries)',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
save_fig(fig,'f1b_nvas_by_framing.pdf')

# F1c: Is Persona intermediate? Test per question × model × country
plog('  F1c: Persona intermediary test...')
rows_f1c = []
sheets = {
    'no_mention': MASTER[MASTER['sheet']=='No Mention'][['model','question_id','extracted','norm_probs']],
    'persona':    MASTER[MASTER['sheet']=='Personalization'][['model','question_id','country','extracted','norm_probs','human_mean','vmin','vmax']],
    'third':      MASTER[MASTER['sheet']=='Third'][['model','question_id','country','extracted','norm_probs','human_mean','vmin','vmax']],
}
nm_df = sheets['no_mention'].rename(columns={'extracted':'nm_ext','norm_probs':'nm_probs'})
pe_df = sheets['persona'].rename(columns={'extracted':'pe_ext','norm_probs':'pe_probs'})
th_df = sheets['third'].rename(columns={'extracted':'th_ext','norm_probs':'th_probs'})

merged = pe_df.merge(th_df,on=['model','question_id','country'],suffixes=('','_th'))
merged = merged.merge(nm_df,on=['model','question_id'])

valid = (merged['human_mean'].notna() & merged['nm_ext'].notna() &
         merged['pe_ext'].notna() & merged['th_ext'].notna() &
         merged['vmin'].notna() & merged['vmax'].notna())
merged = merged[valid].copy()
rng = (merged['vmax']-merged['vmin']).replace(0,np.nan)

merged['dist_nm'] = (merged['nm_ext']-merged['human_mean']).abs()/rng
merged['dist_pe'] = (merged['pe_ext']-merged['human_mean']).abs()/rng
merged['dist_th'] = (merged['th_ext']-merged['human_mean']).abs()/rng

# For each row: is persona strictly between no_mention and third?
merged['nm_closer_than_pe'] = merged['dist_nm'] > merged['dist_pe']   # nm farther than pe
merged['pe_closer_than_th'] = merged['dist_pe'] > merged['dist_th']   # pe farther than th (third closer)
merged['pe_intermediate']   = merged['nm_closer_than_pe'] & merged['pe_closer_than_th']  # NM > PE > TH distance (third is closest)
merged['th_closest'] = merged['dist_th'] < merged['dist_pe']

for model, g in merged.groupby('model'):
    rows_f1c.append({
        'model': model,
        'pct_nm_farther_than_pe': g['nm_closer_than_pe'].mean(),
        'pct_th_closer_than_pe':  g['pe_closer_than_th'].mean(),
        'pct_pe_intermediate':    g['pe_intermediate'].mean(),
        'mean_dist_nm': g['dist_nm'].mean(),
        'mean_dist_pe': g['dist_pe'].mean(),
        'mean_dist_th': g['dist_th'].mean(),
        'n': len(g),
    })
f1c_df = pd.DataFrame(rows_f1c).sort_values('mean_dist_th')
f1c_df.to_csv(os.path.join(RESULTS,'f1c_persona_intermediate.csv'),index=False)

# Plot: mean distance from human data by framing
fig, axes = plt.subplots(1,2,figsize=(16,7))
ax = axes[0]
models_sorted = f1c_df.sort_values('mean_dist_nm').model.tolist()
x = np.arange(len(models_sorted)); w = 0.25
for i, (col, lbl, clr) in enumerate([('mean_dist_nm','No Mention','#607D8B'),
                                       ('mean_dist_pe','Persona','#1976D2'),
                                       ('mean_dist_th','Third','#388E3C')]):
    vals = f1c_df.set_index('model').reindex(models_sorted)[col].fillna(0).values
    ax.bar(x+i*w, vals, w, label=lbl, color=clr, alpha=0.85)
ax.set_xticks(x+w); ax.set_xticklabels([m.replace('_',' ') for m in models_sorted],rotation=45,ha='right',fontsize=7)
ax.set_ylabel('Mean |model − human| / range'); ax.set_title('Distance from Human Data by Framing')
ax.legend()

ax2 = axes[1]
piv2 = f1c_df.set_index('model')[['mean_dist_nm','mean_dist_pe','mean_dist_th']].reindex(models_sorted)
piv2.columns=['No Mention','Persona','Third']
for i, model in enumerate(models_sorted):
    vals = piv2.loc[model].values
    ax2.plot(vals,'-o',alpha=0.6,lw=1,color='gray',ms=4)
ax2.set_xticks([0,1,2]); ax2.set_xticklabels(['No Mention','Persona','Third'])
avg_vals = piv2.mean()
ax2.plot(avg_vals.values,'o-',lw=2.5,color='#D32F2F',ms=8,label='Mean across models')
ax2.set_ylabel('Mean distance from human'); ax2.set_title('Framing → Human Proximity (lower = closer)')
ax2.legend()
fig.suptitle('Is Persona intermediate between No-Mention and Third?',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
save_fig(fig,'f1c_persona_intermediate.pdf')

# LaTeX summary table for F1c
tbl_c = f1c_df[['model','mean_dist_nm','mean_dist_pe','mean_dist_th','pct_pe_intermediate']].copy()
tbl_c.columns = ['Model','Dist-NoMention','Dist-Persona','Dist-Third','\\%Persona Intermediate']
tbl_c = tbl_c.set_index('Model')
save_latex(latex_table(tbl_c.round(3),
    'Framing proximity to human data. Distance = |model$-$human|/range. '
    'Persona Intermediate\\% = fraction of questions where NM\$>\$Persona\$>\$Third distance.',
    'tab:f1c'), 'f1c_persona_intermediate.tex')

# F1d: JS divergence from human distribution by framing
plog('  F1d: JS divergence from human distribution...')
rows_f1d = []
for model, m_df in MASTER.groupby('model'):
    for sheet, (lang, framing) in {**FRAMING_SHEETS, **FRAMING_DIFF_SHEETS}.items():
        sub = m_df[(m_df['sheet']==sheet) & m_df['human_dist'].apply(bool) &
                   m_df['norm_probs'].apply(bool)]
        if len(sub)==0: continue
        jsd_vals = []
        for _, row in sub.iterrows():
            jsd = js_divergence(row['human_dist'], row['norm_probs'])
            if not np.isnan(jsd): jsd_vals.append(jsd)
        if not jsd_vals: continue
        obs,lo,hi = bootstrap_ci(jsd_vals)
        rows_f1d.append({'model':model,'sheet':sheet,'framing':framing,'lang':lang,'jsd':obs,'ci_lo':lo,'ci_hi':hi,'n':len(jsd_vals)})
f1d_df = pd.DataFrame(rows_f1d)
f1d_df.to_csv(os.path.join(RESULTS,'f1d_jsd_from_human.csv'),index=False)

fig, ax = plt.subplots(figsize=(12,6))
models_u = sorted(f1d_df['model'].unique())
x = np.arange(len(models_u)); w = 0.15
sheet_order_jsd = ['No Mention','Personalization','Personalization_Diff','Third','Third_Diff']
for i, sheet in enumerate(sheet_order_jsd):
    sub = f1d_df[f1d_df['sheet']==sheet].set_index('model').reindex(models_u)
    if sub.empty: continue
    ax.bar(x+i*w, sub['jsd'].fillna(0).values, w, label=sheet.replace('_',' '),
           color=list(FCOLORS.values())[i], alpha=0.85)
ax.set_xticks(x+2*w); ax.set_xticklabels([m.replace('_',' ') for m in models_u],rotation=45,ha='right',fontsize=7)
ax.set_ylabel('Mean JS Divergence from Human Distribution'); ax.set_title('Logit-Human Divergence by Framing')
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig,'f1d_jsd_from_human.pdf')

plog('  Exp F1 done.')

# ─────────────────────────────────────────────────────────────────────────────
# EXP F2: Linguistic Analysis (English vs Native)
# ─────────────────────────────────────────────────────────────────────────────
plog('\n=== Exp F2: Linguistic Analysis ===')

# F2a: NVAS English vs Native per model × language family
rows_f2a = []
for model, m_df in MASTER.groupby('model'):
    for framing_en, framing_nat in [('Personalization','Personalization_Diff'),('Third','Third_Diff')]:
        for fam, countries in LANG_FAMILY.items():
            sub_en  = m_df[(m_df['sheet']==framing_en)  & m_df['country'].isin(countries) &
                           ~m_df['refusal'] & m_df['human_mean'].notna() & m_df['extracted'].notna() &
                           m_df['vmin'].notna() & m_df['vmax'].notna()]
            sub_nat = m_df[(m_df['sheet']==framing_nat) & m_df['country'].isin(countries) &
                           ~m_df['refusal'] & m_df['human_mean'].notna() & m_df['extracted'].notna() &
                           m_df['vmin'].notna() & m_df['vmax'].notna()]
            if len(sub_en)<5 or len(sub_nat)<5: continue
            nv_en  = compute_nvas(sub_en).dropna()
            nv_nat = compute_nvas(sub_nat).dropna()
            obs_en,lo_en,hi_en   = bootstrap_ci(nv_en.values)
            obs_nat,lo_nat,hi_nat = bootstrap_ci(nv_nat.values)
            rows_f2a.append({'model':model,'framing':framing_en.replace('Personalization','Persona').replace('Third','Third'),
                             'lang_family':fam,'nvas_en':obs_en,'nvas_nat':obs_nat,
                             'ci_en_lo':lo_en,'ci_en_hi':hi_en,'ci_nat_lo':lo_nat,'ci_nat_hi':hi_nat,
                             'n_en':len(nv_en),'n_nat':len(nv_nat)})
f2a_df = pd.DataFrame(rows_f2a)
f2a_df.to_csv(os.path.join(RESULTS,'f2a_nvas_lang.csv'),index=False)

# Plot: NVAS gain (native - english) by model × language family
f2a_df['nvas_gain'] = f2a_df['nvas_nat'] - f2a_df['nvas_en']
fig, axes = plt.subplots(1,2,figsize=(16,7))
for ax_idx, framing in enumerate(['Persona','Third']):
    ax = axes[ax_idx]
    sub = f2a_df[f2a_df['framing'].str.contains(framing)]
    if sub.empty: continue
    piv = sub.pivot_table(index='model',columns='lang_family',values='nvas_gain')
    piv_plot = piv.reindex(columns=['Arabic','Persian','Turkish']).fillna(0)
    x = np.arange(len(piv_plot)); w=0.25
    for i, fam in enumerate(['Arabic','Persian','Turkish']):
        if fam not in piv_plot.columns: continue
        ax.bar(x+i*w, piv_plot[fam].values, w, label=fam, color=FAM_COL[fam], alpha=0.85)
    ax.axhline(0,color='black',lw=0.8,ls='--')
    ax.set_xticks(x+w); ax.set_xticklabels([m.replace('_',' ') for m in piv_plot.index],rotation=45,ha='right',fontsize=7)
    ax.set_ylabel('NVAS gain (native − english)'); ax.set_title(f'{framing}: NVAS change when switching to native language')
    ax.legend()
fig.suptitle('Does native language improve alignment with human data?',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
save_fig(fig,'f2a_nvas_language_gain.pdf')

# F2b: JS divergence between English and Native distributions
plog('  F2b: JS divergence between English and native distributions...')
rows_f2b = []
for model, m_df in MASTER.groupby('model'):
    for framing_en, framing_nat in [('Personalization','Personalization_Diff'),('Third','Third_Diff')]:
        sub_en  = m_df[m_df['sheet']==framing_en].set_index(['country','question_id'])
        sub_nat = m_df[m_df['sheet']==framing_nat].set_index(['country','question_id'])
        common = sub_en.index.intersection(sub_nat.index)
        if len(common)==0: continue
        jsd_vals_by_fam = {fam: [] for fam in LANG_FAMILY}
        for (country, qid) in common:
            fam = COUNTRY_FAM.get(country)
            if fam is None: continue
            p_en  = sub_en.loc[(country,qid),'norm_probs']
            p_nat = sub_nat.loc[(country,qid),'norm_probs']
            jsd = js_divergence(p_en, p_nat)
            if not np.isnan(jsd): jsd_vals_by_fam[fam].append(jsd)
        for fam, vals in jsd_vals_by_fam.items():
            if not vals: continue
            obs,lo,hi = bootstrap_ci(vals)
            rows_f2b.append({'model':model,'framing':framing_en,'lang_family':fam,'jsd_en_nat':obs,'ci_lo':lo,'ci_hi':hi,'n':len(vals)})
f2b_df = pd.DataFrame(rows_f2b)
f2b_df.to_csv(os.path.join(RESULTS,'f2b_jsd_en_native.csv'),index=False)

# Plot: JS divergence English vs Native by language family
fig, axes = plt.subplots(1,2,figsize=(16,6))
for ax_idx, framing in enumerate(['Personalization','Third']):
    ax = axes[ax_idx]
    sub = f2b_df[f2b_df['framing']==framing]
    if sub.empty: continue
    piv = sub.pivot_table(index='model',columns='lang_family',values='jsd_en_nat').fillna(0)
    x = np.arange(len(piv)); w=0.25
    for i, fam in enumerate(['Arabic','Persian','Turkish']):
        if fam not in piv.columns: continue
        ax.bar(x+i*w, piv[fam].values, w, label=fam, color=FAM_COL[fam], alpha=0.85)
    ax.set_xticks(x+w); ax.set_xticklabels([m.replace('_',' ') for m in piv.index],rotation=45,ha='right',fontsize=7)
    ax.set_ylabel('JS Divergence (EN vs Native)'); ax.set_title(f'{framing}: EN vs Native distribution divergence')
    ax.legend()
fig.suptitle('How much does switching to native language change logit distributions?',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
save_fig(fig,'f2b_jsd_en_native.pdf')

# F2c: Refusal rate: English vs Native
rows_f2c = []
for model, m_df in MASTER.groupby('model'):
    for framing_en, framing_nat in [('Personalization','Personalization_Diff'),('Third','Third_Diff')]:
        for fam, countries in LANG_FAMILY.items():
            sub_en  = m_df[(m_df['sheet']==framing_en)  & m_df['country'].isin(countries)]
            sub_nat = m_df[(m_df['sheet']==framing_nat) & m_df['country'].isin(countries)]
            if len(sub_en)==0 or len(sub_nat)==0: continue
            r_en  = sub_en['refusal'].astype(float).mean()
            r_nat = sub_nat['refusal'].astype(float).mean()
            rows_f2c.append({'model':model,'framing':framing_en,'lang_family':fam,'refusal_en':r_en,'refusal_nat':r_nat,'refusal_gain':r_nat-r_en})
f2c_df = pd.DataFrame(rows_f2c)
f2c_df.to_csv(os.path.join(RESULTS,'f2c_refusal_lang.csv'),index=False)

fig, ax = plt.subplots(figsize=(12,6))
sub_third = f2c_df[f2c_df['framing']=='Third']
piv_r = sub_third.pivot_table(index='model',columns='lang_family',values='refusal_gain').fillna(0)
x = np.arange(len(piv_r)); w=0.25
for i, fam in enumerate(['Arabic','Persian','Turkish']):
    if fam not in piv_r.columns: continue
    ax.bar(x+i*w, piv_r[fam].values, w, label=fam, color=FAM_COL[fam], alpha=0.85)
ax.axhline(0,color='black',lw=0.8,ls='--')
ax.set_xticks(x+w); ax.set_xticklabels([m.replace('_',' ') for m in piv_r.index],rotation=45,ha='right',fontsize=7)
ax.set_ylabel('Refusal rate increase (native − EN)'); ax.set_title('Third framing: does native language increase refusals?')
ax.legend()
fig.tight_layout(); save_fig(fig,'f2c_refusal_lang_change.pdf')

plog('  Exp F2 done.')

# ─────────────────────────────────────────────────────────────────────────────
# EXP F3: Tier 3 Framing × Logit Distributions
# ─────────────────────────────────────────────────────────────────────────────
plog('\n=== Exp F3: Tier 3 Framing Effects on Logit Distributions ===')

# F3a: Logit entropy by framing × tier
rows_f3a = []
for model, m_df in MASTER.groupby('model'):
    for sheet, (lang, framing) in {**FRAMING_SHEETS, **FRAMING_DIFF_SHEETS}.items():
        sub = m_df[m_df['sheet']==sheet]
        for tier in [1,2,3]:
            t_df = sub[sub['tier']==tier]
            if len(t_df)==0: continue
            ent_vals = t_df['norm_probs'].apply(entropy).dropna().values
            if len(ent_vals)==0: continue
            obs,lo,hi = bootstrap_ci(ent_vals)
            rows_f3a.append({'model':model,'sheet':sheet,'framing':framing,'lang':lang,'tier':tier,'entropy':obs,'ci_lo':lo,'ci_hi':hi,'n':len(ent_vals)})
f3a_df = pd.DataFrame(rows_f3a)
f3a_df.to_csv(os.path.join(RESULTS,'f3a_entropy_by_framing.csv'),index=False)

fig, axes = plt.subplots(1,3,figsize=(18,6))
for ax_idx, tier in enumerate([1,2,3]):
    ax = axes[ax_idx]
    sub = f3a_df[f3a_df['tier']==tier]
    piv = sub.pivot_table(index='model',columns='sheet',values='entropy')
    sheet_order = ['No Mention','Personalization','Personalization_Diff','Third','Third_Diff']
    piv = piv.reindex(columns=[s for s in sheet_order if s in piv.columns])
    if piv.empty: continue
    sns.heatmap(piv,ax=ax,cmap='YlOrRd',annot=True,fmt='.2f',linewidths=0.5,
                cbar_kws={'label':'Entropy (bits)'},annot_kws={'size':6})
    ax.set_title(f'Tier {tier}: Logit Entropy by Framing')
    ax.tick_params(axis='x',rotation=45,labelsize=7)
    ax.tick_params(axis='y',labelsize=6)
fig.suptitle('Logit Distribution Entropy by Framing and Tier',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
save_fig(fig,'f3a_entropy_heatmap.pdf')

# F3b: Digit-token probability mass on Tier 3 refusals by framing
rows_f3b = []
for model, m_df in MASTER.groupby('model'):
    for sheet, (lang, framing) in {**FRAMING_SHEETS, **FRAMING_DIFF_SHEETS}.items():
        sub = m_df[(m_df['sheet']==sheet) & m_df['refusal'] &
                   (m_df['tier']==3) & m_df['vmin'].notna() & m_df['vmax'].notna()]
        if len(sub)<3: continue
        dm_vals = []
        for _, row in sub.iterrows():
            dm = digit_mass(row['norm_probs'], row['vmin'], row['vmax'])
            if not np.isnan(dm): dm_vals.append(dm)
        if not dm_vals: continue
        obs,lo,hi = bootstrap_ci(dm_vals)
        rows_f3b.append({'model':model,'sheet':sheet,'framing':framing,'digit_mass':obs,'ci_lo':lo,'ci_hi':hi,'n':len(dm_vals)})
f3b_df = pd.DataFrame(rows_f3b)
f3b_df.to_csv(os.path.join(RESULTS,'f3b_digit_mass_refusals.csv'),index=False)

if len(f3b_df)>0:
    fig, ax = plt.subplots(figsize=(12,6))
    models_u = sorted(f3b_df['model'].unique())
    x = np.arange(len(models_u)); w=0.15
    for i, sheet in enumerate(['No Mention','Personalization','Personalization_Diff','Third','Third_Diff']):
        sub = f3b_df[f3b_df['sheet']==sheet].set_index('model').reindex(models_u)
        if sub.empty: continue
        ax.bar(x+i*w, sub['digit_mass'].fillna(0).values, w, label=sheet.replace('_',' '),
               color=list(FCOLORS.values())[i], alpha=0.85)
    ax.set_xticks(x+2*w); ax.set_xticklabels([m.replace('_',' ') for m in models_u],rotation=45,ha='right',fontsize=7)
    ax.set_ylabel('Digit-token probability mass'); ax.set_title('Tier 3 Refusals: How much mass stays on valid digits?')
    ax.legend(fontsize=8)
    fig.tight_layout(); save_fig(fig,'f3b_digit_mass_refusals.pdf')

plog('  Exp F3 done.')

# ─────────────────────────────────────────────────────────────────────────────
# EXP M1: Mechanistic — Cross-framing Logit Similarity & Entropy
# ─────────────────────────────────────────────────────────────────────────────
plog('\n=== Exp M1: Mechanistic — Cross-framing Logit Similarity ===')

# M1a: Cosine similarity between Persona and Third logit distributions
# (same model × country × question)
plog('  M1a: Persona vs Third cosine similarity...')
pe_sub = MASTER[MASTER['sheet']=='Personalization'][['model','question_id','country','norm_probs','tier']].copy()
th_sub = MASTER[MASTER['sheet']=='Third'][['model','question_id','country','norm_probs','tier']].copy()

merged_m1 = pe_sub.merge(th_sub,on=['model','question_id','country'],suffixes=('_pe','_th'))
rows_m1a = []
for model, g in merged_m1.groupby('model'):
    cs_all, cs_t3 = [], []
    for _, row in g.iterrows():
        cs = cosine_sim(row['norm_probs_pe'], row['norm_probs_th'])
        if np.isnan(cs): continue
        cs_all.append(cs)
        if row.get('tier_pe',None)==3: cs_t3.append(cs)
    obs,lo,hi = bootstrap_ci(cs_all)
    obs3,lo3,hi3 = bootstrap_ci(cs_t3) if cs_t3 else (np.nan,np.nan,np.nan)
    rows_m1a.append({'model':model,'cosine_all':obs,'ci_lo':lo,'ci_hi':hi,
                     'cosine_tier3':obs3,'ci3_lo':lo3,'ci3_hi':hi3,'n':len(cs_all)})
m1a_df = pd.DataFrame(rows_m1a).sort_values('cosine_all',ascending=False)
m1a_df.to_csv(os.path.join(RESULTS,'m1a_cosine_persona_third.csv'),index=False)

fig, ax = plt.subplots(figsize=(12,5))
x = np.arange(len(m1a_df))
ax.barh(x, m1a_df['cosine_all'].values, alpha=0.85, color='#1976D2', label='All tiers')
ax.barh(x, m1a_df['cosine_tier3'].fillna(0).values, alpha=0.6, color='#F44336', label='Tier 3 only')
ax.set_yticks(x); ax.set_yticklabels([m.replace('_',' ') for m in m1a_df['model']],fontsize=7)
ax.set_xlabel('Cosine similarity (Persona vs Third norm_probs)')
ax.set_title('How similar are Persona and Third-person logit distributions?')
ax.axvline(1.0,lw=0.8,color='black',ls='--')
ax.legend()
fig.tight_layout(); save_fig(fig,'m1a_cosine_persona_third.pdf')

# M1b: Entropy across framings — does framing make model more/less certain?
# Already computed above (f3a). Here compute per-question framing entropy change.
plog('  M1b: Entropy shift: No Mention → Persona → Third...')
nm_ent  = MASTER[MASTER['sheet']=='No Mention'][['model','question_id','norm_probs','tier']].copy()
pe_ent  = MASTER[MASTER['sheet']=='Personalization'][['model','question_id','country','norm_probs','tier']].copy()
th_ent  = MASTER[MASTER['sheet']=='Third'][['model','question_id','country','norm_probs','tier']].copy()
nm_ent['ent'] = nm_ent['norm_probs'].apply(entropy)
pe_ent['ent'] = pe_ent['norm_probs'].apply(entropy)
th_ent['ent'] = th_ent['norm_probs'].apply(entropy)

rows_m1b = []
for model in MASTER['model'].unique():
    nm_mod = nm_ent[nm_ent['model']==model].set_index('question_id')['ent']
    pe_mod = pe_ent[pe_ent['model']==model].groupby('question_id')['ent'].mean()
    th_mod = th_ent[th_ent['model']==model].groupby('question_id')['ent'].mean()
    common = nm_mod.index.intersection(pe_mod.index).intersection(th_mod.index)
    if len(common)<5: continue
    rows_m1b.append({'model':model,'ent_nm':nm_mod[common].mean(),'ent_pe':pe_mod[common].mean(),'ent_th':th_mod[common].mean(),'n':len(common)})
m1b_df = pd.DataFrame(rows_m1b)
m1b_df.to_csv(os.path.join(RESULTS,'m1b_entropy_shift.csv'),index=False)

fig, axes = plt.subplots(1,2,figsize=(16,6))
ax = axes[0]
models_b = m1b_df.sort_values('ent_nm').model.tolist()
x = np.arange(len(models_b)); w=0.25
for i,(col,lbl,clr) in enumerate([('ent_nm','No Mention','#607D8B'),('ent_pe','Persona','#1976D2'),('ent_th','Third','#388E3C')]):
    ax.bar(x+i*w, m1b_df.set_index('model').reindex(models_b)[col].fillna(0), w, label=lbl, color=clr, alpha=0.85)
ax.set_xticks(x+w); ax.set_xticklabels([m.replace('_',' ') for m in models_b],rotation=45,ha='right',fontsize=7)
ax.set_ylabel('Mean entropy (bits)'); ax.set_title('Logit Entropy by Framing')
ax.legend()

ax2 = axes[1]
for model in models_b:
    r = m1b_df[m1b_df['model']==model]
    if r.empty: continue
    vals = [r['ent_nm'].values[0], r['ent_pe'].values[0], r['ent_th'].values[0]]
    ax2.plot(vals,'-o',alpha=0.5,lw=1.2,ms=5,color='gray')
avg_ent = m1b_df[['ent_nm','ent_pe','ent_th']].mean()
ax2.plot(avg_ent.values,'o-',lw=2.5,color='#D32F2F',ms=8,label='Mean')
ax2.set_xticks([0,1,2]); ax2.set_xticklabels(['No Mention','Persona','Third'])
ax2.set_ylabel('Mean entropy (bits)'); ax2.set_title('Entropy trajectory across framings')
ax2.legend()
fig.suptitle('How does framing change model certainty (logit entropy)?',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95])
save_fig(fig,'m1b_entropy_shift.pdf')

# M1c: Framing shift direction — does Persona/Third push answers higher or lower?
plog('  M1c: Framing shift direction...')
rows_m1c = []
pe_vals = MASTER[MASTER['sheet']=='Personalization'][['model','question_id','country','extracted']].rename(columns={'extracted':'pe'})
th_vals = MASTER[MASTER['sheet']=='Third'][['model','question_id','country','extracted']].rename(columns={'extracted':'th'})
nm_vals = MASTER[MASTER['sheet']=='No Mention'][['model','question_id','extracted']].rename(columns={'extracted':'nm'})

merged_dir = pe_vals.merge(th_vals,on=['model','question_id','country'])
merged_dir = merged_dir.merge(nm_vals,on=['model','question_id'])
merged_dir = merged_dir[merged_dir['nm'].notna() & merged_dir['pe'].notna() & merged_dir['th'].notna()].copy()
merged_dir['pe_minus_nm'] = merged_dir['pe'] - merged_dir['nm']
merged_dir['th_minus_nm'] = merged_dir['th'] - merged_dir['nm']

for model, g in merged_dir.groupby('model'):
    for country, cg in g.groupby('country'):
        fam = COUNTRY_FAM.get(country,'Unknown')
        rows_m1c.append({'model':model,'country':country,'lang_family':fam,
                         'pe_shift_mean':cg['pe_minus_nm'].mean(),
                         'th_shift_mean':cg['th_minus_nm'].mean(),
                         'pe_positive_pct':(cg['pe_minus_nm']>0).mean(),
                         'th_positive_pct':(cg['th_minus_nm']>0).mean(),
                         'n':len(cg)})
m1c_df = pd.DataFrame(rows_m1c)
m1c_df.to_csv(os.path.join(RESULTS,'m1c_framing_shift_direction.csv'),index=False)

# Plot: shift direction heatmap (country × framing averaged over models)
fig, axes = plt.subplots(1,2,figsize=(16,8))
for ax_idx, (col, title) in enumerate([('pe_shift_mean','Persona − No Mention'),('th_shift_mean','Third − No Mention')]):
    ax = axes[ax_idx]
    piv = m1c_df.groupby(['model','country'])[col].mean().unstack('country')
    piv = piv.reindex(columns=COUNTRIES)
    sns.heatmap(piv,ax=ax,cmap='RdBu_r',center=0,annot=True,fmt='.2f',linewidths=0.3,
                cbar_kws={'label':'Mean shift'},annot_kws={'size':6})
    ax.set_title(f'Answer shift: {title}')
    ax.tick_params(axis='x',rotation=45,labelsize=7)
    ax.tick_params(axis='y',labelsize=6)
fig.suptitle('In which direction does framing shift model answers vs No Mention?',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
save_fig(fig,'m1c_framing_shift_direction.pdf')

plog('  Exp M1 done.')

# ─────────────────────────────────────────────────────────────────────────────
# EXP M2: Mechanistic — RSA on cached activations
# ─────────────────────────────────────────────────────────────────────────────
plog('\n=== Exp M2: Representational Similarity Analysis ===')
import glob

act_files = glob.glob(os.path.join(RESULTS,'probe_acts_*.npy'))
plog(f'  Found {len(act_files)} activation files')

rsa_rows = []
for act_file in act_files:
    model_nm = os.path.basename(act_file).replace('probe_acts_','').replace('.npy','')
    tgt_file = act_file.replace('probe_acts_','probe_targets_')
    if not os.path.exists(tgt_file): continue
    acts = np.load(act_file)   # (N, L, D)
    tgts = np.load(tgt_file)   # (N,) — normalized values

    N, L, D = acts.shape
    plog(f'  {model_nm}: acts={acts.shape}, tgts={tgts.shape}')

    # RSA: for each layer, compute correlation between representational distance matrix
    # and value-difference matrix. Higher = layer encodes human-value-like structure.
    # Use subset for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(N, min(500,N), replace=False)
    acts_sub = acts[idx]   # (500, L, D)
    tgts_sub = tgts[idx]  # (500,)

    # Value distance matrix
    val_dist = np.abs(tgts_sub[:,None] - tgts_sub[None,:])  # (500,500)
    val_dist_flat = val_dist[np.triu_indices(len(idx),k=1)]

    layer_rsa = []
    for l in range(L):
        h = acts_sub[:,l,:]  # (500, D)
        # Cosine distance matrix
        norms = np.linalg.norm(h,axis=1,keepdims=True) + 1e-8
        h_norm = h/norms
        cos_sim_mat = h_norm @ h_norm.T
        cos_dist_mat = 1.0 - cos_sim_mat
        rep_dist_flat = cos_dist_mat[np.triu_indices(len(idx),k=1)]
        r, p = spearmanr(rep_dist_flat, val_dist_flat)
        layer_rsa.append(r)

    rsa_rows.append({'model': model_nm, 'layer_rsa': layer_rsa})

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(range(L), layer_rsa, '-o', ms=4, lw=1.5, color='#1976D2')
    peak_layer = int(np.nanargmax(layer_rsa))
    ax.axvline(peak_layer, ls='--', color='red', lw=1, label=f'Peak layer {peak_layer}')
    ax.set_xlabel('Layer'); ax.set_ylabel('Spearman r (rep dist vs value dist)')
    ax.set_title(f'RSA: {model_nm} — which layers encode value structure?')
    ax.legend()
    fig.tight_layout(); save_fig(fig, f'm2_rsa_{model_nm}.pdf')

if rsa_rows:
    # Summary: peak RSA layer and value
    fig, ax = plt.subplots(figsize=(8,5))
    for r in rsa_rows:
        L = len(r['layer_rsa'])
        ax.plot(range(L), r['layer_rsa'], '-o', ms=3, lw=1.5, label=r['model'].replace('_',' '), alpha=0.8)
    ax.set_xlabel('Layer'); ax.set_ylabel('Spearman r (RSA)')
    ax.set_title('RSA across layers: do representations encode human value structure?')
    ax.legend(fontsize=8)
    fig.tight_layout(); save_fig(fig, 'm2_rsa_all_models.pdf')

plog('  Exp M2 done.')

# ─────────────────────────────────────────────────────────────────────────────
# EXP M3: SAE Feature Analysis — framing-sensitive features
# ─────────────────────────────────────────────────────────────────────────────
plog('\n=== Exp M3: SAE Feature Analysis ===')
sae_files = glob.glob(os.path.join(RESULTS,'sae_codes_*.npy'))
plog(f'  Found {len(sae_files)} SAE code files')

for sae_file in sae_files:
    model_nm = os.path.basename(sae_file).replace('sae_codes_','').replace('.npy','')
    tgt_file = sae_file.replace('sae_codes_','probe_targets_')
    act_file = sae_file.replace('sae_codes_','probe_acts_')
    if not os.path.exists(tgt_file): continue
    codes = np.load(sae_file)   # (N, F) — sparse SAE activations
    tgts  = np.load(tgt_file)   # (N,)

    N, F = codes.shape
    plog(f'  {model_nm}: SAE codes {codes.shape}')

    # Sparsity
    sparsity = (codes>0).mean()
    k_avg = (codes>0).sum(axis=1).mean()
    plog(f'    Sparsity: {sparsity:.3f}, avg active features: {k_avg:.1f}')

    # Find features that correlate with value targets (NVAS proxy)
    # Compute point-biserial correlation: for each feature f, binary (active/not) vs tgt value
    from scipy.stats import pearsonr as prs
    feature_corrs = []
    for f in range(F):
        feat_active = (codes[:,f]>0).astype(float)
        if feat_active.mean() < 0.005 or feat_active.mean() > 0.995:
            feature_corrs.append(0.0); continue
        try:
            r, _ = prs(feat_active, tgts)
            feature_corrs.append(r)
        except: feature_corrs.append(0.0)
    feature_corrs = np.array(feature_corrs)
    top_feat_idx = np.argsort(np.abs(feature_corrs))[-20:][::-1]

    plog(f'    Top correlated features: {top_feat_idx[:5]} (r={feature_corrs[top_feat_idx[:5]].round(3)})')

    fig, axes = plt.subplots(1,2,figsize=(14,5))
    ax = axes[0]
    sorted_corrs = np.sort(np.abs(feature_corrs))[::-1]
    ax.plot(range(min(200,F)), sorted_corrs[:200], color='#1976D2', lw=1.5)
    ax.set_xlabel('Feature rank'); ax.set_ylabel('|Pearson r| with value target')
    ax.set_title(f'{model_nm}: SAE features vs value targets')

    ax2 = axes[1]
    ax2.bar(range(20), feature_corrs[top_feat_idx], color=['#D32F2F' if r>0 else '#1976D2' for r in feature_corrs[top_feat_idx]])
    ax2.set_xlabel('Feature index (top 20)'); ax2.set_ylabel('Pearson r')
    ax2.set_xticks(range(20)); ax2.set_xticklabels([str(i) for i in top_feat_idx],rotation=45,fontsize=7)
    ax2.set_title('Top 20 value-correlated SAE features')
    ax2.axhline(0,lw=0.8,color='black')
    fig.suptitle(f'SAE Feature-Value Correlation: {model_nm}',fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95])
    save_fig(fig, f'm3_sae_feature_corr_{model_nm}.pdf')

plog('  Exp M3 done.')

# ─────────────────────────────────────────────────────────────────────────────
# Summary LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────
plog('\n=== Generating summary tables ===')

# Table: Refusal rate by framing (averaged across models, Tier 3)
if len(f1_df)>0:
    t3 = f1_df[f1_df['tier']==3].groupby('sheet')['refusal_rate'].agg(['mean','std']).round(3)
    t3.columns = ['Mean Refusal','Std']
    save_latex(latex_table(t3, 'Tier~3 refusal rates by framing (averaged across all models).','tab:f1_refusal_t3'),'f1_refusal_tier3.tex')

# Table: NVAS by framing
if len(f1b_df)>0:
    nvas_summary = f1b_df.groupby('sheet')['nvas'].agg(['mean','std']).round(3)
    nvas_summary.columns = ['Mean NVAS','Std']
    save_latex(latex_table(nvas_summary,'NVAS by framing (averaged across all models and tiers).','tab:f1b_nvas'),'f1b_nvas_summary.tex')

# Table: native language NVAS gain
if len(f2a_df)>0:
    gain_summary = f2a_df.groupby(['framing','lang_family'])['nvas_gain'].mean().unstack('lang_family').round(3)
    save_latex(latex_table(gain_summary,'Mean NVAS gain (native$-$English) by framing and language family.','tab:f2a_gain'),'f2a_nvas_gain.tex')

# Table: Entropy summary
if len(f3a_df)>0:
    ent_summary = f3a_df.groupby(['tier','sheet'])['entropy'].mean().unstack('sheet').round(3)
    save_latex(latex_table(ent_summary,'Mean logit entropy (bits) by tier and framing.','tab:f3a_entropy'),'f3a_entropy_summary.tex')

# Table: Mechanistic summary (Persona vs Third cosine sim)
if len(m1a_df)>0:
    m1a_tex = m1a_df[['model','cosine_all','cosine_tier3']].set_index('model').round(3)
    m1a_tex.columns = ['Cosine Sim (All)','Cosine Sim (T3)']
    save_latex(latex_table(m1a_tex,'Cosine similarity between Persona and Third-person logit distributions.','tab:m1a_cosine'),'m1a_cosine_summary.tex')

plog('\n=== All framing analyses done ===')
