"""
Gap 7: Qualitative examples table — T3 questions with model responses across framings.
Generates LaTeX table + CSV of the most illustrative T3 examples.
"""
import os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')

BASE = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

mt = pd.read_pickle(os.path.join(RESULTS, 'master_table.pkl'))

def nvas(row):
    if pd.isna(row['extracted']) or pd.isna(row['human_mean']): return np.nan
    d = row['vmax'] - row['vmin']
    return np.nan if d == 0 else 1 - abs(row['extracted'] - row['human_mean']) / d

mt['nvas'] = mt.apply(nvas, axis=1)

# Load question text if available
qpath = os.path.join(BASE, 'questions.csv')
qpath2 = os.path.join(BASE, 'data', 'questions.csv')
qpath3 = os.path.join(BASE, 'results', 'questions.csv')
q_df = None
for p in [qpath, qpath2, qpath3]:
    if os.path.exists(p):
        q_df = pd.read_csv(p)
        print(f'Loaded questions from {p}')
        print('Columns:', q_df.columns.tolist())
        break

if q_df is None:
    # Try to find any CSV with question text
    import glob
    csvs = glob.glob(os.path.join(BASE, '**/*.csv'), recursive=True)
    for c in csvs:
        try:
            tmp = pd.read_csv(c, nrows=3)
            if any('question' in col.lower() or 'text' in col.lower() for col in tmp.columns):
                print(f'Found potential question file: {c}')
                print('Columns:', tmp.columns.tolist())
        except: pass

# Find T3 questions with interesting behavior
# 1. High refusal rate in Persona but answered in Third
# 2. High human WVS value (interesting cultural question)
t3 = mt[mt['tier']==3].copy()
t3['human_mean_round'] = t3['human_mean'].round(1)

# Per question: refusal rate by framing, mean NVAS by framing
q_stats = []
for qid in t3['question_id'].unique():
    sub = t3[t3['question_id']==qid]
    for country in sub['country'].unique():
        sc = sub[sub['country']==country]
        ref_p  = sc[sc['sheet']=='Personalization']['refusal'].mean()
        ref_th = sc[sc['sheet']=='Third']['refusal'].mean()
        nv_p   = sc[(sc['sheet']=='Personalization')&(~sc['refusal'])]['nvas'].mean()
        nv_th  = sc[(sc['sheet']=='Third')&(~sc['refusal'])]['nvas'].mean()
        hm     = sc['human_mean'].mean()
        vm, vx = sc['vmin'].mean(), sc['vmax'].mean()
        q_stats.append({'qid':qid,'country':country,'ref_persona':ref_p,'ref_third':ref_th,
                        'nvas_persona':nv_p,'nvas_third':nv_th,'human_mean':hm,
                        'vmin':vm,'vmax':vx,'delta_ref':ref_th-ref_p,'delta_nvas':nv_th-nv_p})
qs = pd.DataFrame(q_stats)

# Best examples: high ref in persona, low in third, high delta NVAS
qs['score'] = -qs['delta_ref'] + qs['delta_nvas'].fillna(0)  # high = good example
top = qs.sort_values('score', ascending=False).head(50)

# Select diverse countries + questions
examples = []
seen_qids = set()
for _, row in top.iterrows():
    if len(examples) >= 8: break
    if row['qid'] in seen_qids: continue
    seen_qids.add(row['qid'])
    qid, country = row['qid'], row['country']
    # Get actual model responses
    q_sub = t3[(t3['question_id']==qid) & (t3['country']==country)]
    # Find a model that refuses in Persona but answers in Third
    model_ex = None
    for mk in ['olmo_3_7b_instruct','tulu_3_8b_dpo','aya_expanse_8b','allam_7b_instruct','gpt_5']:
        p_sub  = q_sub[(q_sub['model']==mk)&(q_sub['sheet']=='Personalization')]
        th_sub = q_sub[(q_sub['model']==mk)&(q_sub['sheet']=='Third')]
        nm_sub = q_sub[(q_sub['model']==mk)&(q_sub['sheet']=='No Mention')]
        if len(p_sub)>0 and len(th_sub)>0:
            ref_p  = p_sub['refusal'].values[0]
            ext_p  = p_sub['extracted'].values[0] if not p_sub['refusal'].values[0] else None
            ref_th = th_sub['refusal'].values[0]
            ext_th = th_sub['extracted'].values[0] if not th_sub['refusal'].values[0] else None
            ext_nm = nm_sub['extracted'].values[0] if len(nm_sub)>0 and not nm_sub['refusal'].values[0] else None
            if ref_p and not ref_th:
                model_ex = {'model':mk,'ref_p':ref_p,'ext_p':ext_p,'ref_th':ref_th,'ext_th':ext_th,'ext_nm':ext_nm}
                break
    if model_ex is None:
        # Just take any response pair
        for mk in q_sub['model'].unique():
            p_sub  = q_sub[(q_sub['model']==mk)&(q_sub['sheet']=='Personalization')]
            th_sub = q_sub[(q_sub['model']==mk)&(q_sub['sheet']=='Third')]
            if len(p_sub)>0 and len(th_sub)>0:
                model_ex = {'model':mk,
                            'ref_p':p_sub['refusal'].values[0],
                            'ext_p':p_sub['extracted'].values[0] if not p_sub['refusal'].values[0] else None,
                            'ref_th':th_sub['refusal'].values[0],
                            'ext_th':th_sub['extracted'].values[0] if not th_sub['refusal'].values[0] else None,
                            'ext_nm':None}
                break
    examples.append({
        'qid': qid, 'country': country,
        'human_mean': round(row['human_mean'],1),
        'scale': f'[{int(row["vmin"])}–{int(row["vmax"])}]',
        'ref_persona': round(row['ref_persona'],2) if not np.isnan(row['ref_persona']) else None,
        'ref_third':   round(row['ref_third'],2)   if not np.isnan(row['ref_third'])   else None,
        'nvas_third':  round(row['nvas_third'],3)  if not np.isnan(row['nvas_third'])  else None,
        'model_example': model_ex,
    })

ex_df = pd.DataFrame(examples)
print('\nQualitative examples found:')
print(ex_df[['qid','country','human_mean','scale','ref_persona','ref_third','nvas_third']].to_string())

# Load question texts from tiers.csv
tiers_csv_path = os.path.join(BASE, 'tiers.csv')
question_texts = {}
try:
    tiers_df = pd.read_csv(tiers_csv_path)
    for _, r in tiers_df.iterrows():
        question_texts[int(r['question_id'])] = str(r['No Mention'])
    print(f'Loaded {len(question_texts)} question texts from tiers.csv')
except Exception as e:
    print(f'Could not load tiers.csv: {e}')

def latex_escape(s):
    return (str(s).replace('&','\\&').replace('%','\\%').replace('$','\\$')
                  .replace('#','\\#').replace('_','\\_').replace('{','\\{')
                  .replace('}','\\}').replace('~','\\textasciitilde{}')
                  .replace('^','\\textasciicircum{}').replace('\\\\','\\'))

def short_snippet(qid, max_len=55):
    text = question_texts.get(int(qid) if str(qid).isdigit() else qid, '')
    if not text or text == 'nan':
        return f'Q{qid}'
    # Strip leading "Please indicate ... " boilerplate
    for prefix in ["Please indicate ", "Please tell ", "Using the scale", "How ", "Do you ", "Would you "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    # Find a natural cut
    if len(text) > max_len:
        cut = text[:max_len].rfind(' ')
        text = text[:cut if cut > 20 else max_len] + '…'
    return latex_escape(text)

# Generate LaTeX table
latex_rows = []
for ex in examples[:6]:
    qid_str = ex['qid']
    snippet = short_snippet(qid_str)
    hm = ex['human_mean']
    sc = ex['scale']
    country = ex['country']
    me = ex['model_example']
    if me is None: continue
    model_short = me['model'].replace('olmo_3_7b_','OLMo-').replace('instruct','IT').replace(
        'tulu_3_8b_','Tulu-').replace('aya_expanse_','AYA-').replace('_instruct','').replace(
        'allam_7b_','ALLAM-').replace('gpt_5','GPT-5')
    persona_str = 'REF' if me['ref_p'] else (f'{me["ext_p"]:.0f}' if me['ext_p'] is not None else '—')
    third_str   = 'REF' if me['ref_th'] else (f'{me["ext_th"]:.0f}' if me['ext_th'] is not None else '—')
    latex_rows.append(
        f'{snippet} & {country} & {hm} {sc} & {model_short} & '
        f'{persona_str} & {third_str} \\\\'
    )

latex_table = r"""\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Qualitative T3 examples: model response under Persona vs.\ Third-person framing.
``REF'' = refusal. Human mean is the WVS survey mean for that country and question.
Third-person framing elicits substantive answers from models that refuse under Persona framing,
and the answers are closer to the human WVS mean.}
\label{tab:qualitative}
\begin{tabular}{p{3.8cm}p{1.5cm}p{1.6cm}p{1.5cm}cc}
\toprule
Question snippet & Country & Human mean (scale) & Model & Persona & Third \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""

with open(os.path.join(RESULTS, 'gap7_qualitative_examples.tex'), 'w') as f:
    f.write(latex_table)
print(f'\nSaved: gap7_qualitative_examples.tex')

ex_df.to_csv(os.path.join(RESULTS, 'gap7_qualitative_examples.csv'), index=False)
print('Saved: gap7_qualitative_examples.csv')
print('Done.')
