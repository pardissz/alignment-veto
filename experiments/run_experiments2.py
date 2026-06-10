"""
MENA LLM Experiments — loads cached master table and runs all analysis.
"""
import os, ast, json, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    'Persian': ['Iran'], 'Turkish': ['Turkey'],
}
OLMO_7B  = ['olmo_3_7b_base','olmo_3_7b_sft','olmo_3_7b_dpo','olmo_3_7b_instruct']
OLMO_32B = ['olmo_3_32b_base','olmo_3_32b_sft','olmo_3_32b_dpo','olmo_3_32b_instruct']
TULU     = ['tulu_3_8b_sft','tulu_3_8b_dpo','tulu_3.1_8b']
STAGE_LBL = {
    'olmo_3_7b_base':'7B-Base','olmo_3_7b_sft':'7B-SFT',
    'olmo_3_7b_dpo':'7B-DPO','olmo_3_7b_instruct':'7B-Inst',
    'olmo_3_32b_base':'32B-Base','olmo_3_32b_sft':'32B-SFT',
    'olmo_3_32b_dpo':'32B-DPO','olmo_3_32b_instruct':'32B-Inst',
    'tulu_3_8b_sft':'T-SFT','tulu_3_8b_dpo':'T-DPO','tulu_3.1_8b':'Tulu-3.1',
}
FAM_COL = {'Arabic':'#E53935','Persian':'#1E88E5','Turkish':'#43A047'}

plt.rcParams.update({'font.family':'serif','font.size':10,'axes.titlesize':11,
                     'figure.dpi':150,'savefig.dpi':200,'figure.facecolor':'white'})

def bootstrap_ci(data, n=1000, alpha=0.05, seed=42):
    a = np.asarray(data, dtype=float); a = a[~np.isnan(a)]
    if len(a)<2: return np.nan,np.nan,np.nan
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(a,len(a),replace=True)) for _ in range(n)]
    return np.mean(a), np.percentile(boots,100*alpha/2), np.percentile(boots,100*(1-alpha/2))

def save_fig(fig, fname):
    fig.savefig(os.path.join(RESULTS,fname),bbox_inches='tight'); plt.close(fig)
    plog(f'  Saved: {fname}')

def save_latex(text, fname):
    with open(os.path.join(RESULTS,fname),'w') as f: f.write(text)
    plog(f'  Saved: {fname}')

def latex_tbl(df, caption, label, cf=None):
    if cf is None: cf = 'l'+'r'*len(df.columns)
    return '\n'.join([
        r'\begin{table}[htbp]',r'\centering',r'\small',
        f'\\caption{{{caption}}}',f'\\label{{{label}}}',
        df.to_latex(index=True,escape=True,float_format='%.3f',
                    column_format=cf,na_rep='—'),
        r'\end{table}',])

# ── Load master table ──────────────────────────────────────────────────────────
plog('Loading master table...')
MASTER = pd.read_pickle(os.path.join(RESULTS,'master_table.pkl'))
plog(f'  {len(MASTER):,} rows, {MASTER.model.nunique()} models')
plog(f'  Sheets: {sorted(MASTER.sheet.unique())}')
plog(f'  Tier counts: {MASTER.tier.value_counts().sort_index().to_dict()}')

# ── Load human data for PCA ────────────────────────────────────────────────────
plog('Loading human data for PCA...')
def load_human():
    df = pd.read_excel(os.path.join(BASE,'new_weights_transposed.xlsx'))
    df = df.rename(columns={'question_number':'question_id'})
    for c in COUNTRIES:
        if c not in df.columns: continue
        means, dists = [], []
        for v in df[c]:
            if pd.isna(v): means.append(np.nan); dists.append({}); continue
            try:
                tup = ast.literal_eval(str(v))
                means.append(float(tup[0]))
                dists.append({k:float(str(vv).strip('%'))/100 for k,vv in tup[1].items()})
            except: means.append(np.nan); dists.append({})
        df[f'{c}_mean'] = means; df[f'{c}_dist'] = dists
    return df.set_index('question_id')
HUMAN = load_human()

# ── EXP 2a: Safety Tax ──────────────────────────────────────────────────────────
plog('\n=== Exp 2a: Safety Tax ===')

rows=[]
for (model,tier),g in MASTER.groupby(['model','tier']):
    obs,lo,hi = bootstrap_ci(g['refusal'].astype(float).values)
    rows.append({'model':model,'tier':int(tier),'refusal_rate':obs,'ci_lo':lo,'ci_hi':hi,'n':len(g)})
refusal_df = pd.DataFrame(rows)

piv = refusal_df.pivot(index='model',columns='tier',values='refusal_rate')
piv.columns=[int(c) for c in piv.columns]
if 3 in piv.columns and 1 in piv.columns:
    piv['T3_m_T1']=piv[3]-piv[1]
piv=piv.sort_values('T3_m_T1',ascending=False)

models_ord = piv.index.tolist(); x=np.arange(len(models_ord)); w=0.25
cols={1:'#4CAF50',2:'#FF9800',3:'#F44336'}

fig,axes=plt.subplots(1,2,figsize=(16,7))
ax=axes[0]
for i,tier in enumerate([1,2,3]):
    t=refusal_df[refusal_df['tier']==tier].set_index('model').reindex(models_ord)
    elo=(t['refusal_rate']-t['ci_lo']).fillna(0).values
    ehi=(t['ci_hi']-t['refusal_rate']).fillna(0).values
    ax.bar(x+i*w, t['refusal_rate'].fillna(0).values, w,
           label=f'Tier {tier}', color=cols[tier], alpha=0.85,
           yerr=[elo,ehi], capsize=2, error_kw={'elinewidth':0.8})
ax.set_xticks(x+w); ax.set_xticklabels([m.replace('_',' ') for m in models_ord],rotation=45,ha='right',fontsize=7)
ax.set_ylabel('Refusal Rate'); ax.set_title('Refusal Rate by Tier & Model'); ax.legend()
ax2=axes[1]
gap=piv.get('T3_m_T1',pd.Series(dtype=float)).fillna(0)
ax2.barh(range(len(gap)),gap.values,
         color=['#D32F2F' if v>0 else '#388E3C' for v in gap],alpha=0.85)
ax2.set_yticks(range(len(gap))); ax2.set_yticklabels([m.replace('_',' ') for m in gap.index],fontsize=7)
ax2.axvline(0,color='black',lw=0.8)
ax2.set_xlabel('Tier 3 − Tier 1 Refusal Gap'); ax2.set_title('Safety Tax (T3−T1)')
fig.tight_layout(); save_fig(fig,'exp2a_safety_tax.pdf')

piv_tex=piv.copy(); piv_tex.columns=[f'T{c}' if isinstance(c,int) else c for c in piv_tex.columns]
save_latex(latex_tbl(piv_tex.round(3),
    'Refusal rates by model and tier. T3\_m\_T1 = Tier~3$-$Tier~1 safety-tax gap.',
    'tab:exp2a'),'exp2a_safety_tax.tex')
refusal_df.to_csv(os.path.join(RESULTS,'exp2a_refusal.csv'),index=False)
plog('  Exp 2a done.')

# ── EXP 2b: NVAS by Tier ──────────────────────────────────────────────────────
plog('\n=== Exp 2b: NVAS by Tier ===')

sub2b = MASTER[
    MASTER['sheet'].isin(['Personalization','Third']) &
    ~MASTER['refusal'] & MASTER['human_mean'].notna() &
    MASTER['extracted'].notna() & MASTER['vmin'].notna() & MASTER['vmax'].notna()
].copy()
rng_v = (sub2b['vmax']-sub2b['vmin']).replace(0,np.nan)
sub2b['nvas'] = 1.0 - (sub2b['extracted']-sub2b['human_mean']).abs()/rng_v

rows2b=[]
for (model,tier),g in sub2b.groupby(['model','tier']):
    obs,lo,hi=bootstrap_ci(g['nvas'].dropna().values)
    rows2b.append({'model':model,'tier':int(tier),'nvas':obs,'ci_lo':lo,'ci_hi':hi,'n':len(g)})
nvas_df=pd.DataFrame(rows2b)

nv_piv=nvas_df.pivot(index='model',columns='tier',values='nvas')
nv_piv.columns=[int(c) for c in nv_piv.columns]
if 3 in nv_piv.columns and 1 in nv_piv.columns:
    nv_piv['T3_m_T1']=nv_piv[3]-nv_piv[1]
nv_piv=nv_piv.sort_values('T3_m_T1',ascending=True)

fig,axes=plt.subplots(1,2,figsize=(16,7))
models_nv=nv_piv.index.tolist(); x2=np.arange(len(models_nv))
ax=axes[0]
for i,tier in enumerate([1,2,3]):
    t=nvas_df[nvas_df['tier']==tier].set_index('model').reindex(models_nv)
    elo=(t['nvas']-t['ci_lo']).fillna(0).values; ehi=(t['ci_hi']-t['nvas']).fillna(0).values
    ax.bar(x2+i*w,t['nvas'].fillna(0).values,w,label=f'Tier {tier}',
           color=cols[tier],alpha=0.85,yerr=[elo,ehi],capsize=2,error_kw={'elinewidth':0.8})
ax.set_xticks(x2+w); ax.set_xticklabels([m.replace('_',' ') for m in models_nv],rotation=45,ha='right',fontsize=7)
ax.set_ylabel('NVAS'); ax.set_ylim(0,1.05); ax.set_title('NVAS by Tier & Model'); ax.legend()
ax2=axes[1]
gap2=nv_piv.get('T3_m_T1',pd.Series(dtype=float)).fillna(0)
ax2.barh(range(len(gap2)),gap2.values,
         color=['#D32F2F' if v<0 else '#388E3C' for v in gap2],alpha=0.85)
ax2.set_yticks(range(len(gap2))); ax2.set_yticklabels([m.replace('_',' ') for m in gap2.index],fontsize=7)
ax2.axvline(0,color='black',lw=0.8)
ax2.set_xlabel('NVAS(T3)−NVAS(T1)'); ax2.set_title('NVAS Accuracy Gap T3−T1')
fig.tight_layout(); save_fig(fig,'exp2b_nvas_by_tier.pdf')

# OLMo Tier-3
olmo_t3=nvas_df[(nvas_df['tier']==3)&nvas_df['model'].str.contains('olmo')].copy()
if len(olmo_t3):
    fig2,ax3=plt.subplots(figsize=(8,5))
    y=np.arange(len(olmo_t3))
    bcols=['#1565C0' if 'base' in m else '#E53935' for m in olmo_t3['model']]
    ax3.barh(y,olmo_t3['nvas'].values,
             xerr=[(olmo_t3['nvas']-olmo_t3['ci_lo']).fillna(0).values,
                   (olmo_t3['ci_hi']-olmo_t3['nvas']).fillna(0).values],
             capsize=3,color=bcols,alpha=0.85)
    ax3.set_yticks(y); ax3.set_yticklabels([m.replace('_',' ') for m in olmo_t3['model']],fontsize=8)
    ax3.set_xlabel('NVAS on Tier 3'); ax3.set_title('OLMo variants Tier 3 NVAS (blue=base, red=aligned)')
    fig2.tight_layout(); save_fig(fig2,'exp2b_olmo_tier3.pdf')

nv_tex=nv_piv.copy(); nv_tex.columns=[f'T{c}' if isinstance(c,int) else c for c in nv_tex.columns]
save_latex(latex_tbl(nv_tex.round(3),'NVAS by model and tier (non-refused answers).','tab:exp2b'),
           'exp2b_nvas.tex')
nvas_df.to_csv(os.path.join(RESULTS,'exp2b_nvas.csv'),index=False)
plog('  Exp 2b done.')

# ── EXP 3: Direction of Suppression ──────────────────────────────────────────
plog('\n=== Exp 3: Direction of Suppression ===')

def majority_opt(d):
    if not d: return None
    try: return max(d, key=lambda k: float(d[k]))
    except: return None

def pool_mass(d, toks=None):
    if not d: return np.nan
    toks = toks or {str(i) for i in range(1,10)}
    try:
        tot=sum(float(v) for v in d.values());
        if tot==0: return np.nan
        return sum(float(v) for k,v in d.items() if str(k).strip() in toks)/tot
    except: return np.nan

refused3 = MASTER[
    MASTER['sheet'].isin(['Personalization','Third']) &
    MASTER['refusal'] & MASTER['human_mean'].notna()
].copy()

plog(f'  Refused with human ground truth: {len(refused3)}')

if len(refused3):
    refused3['digit_mass'] = refused3['norm_probs'].apply(pool_mass)
    refused3['model_maj']  = refused3['norm_probs'].apply(majority_opt)
    refused3['human_maj']  = refused3['human_dist'].apply(majority_opt)
    refused3['agrees']     = (
        refused3['model_maj'].notna() & refused3['human_maj'].notna() &
        (refused3['model_maj'].astype(str)==refused3['human_maj'].astype(str))
    ).astype(float)

    rows3=[]
    for (model,tier),g in refused3.groupby(['model','tier']):
        m_obs,m_lo,m_hi=bootstrap_ci(g['digit_mass'].dropna().values)
        for thresh in [0.50,0.75,0.90]:
            above=g[g['digit_mass']>=thresh]
            if len(above)<2:
                rows3.append({'model':model,'tier':int(tier),'threshold':thresh,
                              'n_above':len(above),'agreement_rate':np.nan,
                              'ci_lo':np.nan,'ci_hi':np.nan,'pool_mass':m_obs})
            else:
                obs,lo,hi=bootstrap_ci(above['agrees'].values)
                rows3.append({'model':model,'tier':int(tier),'threshold':thresh,
                              'n_above':len(above),'agreement_rate':obs,
                              'ci_lo':lo,'ci_hi':hi,'pool_mass':m_obs})
    supp_df=pd.DataFrame(rows3)
    supp_df.to_csv(os.path.join(RESULTS,'exp3_suppression.csv'),index=False)

    fig,axes=plt.subplots(1,3,figsize=(18,6))
    ax=axes[0]
    for thresh,lw,ls in zip([0.50,0.75,0.90],[2.5,2,1.5],['-','--',':']):
        avg=supp_df[supp_df['threshold']==thresh].groupby('tier')['agreement_rate'].mean()
        if len(avg): ax.plot(avg.index,avg.values,marker='o',lw=lw,ls=ls,label=f'{thresh:.0%}')
    ax.set_xticks([2,3]); ax.set_xticklabels(['Tier 2','Tier 3'])
    ax.set_ylabel('WVS agreement rate'); ax.set_ylim(0,1)
    ax.set_title('WVS Agreement in Refused Qs by Tier'); ax.legend(title='Threshold')

    ax2=axes[1]
    t75=supp_df[supp_df['threshold']==0.75].pivot(index='model',columns='tier',values='agreement_rate')
    if 3 in t75.columns and 2 in t75.columns:
        g75=(t75[3]-t75[2]).dropna().sort_values(ascending=False)
        ax2.barh(range(len(g75)),g75.values,
                 color=['#D32F2F' if v>0 else '#388E3C' for v in g75],alpha=0.85)
        ax2.set_yticks(range(len(g75))); ax2.set_yticklabels([m.replace('_',' ') for m in g75.index],fontsize=7)
        ax2.axvline(0,color='black',lw=0.8)
        ax2.set_xlabel('Agreement T3−T2 (thresh=75%)'); ax2.set_title('Per-model WVS Gap T3−T2')

    ax3=axes[2]
    pa=supp_df.groupby(['model','tier'])['pool_mass'].mean().reset_index()
    for tier,col in [(2,'#FF9800'),(3,'#F44336')]:
        sub=pa[pa['tier']==tier]
        ax3.scatter(range(len(sub)),sub['pool_mass'].values,label=f'Tier {tier}',color=col,s=30,alpha=0.7)
    ax3.set_ylabel('Digit-token mass fraction'); ax3.set_title('Pool Mass on Digits (Refusals)'); ax3.legend()
    fig.tight_layout(); save_fig(fig,'exp3_suppression.pdf')

    if supp_df['agreement_rate'].notna().any():
        smry=supp_df.groupby(['tier','threshold'])['agreement_rate'].mean().reset_index()
        pt=smry.pivot(index='threshold',columns='tier',values='agreement_rate')
        pt.columns=[f'Tier {c}' for c in pt.columns]
        save_latex(latex_tbl(pt.round(3),
            r'WVS majority agreement (refused qs) by tier and threshold.',
            'tab:exp3'),'exp3_suppression.tex')
else:
    supp_df=pd.DataFrame()
    plog('  No refusals found — skipping Exp 3 plots.')

plog('  Exp 3 done.')

# ── Consistency Metrics ────────────────────────────────────────────────────────
plog('\n=== Consistency Metrics (FCS, CLCS, SPD) ===')

def get_vals(master, model, sheet, country, neutral=False):
    c = 'NEUTRAL' if neutral else country
    q = (master['model']==model) & (master['sheet']==sheet) & (master['country']==c) & \
        ~master['refusal'] & master['extracted'].notna()
    sub = master[q][['question_id','extracted','vmin','vmax']].dropna()
    return sub.set_index('question_id')

cons_rows=[]
for model in sorted(MASTER['model'].unique()):
    neutral_en = get_vals(MASTER,model,'No Mention','',neutral=True)
    for country in COUNTRIES:
        p_en = get_vals(MASTER,model,'Personalization',country)
        p_nat= get_vals(MASTER,model,'Personalization_Diff',country)
        o_en = get_vals(MASTER,model,'Third',country)

        def dist_score(df1, df2, cn1='extracted'):
            if len(df1)==0 or len(df2)==0: return []
            common=df1.index.intersection(df2.index)
            out=[]
            for qid in common:
                v1=float(df1.loc[qid,cn1]) if cn1 in df1.columns else float(df1.loc[qid,'extracted'])
                v2=float(df2.loc[qid,'extracted'])
                vm=float(df1.loc[qid,'vmin']); vx=float(df1.loc[qid,'vmax'])
                if not (pd.notna(v1) and pd.notna(v2)): continue
                rng=vx-vm; rng=rng if rng>0 else 1.0
                out.append(1-abs(v1-v2)/rng)
            return out

        fcs_vals  = dist_score(p_en, o_en)
        clcs_vals = dist_score(p_en, p_nat)
        spd_vals  = dist_score(neutral_en, p_en)

        fcs  = bootstrap_ci(fcs_vals)  if len(fcs_vals)>=2  else (np.nan,)*3
        clcs = bootstrap_ci(clcs_vals) if len(clcs_vals)>=2 else (np.nan,)*3
        spd  = bootstrap_ci(spd_vals)  if len(spd_vals)>=2  else (np.nan,)*3
        cons_rows.append({
            'model':model,'country':country,
            'FCS':fcs[0],'FCS_lo':fcs[1],'FCS_hi':fcs[2],
            'CLCS':clcs[0],'CLCS_lo':clcs[1],'CLCS_hi':clcs[2],
            'SPD':spd[0],'SPD_lo':spd[1],'SPD_hi':spd[2],
        })

cons_df=pd.DataFrame(cons_rows)
cons_df.to_csv(os.path.join(RESULTS,'consistency_metrics.csv'),index=False)

cons_sum=cons_df.groupby('model')[['FCS','CLCS','SPD']].mean().round(3).sort_values('FCS')
save_latex(latex_tbl(cons_sum,
    'Consistency metrics averaged over countries. '
    'FCS=Framing Consistency; CLCS=Cross-Lingual Consistency; SPD=Self-Persona Deviation.',
    'tab:consistency'),'consistency_metrics.tex')

fig,axes=plt.subplots(1,3,figsize=(22,10))
for ax,metric,title in zip(axes,['FCS','CLCS','SPD'],
                            ['Framing Consistency (FCS)','Cross-Lingual Consistency (CLCS)',
                             'Self-Persona Deviation (SPD)']):
    piv=cons_df.pivot(index='model',columns='country',values=metric)
    piv=piv.reindex(sorted(piv.index))
    sns.heatmap(piv,ax=ax,cmap='RdYlGn',vmin=0,vmax=1,cbar_kws={'shrink':0.6})
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right',fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=7)
fig.tight_layout(); save_fig(fig,'consistency_heatmaps.pdf')
plog('  Consistency metrics done.')

# ── PCA 7.1: Human Survey ────────────────────────────────────────────────────
plog('\n=== PCA 7.1: Human Survey ===')
mat71=[]; vc71=[]
for c in COUNTRIES:
    row=HUMAN[f'{c}_mean'].values.astype(float)
    if not np.all(np.isnan(row)): mat71.append(row); vc71.append(c)
mat71=np.array(mat71); cm71=np.nanmean(mat71,axis=0)
ix71=np.where(np.isnan(mat71)); mat71[ix71]=np.take(cm71,ix71[1])
pca71=PCA(2).fit(StandardScaler().fit_transform(mat71))
coord71=pca71.transform(StandardScaler().fit_transform(mat71))
ve71=pca71.explained_variance_ratio_

fig,axes=plt.subplots(1,2,figsize=(14,6))
ax=axes[0]
for i,c in enumerate(vc71):
    fam=[f for f,cs in LANG_FAMILY.items() if c in cs]; fam=fam[0] if fam else 'Other'
    ax.scatter(coord71[i,0],coord71[i,1],color=FAM_COL.get(fam,'gray'),s=80,zorder=5)
    ax.annotate(c,(coord71[i,0],coord71[i,1]),fontsize=7,xytext=(4,4),textcoords='offset points')
handles=[mpatches.Patch(color=v,label=k) for k,v in FAM_COL.items()]
ax.legend(handles=handles)
ax.set_xlabel(f'PC1 ({ve71[0]:.1%})'); ax.set_ylabel(f'PC2 ({ve71[1]:.1%})')
ax.set_title('PCA 7.1: Human Survey Country Vectors')

ax2=axes[1]
n71=len(vc71); jmat=np.zeros((n71,n71))
for i in range(n71):
    for j in range(n71):
        if i==j: continue
        ci,cj=vc71[i],vc71[j]
        jsds=[]
        for qid in HUMAN.index:
            di=HUMAN.loc[qid,f'{ci}_dist']; dj=HUMAN.loc[qid,f'{cj}_dist']
            if di and dj:
                ks=sorted(set(di)|set(dj))
                p=np.array([di.get(k,1e-9) for k in ks],dtype=float); p/=p.sum()
                q=np.array([dj.get(k,1e-9) for k in ks],dtype=float); q/=q.sum()
                m=(p+q)/2
                def kl(a,b): return np.sum(a*np.log(a/(b+1e-12)+1e-12))
                jsds.append(0.5*kl(p,m)+0.5*kl(q,m))
        jmat[i,j]=np.nanmean(jsds) if jsds else np.nan
sns.heatmap(1-jmat,ax=ax2,xticklabels=vc71,yticklabels=vc71,cmap='Blues',vmin=0,vmax=1)
ax2.set_title('JSD Similarity Matrix (Human)')
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45,ha='right',fontsize=7)
ax2.set_yticklabels(ax2.get_yticklabels(),fontsize=7)
fig.tight_layout(); save_fig(fig,'pca_7_1_human.pdf')
plog(f'  PCA 7.1: PC1={ve71[0]:.1%}, PC2={ve71[1]:.1%}')

# ── PCA helpers ───────────────────────────────────────────────────────────────
def country_matrix(model, sheet, min_q=30):
    qids=sorted(MASTER['question_id'].unique()); mat=[]; vc=[]
    for country in COUNTRIES:
        sub=MASTER[(MASTER['model']==model)&(MASTER['sheet']==sheet)&
                   (MASTER['country']==country)&~MASTER['refusal']&MASTER['extracted'].notna()
                   ].set_index('question_id')
        if len(sub)<min_q: continue
        vec=[float(sub.loc[q,'extracted']) if q in sub.index else np.nan for q in qids]
        mat.append(vec); vc.append(country)
    if len(mat)<3: return None,None
    mat=np.array(mat,dtype=float); cm=np.nanmean(mat,axis=0)
    ix=np.where(np.isnan(mat)); mat[ix]=np.take(cm,ix[1])
    try: return PCA(2).fit_transform(StandardScaler().fit_transform(mat)),vc
    except: return None,None

# ── PCA 7.2: LLM Observer English ────────────────────────────────────────────
plog('=== PCA 7.2: Observer English ===')
models72=sorted(MASTER['model'].unique())
nc=7; nr=(len(models72)+nc-1)//nc
fig72,axes72=plt.subplots(nr,nc,figsize=(28,4*nr))
axes72=axes72.flatten()
for idx,model in enumerate(models72):
    if idx>=len(axes72): break
    ax=axes72[idx]
    coords,vc=country_matrix(model,'Third')
    if coords is None: ax.axis('off'); ax.set_title(model.replace('_',' '),fontsize=7); continue
    for i,c in enumerate(vc):
        fam=[f for f,cs in LANG_FAMILY.items() if c in cs]; fam=fam[0] if fam else 'Other'
        ax.scatter(coords[i,0],coords[i,1],color=FAM_COL.get(fam,'gray'),s=25)
        ax.annotate(c[:3],(coords[i,0],coords[i,1]),fontsize=5)
    ax.set_title(model.replace('_',' '),fontsize=7); ax.tick_params(labelsize=5)
for ax in axes72[idx+1:]: ax.axis('off')
fig72.suptitle('PCA 7.2: LLM Observer (English) Country Vectors',fontsize=12)
fig72.tight_layout(); save_fig(fig72,'pca_7_2_observer_english.pdf')

# ── PCA 7.3: Native Language ─────────────────────────────────────────────────
plog('=== PCA 7.3: Native Language ===')
focus73=['olmo_3_7b_instruct','llama_3.1_8b_instruct','qwen2.5_7b_instruct',
         'gemma_3_12b_it','aya_expanse_8b','mistral_7b_instruct']
focus73=[m for m in focus73 if m in MASTER['model'].unique()]
if focus73:
    fig73,axes73=plt.subplots(1,len(focus73),figsize=(5*len(focus73),5))
    if len(focus73)==1: axes73=[axes73]
    for ax,model in zip(axes73,focus73):
        coords,vc=country_matrix(model,'Third_Diff')
        if coords is None: ax.axis('off'); ax.set_title(model,fontsize=8); continue
        for i,c in enumerate(vc):
            fam=[f for f,cs in LANG_FAMILY.items() if c in cs]; fam=fam[0] if fam else 'Other'
            ax.scatter(coords[i,0],coords[i,1],color=FAM_COL.get(fam,'gray'),s=60)
            ax.annotate(c[:3],(coords[i,0],coords[i,1]),fontsize=6)
        ax.legend(handles=[mpatches.Patch(color=v,label=k) for k,v in FAM_COL.items()],fontsize=7)
        ax.set_title(model.replace('_',' '),fontsize=8)
    fig73.suptitle('PCA 7.3: Native Language Observer — Language-Family Collapse',fontsize=11)
    fig73.tight_layout(); save_fig(fig73,'pca_7_3_native.pdf')

# ── PCA 7.4: Persona + Neutral ───────────────────────────────────────────────
plog('=== PCA 7.4: Persona + Neutral ===')
focus74=['olmo_3_7b_instruct','llama_3.1_8b_instruct','qwen2.5_7b_instruct','gemma_3_12b_it']
focus74=[m for m in focus74 if m in MASTER['model'].unique()]
qids74=sorted(MASTER['question_id'].unique())
if focus74:
    fig74,axes74=plt.subplots(1,len(focus74),figsize=(5*len(focus74),5))
    if len(focus74)==1: axes74=[axes74]
    for ax,model in zip(axes74,focus74):
        mat=[]; labels=[]; colors=[]
        for country in COUNTRIES:
            sub=MASTER[(MASTER['model']==model)&(MASTER['sheet']=='Personalization')&
                       (MASTER['country']==country)&~MASTER['refusal']&MASTER['extracted'].notna()].set_index('question_id')
            if len(sub)<30: continue
            vec=[float(sub.loc[q,'extracted']) if q in sub.index else np.nan for q in qids74]
            mat.append(vec)
            fam=[f for f,cs in LANG_FAMILY.items() if country in cs]; fam=fam[0] if fam else 'Other'
            colors.append(FAM_COL.get(fam,'gray')); labels.append(country[:3])
        neut=MASTER[(MASTER['model']==model)&(MASTER['sheet']=='No Mention')&
                    (MASTER['country']=='NEUTRAL')&~MASTER['refusal']&MASTER['extracted'].notna()].set_index('question_id')
        if len(neut)>=30:
            vec=[float(neut.loc[q,'extracted']) if q in neut.index else np.nan for q in qids74]
            mat.append(vec); colors.append('black'); labels.append('LLM★')
        if len(mat)<3: ax.axis('off'); continue
        mat=np.array(mat,dtype=float); cm=np.nanmean(mat,axis=0)
        ix=np.where(np.isnan(mat)); mat[ix]=np.take(cm,ix[1])
        try: coords=PCA(2).fit_transform(StandardScaler().fit_transform(mat))
        except: ax.axis('off'); continue
        for i,(lbl,col) in enumerate(zip(labels,colors)):
            ms=150 if lbl=='LLM★' else 40; mk='*' if lbl=='LLM★' else 'o'
            ax.scatter(coords[i,0],coords[i,1],color=col,s=ms,marker=mk,zorder=5)
            ax.annotate(lbl,(coords[i,0],coords[i,1]),fontsize=5,xytext=(2,2),textcoords='offset points')
        ax.set_title(model.replace('_',' '),fontsize=8)
    fig74.suptitle('PCA 7.4: Persona + Neutral ★ — Cultural Identity Crisis',fontsize=11)
    fig74.tight_layout(); save_fig(fig74,'pca_7_4_persona_neutral.pdf')

# ── PCA 7.5: Cross-Lingual ───────────────────────────────────────────────────
plog('=== PCA 7.5: Cross-Lingual Value Shift ===')
focus75=['olmo_3_7b_instruct','llama_3.1_8b_instruct','qwen2.5_7b_instruct',
         'gemma_3_12b_it','aya_expanse_8b']
focus75=[m for m in focus75 if m in MASTER['model'].unique()]
if focus75:
    fig75,axes75=plt.subplots(1,len(focus75),figsize=(5*len(focus75),5))
    if len(focus75)==1: axes75=[axes75]
    qids75=sorted(MASTER['question_id'].unique())
    for ax,model in zip(axes75,focus75):
        en_sub=MASTER[(MASTER['model']==model)&(MASTER['sheet']=='No Mention')&
                      (MASTER['country']=='NEUTRAL')&~MASTER['refusal']&MASTER['extracted'].notna()].set_index('question_id')
        nat_sub=MASTER[(MASTER['model']==model)&(MASTER['sheet']=='No Mention Diff')&
                       (MASTER['country']=='NEUTRAL')&~MASTER['refusal']&MASTER['extracted'].notna()].set_index('question_id')
        if len(nat_sub)==0:  # Try Personalization_Diff as proxy
            nat_sub=MASTER[(MASTER['model']==model)&(MASTER['sheet']=='Personalization_Diff')&
                           ~MASTER['refusal']&MASTER['extracted'].notna()].groupby('question_id')['extracted'].mean()
            nat_sub=pd.DataFrame({'extracted':nat_sub})
        common=en_sub.index.intersection(nat_sub.index)
        if len(common)>10:
            en_vals=np.array([float(en_sub.loc[q,'extracted']) for q in common],dtype=float)
            nat_vals=np.array([float(nat_sub.loc[q,'extracted']) if 'extracted' in nat_sub.columns else np.nan for q in common],dtype=float)
            diff=np.abs(en_vals-nat_vals)
            valid=~np.isnan(diff)
            if valid.sum()>5:
                ax.hist(diff[valid],bins=20,color='steelblue',alpha=0.8,edgecolor='white')
                ax.axvline(np.nanmean(diff[valid]),color='red',ls='--',
                           label=f'Mean diff={np.nanmean(diff[valid]):.2f}')
                ax.set_xlabel('|English − Native| answer')
                ax.set_ylabel('Count'); ax.legend(fontsize=7)
        ax.set_title(model.replace('_',' '),fontsize=8)
    fig75.suptitle('PCA 7.5: Cross-Lingual Value Shift (Neutral condition)',fontsize=11)
    fig75.tight_layout(); save_fig(fig75,'pca_7_5_cross_lingual.pdf')

plog('  PCA analyses done.')

# ── EXP 5: Training Ablation ─────────────────────────────────────────────────
plog('\n=== Exp 5: Training Ablation ===')
stage_groups=[('OLMo-7B',OLMO_7B),('OLMo-32B',OLMO_32B),('Tulu',TULU)]
supp_df_local = supp_df if 'supp_df' in dir() and len(supp_df) else pd.DataFrame()

fig5,axes5=plt.subplots(3,3,figsize=(18,14))
for ri,(name,stages) in enumerate(stage_groups):
    avail=[s for s in stages if s in refusal_df['model'].values]
    if not avail: continue
    xlbls=[STAGE_LBL.get(m,m.split('_')[-1]) for m in avail]
    c1,c2,c3=axes5[ri]

    for tier,col in [(1,'#4CAF50'),(2,'#FF9800'),(3,'#F44336')]:
        def get_metric(df,col_name):
            return [df[(df['model']==m)&(df['tier']==tier)][col_name].values for m in avail]
        vals=[v[0] if len(v) else np.nan for v in get_metric(refusal_df,'refusal_rate')]
        los =[v[0] if len(v) else np.nan for v in get_metric(refusal_df,'ci_lo')]
        his =[v[0] if len(v) else np.nan for v in get_metric(refusal_df,'ci_hi')]
        c1.plot(range(len(avail)),vals,marker='o',color=col,label=f'T{tier}',lw=2)
        c1.fill_between(range(len(avail)),
                        [l if not np.isnan(l) else v for l,v in zip(los,vals)],
                        [h if not np.isnan(h) else v for h,v in zip(his,vals)],alpha=0.15,color=col)

        nv=[v[0] if len(v) else np.nan for v in get_metric(nvas_df,'nvas')]
        nl=[v[0] if len(v) else np.nan for v in get_metric(nvas_df,'ci_lo')]
        nh=[v[0] if len(v) else np.nan for v in get_metric(nvas_df,'ci_hi')]
        c2.plot(range(len(avail)),nv,marker='o',color=col,label=f'T{tier}',lw=2)
        c2.fill_between(range(len(avail)),
                        [l if not np.isnan(l) else v for l,v in zip(nl,nv)],
                        [h if not np.isnan(h) else v for h,v in zip(nh,nv)],alpha=0.15,color=col)

    if len(supp_df_local):
        for tier,col in [(2,'#FF9800'),(3,'#F44336')]:
            sv=[supp_df_local[(supp_df_local['model']==m)&(supp_df_local['tier']==tier)&
                              (supp_df_local['threshold']==0.75)]['agreement_rate'].mean()
                if m in supp_df_local['model'].values else np.nan for m in avail]
            c3.plot(range(len(avail)),sv,marker='o',color=col,label=f'T{tier}',lw=2)

    for ax,ttl in [(c1,f'{name}: Refusal Rate'),(c2,f'{name}: NVAS'),(c3,f'{name}: WVS Agreement')]:
        ax.set_xticks(range(len(avail))); ax.set_xticklabels(xlbls)
        ax.set_title(ttl); ax.legend(fontsize=8); ax.set_ylim(0,1)

fig5.suptitle('Exp 5: Post-Training Ablation (OLMo-7B, OLMo-32B, Tulu)',fontsize=12)
fig5.tight_layout(); save_fig(fig5,'exp5_ablation.pdf')

abl_models=OLMO_7B+OLMO_32B+TULU
abl_models=[m for m in abl_models if m in refusal_df['model'].values]
if abl_models:
    rp=refusal_df[refusal_df['model'].isin(abl_models)].pivot(index='model',columns='tier',values='refusal_rate')
    np_=nvas_df[nvas_df['model'].isin(abl_models)].pivot(index='model',columns='tier',values='nvas')
    rp.columns=[f'Ref T{c}' for c in rp.columns]; np_.columns=[f'NVAS T{c}' for c in np_.columns]
    comb=pd.concat([rp,np_],axis=1)
    comb.index=[STAGE_LBL.get(m,m) for m in comb.index]
    save_latex(latex_tbl(comb.round(3),'Refusal rates and NVAS across training stages.','tab:exp5'),'exp5_ablation.tex')
plog('  Exp 5 done.')

# ── EXP 10: Directional SPD ──────────────────────────────────────────────────
plog('\n=== Exp 10: Directional SPD ===')

neutral_map = MASTER[
    (MASTER['sheet']=='No Mention') & (MASTER['country']=='NEUTRAL') &
    ~MASTER['refusal'] & MASTER['extracted'].notna()
][['model','question_id','extracted','vmin','vmax']].rename(columns={'extracted':'n_val','vmin':'n_vm','vmax':'n_vx'})

pers_map = MASTER[
    (MASTER['sheet']=='Personalization') & MASTER['human_mean'].notna() &
    ~MASTER['refusal'] & MASTER['extracted'].notna()
][['model','country','question_id','tier','extracted','human_mean','vmin','vmax']].rename(columns={'extracted':'p_val'})

obs_map = MASTER[
    (MASTER['sheet']=='Third') & MASTER['human_mean'].notna() &
    ~MASTER['refusal'] & MASTER['extracted'].notna()
][['model','country','question_id','tier','extracted','human_mean','vmin','vmax']].rename(columns={'extracted':'o_val'})

pn10=pers_map.merge(neutral_map,on=['model','question_id'],how='inner')
on10=obs_map.merge(neutral_map,on=['model','question_id'],how='inner')

def nvas_shift(df, val, ref, vm, vx, hm):
    rng=(df[vx]-df[vm]).replace(0,np.nan)
    return (1-(df[val]-df[hm]).abs()/rng) - (1-(df[ref]-df[hm]).abs()/rng)

pn10['shift_NP'] = nvas_shift(pn10,'p_val','n_val','vmin','vmax','human_mean')
on10['shift_NO'] = nvas_shift(on10,'o_val','n_val','vmin','vmax','human_mean')

rows10np=[]; rows10no=[]
for (model,tier),g in pn10.groupby(['model','tier']):
    obs,lo,hi=bootstrap_ci(g['shift_NP'].dropna().values)
    rows10np.append({'model':model,'tier':int(tier),'shift':obs,'lo':lo,'hi':hi,'type':'N→P'})
for (model,tier),g in on10.groupby(['model','tier']):
    obs,lo,hi=bootstrap_ci(g['shift_NO'].dropna().values)
    rows10no.append({'model':model,'tier':int(tier),'shift':obs,'lo':lo,'hi':hi,'type':'N→O'})
spd10=pd.DataFrame(rows10np+rows10no)
spd10.to_csv(os.path.join(RESULTS,'exp10_directional_spd.csv'),index=False)

models10=OLMO_7B+OLMO_32B+TULU+[m for m in sorted(MASTER['model'].unique())
                                   if m not in OLMO_7B+OLMO_32B+TULU][:5]
fig10,axes10=plt.subplots(2,3,figsize=(18,10))
for ri,(shift_type,lbl) in enumerate([('N→P','Shift N→P (Persona)'),('N→O','Shift N→O (Observer)')]):
    for ci,tier in enumerate([1,2,3]):
        ax=axes10[ri][ci]
        sub=spd10[(spd10['type']==shift_type)&(spd10['tier']==tier)]
        avail=[m for m in models10 if m in sub['model'].values]
        vals=[sub[sub['model']==m]['shift'].values[0] if m in sub['model'].values else np.nan for m in avail]
        los=[sub[sub['model']==m]['lo'].values[0] if m in sub['model'].values else np.nan for m in avail]
        his=[sub[sub['model']==m]['hi'].values[0] if m in sub['model'].values else np.nan for m in avail]
        va=np.array(vals,dtype=float); la=np.array(los,dtype=float); ha_=np.array(his,dtype=float)
        valid=~np.isnan(va)
        if valid.any():
            bcols=['#1565C0' if 'base' in m else '#E53935' for m in avail]
            el=va[valid]-la[valid]; eh=ha_[valid]-va[valid]
            el=np.where(np.isnan(el),0,el); eh=np.where(np.isnan(eh),0,eh)
            ax.barh(np.where(valid)[0],va[valid],xerr=[el,eh],
                    capsize=3,color=np.array(bcols)[valid],alpha=0.8)
            ax.set_yticks(range(len(avail)))
            ax.set_yticklabels([m.replace('_',' ')[:20] for m in avail],fontsize=7)
        ax.axvline(0,color='black',lw=0.8)
        ax.set_xlabel(lbl); ax.set_title(f'Tier {tier}')
fig10.suptitle('Exp 10: Directional SPD (+= adoption improves cultural alignment)',fontsize=11)
fig10.tight_layout(); save_fig(fig10,'exp10_directional_spd.pdf')

np_piv=spd10[spd10['type']=='N→P'].pivot(index='model',columns='tier',values='shift').round(3)
no_piv=spd10[spd10['type']=='N→O'].pivot(index='model',columns='tier',values='shift').round(3)
np_piv.columns=[f'NP T{c}' for c in np_piv.columns]
no_piv.columns=[f'NO T{c}' for c in no_piv.columns]
save_latex(latex_tbl(pd.concat([np_piv,no_piv],axis=1),
    r'Directional SPD: N$\to$P = NVAS(Persona)$-$NVAS(Neutral). Positive = improves.',
    'tab:exp10'),'exp10_directional_spd.tex')
plog('  Exp 10 done.')

# ── EXP 11: Suppression Index ─────────────────────────────────────────────────
plog('\n=== Exp 11: Suppression Index ===')

pers11 = MASTER[
    (MASTER['sheet']=='Personalization') & MASTER['human_mean'].notna() &
    ~MASTER['refusal'] & MASTER['extracted'].notna()
][['model','country','question_id','tier','extracted','human_mean','vmin','vmax']].rename(columns={'extracted':'p_val'})

obs11 = MASTER[
    (MASTER['sheet']=='Third') & MASTER['human_mean'].notna() &
    ~MASTER['refusal'] & MASTER['extracted'].notna()
][['model','country','question_id','tier','extracted','human_mean','vmin','vmax']].rename(columns={'extracted':'o_val'})

m11 = pers11.merge(obs11[['model','country','question_id','o_val']], on=['model','country','question_id'],how='inner')
rng11=(m11['vmax']-m11['vmin']).replace(0,np.nan)
m11['nvas_p']=1-(m11['p_val']-m11['human_mean']).abs()/rng11
m11['nvas_o']=1-(m11['o_val']-m11['human_mean']).abs()/rng11
m11['sup']=m11['nvas_o']-m11['nvas_p']

rows11=[]
for (model,tier),g in m11.groupby(['model','tier']):
    obs,lo,hi=bootstrap_ci(g['sup'].dropna().values)
    rows11.append({'model':model,'tier':int(tier),'sup':obs,'lo':lo,'hi':hi,'n':len(g)})
sup11=pd.DataFrame(rows11)
sup11.to_csv(os.path.join(RESULTS,'exp11_suppression.csv'),index=False)

piv11=sup11.pivot(index='model',columns='tier',values='sup')
sort_col=3 if 3 in piv11.columns else piv11.columns[-1]
piv11=piv11.sort_values(sort_col,ascending=False,na_position='last')

fig11,axes11=plt.subplots(1,2,figsize=(16,8))
x11=np.arange(len(piv11)); w11=0.25
ax=axes11[0]
for i,tier in enumerate([1,2,3]):
    if tier not in piv11.columns: continue
    t=sup11[(sup11['tier']==tier)].set_index('model').reindex(piv11.index)
    elo=(t['sup']-t['lo']).fillna(0).values; ehi=(t['hi']-t['sup']).fillna(0).values
    ax.bar(x11+i*w11,piv11[tier].fillna(0),w11,label=f'T{tier}',
           color=cols[tier],alpha=0.85,yerr=[elo,ehi],capsize=2,error_kw={'elinewidth':0.8})
ax.axhline(0,color='black',lw=0.8,ls='--')
ax.set_xticks(x11+w11); ax.set_xticklabels([m.replace('_',' ') for m in piv11.index],rotation=45,ha='right',fontsize=7)
ax.set_ylabel('SUP = NVAS(Obs)−NVAS(Persona)')
ax.set_title('Exp 11: Suppression Index\n(>0 = knows but hides)'); ax.legend()

ax2=axes11[1]
olmo_a=[m for m in OLMO_7B+OLMO_32B+TULU if m in sup11['model'].values]
for tier,col,ls in [(1,'#4CAF50','-'),(2,'#FF9800','--'),(3,'#F44336',':')]:
    vals=[sup11[(sup11['model']==m)&(sup11['tier']==tier)]['sup'].values for m in olmo_a]
    vals=[v[0] if len(v) else np.nan for v in vals]
    ax2.plot(range(len(olmo_a)),vals,marker='o',color=col,ls=ls,label=f'T{tier}',lw=2)
ax2.axhline(0,color='black',lw=0.8,ls='--')
ax2.set_xticks(range(len(olmo_a)))
ax2.set_xticklabels([STAGE_LBL.get(m,m) for m in olmo_a],rotation=30,ha='right',fontsize=8)
ax2.set_ylabel('Suppression Index'); ax2.set_title('OLMo/Tulu Stages'); ax2.legend()
fig11.tight_layout(); save_fig(fig11,'exp11_suppression_index.pdf')

pt11=piv11.copy(); pt11.columns=[f'Tier {c}' for c in pt11.columns]
save_latex(latex_tbl(pt11.round(3),
    r'Suppression Index = NVAS(Observer)$-$NVAS(Persona). SUP$>0$ = ``knows but hides.''',
    'tab:exp11'),'exp11_suppression_index.tex')
plog('  Exp 11 done.')

# ── Summary Figure ────────────────────────────────────────────────────────────
plog('\n=== Summary Figure ===')
fig_s,axes_s=plt.subplots(2,3,figsize=(22,12))

ax=axes_s[0][0]
t3r=refusal_df[refusal_df['tier']==3].sort_values('refusal_rate',ascending=False)
ax.barh(range(len(t3r)),t3r['refusal_rate'].values,
        color=['#1565C0' if 'base' in m else '#E53935' for m in t3r['model']],alpha=0.85)
ax.set_yticks(range(len(t3r))); ax.set_yticklabels([m.replace('_',' ') for m in t3r['model']],fontsize=7)
ax.set_xlabel('Refusal Rate'); ax.set_title('Tier 3 Refusal Rate (blue=base, red=aligned)')

ax=axes_s[0][1]
t3n=nvas_df[nvas_df['tier']==3].sort_values('nvas',ascending=False)
ax.barh(range(len(t3n)),t3n['nvas'].values,
        color=['#1565C0' if 'base' in m else '#E53935' for m in t3n['model']],alpha=0.85)
ax.set_yticks(range(len(t3n))); ax.set_yticklabels([m.replace('_',' ') for m in t3n['model']],fontsize=7)
ax.set_xlabel('NVAS (Tier 3)'); ax.set_title('Tier 3 NVAS')

ax=axes_s[0][2]
t3s=sup11[sup11['tier']==3].sort_values('sup',ascending=False)
ax.barh(range(len(t3s)),t3s['sup'].values,
        color=['#D32F2F' if v>0 else '#388E3C' for v in t3s['sup']],alpha=0.85)
ax.set_yticks(range(len(t3s))); ax.set_yticklabels([m.replace('_',' ') for m in t3s['model']],fontsize=7)
ax.axvline(0,color='black',lw=0.8)
ax.set_xlabel('Suppression Index'); ax.set_title('Tier 3 Suppression Index (red=suppressed)')

ax=axes_s[1][0]
fcs_a=cons_df.groupby('model')['FCS'].mean().sort_values()
ax.barh(range(len(fcs_a)),fcs_a.values,color='steelblue',alpha=0.85)
ax.set_yticks(range(len(fcs_a))); ax.set_yticklabels([m.replace('_',' ') for m in fcs_a.index],fontsize=7)
ax.set_xlabel('FCS'); ax.set_xlim(0,1); ax.set_title('Framing Consistency Score')

ax=axes_s[1][1]
clcs_a=cons_df.groupby('model')['CLCS'].mean().sort_values()
ax.barh(range(len(clcs_a)),clcs_a.values,color='darkorange',alpha=0.85)
ax.set_yticks(range(len(clcs_a))); ax.set_yticklabels([m.replace('_',' ') for m in clcs_a.index],fontsize=7)
ax.set_xlabel('CLCS'); ax.set_xlim(0,1); ax.set_title('Cross-Lingual Consistency Score')

ax=axes_s[1][2]
if 'T3_m_T1' in piv.columns:
    gap_s=piv['T3_m_T1'].dropna().sort_values(ascending=True)
    ax.barh(range(len(gap_s)),gap_s.values,
            color=['#D32F2F' if v>0 else '#388E3C' for v in gap_s],alpha=0.85)
    ax.set_yticks(range(len(gap_s))); ax.set_yticklabels([m.replace('_',' ') for m in gap_s.index],fontsize=7)
    ax.axvline(0,color='black',lw=0.8)
    ax.set_xlabel('Safety Tax'); ax.set_title('Safety Tax (T3−T1 Refusal Gap)')

fig_s.suptitle('MENA LLM Value Alignment — Summary',fontsize=14,fontweight='bold')
fig_s.tight_layout(); save_fig(fig_s,'summary_overview.pdf')

# ── LaTeX Report ──────────────────────────────────────────────────────────────
plog('\n=== Writing LaTeX Report ===')
report = r"""\documentclass[10pt,a4paper]{article}
\usepackage{booktabs,longtable,graphicx,float,caption}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref,amsmath,amssymb}
\title{MENA LLM Value Alignment: Experimental Results}
\author{Automated Analysis Pipeline}
\date{\today}
\begin{document}
\maketitle\tableofcontents\newpage

\section{Data Overview}
\begin{itemize}
  \item \textbf{Questions}: 864 survey questions (WVS/Arab Opinion Index).
  \item \textbf{Tiers}: T1 ($n=47$, benign), T2 ($n=788$, moderate), T3 ($n=29$, safety-sensitive).
  \item \textbf{Models}: 26 — OLMo-7B/32B (Base/SFT/DPO/Instruct), Tulu-3, Llama-3.1-8B, Qwen-2.5/3, Gemma-3, GPT-4o-mini, Mistral-7B, AYA-8B/32B, JAIS, FANAR, ALLAM.
  \item \textbf{Countries}: 16 MENA (Arabic/Persian/Turkish language families).
  \item \textbf{Conditions}: No~Mention (neutral EN), Personalization (EN/native), Third/Observer (EN/native).
\end{itemize}

\section{Experiment 2a: Safety Tax}
Refusal = non-numeric extracted answer. 95\% bootstrap CIs, $B=1000$.
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp2a_safety_tax.pdf}
\caption{Left: refusal rates by tier. Right: safety-tax gap (T3$-$T1).}\end{figure}
\input{exp2a_safety_tax}

\section{Experiment 2b: NVAS by Tier}
$\text{NVAS}=1-|v_m-v_h|/(v_{\max}-v_{\min})$ for non-refused answers.
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp2b_nvas_by_tier.pdf}
\caption{NVAS by tier and accuracy gap T3$-$T1.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=.6\textwidth]{exp2b_olmo_tier3.pdf}
\caption{OLMo variants: Tier 3 NVAS (base=blue, aligned=red).}\end{figure}
\input{exp2b_nvas}

\section{Experiment 3: Direction of Suppression}
For refused questions: does the renormalized distribution match the WVS majority?
Multi-threshold analysis (50\%, 75\%, 90\%). Conservative framing.
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp3_suppression.pdf}
\caption{WVS agreement for refused qs by tier/threshold; pool mass on digit tokens.}\end{figure}
\input{exp3_suppression}

\section{Experiment 5: Post-Training Ablation}
Refusal rate, NVAS, WVS-agreement as function of training stage.
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp5_ablation.pdf}
\caption{Ablation: OLMo-7B (row 1), OLMo-32B (row 2), Tulu (row 3).}\end{figure}
\input{exp5_ablation}

\section{Consistency Metrics (FCS, CLCS, SPD)}
$\text{FCS}=1-D(v^{\text{persona}},v^{\text{observer}})$;
$\text{CLCS}=1-D(v^{\text{EN}},v^{\text{native}})$;
$\text{SPD}=1-D(v^{\text{neutral}},v^{\text{persona}})$.
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{consistency_heatmaps.pdf}
\caption{FCS, CLCS, SPD heatmaps per model and country.}\end{figure}
\input{consistency_metrics}

\section{PCA Analyses}
\subsection{PCA 7.1: Human Survey}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{pca_7_1_human.pdf}
\caption{Human country vectors (left) and JSD similarity (right).}\end{figure}

\subsection{PCA 7.2: LLM Observer English}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{pca_7_2_observer_english.pdf}
\caption{Observer (English) PCA for all models.}\end{figure}

\subsection{PCA 7.3: Native Language}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{pca_7_3_native.pdf}
\caption{Native language: countries collapse into language-family clusters.}\end{figure}

\subsection{PCA 7.4: Persona + Neutral}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{pca_7_4_persona_neutral.pdf}
\caption{LLM neutral ($\star$) as outlier — Cultural Identity Crisis.}\end{figure}

\subsection{PCA 7.5: Cross-Lingual Value Shift}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{pca_7_5_cross_lingual.pdf}
\caption{|English$-$Native| answer distribution.}\end{figure}

\section{Experiment 10: Directional SPD}
$\text{Shift}_{N\to P}=\text{NVAS(Persona)}-\text{NVAS(Neutral)}$. Positive = improves.
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp10_directional_spd.pdf}
\caption{Signed shift: N$\to$P (top) and N$\to$O (bottom) by tier.}\end{figure}
\input{exp10_directional_spd}

\section{Experiment 11: Suppression Index}
$\text{SUP}=\text{NVAS(Observer)}-\text{NVAS(Persona)}$. SUP$>0$ = ``knows but hides.''
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{exp11_suppression_index.pdf}
\caption{Suppression index by tier and OLMo/Tulu stages.}\end{figure}
\input{exp11_suppression_index}

\section{Summary}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{summary_overview.pdf}
\caption{6-panel summary overview.}\end{figure}

\section{Experiments 4 \& 7 (Require Model Weights)}
Experiment~4 (linear probing on residual-stream activations) and Experiment~7 (SAE
language-family dominance) require loading model weights.
Run \texttt{python run\_probing.py} (GPU required).
\end{document}
"""
with open(os.path.join(RESULTS,'report.tex'),'w') as f: f.write(report)
plog('  Saved report.tex')

plog('\n=== ALL EXPERIMENTS COMPLETE ===')
plog(f'Results directory: {RESULTS}')
for f in sorted(os.listdir(RESULTS)):
    sz=os.path.getsize(os.path.join(RESULTS,f))
    plog(f'  {f:40s} {sz:8,d} bytes')
