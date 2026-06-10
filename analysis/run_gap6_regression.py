"""
Gap 6: Formal regression model.
Mixed-effects OLS: NVAS ~ tier * framing + lang_family + scale_type + model_family
Also logistic regression for refusal.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

BASE = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

mt = pd.read_pickle(os.path.join(RESULTS, 'master_table.pkl'))

def nvas_f(row):
    if pd.isna(row['extracted']) or pd.isna(row['human_mean']): return np.nan
    d = row['vmax'] - row['vmin']
    return np.nan if d == 0 else 1 - abs(row['extracted'] - row['human_mean']) / d

mt_acc = mt[~mt['refusal']].copy()
mt_acc['nvas'] = mt_acc.apply(nvas_f, axis=1)

# Model family classification
BASE_MODELS = {'olmo_3_7b_base','olmo_3_32b_base','llama_3.1_8b_base'}
MENA_MODELS = {'allam_7b_instruct','fanar_1_9b_instruct','jais_2_8b_chat','aya_expanse_8b','aya_expanse_32b'}
FRONTIER    = {'gpt4o_mini','gpt_5'}

def model_family(mk):
    if mk in BASE_MODELS: return 'base'
    if mk in MENA_MODELS: return 'mena'
    if mk in FRONTIER: return 'frontier'
    return 'instruct'

def scale_type(row):
    r = row['vmax'] - row['vmin']
    if r <= 1: return 'binary'
    if r <= 4: return 'short'
    if r <= 9: return 'medium'
    return 'long'

# Prepare full dataset
full = mt_acc.copy()
full['nvas'] = full['nvas']
full = full.dropna(subset=['nvas','tier','lang_family'])
full['tier_cat']    = full['tier'].map({1:'T1',2:'T2',3:'T3'})
full['framing']     = full['sheet'].map({
    'Personalization':'Persona', 'Personalization_Diff':'Persona_Nat',
    'Third':'Third', 'Third_Diff':'Third_Nat',
    'No Mention':'NM', 'No Mention Diff':'NM_Nat'
})
full['model_fam']   = full['model'].apply(model_family)
full['scale_type']  = full.apply(scale_type, axis=1)
full = full.dropna(subset=['framing'])

print(f'Regression dataset: {len(full)} rows (accepted responses with NVAS)')

# ── OLS regression with statsmodels ───────────────────────────────────────────
try:
    import statsmodels.formula.api as smf

    # Main effects model
    formula_main = ('nvas ~ C(tier_cat, Treatment("T1")) '
                    '+ C(framing, Treatment("Persona")) '
                    '+ C(lang_family, Treatment("Arabic")) '
                    '+ C(model_fam, Treatment("instruct")) '
                    '+ C(scale_type, Treatment("medium"))')
    mod_main = smf.ols(formula_main, data=full).fit()
    print('\n=== OLS: NVAS ~ tier + framing + lang_family + model_fam + scale_type ===')
    print(mod_main.summary().tables[1])
    print(f'R² = {mod_main.rsquared:.4f}  Adj-R² = {mod_main.rsquared_adj:.4f}')

    # Interaction model: tier × framing
    formula_int = ('nvas ~ C(tier_cat, Treatment("T1")) * C(framing, Treatment("Persona")) '
                   '+ C(lang_family, Treatment("Arabic")) '
                   '+ C(model_fam, Treatment("instruct")) '
                   '+ C(scale_type, Treatment("medium"))')
    mod_int = smf.ols(formula_int, data=full).fit()
    print('\n=== OLS + tier×framing interaction ===')
    # Print just the interaction terms
    coefs = mod_int.params
    pvals = mod_int.pvalues
    ses   = mod_int.bse
    inter = [(k,v,ses[k],pvals[k]) for k,v in coefs.items() if ':' in k]
    print(f'R² = {mod_int.rsquared:.4f}  Adj-R² = {mod_int.rsquared_adj:.4f}')
    print('\nInteraction terms (tier × framing):')
    print(f"{'Term':<60} {'coef':>8} {'se':>6} {'p':>8}")
    for k, c, s, p in sorted(inter, key=lambda x: abs(x[1]), reverse=True):
        print(f'  {k:<58} {c:>8.4f} {s:>6.4f} {p:>8.4f}')

    # Save coefficient table
    coef_df = pd.DataFrame({'coef': mod_int.params, 'se': mod_int.bse,
                            'pval': mod_int.pvalues, 'ci_low': mod_int.conf_int()[0],
                            'ci_high': mod_int.conf_int()[1]})
    coef_df.to_csv(os.path.join(RESULTS, 'gap6_regression_coefs.csv'))
    print('\nSaved: gap6_regression_coefs.csv')

    # ── Forest plot of key coefficients ───────────────────────────────────────
    key_terms = {k: v for k, v in coefs.items()
                 if any(x in k for x in ['T2','T3','Third','Persona_Nat','NM','mena','frontier','Arabic','Turkish'])}
    key_terms = dict(sorted(key_terms.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(10, max(5, len(key_terms)*0.38)))
    fig.patch.set_facecolor('white'); ax.set_facecolor('#FAFAFA')
    y = np.arange(len(key_terms))
    for i, (k, c) in enumerate(key_terms.items()):
        s   = ses[k]; p = pvals[k]
        col = '#E53935' if p < 0.05 else '#90A4AE'
        ax.barh(i, c, color=col, alpha=0.7 if p<0.05 else 0.4, edgecolor='white')
        ax.errorbar(c, i, xerr=1.96*s, fmt='none', color=col, lw=1.5, capsize=3)
        stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        ax.text(c + (0.005 if c>=0 else -0.005), i, stars, va='center',
                ha='left' if c>=0 else 'right', fontsize=8, color='#333')
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace('C(','').replace(', Treatment(','[ref=').replace(')','')
                        .replace('"','').replace('[T.','(').replace(']',')') for k in key_terms], fontsize=7.5)
    ax.axvline(0, color='#333', lw=1.0, ls='--', alpha=0.6)
    ax.set_xlabel('OLS coefficient (NVAS change)', fontsize=9.5)
    ax.set_title(f'Regression coefficients: NVAS ~ tier × framing + controls\n'
                 f'R²={mod_int.rsquared:.3f}; red = significant (p<0.05)',
                 fontsize=10)
    ax.grid(axis='x', lw=0.3, alpha=0.4, color='#ccc')
    fig.tight_layout()
    out = os.path.join(RESULTS, 'gap6_regression_forest.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
    print(f'Saved: {out}')

    # Logistic regression for refusal
    print('\n=== Logistic regression for refusal ===')
    ref_data = mt[mt['sheet'].isin(['Personalization','Third'])].copy()
    ref_data['tier_cat']  = ref_data['tier'].map({1:'T1',2:'T2',3:'T3'})
    ref_data['framing']   = ref_data['sheet'].map({'Personalization':'Persona','Third':'Third'})
    ref_data['model_fam'] = ref_data['model'].apply(model_family)
    ref_data['scale_type']= ref_data.apply(scale_type, axis=1)
    ref_data = ref_data.dropna(subset=['tier_cat','framing','lang_family'])
    ref_data['refused'] = ref_data['refusal'].astype(int)
    ref_formula = ('refused ~ C(tier_cat, Treatment("T1")) * C(framing, Treatment("Persona")) '
                   '+ C(lang_family, Treatment("Arabic")) + C(model_fam, Treatment("instruct"))')
    mod_logit = smf.logit(ref_formula, data=ref_data).fit(disp=0, maxiter=200)
    print(mod_logit.summary().tables[1])
    logit_df = pd.DataFrame({'coef': mod_logit.params, 'se': mod_logit.bse,
                              'p': mod_logit.pvalues}).reset_index()
    logit_df.columns = ['term','coef','se','p']
    logit_df.to_csv(os.path.join(RESULTS,'gap6_logit_coefs.csv'), index=False)
    print('Saved: gap6_logit_coefs.csv')

except ImportError:
    print('statsmodels not available, using scipy OLS')
    # Simple scipy OLS as fallback
    from scipy import stats as sc_stats
    mt_t3 = mt_acc[mt_acc['tier']==3].dropna(subset=['nvas'])
    mt_t2 = mt_acc[mt_acc['tier']==2].dropna(subset=['nvas'])
    t, p = sc_stats.ttest_ind(mt_t3['nvas'], mt_t2['nvas'])
    print(f'T3 vs T2 NVAS: t={t:.2f}, p={p:.4f}')

print('\nDone.')
