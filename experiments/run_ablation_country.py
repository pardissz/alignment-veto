"""
Priority 3: Causal test — does the country cue actually inform the model's
latent distribution during T3 refusals?

Method: cross-country permutation test using existing master_table data.
No new model inference needed.

For each T3 refused row (model M, country X, question Q):
  - true_mass   = latent probability on WVS majority answer for country X
  - mean_other  = mean latent probability on WVS majority answers for all
                  OTHER countries that have WVS data for question Q
  - delta = true_mass - mean_other

If delta > 0 consistently → the model's latent distribution specifically
agrees with the target country, not with a generic MENA distribution.
This provides causal-style evidence that the country name in the prompt
causally shapes the suppressed distribution.

Secondary test: rank the true country among all 16 countries by latent mass
on that country's WVS majority. Mean rank << 8.5 (random baseline) confirms
country-specificity.

Outputs:
  results/ablation_country_delta.csv  — per-row deltas
  results/ablation_country_summary.csv — per-model summary
  results/ablation_country_rank.pdf   — plot
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

print('Loading master_table.pkl ...')
mt = pickle.load(open(os.path.join(RESULTS, 'master_table.pkl'), 'rb'))

# Focus on Personalization + Third (country-aware framings), T3, refused, with WVS data
t3_ref = mt[
    (mt['tier'] == 3) &
    (mt['refusal'] == True) &
    (mt['country'] != 'NEUTRAL') &
    mt['human_dist'].apply(lambda x: isinstance(x, dict) and len(x) > 0) &
    mt['norm_probs'].apply(lambda x: isinstance(x, dict) and len(x) > 0) &
    mt['sheet'].isin(['Personalization', 'Third'])
].copy().reset_index(drop=True)

print(f'T3 refused rows (Persona+Third, with WVS): {len(t3_ref)}')

# Build a lookup: (question_id, country) → wvs majority answer
# We need WVS data for ALL countries for each question
# Get it from all non-refusal rows in t3 with WVS data
t3_all = mt[
    (mt['tier'] == 3) &
    (mt['country'] != 'NEUTRAL') &
    mt['human_dist'].apply(lambda x: isinstance(x, dict) and len(x) > 0)
].copy()

# WVS majority per (question_id, country)
wvs_lookup = {}  # (qid, country) → majority_answer (int)
for _, row in t3_all.drop_duplicates(['question_id', 'country']).iterrows():
    qid = int(row['question_id'])
    cty = row['country']
    hd  = row['human_dist']
    if hd:
        maj = int(max(hd, key=lambda k: hd[k]))
        wvs_lookup[(qid, cty)] = maj

# All countries with WVS data per question
from collections import defaultdict
q_countries = defaultdict(set)  # qid → set of countries with WVS
for (qid, cty) in wvs_lookup:
    q_countries[qid].add(cty)

print(f'WVS lookup entries: {len(wvs_lookup)}')
print(f'T3 questions with ≥2 WVS countries: {sum(1 for v in q_countries.values() if len(v)>=2)}')

# ── Compute delta per row ─────────────────────────────────────────────────────
delta_rows = []
for _, row in t3_ref.iterrows():
    qid = int(row['question_id'])
    cty = row['country']
    np_  = row['norm_probs']

    true_maj = wvs_lookup.get((qid, cty))
    if true_maj is None:
        continue

    true_mass = np_.get(str(true_maj), 0.0)

    # Mass on WVS majority for all OTHER countries for this question
    other_masses = []
    for other_cty in q_countries.get(qid, set()):
        if other_cty == cty:
            continue
        other_maj = wvs_lookup.get((qid, other_cty))
        if other_maj is None:
            continue
        other_masses.append(np_.get(str(other_maj), 0.0))

    if len(other_masses) < 2:
        continue  # need at least 2 other countries

    mean_other = np.mean(other_masses)
    delta = true_mass - mean_other

    # Rank of true country (1 = highest mass)
    all_masses = [true_mass] + other_masses
    all_masses_sorted = sorted(all_masses, reverse=True)
    rank = all_masses_sorted.index(true_mass) + 1  # 1-indexed

    delta_rows.append({
        'model': row['model'],
        'sheet': row['sheet'],
        'country': cty,
        'question_id': qid,
        'true_mass': true_mass,
        'mean_other_mass': mean_other,
        'delta': delta,
        'rank': rank,
        'n_countries': len(other_masses) + 1,
    })

delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(os.path.join(RESULTS, 'ablation_country_delta.csv'), index=False)
print(f'\nComputed deltas for {len(delta_df)} rows.')
print(f'Mean delta (true - other): {delta_df["delta"].mean():.4f}')
print(f'Fraction positive: {(delta_df["delta"] > 0).mean():.3f}')
print(f'Mean rank (1=best): {delta_df["rank"].mean():.2f} (random baseline: {(delta_df["n_countries"]+1).mean()/2:.2f})')

# One-sample t-test: delta > 0?
t, p = stats.ttest_1samp(delta_df['delta'], 0, alternative='greater')
print(f'One-sample t-test (delta > 0): t={t:.2f}, p={p:.2e}')

# Wilcoxon signed-rank
w, pw = stats.wilcoxon(delta_df['delta'], alternative='greater')
print(f'Wilcoxon signed-rank (delta > 0): W={w:.0f}, p={pw:.2e}')

# ── Per-model summary ─────────────────────────────────────────────────────────
label_map = {
    'allam_7b_instruct': 'ALLAM-7B',
    'aya_expanse_8b': 'AYA-8B',
    'aya_expanse_32b': 'AYA-32B',
    'fanar_1_9b_instruct': 'FANAR-9B',
    'gemma_3_4b_it': 'Gemma3-4B',
    'gemma_3_12b_it': 'Gemma3-12B',
    'gemma_3_27b_it': 'Gemma3-27B',
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt_4o_mini': 'GPT-4o-mini',
    'jais_2_8b_chat': 'JAIS-8B',
    'llama_3.1_8b_instruct': 'Llama3.1-8B-IT',
    'llama_3.1_8b_base': 'Llama3.1-8B-Base',
    'olmo_3_7b_instruct': 'OLMo3-7B-IT',
    'olmo_3_7b_dpo': 'OLMo3-7B-DPO',
    'olmo_3_7b_sft': 'OLMo3-7B-SFT',
    'olmo_3_7b_base': 'OLMo3-7B-Base',
    'olmo_3_32b_instruct': 'OLMo3-32B-IT',
    'olmo_3_32b_base': 'OLMo3-32B-Base',
    'qwen2.5_7b_instruct': 'Qwen2.5-7B',
    'qwen3_4b_instruct': 'Qwen3-4B',
    'qwen3_30b_a3b_instruct': 'Qwen3-30B',
    'mistral_7b_instruct': 'Mistral-7B',
    'tulu_3_8b_instruct': 'Tulu3-8B-IT',
    'tulu_3_8b_dpo': 'Tulu3-8B-DPO',
    'tulu_3_8b_sft': 'Tulu3-8B-SFT',
}

def family_color(m):
    if 'olmo' in m:   return '#1565C0'
    if 'tulu' in m:   return '#6A1B9A'
    if 'llama' in m:  return '#2E7D32'
    if 'gemma' in m:  return '#F57F17'
    if 'gpt' in m:    return '#B71C1C'
    if 'qwen' in m:   return '#00838F'
    if 'aya' in m:    return '#AD1457'
    if 'mistral' in m: return '#4E342E'
    return '#37474F'

summary_rows = []
for model in delta_df['model'].unique():
    sub = delta_df[delta_df['model'] == model]
    n   = len(sub)
    mean_d = sub['delta'].mean()
    se_d   = sub['delta'].std() / np.sqrt(n)
    frac_pos = (sub['delta'] > 0).mean()
    mean_rank = sub['rank'].mean()
    n_countries_mean = sub['n_countries'].mean()
    t_stat, p_val = stats.ttest_1samp(sub['delta'], 0, alternative='greater')
    summary_rows.append({
        'model': model,
        'label': label_map.get(model, model),
        'n': n,
        'mean_delta': mean_d,
        'se_delta': se_d,
        'frac_positive': frac_pos,
        'mean_rank': mean_rank,
        'expected_rank': (n_countries_mean + 1) / 2,
        't_stat': t_stat,
        'p_value': p_val,
    })
summary = pd.DataFrame(summary_rows).sort_values('mean_delta', ascending=False)
summary.to_csv(os.path.join(RESULTS, 'ablation_country_summary.csv'), index=False)

print('\n--- Per-model country cue causal test ---')
print(f"{'Model':<22} {'N':>5} {'Mean Δ':>8} {'Frac+':>7} {'Mean rank':>10} {'p':>10}")
for _, r in summary.iterrows():
    sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else ('*' if r['p_value'] < 0.05 else ''))
    print(f"  {r['label']:<20} {r['n']:>5} {r['mean_delta']:>8.4f} "
          f"{r['frac_positive']:>7.3f} {r['mean_rank']:>10.2f} "
          f"{r['p_value']:>10.4f} {sig}")

# ── Figure: horizontal bar chart of mean_delta per model ─────────────────────
summary_plot = summary.sort_values('mean_delta', ascending=True).reset_index(drop=True)
n = len(summary_plot)
fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n * 0.38)))
fig.patch.set_facecolor('white')

# Left: mean delta
ax1 = axes[0]
ax1.set_facecolor('#FAFAFA')
y = np.arange(n)
colors = [family_color(m) for m in summary_plot['model']]
ax1.barh(y, summary_plot['mean_delta'], color=colors, height=0.65,
         edgecolor='white', linewidth=0.4, zorder=3)
ax1.errorbar(summary_plot['mean_delta'], y,
             xerr=1.96 * summary_plot['se_delta'],
             fmt='none', ecolor='#333', elinewidth=0.9, capsize=3, alpha=0.6, zorder=4)
ax1.axvline(0, color='#333', lw=1.2, zorder=5)
ax1.set_yticks(y)
ax1.set_yticklabels(summary_plot['label'], fontsize=8.5)
ax1.set_xlabel('Mean Δ (true country mass − mean other country mass)', fontsize=9.5)
ax1.set_title('Country cue causal test:\nDoes the model agree specifically with\n'
              'the prompted country\'s WVS majority?', fontsize=10)
ax1.grid(axis='x', lw=0.4, alpha=0.5, color='#ccc', zorder=0)
for i, (_, row) in enumerate(summary_plot.iterrows()):
    sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else ''))
    if sig:
        x_pos = row['mean_delta'] + 1.96 * row['se_delta'] + 0.002
        ax1.text(x_pos, i, sig, va='center', ha='left', fontsize=8, color='#333')

# Right: mean rank vs expected rank (random baseline)
ax2 = axes[1]
ax2.set_facecolor('#FAFAFA')
ax2.barh(y, summary_plot['expected_rank'] - summary_plot['mean_rank'],
         color=colors, height=0.65, edgecolor='white', linewidth=0.4, zorder=3)
ax2.axvline(0, color='#333', lw=1.2, zorder=5)
ax2.set_yticks(y)
ax2.set_yticklabels(summary_plot['label'], fontsize=8.5)
ax2.set_xlabel('Expected rank − Mean rank\n(positive = true country ranked higher than chance)', fontsize=9.5)
ax2.set_title('Rank improvement over random:\nhigher = model more country-specific', fontsize=10)
ax2.grid(axis='x', lw=0.4, alpha=0.5, color='#ccc', zorder=0)

fig.suptitle('Does the country name causally inform the suppressed distribution?\n'
             'Cross-country permutation test on T3 refusals (no new inference needed)',
             fontsize=11, y=1.01)
fig.tight_layout()
out = os.path.join(RESULTS, 'ablation_country_rank.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'\nSaved: {out}')
print('Done.')
