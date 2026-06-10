"""
Priority 2: Qualitative examples of the suppression phenomenon.

Finds T3 (safety-sensitive) questions where a model refuses to answer
(extracted number outside [vmin, vmax]) but its latent digit-token
distribution strongly agrees with the WVS majority answer.

These are the concrete smoking-gun cases: the model "knows" the culturally
accurate answer but suppresses it to outside the valid range.

Outputs:
  results/qualitative_examples.csv  — machine-readable table
  results/qualitative_examples.tex  — LaTeX table for paper
"""

import os, re, pickle
import numpy as np
import pandas as pd

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')
XLSX    = os.path.join(BASE, 'Fixed_no_mention_corrected (6).xlsx')

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading master_table.pkl ...')
mt = pickle.load(open(os.path.join(RESULTS, 'master_table.pkl'), 'rb'))

# Load question texts (Personalization sheet — has country-specific prompts)
xl = pd.ExcelFile(XLSX)
pers_df = xl.parse('Personalization').set_index('question_id')
third_df = xl.parse('Third').set_index('question_id')

def get_prompt_text(qid, country, sheet='Personalization'):
    df = pers_df if sheet == 'Personalization' else third_df
    if qid in df.index and country in df.columns:
        txt = str(df.loc[qid, country])
        if txt != 'nan':
            return txt
    return None


# ── Helper: compute WVS majority answer and latent agreement ──────────────────
def wvs_majority(human_dist: dict) -> int | None:
    """Return the scale value with the highest human fraction."""
    if not human_dist:
        return None
    return int(max(human_dist, key=lambda k: human_dist[k]))


def latent_agrees_with_wvs(norm_probs: dict, wvs_maj: int, threshold: float = 0.50) -> bool:
    """Does the latent distribution concentrate >= threshold mass on wvs_maj?"""
    if wvs_maj is None or not norm_probs:
        return False
    mass = norm_probs.get(str(wvs_maj), 0.0)
    return mass >= threshold


def top_latent_token(norm_probs: dict) -> tuple[str, float]:
    """Return (token, probability) of the highest-mass digit token."""
    if not norm_probs:
        return ('?', 0.0)
    tok = max(norm_probs, key=lambda k: norm_probs[k])
    return tok, norm_probs[tok]


# ── Filter to T3 refused rows with WVS data ───────────────────────────────────
print('Filtering T3 refusals with WVS data...')
t3_ref = mt[
    (mt['tier'] == 3) &
    (mt['refusal'] == True) &
    (mt['country'] != 'NEUTRAL') &
    mt['human_dist'].apply(lambda x: len(x) > 0) &
    mt['norm_probs'].apply(lambda x: isinstance(x, dict) and len(x) > 0)
].copy()

print(f'T3 refused rows with WVS data: {len(t3_ref)}')

# Compute WVS majority and latent agreement
t3_ref['wvs_maj'] = t3_ref['human_dist'].apply(wvs_majority)
t3_ref['wvs_maj_frac'] = t3_ref.apply(
    lambda r: r['human_dist'].get(str(r['wvs_maj']), 0.0), axis=1)
t3_ref['latent_top_token'] = t3_ref['norm_probs'].apply(
    lambda p: top_latent_token(p)[0])
t3_ref['latent_top_prob']  = t3_ref['norm_probs'].apply(
    lambda p: top_latent_token(p)[1])
t3_ref['latent_agrees_50'] = t3_ref.apply(
    lambda r: latent_agrees_with_wvs(r['norm_probs'], r['wvs_maj'], 0.50), axis=1)
t3_ref['latent_agrees_75'] = t3_ref.apply(
    lambda r: latent_agrees_with_wvs(r['norm_probs'], r['wvs_maj'], 0.75), axis=1)
t3_ref['latent_wvs_mass']  = t3_ref.apply(
    lambda r: r['norm_probs'].get(str(r['wvs_maj']), 0.0), axis=1)

# Sheets to include (avoid diff/native sheets for clarity)
t3_ref = t3_ref[t3_ref['sheet'].isin(['Personalization', 'Third'])]
print(f'After filtering to Personalization+Third sheets: {len(t3_ref)}')

# Sort by latent_wvs_mass descending (strongest suppression first)
t3_ref = t3_ref.sort_values('latent_wvs_mass', ascending=False).reset_index(drop=True)

# ── Select diverse examples: different models, questions, countries ─────────────
print('\nSelecting diverse top examples...')

selected = []
seen_qids = set()
seen_models = set()
model_count = {}

# Prefer instruction-tuned models with strong suppression
preferred_models = [
    'gemma_3_27b_it', 'gemma_3_12b_it', 'gemma_3_4b_it',
    'gpt4o_mini', 'allam_7b_instruct', 'fanar_1_9b_instruct',
    'aya_expanse_8b', 'aya_expanse_32b', 'qwen2.5_7b_instruct',
    'olmo_3_7b_instruct', 'olmo_3_32b_instruct', 'llama_3.1_8b_instruct',
]

def score_row(row):
    # Prefer high latent mass on WVS majority, high WVS consensus, clean prompt
    return row['latent_wvs_mass'] * (0.5 + row['wvs_maj_frac'])

for _, row in t3_ref.iterrows():
    if len(selected) >= 8:
        break
    if row['question_id'] in seen_qids and len(seen_qids) < 15:
        continue  # variety of questions
    m = row['model']
    if model_count.get(m, 0) >= 2:
        continue  # max 2 per model

    # Get the prompt text
    sheet = row['sheet']
    prompt = get_prompt_text(int(row['question_id']), row['country'], sheet)
    if prompt is None:
        continue

    # Shorten prompt: remove "Provide the answer number only." suffix, keep body
    prompt_short = re.sub(r'\s*Provide the answer number only\.\s*$', '', prompt).strip()
    prompt_short = re.sub(r'^Imagine you are [aA]n? \w+ person\. ', '', prompt_short).strip()
    prompt_short = re.sub(r'^Imagine you are a citizen of \w+\.? ', '', prompt_short).strip()
    if len(prompt_short) > 120:
        prompt_short = prompt_short[:117] + '...'

    selected.append({
        'model': row['model'],
        'sheet': row['sheet'],
        'country': row['country'],
        'question_id': int(row['question_id']),
        'prompt_short': prompt_short,
        'vmin': int(row['vmin']),
        'vmax': int(row['vmax']),
        'extracted': row['extracted'],
        'wvs_maj': row['wvs_maj'],
        'wvs_maj_frac': row['wvs_maj_frac'],
        'latent_top_token': row['latent_top_token'],
        'latent_top_prob': row['latent_top_prob'],
        'latent_wvs_mass': row['latent_wvs_mass'],
        'norm_probs_str': ', '.join(
            f'{k}:{v:.2f}' for k, v in
            sorted(row['norm_probs'].items(), key=lambda x: -x[1])[:4]
        ),
    })
    seen_qids.add(row['question_id'])
    model_count[m] = model_count.get(m, 0) + 1

if len(selected) < 6:
    # Relax constraint on unique questions
    for _, row in t3_ref.iterrows():
        if len(selected) >= 8:
            break
        m = row['model']
        if model_count.get(m, 0) >= 2:
            continue
        sheet = row['sheet']
        prompt = get_prompt_text(int(row['question_id']), row['country'], sheet)
        if prompt is None:
            continue
        prompt_short = re.sub(r'\s*Provide the answer number only\.\s*$', '', prompt).strip()
        prompt_short = re.sub(r'^Imagine you are [aA]n? \w+ person\. ', '', prompt_short).strip()
        if len(prompt_short) > 120:
            prompt_short = prompt_short[:117] + '...'
        if not any(s['question_id'] == int(row['question_id']) and
                   s['country'] == row['country'] and s['model'] == m
                   for s in selected):
            selected.append({
                'model': row['model'],
                'sheet': row['sheet'],
                'country': row['country'],
                'question_id': int(row['question_id']),
                'prompt_short': prompt_short,
                'vmin': int(row['vmin']),
                'vmax': int(row['vmax']),
                'extracted': row['extracted'],
                'wvs_maj': row['wvs_maj'],
                'wvs_maj_frac': row['wvs_maj_frac'],
                'latent_top_token': row['latent_top_token'],
                'latent_top_prob': row['latent_top_prob'],
                'latent_wvs_mass': row['latent_wvs_mass'],
                'norm_probs_str': ', '.join(
                    f'{k}:{v:.2f}' for k, v in
                    sorted(row['norm_probs'].items(), key=lambda x: -x[1])[:4]
                ),
            })
            model_count[m] = model_count.get(m, 0) + 1

ex = pd.DataFrame(selected)
ex.to_csv(os.path.join(RESULTS, 'qualitative_examples.csv'), index=False)
print(f'Selected {len(ex)} examples.')

# ── Pretty console print ──────────────────────────────────────────────────────
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
    'olmo_3_7b_instruct': 'OLMo3-7B-IT',
    'olmo_3_32b_instruct': 'OLMo3-32B-IT',
    'qwen2.5_7b_instruct': 'Qwen2.5-7B',
}

print('\n' + '='*80)
for i, row in ex.iterrows():
    mlabel = label_map.get(row['model'], row['model'])
    print(f"\nExample {i+1}: {mlabel} | {row['sheet']} | {row['country']} | Q{row['question_id']}")
    print(f"  Question: {row['prompt_short']}")
    print(f"  Scale: [{row['vmin']}, {row['vmax']}]")
    print(f"  Model output (refused): {row['extracted']}")
    print(f"  WVS majority: {row['wvs_maj']} ({row['wvs_maj_frac']:.1%} of respondents)")
    print(f"  Latent probs: {row['norm_probs_str']}")
    print(f"  → Mass on WVS majority: {row['latent_wvs_mass']:.1%}")

# ── LaTeX table ───────────────────────────────────────────────────────────────
lines = []
lines.append(r'\begin{table}[H]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\setlength{\tabcolsep}{4pt}')
lines.append(r'\caption{%')
lines.append(r'Qualitative examples of the suppression phenomenon on Tier~3 (safety-sensitive)')
lines.append(r'questions. Each row shows a refused response (model output falls outside the')
lines.append(r'valid scale) where the model\textquotesingle s latent digit-token distribution')
lines.append(r'nonetheless concentrates the majority of its probability mass on the answer')
lines.append(r'that matches the WVS majority view. The model ``knows\textquotesingle\textquotesingle')
lines.append(r'the culturally accurate answer but suppresses it below the scale minimum.')
lines.append(r'}')
lines.append(r'\label{tab:qualitative_examples}')
lines.append(r'\begin{tabular}{llcp{5cm}cccc}')
lines.append(r'\toprule')
lines.append(r'Model & Country & Framing & Question (abbreviated) & Scale & '
             r'Output & WVS maj. & Latent mass \\')
lines.append(r'\midrule')

for _, row in ex.iterrows():
    mlabel = label_map.get(row['model'], row['model'])
    framing = 'Persona' if 'Persona' in row['sheet'] else 'Third'
    q_tex   = row['prompt_short'].replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
    scale   = f"[{row['vmin']}, {row['vmax']}]"
    out_val = str(row['extracted']) if not pd.isna(row['extracted']) else r'\textit{out}'
    wvs_str = f"{row['wvs_maj']} ({row['wvs_maj_frac']:.0%})"
    mass_str = f"\\textbf{{{row['latent_wvs_mass']:.0%}}}"
    lines.append(
        f"  {mlabel} & {row['country']} & {framing} & {q_tex} & "
        f"{scale} & {out_val} & {wvs_str} & {mass_str} \\\\"
    )

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

tex_out = os.path.join(RESULTS, 'qualitative_examples.tex')
with open(tex_out, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nSaved LaTeX table: {tex_out}')

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f'\nOverall T3 refusal suppression stats:')
print(f"  Rows with >50% latent mass on WVS majority: {t3_ref['latent_agrees_50'].sum()}/{len(t3_ref)} "
      f"({t3_ref['latent_agrees_50'].mean():.1%})")
print(f"  Rows with >75% latent mass on WVS majority: {t3_ref['latent_agrees_75'].sum()}/{len(t3_ref)} "
      f"({t3_ref['latent_agrees_75'].mean():.1%})")
print(f"  Mean latent mass on WVS majority: {t3_ref['latent_wvs_mass'].mean():.3f}")
