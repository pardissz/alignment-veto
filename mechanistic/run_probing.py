"""
Experiment 4: Linear Probing on Residual Stream Activations
Trains linear probes to predict human distribution from LLM hidden states.
Tests country generalization and per-language breakdown.
"""

import os, json, warnings, ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
warnings.filterwarnings('ignore')

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

COUNTRIES = ['Algeria','Egypt','Iran','Iraq','Jordan','Kuwait','Lebanon',
             'Libya','Mauritania','Morocco','Palestine','Qatar',
             'Saudi Arabia','Sudan','Tunisia','Turkey']

LANG_FAMILY = {
    'Arabic': ['Algeria','Egypt','Iraq','Jordan','Kuwait','Lebanon',
               'Libya','Mauritania','Morocco','Palestine','Qatar',
               'Saudi Arabia','Sudan','Tunisia'],
    'Persian': ['Iran'],
    'Turkish': ['Turkey'],
}

# Models to probe (pick small ones that fit in GPU memory)
PROBE_MODELS = {
    'olmo_3_7b_base':     'allenai/OLMo-2-1124-7B',
    'olmo_3_7b_instruct': 'allenai/OLMo-2-1124-7B-Instruct',
    'llama_3.1_8b_instruct': 'meta-llama/Llama-3.1-8B-Instruct',
}

def load_human_data():
    df = pd.read_excel(os.path.join(BASE, 'new_weights_transposed.xlsx'))
    df = df.rename(columns={'question_number':'question_id'})
    for c in COUNTRIES:
        if c not in df.columns: continue
        parsed = []
        for v in df[c]:
            if pd.isna(v):
                parsed.append((np.nan, {}))
                continue
            try:
                tup = ast.literal_eval(str(v))
                mean_val = float(tup[0])
                dist = {k: float(str(vv).strip('%'))/100 for k,vv in tup[1].items()}
                parsed.append((mean_val, dist))
            except:
                parsed.append((np.nan, {}))
        df[f'{c}_mean'] = [p[0] for p in parsed]
    return df.set_index('question_id')

def load_tiers():
    t = pd.read_csv(os.path.join(BASE, 'tiers.csv'))
    return t

def safe_parse_probs(x):
    if pd.isna(x): return {}
    try: return json.loads(str(x).replace("'",'"'))
    except:
        try: return ast.literal_eval(str(x))
        except: return {}

def extract_activations_batch(model, tokenizer, prompts, device, batch_size=8, max_new=1):
    """Extract residual stream activations at the last token for a batch of prompts."""
    all_hidden = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors='pt', padding=True,
                            truncation=True, max_length=512).to(device)
            out = model(**enc, output_hidden_states=True)
            # hidden_states: (n_layers+1) × (batch × seq_len × hidden_dim)
            # Take last token position, all layers
            batch_hidden = torch.stack(
                [hs[:, -1, :] for hs in out.hidden_states], dim=1
            ).float().cpu().numpy()  # (batch, n_layers+1, hidden_dim)
            all_hidden.append(batch_hidden)
    return np.concatenate(all_hidden, axis=0)  # (N, n_layers+1, hidden_dim)

def build_probe_dataset(model_name, hf_name, human_df, tiers_df):
    """
    For each (country, question) pair, get:
    - Activation at last token of the prompt (all layers)
    - Human mean value as regression target
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count()
    print(f'  Loading {hf_name} on {device} ({n_gpu} GPUs)...')

    tokenizer = AutoTokenizer.from_pretrained(hf_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model.eval()

    # Build prompts using Observer framing (English) for all (country, question)
    tiers_df_idx = tiers_df.set_index('question_id')
    prompts, targets, meta = [], [], []

    xl = pd.ExcelFile(os.path.join(BASE, 'MENA_TRANSLATED_reasoning', f'{model_name}.xlsx'))
    third_df = pd.read_excel(xl, sheet_name='Third')

    for _, row in third_df.iterrows():
        qid = int(row['question_id'])
        if qid not in tiers_df_idx.index: continue
        q_text = tiers_df_idx.loc[qid, 'No Mention']  # the question text
        vmin = float(tiers_df_idx.loc[qid, 'Min'])
        vmax = float(tiers_df_idx.loc[qid, 'MAX'])
        tier = int(tiers_df_idx.loc[qid, 'Tier'])

        for country in COUNTRIES:
            lang = [f for f,cs in LANG_FAMILY.items() if country in cs]
            lang = lang[0] if lang else 'Other'

            if qid not in human_df.index: continue
            h_mean = human_df.loc[qid, f'{country}_mean']
            if pd.isna(h_mean): continue

            # Observer prompt
            prompt = (f"You are observing people from {country}. "
                      f"Based on your knowledge of {country}'s culture and values, "
                      f"please answer the following survey question as they typically would:\n"
                      f"{q_text}")
            prompts.append(prompt)
            targets.append(h_mean)
            meta.append({
                'qid': qid, 'country': country, 'lang_family': lang,
                'vmin': vmin, 'vmax': vmax, 'tier': tier
            })

    print(f'  Total prompts: {len(prompts)}')

    # Extract activations in batches
    print(f'  Extracting activations...')
    activations = extract_activations_batch(model, tokenizer, prompts, device, batch_size=16)
    # activations: (N, n_layers, hidden_dim)
    print(f'  Activations shape: {activations.shape}')

    meta_df = pd.DataFrame(meta)
    return activations, np.array(targets), meta_df

def train_probes(activations, targets, meta_df):
    """
    Train linear probe per layer. Pick best layer on validation.
    Country generalization: LOGO cross-validation (leave-one-country-out).
    """
    n_samples, n_layers, hidden_dim = activations.shape
    countries = meta_df['country'].values
    lang_fam = meta_df['lang_family'].values
    tiers = meta_df['tier'].values

    results = []
    best_layer_scores = {}

    # Layer selection: validate on a held-out country
    val_countries = COUNTRIES[:4]  # hold out 4 countries for layer selection
    train_mask = ~meta_df['country'].isin(val_countries).values
    val_mask   =  meta_df['country'].isin(val_countries).values

    layer_scores = []
    for layer in range(n_layers):
        X = activations[:, layer, :]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_val   = scaler.transform(X[val_mask])
        y_train = targets[train_mask]
        y_val   = targets[val_mask]
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        score = r2_score(y_val, probe.predict(X_val))
        layer_scores.append(score)

    best_layer = int(np.argmax(layer_scores))
    print(f'  Best layer: {best_layer} (val R²={layer_scores[best_layer]:.3f})')

    # Country generalization: LOGO
    logo = LeaveOneGroupOut()
    country_scores = []
    for train_idx, test_idx in logo.split(np.arange(n_samples), groups=countries):
        X = activations[:, best_layer, :]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test  = scaler.transform(X[test_idx])
        y_train = targets[train_idx]
        y_test  = targets[test_idx]
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        score = r2_score(y_test, probe.predict(X_test))
        test_country = countries[test_idx[0]]
        country_scores.append({'country': test_country, 'r2': score,
                                'lang_family': lang_fam[test_idx[0]]})
    country_df = pd.DataFrame(country_scores)

    # Per-language breakdown
    lang_df = country_df.groupby('lang_family')['r2'].agg(['mean','std']).reset_index()

    # Dataset-prior baseline: predict mean of training targets
    logo_scores_baseline = []
    for train_idx, test_idx in logo.split(np.arange(n_samples), groups=countries):
        baseline = np.mean(targets[train_idx])
        preds = np.full(len(test_idx), baseline)
        logo_scores_baseline.append(r2_score(targets[test_idx], preds))

    # Prompting baseline: use the model's own Third-sheet extracted answers
    return {
        'best_layer': best_layer,
        'layer_scores': layer_scores,
        'country_generalization': country_df,
        'lang_breakdown': lang_df,
        'baseline_r2': np.mean(logo_scores_baseline),
    }

def run_experiment_4():
    print('=== Experiment 4: Linear Probing ===')
    human_df = load_human_data()
    tiers_df = load_tiers()

    all_results = {}
    for model_name, hf_name in PROBE_MODELS.items():
        print(f'\n--- Probing: {model_name} ---')
        try:
            activations, targets, meta_df = build_probe_dataset(
                model_name, hf_name, human_df, tiers_df)
            np.save(os.path.join(RESULTS, f'probe_acts_{model_name}.npy'), activations)
            np.save(os.path.join(RESULTS, f'probe_targets_{model_name}.npy'), targets)
            meta_df.to_csv(os.path.join(RESULTS, f'probe_meta_{model_name}.csv'), index=False)

            result = train_probes(activations, targets, meta_df)
            all_results[model_name] = result

            # Plot layer scores
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ax = axes[0]
            ax.plot(result['layer_scores'], color='steelblue', lw=2)
            ax.axvline(result['best_layer'], color='red', ls='--', label=f"Best: L{result['best_layer']}")
            ax.axhline(result['baseline_r2'], color='gray', ls=':', label=f"Dataset prior baseline R²={result['baseline_r2']:.3f}")
            ax.set_xlabel('Layer')
            ax.set_ylabel('R² (val set)')
            ax.set_title(f'Probe Layer Selection: {model_name}')
            ax.legend()

            ax2 = axes[1]
            cdf = result['country_generalization'].sort_values('r2', ascending=False)
            cols = {'Arabic':'#E53935','Persian':'#1E88E5','Turkish':'#43A047'}
            bar_colors = [cols.get(row['lang_family'],'gray') for _,row in cdf.iterrows()]
            ax2.barh(range(len(cdf)), cdf['r2'].values, color=bar_colors, alpha=0.85)
            ax2.set_yticks(range(len(cdf)))
            ax2.set_yticklabels(cdf['country'].values, fontsize=8)
            ax2.axvline(0, color='black', lw=0.8)
            ax2.axvline(result['baseline_r2'], color='gray', ls=':', label='Dataset prior')
            ax2.set_xlabel('R² (leave-one-country-out)')
            ax2.set_title('Country Generalization')
            ax2.legend()

            handles = [mpatches.Patch(color='#E53935',label='Arabic'),
                       mpatches.Patch(color='#1E88E5',label='Persian'),
                       mpatches.Patch(color='#43A047',label='Turkish')]
            ax2.legend(handles=handles + [plt.Line2D([],[],color='gray',ls=':',label='Dataset prior')],
                       fontsize=7)
            fig.suptitle(f'Experiment 4: Linear Probing — {model_name}', fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(RESULTS, f'exp4_probe_{model_name}.pdf'), bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: exp4_probe_{model_name}.pdf')

        except Exception as e:
            print(f'  ERROR for {model_name}: {e}')
            import traceback; traceback.print_exc()
            continue

    # Summary table
    if all_results:
        rows = []
        for model_name, res in all_results.items():
            for _, row in res['country_generalization'].iterrows():
                rows.append({
                    'model': model_name,
                    'country': row['country'],
                    'lang_family': row['lang_family'],
                    'r2_logo': row['r2'],
                    'best_layer': res['best_layer'],
                    'baseline_r2': res['baseline_r2'],
                })
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(RESULTS, 'exp4_probing_results.csv'), index=False)

        # LaTeX table: per-language breakdown
        lang_table = summary.groupby(['model','lang_family'])['r2_logo'].agg(['mean','std']).round(3)
        print('\nPer-language R² (LOGO):')
        print(lang_table)
        lang_table.to_csv(os.path.join(RESULTS, 'exp4_lang_breakdown.csv'))

        tex = r"""\begin{table}[htbp]
\centering
\caption{Experiment 4: Linear probe R$^2$ (leave-one-country-out) by model and language family.
  Higher R$^2$ = probe generalizes better to held-out countries.
  Baseline = dataset-prior (mean of training targets).}
\label{tab:exp4_probing}
""" + lang_table.to_latex(escape=True) + r"\end{table}"
        with open(os.path.join(RESULTS, 'exp4_probing.tex'), 'w') as f:
            f.write(tex)
        print('  Saved: exp4_probing.tex')

    print('Experiment 4 done.')

if __name__ == '__main__':
    run_experiment_4()
