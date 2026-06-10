"""
Gap 2: Linear probing for cultural knowledge.
Trains a ridge-regression probe at each layer of OLMo SFT/DPO/IT
to predict human_mean (WVS target) from residual-stream activations.
Key claim: if probing accuracy is stable/increasing across training stages
even as output NVAS fluctuates → knowledge exists but is suppressed.
"""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

MODELS = {
    'olmo_3_7b_sft':      {'label': 'OLMo-7B-SFT',   'color': '#7986CB', 'stage': 1},
    'olmo_3_7b_dpo':      {'label': 'OLMo-7B-DPO',   'color': '#3F51B5', 'stage': 2},
    'olmo_3_7b_instruct': {'label': 'OLMo-7B-IT',    'color': '#1A237E', 'stage': 3},
    'tulu_3_8b_sft':      {'label': 'Tulu3-8B-SFT',  'color': '#F3A046', 'stage': 4},
    'tulu_3_8b_dpo':      {'label': 'Tulu3-8B-DPO',  'color': '#E67C13', 'stage': 5},
    'tulu_3_8b_instruct': {'label': 'Tulu3-8B-IT',   'color': '#C65D00', 'stage': 6},
    'llama_3.1_8b_instruct': {'label': 'LLaMA-8B-IT','color': '#2E7D32', 'stage': 7},
}

def nvas_f(row):
    if pd.isna(row['extracted']) or pd.isna(row['human_mean']): return np.nan
    d = row['vmax'] - row['vmin']
    return np.nan if d == 0 else 1 - abs(row['extracted'] - row['human_mean']) / d

mt = pd.read_pickle(os.path.join(RESULTS, 'master_table.pkl'))

# Get behavioral NVAS (accepted, Persona EN) per model
mt_acc = mt[~mt['refusal']].copy()
mt_acc['nvas'] = mt_acc.apply(nvas_f, axis=1)

behavioral = {}
for mk in MODELS:
    sub = mt_acc[(mt_acc['model']==mk) & (mt_acc['sheet']=='Personalization')]
    behavioral[mk] = {
        't1': sub[sub['tier']==1]['nvas'].mean(),
        't2': sub[sub['tier']==2]['nvas'].mean(),
        't3': sub[sub['tier']==3]['nvas'].mean(),
    }
    print(f'{MODELS[mk]["label"]}: T1={behavioral[mk]["t1"]:.3f}  '
          f'T2={behavioral[mk]["t2"]:.3f}  T3={behavioral[mk]["t3"]:.3f}', flush=True)

all_rows = []

for mk, info in MODELS.items():
    acts_path = os.path.join(RESULTS, f'mix_acts_pe_{mk}.npy')
    meta_path = os.path.join(RESULTS, f'mix_meta_{mk}.csv')
    if not os.path.exists(acts_path):
        print(f'[skip] {info["label"]} — no mix_acts', flush=True)
        continue

    print(f'\n--- {info["label"]} ---', flush=True)
    acts = np.load(acts_path, mmap_mode='r')   # (N, L, D)
    meta = pd.read_csv(meta_path)
    N    = min(len(acts), len(meta))
    acts = acts[:N]; meta = meta.iloc[:N].reset_index(drop=True)

    # Join human_mean from master_table (question_id, country → human_mean)
    mt_sub = mt[['question_id','country','human_mean','vmin','vmax']].drop_duplicates(
        subset=['question_id','country'])
    mt_sub = mt_sub.rename(columns={'vmin':'wvs_vmin','vmax':'wvs_vmax'})
    meta2  = meta.merge(mt_sub, left_on=['qid','country'], right_on=['question_id','country'], how='left')
    # Normalize human_mean to [0,1] range
    hm = meta2['human_mean'].values.astype(np.float32)
    vm = meta2['wvs_vmin'].values.astype(np.float32)
    vx = meta2['wvs_vmax'].values.astype(np.float32)
    denom = vx - vm
    with np.errstate(invalid='ignore', divide='ignore'):
        y_norm = np.where(denom > 0, (hm - vm) / denom, np.nan).astype(np.float32)

    tiers  = meta['tier'].values   # tier comes from mix_meta (already there)
    valid  = ~np.isnan(hm) & ~np.isnan(y_norm)

    N_L = acts.shape[1]
    print(f'  Activations: {acts.shape}  valid targets: {valid.sum()}', flush=True)

    # Sample layers for efficiency: every 2nd layer
    layer_sample = list(range(0, N_L, max(1, N_L//20)))
    if N_L-1 not in layer_sample: layer_sample.append(N_L-1)

    for layer in layer_sample:
        X = acts[:N, layer, :][valid].astype(np.float32)
        y = y_norm[valid]   # predict normalized position on scale
        t = tiers[valid]

        if X.shape[0] < 30: continue

        # Scale
        sc = StandardScaler()
        Xs = sc.fit_transform(X)

        # Full probe
        probe = Ridge(alpha=1.0)
        cv_r2 = cross_val_score(probe, Xs, y, cv=5, scoring='r2')
        r2_all = float(cv_r2.mean())

        # T3-only probe
        t3_mask = t == 3
        if t3_mask.sum() >= 10:
            cv_r2_t3 = cross_val_score(probe, Xs[t3_mask], y[t3_mask], cv=min(5,t3_mask.sum()//2), scoring='r2')
            r2_t3 = float(cv_r2_t3.mean())
        else:
            r2_t3 = np.nan

        # T2-only probe
        t2_mask = t == 2
        if t2_mask.sum() >= 20:
            cv_r2_t2 = cross_val_score(probe, Xs[t2_mask], y[t2_mask], cv=5, scoring='r2')
            r2_t2 = float(cv_r2_t2.mean())
        else:
            r2_t2 = np.nan

        all_rows.append({
            'model': mk, 'label': info['label'], 'stage': info['stage'],
            'color': info['color'], 'layer': layer,
            'r2_all': r2_all, 'r2_t2': r2_t2, 'r2_t3': r2_t3,
            'beh_t3': behavioral[mk]['t3'], 'beh_t2': behavioral[mk]['t2'],
        })

    print(f'  Best layer all: {max(all_rows[-len(layer_sample):], key=lambda x: x["r2_all"])["layer"]}  '
          f'R²={max(all_rows[-len(layer_sample):], key=lambda x: x["r2_all"])["r2_all"]:.3f}', flush=True)

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(RESULTS, 'gap2_probing_results.csv'), index=False)
print('\nSaved: gap2_probing_results.csv')

# ── Figure 1: Probing R² vs layer, colored by model/stage ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')

for ax_i, (metric, title) in enumerate([
    ('r2_all', 'All tiers'),
    ('r2_t2',  'T2 (moderate) only'),
    ('r2_t3',  'T3 (sensitive) only'),
]):
    ax = axes[ax_i]; ax.set_facecolor('#FAFAFA')
    for mk, info in MODELS.items():
        sub = df[df['model']==mk].sort_values('layer')
        if len(sub) == 0: continue
        y_vals = sub[metric].values
        valid_l = ~np.isnan(y_vals)
        if valid_l.sum() < 2: continue
        ax.plot(sub['layer'].values[valid_l], y_vals[valid_l],
                color=info['color'], lw=1.8, label=info['label'], marker='o', ms=3, alpha=0.9)
    ax.set_xlabel('Layer', fontsize=9.5)
    ax.set_ylabel('Probe R² (CV)', fontsize=9.5)
    ax.set_title(f'{title}', fontsize=10)
    ax.set_ylim(-0.1, 0.7)
    ax.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
    ax.grid(lw=0.3, alpha=0.4, color='#ccc')
    if ax_i == 0: ax.legend(fontsize=7.5, framealpha=0.9)

fig.suptitle('Layer-wise Probing: Can we decode WVS human values from residual-stream activations?\n'
             'Positive R² = cultural knowledge is geometrically accessible even when output is suppressed',
             fontsize=10.5, y=1.01)
fig.tight_layout()
out1 = os.path.join(RESULTS, 'gap2_probing_layers.pdf')
fig.savefig(out1, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'Saved: {out1}')

# ── Figure 2: Best-layer R² vs behavioral NVAS (knowledge vs output) ──────────
best_layer_df = df.loc[df.groupby('model')['r2_all'].idxmax()]
best_t3_df    = df.loc[df.groupby('model')['r2_t3'].idxmax()]

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
fig2.patch.set_facecolor('white')

for ax_i, (bd, title_str, xlab) in enumerate([
    (best_layer_df, 'All-tier probing R² vs behavioral NVAS',  'Best-layer probe R² (all tiers)'),
    (best_t3_df,    'T3-only probing R² vs T3 behavioral NVAS', 'Best-layer probe R² (T3 only)'),
]):
    ax = axes2[ax_i]; ax.set_facecolor('#FAFAFA')
    beh_col = 'beh_t2' if ax_i == 0 else 'beh_t3'
    r2_col  = 'r2_all' if ax_i == 0 else 'r2_t3'
    for _, row in bd.iterrows():
        if np.isnan(row[r2_col]): continue
        col = MODELS[row['model']]['color']
        ax.scatter([row[r2_col]], [row[beh_col]], s=80, color=col, zorder=5,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(row['label'], (row[r2_col], row[beh_col]),
                    fontsize=7, xytext=(3,2), textcoords='offset points', color='#333')
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel('Behavioral NVAS (output)', fontsize=9)
    ax.set_title(title_str, fontsize=9.5)
    ax.grid(lw=0.3, alpha=0.4, color='#ccc')

# Key insight annotation
fig2.text(0.5, -0.02,
          'High probe R² with lower behavioral NVAS = cultural knowledge present in representations but suppressed in output',
          ha='center', fontsize=9, color='#555', style='italic')

fig2.suptitle('Cultural Knowledge in Representations vs. Output: Gating Not Erasing\n'
              'Probe R² measures what the model "knows"; behavioral NVAS measures what it "says"',
              fontsize=10.5, y=1.02)
fig2.tight_layout()
out2 = os.path.join(RESULTS, 'gap2_probing_vs_nvas.pdf')
fig2.savefig(out2, bbox_inches='tight', dpi=200); plt.close(fig2)
print(f'Saved: {out2}')

# Print summary
print('\n=== Probing Summary ===')
print(best_layer_df[['label','layer','r2_all','r2_t3','beh_t3']].round(3).to_string())
print('\nDone.')
