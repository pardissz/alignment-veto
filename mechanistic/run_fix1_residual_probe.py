"""
Fix 1: Residualized probing — controls for question-level confound.
Standard probe predicts human_mean (which is 66.6% explained by question_id alone).
Residualized probe predicts country-specific deviation from the per-question mean.
R²_residual > 0 confirms genuine country-specific cultural encoding, not just question recall.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

MODELS = {
    'olmo_3_7b_sft':      {'label': 'OLMo-7B-SFT',  'color': '#7986CB', 'stage': 1},
    'olmo_3_7b_dpo':      {'label': 'OLMo-7B-DPO',  'color': '#3F51B5', 'stage': 2},
    'olmo_3_7b_instruct': {'label': 'OLMo-7B-IT',   'color': '#1A237E', 'stage': 3},
    'tulu_3_8b_sft':      {'label': 'Tulu3-8B-SFT', 'color': '#F3A046', 'stage': 4},
    'tulu_3_8b_dpo':      {'label': 'Tulu3-8B-DPO', 'color': '#E67C13', 'stage': 5},
    'tulu_3_8b_instruct': {'label': 'Tulu3-8B-IT',  'color': '#C65D00', 'stage': 6},
    'llama_3.1_8b_instruct': {'label': 'LLaMA-8B-IT', 'color': '#2E7D32', 'stage': 7},
}

mt = pd.read_pickle(os.path.join(RESULTS, 'master_table.pkl'))

# ── Compute baseline: what does question_id alone explain? ───────────────────
from sklearn.preprocessing import OneHotEncoder
sub_unique = mt[['question_id','country','human_mean','vmin','vmax']].dropna().drop_duplicates(
    subset=['question_id','country'])
sub_unique['y_norm'] = ((sub_unique['human_mean'] - sub_unique['vmin']) /
                        (sub_unique['vmax'] - sub_unique['vmin']).clip(lower=1e-6))
sub_unique = sub_unique.dropna(subset=['y_norm'])

enc_q = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Xq = enc_q.fit_transform(sub_unique[['question_id']])
y_all = sub_unique['y_norm'].values
r2_question_baseline = cross_val_score(Ridge(alpha=1.0), Xq, y_all, cv=5, scoring='r2').mean()
print(f'Question-only baseline R²: {r2_question_baseline:.4f}')

# ── Per-question mean for residualization ────────────────────────────────────
q_means = sub_unique.groupby('question_id')['y_norm'].mean().to_dict()
print(f'Computed per-question means for {len(q_means)} questions')

# ── Behavioral NVAS residual (how much better is model output vs question mean?) ─
def nvas_f(row):
    if pd.isna(row['extracted']) or pd.isna(row['human_mean']): return np.nan
    d = row['vmax'] - row['vmin']
    return np.nan if d == 0 else 1 - abs(row['extracted'] - row['human_mean']) / d

mt['nvas'] = mt.apply(nvas_f, axis=1)

# ── Main probing loop ─────────────────────────────────────────────────────────
all_rows = []

for mk, info in MODELS.items():
    acts_path = os.path.join(RESULTS, f'mix_acts_pe_{mk}.npy')
    meta_path = os.path.join(RESULTS, f'mix_meta_{mk}.csv')
    if not os.path.exists(acts_path):
        print(f'[skip] {info["label"]} — no activations')
        continue

    print(f'\n--- {info["label"]} ---')
    acts = np.load(acts_path, mmap_mode='r')
    meta = pd.read_csv(meta_path)
    N = min(len(acts), len(meta))
    acts = acts[:N]; meta = meta.iloc[:N].reset_index(drop=True)

    # Join human_mean
    mt_sub = mt[['question_id','country','human_mean','vmin','vmax']].drop_duplicates(
        subset=['question_id','country']).rename(columns={'vmin':'wvs_vmin','vmax':'wvs_vmax'})
    meta2 = meta.merge(mt_sub, left_on=['qid','country'], right_on=['question_id','country'], how='left')

    hm = meta2['human_mean'].values.astype(np.float32)
    vm = meta2['wvs_vmin'].values.astype(np.float32)
    vx = meta2['wvs_vmax'].values.astype(np.float32)
    denom = vx - vm
    with np.errstate(invalid='ignore', divide='ignore'):
        y_total = np.where(denom > 0, (hm - vm) / denom, np.nan).astype(np.float32)

    # Residualized target: subtract per-question mean
    qids = meta2['question_id'].values
    q_mean_arr = np.array([q_means.get(q, np.nan) for q in qids], dtype=np.float32)
    y_resid = (y_total - q_mean_arr).astype(np.float32)

    valid_total = ~np.isnan(y_total)
    valid_resid = ~np.isnan(y_resid)
    tiers = meta['tier'].values

    N_L = acts.shape[1]
    layer_sample = list(range(0, N_L, max(1, N_L // 20)))
    if N_L - 1 not in layer_sample: layer_sample.append(N_L - 1)

    model_rows = []
    for layer in layer_sample:
        X = acts[:N, layer, :]

        # Total probe (original)
        mask_t = valid_total
        if mask_t.sum() >= 30:
            Xs = StandardScaler().fit_transform(X[mask_t].astype(np.float32))
            r2_tot = float(cross_val_score(Ridge(alpha=1.0), Xs, y_total[mask_t], cv=5, scoring='r2').mean())
        else:
            r2_tot = np.nan

        # T3 total probe
        mask_t3 = valid_total & (tiers == 3)
        if mask_t3.sum() >= 10:
            Xs3 = StandardScaler().fit_transform(X[mask_t3].astype(np.float32))
            r2_t3_tot = float(cross_val_score(Ridge(alpha=1.0), Xs3, y_total[mask_t3],
                                               cv=min(5, mask_t3.sum() // 2), scoring='r2').mean())
        else:
            r2_t3_tot = np.nan

        # Residualized probe
        mask_r = valid_resid
        if mask_r.sum() >= 30:
            Xsr = StandardScaler().fit_transform(X[mask_r].astype(np.float32))
            r2_res = float(cross_val_score(Ridge(alpha=1.0), Xsr, y_resid[mask_r], cv=5, scoring='r2').mean())
        else:
            r2_res = np.nan

        # T3 residualized probe
        mask_r3 = valid_resid & (tiers == 3)
        if mask_r3.sum() >= 10:
            Xsr3 = StandardScaler().fit_transform(X[mask_r3].astype(np.float32))
            r2_t3_res = float(cross_val_score(Ridge(alpha=1.0), Xsr3, y_resid[mask_r3],
                                               cv=min(5, mask_r3.sum() // 2), scoring='r2').mean())
        else:
            r2_t3_res = np.nan

        model_rows.append({
            'model': mk, 'label': info['label'], 'stage': info['stage'],
            'color': info['color'], 'layer': layer,
            'r2_total': r2_tot, 'r2_t3_total': r2_t3_tot,
            'r2_resid': r2_res, 'r2_t3_resid': r2_t3_res,
        })
        all_rows.extend([model_rows[-1]])

    best = max(model_rows, key=lambda x: x['r2_total'] if not np.isnan(x['r2_total']) else -1)
    best3 = max(model_rows, key=lambda x: x['r2_t3_resid'] if not np.isnan(x['r2_t3_resid']) else -1)
    print(f'  Best total R²={best["r2_total"]:.3f} @ layer {best["layer"]}')
    print(f'  Best T3 residualized R²={best3["r2_t3_resid"]:.3f} @ layer {best3["layer"]}')

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(RESULTS, 'fix1_residual_probe_results.csv'), index=False)

# ── Figure ───────────────────────────────────────────────────────────────────
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(13, 9))
fig.patch.set_facecolor('white')
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
ax_tl = fig.add_subplot(gs[0, 0])
ax_tr = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, :])

top_panels = [
    (ax_tl, 'r2_total',    'Standard probe (all tiers)',       True),
    (ax_tr, 'r2_t3_total', 'Standard T3 probe',                True),
]

for ax, metric, title, show_baseline in top_panels:
    ax.set_facecolor('#FAFAFA')
    for mk, info in MODELS.items():
        sub = df[df['model'] == mk].sort_values('layer')
        if len(sub) == 0: continue
        y_vals = sub[metric].values
        valid_l = ~np.isnan(y_vals)
        if valid_l.sum() < 2: continue
        ax.plot(sub['layer'].values[valid_l], y_vals[valid_l],
                color=info['color'], lw=1.8, label=info['label'],
                marker='o', ms=3, alpha=0.9)
    ax.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
    if show_baseline:
        ax.axhline(r2_question_baseline, color='#E53935', lw=1.2, ls=':', alpha=0.7,
                   label=f'Q-only baseline ({r2_question_baseline:.3f})')
    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('Probe R²', fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(-0.15, 0.85)
    ax.grid(lw=0.3, alpha=0.4, color='#ccc')

ax_tl.legend(fontsize=6.5, framealpha=0.9, ncol=2)
ax_tr.legend(fontsize=6.5, framealpha=0.9, loc='upper left',
             handles=[plt.Line2D([],[],color='#E53935',ls=':',lw=1.2,
                                  label=f'Q-only baseline ({r2_question_baseline:.3f})')])

# Bottom panel: residualized T3 probe (full width)
ax_bot.set_facecolor('#FAFAFA')
for mk, info in MODELS.items():
    sub = df[df['model'] == mk].sort_values('layer')
    if len(sub) == 0: continue
    y_vals = sub['r2_t3_resid'].values
    valid_l = ~np.isnan(y_vals)
    if valid_l.sum() < 2: continue
    ax_bot.plot(sub['layer'].values[valid_l], y_vals[valid_l],
                color=info['color'], lw=1.8, label=info['label'],
                marker='o', ms=3, alpha=0.9)
ax_bot.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
ax_bot.axvspan(28, 32, alpha=0.12, color='#E53935', label='Erasure zone (layers 28–32)')
ax_bot.text(29.0, 0.06, 'erasure\nzone', fontsize=7.5, color='#E53935',
            ha='center', va='bottom', alpha=0.85)
ax_bot.set_xlabel('Layer', fontsize=9)
ax_bot.set_ylabel('Probe R²', fontsize=9)
ax_bot.set_title('Residualized T3 probe — pure country-specific cultural encoding\n'
                 '(per-question mean subtracted; erasure zone = layers 28–32)', fontsize=9)
ax_bot.set_ylim(-0.15, 0.85)
ax_bot.grid(lw=0.3, alpha=0.4, color='#ccc')
ax_bot.legend(fontsize=6.5, framealpha=0.9, ncol=4)

fig.suptitle(
    'Probing Confound Control: Standard vs. Residualized\n'
    f'Standard probe R² includes question-level baseline ({r2_question_baseline:.3f}); '
    'residualized probe isolates country-specific cultural signal',
    fontsize=10.5)
fig.tight_layout()
out = os.path.join(RESULTS, 'fix1_residual_probe.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'\nSaved: {out}')

# ── Summary table ─────────────────────────────────────────────────────────────
best_df = df.loc[df.groupby('model')['r2_t3_resid'].idxmax()]
print('\n=== Residualized Probe Summary (best layer per model) ===')
print(f'Question-only baseline R²: {r2_question_baseline:.4f}')
print(best_df[['label','layer','r2_total','r2_t3_total','r2_resid','r2_t3_resid']].round(3).to_string())
print('\nDone.')
