"""
Experiment 7: SAE Language-Family Dominance
Train Top-K SAE on residual-stream activations; compute country vs language-family selectivity.
Uses cached activation files from Experiment 4.
"""
import os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

def plog(msg): print(msg, flush=True)

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

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

plt.rcParams.update({'font.family':'serif','font.size':10,'axes.titlesize':11,
                     'figure.dpi':150,'savefig.dpi':200,'figure.facecolor':'white'})

# ── SAE ──────────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, k: int):
        super().__init__()
        self.k = k
        self.W_enc = nn.Linear(input_dim, latent_dim, bias=True)
        self.W_dec = nn.Linear(latent_dim, input_dim, bias=True)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.W_enc.weight)
            nn.init.zeros_(self.W_enc.bias)
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)
            nn.init.zeros_(self.W_dec.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.W_enc(x))
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        mask = torch.zeros_like(z).scatter(-1, topk_idx, 1.0)
        return z * mask

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.W_dec(z)
        recon_loss = F.mse_loss(x_hat, x)
        return recon_loss, z, x_hat


def train_sae(acts_2d: np.ndarray, latent_dim: int = 8192, k: int = 32,
              epochs: int = 300, batch_size: int = 512, lr: float = 3e-4,
              device: str = 'cuda') -> tuple:
    """Train a Top-K SAE. Returns (model, final_loss)."""
    input_dim = acts_2d.shape[1]
    X = torch.from_numpy(acts_2d.astype(np.float32)).to(device)

    # Normalise (zero-centre, unit variance)
    mu  = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True).clamp(min=1e-6)
    X_norm = (X - mu) / std

    ds  = TensorDataset(X_norm)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    sae = TopKSAE(input_dim, latent_dim, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    best_loss, patience, patience_budget = 1e9, 0, 20
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            loss, _, _ = sae(xb)
            loss.backward()
            # Re-normalise decoder columns after each step
            with torch.no_grad():
                sae.W_dec.weight.data = F.normalize(sae.W_dec.weight.data, dim=0)
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(ds)
        if ep % 50 == 0:
            plog(f'    epoch {ep}/{epochs}  loss={epoch_loss:.4f}')
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            patience  = 0
        else:
            patience += 1
            if patience >= patience_budget:
                plog(f'    early stop at epoch {ep}, loss={best_loss:.4f}')
                break

    # Get codes for full dataset
    sae.eval()
    with torch.no_grad():
        codes = sae.encode(X_norm).cpu().numpy()   # (N, F)
    return sae, best_loss, codes


# ── Selectivity ──────────────────────────────────────────────────────────────

def max_f1_selectivity(codes: np.ndarray, labels_binary: np.ndarray) -> tuple:
    """
    For each SAE feature f, compute F1 for binary label.
    Returns (max_f1, best_feature_idx).
    """
    fires  = (codes > 0).astype(np.float32)          # (N, F)
    lab    = labels_binary.astype(np.float32)         # (N,)
    tp     = (fires * lab[:, None]).sum(0)            # (F,)
    fp     = (fires * (1 - lab[:, None])).sum(0)
    fn     = ((1 - fires) * lab[:, None]).sum(0)
    prec   = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    denom  = prec + rec
    f1     = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
    best   = int(f1.argmax())
    return float(f1[best]), best


def compute_all_selectivity(codes: np.ndarray, meta: pd.DataFrame) -> dict:
    results = {}
    # Language-family labels
    for fam in LANG_FAMILY:
        lab = (meta['lang_family'] == fam).values
        mf1, bf = max_f1_selectivity(codes, lab)
        results[f'fam:{fam}'] = {'label': fam, 'type': 'lang_family',
                                  'max_f1': mf1, 'best_feature': bf,
                                  'prevalence': float(lab.mean())}
    # Country labels
    for c in COUNTRIES:
        lab = (meta['country'] == c).values
        mf1, bf = max_f1_selectivity(codes, lab)
        results[f'country:{c}'] = {'label': c, 'type': 'country',
                                    'max_f1': mf1, 'best_feature': bf,
                                    'prevalence': float(lab.mean())}
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_selectivity(all_sel: dict, model_name: str, save_path: str):
    fam_rows = [(k, v) for k, v in all_sel.items() if v['type'] == 'lang_family']
    cty_rows = [(k, v) for k, v in all_sel.items() if v['type'] == 'country']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: language-family selectivity ──
    ax = axes[0]
    fam_names = [v['label'] for _, v in fam_rows]
    fam_f1    = [v['max_f1'] for _, v in fam_rows]
    fam_cols  = [FAM_COL.get(n, '#888') for n in fam_names]
    bars = ax.barh(fam_names, fam_f1, color=fam_cols, alpha=0.85)
    ax.set_xlabel('Max-F1 selectivity'); ax.set_title('Language-Family Labels')
    ax.set_xlim(0, 1); ax.axvline(0.5, color='gray', ls='--', lw=0.8)
    for bar, f1 in zip(bars, fam_f1):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{f1:.3f}', va='center', fontsize=9)

    # ── right: country selectivity ──
    ax2 = axes[1]
    cty_names = [v['label'] for _, v in cty_rows]
    cty_f1    = [v['max_f1'] for _, v in cty_rows]
    fam_of_c  = {c: [f for f, cs in LANG_FAMILY.items() if c in cs][0]
                 for c in COUNTRIES}
    cty_cols  = [FAM_COL.get(fam_of_c.get(n, ''), '#888') for n in cty_names]
    # Sort by f1
    order = np.argsort(cty_f1)[::-1]
    cty_names = [cty_names[i] for i in order]
    cty_f1    = [cty_f1[i]    for i in order]
    cty_cols  = [cty_cols[i]  for i in order]
    ax2.barh(range(len(cty_names)), cty_f1, color=cty_cols, alpha=0.85)
    ax2.set_yticks(range(len(cty_names)))
    ax2.set_yticklabels(cty_names, fontsize=8)
    ax2.set_xlabel('Max-F1 selectivity'); ax2.set_title('Country Labels')
    ax2.set_xlim(0, 1); ax2.axvline(0.5, color='gray', ls='--', lw=0.8)
    handles = [mpatches.Patch(color=v, label=k) for k, v in FAM_COL.items()]
    ax2.legend(handles=handles, fontsize=8, loc='lower right')

    fig.suptitle(f'Exp 7: SAE Selectivity — {model_name}', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_summary(summary_df: pd.DataFrame):
    """Grouped bar: mean language-family F1 vs mean country F1 per model."""
    models = summary_df['model'].unique()
    fam_means  = []
    cty_means  = []
    for m in models:
        sub = summary_df[summary_df['model'] == m]
        fam_means.append(sub[sub['type'] == 'lang_family']['max_f1'].mean())
        cty_means.append(sub[sub['type'] == 'country']['max_f1'].mean())

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, fam_means, w, label='Language family', color='#1E88E5', alpha=0.85)
    ax.bar(x + w/2, cty_means, w, label='Country',         color='#E53935', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax.set_ylabel('Mean max-F1')
    ax.set_title('Exp 7: SAE Selectivity — Language Family vs Country')
    ax.legend(); ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', ls='--', lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'exp7_sae_selectivity.pdf'), bbox_inches='tight')
    plt.close(fig)
    plog('  Saved: exp7_sae_selectivity.pdf')


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def write_latex(summary_df: pd.DataFrame, layer_choices: dict):
    fam_tbl = (summary_df[summary_df['type'] == 'lang_family']
               .groupby(['model', 'label'])['max_f1'].mean().unstack('label').round(3))
    cty_tbl = (summary_df[summary_df['type'] == 'country']
               .groupby(['model', 'label'])['max_f1'].mean().unstack('label').round(3))

    mean_fam = fam_tbl.mean(axis=1)
    mean_cty = cty_tbl.mean(axis=1)
    combo = pd.DataFrame({'Mean Fam F1': mean_fam, 'Mean Country F1': mean_cty})
    combo['Layer'] = [layer_choices.get(m, '—') for m in combo.index]

    tex = combo.to_latex(
        caption=(
            r'Experiment~7: SAE Selectivity. '
            r'Mean max-F1 across language-family labels vs.\ country labels. '
            r'Higher language-family F1 relative to country F1 indicates '
            r'the SAE encodes language family more strongly than individual country identity.'
        ),
        label='tab:exp7',
        float_format='%.3f',
        bold_rows=False,
        column_format='lrrr',
    )
    with open(os.path.join(RESULTS, 'exp7_sae.tex'), 'w') as f:
        f.write(tex)
    plog('  Saved: exp7_sae.tex')


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exp7():
    plog('\n=== Experiment 7: SAE Language-Family Dominance ===')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plog(f'  Device: {device}')

    # SAE hyper-parameters
    LATENT_DIM = 8192
    K          = 32
    EPOCHS     = 300
    BATCH_SIZE = 512

    # Find available cached activation files
    act_files = [f for f in os.listdir(RESULTS) if f.startswith('probe_acts_') and f.endswith('.npy')]
    if not act_files:
        plog('  No cached activations found. Run Experiment 4 first.'); return

    all_sel_rows = []
    layer_choices = {}

    for act_file in sorted(act_files):
        model_name = act_file.replace('probe_acts_', '').replace('.npy', '')
        acts_path  = os.path.join(RESULTS, act_file)
        meta_path  = os.path.join(RESULTS, f'probe_meta_{model_name}.csv')
        csv_path   = os.path.join(RESULTS, f'probe_probing.csv')  # fallback

        if not os.path.exists(meta_path):
            # Try the summary CSV
            csv_path2 = os.path.join(RESULTS, 'exp4_probing.csv')
            if os.path.exists(csv_path2):
                meta_full = pd.read_csv(csv_path2)
                if model_name in meta_full['model'].values:
                    meta_sub = meta_full[meta_full['model'] == model_name].copy()
                    # We need country and lang_family per sample — use targets file
                    tgt_path = os.path.join(RESULTS, f'probe_targets_{model_name}.npy')
                    plog(f'  Using exp4_probing.csv meta for {model_name}')
                    meta_df = meta_sub[['country', 'lang_family']].drop_duplicates()
                    # Rebuild per-sample meta from targets (we don't have per-sample here)
                    # Fall back to loading the actual meta csv
                    plog(f'  WARNING: probe_meta_{model_name}.csv not found, skipping.')
                    continue
            else:
                plog(f'  Skipping {model_name}: no meta file.'); continue

        plog(f'\n--- {model_name} ---')
        acts_full = np.load(acts_path)    # (N, n_layers+1, hidden)
        meta_df   = pd.read_csv(meta_path)
        N, n_layers, hidden = acts_full.shape
        plog(f'  Activations: {acts_full.shape}')

        # Use middle layer (proxy for early-middle transformer layer)
        # Also try best_layer from probing if available
        probe_csv = os.path.join(RESULTS, 'exp4_probing.csv')
        if os.path.exists(probe_csv):
            p = pd.read_csv(probe_csv)
            pm = p[p['model'] == model_name]
            if len(pm):
                best_layer = int(pm['best_layer'].iloc[0])
            else:
                best_layer = n_layers // 2
        else:
            best_layer = n_layers // 2

        # Also run SAE on middle layer for more interesting features
        mid_layer = n_layers // 2
        plog(f'  Using layer {mid_layer} (middle) for SAE '
             f'[best probe layer was {best_layer}]')
        layer_choices[model_name] = mid_layer

        acts_2d = acts_full[:, mid_layer, :].astype(np.float32)  # (N, hidden)
        plog(f'  Training SAE (F={LATENT_DIM}, K={K})...')

        cache_codes = os.path.join(RESULTS, f'sae_codes_{model_name}.npy')
        if os.path.exists(cache_codes):
            plog(f'  Loading cached SAE codes...')
            codes = np.load(cache_codes)
        else:
            _, loss, codes = train_sae(acts_2d, LATENT_DIM, K, EPOCHS, BATCH_SIZE,
                                       device=device)
            np.save(cache_codes, codes)
            plog(f'  SAE trained. Final loss={loss:.4f}  Codes: {codes.shape}')

        # Make sure meta aligns with activation samples
        if len(meta_df) != N:
            plog(f'  Meta length mismatch ({len(meta_df)} vs {N}), truncating.')
            meta_df = meta_df.iloc[:N].reset_index(drop=True)

        plog(f'  Computing selectivity...')
        sel = compute_all_selectivity(codes, meta_df)

        for key, vals in sel.items():
            all_sel_rows.append({
                'model': model_name,
                'label': vals['label'],
                'type':  vals['type'],
                'max_f1': vals['max_f1'],
                'best_feature': vals['best_feature'],
                'prevalence': vals['prevalence'],
                'layer': mid_layer,
            })

        # Per-model plot
        save_path = os.path.join(RESULTS, f'exp7_sae_{model_name}.pdf')
        plot_selectivity(sel, model_name, save_path)
        plog(f'  Saved: exp7_sae_{model_name}.pdf')

        # Print summary
        fam_rows = {k: v for k, v in sel.items() if v['type'] == 'lang_family'}
        cty_rows = {k: v for k, v in sel.items() if v['type'] == 'country'}
        plog('  Language-family max-F1:')
        for k, v in fam_rows.items():
            plog(f'    {v["label"]:12s}: {v["max_f1"]:.3f}  (feature #{v["best_feature"]})')
        plog('  Country max-F1 (top 5):')
        cty_sorted = sorted(cty_rows.items(), key=lambda x: -x[1]['max_f1'])[:5]
        for k, v in cty_sorted:
            plog(f'    {v["label"]:15s}: {v["max_f1"]:.3f}  (feature #{v["best_feature"]})')

    if not all_sel_rows:
        plog('  No results produced.'); return

    summary_df = pd.DataFrame(all_sel_rows)
    summary_df.to_csv(os.path.join(RESULTS, 'exp7_sae.csv'), index=False)
    plog('\n  Saved: exp7_sae.csv')

    plot_summary(summary_df)
    write_latex(summary_df, layer_choices)

    plog('\nExperiment 7 done.')


if __name__ == '__main__':
    run_exp7()
