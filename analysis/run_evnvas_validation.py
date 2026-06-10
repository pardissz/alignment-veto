"""
EV-NVAS Validation Experiment
==============================
Addresses reviewer concern: "EV-NVAS is not validated as a measure of
suppressed cultural knowledge."

Part A (runs on saved data — no re-inference needed):
  1. Argmax agreement: does argmax(norm_probs) == actual extracted answer?
  2. Pearson/Spearman correlation between EV(norm_probs) and extracted answer.
  3. Paired EV-NVAS gap (refused minus answered, same question):
     T3 gap should exceed T1 gap under suppression hypothesis.

Part B (option-order sensitivity — requires re-inference via run_evnvas_sensitivity.py):
  Loads logits under original / reversed / letter-labeled option orders and
  reports per-prompt EV-NVAS stability (fraction with >0.05 NVAS change).

Usage:
  python run_evnvas_validation.py              # Part A only
  python run_evnvas_validation.py --part-b PATH  # Part A + B (PATH = CSV from sensitivity run)
"""

import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── helpers ────────────────────────────────────────────────────────────────────

def ev_from_probs(p):
    if isinstance(p, str):
        p = json.loads(p)
    if not p:
        return np.nan
    vals = np.array([int(k) for k in p.keys()])
    probs = np.array(list(p.values()), dtype=float)
    return float(np.dot(probs, vals))


def nvas(pred, human_mean, vmin, vmax):
    return 1.0 - abs(pred - human_mean) / (vmax - vmin)


def argmax_label(p):
    if isinstance(p, str):
        p = json.loads(p)
    if not p:
        return np.nan
    return int(max(p, key=lambda k: p[k]))


# ── Part A ─────────────────────────────────────────────────────────────────────

def run_part_a(df, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Focus on instruction-tuned models (they drive refusals)
    it_keywords = ["instruct", "_it", "chat", "mini", "gpt", "aya", "allam",
                   "fanar", "jais", "tulu", "qwen"]
    it_models = [m for m in df.model.unique()
                 if any(x in m.lower() for x in it_keywords)]
    dfi = df[df.model.isin(it_models) & df.human_mean.notna()].copy()
    print(f"Instruction-tuned rows: {len(dfi)}  models: {len(it_models)}")

    # ── 1. Argmax match rate ────────────────────────────────────────────────
    ans = dfi[(dfi.refusal == False) & dfi.extracted.notna()].copy()
    ans["argmax"] = [argmax_label(r.norm_probs) for _, r in ans.iterrows()]
    ans["argmax_ok"] = ans.argmax == ans.extracted.astype(int)

    print("\n=== 1. Argmax(logit) == actual answer ===")
    rows_am = []
    for t in [1, 2, 3]:
        sub = ans[ans.tier == t]
        rate = sub.argmax_ok.mean()
        rows_am.append(dict(tier=f"T{t}", n=len(sub), match_rate=f"{rate:.3f}"))
        print(f"  T{t}: {rate:.3f}  (n={len(sub):,})")
    print(f"  Overall: {ans.argmax_ok.mean():.3f}  (n={len(ans):,})")

    # ── 2. EV correlation ──────────────────────────────────────────────────
    ans["ev"] = [ev_from_probs(r.norm_probs) for _, r in ans.iterrows()]
    # Drop NaN ev
    ans_ev = ans.dropna(subset=["ev"])
    r_p, _ = stats.pearsonr(ans_ev.ev, ans_ev.extracted)
    r_s, _ = stats.spearmanr(ans_ev.ev, ans_ev.extracted)
    mae = (ans_ev.ev - ans_ev.extracted).abs().mean()
    print(f"\n=== 2. EV correlation with actual answer ===")
    print(f"  Pearson r = {r_p:.4f}")
    print(f"  Spearman r = {r_s:.4f}")
    print(f"  Mean |EV - extracted| = {mae:.4f}")

    for t in [1, 2, 3]:
        sub = ans_ev[ans_ev.tier == t]
        rp, _ = stats.pearsonr(sub.ev, sub.extracted)
        print(f"  T{t}: Pearson r={rp:.3f}  (n={len(sub):,})")

    # ── 3. Paired gap ─────────────────────────────────────────────────────
    dfi2 = dfi.copy()
    dfi2["ev"] = [ev_from_probs(r.norm_probs) for _, r in dfi2.iterrows()]
    dfi2["ev_nvas"] = [nvas(r.ev, r.human_mean, r.vmin, r.vmax)
                       for _, r in dfi2.iterrows()]

    print("\n=== 3. Paired EV-NVAS gap: refused minus answered (same question) ===")
    rows_pg = []
    for t in [1, 2, 3]:
        sub = dfi2[dfi2.tier == t]
        sub_ans = sub[~sub.refusal].groupby("question_id").ev_nvas.mean()
        sub_ref = sub[sub.refusal].groupby("question_id").ev_nvas.mean()
        common = sub_ans.index.intersection(sub_ref.index)
        gap = (sub_ref[common] - sub_ans[common]).mean()
        t_stat, p_val = stats.ttest_rel(sub_ref[common].values, sub_ans[common].values)
        rows_pg.append(dict(tier=f"T{t}", n_questions=len(common),
                            paired_gap=f"{gap:+.4f}",
                            t=f"{t_stat:.2f}", p=f"{p_val:.4g}"))
        print(f"  T{t}: gap={gap:+.4f}  (n_questions={len(common)}, "
              f"t={t_stat:.2f}, p={p_val:.4g})")

    # ── Scatter plot: EV vs extracted ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {1: "#2196F3", 2: "#FF9800", 3: "#E91E63"}
    for t in [1, 2, 3]:
        sub = ans_ev[ans_ev.tier == t].sample(min(3000, len(ans_ev[ans_ev.tier == t])), random_state=42)
        ax.scatter(sub.extracted, sub.ev, alpha=0.08, s=4, color=colors[t], label=f"T{t}")
    lo = min(ans_ev.extracted.min(), ans_ev.ev.min())
    hi = max(ans_ev.extracted.max(), ans_ev.ev.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Actual extracted answer")
    ax.set_ylabel("EV from first-token logits")
    ax.set_title(f"Logit EV vs actual answer (Pearson r={r_p:.3f})")
    ax.legend(markerscale=3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "evnvas_validation_scatter.pdf"), bbox_inches="tight")
    plt.close()
    print(f"\nSaved scatter → {outdir}/evnvas_validation_scatter.pdf")

    # ── Paired gap bar plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    gaps = []
    for t in [1, 2, 3]:
        sub = dfi2[dfi2.tier == t]
        sub_ans = sub[~sub.refusal].groupby("question_id").ev_nvas.mean()
        sub_ref = sub[sub.refusal].groupby("question_id").ev_nvas.mean()
        common = sub_ans.index.intersection(sub_ref.index)
        gaps.append((sub_ref[common] - sub_ans[common]).mean())
    ax.bar(["T1\n(benign)", "T2\n(moderate)", "T3\n(sensitive)"],
           gaps, color=["#2196F3", "#FF9800", "#E91E63"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Paired EV-NVAS gap\n(refused − answered)")
    ax.set_title("Refused rows carry higher cultural signal\n(T3 gap > T1 gap)")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "evnvas_validation_paired_gap.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved paired-gap bar → {outdir}/evnvas_validation_paired_gap.pdf")

    return dict(r_pearson=r_p, r_spearman=r_s, mae=mae, gaps=gaps)


# ── Part B ─────────────────────────────────────────────────────────────────────

def run_part_b(csv_path, outdir):
    """
    Load sensitivity CSV produced by run_evnvas_sensitivity.py and report
    per-prompt EV-NVAS stability across option-order perturbations.

    Expected CSV columns:
      prompt_id, model, country, question_id, human_distribution,
      logits_original, logits_reversed, logits_letters, logits_numeric
    where logit_* columns contain JSON lists in the same option order.
    """
    os.makedirs(outdir, exist_ok=True)
    df_s = pd.read_csv(csv_path)
    print(f"\nLoaded sensitivity CSV: {len(df_s)} rows from {csv_path}")

    logit_cols = [c for c in df_s.columns if c.startswith("logits_")]
    print(f"Perturbation conditions: {logit_cols}")

    def softmax(x):
        x = np.asarray(x, dtype=float)
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def ev_from_list(lst):
        probs = softmax(json.loads(lst))
        n = len(probs)
        vals = np.arange(1, n + 1)
        return float(np.dot(probs, vals))

    df_s["human_mean"] = df_s["human_distribution"].apply(
        lambda x: np.dot(softmax(json.loads(x)), np.arange(1, len(json.loads(x)) + 1))
    )

    # EV-NVAS for each condition (assumes 1..n scale)
    for col in logit_cols:
        df_s[f"ev_{col}"] = df_s[col].apply(
            lambda x: ev_from_list(x) if pd.notna(x) else np.nan
        )
        n_opts = len(json.loads(df_s[col].iloc[0]))
        df_s[f"nvas_{col}"] = 1 - (df_s[f"ev_{col}"] - df_s["human_mean"]).abs() / (n_opts - 1)

    nvas_cols = [f"nvas_{c}" for c in logit_cols]
    ref_col = nvas_cols[0]
    print("\n=== Part B: EV-NVAS stability across option-order perturbations ===")
    for nc in nvas_cols[1:]:
        diff = (df_s[nc] - df_s[ref_col]).abs()
        frac_unstable = (diff > 0.05).mean()
        mean_diff = diff.mean()
        print(f"  {nc} vs {ref_col}: mean |ΔNVAS|={mean_diff:.4f}, "
              f"fraction >0.05 NVAS change={frac_unstable:.3f}")

    # Summary figure
    fig, axes = plt.subplots(1, len(nvas_cols) - 1,
                             figsize=(4 * (len(nvas_cols) - 1), 4), sharey=True)
    if len(nvas_cols) == 2:
        axes = [axes]
    for ax, nc in zip(axes, nvas_cols[1:]):
        ax.scatter(df_s[ref_col], df_s[nc], alpha=0.3, s=6)
        lo = min(df_s[ref_col].min(), df_s[nc].min())
        hi = max(df_s[ref_col].max(), df_s[nc].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("EV-NVAS (original order)")
        ax.set_ylabel(f"EV-NVAS ({nc.replace('nvas_logits_', '')})")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "evnvas_sensitivity_scatter.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved sensitivity scatter → {outdir}/evnvas_sensitivity_scatter.pdf")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-table",
                    default="/shared/storage-01/users/zahraei2/mena_normal/results/master_table.pkl")
    ap.add_argument("--outdir",
                    default="/shared/storage-01/users/zahraei2/mena_normal/results/evnvas_validation")
    ap.add_argument("--part-b", default=None,
                    help="Path to sensitivity CSV from run_evnvas_sensitivity.py")
    args = ap.parse_args()

    print("Loading master table...")
    df = pd.read_pickle(args.master_table)
    print(f"  Rows: {len(df):,}  Refusal rate: {df.refusal.mean():.3f}")

    run_part_a(df, args.outdir)

    if args.part_b:
        run_part_b(args.part_b, args.outdir)
    else:
        print("\n[Part B skipped — run run_evnvas_sensitivity.py first, then pass --part-b PATH]")

    print("\nDone. Figures saved to:", args.outdir)


if __name__ == "__main__":
    main()
