"""
EV-NVAS Option-Order Sensitivity Test
======================================
For all 29 T3 questions, runs inference under three prompt conditions:
  original  — the exact question text used in the experiment
  reversed  — same question but scale endpoints swapped (1↔N in labels)
  letter    — digit labels replaced with A/B/C/… labels

Extracts first-token forced-choice logits (reusing a.py logic).
Computes EV-NVAS stability: |NVAS_original - NVAS_condition| per prompt.

Saves results/evnvas_sensitivity.csv for run_evnvas_validation.py --part-b.

Usage:
  python3 run_evnvas_sensitivity.py
  python3 run_evnvas_sensitivity.py --model allenai/Llama-3.1-Tulu-3-8B-DPO
"""

import argparse, json, os, re, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers

# ── prompt construction ─────────────────────────────────────────────────────

def _parse_scale_anchors(question_text: str):
    """
    Return (lo_digit, lo_label, hi_digit, hi_label, match_obj, fmt)
    where fmt is 'A' (quoted endpoints) or 'B' (colon-list).
    Returns None if no recognised pattern found.
    """
    # Format A-and: "with '1' being 'X' and 'N' being 'Y'"
    # Format A-comma: "With '1' being 'X', '2' being 'Y'"  (binary)
    pat_a = re.compile(
        r"[Ww]ith '(\d+)' being '([^']+)'(?:,|,? and) '(\d+)' being '([^']+)'"
    )
    m = pat_a.search(question_text)
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4), m, "A"

    # Format B: "1: X, 2: Y, 3: Z, 4: W"
    pat_b = re.compile(r"(\d+):\s*([A-Za-z][^,\.]*)")
    matches = list(pat_b.finditer(question_text))
    if len(matches) >= 2:
        return (matches[0].group(1), matches[0].group(2).strip(),
                matches[-1].group(1), matches[-1].group(2).strip(),
                matches, "B")

    return None


def make_reversed_prompt(question_text: str, vmin: int, vmax: int) -> str:
    """
    Swap endpoint label descriptions so the semantic meaning of vmin and vmax
    are exchanged. This tests whether model outputs are driven by digit-token
    priors or by the semantic content of the question.

    Handles the three formats observed in the MENA question set:
      Format A-and:   "with '1' being 'X' and '4' being 'Y'"
      Format A-comma: "With '1' being 'No', '2' being 'Yes'"  (binary)
      Format B:       "1: Strongly agree, 2: Agree, 3: Disagree, 4: Strongly disagree"
    """
    parsed = _parse_scale_anchors(question_text)
    if parsed is None:
        return question_text  # fallback: unchanged

    lo_digit, lo_label, hi_digit, hi_label, match_obj, fmt = parsed

    if fmt == "A":
        # Swap lo_label ↔ hi_label, keeping digits in place
        original_chunk = match_obj.group(0)
        # Reconstruct with same connector as original
        connector = ", " if ", '" + hi_digit in original_chunk else " and "
        new_chunk = (f"with '{lo_digit}' being '{hi_label}'"
                     + connector
                     + f"'{hi_digit}' being '{lo_label}'")
        # Preserve original case of "with"
        if original_chunk[0] == "W":
            new_chunk = "W" + new_chunk[1:]
        return (question_text[: match_obj.start()]
                + new_chunk
                + question_text[match_obj.end():])

    else:  # Format B — reverse ALL labels
        matches = match_obj  # list of re.Match
        all_labels = [m.group(2).strip() for m in matches]
        rev_labels = all_labels[::-1]
        result = question_text
        for m, rl in zip(reversed(matches), reversed(rev_labels)):
            # Replace in reverse order to preserve offsets
            orig = m.group(0)
            new  = f"{m.group(1)}: {rl}"
            result = result[: m.start()] + new + result[m.start() + len(orig):]
        return result


def make_letter_prompt(question_text: str, vmin: int, vmax: int) -> str:
    """
    Replace numeric labels with letter labels (1→A, 2→B, …) keeping semantic
    order identical to original. Changes suffix to 'Answer with the letter only'.
    """
    n = vmax - vmin + 1
    letters = [chr(ord("A") + i) for i in range(n)]
    parsed = _parse_scale_anchors(question_text)

    result = question_text

    if parsed is not None:
        lo_digit, lo_label, hi_digit, hi_label, match_obj, fmt = parsed
        if fmt == "A":
            lo_letter = letters[int(lo_digit) - vmin]
            hi_letter = letters[int(hi_digit) - vmin]
            original_chunk = match_obj.group(0)
            connector = ", " if ", '" + hi_digit in original_chunk else " and "
            new_chunk = (f"with '{lo_letter}' being '{lo_label}'"
                         + connector
                         + f"'{hi_letter}' being '{hi_label}'")
            if original_chunk[0] == "W":
                new_chunk = "W" + new_chunk[1:]
            result = (question_text[: match_obj.start()]
                      + new_chunk
                      + question_text[match_obj.end():])
        else:  # Format B
            matches = match_obj
            for m in reversed(matches):
                d = int(m.group(1))
                letter = letters[d - vmin]
                orig = m.group(0)
                new  = f"{letter}: {m.group(2).strip()}"
                result = result[: m.start()] + new + result[m.start() + len(orig):]

    result = result.replace("Provide the answer number only", "Answer with the letter only")
    result = result.replace("Provide the answer number", "Answer with the letter")
    return result


# ── logit extraction ────────────────────────────────────────────────────────

def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_option_token_ids(tokenizer, labels: list[str]) -> dict[str, int]:
    """Return {label: best_token_id} using the same logic as a.py's
    get_single_token_options (direct encode + space-prefixed + vocab scan)."""
    out = {}
    seen = set()
    for lab in labels:
        candidates = set()
        for s in [lab, " " + lab]:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                candidates.add(ids[0])
        for tid in range(min(1000, tokenizer.vocab_size)):
            try:
                if tokenizer.decode([tid]).strip() == lab:
                    candidates.add(tid)
            except Exception:
                pass
        candidates -= seen
        seen |= candidates
        out[lab] = sorted(candidates)[0] if candidates else None
    return out


@torch.no_grad()
def forced_choice_probs(model, tokenizer, prompt: str,
                        labels: list[str], device: str) -> dict[str, float] | None:
    """Run one forward pass; return softmax-renormalised P(label) dict."""
    formatted = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits[0, -1, :].float()  # last token position

    tok_ids = get_option_token_ids(tokenizer, labels)
    missing = [l for l, tid in tok_ids.items() if tid is None]
    if missing:
        print(f"  WARNING: no single token for labels {missing} — skipping row")
        return None

    option_logits = torch.tensor([logits[tok_ids[l]] for l in labels])
    probs = F.softmax(option_logits, dim=0).cpu().numpy()
    return {l: float(p) for l, p in zip(labels, probs)}


# ── EV and NVAS helpers ─────────────────────────────────────────────────────

def ev(probs: dict, label_to_value: dict[str, float]) -> float:
    return sum(probs[l] * label_to_value[l] for l in probs)


def nvas_score(ev_val: float, human_mean: float, vmin: int, vmax: int) -> float:
    return 1.0 - abs(ev_val - human_mean) / (vmax - vmin)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Llama-3.1-Tulu-3-8B-DPO",
                    help="HuggingFace model ID (cached at HF_HOME)")
    ap.add_argument("--master-table",
                    default="results/master_table.pkl")
    ap.add_argument("--questions-file",
                    default="Fixed_no_mention_corrected (6).xlsx")
    ap.add_argument("--output", default="results/evnvas_sensitivity.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ── load T3 metadata ──────────────────────────────────────────────────
    df = pd.read_pickle(args.master_table)
    t3_meta = (df[df.tier == 3]
               .groupby("question_id")
               .agg(vmin=("vmin", "first"), vmax=("vmax", "first"),
                    human_mean=("human_mean", "mean"))
               .reset_index())
    print(f"T3 questions: {len(t3_meta)}")

    # ── load question texts ───────────────────────────────────────────────
    qdf = pd.read_excel(args.questions_file, sheet_name="No Mention")
    qmap = dict(zip(qdf["question_id"], qdf["No Mention"]))

    # ── load model ────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, use_fast=True, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()
    print("Model loaded.")

    # ── run sensitivity test ──────────────────────────────────────────────
    rows = []
    for idx, qrow in t3_meta.iterrows():
        qid = int(qrow.question_id)
        vmin, vmax = int(qrow.vmin), int(qrow.vmax)
        human_mean = float(qrow.human_mean) if pd.notna(qrow.human_mean) else None
        question_text = qmap.get(qid)
        if question_text is None:
            print(f"  WARNING: question_id {qid} not found in xlsx — skipping")
            continue

        # Digit labels (original scale)
        digit_labels = [str(v) for v in range(vmin, vmax + 1)]
        digit_to_val = {str(v): float(v) for v in range(vmin, vmax + 1)}

        # Letter labels (same semantic order as original)
        n_opts = vmax - vmin + 1
        letter_labels = [chr(ord("A") + i) for i in range(n_opts)]
        letter_to_val = {chr(ord("A") + i): float(vmin + i) for i in range(n_opts)}

        # Construct the three prompts
        prompt_orig = question_text
        prompt_rev  = make_reversed_prompt(question_text, vmin, vmax)
        prompt_let  = make_letter_prompt(question_text, vmin, vmax)

        print(f"\n[{idx+1}/{len(t3_meta)}] Q{qid} (scale {vmin}-{vmax})")
        print(f"  ORIG: {prompt_orig[:90]}…")
        print(f"  REV:  {prompt_rev[:90]}…")
        print(f"  LET:  {prompt_let[:90]}…")

        # Extract logits
        p_orig = forced_choice_probs(model, tokenizer, prompt_orig, digit_labels, args.device)
        p_rev  = forced_choice_probs(model, tokenizer, prompt_rev,  digit_labels, args.device)
        p_let  = forced_choice_probs(model, tokenizer, prompt_let,  letter_labels, args.device)

        if p_orig is None:
            continue

        ev_orig = ev(p_orig, digit_to_val)

        # Reversed: token k now semantically means vmin + (vmax - (vmin + (k-vmin))) = vmax - (k-vmin)
        # i.e., digit vmin now labels the hi endpoint → corrected value = vmax+vmin - digit
        ev_rev_corrected = None
        if p_rev is not None:
            ev_rev_raw = ev(p_rev, digit_to_val)
            # Because scale endpoints are swapped, each digit k maps to (vmax+vmin-k)
            rev_digit_to_val = {str(v): float(vmax + vmin - v) for v in range(vmin, vmax + 1)}
            ev_rev_corrected = ev(p_rev, rev_digit_to_val)

        ev_let = ev(p_let, letter_to_val) if p_let is not None else None

        row = dict(
            question_id=qid, vmin=vmin, vmax=vmax, human_mean=human_mean,
            prompt_orig=prompt_orig, prompt_rev=prompt_rev, prompt_let=prompt_let,
            ev_orig=ev_orig,
            ev_rev_corrected=ev_rev_corrected,
            ev_let=ev_let,
            probs_orig=json.dumps(p_orig),
            probs_rev=json.dumps(p_rev) if p_rev else None,
            probs_let=json.dumps(p_let) if p_let else None,
        )
        if human_mean is not None:
            row["nvas_orig"] = nvas_score(ev_orig, human_mean, vmin, vmax)
            row["nvas_rev"]  = nvas_score(ev_rev_corrected, human_mean, vmin, vmax) if ev_rev_corrected is not None else None
            row["nvas_let"]  = nvas_score(ev_let, human_mean, vmin, vmax) if ev_let is not None else None

        rows.append(row)
        print(f"  EV: orig={ev_orig:.3f}, rev_corrected={ev_rev_corrected:.3f}, let={ev_let:.3f}")
        if human_mean and "nvas_orig" in row:
            print(f"  NVAS: orig={row['nvas_orig']:.3f}, rev={row.get('nvas_rev','—')}, let={row.get('nvas_let','—')}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")

    # ── summary stats ─────────────────────────────────────────────────────
    valid = out_df.dropna(subset=["nvas_orig", "nvas_rev", "nvas_let"])
    print(f"\n=== Stability summary ({len(valid)} questions with full human_mean) ===")
    for cond, col in [("Reversed", "nvas_rev"), ("Letter", "nvas_let")]:
        diff = (valid[col] - valid["nvas_orig"]).abs()
        print(f"  {cond}: mean |ΔNVAS|={diff.mean():.4f}, "
              f"max={diff.max():.4f}, "
              f"fraction >0.05: {(diff > 0.05).mean():.3f}")

    return out_df


if __name__ == "__main__":
    main()
