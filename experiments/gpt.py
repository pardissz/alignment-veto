"""
GPT-4o-mini evaluation — mirrors code.py but uses the OpenAI API.
Reads Fixed_no_mention_reasoning.xlsx, writes gpt.xlsx (all sheets).

Auto-resume: checkpoint JSONs in gpt_checkpoints/ allow resuming within
a sheet if the run is interrupted.

Output columns per prompt column (same naming convention as code.py):
  {col}_gpt4o_mini_full_answer
  {col}_gpt4o_mini_extracted_number
  {col}_gpt4o_mini_analysis
  {col}_gpt4o_mini_normalized_probs

USAGE:
  python gpt.py
  python gpt.py --input my_prompts.xlsx --output results.xlsx
  python gpt.py --max_per_column 5   # for quick testing
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ---- API configuration ----
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-Ov8trgbrzTtZ3zTa3-mlYA")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://modelrouter.sumuk.org/v1")

MODEL = "gpt-4o-mini"
MODEL_NAME = "gpt4o_mini"
MAX_TOKENS = 64

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---- Column classification (identical to code.py) ----
PROMPT_COLUMN_HEADERS = {
    "Algeria", "Egypt", "Iran", "Iraq", "Jordan", "Kuwait", "Lebanon",
    "Libya", "Mauritania", "Morocco", "Palestine", "Qatar", "Saudi Arabia",
    "Sudan", "Tunisia", "Turkey", "LLM",
    "No Mention", "Arabic No Mention", "Persian No Mention",
    "Turkey No Mention", "Turkish No Mention",
}

METADATA_COLUMNS = {"Min", "MAX", "question_id", "question_number"}


def is_prompt_column(col_name) -> bool:
    if col_name is None:
        return False
    s = str(col_name).strip()
    if s in METADATA_COLUMNS:
        return False
    suffixes = ("_full_answer", "_extracted_number", "_analysis", "_normalized_probs")
    if any(s.endswith(suf) for suf in suffixes):
        return False
    return s in PROMPT_COLUMN_HEADERS


def extract_number(text: str, valid_options: list) -> str:
    """Extract first valid option from generated text (same logic as code.py)."""
    if not text:
        return text
    sorted_options = sorted(valid_options, key=len, reverse=True)
    for option in sorted_options:
        if re.search(r'\b' + re.escape(option) + r'\b', text):
            return option
    for option in sorted_options:
        if option in text:
            return option
    return text.strip()


def evaluate_prompt_gpt(prompt: str, valid_options: list) -> dict:
    """Call GPT-4o-mini and extract text + first-token option probabilities.

    Uses logprobs=True / top_logprobs=20 to approximate P(option) at the
    first generated token position — the same quantity code.py measures from
    local model logits. Options absent from the top-20 get probability 0.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
        )
    except Exception as e:
        uniform = 1.0 / len(valid_options)
        return {
            "generated_text": f"Error: {e}",
            "extracted_number": "Error",
            "option_probabilities": {
                opt: {"probability": 0.0, "normalized_probability": uniform}
                for opt in valid_options
            },
            "normalized_probabilities": {opt: uniform for opt in valid_options},
        }

    generated_text = response.choices[0].message.content or ""
    extracted_number = extract_number(generated_text, valid_options)

    # Build a token → probability map from the first generated token's top logprobs.
    # Tokens are stripped of surrounding whitespace so "1", " 1", "1 " all map to "1".
    token_prob_map: dict[str, float] = {}
    logprobs_content = (response.choices[0].logprobs or {}).content or []  # type: ignore[union-attr]
    if logprobs_content:
        for tlp in logprobs_content[0].top_logprobs:
            key = tlp.token.strip()
            prob = math.exp(tlp.logprob)
            if key not in token_prob_map or prob > token_prob_map[key]:
                token_prob_map[key] = prob

    raw_probs = {opt: token_prob_map.get(opt, 0.0) for opt in valid_options}

    total = sum(raw_probs.values())
    if total > 0:
        norm_probs = {opt: raw_probs[opt] / total for opt in valid_options}
    else:
        uniform = 1.0 / len(valid_options)
        norm_probs = {opt: uniform for opt in valid_options}

    option_probabilities = {
        opt: {
            "probability": raw_probs[opt],
            "normalized_probability": norm_probs[opt],
        }
        for opt in valid_options
    }

    return {
        "generated_text": generated_text,
        "extracted_number": extracted_number,
        "option_probabilities": option_probabilities,
        "normalized_probabilities": norm_probs,
    }


def process_sheet(df: pd.DataFrame, sheet_name: str,
                  ckpt_dir: Path, max_per_column: int = None) -> pd.DataFrame:
    prompt_cols = [c for c in df.columns if is_prompt_column(c)]
    print(f"\n[{sheet_name}] {len(prompt_cols)} prompt columns: {prompt_cols}")

    if "Min" not in df.columns or "MAX" not in df.columns:
        print(f"[{sheet_name}] WARNING: missing Min/MAX columns. Skipping.")
        return df

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for col in tqdm(prompt_cols, desc=sheet_name):
        ckpt_file = ckpt_dir / f"{col.replace('/', '_')}.json"
        cache = json.load(open(ckpt_file, encoding="utf-8")) if ckpt_file.exists() else {}
        if ckpt_file.exists():
            print(f"  [{col}] resuming from checkpoint ({len(cache)} cached rows)")

        valid_indices = df[col].dropna().index.tolist()
        if max_per_column:
            valid_indices = valid_indices[:max_per_column]
        pending = [idx for idx in valid_indices if str(idx) not in cache]

        for idx in tqdm(pending, desc=f"  {col}", leave=False):
            entry = df.loc[idx, col]
            if pd.isna(entry) or not str(entry).strip():
                continue
            try:
                row_min = int(float(df.loc[idx, "Min"]))
                row_max = int(float(df.loc[idx, "MAX"]))
            except (ValueError, TypeError):
                continue

            valid_options = [str(i) for i in range(row_min, row_max + 1)]
            result = evaluate_prompt_gpt(str(entry), valid_options)

            cache[str(idx)] = {
                "full_answer": result["generated_text"],
                "extracted_number": result["extracted_number"],
                "analysis": {
                    "extracted_answer": result["extracted_number"],
                    "normalized_probabilities": result["normalized_probabilities"],
                    "option_details": result["option_probabilities"],
                },
                "norm_probs": result["normalized_probabilities"],
            }

            if len(cache) % 50 == 0:
                with open(ckpt_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)

        with open(ckpt_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

        # Write results back into the dataframe
        full_col = pd.Series(index=df.index, dtype=object)
        num_col  = pd.Series(index=df.index, dtype=object)
        an_col   = pd.Series(index=df.index, dtype=object)
        np_col   = pd.Series(index=df.index, dtype=object)
        for idx in df.index:
            v = cache.get(str(idx))
            if v:
                full_col[idx] = v["full_answer"]
                num_col[idx]  = v["extracted_number"]
                an_col[idx]   = json.dumps(v["analysis"],   ensure_ascii=False)
                np_col[idx]   = json.dumps(v["norm_probs"], ensure_ascii=False)

        df[f"{col}_{MODEL_NAME}_full_answer"]       = full_col
        df[f"{col}_{MODEL_NAME}_extracted_number"]  = num_col
        df[f"{col}_{MODEL_NAME}_analysis"]          = an_col
        df[f"{col}_{MODEL_NAME}_normalized_probs"]  = np_col

    return df


def is_sheet_complete(df: pd.DataFrame) -> bool:
    prompt_cols = [c for c in df.columns if is_prompt_column(c)]
    if not prompt_cols:
        return True
    for pc in prompt_cols:
        out_col = f"{pc}_{MODEL_NAME}_full_answer"
        if out_col not in df.columns:
            return False
        if (df[pc].notna() & df[out_col].isna()).any():
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("Fixed_no_mention_corrected (6).xlsx"))
    parser.add_argument("--output", type=Path, default=Path("gpt.xlsx"))
    parser.add_argument("--max_per_column", type=int, default=None,
                        help="Limit rows per column (useful for testing)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return

    ckpt_root = Path("gpt_checkpoints")
    ckpt_root.mkdir(exist_ok=True)

    sheet_names = pd.ExcelFile(args.input).sheet_names
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Sheets: {sheet_names}")

    # Load from existing output if present (for auto-resume), else from input
    all_dfs: dict[str, pd.DataFrame] = {}
    existing_sheets: set[str] = set()
    if args.output.exists():
        try:
            xls = pd.ExcelFile(args.output)
            existing_sheets = set(xls.sheet_names)
        except Exception:
            pass
    for s in sheet_names:
        if s in existing_sheets:
            all_dfs[s] = pd.read_excel(args.output, sheet_name=s)
        else:
            all_dfs[s] = pd.read_excel(args.input, sheet_name=s)

    for sheet_name in sheet_names:
        df = all_dfs[sheet_name]
        if is_sheet_complete(df):
            print(f"\n[{sheet_name}] Already complete — skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Sheet: {sheet_name}")
        print(f"{'=' * 60}")
        ckpt_dir = ckpt_root / sheet_name.replace("/", "_").replace(" ", "_")
        df = process_sheet(df, sheet_name, ckpt_dir, args.max_per_column)
        all_dfs[sheet_name] = df

        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            for sn in sheet_names:
                all_dfs[sn].to_excel(writer, sheet_name=sn, index=False)
        print(f"Saved {args.output}  (sheet '{sheet_name}' done)")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
"""
GPT-4o-mini evaluation — mirrors code.py but uses the OpenAI API.
Reads Fixed_no_mention_reasoning.xlsx, writes gpt.xlsx (all sheets).

Auto-resume: checkpoint JSONs in gpt_checkpoints/ allow resuming within
a sheet if the run is interrupted.

Output columns per prompt column (same naming convention as code.py):
  {col}_gpt4o_mini_full_answer
  {col}_gpt4o_mini_extracted_number
  {col}_gpt4o_mini_analysis
  {col}_gpt4o_mini_normalized_probs

USAGE:
  python gpt.py
  python gpt.py --input my_prompts.xlsx --output results.xlsx
  python gpt.py --max_per_column 5   # for quick testing
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ---- API configuration ----
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-Ov8trgbrzTtZ3zTa3-mlYA")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://modelrouter.sumuk.org/v1")

MODEL = "gpt-4o-mini"
MODEL_NAME = "gpt4o_mini"
MAX_TOKENS = 256

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---- Column classification (identical to code.py) ----
PROMPT_COLUMN_HEADERS = {
    "Algeria", "Egypt", "Iran", "Iraq", "Jordan", "Kuwait", "Lebanon",
    "Libya", "Mauritania", "Morocco", "Palestine", "Qatar", "Saudi Arabia",
    "Sudan", "Tunisia", "Turkey", "LLM",
    "No Mention", "Arabic No Mention", "Persian No Mention",
    "Turkey No Mention", "Turkish No Mention",
}

METADATA_COLUMNS = {"Min", "MAX", "question_id", "question_number"}


def is_prompt_column(col_name) -> bool:
    if col_name is None:
        return False
    s = str(col_name).strip()
    if s in METADATA_COLUMNS:
        return False
    suffixes = ("_full_answer", "_extracted_number", "_analysis", "_normalized_probs")
    if any(s.endswith(suf) for suf in suffixes):
        return False
    return s in PROMPT_COLUMN_HEADERS


def extract_number(text: str, valid_options: list) -> str:
    """Extract first valid option from generated text (same logic as code.py)."""
    if not text:
        return text
    sorted_options = sorted(valid_options, key=len, reverse=True)
    for option in sorted_options:
        if re.search(r'\b' + re.escape(option) + r'\b', text):
            return option
    for option in sorted_options:
        if option in text:
            return option
    return text.strip()


def evaluate_prompt_gpt(prompt: str, valid_options: list) -> dict:
    """Call GPT-4o-mini and extract text + first-token option probabilities.

    Uses logprobs=True / top_logprobs=20 to approximate P(option) at the
    first generated token position — the same quantity code.py measures from
    local model logits. Options absent from the top-20 get probability 0.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
        )
    except Exception as e:
        uniform = 1.0 / len(valid_options)
        return {
            "generated_text": f"Error: {e}",
            "extracted_number": "Error",
            "option_probabilities": {
                opt: {"probability": 0.0, "normalized_probability": uniform}
                for opt in valid_options
            },
            "normalized_probabilities": {opt: uniform for opt in valid_options},
        }

    generated_text = response.choices[0].message.content or ""
    extracted_number = extract_number(generated_text, valid_options)

    # Build a token → probability map from the first generated token's top logprobs.
    # Tokens are stripped of surrounding whitespace so "1", " 1", "1 " all map to "1".
    token_prob_map: dict[str, float] = {}
    logprobs_content = (response.choices[0].logprobs or {}).content or []  # type: ignore[union-attr]
    if logprobs_content:
        for tlp in logprobs_content[0].top_logprobs:
            key = tlp.token.strip()
            prob = math.exp(tlp.logprob)
            if key not in token_prob_map or prob > token_prob_map[key]:
                token_prob_map[key] = prob

    raw_probs = {opt: token_prob_map.get(opt, 0.0) for opt in valid_options}

    total = sum(raw_probs.values())
    if total > 0:
        norm_probs = {opt: raw_probs[opt] / total for opt in valid_options}
    else:
        uniform = 1.0 / len(valid_options)
        norm_probs = {opt: uniform for opt in valid_options}

    option_probabilities = {
        opt: {
            "probability": raw_probs[opt],
            "normalized_probability": norm_probs[opt],
        }
        for opt in valid_options
    }

    return {
        "generated_text": generated_text,
        "extracted_number": extracted_number,
        "option_probabilities": option_probabilities,
        "normalized_probabilities": norm_probs,
    }


def process_sheet(df: pd.DataFrame, sheet_name: str,
                  ckpt_dir: Path, max_per_column: int = None) -> pd.DataFrame:
    prompt_cols = [c for c in df.columns if is_prompt_column(c)]
    print(f"\n[{sheet_name}] {len(prompt_cols)} prompt columns: {prompt_cols}")

    if "Min" not in df.columns or "MAX" not in df.columns:
        print(f"[{sheet_name}] WARNING: missing Min/MAX columns. Skipping.")
        return df

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for col in tqdm(prompt_cols, desc=sheet_name):
        ckpt_file = ckpt_dir / f"{col.replace('/', '_')}.json"
        cache = json.load(open(ckpt_file, encoding="utf-8")) if ckpt_file.exists() else {}
        if ckpt_file.exists():
            print(f"  [{col}] resuming from checkpoint ({len(cache)} cached rows)")

        valid_indices = df[col].dropna().index.tolist()
        if max_per_column:
            valid_indices = valid_indices[:max_per_column]
        pending = [idx for idx in valid_indices if str(idx) not in cache]

        for idx in tqdm(pending, desc=f"  {col}", leave=False):
            entry = df.loc[idx, col]
            if pd.isna(entry) or not str(entry).strip():
                continue
            try:
                row_min = int(float(df.loc[idx, "Min"]))
                row_max = int(float(df.loc[idx, "MAX"]))
            except (ValueError, TypeError):
                continue

            valid_options = [str(i) for i in range(row_min, row_max + 1)]
            result = evaluate_prompt_gpt(str(entry), valid_options)

            cache[str(idx)] = {
                "full_answer": result["generated_text"],
                "extracted_number": result["extracted_number"],
                "analysis": {
                    "extracted_answer": result["extracted_number"],
                    "normalized_probabilities": result["normalized_probabilities"],
                    "option_details": result["option_probabilities"],
                },
                "norm_probs": result["normalized_probabilities"],
            }

            if len(cache) % 50 == 0:
                with open(ckpt_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)

        with open(ckpt_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

        # Write results back into the dataframe
        full_col = pd.Series(index=df.index, dtype=object)
        num_col  = pd.Series(index=df.index, dtype=object)
        an_col   = pd.Series(index=df.index, dtype=object)
        np_col   = pd.Series(index=df.index, dtype=object)
        for idx in df.index:
            v = cache.get(str(idx))
            if v:
                full_col[idx] = v["full_answer"]
                num_col[idx]  = v["extracted_number"]
                an_col[idx]   = json.dumps(v["analysis"],   ensure_ascii=False)
                np_col[idx]   = json.dumps(v["norm_probs"], ensure_ascii=False)

        df[f"{col}_{MODEL_NAME}_full_answer"]       = full_col
        df[f"{col}_{MODEL_NAME}_extracted_number"]  = num_col
        df[f"{col}_{MODEL_NAME}_analysis"]          = an_col
        df[f"{col}_{MODEL_NAME}_normalized_probs"]  = np_col

    return df


def is_sheet_complete(df: pd.DataFrame) -> bool:
    prompt_cols = [c for c in df.columns if is_prompt_column(c)]
    if not prompt_cols:
        return True
    for pc in prompt_cols:
        out_col = f"{pc}_{MODEL_NAME}_full_answer"
        if out_col not in df.columns:
            return False
        if (df[pc].notna() & df[out_col].isna()).any():
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("Fixed_no_mention_reasoning.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("gpt.xlsx"))
    parser.add_argument("--max_per_column", type=int, default=None,
                        help="Limit rows per column (useful for testing)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return

    ckpt_root = Path("gpt_checkpoints")
    ckpt_root.mkdir(exist_ok=True)

    sheet_names = pd.ExcelFile(args.input).sheet_names
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Sheets: {sheet_names}")

    # Load from existing output if present (for auto-resume), else from input
    all_dfs: dict[str, pd.DataFrame] = {}
    existing_sheets: set[str] = set()
    if args.output.exists():
        try:
            xls = pd.ExcelFile(args.output)
            existing_sheets = set(xls.sheet_names)
        except Exception:
            pass
    for s in sheet_names:
        if s in existing_sheets:
            all_dfs[s] = pd.read_excel(args.output, sheet_name=s)
        else:
            all_dfs[s] = pd.read_excel(args.input, sheet_name=s)

    for sheet_name in sheet_names:
        df = all_dfs[sheet_name]
        if is_sheet_complete(df):
            print(f"\n[{sheet_name}] Already complete — skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Sheet: {sheet_name}")
        print(f"{'=' * 60}")
        ckpt_dir = ckpt_root / sheet_name.replace("/", "_").replace(" ", "_")
        df = process_sheet(df, sheet_name, ckpt_dir, args.max_per_column)
        all_dfs[sheet_name] = df

        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            for sn in sheet_names:
                all_dfs[sn].to_excel(writer, sheet_name=sn, index=False)
        print(f"Saved {args.output}  (sheet '{sheet_name}' done)")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
