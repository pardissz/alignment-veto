"""
Parallel launcher for filling missing base-model sheets.

For 7B models (OLMo-7B-base, Llama-3.1-8B-base):
  - 4 workers, one per GPU, each handling 4 of 16 countries
  - Each GPU loads a full model copy (~14 GB bfloat16), batch_size=32

For 32B model (OLMo-3-32B-base):
  - 2 workers, 2 GPUs each (model needs ~64 GB), batch_size=16

After all workers finish, results are merged column-by-column back into the
original xlsx (read checkpoint JSONs directly to avoid xlsx write conflicts).
"""

import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

BASE  = Path('/shared/storage-01/users/zahraei2/mena_normal')
DATA  = BASE / 'MENA_TRANSLATED_reasoning'
HF    = Path('/shared/storage-01/huggingface/hub')
HF2   = Path('/shared/storage-01/models/hub')
LOG   = BASE / 'results' / 'base_missing_log.txt'

COUNTRIES = [
    'Algeria','Egypt','Iran','Iraq','Jordan','Kuwait','Lebanon',
    'Libya','Mauritania','Morocco','Palestine','Qatar',
    'Saudi Arabia','Sudan','Tunisia','Turkey',
]

BASE_MODELS = {
    'olmo_3_7b_base': {
        'path': HF / 'models--allenai--Olmo-3-1025-7B' / 'snapshots' /
                'a81bae42db3975be1671e27b9c9a56da1a9f980f',
        'gpus': [['0'], ['1'], ['2'], ['3']],   # 4 workers, 1 GPU each
        'batch_size': 32,
    },
    'llama_3.1_8b_base': {
        'path': HF2 / 'models--meta-llama--Llama-3.1-8B' / 'snapshots' /
                'd04e592bb4f6aa9cfee91e2e20afa771667e1d4b',
        'gpus': [['0'], ['1'], ['2'], ['3']],
        'batch_size': 32,
    },
    'olmo_3_32b_base': {
        'path': HF / 'models--allenai--Olmo-3-1125-32B' / 'snapshots' /
                'c2b61dae89a1ad10e4ad5653d0e46b590902607b',
        'gpus': [['0', '1'], ['2', '3']],       # 2 workers, 2 GPUs each
        'batch_size': 16,
    },
}

# All prompt-column names that could appear in any sheet
ALL_PROMPT_COLS = COUNTRIES + [
    'LLM',
    'No Mention', 'Arabic No Mention', 'Persian No Mention', 'Turkey No Mention',
]


def split_columns(cols, n_workers):
    """Round-robin split of cols into n_workers groups."""
    groups = [[] for _ in range(n_workers)]
    for i, c in enumerate(cols):
        groups[i % n_workers].append(c)
    return groups


def pending_sheets(xlsx_path, model_name):
    """Return list of sheet names that still need model answers."""
    xls = pd.ExcelFile(xlsx_path)
    pending = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        # Check if every prompt column that exists in the sheet has its output col
        prompt_cols = [c for c in df.columns if c in ALL_PROMPT_COLS]
        for pc in prompt_cols:
            out = f'{pc}_{model_name}_full_answer'
            if out not in df.columns:
                pending.append(sheet); break
            pmask = df[pc].notna()
            omask = df[out].notna()
            if (pmask & ~omask).any():
                pending.append(sheet); break
    return pending


def merge_worker_results(xlsx_path, model_name, ckpt_dir, sheet_names):
    """
    Read the original xlsx, overlay all checkpoint data column by column,
    and write back the complete file.
    """
    xls = pd.ExcelFile(xlsx_path)
    all_dfs = {s: xls.parse(s) for s in sheet_names}

    for sheet in sheet_names:
        df       = all_dfs[sheet]
        sheet_ck = ckpt_dir / sheet
        if not sheet_ck.exists():
            continue

        for ckpt_file in sheet_ck.glob('*.json'):
            col = ckpt_file.stem.replace('_', ' ')
            # Try both underscore and space versions
            if col not in df.columns:
                col = ckpt_file.stem  # keep underscores
            if col not in df.columns:
                # Try to find matching col
                stem = ckpt_file.stem
                matches = [c for c in df.columns if c.replace(' ', '_') == stem]
                if matches:
                    col = matches[0]
                else:
                    continue

            cache = json.load(open(ckpt_file, encoding='utf-8'))
            if not cache:
                continue

            full_s = pd.Series(index=df.index, dtype=object)
            num_s  = pd.Series(index=df.index, dtype=object)
            an_s   = pd.Series(index=df.index, dtype=object)
            np_s   = pd.Series(index=df.index, dtype=object)
            ref_s  = pd.Series(index=df.index, dtype=object)

            for idx in df.index:
                v = cache.get(str(idx))
                if not v:
                    continue
                full_s[idx] = v['full_answer']
                num_s[idx]  = v['extracted_number']
                an_s[idx]   = json.dumps(v['analysis'], ensure_ascii=False)
                np_s[idx]   = json.dumps(v['norm_probs'], ensure_ascii=False)
                # Refusal: extracted_number not in valid range
                try:
                    row_min = int(float(df.loc[idx, 'Min']))
                    row_max = int(float(df.loc[idx, 'MAX']))
                    valid = set(str(i) for i in range(row_min, row_max + 1))
                    if v['extracted_number'] not in valid:
                        ref_s[idx] = 'refusal'
                except Exception:
                    pass

            df[f'{col}_{model_name}_full_answer']      = full_s
            df[f'{col}_{model_name}_extracted_number'] = num_s
            df[f'{col}_{model_name}_refusal']          = ref_s
            df[f'{col}_{model_name}_analysis']         = an_s
            df[f'{col}_{model_name}_normalized_probs'] = np_s

        all_dfs[sheet] = df

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sn in sheet_names:
            all_dfs[sn].to_excel(writer, sheet_name=sn, index=False)
    print(f'  Merged → {xlsx_path}', flush=True)


def run_model(model_name, cfg):
    xlsx_path  = DATA / f'{model_name}.xlsx'
    model_path = cfg['path']
    gpu_groups = cfg['gpus']
    batch_size = cfg['batch_size']
    n_workers  = len(gpu_groups)

    print(f'\n{"="*60}', flush=True)
    print(f'MODEL: {model_name}  ({n_workers} workers)', flush=True)
    print(f'Path:  {model_path}', flush=True)

    if not xlsx_path.exists():
        print(f'  xlsx not found: {xlsx_path}', flush=True); return
    if not model_path.exists():
        print(f'  model not found: {model_path}', flush=True); return

    sheets = pd.ExcelFile(xlsx_path).sheet_names
    missing = pending_sheets(xlsx_path, model_name)
    print(f'  Pending sheets: {missing}', flush=True)
    if not missing:
        print('  Nothing to do.', flush=True); return

    # Determine which prompt columns to process across ALL pending sheets
    xls = pd.ExcelFile(xlsx_path)
    cols_to_process = set()
    for sheet in missing:
        df = xls.parse(sheet)
        for c in df.columns:
            if c in ALL_PROMPT_COLS:
                cols_to_process.add(c)
    cols_to_process = sorted(cols_to_process, key=lambda c: ALL_PROMPT_COLS.index(c)
                             if c in ALL_PROMPT_COLS else 999)
    print(f'  Columns to process: {cols_to_process}', flush=True)

    # Split columns across workers
    col_groups = split_columns(cols_to_process, n_workers)
    ckpt_dir   = DATA / '_checkpoints' / model_name

    # Launch workers
    procs = []
    log_files = []
    for w, (gpus, cols) in enumerate(zip(gpu_groups, col_groups)):
        if not cols:
            continue
        gpu_str = ','.join(gpus)
        log_path = LOG.parent / f'base_{model_name}_w{w}.txt'
        log_files.append(log_path)
        lf = open(log_path, 'w')
        cmd = [
            sys.executable, '-u',
            str(BASE / 'run_base_worker.py'),
            '--model_name', model_name,
            '--model_path', str(model_path),
            '--xlsx',       str(xlsx_path),
            '--columns',    ','.join(cols),
            '--batch_size', str(batch_size),
            '--ckpt_dir',   str(ckpt_dir),
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_str
        print(f'  Worker {w}: GPU={gpu_str}  cols={cols}', flush=True)
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
        procs.append((p, log_path, w))

    # Wait for all workers
    print(f'  Waiting for {len(procs)} workers...', flush=True)
    while True:
        still_running = [(p, lf, w) for p, lf, w in procs if p.poll() is None]
        if not still_running:
            break
        time.sleep(30)
        for p, lf, w in still_running:
            # Print last line of each worker log
            try:
                lines = open(lf).readlines()
                if lines:
                    print(f'    W{w}: {lines[-1].rstrip()}', flush=True)
            except Exception:
                pass

    # Check exit codes
    failed = [w for p, lf, w in procs if p.returncode != 0]
    if failed:
        print(f'  WARNING: workers {failed} exited with error.', flush=True)

    print(f'  All workers done. Merging results...', flush=True)
    merge_worker_results(xlsx_path, model_name, ckpt_dir, sheets)
    print(f'  {model_name} complete.', flush=True)


def main():
    LOG.parent.mkdir(exist_ok=True)
    # Run models sequentially (they share GPUs, can't overlap 7B and 32B)
    for model_name, cfg in BASE_MODELS.items():
        run_model(model_name, cfg)
    print('\n=== All base models done ===', flush=True)


if __name__ == '__main__':
    main()
