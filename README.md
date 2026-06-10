# The Alignment Veto: How Safety Training Suppresses Cultural Knowledge in LLMs

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-PardisSzah-blue)](https://huggingface.co/datasets/PardisSzah/alignment-veto-responses)
[![Website](https://img.shields.io/badge/🌐%20Website-GitHub%20Pages-teal)](https://pardissz.github.io/alignment-veto)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Pardis Sadat Zahraei, Gokhan Tur, Dilek Hakkani-Tür, Ehsaneddin Asgari**

---

## TL;DR

When a language model refuses a culturally sensitive question, is the knowledge **erased** or **suppressed**?

We show it's **suppressed**: at the moment of refusal, a model's internal logit distributions correlate with human survey data **more strongly** than its freely generated answers. We call this the **alignment veto**.

<p align="center">
  <img src="figures/main_figure.png" width="680" alt="Alignment Veto Illustration"/>
</p>

---

## Abstract

Across 16 MENA countries, 26 models, and 1.53M human survey responses, we show the answer is suppression, not erasure: at the moment of refusal, a model's internal logit distribution correlates with human survey data more strongly than its freely generated answers. We call this the **alignment veto**.

We distinguish **suppression failures** (accurate internal distributions blocked at output) from **representational bias failures** (the encoding itself diverges from human values), and show the two require different interventions.

The gate is inequitable: the **safety tax reaches 37.6%**, with a **19.8% alignment-quality gap** between best- and worst-served nations, and native-language prompting **widens rather than closes** it. Sparse autoencoder analysis corroborated by comparisons across alignment stages identifies a **candidate DPO-stage feature** mediating suppression in Tulu-3-8B.

---

## Key Findings

| Finding | Result |
|---|---|
| Safety tax (T3 vs T1 refusal rate) | **+11.5%** mean; up to **37.6%** (ALLAM-7B) |
| Internal alignment at refusal | Refused T3 EV-NVAS **0.718** > Accepted T3 NVAS **0.690** |
| Country equity gap | **19.8%** (Algeria 0.532 vs Palestine 0.731) |
| Native-language NVAS loss | **−0.050** points (all 26 models drop) |
| Arabic country collapse | **66.5%** identical responses across 14 Arabic countries |
| SAE veto feature prevalence | T3: **28.6%** vs T2: **<0.4%** vs T1: **0%** (70× ratio) |
| DPO ablation T3 shift | Mean **+0.250** (p=0.016) across 20 seeds; T1 = 0.000 in all 40 seeds |
| Third-person framing gain | **2.6×** more NVAS benefit on T3 than Persona framing |

---

## Two Failure Modes

```
                  High Refusal Rate
                        ▲
         SUPPRESSION    │    
         FAILURES       │    
         (T3-dominant)  │  model refuses BUT
         ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─
                        │   internal logits ≈ human
         REPRESENTATIONAL│
         BIAS FAILURES  │
         (gender eq.)   │    
                        └──────────────────▶
                           High NVAS when answered
```

| Failure Mode | Root Cause | Fix |
|---|---|---|
| **Suppression failure** | Alignment gate blocks accurate internal distribution | Third-person framing, targeted DPO data curation |
| **Representational bias** | Model encoding itself diverges from human data | MENA-specific training data; prompt engineering cannot help |

---

## Dataset

Model responses (~1.53M) for 26 open models × 864 questions × 16 countries × 6 framings × 2 languages are released on HuggingFace:

→ [PardisSzah/alignment-veto-responses](https://huggingface.co/datasets/PardisSzah/alignment-veto-responses)

### Question Tiers

| Tier | n | Description |
|---|---|---|
| **T1 — Benign** | 47 | Demographics and preferences (e.g., importance of family) |
| **T2 — Moderate** | 788 | Value-laden but not directly safety-targeted |
| **T3 — Sensitive** | 29 | LGBTQ+ acceptance, domestic violence norms, gender equality, religious tolerance |

### Framing Conditions (3 × 2 design)

| Framing | Language | Prompt style |
|---|---|---|
| Neutral | EN / Native | Direct query, no identity framing |
| Persona | EN / Native | "Imagine you are [nationality]..." |
| Observer (Third) | EN / Native | "How would an average [nationality] respond..." |

### Countries (16 MENA nations)

Algeria, Egypt, Iran, Iraq, Jordan, Kuwait, Lebanon, Libya, Mauritania, Morocco, Palestine, Qatar, Saudi Arabia, Sudan, Tunisia, Turkey

---

## Models Evaluated

| Family | Models | Stages |
|---|---|---|
| OLMo-3 | 7B, 32B | Base, SFT, DPO, IT |
| Tulu-3 | 8B | SFT, DPO; 3.1-8B IT |
| LLaMA-3.1 | 8B | Base, IT |
| Gemma-3 | 4B, 12B, 27B | IT |
| Qwen | 2.5-7B, 3-4B, 3-30B-MoE | IT |
| GPT | 4o-mini, 5 | — |
| Arabic-specialized | ALLAM-7B, AYA 8B/32B, FANAR-1.9B, Jais-2-8B | IT |

---

## Metrics

**NVAS** (Normalised Value Alignment Score):
```
NVAS = 1 − |ŷ − y_human| / (y_max − y_min)
```

**EV-NVAS** (Expected Value NVAS): extracts internal logit distribution at the first generated token during refusal, renormalized over valid scale options. Validated at 92.5% argmax match on answered rows.

**Safety Tax**: Mean refusal-rate difference between T3 and T1 questions.

---

## Repository Structure

```
alignment-veto/
├── analysis/
│   ├── run_experiments.py        # Main analysis: NVAS, refusal rates, framing
│   ├── run_framing_analysis.py   # Framing condition comparison
│   ├── run_gap1_third_framing.py # Third-framing intervention analysis
│   ├── run_gap2_probing.py       # Residualized ridge-regression probing
│   ├── run_gap3_frontier.py      # Frontier model (GPT-4o-mini, GPT-5) analysis
│   ├── run_gap4_native.py        # Native-language analysis
│   ├── run_gap6_regression.py    # Logistic regression (tier × framing × country)
│   ├── run_gap7_qualitative.py   # Qualitative example generation
│   └── run_evnvas_validation.py  # EV-NVAS validation
├── experiments/
│   ├── run_base_parallel.py      # Parallel inference on open models
│   ├── run_experiments2.py       # Extended experiments
│   ├── run_ablation_country.py   # Country-level ablation
│   └── gpt.py                    # GPT-4o-mini / GPT-5 inference
├── mechanistic/
│   ├── run_fix1_residual_probe.py # Residualized probing across layers
│   ├── run_fix2_sae_random_control.py # SAE random feature control
│   ├── run_fix3_t3_topics.py     # T3 topic taxonomy
│   └── run_extract_lm_head.py    # LM head activation extraction
├── figures/
│   ├── make_fixed_figures.py     # Paper figure generation
│   ├── make_missing_figures.py   # Supplementary figures
│   ├── make_qualitative_examples.py
│   ├── make_residual_probe_plot.py
│   ├── make_suppression_delta.py
│   └── make_suppression_scatter.py
├── data/
│   └── README.md                 # Data access instructions
├── docs/
│   └── index.html                # GitHub Pages website
└── main3.tex                     # Paper LaTeX source
```

---

## Quick Start

### Reproduce the main analysis

```bash
git clone https://github.com/pardissz/alignment-veto
cd alignment-veto

pip install -r requirements.txt

# Run main NVAS / refusal / framing analysis
python analysis/run_experiments.py

# Run third-person framing intervention
python analysis/run_gap1_third_framing.py

# Run native-language analysis
python analysis/run_gap4_native.py

# Run residualized probing
python analysis/run_gap2_probing.py
```

### Load the response dataset

```python
from datasets import load_dataset

ds = load_dataset("PardisSzah/alignment-veto-responses")
# Each row: model, country, framing, language, tier, question, response, nvas, ev_nvas, refused
```

---

## Citation

```bibtex
@article{zahraei2026alignmentveto,
  title     = {The Alignment Veto: How Safety Training Suppresses Cultural Knowledge in LLMs},
  author    = {Zahraei, Pardis Sadat and Tur, Gokhan and Hakkani-T\"{u}r, Dilek and Asgari, Ehsaneddin},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

---

## License

Released under the [MIT License](LICENSE).  
Human survey data (WVS Wave 7, Arab Opinion Index) are used under academic licenses — see [data/README.md](data/README.md).
