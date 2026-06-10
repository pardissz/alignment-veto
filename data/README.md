# Data

## Model Response Dataset

The full model response dataset (~1.53M responses across 26 open models) is hosted on HuggingFace:

→ **[PardisSzah/alignment-veto-responses](https://huggingface.co/datasets/PardisSzah/alignment-veto-responses)**

```python
from datasets import load_dataset
ds = load_dataset("PardisSzah/alignment-veto-responses")
```

### Schema

| Column | Type | Description |
|---|---|---|
| `model` | str | Model identifier (e.g., `olmo_3_7b_instruct`) |
| `country` | str | One of 16 MENA countries |
| `framing` | str | `Personalization`, `Third`, or `No_Mention` |
| `language` | str | `EN` or native (`AR`, `FA`, `TR`) |
| `tier` | int | 1 (benign), 2 (moderate), 3 (sensitive) |
| `question_id` | str | WVS/AOI question identifier |
| `question_text` | str | Full question text |
| `response` | str | Model's generated response |
| `refused` | bool | Whether response is a refusal |
| `nvas` | float | NVAS score (0–1); NaN if refused |
| `ev_nvas` | float | EV-NVAS score (internal logits at refusal moment) |
| `human_mean` | float | WVS/AOI human survey mean for this country×question |

## Human Survey Sources

| Source | License | Access |
|---|---|---|
| World Values Survey Wave 7 | Academic license | [worldvaluessurvey.org](https://www.worldvaluessurvey.org) |
| Arab Opinion Index | Academic license | [arabcenterdc.org](https://www.arabcenterdc.org) |

Questions were restricted to those with ≥4 MENA nations represented in the data.
