# dch-hsgr

Experiment code for hierarchical multi-candidate reasoning over math word problems
(GSM8K / MATH-500), with a Qwen2.5 backbone. Self-contained: scripts, derived
datasets, and prior pipeline outputs in JSONL.

## Setup

```bash
pip install -r requirements.txt
# Models are pulled from HuggingFace on first use:
#   Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-1.5B-Instruct
# Offline mode: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Default batch sizes in `pilot.py` assume a 141GB GPU. Halve them (or more) on
24GB cards.

## Data

```bash
python data_prep.py --which all --data-dir data
```

| File | Contents |
|---|---|
| `data/math_l5.jsonl` | Level-5 subset of MATH-500 (134 problems) |
| `data/gsm_deep_test.jsonl` | GSM8K problems with >=4 annotated steps (256), with gold intermediate values |
| `data/gsm_deep_train.jsonl` | Same, train split (4661) |
| `data/gsm_chain_test.jsonl` | Compositional GSM8K (400): one number in problem B is replaced by the answer to problem A, gold recomputed |
| `data/gsm_join_test.jsonl` | Three-node GSM join graphs (400): two independently causal parent values feed one recomputed root |

`gsm_chain` entries pass an identity-substitution check (re-evaluating with the
original number must reproduce the original gold) and filter out percentages,
numbers occurring more than once, non-integer golds, and substitutions off by
more than 3x in magnitude.

## Pipelines

```bash
bash run_s1.sh                                   # depth-2 pipeline on the new datasets
python s1_gate.py --dirs outputs_chain,outputs_mathl5

bash run_s2.sh                                   # depth-3: tree -> leaf -> mid -> root -> bp
bash run_g1_g3.sh                                # extended root sampling; trace-position probes
```

`run_s2.sh` writes `outputs_deep/s2_bp_report.json`, whose `acc_by_round` field
reports accuracy after each belief-propagation round.

## Analysis (CPU only)

```bash
python verify_latent.py    --stage all    # probe architecture, cross-domain, reasoning validity
python verify_credit.py                   # leave-one-out vs exact counterfactual vs frequency
python verify_structure.py --stage all    # conformal set size, message passing, token budget
python verify_headroom.py                 # remaining-gap decomposition
```

These need `outputs*/hidden_feats.pt`, which is not committed (~100MB).
Regenerate with:

```bash
python phase06_hidden_probe.py --stage extract \
    --dirs outputs_gsm_train,outputs_math_train,outputs,outputs_gsm_test
```

## Layout

```
*.py                  pipelines, probes, scorers, analysis
data/                 source and derived datasets
outputs*/             prior pipeline outputs (JSONL; .pt features not committed)
run_s1.sh run_s2.sh run_g1_g3.sh
```
