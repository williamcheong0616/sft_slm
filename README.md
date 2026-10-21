# SEALion SFT — Supervised Fine-Tuning

Fine-tune [SEA-LION](https://sea-lion.ai/) models on your own data using **QLoRA**.

## Supported Models

| Directory | Model | Architecture | Params | Context |
|---|---|---|---|---|
| `gemma/` | [Gemma-SEA-LION-v4-4B-VL](https://hf.co/aisingapore/Gemma-SEA-LION-v4-4B-VL) | Gemma 3 | 4B | 128K |
| `llama/` | [Llama-SEA-LION-v3.5-8B-R](https://hf.co/aisingapore/Llama-SEA-LION-v3.5-8B-R) | Llama 3.1 | 8B | 128K |
| `qwen/` | [Qwen-SEA-LION-v4-8B-VL](https://hf.co/aisingapore/Qwen-SEA-LION-v4-8B-VL) | Qwen3 | 8B | 256K |

## Project Structure

```
sft_slm/
├── gemma/                         # Gemma-SEA-LION-v4-4B-VL
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── merge_weights.py
├── llama/                         # Llama-SEA-LION-v3.5-8B-R
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── merge_weights.py
├── qwen/                          # Qwen-SEA-LION-v4-8B-VL
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── merge_weights.py
├── scripts/
│   └── prepare_data.py            # Validate & split JSONL data (shared)
├── data/
│   ├── sample_instruction.jsonl
│   └── sample_chat.jsonl
├── eval_metrics.py                # Benchmark generation + metrics (CIs, paired tests)
├── human_eval_analyze.py          # Human-eval agreement + significance tests
├── tests/                         # CPU-only smoke tests
├── requirements.txt
└── README.md
```

Each model directory is **self-contained** — no shared config or imports between them.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your data (shared across all models)

```bash
python scripts/prepare_data.py --input data/your_data.jsonl
```

### 3. Train — pick a model directory

```bash
python gemma/train.py --no-wandb
python llama/train.py --no-wandb
python qwen/train.py --no-wandb

# With overrides
python llama/train.py --epochs 5 --lr 1e-4 --batch-size 1 --lora-rank 32
```

### 4. Inference

```bash
python gemma/inference.py --prompt "Explain what SEALion is"
python llama/inference.py --interactive
python qwen/inference.py --prompt "Translate to Malay: Hello"
```

### 5. Evaluate

```bash
python llama/evaluate.py --eval-file data/eval.jsonl
```

### 6. Merge weights (optional)

```bash
python llama/merge_weights.py
```

### 7. Benchmark evaluation (`eval_metrics.py`)

Generates responses for the base / zero-shot / fine-tuned variants and scores them
with BLEU-4, chrF, ROUGE-L, BERTScore and CMI. Every metric gets a 95% bootstrap CI,
and each family gets a paired-bootstrap p-value for finetuned vs zeroshot.

```bash
pip install sacrebleu rouge-score bert-score openpyxl numpy

# Old eval_results/ were sampled from curated_data.jsonl, which includes the training rows. Archive them first.
mv eval_results eval_results_contaminated

python eval_metrics.py --resample --force --seed 42 \
    --vanilla-model qwen=<upstream base checkpoint from the Qwen-SEA-LION-v4-8B-VL model card>
```

| Flag | Default | Purpose |
|---|---|---|
| `--test-file` | `data/eval.jsonl` | Held-out split to sample the test set from. Both `messages` (prepare_data.py) and `conversations` records are accepted. The script stops with an error if there are fewer than `--samples` usable records. |
| `--train-file` | `data/train.jsonl` | The file the adapters were trained on. Used to report exact and normalised (lowercase, whitespace-collapsed) test/train overlap in `overlap_report.json`. |
| `--resample` | off | Rebuild `eval_results/test_samples.json`. The cache stores the source path and sha256. The script refuses to reuse a cache built from a different file, size or seed. |
| `--seed` | `42` | Generation seed (`transformers.set_seed`). It is recorded in each `responses_<label>.json` and in `benchmark_raw.json`. |
| `--vanilla-model FAMILY=ID` | see `VANILLA_MODEL_IDS` | Override the vanilla base checkpoint (repeatable). The default `Qwen/Qwen2.5-7B-Instruct` is **not** the upstream base of Qwen-SEA-LION-v4-8B-VL. |
| `--n-bootstrap` / `--bootstrap-seed` | `1000` / `12345` | Item resamples for the CIs and paired tests. |
| `--samples`, `--model`, `--no-base`, `--no-zeroshot`, `--no-finetuned`, `--force`, `--skip-generate` | | Unchanged. |

Notes:
- The script refuses response files whose questions don't match the current test set,
  or that were generated with a different model ID. Use `--force` to regenerate them.
- For Qwen-family templates the chat template gets `enable_thinking=False`.
  Before scoring, any `<think>…</think>` trace is removed: full blocks, an orphan
  `</think>`, or an unclosed `<think>` cut off at `max_new_tokens`. The count of each
  is logged and saved. The raw text stays in `responses_<label>.json`.
- `cmi_standard` is the Gambäck–Das CMI, 100 × (1 − max_lang / (n − u)), where u is
  the tokens matched by neither lexicon. `cmi_heuristic` is the old formula, kept for
  comparison. Both use the small hand-built Malay and English lexicons in `eval_metrics.py`.
- CIs are percentile bootstrap intervals over items, using one resample matrix shared
  by all variants. BLEU and chrF are recomputed at corpus level on each resample.
  The p-values are two-sided paired-bootstrap tests with null-centred deltas.
- Outputs: `benchmark_summary.csv` (scores, CIs, and delta/p vs zeroshot on finetuned
  rows), `benchmark_paired.csv`, `benchmark_results.xlsx` and `benchmark_raw.json`.
  The JSON metadata covers the test-set hash, overlap, seeds, model IDs and think-strip counts.
- The overlap check only catches exact and normalised duplicates, not paraphrases. It is
  only meaningful against the exact file the adapters were trained on. If an adapter
  came from `train_iterative.py`, pass that run's training data instead.

### 8. Human evaluation analysis

```bash
pip install pandas numpy scipy openpyxl krippendorff   # krippendorff is optional
python human_eval_analyze.py
```

This reports ordinal Krippendorff's α per dimension. α uses every (question, variant)
unit with at least two ratings. Fleiss' κ is kept for comparison. It also runs
two-sided Wilcoxon signed-rank tests between variants for each dimension. Scores are
averaged over annotators for each question, and Holm correction is applied over all
variant pairs × dimensions. Results go to `human_eval_kappa.json`,
`human_eval_wilcoxon.csv` and `human_eval_summary.xlsx`.

### Tests

CPU-only smoke tests (no model loading):

```bash
pip install pytest numpy pandas scipy sacrebleu krippendorff
python -m pytest tests -q
```

## Data Format

All three models share the same data format. See `data/sample_instruction.jsonl` and `data/sample_chat.jsonl` for examples.
