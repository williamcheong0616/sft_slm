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

## Data Format

All three models share the same data format. See `data/sample_instruction.jsonl` and `data/sample_chat.jsonl` for examples.
