# SemEval-2026 Task 3 (Track A) — Efficient DimABSA (AILS-NTUA)

Code for our AILS-NTUA submission to **SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA), Track-A**, covering:

- **Subtask 1 (DimASR)**: Dimensional Aspect Sentiment Regression (predict VA scores per aspect)
- **Subtask 2 (DimASTE)**: Dimensional Aspect Sentiment Triplet Extraction (A, O, VA)
- **Subtask 3 (DimASQP)**: Dimensional Aspect Sentiment Quadruplet Prediction (A, C, O, VA)

Our approach combines:
- **Encoder fine-tuning** for VA regression (DimASR)
- **Instruction-tuned LLM generation with LoRA** for structured JSON extraction (DimASTE/DimASQP) :contentReference[oaicite:2]{index=2}

> Paper/system description: *AILS-NTUA at SemEval-2026 Task 3: Efficient Dimensional Aspect-Based Sentiment Analysis* :contentReference[oaicite:3]{index=3}  
> Repo referenced in the paper: `stavgaz/Semeval2026-Efficient-DimABSA` :contentReference[oaicite:4]{index=4}

---

## Task format

- **Categories** (DimASQP): `ENTITY#ATTRIBUTE`
- **Valence–Arousal (VA)**: `V#A` with each in `[1.00, 9.00]` :contentReference[oaicite:5]{index=5}

---

## Models

We use language-appropriate LLM backbones for structured extraction:
- **ENG**: Llama 3.1 8B Instruct
- **ZHO**: Qwen 2.5 7B Instruct
- **JPN/RUS/UKR/TAT**: Qwen 2.5 14B Instruct :contentReference[oaicite:6]{index=6}

Prompting uses each model’s native chat template:
- Llama-style: `<|begin_of_text|> ... <|eot_id|>`
- Qwen-style: `<|im_start|> ... <|im_end|>` :contentReference[oaicite:7]{index=7}

---

## Installation

### Requirements
- Python 3.10+ recommended
- CUDA GPU recommended (the paper experiments used a single A100) :contentReference[oaicite:8]{index=8}

### Install deps
```bash
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA
pip install transformers datasets accelerate peft trl bitsandbytes tqdm numpy requests wandb
pip install unsloth
```

---

## Example Runs


### 0) (Optional) login for gated models like Llama
```bash
huggingface-cli login
```

### 1) Train LoRA (example: subtask_2, Tatar, restaurant)
```bash
python DimASTE/Training.py \
  --subtask subtask_2 \
  --language tat \
  --domain restaurant \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 4 \
  --lr 2e-4
```

### 2) Run inference (generates predictions JSONL)
```bash
python DimASTE/Inference.py \
  --subtask subtask_2 \
  --language tat \
  --domain restaurant
  --batch_size 8
```
---
