# --- 1. IMPORT UNSLOTH FIRST (REQUIRED) ---
# Unsloth must be imported early because it can patch/optimize parts of the HF stack.
from unsloth import FastLanguageModel

# --- 2. IMPORT TRL SFTTrainer (THE CLASS TO PATCH) ---
# NOTE: This import path is version-sensitive; some TRL versions moved SFTTrainer.
from trl.trainer.sft_trainer import SFTTrainer

# --- 3. PATCH TRL'S DEFAULT EOS BEFORE ANY TRAINER IS CREATED ---
SFTTrainer.default_eos_token = "<|eot_id|>"
SFTTrainer.default_eos_token_id = 128009

print("TRL DEFAULT EOS TOKEN:", SFTTrainer.default_eos_token)
print("TRL DEFAULT EOS TOKEN ID:", SFTTrainer.default_eos_token_id)

# --- 4. NOW IMPORT THE REST OF YOUR DEPENDENCIES ---
import json
import requests
import argparse
import os
import random
import re
import numpy as np
import torch
import math
import logging
from typing import List, Dict, Set, Tuple, Any
import wandb  # NOTE: imported but not used below (not fatal)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    PeftModel,
)

# --- 5. IMPORTANT: NOW SAFE TO IMPORT TRL CONFIG ---
from trl import SFTConfig

from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# Utility helpers
# ============================================================

def set_seed(seed: int):
    """Set RNG seeds for reproducibility (Python / NumPy / PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl_url(url: str) -> List[Dict]:
    """Download a JSONL file from a URL and parse it into a list of dicts."""
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def convert_data(raw_data: List[Dict], output_file: str, model_name: str, language: str = "eng", domain: str = "res"):
    """
    Convert raw training items into a JSONL with a single 'text' field:
      {"text": "<chat prompt + gold answer>"}

    NOTE: This function depends on create_new_instruction_prompt.
    """
    new_dataset = []
    for data_sample in raw_data:
        try:
            # Build a full supervised training sample (prompt + answer JSON)
            prompt_string = create_new_instruction_prompt(data_sample, model_name, language, domain)
            new_dataset.append({"text": prompt_string})
        except KeyError:
            # Missing expected keys like 'Text' or 'Quadruplet'
            pass
        except Exception as e:
            print(f"Error processing sample: {e}\nSample ID: {data_sample.get('ID', 'Unknown')}")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # ensure_ascii=False keeps multilingual text readable (no \uXXXX escapes)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Data conversion complete! File saved to {output_file}.")
    print(f"Total prompts generated: {len(new_dataset)}")

def load_prompt(lang, domain, is_train=True):
    """
    Load prompt template from prompts2.jsonl.
    """
    with open("./prompts2.jsonl", "r") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]

    prompt = p["prompt"]
    return prompt


# ============================================================
# Prompt builder (SFT)
# ============================================================

def create_new_instruction_prompt(data_sample: Dict, model_name: str, language: str = "eng", domain: str = 'res') -> str:
    """
    Build one supervised training string: instruction + review + gold JSON answer.

    ⚠️ FATAL RISK: `prompt` is only assigned inside certain branches:
      - Llama only supports language == 'eng' here
      - Qwen supports ukr/tat/zho/jpn/rus but NOT 'eng'
    If you choose a model/language combination not covered, you'll get:
      UnboundLocalError: local variable 'prompt' referenced before assignment

    Possible fix:
      - initialize prompt = None at the top
      - after branches, if prompt is None: raise ValueError(...)
    """
    instruction = load_prompt(language, domain)

    # Convert "VA" field from "V#A" into separate numeric Valence/Arousal
    new_quad_list = []
    for quad in data_sample['Quadruplet']:
        try:
            v_str, a_str = quad['VA'].split('#')
            v = float(v_str)
            a = float(a_str)
        except (ValueError, TypeError, AttributeError):
            # fallback if malformed/missing
            v, a = 5.00, 5.00

        # NOTE: You are not including Category here. If your task expects it, this is a
        # schema mismatch (might not crash training, but can harm consistency).
        # Possible fix (optional):
        #   "Category": quad.get("Category"),
        new_quad_list.append({
            "Aspect": quad.get("Aspect"),
            "Opinion": quad.get("Opinion"),
            "Valence": v,
            "Arousal": a
        })

    # Pretty JSON can help models learn structured output
    answer_json_string = json.dumps(new_quad_list, indent=2, ensure_ascii=False)

    # Force floats like 5.0 -> 5.00
    answer_json_string = re.sub(
        r'("(?:Valence|Arousal)":\s*)(\d+\.\d)(?!\d)',
        r'\g<1>\g<2>0',
        answer_json_string
    )
    # Force ints like 5 -> 5.00
    answer_json_string = re.sub(
        r'("(?:Valence|Arousal)":\s*)(\d+)(?!\d|\.)',
        r'\g<1>\g<2>.00',
        answer_json_string
    )

    text = data_sample['Text'].replace("` ` ", "")

    # Llama chat formatting (only eng supported in your code)
    if "llama" in model_name.lower():
        if language == 'eng':
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Review: \"{text}\"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{answer_json_string}<|eot_id|>"""

    # Qwen chat formatting (no 'eng' case in your code)
    if "qwen" in model_name.lower():
        if language == 'ukr':
            prompt = f"""<|im_start|>user
{instruction}

Текст відгуку: \"{text}\"
<|im_end|>
<|im_start|>assistant
{answer_json_string}
<|im_end|>"""
        elif language == 'tat':
            prompt = f"""<|im_start|>user
{instruction}

Текст бәяләмәсе \"{text}\"
<|im_end|>
<|im_start|>assistant
{answer_json_string}
<|im_end|>"""
        elif language == 'zho':
            prompt = f"""<|im_start|>user
{instruction}

評論: \"{text}\"
<|im_end|>
<|im_start|>assistant
{answer_json_string}
<|im_end|>"""
        elif language == 'jpn':
            prompt = f"""<|im_start|>user
{instruction}

レビュー本文：\"{text}\"
<|im_end|>
<|im_start|>assistant
{answer_json_string}
<|im_end|>"""
        elif language == 'rus':
            prompt = f"""<|im_start|>user
{instruction}

Отзыв: \"{text}\"
<|im_end|>
<|im_start|>assistant
{answer_json_string}
<|im_end|>"""

    return prompt


# ============================================================
# Model choice helper
# ============================================================

def get_model_name_for_language(language_code: str) -> str:
    """
    Choose HF base model per language.
    """
    lang = language_code.lower().strip()
    model_map = {
        "eng": "meta-llama/Llama-3.1-8B-Instruct",
        "zho": "Qwen/Qwen2.5-7B-Instruct",
        "jpn": "Qwen/Qwen2.5-14B-Instruct",
        "rus": "Qwen/Qwen2.5-14B-Instruct",
        "ukr": "Qwen/Qwen2.5-14B-Instruct",
        "tat": "Qwen/Qwen2.5-14B-Instruct"
    }
    return model_map.get(lang, "meta-llama/Llama-3.1-8B-Instruct")


# ============================================================
# Main training entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train a LoRA model.")
    parser.add_argument('--domain', type=str, default='restaurant', help="Domain")
    parser.add_argument('--language', type=str, default='tat', help="Language")
    parser.add_argument('--subtask', type=str, default='subtask_2', help="Subtask")
    parser.add_argument('--base_url', type=str, default="https://cdn.jsdelivr.net/gh/DimABSA/DimABSA2026@main/task-dataset/track_a/", help="Base URL for data")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--epochs', type=int, default=1, help="Max number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Per-device batch size")
    parser.add_argument('--grad_accum', type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay")
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Warmup ratio")
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help="LR scheduler type")
    parser.add_argument('--lora_r', type=int, default=16, help="LoRA rank 'r'")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.2, help="LoRA dropout")
    parser.add_argument('--lora_target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help="LoRA target modules")
    args = parser.parse_args()

    global tokenizer

    # --- 1. Set Seed ---
    set_seed(args.seed)
    print(f"Set random seed to {args.seed}")

    # --- 2. Prepare Data ---
    print("--- Starting Data Preparation ---")
    output_dir = f"./{args.subtask}/{args.language}"
    train_output_path = f"{output_dir}/{args.language}_{args.domain}_train_prompts.jsonl"

    train_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_train_alltasks.jsonl"
    print(f"Downloading training data from: {train_url}")
    train_raw = load_jsonl_url(train_url)
    random.shuffle(train_raw)

    # Estimate training steps for warmup calculation
    num_update_steps_per_epoch = len(train_raw) // (args.batch_size * args.grad_accum)
    total_training_steps = num_update_steps_per_epoch * args.epochs
    num_warmup_steps = int(total_training_steps * args.warmup_ratio)
    print(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")

    selected_model_name = get_model_name_for_language(args.language)

    # Convert raw training examples to SFT prompts JSONL
    convert_data(train_raw, train_output_path, selected_model_name, args.language, args.domain)
    print("--- Data Preparation Complete ---")

    # --- 3. Load Datasets ---
    train_dataset = load_dataset('json', data_files=train_output_path, split='train')

    # --- 4. Model & Tokenizer Setup (4-BIT) ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=selected_model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # --- 5. Add LoRA adapters ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    # Ensure pad token exists for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --- 6. Training setup ---
    print("Setting up training arguments...")

    sane_model_name = selected_model_name.replace("/", "_")
    model_output_dir = f"./models/{sane_model_name}_{args.language}_{args.domain}"
    print(f"Model output directory: {model_output_dir}")

    training_args = SFTConfig(
        output_dir=f"{model_output_dir}-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_steps=num_warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy="steps",
        save_total_limit=1,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    print("\nTrainable parameters (BEFORE training):")
    trainer.model.print_trainable_parameters()

    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")

    trainer.model.save_pretrained(model_output_dir)
    print(f"Best model adapters saved to {model_output_dir}")


if __name__ == "__main__":
    main()
