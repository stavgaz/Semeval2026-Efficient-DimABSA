# ============================================================
# 1) IMPORT UNSLOTH FIRST (REQUIRED)
# ============================================================
from unsloth import FastLanguageModel

# ============================================================
# 2) IMPORT TRL SFTTrainer (CLASS TO PATCH)
# ============================================================
from trl.trainer.sft_trainer import SFTTrainer

# ============================================================
# 3) PATCH TRL'S DEFAULT EOS BEFORE ANY TRAINER IS CREATED
# ============================================================
SFTTrainer.default_eos_token = "<|eot_id|>"
SFTTrainer.default_eos_token_id = 128009

print("TRL DEFAULT EOS TOKEN:", SFTTrainer.default_eos_token)
print("TRL DEFAULT EOS TOKEN ID:", SFTTrainer.default_eos_token_id)

# ============================================================
# 4) NOW IMPORT THE REST OF YOUR DEPENDENCIES
# ============================================================
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

# ============================================================
# 5) IMPORTANT: NOW SAFE TO IMPORT TRL CONFIG
# ============================================================
from trl import SFTConfig

from datasets import load_dataset
from tqdm import tqdm


# ============================================================
# Category construction helpers
# ============================================================

def combine_lists(list1, list2):
    """
    Create all combinations "<ENTITY>#<ATTRIBUTE>" for labeling.

    Returns:
      - result_dict: map from category string -> index
      - combinations: list of category strings in index order
    """
    combinations = [f"{s1}#{s2}" for s1 in list1 for s2 in list2]
    result_dict = {}
    for index, combo in enumerate(combinations):
        result_dict[combo] = index
    return result_dict, combinations


# Domain-specific label spaces
laptop_entity_labels = ['LAPTOP', 'DISPLAY', 'KEYBOARD', 'MOUSE', 'MOTHERBOARD', 'CPU', 'FANS_COOLING', 'PORTS', 'MEMORY', 'POWER_SUPPLY', 'OPTICAL_DRIVES', 'BATTERY', 'GRAPHICS', 'HARD_DISK', 'MULTIMEDIA_DEVICES', 'HARDWARE', 'SOFTWARE', 'OS', 'WARRANTY', 'SHIPPING', 'SUPPORT', 'COMPANY'] + ['OUT_OF_SCOPE']
laptop_attribute_labels = ['GENERAL', 'PRICE', 'QUALITY', 'DESIGN_FEATURES', 'OPERATION_PERFORMANCE', 'USABILITY', 'PORTABILITY', 'CONNECTIVITY', 'MISCELLANEOUS']
laptop_category_dict, laptop_category_list = combine_lists(laptop_entity_labels, laptop_attribute_labels)

restaurant_entity_labels = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
restaurant_attribute_labels= ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']
restaurant_category_dict, restaurant_category_list = combine_lists(restaurant_entity_labels, restaurant_attribute_labels)

hotel_entity_labels = ['HOTEL', 'ROOMS', 'FACILITIES', 'ROOM_AMENITIES', 'SERVICE', 'LOCATION', 'FOOD_DRINKS']
hotel_attribute_labels = ['GENERAL', 'PRICE', 'COMFORT', 'CLEANLINESS', 'QUALITY', 'DESIGN_FEATURES', 'STYLE_OPTIONS', 'MISCELLANEOUS']
hotel_category_dict, hotel_category_list = combine_lists(hotel_entity_labels, hotel_attribute_labels)

finance_entity_labels = ['MARKET', 'COMPANY', 'BUSINESS', 'PRODUCT']
finance_attribute_labels = ['GENERAL', 'SALES', 'PROFIT', 'AMOUNT', 'PRICE', 'COST']
finance_category_dict, finance_category_list = combine_lists(finance_entity_labels, finance_attribute_labels)

# Quick access maps
category_map = {
    'lap': (laptop_category_dict, laptop_category_list),
    'res': (restaurant_category_dict, restaurant_category_list),
    'hot': (hotel_category_dict, hotel_category_list),
    'fin': (finance_category_dict, finance_category_list),
}
domain_mapping = {
    'restaurant': 'res',
    'laptop': 'lap',
    'hotel': 'hot',
    'finance': 'fin'
}

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    """
    Set Python/NumPy/PyTorch RNG seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Data loading helpers
# ============================================================

def load_jsonl_url(url: str) -> List[Dict]:
    """
    Download JSONL from a URL and parse each line as JSON.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

# ============================================================
# Data conversion (build SFT training text)
# ============================================================

def convert_data(
    raw_data: List[Dict],
    category_list: List[str],
    output_file: str,
    model_name: str,
    language: str = "eng",
    domain: str = "res",
    is_prediction: bool = False
):
    """
    Convert raw dataset entries into a JSONL where each line is:
      {"text": "<FULL TRAINING PROMPT + ANSWER>"}

    NOTE: `is_prediction` is unused here; you might remove it or implement branching.
    """
    new_dataset = []
    for data_sample in raw_data:
        try:
            # Build the full supervised training string (prompt + gold JSON answer)
            prompt_string = create_new_instruction_prompt(
                data_sample, category_list, model_name, language, domain
            )
            new_dataset.append({"text": prompt_string})
        except KeyError:
            # Missing expected fields (e.g., Text or Quadruplet)
            pass
        except Exception as e:
            print(f"Error processing sample: {e}\nSample ID: {data_sample.get('ID', 'Unknown')}")

    # Make sure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # ensure_ascii=False keeps multilingual readable instead of \uXXXX escapes
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Data conversion complete! File saved to {output_file}.")
    print(f"Total prompts generated: {len(new_dataset)}")

# ============================================================
# Prompt template loading
# ============================================================

def load_prompt(lang, domain, categories, is_train=True):
    """
    Load per-language/domain prompt templates and inject categories.
    """
    with open("./prompts.jsonl", "r") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]

    if is_train:
        prompt = p["train_prompt"].replace("{CATEGORIES}", str(categories))
    else:
        prompt = p["infer_prompt"].replace("{CATEGORIES}", str(categories))
    return prompt

# ============================================================
# Prompt builder (supervised: prompt + answer JSON)
# ============================================================

def create_new_instruction_prompt(
    data_sample: Dict,
    category_list: List[str],
    model_name: str,
    language: str = "eng",
    domain: str = 'res'
) -> str:
    """
    Build the full SFT training sample:
      <chat prompt> + <gold JSON answer>

    Gold answer format:
      [
        {"Aspect": ..., "Category": ..., "Opinion": ..., "Valence": float, "Arousal": float},
        ...
      ]
    """
    # 1) Allowed categories inserted into instruction
    possible_categories = ", ".join(f'"{cat}"' for cat in category_list)
    instruction = load_prompt(language, domain, possible_categories)

    # 2) Convert old "VA": "V#A" string into separate floats Valence/Arousal
    new_quad_list = []
    for quad in data_sample['Quadruplet']:
        try:
            v_str, a_str = quad['VA'].split('#')
            v = float(v_str)
            a = float(a_str)
        except (ValueError, TypeError, AttributeError):
            # Fallback if missing/bad format
            v, a = 5.00, 5.00

        new_quad_list.append({
            "Aspect": quad.get("Aspect"),
            "Category": quad.get("Category"),
            "Opinion": quad.get("Opinion"),
            "Valence": v,
            "Arousal": a
        })

    # Indented JSON can help the model learn structure
    answer_json_string = json.dumps(new_quad_list, indent=2, ensure_ascii=False)

    # Force 1-decimal floats to 2 decimals (e.g., 5.0 -> 5.00)
    answer_json_string = re.sub(
        r'("(?:Valence|Arousal)":\s*)(\d+\.\d)(?!\d)',
        r'\g<1>\g<2>0',
        answer_json_string
    )
    # Force ints to 2-decimal floats (e.g., 5 -> 5.00)
    answer_json_string = re.sub(
        r'("(?:Valence|Arousal)":\s*)(\d+)(?!\d|\.)',
        r'\g<1>\g<2>.00',
        answer_json_string
    )

    # 3) Review text
    text = data_sample['Text'].replace("` ` ", "")

    # 4) Wrap into the model-specific chat format
    if "llama" in model_name.lower():
        if language == 'eng':
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Review: \"{text}\"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{answer_json_string}<|eot_id|>"""

    if "qwen" in model_name.lower():
        if language == 'eng':
            prompt = f"""<|im_start|>user
{instruction}

Review: \"{text}\"
<|im_end|>
<|im_start|>assistant
{answer_json_string}
<|im_end|>"""
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
# Model selection by language
# ============================================================

def get_model_name_for_language(language_code: str) -> str:
    """
    Choose base HF model per language.
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
    """
    Pipeline:
      1) Parse args
      2) Download raw training JSONL
      3) Convert to TRL SFT JSONL with {"text": "..."}
      4) Load dataset
      5) Load base model 4-bit with Unsloth
      6) Attach LoRA
      7) Train with TRL SFTTrainer
      8) Save adapter weights
    """
    parser = argparse.ArgumentParser(description="Train a LoRA model.")
    parser.add_argument('--domain', type=str, default='restaurant', help="Domain")
    parser.add_argument('--language', type=str, default='zho', help="Language")
    parser.add_argument('--subtask', type=str, default='subtask_3', help="Subtask")
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

    # --- 1) Set Seed ---
    set_seed(args.seed)
    print(f"Set random seed to {args.seed}")

    # --- 2) Prepare Data ---
    print("--- Starting Data Preparation ---")

    # Map full domain name -> short domain key used in category_map
    domain_key = domain_mapping.get(args.domain, args.domain)
    categories_for_this_domain = category_map[domain_key][1]

    output_dir = f"./{args.subtask}/{args.language}"

    train_output_path = f"{output_dir}/{args.language}_{args.domain}_train_prompts.jsonl"

    # Remote JSONL URL for training
    train_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_train_alltasks.jsonl"
    print(f"Downloading training data from: {train_url}")
    train_raw = load_jsonl_url(train_url)

    # Shuffle to reduce ordering bias
    random.shuffle(train_raw)

    # Compute warmup steps (rough estimate; ignores remainder and multi-GPU nuances)
    num_update_steps_per_epoch = len(train_raw) // (args.batch_size * args.grad_accum)
    total_training_steps = num_update_steps_per_epoch * args.epochs
    num_warmup_steps = int(total_training_steps * args.warmup_ratio)
    print(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")

    # Choose model based on language
    selected_model_name = get_model_name_for_language(args.language)

    # Convert raw samples -> SFT JSONL: {"text": "<prompt+answer>"}
    convert_data(
        train_raw,
        categories_for_this_domain,
        train_output_path,
        selected_model_name,
        args.language,
        args.domain,
        is_prediction=False
    )
    print("--- Data Preparation Complete ---")

    # --- 3) Load Dataset ---
    # Loads the JSONL where each record has field "text"
    train_dataset = load_dataset('json', data_files=train_output_path, split='train')

    # --- 4) Model & Tokenizer Setup (4-BIT) ---
    # Uses Unsloth's fast loader, optionally quantized.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=selected_model_name,
        max_seq_length=2048,
        dtype=None,          # Auto-select dtype
        load_in_4bit=True,   # 4-bit quantization
    )

    # --- 5) Add LoRA adapters ---
    # Unsloth wraps PEFT under the hood.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,  # Unsloth often recommends 0.0 for speed/stability
        bias="none",
    )

    # Ensure padding is defined for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 6) Training setup ---
    sane_model_name = selected_model_name.replace("/", "_")
    model_output_dir = f"./models/{sane_model_name}_{args.language}_{args.domain}"
    print(f"Model output directory: {model_output_dir}")

    # WARNING: bf16=True may crash on GPUs without bf16 support.
    training_args = SFTConfig(
        output_dir=f"{model_output_dir}-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        bf16=True,          # switch to fp16=True if bf16 not supported
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_steps=num_warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy="steps",
        save_total_limit=1,
        dataset_text_field="text",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    # Shows how many parameters are trainable (LoRA layers)
    print("\nTrainable parameters (BEFORE training):")
    trainer.model.print_trainable_parameters()

    # --- 7) Train ---
    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")

    # --- 8) Save LoRA adapters ---
    print(f"Saving model adapters to {model_output_dir}")
    trainer.model.save_pretrained(model_output_dir)
    print(f"Adapters saved to {model_output_dir}")

if __name__ == "__main__":
    main()
