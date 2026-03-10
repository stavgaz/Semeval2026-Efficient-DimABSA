# --- 1. IMPORT UNSLOTH FIRST (REQUIRED) ---
# Unsloth provides faster loading/training utilities and integrates cleanly with LoRA.
from unsloth import FastLanguageModel

# --- 2. IMPORT TRL SFTTrainer ---
# TRL's SFTTrainer runs supervised fine-tuning over a dataset field (here: "text").
from trl.trainer.sft_trainer import SFTTrainer

# --- 3. PATCH TRL'S DEFAULT EOS BEFORE ANY TRAINER IS CREATED ---
# We set the end-of-turn token used by TRL when it needs an EOS marker for chat-style text.
SFTTrainer.default_eos_token = "<|eot_id|>"
SFTTrainer.default_eos_token_id = 128009

print("TRL DEFAULT EOS TOKEN:", SFTTrainer.default_eos_token)
print("TRL DEFAULT EOS TOKEN ID:", SFTTrainer.default_eos_token_id)

# --- 4. IMPORT DEPENDENCIES ---
import json
import requests
import argparse
import os
import random
import re
import numpy as np
import torch
from typing import List, Dict, Any

from trl import SFTConfig
from datasets import load_dataset


# ============================================================
# Category helpers / label spaces
# ============================================================

def combine_lists(list1, list2):
    """Build all combinations like '<ENTITY>#<ATTRIBUTE>' and return dict+list."""
    combinations = [f"{s1}#{s2}" for s1 in list1 for s2 in list2]
    result_dict = {combo: idx for idx, combo in enumerate(combinations)}
    return result_dict, combinations

# Laptop
laptop_entity_labels = [
    'LAPTOP', 'DISPLAY', 'KEYBOARD', 'MOUSE', 'MOTHERBOARD', 'CPU', 'FANS_COOLING',
    'PORTS', 'MEMORY', 'POWER_SUPPLY', 'OPTICAL_DRIVES', 'BATTERY', 'GRAPHICS',
    'HARD_DISK', 'MULTIMEDIA_DEVICES', 'HARDWARE', 'SOFTWARE', 'OS', 'WARRANTY',
    'SHIPPING', 'SUPPORT', 'COMPANY', 'OUT_OF_SCOPE'
]
laptop_attribute_labels = [
    'GENERAL', 'PRICE', 'QUALITY', 'DESIGN_FEATURES', 'OPERATION_PERFORMANCE',
    'USABILITY', 'PORTABILITY', 'CONNECTIVITY', 'MISCELLANEOUS'
]
laptop_category_dict, laptop_category_list = combine_lists(laptop_entity_labels, laptop_attribute_labels)

# Restaurant
restaurant_entity_labels = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
restaurant_attribute_labels = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']
restaurant_category_dict, restaurant_category_list = combine_lists(restaurant_entity_labels, restaurant_attribute_labels)

# Hotel
hotel_entity_labels = ['HOTEL', 'ROOMS', 'FACILITIES', 'ROOM_AMENITIES', 'SERVICE', 'LOCATION', 'FOOD_DRINKS']
hotel_attribute_labels = ['GENERAL', 'PRICE', 'COMFORT', 'CLEANLINESS', 'QUALITY', 'DESIGN_FEATURES', 'STYLE_OPTIONS', 'MISCELLANEOUS']
hotel_category_dict, hotel_category_list = combine_lists(hotel_entity_labels, hotel_attribute_labels)

# Finance
finance_entity_labels = ['MARKET', 'COMPANY', 'BUSINESS', 'PRODUCT']
finance_attribute_labels = ['GENERAL', 'SALES', 'PROFIT', 'AMOUNT', 'PRICE', 'COST']
finance_category_dict, finance_category_list = combine_lists(finance_entity_labels, finance_attribute_labels)

# Maps domain short-key -> (dict, list)
category_map = {
    'lap': (laptop_category_dict, laptop_category_list),
    'res': (restaurant_category_dict, restaurant_category_list),
    'hot': (hotel_category_dict, hotel_category_list),
    'fin': (finance_category_dict, finance_category_list),
}

# Maps CLI domain name -> short-key
domain_mapping = {
    'restaurant': 'res',
    'laptop': 'lap',
    'hotel': 'hot',
    'finance': 'fin'
}


# ============================================================
# Repro + Data loading
# ============================================================

def set_seed(seed: int):
    """Set RNG seeds for reproducible shuffling/training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl_url(url: str) -> List[Dict]:
    """Download JSONL from URL and parse each line."""
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]


# ============================================================
# Data conversion (raw -> SFT text)
# ============================================================

def convert_data(
    raw_data: List[Dict],
    category_list: List[str],
    output_file: str,
    model_name: str,
    language: str = "eng",
    domain: str = "res",
):
    """
    Convert raw training samples into SFT-ready JSONL.

    Output JSONL format:
      {"text": "<chat prompt + gold JSON answer>"}

    TRL SFTTrainer will read the field "text" and train on it.
    """
    new_dataset = []
    for data_sample in raw_data:
        try:
            prompt_string = create_new_instruction_prompt(
                data_sample, category_list, model_name, language, domain
            )
            new_dataset.append({"text": prompt_string})
        except KeyError:
            # Sample missing expected keys (e.g., Text/Quadruplet)
            pass
        except Exception as e:
            print(f"Error processing sample: {e}\nSample ID: {data_sample.get('ID', 'Unknown')}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Data conversion complete! File saved to {output_file}.")
    print(f"Total prompts generated: {len(new_dataset)}")


# ============================================================
# Prompt loading / building
# ============================================================

def load_prompt(lang, domain, categories, is_train=True):
    """
    Load a prompt template and inject the category list.

    Expected prompts.jsonl structure (must be a single JSON object):
      {
        "eng_restaurant": {"train_prompt": "...{CATEGORIES}...", "infer_prompt": "..."},
        ...
      }
    """
    with open("./prompts.jsonl", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]

    if is_train:
        return p["train_prompt"].replace("{CATEGORIES}", str(categories))
    return p["infer_prompt"].replace("{CATEGORIES}", str(categories))


def create_new_instruction_prompt(
    data_sample: Dict,
    category_list: List[str],
    model_name: str,
    language: str = "eng",
    domain: str = "res",
) -> str:
    """
    Build one supervised training example:
      - Instruction (includes list of valid categories)
      - Review text
      - Gold output JSON list (Aspect/Category/Opinion + Valence/Arousal)

    The final string is formatted in the model's chat template style (Llama in this script).
    """
    possible_categories = ", ".join(f"\"{cat}\"" for cat in category_list)
    instruction = load_prompt("eng", domain, possible_categories, is_train=True)

    # Convert VA string "V#A" into separate float fields
    new_quad_list = []
    for quad in data_sample["Quadruplet"]:
        try:
            v_str, a_str = quad["VA"].split("#")
            v = float(v_str)
            a = float(a_str)
        except (ValueError, TypeError, AttributeError):
            v, a = 5.00, 5.00

        new_quad_list.append({
            "Aspect": quad.get("Aspect"),
            "Category": quad.get("Category"),
            "Opinion": quad.get("Opinion"),
            "Valence": v,
            "Arousal": a,
        })

    answer_json_string = json.dumps(new_quad_list, indent=2, ensure_ascii=False)

    # Normalize numeric formatting to 2 decimals
    answer_json_string = re.sub(
        r'("(?:Valence|Arousal)":\s*)(\d+\.\d)(?!\d)',
        r"\g<1>\g<2>0",
        answer_json_string,
    )
    answer_json_string = re.sub(
        r'("(?:Valence|Arousal)":\s*)(\d+)(?!\d|\.)',
        r"\g<1>\g<2>.00",
        answer_json_string,
    )

    text = data_sample["Text"].replace("` ` ", "")

    # Llama chat-format: user message + assistant message containing gold JSON
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Review: \"{text}\"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{answer_json_string}<|eot_id|>"""

    return prompt


# ============================================================
# Model selection
# ============================================================

def get_model_name_for_language(language_code: str) -> str:
    """
    Select the base model to train.

    In this version we use Llama 3.1 8B Instruct for all languages.
    """
    lang = language_code.lower().strip()
    model_map = {
        "eng": "meta-llama/Llama-3.1-8B-Instruct",
        "zho": "meta-llama/Llama-3.1-8B-Instruct",
        "jpn": "meta-llama/Llama-3.1-8B-Instruct",
        "rus": "meta-llama/Llama-3.1-8B-Instruct",
        "ukr": "meta-llama/Llama-3.1-8B-Instruct",
        "tat": "meta-llama/Llama-3.1-8B-Instruct",
    }
    return model_map.get(lang, "meta-llama/Llama-3.1-8B-Instruct")


# ============================================================
# Main training entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train a LoRA model with Unsloth + TRL SFTTrainer.")
    parser.add_argument("--domain", type=str, default="restaurant")
    parser.add_argument("--language", type=str, default="zho")
    parser.add_argument("--subtask", type=str, default="subtask_3")
    parser.add_argument("--base_url", type=str, default="https://cdn.jsdelivr.net/gh/DimABSA/DimABSA2026@main/task-dataset/track_a/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.2)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Set random seed to {args.seed}")

    # Determine category set for this domain
    domain_key = domain_mapping.get(args.domain, args.domain)
    categories_for_this_domain = category_map[domain_key][1]

    # Paths for generated SFT prompt file
    output_dir = f"./{args.subtask}/{args.language}"
    train_output_path = f"{output_dir}/{args.language}_{args.domain}_train_prompts.jsonl"

    # Training data is expected to exist locally (e.g., produced by a translation script)
    train_path = f"./translated_{args.language}_{args.domain}_final.jsonl"
    print(f"Loading training data from: {train_path}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f if line.strip()]

    random.shuffle(train_raw)

    # Warmup steps computed from dataset size and effective batch size
    num_update_steps_per_epoch = len(train_raw) // (args.batch_size * args.grad_accum)
    total_training_steps = num_update_steps_per_epoch * args.epochs
    num_warmup_steps = int(total_training_steps * args.warmup_ratio)
    print(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")

    # Base model selection
    selected_model_name = get_model_name_for_language(args.language)

    # Create SFT dataset JSONL
    convert_data(
        train_raw,
        categories_for_this_domain,
        train_output_path,
        selected_model_name,
        args.language,
        args.domain,
    )

    # Load dataset as Hugging Face Dataset
    train_dataset = load_dataset("json", data_files=train_output_path, split="train")

    # Load base model in 4-bit to reduce memory usage
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=selected_model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        local_files_only=False,  # allow download if not cached
    )

    # Attach LoRA adapters on top of the base model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    # Ensure the tokenizer has a pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Output directory for adapters/checkpoints
    sane_model_name = selected_model_name.replace("/", "_")
    model_output_dir = f"./models/{sane_model_name}_{args.language}_{args.domain}"
    print(f"Model output directory: {model_output_dir}")

    # Mixed precision selection based on GPU capability
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        output_dir=f"{model_output_dir}-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        bf16=bf16_ok,
        fp16=not bf16_ok,
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

    # Save LoRA adapters
    trainer.model.save_pretrained(model_output_dir)
    print(f"Adapters saved to {model_output_dir}")


if __name__ == "__main__":
    main()
