# --- 1. IMPORT UNSLOTH FIRST (REQUIRED) ---
from unsloth import FastLanguageModel

# --- 2. IMPORT TRL SFTTrainer (THE CLASS TO PATCH) ---
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
import wandb

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl_url(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def convert_data(raw_data: List[Dict], output_file: str, model_name: str, language: str = "eng", domain: str = "res"):
    """
    Converts data using the new strategy (separate Valence/Arousal keys).
    Supports multilingual data (Chinese, Japanese, Cyrillic) via ensure_ascii=False.
    """
    new_dataset = []
    for data_sample in raw_data:
        try:
            # TRAINING: Full prompt + answer
            prompt_string = create_new_instruction_prompt(data_sample, model_name, language, domain)
            new_dataset.append({
                "text": prompt_string
            })
        except KeyError as e:
            pass 
        except Exception as e:
            print(f"Error processing sample: {e}\nSample ID: {data_sample.get('ID', 'Unknown')}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # CRITICAL: ensure_ascii=False prevents Chinese/Russian from turning into \uXXXX garbage
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Data conversion complete! File saved to {output_file}.")
    print(f"Total prompts generated: {len(new_dataset)}")

def load_prompt(lang, domain, is_train=True):
    with open("/leonardo_work/EUHPC_D19_014/dimABSA/dimABSA/prompts2.jsonl", "r") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]

    prompt = p["prompt"]

    return prompt

# --- START: MODIFIED PROMPT FUNCTION (Added V/A Definitions) ---
def create_new_instruction_prompt(data_sample: Dict, model_name: str, language: str = "eng", domain: str = 'res') -> str:
    """
    Creates a formatted prompt string for prediction (no answer).
    Supports multilingual instructions via the 'language' parameter.
    Languages: 'eng', 'zho', 'jpn', 'rus', 'tat', 'ukr'.
    """
    
    # 2. Get the instruction based on language (Default to English 'eng' if not found)
    instruction = load_prompt(language, domain)
    
    # 3. Parse the old "V#A" format and build the new-format "answer" JSON
    new_quad_list = []
    for quad in data_sample['Quadruplet']:
        try:
            v_str, a_str = quad['VA'].split('#')
            v = float(v_str)
            a = float(a_str)
        except (ValueError, TypeError, AttributeError):
            v, a = 5.00, 5.00 # Fallback on error
            
        new_quad_list.append({
            "Aspect": quad.get("Aspect"),
            "Opinion": quad.get("Opinion"),
            "Valence": v,
            "Arousal": a
        })

    # `indent=2` makes the training data human-readable and helps the model learn the structure
    answer_json_string = json.dumps(new_quad_list, indent=2, ensure_ascii=False)

    answer_json_string = re.sub(
        r'("(?:Valence|Arousal)":\s*)(\d+\.\d)(?!\d)', 
        r'\g<1>\g<2>0', 
        answer_json_string
    )
    
    # Matches: "Key": 5 -> "Key": 5.00 (Just in case integer slipped through)
    answer_json_string = re.sub(
        r'("(?:Valence|Arousal)":\s*)(\d+)(?!\d|\.)', 
        r'\g<1>\g<2>.00', 
        answer_json_string
    )

    # 3. Get the review text
    text = data_sample['Text'].replace("` ` ", "")
    
    # 4. Create the final, full prompt string
    if "llama" in model_name.lower():

        if language == 'eng':
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Review: \"{text}\"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{answer_json_string}<|eot_id|>"""

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
    


def resolve_local_model_path(model_id: str, hf_home: str) -> str:
    """
    Map a HF model id to the local directory where we downloaded it.
    Falls back to model_id if not found.
    """
    mapping = {
        "meta-llama/Llama-3.1-8B-Instruct": os.path.join(hf_home, "models", "meta-llama__Llama-3.1-8B-Instruct"),
        "Qwen/Qwen2.5-7B-Instruct":         os.path.join(hf_home, "models", "Qwen__Qwen2.5-7B-Instruct"),
        "Qwen/Qwen2.5-14B-Instruct":        os.path.join(hf_home, "models", "Qwen__Qwen2.5-14B-Instruct"),
    }
    local = mapping.get(model_id, model_id)
    # If it looks like a local path, validate it has config.json
    if os.path.isdir(local):
        cfg = os.path.join(local, "config.json")
        if not os.path.exists(cfg):
            raise RuntimeError(f"Local model dir exists but missing config.json: {local}")
    return local

def get_model_name_for_language(language_code: str) -> str:
    """
    Returns the Hugging Face model ID most appropriate for the given language.
    
    Strategy:
    - Use language-specific fine-tunes for distinct scripts/grammars (Chinese, Japanese).
    - Use Llama 3.1 (instead of 3.0) for Cyrillic languages (Russian, Ukrainian, Tatar) 
      because 3.1 has significantly better multilingual pre-training.
    - Default to Llama 3.1 for English (Best 8B model as of late 2024).
    """
    
    # Normalize input just in case
    lang = language_code.lower().strip()

    model_map = {
        # English: Llama 3.1 is superior to 3.0 for instruction following and JSON adherence.
        "eng": "meta-llama/Llama-3.1-8B-Instruct",

        # Chinese: HFL (Hugging Face Library team) fine-tune. 
        # Fixes English-bias issues and improves tokenizer for Hanzi.
        "zho": "Qwen/Qwen2.5-7B-Instruct",

        # Japanese: ELYZA. The gold standard for Japanese Llama models.
        "jpn": "Qwen/Qwen2.5-14B-Instruct",

        # Russian: Llama 3.1 is highly recommended over 3.0 for Cyrillic.
        # You could also use "IlyaGusev/saiga_llama3_8b" if you prefer a chat-specific finetune.
        "rus": "Qwen/Qwen2.5-14B-Instruct",

        # Ukrainian: Llama 3.1 has native support. 
        # Alternatives: "UberText/Llama-3-8B-UA" (if available/verified), but 3.1 is safest base.
        "ukr": "Qwen/Qwen2.5-14B-Instruct",

        # Tatar: Low-resource Cyrillic. 
        # Llama 3.1 is the best foundation due to massive multilingual training data.
        "tat": "Qwen/Qwen2.5-14B-Instruct"
    }

    # Default to English base if language not found
    return model_map.get(lang, "meta-llama/Llama-3.1-8B-Instruct")

# --- Main Training Function ---
def main():
    parser = argparse.ArgumentParser(description="Train a LoRA model.")
    # (All parser arguments are the same)
    parser.add_argument('--domain', type=str, default='restaurant', help="Domain")
    parser.add_argument('--language', type=str, default='tat', help="Language")
    parser.add_argument('--subtask', type=str, default='subtask_2', help="Subtask")
    parser.add_argument('--base_url', type=str, default=f"https://cdn.jsdelivr.net/gh/DimABSA/DimABSA2026@main/task-dataset/track_a/", help="Base URL for data")
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
    parser.add_argument(
                    "--data_root",
                    type=str,
                    default="/leonardo_work/EUHPC_D19_014/dimABSA/data/track_a",
                    help="Local root for track_a (offline). Layout: data_root/subtask/lang/*.jsonl",
                )
    args = parser.parse_args()

    run = wandb.init(project="llms", config=args)

    global tokenizer
    
    # --- 1. Set Seed ---
    set_seed(args.seed)
    print(f"Set random seed to {args.seed}")

    # --- 2. Prepare Data ---
    print("--- Starting Data Preparation ---")
    output_dir = f"/leonardo_work/EUHPC_D19_014/dimABSA/dimABSA/pure_llm/{args.subtask}/{run.id}/{args.language}"
    train_output_path = f"{output_dir}/{args.language}_{args.domain}_train_prompts.jsonl"
    train_path = os.path.join(
        args.data_root,
        args.subtask,
        args.language,
        f"{args.language}_{args.domain}_train_alltasks.jsonl",
    )
    print(f"Loading training data from: {train_path}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f if line.strip()]

    random.shuffle(train_raw)

    num_update_steps_per_epoch = len(train_raw) // (args.batch_size * args.grad_accum)
    total_training_steps = num_update_steps_per_epoch * args.epochs
    num_warmup_steps = int(total_training_steps * args.warmup_ratio)
    print(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")
    
    selected_model_name = get_model_name_for_language(args.language)
    wandb.config["model_name"] = selected_model_name
    
    convert_data(train_raw,train_output_path, selected_model_name, args.language, args.domain)
    print("--- Data Preparation Complete ---")
    
    # --- 3. Load Datasets ---
    train_dataset = load_dataset('json', data_files=train_output_path, split='train')

    hf_home = os.environ.get("HF_HOME", "")
    model_source = resolve_local_model_path(selected_model_name, hf_home) if hf_home else selected_model_name
    print(f"Loading model from: {model_source}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_source,
        max_seq_length = 2048, # Reduce this if you still OOM (e.g., to 2048)
        dtype = None,          # Auto-detects your GPU's best dtype
        load_in_4bit = True,   # Forces 4-bit to fit in 48GB
    )

    # 2. Add LoRA adapters (Unsloth handles the config internally)
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = args.lora_target_modules,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout, # Unsloth recommends 0 for speed
        bias = "none",
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --- 6. Training ---
    print("Setting up training arguments...")
    
    sane_model_name = selected_model_name.replace("/", "_")
    model_output_dir = f"/leonardo_work/EUHPC_D19_014/dimABSA/dimABSA/pure_llm/{args.subtask}/{run.id}/final_models/{sane_model_name}_{args.language}_{args.domain}"
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
        report_to="wandb",
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

    print(f"Saving best model (based on eval_loss) to {model_output_dir}")
    trainer.model.save_pretrained(model_output_dir)
    print(f"Best model adapters saved to {model_output_dir}")
    

if __name__ == "__main__":
    main()