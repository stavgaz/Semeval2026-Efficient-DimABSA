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
from typing import List, Dict, Set, Tuple, Any, Optional

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
from huggingface_hub import login
import unicodedata


# --- Category Definitions (No changes) ---
def combine_lists(list1, list2):
    combinations = [f"{s1}#{s2}" for s1 in list1 for s2 in list2]
    result_dict = {}
    for index, combo in enumerate(combinations):
        result_dict[combo] = index
    return result_dict, combinations
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
# --- End Category Definitions ---

# --- Data Preparation Functions (No changes) ---
# --- Data Preparation Functions (No changes) ---
def load_jsonl_url(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def load_jsonl_file(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_prompt(lang, domain, categories, is_train=True):
    with open("/leonardo_work/EUHPC_D19_014/dimABSA/dimABSA/prompts.jsonl", "r") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]

    if is_train:
        prompt = p["train_prompt"].replace("{CATEGORIES}", str(categories))
    else:
        prompt = p["infer_prompt"].replace("{CATEGORIES}", str(categories))
    return prompt

# --- START: MODIFIED PREDICTION PROMPT FUNCTION (Few-Shot & Re-ordered) ---
def create_prediction_prompt(data_sample: Dict, category_list: List[str], model_name: str, language: str = "eng", domain: str = 'res') -> str:
    """
    Creates a formatted, prompt string for INFERENCE/PREDICTION.
    Supports multilingual instructions via the 'language' parameter.
    Languages: 'eng', 'zho', 'jpn', 'rus', 'tat', 'ukr'.
    """
    
    # 1. Create the new instruction, asking for separate V/A
    possible_categories = ", ".join(f'"{cat}"' for cat in category_list)
    # 2. Define instructions inside the function to use f-string interpolation for variables

    # 3. Get the instruction based on language (Default to English 'eng' if not found)
    instruction = load_prompt("eng", domain, possible_categories, is_train=False)
    
    text = data_sample['Text'].replace("` ` ", "")
    
    if "llama" in model_name.lower():
       
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Review: \"{text}\"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

# --- END: MODIFIED PREDICTION PROMPT FUNCTION ---

def get_original_span(full_text: str, predicted_span: str) -> str:
    """
    Robustly maps the model's predicted span back to the exact substring in the raw text.
    Handles:
      1. Spacing differences (\t, \n, spaces, ideographic \u3000)
      2. Full-width / half-width characters
      3. Case differences (Latin/Cyrillic)
      4. Tokenization artifacts
    """
    if not predicted_span:
        return predicted_span

    # 1. Check for exact match first
    if predicted_span in full_text:
        return predicted_span

    # 2. Normalize full-width characters
    full_text_norm = unicodedata.normalize("NFKC", full_text)
    predicted_span_norm = unicodedata.normalize("NFKC", predicted_span)

    # 3. Remove all whitespace for fuzzy matching
    full_text_nosp = "".join(c for c in full_text_norm if not c.isspace())
    predicted_nosp = "".join(c for c in predicted_span_norm if not c.isspace())

    # 4. Try exact match (case-sensitive)
    start_idx = full_text_nosp.find(predicted_nosp)
    if start_idx == -1:
        # 5. Case-insensitive match
        start_idx = full_text_nosp.lower().find(predicted_nosp.lower())

    if start_idx != -1:
        end_idx = start_idx + len(predicted_nosp)
        current = 0
        start_real, end_real = -1, -1

        for i, c in enumerate(full_text_norm):
            if c.isspace():
                continue
            if current == start_idx:
                start_real = i
            if current == end_idx - 1:
                end_real = i
                break
            current += 1

        if start_real != -1 and end_real != -1:
            return full_text[start_real:end_real+1]

    # Fallback
    return predicted_span

def convert_prediction_data(raw_data: List[Dict], category_list: List[str], output_file: str, model_name: str, language: str = "eng", domain: str = 'res'):
    """
    Converts data using the new strategy (separate Valence/Arousal keys).
    Supports multilingual data (Chinese, Japanese, Cyrillic) via ensure_ascii=False.
    """
    new_dataset = []
    for data_sample in raw_data:
        try:
            # FINAL EVAL: Prompt only, hide labels
            prompt_string = create_prediction_prompt(data_sample, category_list, model_name, language, domain)
            
            # We MUST use the *original* format for labels for the metric calculator
            # ensure_ascii=False here is critical for human readability of CJK labels
            new_dataset.append({
                "ID": data_sample['ID'], 
                "prompt_text": prompt_string,
                "raw_text": data_sample['Text'].replace("` ` ", ""), 
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

def parse_va_string(va_str: str) -> Optional[Tuple[float, float]]:
    try:
        v_str, a_str = va_str.split('#')
        v = float(v_str)
        a = float(a_str)
        if not (1.0 <= v <= 9.0 and 1.0 <= a <= 9.0):
            return None
        return (v, a)
    except (ValueError, TypeError, AttributeError):
        return None
# --- End Data Preparation Functions ---

def extract_json_list(text: str) -> List[dict]:
    """
    Extracts the FIRST valid JSON list from a model output.
    Returns [] if nothing valid can be recovered.
    Robust for Chinese/Japanese/Korean text.
    """
    if not text or not isinstance(text, str):
        return []

    raw = text.strip()
    
    # Normalize quotes, remove \r and tabs
    repaired = (
        raw.replace("“", '"').replace("”", '"')
           .replace("’", "'")
           .replace("\r", " ")
           .replace("\t", " ")
    )

    # Remove extra commas before closing brackets
    repaired = re.sub(r",(\s*[\]\}])", r"\1", repaired)

    # Try direct load
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except:
        pass

    # Try to extract any JSON lists (including nested)
    matches = re.findall(r"\[[^\]]*?\]", repaired)
    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    # Extract dicts
    matches = re.findall(r"{.*?}", repaired)
    for m in matches:
        try:
            return [json.loads(m)]
        except:
            continue

    # Aggressive repair for quotes / None / True / False
    aggressive = (
        repaired.replace("'", '"')
                .replace("None", "null")
                .replace("True", "true")
                .replace("False", "false")
    )

    # Try lists again
    matches = re.findall(r"\[[^\]]*?\]", aggressive)
    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    # Try dicts again
    matches = re.findall(r"{.*?}", aggressive)
    for m in matches:
        try:
            return [json.loads(m)]
        except:
            continue

    # Fallback: extract between first [ and last ]
    try:
        start = aggressive.find("[")
        end = aggressive.rfind("]")
        if start != -1 and end != -1 and end > start:
            segment = aggressive[start:end+1]
            return json.loads(segment)
    except:
        pass
    
    print('EMPTY')
    return []

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
        "zho": "meta-llama/Llama-3.1-8B-Instruct",

        # Japanese: ELYZA. The gold standard for Japanese Llama models.
        "jpn": "meta-llama/Llama-3.1-8B-Instruct",

        # Russian: Llama 3.1 is highly recommended over 3.0 for Cyrillic.
        # You could also use "IlyaGusev/saiga_llama3_8b" if you prefer a chat-specific finetune.
        "rus": "meta-llama/Llama-3.1-8B-Instruct",

        # Ukrainian: Llama 3.1 has native support. 
        # Alternatives: "UberText/Llama-3-8B-UA" (if available/verified), but 3.1 is safest base.
        "ukr": "meta-llama/Llama-3.1-8B-Instruct",

        # Tatar: Low-resource Cyrillic. 
        # Llama 3.1 is the best foundation due to massive multilingual training data.
        "tat": "meta-llama/Llama-3.1-8B-Instruct"
    }

    # Default to English base if language not found
    return model_map.get(lang, "meta-llama/Llama-3.1-8B-Instruct")

# --- Main Inference Function ---
def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained LoRA model.")
    parser.add_argument('--domain', type=str, default='restaurant', help="Domain (e.g., 'restaurant', 'laptop')")
    parser.add_argument('--language', type=str, default='zho', help="Language (e.g., 'eng')")
    parser.add_argument('--subtask', type=str, default='subtask_3', help="Subtask (e.g., 'quad')")
    parser.add_argument('--task', type=str, default='task3', help="Subtask (e.g., 'quad')")
    parser.add_argument('--base_url', type=str, default=f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a", help="Base URL for data")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference")
    parser.add_argument('--run_id', type=str, default='du19n6qg', help="run_id to inference")
    parser.add_argument(
                    "--data_root",
                    type=str,
                    default="/leonardo_work/EUHPC_D19_014/dimABSA/data/track_a",
                    help="Local root for track_a (offline). Layout: data_root/subtask/lang/*.jsonl",
                )
    
    args = parser.parse_args()

    # --- 1. Prepare Prediction Data ---
    print("--- Starting Data Preparation for Inference ---")
    domain_key = domain_mapping.get(args.domain, args.domain)
    categories_for_this_domain = category_map[domain_key][1]
    output_dir = f"/leonardo_work/EUHPC_D19_014/dimABSA/dimABSA/trans_res/{args.subtask}/{args.run_id}/{args.language}"
    predict_data_path = f"{output_dir}/{args.language}_{args.domain}_prediction_prompts.jsonl"
    output_file = f"{output_dir}/pred_{args.language}_{args.domain}.jsonl"
    
    selected_model_name = get_model_name_for_language(args.language)

    sane_model_name = selected_model_name.replace("/", "_")
    adapter_path = f"/leonardo_work/EUHPC_D19_014/dimABSA/dimABSA/trans_res/{args.subtask}/{args.run_id}/final_models/{sane_model_name}_{args.language}_{args.domain}"
    
    predict_url = f"/leonardo_work/EUHPC_D19_014/dimABSA/dimABSA/translated_for_pred_{args.language}_{args.domain}_final.jsonl"
    print(f"Loading training data from: {predict_url}")

    with open(predict_url, "r", encoding="utf-8") as f:
        predict_raw = [json.loads(line) for line in f if line.strip()]

    convert_prediction_data(predict_raw, categories_for_this_domain, predict_data_path, selected_model_name, args.language, args.domain)
    print("--- Data Preparation Complete ---")

    hf_home = os.environ.get("HF_HOME", "")
    model_source = resolve_local_model_path(selected_model_name, hf_home) if hf_home else selected_model_name
    print(f"Loading model from: {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(model_source)

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left" # Correct for batch generation

    print(f"\nLoading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # --- 3. Load Prediction Prompts ---
    print(f"Loading test data from: {predict_data_path}")
    test_data = load_jsonl_file(predict_data_path)

    final_submissions = []

    # Tokenize just to get lengths
    all_prompts = [x["prompt_text"] for x in test_data]
    all_ids = [x["ID"] for x in test_data]
    all_texts = [x["raw_text"] for x in test_data]

    prompt_lengths = [len(tokenizer(p, truncation=True)["input_ids"]) for p in all_prompts]

    # Combine prompts, IDs, texts, and lengths
    examples = list(zip(all_prompts, all_ids, all_texts, prompt_lengths))

    # Sort by length
    examples.sort(key=lambda x: x[3])

    # Now create batches
    batch_size = args.batch_size
    batches = [examples[i:i+batch_size] for i in range(0, len(examples), batch_size)]

    # Inference loop
    # Inference loop with progress bar
    for batch in tqdm(batches, desc="Generating outputs"):
        batch_prompts = [x[0] for x in batch]
        batch_ids     = [x[1] for x in batch]
        batch_texts   = [x[2] for x in batch]

        inputs = tokenizer(
            text=batch_prompts,
            return_tensors="pt",
            padding=True,          # minimal padding because lengths are similar
            truncation=True,
        ).to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        # decode only generated portion
        gen = output[:, inputs.input_ids.shape[1]:]
        batch_outputs = tokenizer.batch_decode(gen, skip_special_tokens=True)

        # store temporarily for post-processing step
        for j, raw_output in enumerate(batch_outputs):
            final_submissions.append({
                "id": batch_ids[j],
                "text": batch_texts[j],
                "raw_output": raw_output
            })

    processed = []

    for item in final_submissions:
        sample_id       = item["id"]
        raw_review_text = item["text"]
        raw_output      = item["raw_output"]

        parsed_quads = extract_json_list(raw_output)

        final_quads = []
        seen = set()

        for quad in parsed_quads:
            if not isinstance(quad, dict):
                continue

            aspect  = quad.get("Aspect", "NULL")
            opinion = quad.get("Opinion", "NULL")

            if aspect != "NULL":
                aspect = get_original_span(raw_review_text, aspect)

            if opinion != "NULL":
                opinion = get_original_span(raw_review_text, opinion)

            if aspect == "NULL" or opinion == "NULL":
                continue

            key = (aspect, quad.get("Category"), opinion)
            if key in seen:
                continue
            seen.add(key)

            v = quad.get("Valence")
            a = quad.get("Arousal")
            va_str = "NULL#NULL"

            if isinstance(v, (int, float)) and isinstance(a, (int, float)):
                if 1 <= v <= 9 and 1 <= a <= 9:
                    va_str = f"{v:.2f}#{a:.2f}"

            final_quads.append({
                "Aspect": aspect,
                "Category": quad.get("Category"),
                "Opinion": opinion,
                "VA": va_str
            })

        processed.append({
            "ID": sample_id,
            "Text": raw_review_text,
            "Quadruplet": final_quads
        })

    # --- 7. Save ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nInference complete! File: {output_file}")

if __name__ == "__main__":
    main()