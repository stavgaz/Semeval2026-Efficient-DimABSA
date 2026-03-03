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

from trl import SFTConfig

from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login
import unicodedata

# ----------------------------
# Category construction helpers
# ----------------------------

def combine_lists(list1, list2):
    """
    Create all pairwise combinations between list1 and list2 in the form:
      "<ENTITY>#<ATTRIBUTE>"

    Returns:
      - result_dict: mapping combo_str -> index
      - combinations: list of combo_str in the same enumeration order
    """
    combinations = [f"{s1}#{s2}" for s1 in list1 for s2 in list2]
    result_dict = {}
    for index, combo in enumerate(combinations):
        result_dict[combo] = index
    return result_dict, combinations


# Define domain-specific entity + attribute label sets
laptop_entity_labels = [
    'LAPTOP', 'DISPLAY', 'KEYBOARD', 'MOUSE', 'MOTHERBOARD', 'CPU', 'FANS_COOLING',
    'PORTS', 'MEMORY', 'POWER_SUPPLY', 'OPTICAL_DRIVES', 'BATTERY', 'GRAPHICS',
    'HARD_DISK', 'MULTIMEDIA_DEVICES', 'HARDWARE', 'SOFTWARE', 'OS', 'WARRANTY',
    'SHIPPING', 'SUPPORT', 'COMPANY'
] + ['OUT_OF_SCOPE']
laptop_attribute_labels = [
    'GENERAL', 'PRICE', 'QUALITY', 'DESIGN_FEATURES', 'OPERATION_PERFORMANCE',
    'USABILITY', 'PORTABILITY', 'CONNECTIVITY', 'MISCELLANEOUS'
]
laptop_category_dict, laptop_category_list = combine_lists(laptop_entity_labels, laptop_attribute_labels)

restaurant_entity_labels = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
restaurant_attribute_labels = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']
restaurant_category_dict, restaurant_category_list = combine_lists(restaurant_entity_labels, restaurant_attribute_labels)

hotel_entity_labels = ['HOTEL', 'ROOMS', 'FACILITIES', 'ROOM_AMENITIES', 'SERVICE', 'LOCATION', 'FOOD_DRINKS']
hotel_attribute_labels = ['GENERAL', 'PRICE', 'COMFORT', 'CLEANLINESS', 'QUALITY', 'DESIGN_FEATURES', 'STYLE_OPTIONS', 'MISCELLANEOUS']
hotel_category_dict, hotel_category_list = combine_lists(hotel_entity_labels, hotel_attribute_labels)

finance_entity_labels = ['MARKET', 'COMPANY', 'BUSINESS', 'PRODUCT']
finance_attribute_labels = ['GENERAL', 'SALES', 'PROFIT', 'AMOUNT', 'PRICE', 'COST']
finance_category_dict, finance_category_list = combine_lists(finance_entity_labels, finance_attribute_labels)

# A single map to access category lists/dicts by short domain keys
category_map = {
    'lap': (laptop_category_dict, laptop_category_list),
    'res': (restaurant_category_dict, restaurant_category_list),
    'hot': (hotel_category_dict, hotel_category_list),
    'fin': (finance_category_dict, finance_category_list),
}

# Map long domain names -> short keys used above
domain_mapping = {
    'restaurant': 'res',
    'laptop': 'lap',
    'hotel': 'hot',
    'finance': 'fin'
}

# ----------------------------
# Data loading helpers
# ----------------------------

def load_jsonl_url(url: str) -> List[Dict]:
    """
    Download a JSONL file from a URL and parse it line-by-line.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def load_jsonl_file(filepath: str) -> List[Dict]:
    """
    Load a local JSONL file and parse it line-by-line.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_prompt(lang, domain, categories, is_train=True):
    """
    Load prompt templates from prompts.jsonl (NOTE: despite extension, you use json.load).
    It expects a dict keyed by "<lang>_<domain>" containing train_prompt/infer_prompt.
    """
    with open("./prompts.jsonl", "r") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]

    # Replace placeholder with a string representation of categories
    if is_train:
        prompt = p["train_prompt"].replace("{CATEGORIES}", str(categories))
    else:
        prompt = p["infer_prompt"].replace("{CATEGORIES}", str(categories))
    return prompt

# ----------------------------
# Prompt creation for inference
# ----------------------------

def create_prediction_prompt(
    data_sample: Dict,
    category_list: List[str],
    model_name: str,
    language: str = "eng",
    domain: str = 'res'
) -> str:
    """
    Create a formatted prompt for inference/prediction.

    - category_list: list of allowed "ENTITY#ATTRIBUTE" categories for the domain
    - model_name: used to pick the chat template (Llama vs Qwen)
    - language: selects a localized instruction + localized "Review:" label

    Languages supported in your code:
      - eng, zho, jpn, rus, tat, ukr
    """
    # Create a pretty string: "cat1", "cat2", ...
    possible_categories = ", ".join(f'"{cat}"' for cat in category_list)

    # Load domain/language-specific instruction template and inject categories
    instruction = load_prompt(language, domain, possible_categories, is_train=False)

    # Clean up some tokenization artifacts found in your data
    text = data_sample['Text'].replace("` ` ", "")

    # Build the final prompt in the target model's expected chat format.
    # NOTE: Your code only explicitly handles English for Llama here.
    prompt = ""

    if "llama" in model_name.lower():
        if language == 'eng':
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Review: \"{text}\"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

    if "qwen" in model_name.lower():
        # Qwen chat template variants per language
        if language == 'eng':
            prompt = f"""<|im_start|>user
{instruction}

Review: \"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        if language == 'zho':
            prompt = f"""<|im_start|>user
{instruction}

評論: \"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == 'jpn':
            prompt = f"""<|im_start|>user
{instruction}

レビュー本文：\"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == 'ukr':
            prompt = f"""<|im_start|>user
{instruction}

Текст відгуку: \"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == 'tat':
            prompt = f"""<|im_start|>user
{instruction}

Текст бәяләмәсе \"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == 'rus':
            prompt = f"""<|im_start|>user
{instruction}

Отзыв: \"{text}\"
<|im_end|>
<|im_start|>assistant
"""

    return prompt

# ----------------------------
# Span recovery / alignment
# ----------------------------

def get_original_span(full_text: str, predicted_span: str) -> str:
    """
    Map a model-predicted span back to the exact substring in the original raw text.

    Handles:
      1) whitespace differences (tabs/newlines/ideographic spaces)
      2) full-width vs half-width (NFKC)
      3) case differences (Latin/Cyrillic)
      4) minor tokenization artifacts

    Approach:
      - try exact match
      - normalize and remove whitespace, then find
      - translate the no-space indices back to original indices
      - fallback to predicted_span if nothing found
    """
    if not predicted_span:
        return predicted_span

    # Fast path: exact match
    if predicted_span in full_text:
        return predicted_span

    # Normalize full-width characters, etc.
    full_text_norm = unicodedata.normalize("NFKC", full_text)
    predicted_span_norm = unicodedata.normalize("NFKC", predicted_span)

    # Remove all whitespace for robust matching
    full_text_nosp = "".join(c for c in full_text_norm if not c.isspace())
    predicted_nosp = "".join(c for c in predicted_span_norm if not c.isspace())

    # Try case-sensitive, then case-insensitive
    start_idx = full_text_nosp.find(predicted_nosp)
    if start_idx == -1:
        start_idx = full_text_nosp.lower().find(predicted_nosp.lower())

    # If found, map back to original indices
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
            return full_text[start_real:end_real + 1]

    return predicted_span

# ----------------------------
# Convert raw dev data -> prompt JSONL for inference
# ----------------------------

def convert_prediction_data(
    raw_data: List[Dict],
    category_list: List[str],
    output_file: str,
    model_name: str,
    language: str = "eng",
    domain: str = 'res'
):
    """
    Convert raw samples into a JSONL file containing:
      - ID
      - prompt_text (constructed chat prompt)
      - raw_text (cleaned original review text)

    ensure_ascii=False is used to keep CJK/Cyrillic readable.
    """
    new_dataset = []

    for data_sample in raw_data:
        try:
            prompt_string = create_prediction_prompt(
                data_sample,
                category_list,
                model_name,
                language,
                domain
            )

            new_dataset.append({
                "ID": data_sample['ID'],
                "prompt_text": prompt_string,
                "raw_text": data_sample['Text'].replace("` ` ", ""),
            })

        except KeyError:
            # If the sample is missing expected keys (ID/Text), skip it.
            pass
        except Exception as e:
            print(f"Error processing sample: {e}\nSample ID: {data_sample.get('ID', 'Unknown')}")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Data conversion complete! File saved to {output_file}.")
    print(f"Total prompts generated: {len(new_dataset)}")

# ----------------------------
# VA parsing utilities
# ----------------------------

def parse_va_string(va_str: str) -> Optional[Tuple[float, float]]:
    """
    Parse a "V#A" string into floats, ensuring both are in [1, 9].
    Returns None if invalid.
    """
    try:
        v_str, a_str = va_str.split('#')
        v = float(v_str)
        a = float(a_str)
        if not (1.0 <= v <= 9.0 and 1.0 <= a <= 9.0):
            return None
        return (v, a)
    except (ValueError, TypeError, AttributeError):
        return None

# ----------------------------
# Output JSON recovery
# ----------------------------

def extract_json_list(text: str) -> List[dict]:
    """
    Extract the FIRST valid JSON list (or dict) from a model output string.

    Why this exists:
      LLMs often produce extra text, broken quotes, trailing commas, etc.
      This attempts several repairs and fallbacks.

    Returns:
      - A Python list of dicts (possibly length 1 if the model returned a dict)
      - [] if nothing valid could be recovered
    """
    if not text or not isinstance(text, str):
        return []

    raw = text.strip()

    # Basic quote normalization + cleanup
    repaired = (
        raw.replace("“", '"').replace("”", '"')
           .replace("’", "'")
           .replace("\r", " ")
           .replace("\t", " ")
    )

    # Remove dangling commas before ']' or '}'
    repaired = re.sub(r",(\s*[\]\}])", r"\1", repaired)

    # Attempt direct json.loads first
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except:
        pass

    # Try to find bracketed list-like substrings: [ ... ]
    matches = re.findall(r"\[[^\]]*?\]", repaired)
    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    # Try to find dict-like substrings: { ... }
    matches = re.findall(r"{.*?}", repaired)
    for m in matches:
        try:
            return [json.loads(m)]
        except:
            continue

    # More aggressive repairs for Python-y outputs
    aggressive = (
        repaired.replace("'", '"')
                .replace("None", "null")
                .replace("True", "true")
                .replace("False", "false")
    )

    matches = re.findall(r"\[[^\]]*?\]", aggressive)
    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    matches = re.findall(r"{.*?}", aggressive)
    for m in matches:
        try:
            return [json.loads(m)]
        except:
            continue

    # Final fallback: take substring between first '[' and last ']'
    try:
        start = aggressive.find("[")
        end = aggressive.rfind("]")
        if start != -1 and end != -1 and end > start:
            segment = aggressive[start:end + 1]
            return json.loads(segment)
    except:
        pass

    print('EMPTY')
    return []

# ----------------------------
# Model selection by language
# ----------------------------

def get_model_name_for_language(language_code: str) -> str:
    """
    Pick an HF model ID based on language.

    Your comments mention Llama 3.1 for Cyrillic languages, but the map currently
    points rus/ukr/tat/jpn to Qwen2.5-14B-Instruct and zho to Qwen2.5-7B.
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

# ----------------------------
# Main inference pipeline
# ----------------------------

def main():
    """
    End-to-end inference script:

    1) Decide domain/language/category space
    2) Load raw dev/test JSONL
    3) Convert to prompt JSONL
    4) Load base model + tokenizer
    5) Load LoRA adapter
    6) Batched generation (sorted by prompt length for efficiency)
    7) Post-process: recover JSON, normalize spans, validate VA, dedupe
    8) Save final JSONL submission
    """
    parser = argparse.ArgumentParser(description="Run inference with a trained LoRA model.")
    parser.add_argument('--domain', type=str, default='restaurant', help="Domain (e.g., 'restaurant', 'laptop')")
    parser.add_argument('--language', type=str, default='zho', help="Language (e.g., 'eng')")
    parser.add_argument('--subtask', type=str, default='subtask_3', help="Subtask (e.g., 'quad')")
    parser.add_argument('--task', type=str, default='task3', help="Subtask (e.g., 'quad')")
    parser.add_argument(
        '--base_url',
        type=str,
        default="https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a",
        help="Base URL for data"
    )
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()

    # --- 1) Prepare Prediction Data ---
    print("--- Starting Data Preparation for Inference ---")

    # Map long domain name -> internal short key
    domain_key = domain_mapping.get(args.domain, args.domain)

    # Get the list of allowed categories for that domain
    categories_for_this_domain = category_map[domain_key][1]

    # Output directories / files
    output_dir = f"./{args.subtask}/{args.language}"
    predict_data_path = f"{output_dir}/{args.language}_{args.domain}_prediction_prompts.jsonl"
    output_file = f"{output_dir}/pred_{args.language}_{args.domain}.jsonl"

    # Choose base model for this language
    selected_model_name = get_model_name_for_language(args.language)

    # Adapter directory name: replace "/" to make a local folder-safe path
    sane_model_name = selected_model_name.replace("/", "_")
    adapter_path = f"./models/{sane_model_name}_{args.language}_{args.domain}"

    # Path to your dev/test JSONL
    predict_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_test_{args.task}.jsonl"
    print(f"Downloading prediction data from: {predict_url}")

    # Load the raw dev data
    predict_raw = load_jsonl_url(predict_url)
    print(f"Loading test data from: {predict_url}")

    # Convert raw data -> inference prompts JSONL
    convert_prediction_data(
        predict_raw,
        categories_for_this_domain,
        predict_data_path,
        selected_model_name,
        args.language,
        args.domain
    )
    print("--- Data Preparation Complete ---")

    # --- 2) Load tokenizer + base model ---
    tokenizer = AutoTokenizer.from_pretrained(selected_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        selected_model_name,
        dtype=torch.float16,
        device_map="auto"
    )

    # Ensure tokenizer has a pad token; common trick for decoder-only models
    tokenizer.pad_token = tokenizer.eos_token

    # Left padding is preferred for generation with decoder-only models in batches
    tokenizer.padding_side = "left"

    # --- 3) Load LoRA adapters ---
    print(f"\nLoading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # --- 4) Load Prediction Prompts ---
    print(f"Loading test data from: {predict_data_path}")
    test_data = load_jsonl_file(predict_data_path)

    final_submissions = []

    # Extract prompt texts + ids + raw texts
    all_prompts = [x["prompt_text"] for x in test_data]
    all_ids = [x["ID"] for x in test_data]
    all_texts = [x["raw_text"] for x in test_data]

    # Tokenize only to compute prompt lengths (for batching by similar lengths)
    prompt_lengths = [len(tokenizer(p, truncation=True)["input_ids"]) for p in all_prompts]

    # Combine everything into one list of examples
    examples = list(zip(all_prompts, all_ids, all_texts, prompt_lengths))

    # Sort by prompt length so each batch has similar length -> less padding -> faster
    examples.sort(key=lambda x: x[3])

    # Create batches of size args.batch_size
    batch_size = args.batch_size
    batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]

    # --- 5) Inference loop ---
    for batch in tqdm(batches, desc="Generating outputs"):
        batch_prompts = [x[0] for x in batch]
        batch_ids     = [x[1] for x in batch]
        batch_texts   = [x[2] for x in batch]

        # Tokenize with padding so tensors align in batch
        inputs = tokenizer(
            text=batch_prompts,
            return_tensors="pt",
            padding=True,      # minimal padding due to length-sorting
            truncation=True,
        ).to(model.device)

        # Deterministic generation (do_sample=False)
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        # Only decode the generated continuation (exclude the prompt tokens)
        gen = output[:, inputs.input_ids.shape[1]:]
        batch_outputs = tokenizer.batch_decode(gen, skip_special_tokens=True)

        # Store raw outputs for post-processing
        for j, raw_output in enumerate(batch_outputs):
            final_submissions.append({
                "id": batch_ids[j],
                "text": batch_texts[j],
                "raw_output": raw_output
            })

    # --- 6) Post-processing / format normalization ---
    processed = []

    for item in final_submissions:
        sample_id       = item["id"]
        raw_review_text = item["text"]
        raw_output      = item["raw_output"]

        # Attempt to recover JSON list from model output
        parsed_quads = extract_json_list(raw_output)

        final_quads = []
        seen = set()  # dedupe based on (aspect, category, opinion)

        for quad in parsed_quads:
            if not isinstance(quad, dict):
                continue

            # Extract required fields (fallback to "NULL")
            aspect  = quad.get("Aspect", "NULL")
            opinion = quad.get("Opinion", "NULL")

            # Map predicted spans back to exact raw text spans (helps scoring)
            if aspect != "NULL":
                aspect = get_original_span(raw_review_text, aspect)

            if opinion != "NULL":
                opinion = get_original_span(raw_review_text, opinion)

            # Skip incomplete predictions
            if aspect == "NULL" or opinion == "NULL":
                continue

            # Dedupe identical tuples
            key = (aspect, quad.get("Category"), opinion)
            if key in seen:
                continue
            seen.add(key)

            # Validate VA numeric values; otherwise write "NULL#NULL"
            v = quad.get("Valence")
            a = quad.get("Arousal")
            va_str = "NULL#NULL"

            if isinstance(v, (int, float)) and isinstance(a, (int, float)):
                if 1 <= v <= 9 and 1 <= a <= 9:
                    va_str = f"{v:.2f}#{a:.2f}"

            # Final normalized format expected by your evaluator/submission
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

    # --- 7) Save output JSONL ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nInference complete! File: {output_file}")

if __name__ == "__main__":
    main()
