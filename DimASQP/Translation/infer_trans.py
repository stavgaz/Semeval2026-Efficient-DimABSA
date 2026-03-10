import json
import requests
import argparse
import os
import random
import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel
from tqdm import tqdm
import unicodedata


# ============================================================
# Category definitions
# ============================================================

def combine_lists(list1, list2):
    """Create all '<ENTITY>#<ATTRIBUTE>' combinations and return dict + list."""
    combinations = [f"{s1}#{s2}" for s1 in list1 for s2 in list2]
    result_dict = {combo: idx for idx, combo in enumerate(combinations)}
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


# ============================================================
# Data loading helpers
# ============================================================

def load_jsonl_url(url: str) -> List[Dict]:
    """Download JSONL from a URL and parse each line."""
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load JSONL from local disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ============================================================
# Prompt template loading
# ============================================================

def load_prompt(lang, domain, categories, is_train=True):
    """
    Load prompt templates and inject the allowed category list.

    Expected prompts.jsonl format: a single JSON object with keys like "eng_restaurant".
    """
    with open("/leonardo_work/EUHPC_D19_014/dimABSA/dimABSA/prompts.jsonl", "r") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]

    if is_train:
        return p["train_prompt"].replace("{CATEGORIES}", str(categories))
    return p["infer_prompt"].replace("{CATEGORIES}", str(categories))


# ============================================================
# Prompt building (inference)
# ============================================================

def create_prediction_prompt(
    data_sample: Dict,
    category_list: List[str],
    model_name: str,
    language: str = "eng",
    domain: str = "res",
) -> str:
    """
    Build a chat prompt for inference.

    - category_list is inserted into the instruction (as allowed labels).
    - This version uses Llama chat formatting.
    - Instruction language is forced to English via load_prompt("eng", ...).
    """
    possible_categories = ", ".join(f"\"{cat}\"" for cat in category_list)
    instruction = load_prompt("eng", domain, possible_categories, is_train=False)

    text = data_sample["Text"].replace("` ` ", "")

    # Llama chat format: user block + assistant block begins
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Review: \"{text}\"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt


def get_original_span(full_text: str, predicted_span: str) -> str:
    """
    Map a predicted span back to the exact substring in the original review text.
    This helps when evaluation expects exact spans.
    """
    if not predicted_span:
        return predicted_span

    if predicted_span in full_text:
        return predicted_span

    full_text_norm = unicodedata.normalize("NFKC", full_text)
    predicted_span_norm = unicodedata.normalize("NFKC", predicted_span)

    full_text_nosp = "".join(c for c in full_text_norm if not c.isspace())
    predicted_nosp = "".join(c for c in predicted_span_norm if not c.isspace())

    start_idx = full_text_nosp.find(predicted_nosp)
    if start_idx == -1:
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
            return full_text[start_real:end_real + 1]

    return predicted_span


def convert_prediction_data(
    raw_data: List[Dict],
    category_list: List[str],
    output_file: str,
    model_name: str,
    language: str = "eng",
    domain: str = "res",
):
    """
    Convert raw dev/test JSONL into an intermediate prompt JSONL with:
      - ID
      - prompt_text
      - raw_text
    """
    new_dataset = []
    for data_sample in raw_data:
        try:
            prompt_string = create_prediction_prompt(
                data_sample, category_list, model_name, language, domain
            )
            new_dataset.append({
                "ID": data_sample["ID"],
                "prompt_text": prompt_string,
                "raw_text": data_sample["Text"].replace("` ` ", ""),
            })
        except KeyError:
            pass
        except Exception as e:
            print(f"Error processing sample: {e}\nSample ID: {data_sample.get('ID', 'Unknown')}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Data conversion complete! File saved to {output_file}.")
    print(f"Total prompts generated: {len(new_dataset)}")


def extract_json_list(text: str) -> List[dict]:
    """
    Try to recover a JSON list from model output.
    Returns [] if nothing valid can be recovered.
    """
    if not text or not isinstance(text, str):
        return []

    raw = text.strip()
    repaired = (
        raw.replace("“", '"').replace("”", '"')
           .replace("’", "'")
           .replace("\r", " ")
           .replace("\t", " ")
    )

    repaired = re.sub(r",(\s*[\]\}])", r"\1", repaired)

    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except:
        pass

    matches = re.findall(r"\[[^\]]*?\]", repaired)
    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    matches = re.findall(r"{.*?}", repaired)
    for m in matches:
        try:
            return [json.loads(m)]
        except:
            continue

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

    try:
        start = aggressive.find("[")
        end = aggressive.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(aggressive[start:end + 1])
    except:
        pass

    print("EMPTY")
    return []


def get_model_name_for_language(language_code: str) -> str:
    """Select the base model for a given language. Here: Llama 3.1 8B for all."""
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
# Main inference
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained LoRA model.")
    parser.add_argument("--domain", type=str, default="restaurant")
    parser.add_argument("--language", type=str, default="zho")
    parser.add_argument("--subtask", type=str, default="subtask_3")
    parser.add_argument("--task", type=str, default="task3")
    parser.add_argument("--base_url", type=str, default="https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--run_id", type=str, default="du19n6qg")
    args = parser.parse_args()

    # Domain -> category list (ENTITY#ATTRIBUTE)
    domain_key = domain_mapping.get(args.domain, args.domain)
    categories_for_this_domain = category_map[domain_key][1]

    # Output locations
    predict_data_path = f"./{args.language}_{args.domain}_prediction_prompts.jsonl"
    output_file = f"./pred_{args.language}_{args.domain}.jsonl"

    # Base model + adapter path
    selected_model_name = get_model_name_for_language(args.language)
    sane_model_name = selected_model_name.replace("/", "_")
    adapter_path = f"./{sane_model_name}_{args.language}_{args.domain}"

    # ------------------------------------------------------------
    # DATA LOADING FROM URL (requested change)
    # ------------------------------------------------------------
    # If you want dev data:  ..._dev_task3.jsonl
    # If you want test data: ..._test_task3.jsonl (depending on official naming)
    predict_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_dev_{args.task}.jsonl"
    print(f"Loading prediction data from URL: {predict_url}")
    predict_raw = load_jsonl_url(predict_url)

    # Build intermediate prompt file
    convert_prediction_data(
        predict_raw,
        categories_for_this_domain,
        predict_data_path,
        selected_model_name,
        args.language,
        args.domain,
    )

    # ------------------------------------------------------------
    # ONLINE MODEL LOADING
    # ------------------------------------------------------------
    # This will download from Hugging Face if not cached.
    print(f"Loading base model from Hugging Face: {selected_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(selected_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        selected_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load LoRA adapters on top of the base model
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Load prompt file
    test_data = load_jsonl_file(predict_data_path)

    # Sort prompts by length to reduce padding in batch generation
    all_prompts = [x["prompt_text"] for x in test_data]
    all_ids = [x["ID"] for x in test_data]
    all_texts = [x["raw_text"] for x in test_data]

    prompt_lengths = [len(tokenizer(p, truncation=True)["input_ids"]) for p in all_prompts]
    examples = list(zip(all_prompts, all_ids, all_texts, prompt_lengths))
    examples.sort(key=lambda x: x[3])

    batches = [examples[i:i + args.batch_size] for i in range(0, len(examples), args.batch_size)]

    # ------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------
    final_submissions = []

    for batch in tqdm(batches, desc="Generating outputs"):
        batch_prompts = [x[0] for x in batch]
        batch_ids     = [x[1] for x in batch]
        batch_texts   = [x[2] for x in batch]

        inputs = tokenizer(
            text=batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        gen = output[:, inputs.input_ids.shape[1]:]
        batch_outputs = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for j, raw_output in enumerate(batch_outputs):
            final_submissions.append({
                "id": batch_ids[j],
                "text": batch_texts[j],
                "raw_output": raw_output
            })

    # ------------------------------------------------------------
    # Post-processing: parse JSON + align spans + validate VA
    # ------------------------------------------------------------
    processed = []

    for item in final_submissions:
        sample_id = item["id"]
        raw_review_text = item["text"]
        raw_output = item["raw_output"]

        parsed_quads = extract_json_list(raw_output)

        final_quads = []
        seen = set()

        for quad in parsed_quads:
            if not isinstance(quad, dict):
                continue

            aspect = quad.get("Aspect", "NULL")
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

    # Save predictions
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Inference complete! File: {output_file}")


if __name__ == "__main__":
    main()
