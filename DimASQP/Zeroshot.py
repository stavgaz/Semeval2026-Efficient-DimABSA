import json
import requests
import argparse
import os
import re
import torch
from typing import List, Dict, Tuple, Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import unicodedata


# ============================================================
# Category definitions (ENTITY#ATTRIBUTE) per domain
# ============================================================

def combine_lists(list1, list2):
    """Create all combinations s1#s2 and return (dict mapping -> index, list)."""
    combinations = [f"{s1}#{s2}" for s1 in list1 for s2 in list2]
    result_dict = {combo: idx for idx, combo in enumerate(combinations)}
    return result_dict, combinations

laptop_entity_labels = ['LAPTOP', 'DISPLAY', 'KEYBOARD', 'MOUSE', 'MOTHERBOARD', 'CPU', 'FANS_COOLING', 'PORTS', 'MEMORY', 'POWER_SUPPLY', 'OPTICAL_DRIVES', 'BATTERY', 'GRAPHICS', 'HARD_DISK', 'MULTIMEDIA_DEVICES', 'HARDWARE', 'SOFTWARE', 'OS', 'WARRANTY', 'SHIPPING', 'SUPPORT', 'COMPANY'] + ['OUT_OF_SCOPE']
laptop_attribute_labels = ['GENERAL', 'PRICE', 'QUALITY', 'DESIGN_FEATURES', 'OPERATION_PERFORMANCE', 'USABILITY', 'PORTABILITY', 'CONNECTIVITY', 'MISCELLANEOUS']
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
    """Download JSONL data from a URL and parse lines into dicts."""
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load JSONL from a local file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ============================================================
# Prompt template loading
# ============================================================

def load_prompt(lang, domain, categories, is_train=True):
    """
    Loads prompt templates from zeroshot_prompts3.jsonl (must be a single JSON object),
    and injects the allowed categories list.
    """
    with open("./zeroshot_prompts3.jsonl", "r") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]

    # Here you always use infer_prompt (zero-shot inference)
    prompt = p["infer_prompt"].replace("{CATEGORIES}", str(categories))
    return prompt


# ============================================================
# Build inference prompt (Qwen chat template)
# ============================================================

def create_prediction_prompt(data_sample: Dict, category_list: List[str], model_name: str, language: str = "eng", domain: str = 'res') -> str:
    """
    Builds a model chat prompt that asks the model to output JSON quadruplets.
    Uses Qwen's <|im_start|> chat tokens.
    """
    possible_categories = ", ".join(f'"{cat}"' for cat in category_list)
    instruction = load_prompt(language, domain, possible_categories, is_train=False)

    text = data_sample['Text'].replace("` ` ", "")

    # IMPORTANT: initialize to avoid UnboundLocalError
    prompt = None

    if "qwen" in model_name.lower():
        # Each language uses a localized "Review:" label
        if language == 'eng':
            prompt = f"""<|im_start|>user

{instruction}

Review: \"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == 'zho':
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
    if prompt is None:
        raise ValueError(f"Unsupported model or language: model={model_name}, language={language}")

    return prompt


# ============================================================
# Span alignment helper
# ============================================================

def get_original_span(full_text: str, predicted_span: str) -> str:
    """Map predicted span back to exact substring in full_text using normalization."""
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


# ============================================================
# Build intermediate prompt JSONL
# ============================================================

def convert_prediction_data(raw_data: List[Dict], category_list: List[str], output_file: str, model_name: str, language: str = "eng", domain: str = 'res'):
    """
    Writes a prompt JSONL with:
      - ID
      - prompt_text (model input)
      - raw_text (original text for span alignment)
    """
    new_dataset = []
    for data_sample in raw_data:
        try:
            prompt_string = create_prediction_prompt(data_sample, category_list, model_name, language, domain)
            new_dataset.append({
                "ID": data_sample['ID'],
                "prompt_text": prompt_string,
                "raw_text": data_sample['Text'].replace("` ` ", ""),
            })
        except KeyError:
            pass
        except Exception as e:
            print(f"Error processing sample: {e}\nSample ID: {data_sample.get('ID', 'Unknown')}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Data conversion complete! File saved to {output_file}.")
    print(f"Total prompts generated: {len(new_dataset)}")


# ============================================================
# Model output JSON recovery
# ============================================================

def extract_json_list(text: str) -> List[dict]:
    """Extract a JSON list from model output; return [] if not recoverable."""
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

    return []


# ============================================================
# Model selection
# ============================================================

def get_model_name_for_language(language_code: str) -> str:
    """Pick the base model. Here you use Qwen 2.5 14B for all languages."""
    lang = language_code.lower().strip()
    model_map = {
        "eng": "Qwen/Qwen2.5-14B-Instruct",
        "zho": "Qwen/Qwen2.5-14B-Instruct",
        "jpn": "Qwen/Qwen2.5-14B-Instruct",
        "rus": "Qwen/Qwen2.5-14B-Instruct",
        "ukr": "Qwen/Qwen2.5-14B-Instruct",
        "tat": "Qwen/Qwen2.5-14B-Instruct",
    }
    return model_map.get(lang, "Qwen/Qwen2.5-14B-Instruct")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Zero-shot inference (Qwen) for DimABSA.")
    parser.add_argument('--domain', type=str, default='restaurant')
    parser.add_argument('--language', type=str, default='zho')
    parser.add_argument('--subtask', type=str, default='subtask_3')
    parser.add_argument('--task', type=str, default='task3')
    parser.add_argument('--base_url', type=str, default="https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a")
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    # Get category list for this domain
    domain_key = domain_mapping.get(args.domain, args.domain)
    categories_for_this_domain = category_map[domain_key][1]

    # Where we store intermediate prompts + final predictions
    output_dir = f"./zeroshot"
    predict_data_path = f"{output_dir}/{args.language}_{args.domain}_prediction_prompts.jsonl"
    output_file = f"{output_dir}/pred_{args.language}_{args.domain}.jsonl"

    selected_model_name = get_model_name_for_language(args.language)

    # Download dev data for this task from the shared dataset repo
    predict_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_dev_{args.task}.jsonl"
    print(f"Downloading prediction data from: {predict_url}")
    predict_raw = load_jsonl_url(predict_url)

    # Build prompt JSONL
    convert_prediction_data(
        predict_raw,
        categories_for_this_domain,
        predict_data_path,
        selected_model_name,
        args.language,
        args.domain
    )

    # Load tokenizer + base model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(selected_model_name)

    # Use torch_dtype for compatibility across transformers versions
    model = AutoModelForCausalLM.from_pretrained(
        selected_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    # Load prompts to run generation
    test_data = load_jsonl_file(predict_data_path)

    all_prompts = [x["prompt_text"] for x in test_data]
    all_ids = [x["ID"] for x in test_data]
    all_texts = [x["raw_text"] for x in test_data]

    # Sort by prompt length to minimize padding
    prompt_lengths = [len(tokenizer(p, truncation=True)["input_ids"]) for p in all_prompts]
    examples = list(zip(all_prompts, all_ids, all_texts, prompt_lengths))
    examples.sort(key=lambda x: x[3])

    batches = [examples[i:i + args.batch_size] for i in range(0, len(examples), args.batch_size)]

    final_submissions = []

    for batch in tqdm(batches, desc="Generating outputs"):
        batch_prompts = [x[0] for x in batch]
        batch_ids = [x[1] for x in batch]
        batch_texts = [x[2] for x in batch]

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

    # Parse model outputs into final JSONL submission format
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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Inference complete! File: {output_file}")


if __name__ == "__main__":
    main()
