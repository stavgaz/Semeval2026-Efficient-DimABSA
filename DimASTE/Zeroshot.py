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
# Data loading helpers
# ============================================================

def load_jsonl_url(url: str) -> List[Dict]:
    """Download JSONL from a URL and parse each line."""
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load JSONL from disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ============================================================
# Prompt template loading (zero-shot)
# ============================================================

def load_prompt(lang, domain, is_train=True):
    """
    Load the prompt template for (lang, domain) from zeroshot_prompts2.jsonl.

    NOTE: despite .jsonl extension, json.load() expects ONE JSON object in the file.
    """
    with open("./zeroshot_prompts2.jsonl", "r") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]

    # For this script we always use infer_prompt
    return p["infer_prompt"]


# ============================================================
# Prompt creation (Qwen chat format)
# ============================================================

def create_prediction_prompt(data_sample: Dict, model_name: str, language: str = "eng", domain: str = "res") -> str:
    """
    Build a Qwen chat prompt that asks the model to produce triplets as JSON.

    The instruction is loaded from zeroshot_prompts2.jsonl and then the review text is appended.
    """
    instruction = load_prompt(language, domain, is_train=False)
    text = data_sample["Text"].replace("` ` ", "")

    # Initialize so we don't return an undefined variable
    prompt = None

    if "qwen" in model_name.lower():
        if language == "eng":
            prompt = f"""<|im_start|>user

{instruction}

Review: \"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == "zho":
            prompt = f"""<|im_start|>user
{instruction}

評論: \"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == "jpn":
            prompt = f"""<|im_start|>user
{instruction}

レビュー本文：\"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == "ukr":
            prompt = f"""<|im_start|>user
{instruction}

Текст відгуку: \"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == "tat":
            prompt = f"""<|im_start|>user
{instruction}

Текст бәяләмәсе \"{text}\"
<|im_end|>
<|im_start|>assistant
"""
        elif language == "rus":
            prompt = f"""<|im_start|>user
{instruction}

Отзыв: \"{text}\"
<|im_end|>
<|im_start|>assistant
"""

    if prompt is None:
        raise ValueError(f"Unsupported model/language combination: model={model_name}, language={language}")

    return prompt


# ============================================================
# Span alignment helper
# ============================================================

def get_original_span(full_text: str, predicted_span: str) -> str:
    """Map predicted span back to exact substring in original text."""
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

def convert_prediction_data(raw_data: List[Dict], output_file: str, model_name: str, language: str = "eng", domain: str = "res"):
    """
    Convert raw dev/test samples to a prompt JSONL with:
      - ID
      - prompt_text (model input)
      - raw_text (original review)
    """
    new_dataset = []
    for data_sample in raw_data:
        try:
            prompt_string = create_prediction_prompt(data_sample, model_name, language, domain)
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


# ============================================================
# JSON recovery from model output
# ============================================================

def extract_json_list(text: str) -> List[dict]:
    """Recover a JSON list from model output; return [] if not possible."""
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
# Model selection (Qwen everywhere)
# ============================================================

def get_model_name_for_language(language_code: str) -> str:
    """Pick Qwen 2.5 14B Instruct for all languages."""
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
# Main inference
# ============================================================

def main():
    """
    Pipeline:
      1) Download dev data from base_url
      2) Convert each sample into a chat prompt JSONL
      3) Load Qwen model + tokenizer
      4) Generate outputs in batches (sorted by prompt length)
      5) Parse outputs into triplets (Aspect, Opinion, VA)
      6) Save JSONL submission
    """
    parser = argparse.ArgumentParser(description="Zero-shot inference (Qwen) for DimABSA subtask_2.")
    parser.add_argument("--domain", type=str, default="restaurant")
    parser.add_argument("--language", type=str, default="zho")
    parser.add_argument("--subtask", type=str, default="subtask_2")
    parser.add_argument("--task", type=str, default="task2")
    parser.add_argument("--base_url", type=str, default="https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    output_dir = f"./zeroshot"
    predict_data_path = f"{output_dir}/{args.language}_{args.domain}_prediction_prompts.jsonl"
    output_file = f"{output_dir}/pred_{args.language}_{args.domain}.jsonl"

    # Model used for inference
    selected_model_name = get_model_name_for_language(args.language)

    # Download dev data
    predict_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_dev_{args.task}.jsonl"
    print(f"Downloading prediction data from: {predict_url}")
    predict_raw = load_jsonl_url(predict_url)

    # Create prompts file
    convert_prediction_data(predict_raw, predict_data_path, selected_model_name, args.language, args.domain)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(selected_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        selected_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    # Load prompts
    test_data = load_jsonl_file(predict_data_path)

    # Sort by prompt length for efficient batching
    all_prompts = [x["prompt_text"] for x in test_data]
    all_ids = [x["ID"] for x in test_data]
    all_texts = [x["raw_text"] for x in test_data]

    prompt_lengths = [len(tokenizer(p, truncation=True)["input_ids"]) for p in all_prompts]
    examples = list(zip(all_prompts, all_ids, all_texts, prompt_lengths))
    examples.sort(key=lambda x: x[3])

    batches = [examples[i:i + args.batch_size] for i in range(0, len(examples), args.batch_size)]

    # Generation loop
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

    # Post-process into triplet format
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

            # For triplet: dedupe based on (Aspect, Opinion)
            key = (aspect, opinion)
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
                "Opinion": opinion,
                "VA": va_str
            })

        processed.append({
            "ID": sample_id,
            "Text": raw_review_text,
            "Triplet": final_quads
        })

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Inference complete! File: {output_file}")


if __name__ == "__main__":
    main()
