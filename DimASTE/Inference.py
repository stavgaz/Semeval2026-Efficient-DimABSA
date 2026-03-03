# ============================================================
# IMPORTS
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

# TRL config imported but NOT used in this inference script (not fatal, can remove)
from trl import SFTConfig

from datasets import load_dataset  # NOT used (not fatal, can remove)
from tqdm import tqdm
import unicodedata


# ============================================================
# Data loading helpers
# ============================================================

def load_jsonl_url(url: str) -> List[Dict]:
    """
    Download JSONL file from a URL and parse each line.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def load_jsonl_file(filepath: str) -> List[Dict]:
    """
    Load JSONL file from local disk and parse each line.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_prompt(lang, domain):
    """
    Load an inference prompt template from prompts2.jsonl (despite .jsonl, using json.load).
    This requires prompts2.jsonl to be a single JSON object, not JSONL.
    """
    with open("./prompts2.jsonl", "r") as f:
        prompts = json.load(f)

    key = f"{lang}_{domain}"
    p = prompts[key]
    prompt = p["prompt"]
    return prompt


# ============================================================
# Prompt creation (inference)
# ============================================================

def create_prediction_prompt(data_sample: Dict, model_name: str, language: str = "eng", domain: str = 'res') -> str:
    """
    Build a chat prompt that asks the model to output triplets/quads as JSON.

    NOTE: This function is model-family specific:
      - Llama uses <|begin_of_text|> / <|start_header_id|> ... and <|eot_id|>
      - Qwen uses <|im_start|> ... <|im_end|>

    FATAL RISK: For some (model_name, language) pairs, `prompt` might not be assigned.
    Consider setting prompt=None and raising if still None at the end.
    """
    instruction = load_prompt(language, domain)
    text = data_sample['Text'].replace("` ` ", "")

    # Llama prompt format (only English supported here)
    if "llama" in model_name.lower():
        if language == 'eng':
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Review: \"{text}\"<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

    # Qwen prompt format (multiple languages supported, but NO 'eng' branch)
    if "qwen" in model_name.lower():
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


# ============================================================
# Span alignment: map predicted string back to original substring
# ============================================================

def get_original_span(full_text: str, predicted_span: str) -> str:
    """
    Attempt to map a predicted span back to an exact substring in the original review.
    Helps evaluation if the scorer expects exact spans.
    """
    if not predicted_span:
        return predicted_span

    # 1) Exact match
    if predicted_span in full_text:
        return predicted_span

    # 2) Normalize full-width/half-width
    full_text_norm = unicodedata.normalize("NFKC", full_text)
    predicted_span_norm = unicodedata.normalize("NFKC", predicted_span)

    # 3) Remove whitespace for fuzzy matching
    full_text_nosp = "".join(c for c in full_text_norm if not c.isspace())
    predicted_nosp = "".join(c for c in predicted_span_norm if not c.isspace())

    # 4) Search case-sensitive then case-insensitive
    start_idx = full_text_nosp.find(predicted_nosp)
    if start_idx == -1:
        start_idx = full_text_nosp.lower().find(predicted_nosp.lower())

    # 5) Map back to original indices in normalized text
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

        # Return slice from the original raw text indices
        if start_real != -1 and end_real != -1:
            return full_text[start_real:end_real + 1]

    return predicted_span


# ============================================================
# Convert raw test data -> prompt JSONL for inference
# ============================================================

def convert_prediction_data(raw_data: List[Dict], output_file: str, model_name: str, language: str = "eng", domain: str = 'res'):
    """
    Create an intermediate JSONL file with:
      - ID
      - prompt_text (chat prompt)
      - raw_text (original review)
    """
    new_dataset = []
    for data_sample in raw_data:
        try:
            prompt_string = create_prediction_prompt(data_sample, model_name, language, domain)
            new_dataset.append({
                "ID": data_sample['ID'],
                "prompt_text": prompt_string,
                "raw_text": data_sample['Text'].replace("` ` ", ""),
            })
        except KeyError:
            # Missing ID/Text
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
# JSON extraction from model output
# ============================================================

def extract_json_list(text: str) -> List[dict]:
    """
    Try to recover the FIRST valid JSON list/dict from model output.
    Returns [] if nothing can be recovered.
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

    # Direct json.loads
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except:
        pass

    # Extract list-like substring
    matches = re.findall(r"\[[^\]]*?\]", repaired)
    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    # Extract dict-like substring(s)
    matches = re.findall(r"{.*?}", repaired)
    for m in matches:
        try:
            return [json.loads(m)]
        except:
            continue

    # Aggressive replacements for Python-like output
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

    # Fallback: first '[' to last ']'
    try:
        start = aggressive.find("[")
        end = aggressive.rfind("]")
        if start != -1 and end != -1 and end > start:
            segment = aggressive[start:end + 1]
            return json.loads(segment)
    except:
        pass

    print("EMPTY")
    return []


# ============================================================
# Model selection by language
# ============================================================

def get_model_name_for_language(language_code: str) -> str:
    """
    Map language code -> base HF model name.
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
# Main inference function
# ============================================================

def main():
    """
    Main inference routine:
      - build prompt file
      - load model + adapter
      - generate outputs in batches
      - post-process and save JSONL
    """
    parser = argparse.ArgumentParser(description="Run inference with a trained LoRA model.")
    parser.add_argument('--domain', type=str, default='restaurant')
    parser.add_argument('--language', type=str, default='tat')
    parser.add_argument('--subtask', type=str, default='subtask_2')
    parser.add_argument('--task', type=str, default='task2')
    parser.add_argument('--base_url', type=str, default="https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a")
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print("--- Starting Data Preparation for Inference ---")
    output_dir = f"./{args.subtask}/{args.language}"

    predict_data_path = f"{output_dir}/{args.language}_{args.domain}_prediction_prompts.jsonl"
    output_file = f"{output_dir}/pred_{args.language}_{args.domain}.jsonl"

    selected_model_name = get_model_name_for_language(args.language)
    sane_model_name = selected_model_name.replace("/", "_")
    adapter_path = f"./models/{sane_model_name}_{args.language}_{args.domain}"

    # Load data from URL:
    predict_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_dev_{args.task}.jsonl"
    print(f"Loading training data from: {predict_url}")
    predict_raw = load_jsonl_url(predict_url)

    # Build prompts jsonl
    convert_prediction_data(predict_raw, predict_data_path, selected_model_name, args.language, args.domain)
    print("--- Data Preparation Complete ---")

    # Load tokenizer + base model
    tokenizer = AutoTokenizer.from_pretrained(selected_model_name)

    # Prefer torch_dtype instead of dtype for HF compatibility
    model = AutoModelForCausalLM.from_pretrained(
        selected_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Ensure padding works for batch generation
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left padding is typical for decoder-only generation

    # Load LoRA adapter weights
    print(f"\nLoading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Load prompt file (intermediate)
    print(f"Loading test data from: {predict_data_path}")
    test_data = load_jsonl_file(predict_data_path)

    # Prepare batched prompts
    all_prompts = [x["prompt_text"] for x in test_data]
    all_ids = [x["ID"] for x in test_data]
    all_texts = [x["raw_text"] for x in test_data]

    prompt_lengths = [len(tokenizer(p, truncation=True)["input_ids"]) for p in all_prompts]
    examples = list(zip(all_prompts, all_ids, all_texts, prompt_lengths))
    examples.sort(key=lambda x: x[3])  # sort by prompt length for efficiency

    batch_size = args.batch_size
    batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]

    # ============================================================
    # Generation loop
    # ============================================================
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

        # Only decode the continuation (exclude prompt tokens)
        gen = output[:, inputs.input_ids.shape[1]:]
        batch_outputs = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for j, raw_output in enumerate(batch_outputs):
            final_submissions.append({
                "id": batch_ids[j],
                "text": batch_texts[j],
                "raw_output": raw_output
            })

    # ============================================================
    # Post-processing: parse JSON + normalize spans + validate VA
    # ============================================================
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

            # Skip if essential fields missing
            if aspect == "NULL" or opinion == "NULL":
                continue

            # Dedupe includes category, but you don't output category later (schema mismatch risk)
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

            # WARNING: Missing "Category" in output might break evaluator
            final_quads.append({
                "Aspect": aspect,
                # "Category": quad.get("Category"),  # Consider adding back
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
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nInference complete! File: {output_file}")

if __name__ == "__main__":
    main()
