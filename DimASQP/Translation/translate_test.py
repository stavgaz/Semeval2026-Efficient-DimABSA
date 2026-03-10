import argparse
import json
import requests
import time
import re
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import copy


# ============================================================
# I/O helpers
# ============================================================

def read_jsonl_from_url(url):
    """Download JSONL from a URL and parse each line into a dict."""
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def write_jsonl(path, rows):
    """Write a list[dict] to JSONL on disk."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chunked(lst, size):
    """Yield consecutive chunks of `lst` of length `size`."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ============================================================
# Prompt
# ============================================================

def build_prompt(batch, language="eng"):
    """
    Build a strict translation prompt for *prediction* (dev/test) data.

    Here we only translate the field:
      - Text

    The output must preserve the same JSON objects and order.
    """
    if language == "rus":
        p = "Russian"
    elif language == "tat":
        p = "Tatar"
    elif language == "ukr":
        p = "Ukrainian"
    elif language == "zho":
        p = "Chinese"
    elif language == "jpn":
        p = "Japanese"
    else:
        raise ValueError(f"Unsupported language code: {language}")

    return (
        f"You are translating a sentiment annotation dataset from {p} to English.\n"
        "This is a STRICT annotation task, not free translation.\n"
        "Follow ALL rules EXACTLY.\n\n"

        "OUTPUT FORMAT RULES (MANDATORY):\n"
        "1. Output ONLY valid JSON. No explanations. No extra text.\n"
        "2. Return the SAME JSON objects in the SAME order.\n"
        "3. Do NOT add, remove, rename, or reorder any fields.\n\n"

        "FIELD TRANSLATION RULES (MANDATORY):\n"
        "4. You MUST translate the field:\n"
        "   - Text\n\n"

        "FINAL CHECK:\n"
        f"5. Ensure NO {p} characters remain.\n"
        "6. Only then output the JSON.\n\n"

        "Translate the following JSON:\n"
        + json.dumps(batch, ensure_ascii=False)
    )


# ============================================================
# Robust JSON extractor
# ============================================================

def extract_json(text):
    """Extract the first valid JSON object/list from model output."""
    if not text:
        raise ValueError("Empty output")

    # Attempt 1: bracket-stack extraction from the first '{' or '['
    try:
        start = min(i for i in [text.find("["), text.find("{")] if i != -1)
        stack = []
        for i in range(start, len(text)):
            c = text[i]
            if c in "[{":
                stack.append(c)
            elif c in "]}":
                stack.pop()
                if not stack:
                    return json.loads(text[start:i + 1])
    except Exception:
        pass

    # Attempt 2: cleanup trailing commas (common LLM JSON failure mode)
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        start = min(i for i in [cleaned.find("["), cleaned.find("{")] if i != -1)
        return json.loads(cleaned[start:])
    except Exception as e:
        raise ValueError("Model returned invalid JSON") from e


# ============================================================
# Generation wrapper
# ============================================================

def translate(prompt, model_name=None):
    """
    Generate translation using the loaded HF model/tokenizer.
    `model_name` is unused; kept so the call site stays simple.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.0,
            top_p=0.1,
            do_sample=False
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="rus")
    parser.add_argument("--domain", default="restaurant")
    parser.add_argument("--subtask", default="subtask_3")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://cdn.jsdelivr.net/gh/DimABSA/DimABSA2026@main/task-dataset/track_a/",
        help="Base URL for data"
    )
    args = parser.parse_args()

    # Output file
    output = f"./translated_for_pred_{args.language}_{args.domain}_final.jsonl"

    # Input prediction data from URL (dev_task3 as requested)
    pred_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_dev_task3.jsonl"
    print(f"Loading prediction data from: {pred_url}")
    pred_raw = read_jsonl_from_url(pred_url)

    # Create batches
    batches = list(chunked(pred_raw, args.batch_size))

    # ------------------------------------------------------------
    # ONLINE MODEL LOADING
    # ------------------------------------------------------------
    model_source = args.model
    print(f"Loading model from Hugging Face (online): {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    ID_FIELD = "ID"
    out = []

    for batch_idx, batch in enumerate(tqdm(batches, desc="Translating", unit="batch"), 1):
        # Mask IDs before sending to the model so the model doesn't change them.
        # For prediction/dev data we assume each object only has {ID, Text}.
        masked_batch = []
        ids = []
        for obj in batch:
            o = copy.deepcopy(obj)
            ids.append(o.get(ID_FIELD))
            o.pop(ID_FIELD, None)
            masked_batch.append(o)

        prompt = build_prompt(masked_batch, args.language)
        success = False

        for attempt in range(3):
            try:
                translated = translate(prompt, model)

                try:
                    parsed = extract_json(translated)
                except Exception:
                    parsed = None

                if not parsed:
                    print(f"[WARN] Batch {batch_idx}: invalid JSON (attempt {attempt+1})")
                    time.sleep(1)
                    continue

                # We expect a list of objects back
                if not isinstance(parsed, list):
                    print(f"[WARN] Batch {batch_idx}: expected JSON list but got {type(parsed).__name__} (attempt {attempt+1})")
                    time.sleep(1)
                    continue

                # Ensure the model returned one output object per input object
                if len(parsed) != len(batch):
                    print(f"[WARN] Batch {batch_idx}: length mismatch. input={len(batch)} output={len(parsed)} (attempt {attempt+1})")
                    time.sleep(1)
                    continue

                # Restore IDs
                for out_obj, orig_id in zip(parsed, ids):
                    out_obj[ID_FIELD] = orig_id

                # For prediction translation, we enforce only that ID exists and Text exists.
                # (We don't enforce key sets beyond that, because the input is minimal.)
                for out_obj in parsed:
                    if "Text" not in out_obj:
                        raise ValueError("Missing 'Text' in translated output object")
                    if ID_FIELD not in out_obj:
                        raise ValueError("Missing 'ID' after restore")

                out.extend(parsed)
                success = True
                break

            except Exception as e:
                print(f"[WARN] Batch {batch_idx}: generation failure (attempt {attempt+1}) → {e}")
                time.sleep(2)

        if not success:
            print(f"[SKIP] Batch {batch_idx} permanently skipped")
            continue

        if batch_idx % 10 == 0:
            write_jsonl(output, out)

    # Final write
    write_jsonl(output, out)
    print("DONE:", output)
