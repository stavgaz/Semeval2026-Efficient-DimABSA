import argparse
import json
import requests
import time
import re
from tqdm import tqdm
from collections import Counter, defaultdict
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
    """Yield list chunks of length `size`."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ============================================================
# Prompt 
# ============================================================

def build_prompt(batch, language="eng"):
    """
    Build a strict "translate dataset" prompt that asks the model to return JSON only.

    Potential issue:
      - If language == "eng", `p` is never set in your current code,
        which would raise UnboundLocalError. (You default to "rus", so you
        may never hit it, but it's a landmine if someone passes --language eng.)
    """
    if language == "rus":
        p = 'Russian'
    elif language == 'tat':
        p = 'Tatar'
    elif language == 'ukr':
        p = 'Ukrainian'
    elif language == 'zho':
        p = 'Chinese'
    elif language == 'jpn':
        p = 'Japanese'

    return (
        f"You are translating a sentiment annotation dataset from {p} to English.\n"
        "This is a STRICT annotation task, not free translation.\n"
        "Follow ALL rules EXACTLY.\n\n"

        "OUTPUT FORMAT RULES (MANDATORY):\n"
        "1. Output ONLY valid JSON. No explanations. No extra text.\n"
        "2. Return the SAME JSON objects in the SAME order.\n"
        "3. Do NOT add, remove, rename, or reorder any fields.\n\n"

        "FIELD TRANSLATION RULES (MANDATORY):\n"
        "4. You MUST translate ALL of the following fields into English EVERY TIME:\n"
        "   - Text\n"
        "   - Quadruplet[].Aspect (EXCEPT when the value is exactly 'NULL')\n"
        "   - Quadruplet[].Opinion (EXCEPT when the value is exactly 'NULL')\n"
        f"5. NEVER leave Aspect or Opinion in {p}.\n\n"

        "NULL HANDLING RULES (CRITICAL):\n"
        "6. If Quadruplet[].Aspect is exactly 'NULL', keep it exactly as 'NULL'. Do NOT invent, infer, or replace it.\n"
        "7. If Quadruplet[].Opinion is exactly 'NULL', keep it exactly as 'NULL'. Do NOT invent or infer an opinion.\n\n"

        "ASPECT RULES (VERY IMPORTANT):\n"
        "8. Translate Aspect terms LITERALLY.\n"
        "   - Do NOT paraphrase.\n"
        "   - Do NOT normalize.\n"
        "   - Do NOT substitute synonyms.\n"
        "   - Do NOT reinterpret dish names or concepts.\n"
        "9. If Quadruplet[].Aspect is NOT 'NULL', the translated Aspect MUST appear verbatim in the translated Text.\n\n"

        "OPINION RULES (VERY IMPORTANT):\n"
        "10. Translate Opinions as literally as possible.\n"
        "11. Preserve sentiment polarity and intensity EXACTLY.\n"
        "12. Do NOT paraphrase, reinterpret, or generalize opinions.\n\n"

        "FINAL CHECK:\n"
        f"13. Ensure NO {p} characters remain (except 'NULL').\n"
        "14. Only then output the JSON.\n\n"

        "Translate the following JSON:\n"
        + json.dumps(batch, ensure_ascii=False)
    )


# ============================================================
# Robust JSON extractor
# ============================================================

def extract_json(text):
    """
    Extract the first valid JSON object/list from model output.

    Potential issues:
      - If the model emits non-JSON text before/after, extraction might fail.
      - If the output is a JSON list but model returns extra commentary, attempt 1 may miss it.
      - This tries a second pass that strips trailing commas (common LLM failure).
    """
    if not text:
        raise ValueError("Empty output")

    # Attempt 1: bracket-stack extraction from first '{' or '['
    try:
        start = min(i for i in [text.find("["), text.find("{")] if i != -1)
        stack = []
        for i in range(start, len(text)):
            c = text[i]
            if c in "[{":
                stack.append(c)
            elif c in "]}":
                # Potential issue: pop on empty stack would error, but we are in try/except.
                stack.pop()
                if not stack:
                    return json.loads(text[start:i + 1])
    except Exception:
        pass

    # Attempt 2: relaxed cleanup (remove trailing commas)
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
    Runs generation using the loaded HF model/tokenizer.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,   # Potential issue: may be too small for large batches; or too big for RAM.
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
    parser.add_argument('--base_url', type=str, default="https://cdn.jsdelivr.net/gh/DimABSA/DimABSA2026@main/task-dataset/track_a/", help="Base URL for data")
    args = parser.parse_args()

    # Output file path
    output = f"./translated_{args.language}_{args.domain}_final.jsonl"

    # Input training 
    train_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_train_alltasks.jsonl"
    print(f"Loading training data from: {train_url}")

    train_raw = read_jsonl_from_url(train_url)

    # Batch the data for translation
    batches = list(chunked(train_raw, args.batch_size))

    model_source = args.model
    print(f"Loading model from Hugging Face (online): {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,   # Potential issue: executes remote model code; only use trusted repos.
    )

    # Left padding is typical for decoder-only batch generation
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16,  # Potential issue: if GPU doesn't support fp16 well, adjust.
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # ============================================================
    # Translation loop
    # ============================================================

    ID_FIELD = 'ID'
    out = []

    for batch_idx, batch in enumerate(tqdm(batches, desc="Translating", unit="batch"), 1):
        # Mask IDs before sending to model (so it doesn't "translate" them)
        masked_batch = []
        ids = []
        for obj in batch:
            o = copy.deepcopy(obj)
            ids.append(o.get(ID_FIELD))
            o.pop(ID_FIELD, None)
            masked_batch.append(o)

        prompt = build_prompt(masked_batch, args.language)
        success = False

        # Retry each batch up to 3 times if JSON parsing fails
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
                    
                if not isinstance(parsed, list):
                    print(f"[WARN] Batch {batch_idx}: expected a JSON list but got {type(parsed).__name__} (attempt {attempt+1})")
                    time.sleep(1)
                    continue
                
                if len(parsed) != len(batch):
                    print(f"[WARN] Batch {batch_idx}: length mismatch. input={len(batch)} output={len(parsed)} (attempt {attempt+1})")
                    time.sleep(1)
                    continue
                    
                for out_obj, orig_id in zip(parsed, ids):
                    out_obj[ID_FIELD] = orig_id

                # Ensure key sets match original objects (strong schema check)
                for inp_obj, out_obj in zip(batch, parsed):
                    if set(out_obj.keys()) != set(inp_obj.keys()):
                        raise ValueError(f"Keys mismatch after restore for ID={inp_obj.get(ID_FIELD)}")

                out.extend(parsed)
                success = True
                break

            except Exception as e:
                print(f"[WARN] Batch {batch_idx}: generation failure (attempt {attempt+1}) → {e}")
                time.sleep(2)

        if not success:
            print(f"[SKIP] Batch {batch_idx} permanently skipped")
            continue

        # Periodic checkpoint write to avoid losing work
        if batch_idx % 10 == 0:
            write_jsonl(output, out)

    # Final write
    write_jsonl(output, out)
    print("DONE:", output)
