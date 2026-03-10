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

def read_jsonl_from_url(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# ============================================================
# Prompt (UNCHANGED – exactly yours)
# ============================================================

def build_prompt(batch, language="eng"):

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

def extract_json(text):
    if not text:
        raise ValueError("Empty output")

    # Attempt 1: strict extraction
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

    # Attempt 2: relaxed cleanup (remove trailing commas)
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        start = min(i for i in [cleaned.find("["), cleaned.find("{")] if i != -1)
        return json.loads(cleaned[start:])
    except Exception as e:
        raise ValueError("Model returned invalid JSON") from e

# ============================================================
# Main
# ============================================================

def ollama_translate(prompt, model_name=None):
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="rus")
    parser.add_argument("--domain", default="restaurant")
    parser.add_argument("--subtask", default="subtask_3")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
                    "--data_root",
                    type=str,
                    default="/leonardo_work/EUHPC_D19_014/dimABSA/data/track_a",
                    help="Local root for track_a (offline). Layout: data_root/subtask/lang/*.jsonl",
                )
    args = parser.parse_args()

    output = f"/leonardo_work/EUHPC_D19_014/dimABSA/dimABSA/translated_{args.language}_{args.domain}_final.jsonl"
    train_path = os.path.join(
        args.data_root,
        args.subtask,
        args.language,
        f"{args.language}_{args.domain}_train_alltasks.jsonl",
    )
    print(f"Loading training data from: {train_path}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f if line.strip()]

    results = []

    out = []

    batches = list(chunked(train_raw, args.batch_size))

    hf_home = os.environ.get("HF_HOME", "")
    model_source = resolve_local_model_path(args.model, hf_home)
    print(f"Loading model from: {model_source}")


    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left" # Correct for batch generation

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    ID_FIELD = 'ID'

    for batch_idx, batch in enumerate(
        tqdm(batches, desc="Translating", unit="batch"), 1
    ):
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
                translated = ollama_translate(prompt, model)

                try:
                    parsed = extract_json(translated)
                except Exception as e:
                    parsed = None

                if not parsed:
                    print(f"[WARN] Batch {batch_idx}: invalid JSON (attempt {attempt+1})")
                    time.sleep(1)
                    continue

                # restore IDs
                for out_obj, orig_id in zip(parsed, ids):
                    out_obj[ID_FIELD] = orig_id

                # key set must match original
                for inp_obj, out_obj in zip(batch, parsed):
                    if set(out_obj.keys()) != set(inp_obj.keys()):
                        raise ValueError(f"Keys mismatch after restore for ID={inp_obj.get(ID_FIELD)}")

                out.extend(parsed)
                success = True
                break

            except Exception as e:
                print(f"[WARN] Batch {batch_idx}: ollama failure (attempt {attempt+1}) → {e}")
                time.sleep(2)

        if not success:
            print(f"[SKIP] Batch {batch_idx} permanently skipped")
            # OPTIONAL: save skipped batch for later inspection
            # skipped_batches.append(batch)
            continue

        if batch_idx % 10 == 0:
            write_jsonl(output, out)

    print("DONE:", output)
