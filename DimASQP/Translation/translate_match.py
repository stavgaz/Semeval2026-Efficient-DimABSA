import os
import json
import argparse
import unicodedata
from typing import List, Dict, Any, Optional

import requests
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# JSONL loader (URL)
# -----------------------------
def read_jsonl_from_url(url: str) -> List[Dict[str, Any]]:
    """Download JSONL from a URL and parse each line as JSON."""
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]


# -----------------------------
# Robust JSON extractor
# -----------------------------
def extract_json(text: str) -> Optional[Any]:
    """
    Extract the first valid JSON object/array from a text blob.
    Returns dict/list or None.
    """
    if not text:
        return None

    text = text.strip()
    candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not candidates:
        return None
    start = min(candidates)

    stack = []
    for i in range(start, len(text)):
        c = text[i]
        if c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                return None
            stack.pop()
            if not stack:
                snippet = text[start: i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None


# -----------------------------
# Span matcher (your original, kept)
# -----------------------------
def get_original_span(full_text: str, predicted_span: Any) -> str:
    """Map model-chosen span back to an exact substring in the original text."""
    if predicted_span is None:
        return ""

    if not isinstance(predicted_span, str):
        predicted_span = str(predicted_span)

    if not full_text:
        return predicted_span

    if predicted_span in full_text:
        return predicted_span

    full_text_norm = unicodedata.normalize("NFKC", full_text)
    predicted_norm = unicodedata.normalize("NFKC", predicted_span)

    full_text_nosp = "".join(c for c in full_text_norm if not c.isspace())
    predicted_nosp = "".join(c for c in predicted_norm if not c.isspace())

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
            return full_text[start_real: end_real + 1]

    return predicted_span


# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(item: Dict[str, Any], language: str) -> str:
    """
    Given:
      - src_text: original non-English review text
      - quad: English quadruplet (Aspect/Opinion/Category/VA)
    Ask the model to pick the best matching spans from the original text.
    """
    lang_map = {
        "rus": "Russian",
        "tat": "Tatar",
        "ukr": "Ukrainian",
        "zho": "Chinese",
        "jpn": "Japanese",
    }
    p = lang_map.get(language, language)

    return (
        f"You are given a {p} review text and an English ABSA quadruplet.\n"
        f"Your task is to SELECT the closest matching spans from the ORIGINAL {p} text by MEANING.\n\n"
        "YOU MUST FOLLOW THESE RULES:\n"
        "1) Output ONLY valid JSON (no markdown, no extra text).\n"
        "2) Use keys exactly: Aspect, Opinion, Category, VA.\n"
        "3) Category and VA MUST be copied unchanged from the English quadruplet.\n"
        "4) Aspect and Opinion MUST be EXACT SUBSTRINGS copied from the original text (character-for-character).\n"
        "   - Do NOT translate into new words.\n"
        "   - Do NOT paraphrase.\n"
        "   - Copy directly from the text.\n"
        "5) Match by meaning (semantic match), even if different words from English.\n"
        "6) NEVER output empty strings. If the exact meaning is not present, choose the CLOSEST POSSIBLE span anyway.\n"
        "7) Prefer spans that are specific (not the whole sentence) but still meaningful.\n\n"
        f"{p} Text:\n{item['src_text']}\n\n"
        "English Quadruplet:\n"
        f"{json.dumps(item['quad'], ensure_ascii=False)}\n\n"
        "Return JSON now:\n"
    )


# -----------------------------
# Generation
# -----------------------------
@torch.inference_mode()
def model_generate_json(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 256,
) -> str:
    """Generate text continuation from prompt and return decoded string."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="rus")
    parser.add_argument("--domain", default="restaurant")
    parser.add_argument("--subtask", default="subtask_3")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)  # kept, but not used (per-item loop)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--verbose_errors", action="store_true")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://cdn.jsdelivr.net/gh/DimABSA/DimABSA2026@main/task-dataset/track_a/",
        help="Base URL for track_a JSONL",
    )
    args = parser.parse_args()

    # Output: final predictions where Aspect/Opinion are spans from original language text
    output_path = f"pred_{args.language}_{args.domain}_final.jsonl"

    # ------------------------------------------------------------
    # 1) Load original dev data (ID -> original-language Text) FROM URL
    # ------------------------------------------------------------
    dev_url = f"{args.base_url}/{args.subtask}/{args.language}/{args.language}_{args.domain}_dev_task3.jsonl"
    rows = read_jsonl_from_url(dev_url)

    # Map ID -> original-language text
    id2text: Dict[Any, str] = {r["ID"]: r["Text"] for r in rows}

    # ------------------------------------------------------------
    # 2) Load base model ONLINE (instead of local path resolver)
    # ------------------------------------------------------------
    model_source = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ------------------------------------------------------------
    # 3) Load English predictions produced earlier (local file)
    #    These contain English spans; we'll map them back to original spans.
    # ------------------------------------------------------------
    en_preds_path = f"pred_{args.language}_{args.domain}.jsonl"
    
    out_records: List[Dict[str, Any]] = []

    with open(en_preds_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Matching predictions"):
            pred = json.loads(line)
            src_text = id2text.get(pred["ID"], "")

            new_quads = []
            for quad in pred.get("Quadruplet", []):
                item = {"src_text": src_text, "quad": quad}

                fixed: Optional[Dict[str, Any]] = None

                for _ in range(args.retries):
                    try:
                        prompt = build_prompt(item, args.language)
                        resp = model_generate_json(
                            prompt,
                            tokenizer=tokenizer,
                            model=model,
                            max_new_tokens=args.max_new_tokens,
                        )

                        cand = extract_json(resp)

                        # Accept only dicts with required keys
                        if isinstance(cand, dict) and "Aspect" in cand and "Opinion" in cand and "Category" in cand and "VA" in cand:
                            fixed = cand
                            break
                    except Exception as e:
                        if args.verbose_errors:
                            print(f"[WARN] ID={pred.get('ID')} retry failed: {e}")

                # If matching fails, keep the original English quad
                if fixed is None:
                    fixed = quad

                # Anchor spans to the exact substring surface form in the original text
                fixed["Aspect"] = get_original_span(src_text, fixed.get("Aspect", ""))
                fixed["Opinion"] = get_original_span(src_text, fixed.get("Opinion", ""))

                new_quads.append(fixed)

            out_records.append({
                "ID": pred["ID"],
                "Text": src_text,
                "Quadruplet": new_quads,
            })

    # Write final JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
