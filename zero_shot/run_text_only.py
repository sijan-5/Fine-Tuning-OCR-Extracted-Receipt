import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    m = JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0)), None
        except json.JSONDecodeError as e:
            return None, f"regex-extracted but invalid: {e}"
    return None, "no JSON object found"


def load_completed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["image_id"])
            except Exception:
                pass
    return done


def build_messages(system_prompt: str, user_prompt: str, supports_system: bool):
    """Build chat messages, merging system into user if system role is unsupported."""
    if supports_system:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    merged = f"{system_prompt}\n\n{user_prompt}"
    return [{"role": "user", "content": merged}]


def detect_system_support(tok) -> bool:
    """Try rendering a system message; if it fails, we must merge."""
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ocr-column", required=True,
                   choices=["paddle_ocr_text", "tesseract_ocr_text"])
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--quantize", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    supports_system = detect_system_support(tok)
    print(f"System role supported: {supports_system}", flush=True)

    print(f"Loading model: {args.model}", flush=True)
    load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        del load_kwargs["dtype"]

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    print(f"Reading dataset: {args.csv}", flush=True)
    df = pd.read_csv(args.csv, sep=",", encoding="cp1252")
    df = df[df[args.ocr_column].notna() & df[args.ocr_column].astype(str).str.strip().ne("")]
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Total rows after filtering: {len(df)}", flush=True)

    done = load_completed_ids(out_path)
    if done:
        print(f"Resuming - skipping {len(done)} already-completed rows", flush=True)

    from zero_shot.prompts import SYSTEM_PROMPT, build_user_prompt

    with open(out_path, "a") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Receipts"):
            image_id = row["image_id"]
            if image_id in done:
                continue

            ocr_text = row.get(args.ocr_column, "")
            if not isinstance(ocr_text, str) or not ocr_text.strip():
                record = {
                    "image_id": int(image_id),
                    "ground_truth": row["ground_truth_data"],
                    "prediction": None,
                    "error": "empty OCR",
                    "raw_response": None,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                continue

            messages = build_messages(
                SYSTEM_PROMPT,
                build_user_prompt(ocr_text),
                supports_system,
            )
            inputs = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            response = tok.decode(
                out_ids[0][input_len:], skip_special_tokens=True
            )
            prediction, err = extract_json(response)

            record = {
                "image_id": int(image_id),
                "ground_truth": row["ground_truth_data"],
                "prediction": prediction,
                "error": err,
                "raw_response": response if err else None,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
