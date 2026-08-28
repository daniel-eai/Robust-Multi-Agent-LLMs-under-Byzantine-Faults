#!/usr/bin/env python3
"""Fill weak_answer for a MATH-500 dataset file by querying the weak model.

The fixed-mode Byzantine agent replays this answer with confidence 1.0, which
matches the paper's threat model (an out-of-distribution weak-model response
presented with maximal confidence).

Usage:
    python tools/precompute_weak_answers.py \
        --data data/byzantine/math500/math500_500.json \
        --model qwen2.5-1.5b-instruct \
        --base-url http://127.0.0.1:8001/v1
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.api_model import APIModel

SYSTEM_PROMPT = (
    "You are a math expert. Solve the problem step by step. "
    "At the very end, write your final answer on its own line in this exact format:\n"
    "Answer: <number>\n"
    "Only write a single number (integer or decimal). Do not write words, units, or expressions."
)


def extract(text: str) -> str:
    m = re.search(r"^Answer\s*:\s*(-?\d+(?:\.\d+)?)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    boxed = re.findall(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text)
    if boxed:
        cleaned = re.sub(r"[^0-9.\-/]", "", boxed[-1].strip())
        return cleaned or boxed[-1].strip()
    m = re.search(r"[Aa]nswer\s*:\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


async def fill(model: APIModel, item: dict) -> dict:
    try:
        response = await model.generate_with_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ]
        )
        item["weak_answer"] = extract(response)
    except Exception as error:
        print(f"{item['question_id']}: failed ({error})", file=sys.stderr)
        item["weak_answer"] = ""
    return item


async def main_async(args):
    items = json.loads(Path(args.data).read_text())
    model = APIModel(
        args.model,
        api_base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_concurrency=args.concurrency,
    )
    try:
        filled = await asyncio.gather(*[fill(model, item) for item in items])
    finally:
        await model.close()

    out = Path(args.out or args.data)
    out.write_text(json.dumps(filled, ensure_ascii=False, indent=1))
    missing = sum(1 for item in filled if not item.get("weak_answer"))
    print(f"filled {len(filled) - missing}/{len(filled)} weak answers -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Precompute Byzantine replay answers")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=os.getenv("WEAK_MODELS", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.getenv("WEAK_API_BASE_URL"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
