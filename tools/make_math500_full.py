#!/usr/bin/env python3
"""Build the full MATH-500 test set (500 problems) in the byzantine schema.

weak_answer is left empty; fill it with tools/precompute_weak_answers.py so the
fixed-mode Byzantine agent has a plausible wrong answer to replay.

Usage:
    python tools/make_math500_full.py --out data/byzantine/math500/math500_500.json
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build the full MATH-500 test set")
    parser.add_argument("--out", default="data/byzantine/math500/math500_500.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for i, ex in enumerate(ds):
        items.append(
            {
                "question_id": f"math500_{i:03d}",
                "question": ex["problem"],
                "correct_answer": str(ex["answer"]).strip(),
                "weak_answer": "",
                "dataset_type": "math500",
                "level": ex.get("level", ""),
                "type": ex.get("subject", ""),
            }
        )
    if args.limit:
        items = items[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(f"wrote {len(items)} questions -> {out}")


if __name__ == "__main__":
    main()
