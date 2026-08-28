"""Build a 30-question HMMT dataset from MathArena/hmmt_feb_{2024,2025} (15 each).

Output: <repo>/data/byzantine/hmmt/hmmt_30q.json
Schema matches the existing hmmt_20q.json (question_id / question /
correct_answer / source).

Set HF_HOME if you want to control where the dataset cache lands so nothing
is written outside the project directory.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path


from datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/byzantine/hmmt/hmmt_30q.json"
SEED = 20260522
PER_YEAR = 15
SOURCES = [("MathArena/hmmt_feb_2024", "hmmt_feb_2024"),
           ("MathArena/hmmt_feb_2025", "hmmt_feb_2025")]


def main() -> None:
    rng = random.Random(SEED)
    items = []
    for hf_name, tag in SOURCES:
        ds = load_dataset(hf_name, split="train")
        rows = list(ds)
        rng.shuffle(rows)
        for row in rows[:PER_YEAR]:
            items.append({
                "question_id": f"{tag}_{row['problem_idx']:02d}",
                "question": row["problem"],
                "correct_answer": str(row["answer"]).strip(),
                "source": tag,
            })

    assert len(items) == 30, len(items)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(items)} questions to {OUT}")
    for it in items:
        q = it["question"][:70].replace("\n", " ")
        print(f"  {it['question_id']:22s} ans={it['correct_answer']:10s} | {q}...")


if __name__ == "__main__":
    main()
