#!/usr/bin/env python3
"""Build the commonsense evaluation sets in the byzantine dataset schema.

Shuffles each validation split with a fixed seed. Following the paper, HellaSwag
and BoolQ are subsampled to 1000 questions each; pass --no-cap to keep the whole
split.

Usage:
    python tools/make_commonsense.py --out-dir data/byzantine/commonsense
"""
import argparse
import json
import random
from pathlib import Path

SEED = 20260522
CAPS = {"hellaswag": 1000, "boolq": 1000}
PAPER_SUITE = ["arc_c", "hellaswag", "boolq", "obqa", "rte"]


def make_arc_c():
    from datasets import load_dataset

    rows = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation"))
    random.Random(SEED).shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        parts, idx_to_tag = [], {}
        for j, (lab, txt) in enumerate(zip(labels, texts)):
            tag = f"answer{j + 1}"
            idx_to_tag[lab] = tag
            parts.append(f"{tag}: {txt}")
        fmt = "/".join(f"answer{j + 1}" for j in range(len(texts)))
        q = (
            "Please choose the correct answer to the question: "
            f"{r['question']}\n\n" + " ".join(parts) + f"\n\nAnswer format: {fmt}"
        )
        out.append(
            {
                "question_id": f"arc_c_{i:05d}",
                "question": q,
                "correct_answer": idx_to_tag[r["answerKey"]],
                "weak_answer": "",
                "dataset_type": "commonsense170k_mix",
                "source": "arc_challenge",
            }
        )
    return out


def make_hellaswag():
    from datasets import load_dataset

    rows = list(load_dataset("Rowan/hellaswag", split="validation"))
    random.Random(SEED + 1).shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        ctx = (r["ctx_a"] + " " + r["ctx_b"]).strip()
        endings = r["endings"]
        label = int(r["label"])
        parts = " ".join(f"ending{j + 1}: {e}" for j, e in enumerate(endings))
        fmt = "/".join(f"ending{j + 1}" for j in range(len(endings)))
        q = (
            "Please choose the most plausible ending for the following context.\n\n"
            f"Context: {ctx}\n\n{parts}\n\nAnswer format: {fmt}"
        )
        out.append(
            {
                "question_id": f"hellaswag_{i:05d}",
                "question": q,
                "correct_answer": f"ending{label + 1}",
                "weak_answer": "",
                "dataset_type": "commonsense170k_mix",
                "source": "hellaswag",
            }
        )
    return out


def make_winogrande():
    from datasets import load_dataset

    rows = list(
        load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    )
    random.Random(SEED + 2).shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        q = (
            "Please choose the correct answer to fill in the blank ('_') to complete the "
            f"given sentence: {r['sentence']}\n\n"
            f"option1: {r['option1']} option2: {r['option2']}\n\n"
            "Answer format: option1/option2"
        )
        out.append(
            {
                "question_id": f"winogrande_{i:05d}",
                "question": q,
                "correct_answer": f"option{r['answer']}",
                "weak_answer": "",
                "dataset_type": "commonsense170k_mix",
                "source": "winogrande",
            }
        )
    return out


def make_boolq():
    from datasets import load_dataset

    rows = list(load_dataset("google/boolq", split="validation"))
    random.Random(SEED + 3).shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        passage = r["passage"]
        if len(passage) > 1200:
            passage = passage[:1200] + "..."
        q = (
            "Read the following passage and answer the question with true or false.\n\n"
            f"Passage: {passage}\n\nQuestion: {r['question']}?\n\nAnswer format: true/false"
        )
        out.append(
            {
                "question_id": f"boolq_{i:05d}",
                "question": q,
                "correct_answer": "true" if r["answer"] else "false",
                "weak_answer": "",
                "dataset_type": "commonsense170k_mix",
                "source": "boolq",
            }
        )
    return out


def make_obqa():
    from datasets import load_dataset

    rows = list(load_dataset("allenai/openbookqa", "main", split="validation"))
    random.Random(SEED + 4).shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        parts, idx_to_tag = [], {}
        for j, (lab, txt) in enumerate(zip(labels, texts)):
            tag = f"answer{j + 1}"
            idx_to_tag[lab] = tag
            parts.append(f"{tag}: {txt}")
        fmt = "/".join(f"answer{j + 1}" for j in range(len(texts)))
        q = (
            "Please choose the correct answer to the question: "
            f"{r['question_stem']}\n\n" + " ".join(parts) + f"\n\nAnswer format: {fmt}"
        )
        out.append(
            {
                "question_id": f"obqa_{i:05d}",
                "question": q,
                "correct_answer": idx_to_tag[r["answerKey"]],
                "weak_answer": "",
                "dataset_type": "commonsense170k_mix",
                "source": "openbookqa",
            }
        )
    return out


def make_rte():
    from datasets import load_dataset

    rows = list(load_dataset("nyu-mll/glue", "rte", split="validation"))
    random.Random(SEED + 5).shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        q = (
            "Read the premise and decide whether it entails the hypothesis. "
            "Answer true if the premise entails (logically implies) the hypothesis, "
            "otherwise answer false.\n\n"
            f"Premise: {r['sentence1']}\n\nHypothesis: {r['sentence2']}\n\n"
            "Answer format: true/false"
        )
        out.append(
            {
                "question_id": f"rte_{i:05d}",
                "question": q,
                "correct_answer": "true" if r["label"] == 0 else "false",
                "weak_answer": "",
                "dataset_type": "commonsense170k_mix",
                "source": "rte",
            }
        )
    return out


MAKERS = {
    "arc_c": make_arc_c,
    "hellaswag": make_hellaswag,
    "boolq": make_boolq,
    "obqa": make_obqa,
    "rte": make_rte,
    "winogrande": make_winogrande,
}


def main():
    parser = argparse.ArgumentParser(description="Build commonsense evaluation sets")
    parser.add_argument("--datasets", nargs="*", default=PAPER_SUITE, choices=list(MAKERS))
    parser.add_argument("--out-dir", default="data/byzantine/commonsense")
    parser.add_argument("--no-cap", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in args.datasets:
        items = MAKERS[name]()
        if not args.no_cap and name in CAPS:
            items = items[: CAPS[name]]
        if args.limit:
            items = items[: args.limit]
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(items, ensure_ascii=False, indent=1))
        manifest[name] = len(items)
        print(f"{name:12s}: {len(items):5d} questions -> {path}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"total: {sum(manifest.values())} questions")


if __name__ == "__main__":
    main()
