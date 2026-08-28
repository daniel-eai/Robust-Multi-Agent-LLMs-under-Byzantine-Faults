#!/usr/bin/env python3
"""
Prepare MATH-500 dataset for Byzantine fault tolerance experiments.

Downloads from lighteval/MATH-Hard on HuggingFace, calls strong (gpt-4o-mini)
and weak (gpt-3.5-turbo) models, keeps only questions where strong=correct
and weak=wrong. Saves in the same format as gsm8k_final_dataset.

Usage:
    conda run -n agent_LLM python tools/prepare_math500.py \
        --api-key <KEY> \
        --num-questions 10 \
        --output data/byzantine/math500/math500_dataset.json
"""

import argparse
import asyncio
import json
import os
import re
import sys
import hashlib
from pathlib import Path
from typing import Optional

import aiohttp


def extract_boxed_answer(solution: str) -> str:
    """Extract the final boxed answer from a MATH solution string."""
    matches = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', solution)
    if matches:
        ans = matches[-1].strip()
        # Try to extract a clean number
        num = re.sub(r'[^0-9.\-/]', '', ans)
        return num if num else ans
    return ""


async def call_api(session: aiohttp.ClientSession, api_base: str, api_key: str,
                   model: str, messages: list, temperature: float = 0.0) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }
    url = f"{api_base}/chat/completions"
    for attempt in range(3):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                print(f"  API error ({model}): {e}")
                return ""
            await asyncio.sleep(2 ** attempt)
    return ""


def extract_answer_from_response(text: str) -> str:
    """Extract numerical answer from model response."""
    # Look for "Answer: <number>" on its own line (our forced format)
    m = re.search(r'^Answer\s*:\s*(-?[\d]+(?:\.\d+)?)\s*$', text, re.MULTILINE)
    if m:
        return m.group(1).strip()

    # Look for boxed answer
    boxed = re.findall(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if boxed:
        ans = boxed[-1].strip()
        num = re.sub(r'[^0-9.\-/]', '', ans)
        return num if num else ans

    # "Answer: X" where X starts with a number
    m = re.search(r'[Aa]nswer\s*:\s*(-?[\d]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).strip()

    # Last number in response
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else ""


def normalize_answer(ans: str) -> str:
    ans = str(ans).strip().lower()
    ans = re.sub(r'[,\s]', '', ans)
    # remove LaTeX commands
    ans = re.sub(r'\\[a-zA-Z]+', '', ans)
    ans = re.sub(r'[{}]', '', ans)
    # try fraction evaluation: a/b
    m = re.match(r'^(-?\d+)/(-?\d+)$', ans)
    if m:
        try:
            return str(round(int(m.group(1)) / int(m.group(2)), 6))
        except Exception:
            pass
    try:
        return str(round(float(ans), 6))
    except Exception:
        return ans


def is_correct(predicted: str, ground_truth: str) -> bool:
    p = normalize_answer(predicted)
    g = normalize_answer(ground_truth)
    if p == g:
        return True
    # numeric tolerance
    try:
        return abs(float(p) - float(g)) < 1e-3
    except Exception:
        return False


SYSTEM_PROMPT = (
    "You are a math expert. Solve the problem step by step. "
    "At the very end, write your final answer on its own line in this exact format:\n"
    "Answer: <number>\n"
    "Only write a single number (integer or decimal). Do not write words, units, or expressions."
)


async def process_question(session, api_base, api_key, strong_model, weak_model,
                            problem: dict, idx: int):
    question = problem["problem"]
    correct_answer = extract_boxed_answer(problem["solution"])

    if not correct_answer:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Call strong and weak models in parallel
    strong_resp, weak_resp = await asyncio.gather(
        call_api(session, api_base, api_key, strong_model, messages),
        call_api(session, api_base, api_key, weak_model, messages),
    )
    strong_ans = extract_answer_from_response(strong_resp)
    weak_ans = extract_answer_from_response(weak_resp)
    strong_ok = is_correct(strong_ans, correct_answer)
    weak_ok = is_correct(weak_ans, correct_answer)

    print(f"  [{idx}] correct={correct_answer} | strong={strong_ans}({'✓' if strong_ok else '✗'}) | weak={weak_ans}({'✓' if weak_ok else '✗'})")

    q_id = hashlib.md5(question.encode()).hexdigest()[:8]

    return {
        "question": question,
        "correct_answer": correct_answer,
        "strong_model": strong_model,
        "strong_response": strong_resp,
        "strong_answer": strong_ans,
        "strong_correct": strong_ok,
        "weak_model": weak_model,
        "weak_response": weak_resp,
        "weak_answer": weak_ans,
        "weak_correct": weak_ok,
        "is_target": strong_ok and not weak_ok,
        "dataset_type": "math500",
        "question_id": f"math500_{q_id}",
        "level": problem.get("level", ""),
        "type": problem.get("type", ""),
        "stability_rate": 1.0,
        "test_results": {},
    }


async def main():
    parser = argparse.ArgumentParser(description="Prepare MATH-500 dataset")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--strong-model", default="gpt-4o-mini")
    parser.add_argument("--weak-model", default="gpt-3.5-turbo")
    parser.add_argument("--num-questions", type=int, default=10,
                        help="Number of target questions (strong✓, weak✗) to collect")
    parser.add_argument("--max-candidates", type=int, default=300,
                        help="Max questions to try before giving up")
    parser.add_argument("--level", default=None,
                        help="Filter by difficulty level (e.g. 'Level 4', 'Level 5')")
    parser.add_argument("--dataset-source", default="hard",
                        choices=["hard", "full"],
                        help="hard: lighteval/MATH-Hard (Level 5 only) | full: hendrycks/competition_math (Level 1-5)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: data/byzantine/math500/math500_dataset.json)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key required (--api-key or API_KEY env var)")
        sys.exit(1)

    repo_root = Path(__file__).parent.parent
    output_path = Path(args.output) if args.output else repo_root / "data/byzantine/math500/math500_dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    if args.dataset_source == "full":
        print("Loading EleutherAI/hendrycks_math (full, Level 1-5)...")
        subsets = ["algebra", "counting_and_probability", "geometry",
                   "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
        all_problems = []
        for subset in subsets:
            sub = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
            all_problems.extend(list(sub))
        import random as _rnd
        _rnd.seed(args.seed)
        _rnd.shuffle(all_problems)
        dataset = all_problems
    else:
        print("Loading lighteval/MATH-Hard (Level 5 only)...")
        dataset = list(load_dataset("lighteval/MATH-Hard", split="test"))

    problems = list(dataset)
    if args.level:
        problems = [p for p in problems if p.get("level", "") == args.level]
        print(f"Filtered to {len(problems)} problems at {args.level}")

    import random
    random.seed(args.seed)
    random.shuffle(problems)

    results = []
    batch_size = 10  # parallel batch size

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, min(args.max_candidates, len(problems)), batch_size):
            if len(results) >= args.num_questions:
                break

            batch = problems[batch_start:batch_start + batch_size]
            print(f"\nBatch {batch_start//batch_size + 1}: processing {len(batch)} questions in parallel...")

            tasks = [
                process_question(session, args.api_base_url, api_key,
                                 args.strong_model, args.weak_model,
                                 problem, batch_start + i + 1)
                for i, problem in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for item in batch_results:
                if isinstance(item, Exception):
                    print(f"  [ERROR] {item}")
                    continue
                if item is None:
                    print(f"  [SKIP] no boxed answer")
                    continue
                if item["is_target"]:
                    status = "✓✗ ADDED"
                elif not item["strong_correct"] and not item["weak_correct"]:
                    status = "✗✗ both wrong"
                elif item["strong_correct"] and item["weak_correct"]:
                    status = "✓✓ both correct"
                else:
                    status = "✗✓ reversed"
                print(f"  [{status}] correct={item['correct_answer']} strong={item['strong_answer']} weak={item['weak_answer']}")
                if item["is_target"]:
                    results.append(item)
                if len(results) >= args.num_questions:
                    break

            print(f"Collected: {len(results)}/{args.num_questions}")

    print(f"\n{'='*50}")
    print(f"Collected {len(results)}/{args.num_questions} target questions")
    print(f"Saving to {output_path}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
