#!/usr/bin/env python3
"""
Compute IAA, FAA, BFTI, RA metrics from experiment result JSON files.

IAA  (Initial Agent Accuracy)  : fraction of correct initial answers across all agents
FAA  (Final Agent Accuracy)    : fraction of correct final answers across all agents
BFTI (Byzantine Fault Tolerance Improvement) : FAA - IAA
RA   (Round Accuracy)          : fraction of questions where consensus is correct
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


def _norm(val: str, dataset_type: str) -> str:
    s = str(val).strip().lower()
    if dataset_type == "safe":
        if s in ("1", "safe"):
            return "safe"
        return "unsafe"
    return s


def compute_metrics(json_path: str) -> Optional[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    consensus_results = data.get("consensus_results", [])
    metadata = data.get("metadata", {})
    config = metadata.get("config", {})
    dataset_type = str(config.get("dataset_type", "gsm8k")).lower()

    correct_answer_map = {q["question_id"]: q["correct_answer"] for q in questions}

    initial_correct = 0
    final_correct = 0
    total_agent_answers = 0
    consensus_correct = 0
    total_rounds = len(consensus_results)

    for cr in consensus_results:
        question_id = cr.get("question_id")
        correct_answer = correct_answer_map.get(question_id, "")
        correct_norm = _norm(correct_answer, dataset_type)

        # RA: is consensus correct?
        is_correct = cr.get("is_correct")
        if is_correct is None:
            consensus_ans = _norm(cr.get("consensus_answer", ""), dataset_type)
            is_correct = (consensus_ans == correct_norm)
        if is_correct:
            consensus_correct += 1

        # IAA / FAA: per-agent initial and final answers
        for resp in cr.get("individual_responses", []):
            meta = resp.get("metadata", {}) or {}
            initial_ans = _norm(str(meta.get("initial_answer", resp.get("answer", ""))), dataset_type)
            final_ans = _norm(str(meta.get("final_answer", resp.get("answer", ""))), dataset_type)

            if initial_ans == correct_norm:
                initial_correct += 1
            if final_ans == correct_norm:
                final_correct += 1
            total_agent_answers += 1

    if total_agent_answers == 0:
        return None

    IAA = initial_correct / total_agent_answers
    FAA = final_correct / total_agent_answers
    BFTI = FAA - IAA
    RA = consensus_correct / total_rounds if total_rounds > 0 else 0.0

    exp_id = data.get("experiment_id", Path(json_path).stem)
    method = data.get("method_type", "unknown")
    topology = data.get("topology_type", "unknown")
    dishonest = config.get("dishonest_confidence", "none")

    return {
        "experiment_id": exp_id,
        "method": method,
        "topology": topology,
        "dishonest_confidence": str(dishonest) if dishonest else "none",
        "agents": data.get("agent_count", 0),
        "malicious": data.get("malicious_count", 0),
        "IAA": round(IAA * 100, 2),
        "FAA": round(FAA * 100, 2),
        "BFTI": round(BFTI * 100, 2),
        "RA": round(RA * 100, 2),
    }


def scan_directory(root_dir: str) -> List[Dict[str, Any]]:
    results = []
    for path in Path(root_dir).rglob("*.json"):
        if path.name.endswith(".summary.json"):
            continue
        try:
            m = compute_metrics(str(path))
            if m:
                results.append(m)
        except Exception as e:
            pass
    return results


def print_table(results: List[Dict[str, Any]], group_by_mode: bool = True):
    if not results:
        print("No results found.")
        return

    # Sort
    results = sorted(results, key=lambda x: (x["dishonest_confidence"], x["method"], x["topology"]))

    modes = sorted(set(r["dishonest_confidence"] for r in results))

    for mode in modes:
        mode_results = [r for r in results if r["dishonest_confidence"] == mode]
        if not mode_results:
            continue

        print(f"\n{'='*70}")
        print(f"  Dishonest Confidence Mode: {mode}")
        print(f"{'='*70}")
        print(f"{'Method':<12} {'Topology':<15} {'IAA':>7} {'FAA':>7} {'BFTI':>8} {'RA':>7}")
        print("-" * 60)

        for r in mode_results:
            bfti_str = f"{r['BFTI']:+.1f}%"
            print(f"{r['method']:<12} {r['topology']:<15} "
                  f"{r['IAA']:>6.1f}% {r['FAA']:>6.1f}% "
                  f"{bfti_str:>8} {r['RA']:>6.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compute IAA/FAA/BFTI/RA metrics from result JSON files")
    parser.add_argument("--dir", "-d", required=True, help="Root directory containing result JSON files")
    parser.add_argument("--output", "-o", default=None, help="Save results to CSV (optional)")
    args = parser.parse_args()

    print(f"Scanning: {args.dir}")
    results = scan_directory(args.dir)
    print(f"Found {len(results)} experiment results\n")

    print_table(results)

    if args.output:
        import csv
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "method", "topology",
                                                    "dishonest_confidence", "agents", "malicious",
                                                    "IAA", "FAA", "BFTI", "RA"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
