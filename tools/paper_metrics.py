#!/usr/bin/env python3
"""
Compute paper Table 1 / Table 2 metrics from experiment result JSON files.

Table 1: per (method, topology):
    IAA, FAA, BFTI, RA,
    W IAA -> W FAA  (weak honest group),
    S IAA -> S FAA  (strong honest group),
    H-Majority      (majority among honest agents == ground truth).

Table 2: per (method, topology):
    Round-by-round W/S accuracy across rounds 0..T.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --- answer normalization (re-used loosely from prepare_math500) ---

def _norm_answer(val: str, dataset_type: str) -> str:
    s = str(val).strip()
    if dataset_type == "safe":
        s = s.lower()
        return "safe" if s in ("1", "safe") else "unsafe"
    # commonsense / math500 / gsm8k: light normalization
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    s = s.strip()
    s = re.sub(r"\\(?:text|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace(r"^\circ", "").replace(r"^{\circ}", "").replace("°", "")
    s = s.replace(r"\%", "").replace("%", "")
    s = s.replace(r"\$", "").replace("$", "")
    s = re.sub(r"[,\s]", "", s)
    s = s.rstrip(".")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    while True:
        new = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}]", "", s)
    s = s.lower()
    m = re.match(r"^\(?(-?\d+)\)?/\(?(-?\d+)\)?$", s)
    if m:
        try:
            return str(round(int(m.group(1)) / int(m.group(2)), 6))
        except Exception:
            pass
    try:
        return str(round(float(s), 6))
    except Exception:
        return s


def _eq(pred: str, gt: str, dataset_type: str) -> bool:
    a = _norm_answer(pred, dataset_type)
    b = _norm_answer(gt, dataset_type)
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-3
    except Exception:
        return False


# --- agent role inference ---

def _infer_roles(data: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """Return (malicious_ids, weak_honest_ids, strong_honest_ids).

    Reconstructs the same partition base_runner.py uses:
        weak_honest_ids = first `weak_honest` IDs from non-malicious agents.
    """
    metadata = data.get("metadata", {})
    config = metadata.get("config", {})
    malicious_ids = list(metadata.get("malicious_agents", []) or [])
    weak_honest_count = int(config.get("weak_honest", 0) or 0)

    # Collect all agent IDs in their first-seen order.
    seen = []
    for cr in data.get("consensus_results", []):
        for ir in cr.get("individual_responses", []):
            aid = ir.get("agent_id")
            if aid and aid not in seen:
                seen.append(aid)

    malicious_set = set(malicious_ids)
    honest_ids = [a for a in seen if a not in malicious_set]
    weak_honest_ids = honest_ids[:weak_honest_count]
    strong_honest_ids = honest_ids[weak_honest_count:]
    return malicious_ids, weak_honest_ids, strong_honest_ids


# --- per-result-file metric computation ---

def compute_paper_metrics(json_path: str) -> Optional[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    consensus_results = data.get("consensus_results", [])
    if not consensus_results:
        return None
    metadata = data.get("metadata", {})
    config = metadata.get("config", {})
    dataset_type = str(config.get("dataset_type", "gsm8k")).lower()

    correct_map = {q["question_id"]: q["correct_answer"] for q in questions}

    malicious_ids, weak_ids, strong_ids = _infer_roles(data)

    # Counters across all questions x agents.
    counts = {
        "all_init_correct": 0, "all_final_correct": 0, "all_total": 0,
        "weak_init_correct": 0, "weak_final_correct": 0, "weak_total": 0,
        "strong_init_correct": 0, "strong_final_correct": 0, "strong_total": 0,
        "consensus_correct": 0, "honest_majority_correct": 0,
        "n_questions": 0,
    }

    # Round-by-round W/S accuracy (Table 2).
    # round_data is a list per question: [round0, round1, ..., roundT] of {agent_id: answer}.
    rounds_W: List[List[float]] = []
    rounds_S: List[List[float]] = []

    for cr in consensus_results:
        qid = cr.get("question_id")
        gt = correct_map.get(qid, "")
        counts["n_questions"] += 1

        # Per-agent init/final.
        for ir in cr.get("individual_responses", []):
            aid = ir.get("agent_id")
            meta = ir.get("metadata", {}) or {}
            init = str(meta.get("initial_answer", ir.get("answer", "")))
            final = str(meta.get("final_answer", ir.get("answer", "")))

            init_ok = _eq(init, gt, dataset_type)
            final_ok = _eq(final, gt, dataset_type)

            counts["all_init_correct"] += int(init_ok)
            counts["all_final_correct"] += int(final_ok)
            counts["all_total"] += 1

            if aid in weak_ids:
                counts["weak_init_correct"] += int(init_ok)
                counts["weak_final_correct"] += int(final_ok)
                counts["weak_total"] += 1
            elif aid in strong_ids:
                counts["strong_init_correct"] += int(init_ok)
                counts["strong_final_correct"] += int(final_ok)
                counts["strong_total"] += 1

        # Consensus correctness (RA).
        is_correct = cr.get("is_correct")
        if is_correct is None:
            cans = cr.get("consensus_answer", "")
            is_correct = _eq(cans, gt, dataset_type)
        if is_correct:
            counts["consensus_correct"] += 1

        # Honest-Majority: majority over honest agents only.
        cr_meta = cr.get("metadata", {}) or {}
        round_data = cr_meta.get("round_data", [])
        if round_data:
            final_round = round_data[-1]
            honest_set = set(weak_ids) | set(strong_ids)
            from collections import Counter
            honest_finals = [final_round[a] for a in honest_set if a in final_round]
            normed = [_norm_answer(x, dataset_type) for x in honest_finals]
            if normed:
                top, _ = Counter(normed).most_common(1)[0]
                if _eq(top, gt, dataset_type):
                    counts["honest_majority_correct"] += 1

            # Round-by-round W/S accuracy.
            T = len(round_data) - 1
            while len(rounds_W) < len(round_data):
                rounds_W.append([])
                rounds_S.append([])
            for t, snapshot in enumerate(round_data):
                w_n = sum(1 for a in weak_ids if a in snapshot)
                w_c = sum(1 for a in weak_ids if a in snapshot and _eq(snapshot[a], gt, dataset_type))
                s_n = sum(1 for a in strong_ids if a in snapshot)
                s_c = sum(1 for a in strong_ids if a in snapshot and _eq(snapshot[a], gt, dataset_type))
                if w_n:
                    rounds_W[t].append(w_c / w_n)
                if s_n:
                    rounds_S[t].append(s_c / s_n)

    if counts["all_total"] == 0 or counts["n_questions"] == 0:
        return None

    def pct(num, den):
        return round(100 * num / den, 1) if den else 0.0

    IAA = pct(counts["all_init_correct"], counts["all_total"])
    FAA = pct(counts["all_final_correct"], counts["all_total"])
    BFTI = round(FAA - IAA, 1)
    RA = pct(counts["consensus_correct"], counts["n_questions"])
    H_MAJ = pct(counts["honest_majority_correct"], counts["n_questions"])
    W_IAA = pct(counts["weak_init_correct"], counts["weak_total"])
    W_FAA = pct(counts["weak_final_correct"], counts["weak_total"])
    S_IAA = pct(counts["strong_init_correct"], counts["strong_total"])
    S_FAA = pct(counts["strong_final_correct"], counts["strong_total"])

    # Round-by-round (Table 2 format).
    round_W_pct = [round(100 * (sum(xs) / len(xs)), 1) if xs else 0.0 for xs in rounds_W]
    round_S_pct = [round(100 * (sum(xs) / len(xs)), 1) if xs else 0.0 for xs in rounds_S]

    return {
        "experiment_id": data.get("experiment_id", Path(json_path).stem),
        "method": data.get("method_type", "unknown"),
        "topology": data.get("topology_type", config.get("topology", "unknown")),
        "agents": data.get("agent_count", config.get("agents", 0)),
        "malicious": data.get("malicious_count", config.get("malicious", 0)),
        "weak_honest": int(config.get("weak_honest", 0) or 0),
        "rounds_T": len(round_W_pct) - 1 if round_W_pct else 0,
        "n_questions": counts["n_questions"],
        # Table 1
        "IAA": IAA, "FAA": FAA, "BFTI": BFTI, "RA": RA, "H_Majority": H_MAJ,
        "W_IAA": W_IAA, "W_FAA": W_FAA, "S_IAA": S_IAA, "S_FAA": S_FAA,
        # Table 2
        "round_W": round_W_pct,
        "round_S": round_S_pct,
        "json_path": json_path,
    }


def scan(root: str) -> List[Dict[str, Any]]:
    out = []
    for path in sorted(Path(root).rglob("*.json")):
        if path.name.endswith(".summary.json") or path.name == "comparison.json":
            continue
        try:
            m = compute_paper_metrics(str(path))
            if m:
                out.append(m)
        except Exception as e:
            print(f"[skip] {path}: {e}")
    return out


# --- formatting ---

_TOP_ORDER = ["merg", "k_circulant", "preferential", "robust_random",
              "complete", "star", "chain", "tree", "random", "layered_graph"]
_METHOD_ORDER = ["prompt_probe", "sac", "decoder_probe", "pilot"]


def _sort_key(r: Dict[str, Any]):
    m = str(r["method"]).lower()
    t = str(r["topology"]).lower()
    mi = _METHOD_ORDER.index(m) if m in _METHOD_ORDER else 99
    ti = _TOP_ORDER.index(t) if t in _TOP_ORDER else 99
    return (mi, ti)


def render_table1(rows: List[Dict[str, Any]]) -> str:
    rows = sorted(rows, key=_sort_key)
    out = ["# Table 1 — Paper format",
           "",
           "| Method | Topology | IAA | FAA | BFTI | RA | W IAA→FAA | S IAA→FAA | H-Majority |",
           "|--------|----------|-----|-----|------|----|-----------|-----------|------------|"]
    method_seen = None
    for r in rows:
        m = "CP-WBFT" if r["method"] == "prompt_probe" else ("SAC" if r["method"] == "sac" else r["method"])
        m_label = m if m != method_seen else ""
        method_seen = m
        bfti = f"{r['BFTI']:+.1f}%"
        out.append(
            f"| {m_label} | {r['topology']} "
            f"| {r['IAA']:.1f}% | {r['FAA']:.1f}% | {bfti} | {r['RA']:.1f}% "
            f"| {r['W_IAA']:.0f}→{r['W_FAA']:.0f}% | {r['S_IAA']:.0f}→{r['S_FAA']:.0f}% "
            f"| {r['H_Majority']:.1f}% |"
        )
    return "\n".join(out) + "\n"


def render_table2(rows: List[Dict[str, Any]]) -> str:
    rows = sorted(rows, key=_sort_key)
    if not rows:
        return ""
    T = max(len(r["round_W"]) for r in rows)
    out = ["# Table 2 — Per-round W/S honest accuracy",
           ""]
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    for method in methods:
        m_rows = [r for r in rows if r["method"] == method]
        m_label = "CP-WBFT" if method == "prompt_probe" else ("SAC" if method == "sac" else method)
        out.append(f"## {m_label}")
        out.append("")
        header = "| Round | " + " | ".join(f"{r['topology']} (W/S)" for r in m_rows) + " |"
        sep = "|---|" + "|".join(["---"] * len(m_rows)) + "|"
        out.append(header)
        out.append(sep)
        for t in range(T):
            cells = []
            for r in m_rows:
                if t < len(r["round_W"]):
                    cells.append(f"{r['round_W'][t]:.1f} / {r['round_S'][t]:.1f}")
                else:
                    # CP-WBFT often converges and stops storing snapshots after Rnd 1.
                    # Pad with the last available round so the collapse stays visible
                    # across all reported rounds in the table.
                    last = len(r["round_W"]) - 1
                    if last >= 0:
                        cells.append(f"{r['round_W'][last]:.1f} / {r['round_S'][last]:.1f}")
                    else:
                        cells.append("-")
            label = "Init" if t == 0 else f"Rnd {t}"
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", "-d", required=True, help="Result root (e.g. results/paper_math500_t6)")
    ap.add_argument("--out", "-o", default=None, help="Write Markdown report to this path")
    ap.add_argument("--csv", default=None, help="Write per-row metrics to CSV")
    args = ap.parse_args()

    rows = scan(args.dir)
    if not rows:
        print(f"No results found under {args.dir}")
        return

    md = render_table1(rows) + "\n" + render_table2(rows)
    print(md)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\nMarkdown report saved to: {args.out}")

    if args.csv:
        keys = ["method", "topology", "agents", "malicious", "weak_honest", "rounds_T",
                "n_questions", "IAA", "FAA", "BFTI", "RA", "H_Majority",
                "W_IAA", "W_FAA", "S_IAA", "S_FAA", "experiment_id", "json_path"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in sorted(rows, key=_sort_key):
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
