#!/usr/bin/env python3
"""make_tables_commonsense_qwen.py — generate paper-style Table 3 and Table 4
for each of the 6 commonsense benchmarks from the final_commonsense_qwen/
result tree.

For each dataset directory we expect six experiment subdirs
(SAC_{merg,complete,random}, CP_{merg,complete,random}), each containing the
OursNew run output JSON. Roles are fixed by config (agents=7, malicious=1,
weak-honest=2):

  agent_0, agent_1  -> weak honest (2)
  agent_2..agent_5  -> strong honest (4)
  agent_6           -> adversarial Byzantine (1)

Table 3 columns: IAA / FAA / BFTI / RA / W IAA->FAA / S IAA->FAA / H-Majority
Table 4 columns: round-0 (Init) .. round-6, each cell "W/S" (weak honest avg,
strong honest avg) in percent.

Outputs per dataset:
  final_commonsense_qwen/<dataset>/table3.txt
  final_commonsense_qwen/<dataset>/table4.txt
  final_commonsense_qwen/<dataset>/metrics.json
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(os.getenv("RESULTS_ROOT", "results"))
DATASETS = ["arc_c", "hellaswag", "winogrande", "boolq", "obqa", "rte"]
EXPERIMENTS = [
    # (method_label, friendly, topology_label_in_table)
    ("SAC", "SAC_merg",          "MERG"),
    ("SAC", "SAC_k_circulant",   "k-circulant"),
    ("SAC", "SAC_preferential",  "Preferential"),
    ("SAC", "SAC_robust_random", "Erdos-Renyi"),
    ("CP-WBFT", "CP_merg",          "MERG"),
    ("CP-WBFT", "CP_k_circulant",   "k-circulant"),
    ("CP-WBFT", "CP_preferential",  "Preferential"),
    ("CP-WBFT", "CP_robust_random", "Erdos-Renyi"),
]

WEAK = {"agent_0", "agent_1"}
STRONG = {"agent_2", "agent_3", "agent_4", "agent_5"}
BYZANTINE = {"agent_6"}
HONEST = WEAK | STRONG  # 6 agents
ALL_AGENTS = HONEST | BYZANTINE  # 7 agents


def _round_acc(round_answers: Dict[str, str], gt: str, agents: set) -> float:
    if not agents:
        return float("nan")
    correct = sum(1 for a in agents if round_answers.get(a, "") == gt)
    return correct / len(agents)


def _majority(answers: Dict[str, str], agents: set) -> str:
    counts = Counter(answers.get(a, "") for a in agents)
    counts.pop("", None)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _load_result_json(exp_dir: Path) -> dict | None:
    candidates = sorted(exp_dir.glob("sac_*.json")) + sorted(exp_dir.glob("prompt_probe_*.json")) + sorted(exp_dir.glob("prompt_*.json"))
    candidates = [c for c in candidates if "summary" not in c.name and "report" not in c.name]
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def _per_question_metrics(result: dict) -> List[dict]:
    qmap = {q["question_id"]: q for q in result.get("questions", [])}
    out = []
    for cr in result.get("consensus_results", []):
        qid = cr["question_id"]
        q = qmap.get(qid)
        gt = (q.get("correct_answer") or "").strip().lower() if q else ""
        md = cr.get("metadata", {})
        initial = {a: (v or "").strip().lower() for a, v in (md.get("initial_answers") or {}).items()}
        rounds = [
            {a: (v or "").strip().lower() for a, v in (rd or {}).items()}
            for rd in (md.get("round_data") or [])
        ]
        out.append({
            "qid": qid,
            "gt": gt,
            "initial": initial,
            "rounds": rounds,  # length should be 7 (Init + 6 rounds)
        })
    return out


def _table3_row(per_q: List[dict]) -> Dict[str, float]:
    if not per_q:
        return {}
    # IAA / FAA over all 7 agents
    iaa_vals, faa_vals = [], []
    ra_correct = 0
    h_maj_correct = 0
    w_iaa, w_faa, s_iaa, s_faa = [], [], [], []
    for pq in per_q:
        gt = pq["gt"]
        init = pq["initial"]
        rounds = pq["rounds"]
        final = rounds[-1] if rounds else init
        # per-agent accuracy over ALL 7 agents
        iaa_vals.append(_round_acc(init, gt, ALL_AGENTS))
        faa_vals.append(_round_acc(final, gt, ALL_AGENTS))
        # group-wise weak / strong, before and after
        w_iaa.append(_round_acc(init, gt, WEAK))
        w_faa.append(_round_acc(final, gt, WEAK))
        s_iaa.append(_round_acc(init, gt, STRONG))
        s_faa.append(_round_acc(final, gt, STRONG))
        # RA: majority of ALL 7 agents at round T equals gt
        if _majority(final, ALL_AGENTS) == gt:
            ra_correct += 1
        # H-Majority: majority of honest 6 agents at round T equals gt
        if _majority(final, HONEST) == gt:
            h_maj_correct += 1
    n = len(per_q)
    iaa = 100 * sum(iaa_vals) / n
    faa = 100 * sum(faa_vals) / n
    bfti = faa - iaa
    ra = 100 * ra_correct / n
    h_maj = 100 * h_maj_correct / n
    w_iaa_pct = 100 * sum(w_iaa) / n
    w_faa_pct = 100 * sum(w_faa) / n
    s_iaa_pct = 100 * sum(s_iaa) / n
    s_faa_pct = 100 * sum(s_faa) / n
    return {
        "IAA": iaa, "FAA": faa, "BFTI": bfti, "RA": ra,
        "W_IAA": w_iaa_pct, "W_FAA": w_faa_pct,
        "S_IAA": s_iaa_pct, "S_FAA": s_faa_pct,
        "H_Majority": h_maj,
    }


def _table4_rounds(per_q: List[dict]) -> List[Tuple[float, float]]:
    """Return per-round (W%, S%) for rounds 0..6 (Init + 6 communication rounds)."""
    if not per_q:
        return []
    n_rounds = max(len(pq["rounds"]) for pq in per_q)
    out = []
    for r in range(n_rounds):
        w_vals, s_vals = [], []
        for pq in per_q:
            if r < len(pq["rounds"]):
                ra = pq["rounds"][r]
            else:
                ra = pq["rounds"][-1] if pq["rounds"] else pq["initial"]
            gt = pq["gt"]
            w_vals.append(_round_acc(ra, gt, WEAK))
            s_vals.append(_round_acc(ra, gt, STRONG))
        out.append((100 * sum(w_vals) / len(w_vals), 100 * sum(s_vals) / len(s_vals)))
    return out


def format_table3(rows: List[Tuple[str, str, Dict[str, float]]]) -> str:
    hdr = (f"{'Method':<10}{'Topology':<13}{'IAA':>7}{'FAA':>7}{'BFTI':>8}"
           f"{'RA':>7}{'W IAA->FAA':>13}{'S IAA->FAA':>13}{'H-Maj':>8}")
    lines = [hdr, "-" * len(hdr)]
    for method, topo, m in rows:
        if not m:
            lines.append(f"{method:<10}{topo:<13}  (no data)")
            continue
        lines.append(
            f"{method:<10}{topo:<13}"
            f"{m['IAA']:>6.1f}%{m['FAA']:>6.1f}%"
            f"{m['BFTI']:>+7.1f}%{m['RA']:>6.1f}%"
            f"  {m['W_IAA']:>4.0f}->{m['W_FAA']:>4.0f}%"
            f"  {m['S_IAA']:>4.0f}->{m['S_FAA']:>4.0f}%"
            f"{m['H_Majority']:>7.1f}%"
        )
    return "\n".join(lines)


def format_table4(method: str, topo: str, rounds: List[Tuple[float, float]]) -> str:
    labels = ["Init"] + [f"Rnd {i}" for i in range(1, len(rounds))]
    cells = " | ".join(f"{w:5.1f}/{s:5.1f}" for w, s in rounds)
    return f"  {method:<10}{topo:<13} {cells}"


def main() -> None:
    summary = {}
    for ds in DATASETS:
        ds_dir = ROOT / ds
        if not ds_dir.exists():
            print(f"[skip] {ds_dir} missing")
            continue

        t3_rows = []
        t4_rows = []
        ds_metrics = {}
        for method, friendly, topo in EXPERIMENTS:
            exp_dir = ds_dir / friendly
            if not exp_dir.exists():
                t3_rows.append((method, topo, {}))
                continue
            res = _load_result_json(exp_dir)
            if not res:
                t3_rows.append((method, topo, {}))
                continue
            per_q = _per_question_metrics(res)
            m3 = _table3_row(per_q)
            t3_rows.append((method, topo, m3))
            rounds = _table4_rounds(per_q)
            t4_rows.append((method, topo, rounds))
            ds_metrics[friendly] = {"table3": m3, "table4_rounds": rounds, "n_questions": len(per_q)}

        # Write Table 3
        t3_text = f"Table 3 — {ds} (50Q, strong=qwen3-4b, weak=qwen2.5-1.5b-instruct, instruct-only)\n\n" + format_table3(t3_rows)
        (ds_dir / "table3.txt").write_text(t3_text + "\n")

        # Write Table 4
        if t4_rows:
            n_rounds = max(len(r[2]) for r in t4_rows)
            hdr_cols = ["Init"] + [f"Rnd {i}" for i in range(1, n_rounds)]
            hdr = "  " + " " * 23 + " | ".join(f"{c:^11}" for c in hdr_cols)
            lines = [
                f"Table 4 — {ds} per-round W/S (in %, weak honest 2 agents / strong honest 4 agents)",
                "",
                hdr,
                "  " + "-" * (len(hdr) - 2),
            ]
            for method, topo, rounds in t4_rows:
                lines.append(format_table4(method, topo, rounds))
            (ds_dir / "table4.txt").write_text("\n".join(lines) + "\n")

        (ds_dir / "metrics.json").write_text(json.dumps(ds_metrics, indent=2))
        summary[ds] = ds_metrics
        print(f"[ok] {ds}: wrote table3.txt / table4.txt / metrics.json")

    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {ROOT/'summary.json'}")


if __name__ == "__main__":
    main()
