#!/usr/bin/env python3
"""
Plot round-by-round accuracy convergence curve.
X axis: communication round T
Y axis: consensus accuracy

Usage:
    python tools/plot_convergence.py \
        --results results/prompt/.../result.json "CP-WBFT" \
                  results/sac/.../result.json "SAC" \
        --output convergence_curve.png
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def normalize(ans: str) -> str:
    ans = str(ans).strip().lower()
    ans = re.sub(r'[,\s]', '', ans)
    ans = re.sub(r'\\[a-zA-Z]+', '', ans)
    ans = re.sub(r'[{}]', '', ans)
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


def is_correct(pred: str, gt: str) -> bool:
    p, g = normalize(pred), normalize(gt)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-3
    except Exception:
        return False


def compute_three_metrics(result_path: str):
    """
    Returns dict of three per-round metric lists:
      'agreement'   : fraction of questions where ALL honest agents have the same answer
      'convergence' : fraction of questions where ALL honest agents have the correct answer
      'hon_majority': fraction of questions where honest majority vote == correct answer
    """
    from collections import Counter as _Counter
    d = json.load(open(result_path))
    mal = set(d['metadata']['malicious_agents'])
    cr = d.get('consensus_results', [])
    if not cr:
        return {'agreement': [], 'convergence': [], 'hon_majority': []}

    correct_map = {q['question_id']: normalize(q['correct_answer']) for q in d.get('questions', [])}
    all_agents = list(d['evaluation_metrics']['node_accuracy']['node_accuracies'].keys())
    hon_agents = [a for a in all_agents if a not in mal]

    max_rounds = max(len(r.get('metadata', {}).get('round_data', [])) for r in cr)
    if max_rounds == 0:
        return {'agreement': [], 'convergence': [], 'hon_majority': []}

    agreement_list, convergence_list, hon_majority_list = [], [], []

    for t in range(max_rounds):
        agree_count = conv_count = maj_count = total = 0

        for r in cr:
            qid = r['question_id']
            correct_ans = correct_map.get(qid, '')
            if not correct_ans:
                continue

            round_data = r.get('metadata', {}).get('round_data', [])
            snapshot = round_data[t] if t < len(round_data) else (round_data[-1] if round_data else {})

            hon_answers = [normalize(snapshot.get(a, '')) for a in hon_agents if snapshot.get(a)]
            if not hon_answers:
                continue

            total += 1

            # (1) Agreement: all honest agents same answer
            if len(set(hon_answers)) == 1:
                agree_count += 1

            # (2) Convergence: ALL honest agents correct
            if all(is_correct(a, correct_ans) for a in hon_answers):
                conv_count += 1

            # (3) Honest majority vote correct
            hon_consensus = _Counter(hon_answers).most_common(1)[0][0]
            if is_correct(hon_consensus, correct_ans):
                maj_count += 1

        if total > 0:
            agreement_list.append(agree_count / total)
            convergence_list.append(conv_count / total)
            hon_majority_list.append(maj_count / total)

    return {'agreement': agreement_list, 'convergence': convergence_list, 'hon_majority': hon_majority_list}


def compute_round_accuracies_honest_only(result_path: str) -> Tuple[List[float], List[str]]:
    """
    Returns (accuracies_per_round, malicious_agents).
    At each round, majority vote among honest agents only is compared with correct answer.
    """
    from collections import Counter as _Counter
    d = json.load(open(result_path))
    mal = set(d['metadata']['malicious_agents'])
    cr = d.get('consensus_results', [])
    if not cr:
        return [], list(mal)

    # correct answer map from questions field
    correct_map = {q['question_id']: normalize(q['correct_answer']) for q in d.get('questions', [])}

    all_agents = list(d['evaluation_metrics']['node_accuracy']['node_accuracies'].keys())
    hon_agents = [a for a in all_agents if a not in mal]

    max_rounds = max(len(r.get('metadata', {}).get('round_data', [])) for r in cr)
    if max_rounds == 0:
        return [], list(mal)

    accuracies = []
    for t in range(max_rounds):
        correct_count = 0
        total = 0
        for r in cr:
            qid = r['question_id']
            correct_ans = correct_map.get(qid, '')
            if not correct_ans:
                continue

            round_data = r.get('metadata', {}).get('round_data', [])
            snapshot = round_data[t] if t < len(round_data) else (round_data[-1] if round_data else {})

            hon_answers = [normalize(snapshot.get(a, '')) for a in hon_agents if snapshot.get(a)]
            if not hon_answers:
                continue

            hon_consensus = _Counter(hon_answers).most_common(1)[0][0]
            if is_correct(hon_consensus, correct_ans):
                correct_count += 1
            total += 1

        if total > 0:
            accuracies.append(correct_count / total)

    return accuracies, list(mal)


def compute_round_accuracies(result_path: str) -> Tuple[List[float], List[str]]:
    """
    Returns (accuracies_per_round, malicious_agents).
    accuracies_per_round[t] = fraction of questions where
    the majority of honest agents were correct at round t.
    """
    d = json.load(open(result_path))
    mal = set(d['metadata']['malicious_agents'])
    cr = d.get('consensus_results', [])

    if not cr:
        return [], list(mal)

    # Find max rounds from round_data
    max_rounds = max(
        len(r.get('metadata', {}).get('round_data', [])) for r in cr
    )
    if max_rounds == 0:
        return [], list(mal)

    all_agents = list(d['evaluation_metrics']['node_accuracy']['node_accuracies'].keys())
    hon_agents = [a for a in all_agents if a not in mal]

    # Get correct answers from consensus results
    # Use is_correct flag logic: compare with stored correct via question metadata
    # We'll use the final individual responses to infer correct answers
    # Best approach: store correct answer per question from the dataset
    # Since we may not have dataset, use the node_accuracy correct_answer if available

    # Build correct answer map from node_accuracy responses and is_correct
    # Actually, let's use a different approach:
    # For each question, find what the correct answer is by checking which
    # non-Byzantine agent had the right answer in the final round (if any)
    # This is imperfect; better to use dataset.

    # Use consensus_result.is_correct and consensus_answer to find correct answers
    # If consensus is correct, consensus_answer = correct_answer
    # If not, we look at individual responses metadata

    correct_answer_map: Dict[str, str] = {}
    for r in cr:
        qid = r['question_id']
        # Try to get correct answer from node_accuracy data
        node_acc = d['evaluation_metrics']['node_accuracy']['node_accuracies']
        # responses field has agent answers; is_correct per agent
        for aid, info in node_acc.items():
            if info.get('accuracy', 0) == 1.0 and info.get('responses'):
                # This agent got all correct - use their answers as reference
                # Map question index to answer
                pass

        # Simpler: use the question's correct_answer if stored in metadata
        meta = r.get('metadata', {})
        if 'correct_answer' in meta:
            correct_answer_map[qid] = meta['correct_answer']

    # If we couldn't get correct answers from metadata, use a heuristic:
    # For each question, find the most common non-Byzantine initial answer
    # as a proxy for correct answer (works when honest > malicious initially)
    if not correct_answer_map:
        for r in cr:
            qid = r['question_id']
            round_data = r.get('metadata', {}).get('round_data', [])
            if round_data:
                init = round_data[0]
                hon_answers = [init.get(a, '') for a in hon_agents if init.get(a)]
                if hon_answers:
                    # majority vote of honest agents at round 0
                    from collections import Counter
                    most_common = Counter(hon_answers).most_common(1)[0][0]
                    correct_answer_map[qid] = most_common

    # Compute per-round accuracy
    accuracies = []
    for t in range(max_rounds):
        correct_count = 0
        total = 0
        for r in cr:
            qid = r['question_id']
            round_data = r.get('metadata', {}).get('round_data', [])
            if t >= len(round_data):
                # Use last available round
                snapshot = round_data[-1] if round_data else {}
            else:
                snapshot = round_data[t]

            correct_ans = correct_answer_map.get(qid, '')
            if not correct_ans:
                continue

            # Check majority of honest agents at this round
            hon_correct = sum(
                1 for a in hon_agents
                if is_correct(snapshot.get(a, ''), correct_ans)
            )
            if hon_correct > len(hon_agents) / 2:
                correct_count += 1
            total += 1

        if total > 0:
            accuracies.append(correct_count / total)

    return accuracies, list(mal)


def plot_convergence(method_results: List[Tuple[str, str]], output: str, honest_only: bool = False):
    import matplotlib.pyplot as plt

    colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6']
    markers = ['o', 's', '^', 'D', 'v']

    fig, ax = plt.subplots(figsize=(7, 5))

    compute_fn = compute_round_accuracies_honest_only if honest_only else compute_round_accuracies

    for idx, (path, label) in enumerate(method_results):
        accs, _ = compute_fn(path)
        if not accs:
            print(f"Warning: No round_data found in {path}")
            continue
        rounds = list(range(len(accs)))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(rounds, accs, color=color, marker=marker,
                linewidth=2, markersize=6, label=label)
        ax.annotate(f'{accs[-1]:.2f}',
                    xy=(rounds[-1], accs[-1]),
                    xytext=(4, 0), textcoords='offset points',
                    color=color, fontsize=9, va='center')

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    title = 'Honest-Agent Majority Accuracy per Round' if honest_only else 'Round-by-Round Accuracy Convergence'
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_three_metrics(method_results: List[Tuple[str, str]], output: str):
    """Plot Agreement / Convergence / Honest-Majority in 3 subplots, one per method."""
    import matplotlib.pyplot as plt

    metric_keys  = ['agreement', 'convergence', 'hon_majority']
    metric_labels = ['Agreement\n(all honest same)', 'Convergence\n(all honest correct)', 'Honest Majority\n(majority correct)']
    colors = {'agreement': '#3498DB', 'convergence': '#2ECC71', 'hon_majority': '#E74C3C'}

    n_methods = len(method_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, (path, label) in zip(axes, method_results):
        metrics = compute_three_metrics(path)
        rounds = list(range(len(metrics['agreement'])))
        for key, mlabel in zip(metric_keys, metric_labels):
            vals = metrics[key]
            if vals:
                ax.plot(rounds, vals, color=colors[key], marker='o',
                        linewidth=2, markersize=5, label=mlabel)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('Communication Round', fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Fraction of Questions', fontsize=11)
    fig.suptitle('Round-by-Round Honest Agent Metrics', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+', metavar=('PATH', 'LABEL'),
                        help='Pairs of result_path label (e.g. result.json "SAC")')
    parser.add_argument('--output', default='convergence_curve.png')
    parser.add_argument('--honest-only', action='store_true',
                        help='Compute accuracy using honest agents majority vote only')
    parser.add_argument('--three-metrics', action='store_true',
                        help='Plot agreement / convergence / honest-majority per method')
    args = parser.parse_args()

    if not args.results or len(args.results) % 2 != 0:
        parser.error('--results must be pairs of PATH LABEL')

    method_results = []
    for i in range(0, len(args.results), 2):
        method_results.append((args.results[i], args.results[i+1]))

    if args.three_metrics:
        plot_three_metrics(method_results, args.output)
    else:
        plot_convergence(method_results, args.output, honest_only=args.honest_only)


if __name__ == '__main__':
    main()
