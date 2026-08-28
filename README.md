# Robust Multi-Agent LLMs under Byzantine Faults

Official implementation of **Robust Multi-Agent LLMs under Byzantine Faults**.

[![arXiv](https://img.shields.io/badge/arXiv-2605.09076-b31b1b.svg)](https://arxiv.org/abs/2605.09076)

## Overview

Large Language Model Multi-Agent Systems (LLM-MAS) can benefit from collaboration among multiple agents, but their communication also introduces vulnerabilities when some agents are faulty or adversarial.

We propose **Self-Anchored Consensus (SAC)**, a fully decentralized filter-and-refine protocol designed to make LLM multi-agent systems robust against Byzantine agents. Our framework connects Byzantine-resilient consensus theory with LLM-based multi-agent collaboration and establishes **\((F+1)\)-robustness** as a sufficient graph condition for containing Byzantine influence.

## Acknowledgements

This repository is built on top of
[**Z1ivan/Byzantine-Fault-Tolerance-in-LLM-MAS**](https://github.com/Z1ivan/Byzantine-Fault-Tolerance-in-LLM-MAS),
the official implementation of the CP-WBFT baseline. We kept their framework and
directory layout (`config/`, `core/`, `methods/`, `tools/`) and added SAC on top
of it, so the two methods run through the same agents, topologies, data loader,
and evaluation code and are directly comparable. We thank the authors for
releasing their code. This repository keeps the same MIT license.

## Installation

```bash
git clone https://github.com/<your-org>/robust_mas.git
cd robust_mas

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9 or newer.

## API key

Copy the template and fill in your key:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-your-key-here
API_KEY=sk-your-key-here
```

The key is read from the environment (`OPENAI_API_KEY`, then `API_KEY`, then
`LLM_API_KEY`), or you can pass `--api-key` on the command line. No key is
committed to this repository, and `.env` is git-ignored.

For a self-hosted OpenAI-compatible server (vLLM, TGI, Ollama) any placeholder
works:

```bash
export OPENAI_API_KEY=EMPTY
```

## Data

```bash
python tools/make_math500_full.py --out data/byzantine/math500/math500_500.json
python tools/make_commonsense.py --out-dir data/byzantine/commonsense
```

The commonsense builder shuffles each validation split with a fixed seed and,
following the paper, subsamples HellaSwag and BoolQ to 1000 questions each; pass
`--no-cap` to keep the entire split. Resulting sizes: ARC-Challenge 299,
HellaSwag 1000, BoolQ 1000, OpenBookQA 500, RTE 277 (3,076 in total).
`winogrande` is also available but is not part of the five-benchmark suite
reported in the paper.

Each question object follows the base repository's byzantine schema:

```json
{
  "question_id": "math500_000",
  "question": "...",
  "correct_answer": "...",
  "weak_answer": "",
  "dataset_type": "math500"
}
```

`weak_answer` is the answer a fixed-mode Byzantine agent replays with confidence
`1.0`. For MATH-500 it is filled by querying the weak model once per question:

```bash
python tools/precompute_weak_answers.py \
    --data data/byzantine/math500/math500_500.json \
    --model qwen2.5-1.5b-instruct \
    --base-url http://127.0.0.1:8001/v1
```

If `weak_answer` is empty, the Byzantine agent deterministically synthesizes a
wrong answer of the correct format, seeded by `(seed, agent_id, question_id)`.
Any pre-generated adversarial answer can also be placed in `weak_answer`
directly.

## Running

Both methods run through the same entry point as the base framework.

### SAC

```bash
python methods/unified_entry.py sac \
  --dataset-type math500 --data-path data/byzantine/math500/math500_500.json \
  --agents 7 --malicious 1 --weak-honest 2 --byzantine-mode fixed \
  --mode all --rounds 6 --adversary-bound 3 \
  --seed 1234 --position-seed 1234 \
  --strong-model qwen3-4b --weak-model qwen2.5-1.5b-instruct \
  --topology merg --output-dir results/math500
```

### CP-WBFT

```bash
python methods/unified_entry.py cp_wbft \
  --dataset-type math500 --data-path data/byzantine/math500/math500_500.json \
  --agents 7 --malicious 1 --weak-honest 2 --byzantine-mode fixed \
  --mode all --rounds 6 \
  --seed 1234 --position-seed 1234 \
  --strong-model qwen3-4b --weak-model qwen2.5-1.5b-instruct \
  --topology merg --output-dir results/math500
```

`cp_wbft` is an alias for the base repository's `prompt_probe` method; both names
work.

For the commonsense benchmarks use `--dataset-type commonsense170k_mix` and
point `--data-path` at the benchmark file:

```bash
python methods/unified_entry.py sac \
  --dataset-type commonsense170k_mix --data-path data/byzantine/commonsense/arc_c.json \
  --agents 7 --malicious 1 --weak-honest 2 --byzantine-mode fixed \
  --mode all --rounds 6 --adversary-bound 3 \
  --seed 1234 --position-seed 1234 \
  --strong-model qwen3-4b --weak-model qwen2.5-1.5b-instruct \
  --topology merg --output-dir results/commonsense/arc_c
```

Results are written under
`<output-dir>/<sac|prompt>/<dataset>/llm/<topology>_<agents>_<malicious>/<experiment_id>/`.
The result JSON contains, per question, the initial answers, the per-round
answer snapshot of every agent, and for SAC the self-scores, retained set, and
filtered set of every agent at every round.

### Model endpoints

Model names and endpoints are read from the environment, matching the base
framework:

```bash
export STRONG_MODELS=qwen3-4b
export WEAK_MODELS=qwen2.5-1.5b-instruct
export STRONG_API_BASE_URL=http://127.0.0.1:8002/v1
export WEAK_API_BASE_URL=http://127.0.0.1:8001/v1
export API_BASE_URL=$STRONG_API_BASE_URL
```

To use the OpenAI API instead, leave the base URLs unset and pass OpenAI model
names to `--strong-model` / `--weak-model`.

### Full sweeps

The paper uses Qwen3-4B as the strong model and Qwen2.5-1.5B-Instruct as the
weak model, served locally through two OpenAI-compatible endpoints:

```bash
bash scripts/serve_vllm.sh
```

Then, in another shell:

```bash
bash scripts/run_math500.sh
bash scripts/run_commonsense.sh
```

Both scripts sweep `{sac, cp_wbft} x {merg, k_circulant, robust_random}` and call
`tools/paper_metrics.py` at the end. Override `DATA`, `DATA_DIR`, `DATASETS`,
`OUT`, `STRONG_MODELS`, `WEAK_MODELS`, and the base URLs through the
environment.

### Closed-model setting

The appendix experiment uses a single strong closed model for all agents, no
weak honest group, and three Byzantine agents:

```bash
python tools/make_hmmt_30q.py --out data/byzantine/hmmt/hmmt25_30.json

python methods/unified_entry.py sac \
  --dataset-type math500 --data-path data/byzantine/hmmt/hmmt25_30.json \
  --agents 7 --malicious 3 --weak-honest 0 --byzantine-mode fixed \
  --mode all --rounds 6 --adversary-bound 3 \
  --seed 1234 --position-seed 1234 \
  --strong-model gpt-5.6-sol --weak-model gpt-5.6-sol \
  --max-tokens 12000 --topology merg --output-dir results/hmmt25
```

Each Byzantine agent replays the pre-generated answer stored in `weak_answer`
with confidence `1.0`. Set `LLM_TIMEOUT` higher (the paper used 2400) for long
reasoning traces.

## Experimental configuration

| Setting | Flag | Paper value |
|---|---|---|
| Agents | `--agents` | 7 |
| Byzantine agents | `--malicious` | 1 (3 in the closed-model setting) |
| Weak honest agents | `--weak-honest` | 2 (0 in the closed-model setting) |
| Adversary bound `F` | `--adversary-bound` | 3 |
| Communication rounds | `--rounds` | 6 |
| Byzantine behavior | `--byzantine-mode` | `fixed` |
| Temperature | `--temperature` | 0.1 |
| Seeds | `--seed`, `--position-seed` | 1234 |

Role assignment is deterministic given the seeds. With the values above the
Byzantine agent is `agent_6`, the weak honest agents are `agent_0` and
`agent_1`, and the strong honest agents are `agent_2` through `agent_5`.


## Metrics

| Metric | Definition |
|---|---|
| IAA | Initial per-agent accuracy, averaged over all 7 agents |
| FAA | Final per-agent accuracy after the last round |
| BFTI | `FAA - IAA`; positive means consensus helped |
| RA | Fraction of questions where the majority over all 7 agents is correct |
| H-Majority | Fraction where the majority over the 6 honest agents is correct |
| W IAA→FAA | Weak honest group accuracy before and after consensus |
| S IAA→FAA | Strong honest group accuracy before and after consensus |



## Repository layout

```
.
├── config/            experiment configurations (sac_config.py is ours)
├── core/
│   ├── agents/        agent implementations and factory
│   ├── consensus/     consensus engine
│   ├── data/          dataset loading
│   ├── evaluation/    metrics
│   ├── experiment_manager/  Byzantine position control, seeding
│   ├── models/        OpenAI-compatible and local model backends
│   ├── results/       result serialization
│   ├── runners/       base_runner, sac_runner (ours), prompt_probe_runner
│   ├── topologies/    communication graphs
│   ├── utils/         math grading, naming
│   └── visualization/ report generation
├── methods/           unified_entry.py, the single entry point
├── data/byzantine/    generated datasets
├── scripts/           serving and sweep scripts
├── tools/             dataset builders and table generation
└── r_robust.py        r-robustness verifier
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{sac2026robust,
  title={Robust Multi-Agent LLMs under Byzantine Faults},
  author={Lee, Haejoon and Yun, Vincent-Daniel and Panagou, Dimitra and Karimireddy, Sai Praneeth},
  journal={arXiv preprint arXiv:2605.09076},
  year={2026}
}
```

Please also cite the base framework this work builds on:
[Z1ivan/Byzantine-Fault-Tolerance-in-LLM-MAS](https://github.com/Z1ivan/Byzantine-Fault-Tolerance-in-LLM-MAS).
