# Robust Multi-Agent LLMs under Byzantine Faults

Official implementation of **Robust Multi-Agent LLMs under Byzantine Faults**.

[![arXiv](https://img.shields.io/badge/arXiv-2605.09076-b31b1b.svg)](https://arxiv.org/abs/2605.09076)

## Overview

Large Language Model Multi-Agent Systems (LLM-MAS) can benefit from collaboration among multiple agents, but their communication also introduces vulnerabilities when some agents are faulty or adversarial.



We propose **Self-Anchored Consensus (SAC)**, a fully decentralized filter-and-refine protocol designed to make LLM multi-agent systems robust against Byzantine agents.

SAC enables each agent to:

- independently evaluate neighboring responses using receiver-side confidence scores,
- filter potentially unreliable messages,
- refine its own response using trusted neighboring information, and
- operate over communication graphs with principled robustness guarantees.

Our framework connects Byzantine-resilient consensus theory with LLM-based multi-agent collaboration and establishes **\((F+1)\)-robustness** as a sufficient graph condition for containing Byzantine influence.

## Code Release

**Code is coming soon.**

We are currently preparing the implementation, evaluation scripts, prompts, and experiment configurations for public release.


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{sac2026robust,
  title={Robust Multi-Agent LLMs under Byzantine Faults},
  author={Lee, Haejoon and Yun, Vincent-Daniel and Panagou, Dimitra and Karimireddy, Sai Praneeth},
  journal={arXiv preprint arXiv:2605.09076},
  year={2026}
}
