import argparse
from typing import Optional, List
from dataclasses import dataclass, field
from .base_config import BaseConfig, BaseConfigManager


@dataclass
class SACConfig(BaseConfig):

    topology: str = "complete"
    agents: int = 7
    malicious: int = 6
    agent_type: str = "llm"

    mode: str = "single"
    rounds: int = 3           # SAC consensus rounds (T in the paper)
    question: Optional[str] = None

    adversary_bound: int = -1  # F in the paper; -1 means use malicious count

    position_strategy: str = "random"
    specific_positions: Optional[List[int]] = None
    position_seed: Optional[int] = None
    force_random: bool = False

    save: bool = True
    visualize: bool = True
    output_name: Optional[str] = None

    dataset_type: str = "gsm8k"
    data_path: Optional[str] = None
    output_dir: str = "results"

    strong_model: str = "gpt-4o-mini"
    weak_model: str = "gpt-3.5-turbo"
    dishonest_confidence: str = "none"  # none | high(1.0) | medium(0.5) | low(0.2) | random | float
    byzantine_mode: str = "model"  # model | fixed (fixed: use dataset weak_answer, confidence=1.0, no LLM call)
    weak_honest: int = 0  # number of honest agents assigned weak model


    api_base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None

    temperature: float = 0.1
    max_tokens: int = 1000
    seed: int = 1234

    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

    def get_adversary_bound(self) -> int:
        if self.adversary_bound >= 0:
            return self.adversary_bound
        return self.malicious

    def validate(self) -> List[str]:
        errors = []
        if self.agents <= 0:
            errors.append("agents must be > 0")
        if self.malicious < 0:
            errors.append("malicious must be >= 0")
        if self.malicious >= self.agents:
            errors.append("malicious must be < agents")
        if self.rounds <= 0:
            errors.append("rounds must be > 0")
        return errors


def create_sac_config(args_list: Optional[List[str]] = None) -> SACConfig:
    parser = argparse.ArgumentParser(description="Self-Anchored Consensus (SAC)")

    parser.add_argument("--dataset-type", default="gsm8k", choices=["gsm8k", "safe", "commonsense", "math500", "commonsense170k", "commonsense170k_mix"])
    parser.add_argument("--data-path", default=None, help="Custom data file path")
    parser.add_argument("--topology", default="complete",
                        choices=["complete", "star", "chain", "tree", "random", "layered_graph", "merg", "k_circulant", "preferential", "robust_random"])
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--malicious", type=int, default=6)
    parser.add_argument("--mode", required=True, choices=["single", "all"])
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--adversary-bound", type=int, default=-1,
                        help="F in the paper. Defaults to --malicious value.")
    parser.add_argument("--strong-model", default="gpt-4o-mini")
    parser.add_argument("--weak-model", default="gpt-3.5-turbo")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--position-strategy", default="random")
    parser.add_argument("--position-seed", type=int, default=None)
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false")
    parser.add_argument("--question", default=None)
    parser.add_argument("--dishonest-confidence",
                        type=str, default="none",
                        help="Dishonest-confidence adversary mode: none | high (1.0) | medium (0.5) | low (0.2) | random | float value")
    parser.add_argument("--byzantine-mode", type=str, default="model",
                        choices=["model", "fixed"],
                        help="Byzantine agent behavior: model (use weak LLM) | fixed (use dataset weak_answer, confidence=1.0, no LLM call)")
    parser.add_argument("--weak-honest", type=int, default=0,
                        help="Number of honest agents to assign weak model (default: 0)")

    args = parser.parse_args(args_list if args_list is not None else [])

    import os
    api_key = args.api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base_url = args.api_base_url or os.environ.get("API_BASE_URL", "https://api.openai.com/v1")

    return SACConfig(
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        topology=args.topology,
        agents=args.agents,
        malicious=args.malicious,
        mode=args.mode,
        rounds=args.rounds,
        adversary_bound=args.adversary_bound,
        strong_model=args.strong_model,
        weak_model=args.weak_model,
        api_key=api_key,
        api_base_url=api_base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        position_strategy=args.position_strategy,
        position_seed=args.position_seed,
        save=args.save,
        visualize=args.visualize,
        output_dir=args.output_dir,
        question=args.question,
        dishonest_confidence=args.dishonest_confidence,
        byzantine_mode=args.byzantine_mode,
        weak_honest=args.weak_honest,
    )
