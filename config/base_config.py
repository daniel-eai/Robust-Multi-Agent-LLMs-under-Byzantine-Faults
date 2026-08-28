
import argparse
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from .global_params import GlobalParams
except ImportError:

    GlobalParams = None

@dataclass
class BaseConfig:

    topology: str = "complete"
    agents: int = 5
    malicious: int = 1
    agent_type: str = "llm"

    strong_model: Optional[str] = None
    weak_model: Optional[str] = None

    dataset_type: str = "gsm8k"                    
    data_path: Optional[str] = None                 

    mode: str = "all"                    
    rounds: int = 1
    question: Optional[str] = None

    position_strategy: str = "random"
    specific_positions: Optional[List[int]] = None
    position_seed: Optional[int] = None
    force_random: bool = False

    save: bool = True
    visualize: bool = True
    output_name: Optional[str] = None
    save_detailed_data: bool = False

    api_key: Optional[str] = None                 
    api_base_url: str = "https://api.openai.com/v1"             

    @staticmethod
    def get_model_path(model_name: str) -> str:

        if GlobalParams is None:
            raise ImportError("GlobalParams not available")

        if model_name.lower() == "llama3":
            return GlobalParams.LLAMA3_MODEL_PATH
        elif model_name.lower() == "llama31":
            return GlobalParams.LLAMA31_MODEL_PATH
        else:
            raise ValueError(f"Unknown model: {model_name}")

SUPPORTED_TOPOLOGIES = [
    "complete", "star", "chain", "tree", "random", "dynamic", "layered_graph", "merg",
    "k_circulant", "preferential", "robust_random"
]

POSITION_STRATEGIES = [
    "random", "star_center", "star_leaf", 
    "tree_root", "tree_internal", "tree_leaf",
    "chain_head", "chain_middle", "chain_tail",
    "layered_top", "layered_middle", "layered_bottom",
    "high_centrality", "low_centrality", "high_degree", "low_degree"
]

AGENT_TYPES = ["traditional", "llm"]

TEST_MODES = ["single", "all"]

class BaseConfigManager:

    def __init__(self, config_class=BaseConfig):
        self.config_class = config_class
        self._config = None

    def create_base_parser(self, description: str = "Byzantine Fault Tolerance System") -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        parser.add_argument("--dataset-type", choices=["gsm8k", "safe", "commonsense", "math500", "commonsense170k", "commonsense170k_mix"],
                           default="gsm8k", help="Dataset type")
        parser.add_argument("--data-path", default=None,
                           help="Custom data file path (optional, takes priority over default path)")

        parser.add_argument("--topology", choices=SUPPORTED_TOPOLOGIES,
                           default="complete", help="Network topology type")
        parser.add_argument("--agents", type=int, default=5, help="Total number of agents")
        parser.add_argument("--malicious", type=int, default=1, help="Number of malicious agents")
        parser.add_argument("--agent-type", choices=AGENT_TYPES,
                           default="llm", help="Agent type")

        parser.add_argument("--strong-model",
                           default=None,
                           help="Strong model name (used by normal/non-malicious agents)")
        parser.add_argument("--weak-model",
                           default=None,
                           help="Weak model name (used by malicious/weak agents)")

        parser.add_argument("--mode", choices=TEST_MODES, required=True,
                           help="Test mode: 'single' tests 1 question, 'all' tests all questions")
        parser.add_argument("--rounds", type=int, required=True,
                           help="Number of consensus rounds per question")
        parser.add_argument("--question", help="Specify question ID (only in single mode)")

        parser.add_argument("--position-strategy", choices=POSITION_STRATEGIES,
                           default="random", help="Malicious node position strategy")
        parser.add_argument("--specific-positions", type=int, nargs='+',
                           help="Specify malicious node positions (e.g.: --specific-positions 0 2)")
        parser.add_argument("--position-seed", type=int,
                           help="Position random seed")
        parser.add_argument("--force-random", action="store_true",
                           help="Force random position selection")

        parser.add_argument("--save", action="store_true", default=True,
                           help="Save experiment results")
        parser.add_argument("--no-save", dest="save", action="store_false",
                           help="Do not save experiment results")
        parser.add_argument("--visualize", action="store_true", default=True,
                           help="Generate visualization charts")
        parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                           help="Do not generate visualization charts")
        parser.add_argument("--output-name", help="Custom output file name")
        parser.add_argument("--save-detailed-data", action="store_true",
                           help="Save detailed process data")

        parser.add_argument("--llama3-model-path",
                           default="models/LLama-3-8B-Instruct",
                           help="LLaMA3 local model path")
        parser.add_argument("--llama31-model-path",
                           default="models/LLama-3.1-8B-Instruct",
                           help="LLaMA3.1 local model path")

        parser.add_argument("--api-key", help="API key")
        parser.add_argument("--api-base-url",
                           default="https://api.openai.com/v1",
                           help="API base URL (OpenAI-compatible interface, e.g. official OpenAI or compatible proxy gateway)")
        parser.add_argument("--temperature", type=float, default=None,
                           help="Generation temperature (default determined by method/dataset)")
        parser.add_argument("--max-tokens", type=int, default=1000,
                           help="Maximum number of generation tokens")
        parser.add_argument("--seed", type=int, default=1234,
                           help="Random seed")

        return parser

    def parse_args_to_config(self, args: argparse.Namespace) -> BaseConfig:    

        api_key = (
            args.api_key
            or os.getenv("OPENAI_COMPATIBILITY_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
        )

        api_base_url = args.api_base_url or os.getenv("API_BASE_URL") or "https://api.openai.com/v1"

        return self.config_class(
            dataset_type=getattr(args, 'dataset_type', 'gsm8k'),
            data_path=getattr(args, 'data_path', None),
            topology=args.topology,
            agents=args.agents,
            malicious=args.malicious,
            agent_type=args.agent_type,
            strong_model=getattr(args, 'strong_model', None),
            weak_model=getattr(args, 'weak_model', None),
            mode=args.mode,
            rounds=args.rounds,
            question=args.question,
            position_strategy=args.position_strategy,
            specific_positions=args.specific_positions,
            position_seed=args.position_seed,
            force_random=args.force_random,
            save=args.save,
            visualize=args.visualize,
            output_name=args.output_name,
            save_detailed_data=args.save_detailed_data,
            api_key=api_key,
            api_base_url=api_base_url
        )

    def validate_config(self, config: BaseConfig) -> List[str]:

        return self.validate_base_config(config)

    def validate_base_config(self, config: BaseConfig) -> List[str]:

        errors = []

        if config.agents < 2:
            errors.append("Number of agents must be at least 2")

        if config.malicious >= config.agents:
            errors.append("Number of malicious agents cannot be greater than or equal to the total count")

        if config.rounds < 1:
            errors.append("Number of consensus rounds must be at least 1")

        malicious_ratio = config.malicious / config.agents
        if malicious_ratio > 0.33:
            print(f"Warning: Malicious node ratio ({malicious_ratio:.1%}) exceeds the theoretical fault tolerance limit (33%); experiment results may not be fault tolerant.")

        if config.specific_positions:
            if len(config.specific_positions) != config.malicious:
                errors.append(f"Number of specified positions ({len(config.specific_positions)}) does not match number of malicious agents ({config.malicious})")

            if any(pos >= config.agents for pos in config.specific_positions):
                errors.append(f"Specified positions exceed the agent count range (0-{config.agents-1})")

            if len(set(config.specific_positions)) != len(config.specific_positions):
                errors.append("Specified positions cannot be duplicated")

        paths_to_check = [
            (config.llama3_model_path, "LLaMA3 model path"),
            (config.llama31_model_path, "LLaMA3.1 model path"),
        ]

        for path, name in paths_to_check:
            if not Path(path).exists():
                errors.append(f"{name} does not exist: {path}")

        return errors

    def print_base_config_summary(self, config: BaseConfig) -> None:

        print("Byzantine Fault Tolerance System Configuration:")
        print(f"   Topology: {config.topology}")
        print(f"   Total agents: {config.agents}")
        print(f"   Malicious agents: {config.malicious} ({config.malicious/config.agents:.1%})")
        print(f"   Agent type: {config.agent_type}")
        print(f"   Test mode: {config.mode}")
        print(f"   Consensus rounds: {config.rounds}")
        print(f"   Position strategy: {config.position_strategy}")

        if config.specific_positions:
            print(f"   Specified positions: {config.specific_positions}")
        if config.position_seed:
            print(f"   Position seed: {config.position_seed}")

        print(f"   Save results: {'Yes' if config.save else 'No'}")
        print(f"   Generate visualizations: {'Yes' if config.visualize else 'No'}")
        print(f"   API base URL: {config.api_base_url}")
        print(f"   Temperature: {config.temperature}")

def create_base_config_from_args(args_list: Optional[List[str]] = None, description: str = "Byzantine Fault Tolerance System") -> BaseConfig:
    manager = BaseConfigManager()
    parser = manager.create_base_parser(description)
    args = parser.parse_args(args_list)
    return manager.parse_args_to_config(args)

def validate_and_print_config(config: BaseConfig, config_name: str = "System") -> bool:
    manager = BaseConfigManager()
    errors = manager.validate_base_config(config)

    if errors:
        print(f"{config_name} configuration validation failed:")
        for error in errors:
            print(f"   Error: {error}")
        return False

    print(f"{config_name} configuration validation passed")
    manager.print_base_config_summary(config)
    return True

if __name__ == "__main__":

    print("Testing base configuration system")
    config = create_base_config_from_args("Base Configuration Test")

    if validate_and_print_config(config, "Base"):
        print("\nConfiguration system is working correctly")
    else:
        print("\nConfiguration system has issues")
