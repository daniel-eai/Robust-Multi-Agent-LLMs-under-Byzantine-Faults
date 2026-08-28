#!/usr/bin/env python3

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.unified_config_manager import UnifiedConfigManager, MethodType as ConfigMethodType
from config.prompt_probe_config import create_prompt_probe_config
from config.decoder_probe_config import create_decoder_probe_config
from config.base_config import BaseConfig
from core.interfaces import MethodType

logger = logging.getLogger(__name__)

def _normalize_proxy_env() -> None:

    try:
        pairs = [("HTTP_PROXY", "http_proxy"), ("HTTPS_PROXY", "https_proxy")]
        for upper_key, lower_key in pairs:
            upper_val = os.environ.get(upper_key)
            lower_val = os.environ.get(lower_key)
            if upper_val and not lower_val:
                os.environ[lower_key] = upper_val
            if lower_val and not upper_val:
                os.environ[upper_key] = lower_val
    except Exception:

        pass

def _load_env_files() -> None:

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:

        return

    env_candidates = [project_root / ".env", Path.cwd() / ".env"]
    for env_path in env_candidates:
        try:
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=False)
        except Exception:

            continue

    _normalize_proxy_env()

_load_env_files()

class UnifiedMethodEntry:

    def __init__(self):
        self.config_manager = UnifiedConfigManager()
        logger.info("Initializing unified method entry point")

    async def run_experiment(self, method: str, config_path: Optional[str] = None, extra_args: Optional[list] = None, **kwargs) -> Any:

        method_type = self._parse_method_type(method)

        logger.info(f"Starting {method_type.value} experiment...")

        try:

            if config_path:
                config = self._load_config_from_file(method_type, config_path)
            else:
                config = self._parse_command_line_config(method_type, extra_args=extra_args or [], **kwargs)

            result = await self._route_to_runner(method_type, config)

            logger.info(f"{method_type.value} experiment complete: {result.experiment_id}")
            return result

        except Exception as e:
            logger.error(f"{method_type.value} experiment failed: {e}")
            raise

    def _parse_method_type(self, method: str) -> MethodType:

        method_lower = method.lower().replace('_', '').replace('-', '')

        if method_lower in ['pilot']:
            return MethodType.PILOT
        elif method_lower in ['promptprobe', 'probe', 'prompt', 'cpwbft']:
            return MethodType.PROMPT_PROBE
        elif method_lower in ['safe', 'safedecoder']:
            raise NotImplementedError("safe method is disabled")
        elif method_lower in ['decoderprobe', 'decoder', 'hcp']:
            return MethodType.DECODER
        elif method_lower in ['sac', 'self_anchored', 'selfanchored']:
            return MethodType.SAC
        else:
            raise ValueError(f"Unsupported method type: {method}")

    def _load_config_from_file(self, method_type: MethodType, config_path: str) -> Any:

        logger.debug(f"Loading {method_type.value} config from file: {config_path}")

        def _to_config_method_type(mt: MethodType) -> ConfigMethodType:
            return ConfigMethodType(mt.value)

        config = self.config_manager.load_config(
            _to_config_method_type(method_type), config_file=config_path
        )

        if not config:
            raise ValueError(f"Failed to load {method_type.value} config file: {config_path}")

        return config

    def _parse_command_line_config(self, method_type: MethodType, extra_args: Optional[list] = None, **kwargs) -> Any:

        logger.debug(f"Parsing {method_type.value} command-line config")

        if method_type == MethodType.PILOT:

            from config.base_config import create_base_config_from_args
            return create_base_config_from_args(args_list=extra_args or [])
        elif method_type == MethodType.PROMPT_PROBE:
            return create_prompt_probe_config(extra_args)
        elif method_type == MethodType.DECODER:
            return create_decoder_probe_config(extra_args)
        elif method_type == MethodType.SAC:
            from config.sac_config import create_sac_config
            return create_sac_config(extra_args)
        else:
            raise ValueError(f"Unsupported method type: {method_type}")

    async def _route_to_runner(self, method_type: MethodType, config: Any) -> Any:

        logger.debug(f"Routing to {method_type.value} runner")

        if method_type == MethodType.PILOT:

            from core.runners.pilot_runner import run_pilot_experiment as run_gsm8k_experiment
            return await run_gsm8k_experiment(config)
        elif method_type == MethodType.PROMPT_PROBE:

            from core.runners.prompt_probe_runner import run_prompt_probe_experiment
            return await run_prompt_probe_experiment(config)
        elif method_type == MethodType.DECODER:
            from core.runners.decoder_probe_runner import run_decoder_probe_experiment
            return await run_decoder_probe_experiment(config)
        elif method_type == MethodType.SAC:
            from core.runners.sac_runner import run_sac_experiment
            return await run_sac_experiment(config)
        else:
            raise ValueError(f"Unsupported method type: {method_type}")

    def list_supported_methods(self) -> list:

        return [
            {
                "name": "pilot",
                "aliases": [],
                "description": "Pilot experiment unified entry point"
            },
            {
                "name": "prompt_probe",
                "aliases": ["prompt-probe", "probe", "cp_wbft"],
                "description": "CP-WBFT confidence-probing baseline"
            },

            {
                "name": "decoder_probe",
                "aliases": ["decoder", "hcp"],
                "description": "Local decoder hidden-layer confidence probe (HCP), supports GSM8K/SAFE"
            },
            {
                "name": "sac",
                "aliases": ["self_anchored"],
                "description": "Self-Anchored Consensus (SAC): receiver-side scoring with filter-and-refine"
            }
        ]

    def get_method_help(self, method: str) -> str:

        try:
            method_type = self._parse_method_type(method)

            if method_type == MethodType.PILOT:

                from config.base_config import BaseConfigManager
                parser = BaseConfigManager().create_base_parser(description="Pilot experiment method")
                return parser.format_help()
            elif method_type == MethodType.PROMPT_PROBE:

                from config.prompt_probe_config import PromptProbeConfigManager
                parser = PromptProbeConfigManager().create_parser()
                return parser.format_help()
            elif method_type == MethodType.DECODER:

                from config import decoder_probe_config as dpc
                parser = dpc._create_parser()  # type: ignore[attr-defined]
                return parser.format_help()
            else:
                return f"Unknown method: {method}"
        except Exception as e:
            return f"Failed to get help: {e}"

_entry_instance = None

def get_unified_entry() -> UnifiedMethodEntry:

    global _entry_instance
    if _entry_instance is None:
        _entry_instance = UnifiedMethodEntry()
    return _entry_instance

async def run_method_experiment(method: str, config_path: Optional[str] = None, **kwargs) -> Any:

    entry = get_unified_entry()
    return await entry.run_experiment(method, config_path, **kwargs)

def main():

    import argparse

    if "--list-methods" in sys.argv:
        entry = get_unified_entry()
        methods = entry.list_supported_methods()
        print("Supported methods:")
        for method in methods:
            print(f"  {method['name']}: {method['description']}")
            if method['aliases']:
                print(f"    Aliases: {', '.join(method['aliases'])}")
        return

    if "--method-help" in sys.argv:
        entry = get_unified_entry()
        try:
            idx = sys.argv.index("--method-help")
            if idx + 1 < len(sys.argv):
                method_name = sys.argv[idx + 1]
                help_text = entry.get_method_help(method_name)
                print(help_text)
            else:
                print("Error: --method-help requires a method name")
                sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    if len(sys.argv) >= 2 and sys.argv[1] in ["pilot", "prompt_probe", "decoder_probe", "decoder"] and "--help" in sys.argv:
        entry = get_unified_entry()
        help_text = entry.get_method_help(sys.argv[1])
        print(help_text)
        return

    parser = argparse.ArgumentParser(
        description="Byzantine project unified method entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )

    parser.add_argument(
        "method",
        choices=["sac", "prompt_probe", "cp_wbft", "pilot", "decoder_probe", "decoder"],
        help="Method to run (pilot: pilot experiment, prompt_probe: prompt probe, decoder_probe/decoder: decoder probe)"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Config file path"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level"
    )

    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show help message"
    )

    args, unknown_args = parser.parse_known_args()

    if args.help:
        if args.method:
            entry = get_unified_entry()
            help_text = entry.get_method_help(args.method)
            print(help_text)
        else:
            parser.print_help()
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    entry = get_unified_entry()

    async def run_experiment():
        try:
            result = await entry.run_experiment(args.method, args.config, extra_args=unknown_args)

            print(f"\nExperiment complete!")
            print(f"Experiment ID: {result.experiment_id}")
            print(f"Method type: {result.method_type.value}")
            print(f"Agents: {result.agent_count}")
            print(f"Malicious agents: {result.malicious_count}")
            print(f"Execution time: {result.execution_time:.2f}s")

            if result.evaluation_metrics:
                overall = result.evaluation_metrics.get('overall_assessment', {})
                if 'accuracy' in overall:
                    print(f"Accuracy: {overall['accuracy']:.4f}")
                if 'consensus_rate' in overall:
                    print(f"Consensus rate: {overall['consensus_rate']:.4f}")

                safety = result.evaluation_metrics.get('safety_assessment', {})
                if safety:
                    print(f"Safety rate: {safety.get('safety_rate', 'N/A')}")

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            sys.exit(1)

    asyncio.run(run_experiment())

if __name__ == "__main__":
    main()
