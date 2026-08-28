from typing import Dict, Any, Optional
import logging

from ..interfaces import IAgent, IComponentFactory, AgentType, MethodType
from .base_agent import BaseAgent
from .llm_agent import LLMAgent
from .traditional_agent import TraditionalAgent
try:
    from .decoder_agent import DecoderAgent
except ImportError:
    DecoderAgent = None

try:
    from methods.gsm8k_decoder.probe_training.gsm8k_byzantine.gsm8k_agent import GSM8KAgent
except ImportError:
    GSM8KAgent = None

try:
    from methods.prompt_probe.prompt_probe_agent import PromptProbeAgent
except ImportError:
    PromptProbeAgent = None

try:
    from methods.safe_decoder.safe_byzantine.safe_probe_agent import SafeProbeAgent
except ImportError:
    SafeProbeAgent = None

logger = logging.getLogger(__name__)

class StandardizedAgentFactory:

    def __init__(self):
        self.agent_registry = self._build_agent_registry()

    def _build_agent_registry(self) -> Dict[str, type]:

        registry = {

            "llm": LLMAgent,
            "traditional": TraditionalAgent,
        }

        if DecoderAgent:
            registry["decoder"] = DecoderAgent

        if GSM8KAgent:
            registry["gsm8k"] = GSM8KAgent
            registry["gsm8k_llm"] = GSM8KAgent

        if PromptProbeAgent:
            registry["prompt_probe"] = PromptProbeAgent
            registry["prompt_probe_llm"] = PromptProbeAgent

        if SafeProbeAgent:
            registry["safe"] = SafeProbeAgent
            registry["safe_probe"] = SafeProbeAgent

        return registry

    def create_agent(
        self, 
        agent_id: str, 
        agent_type: AgentType, 
        method_type: MethodType,
        config: Any,
        **kwargs
    ) -> IAgent:

        agent_key = self._determine_agent_key(agent_type, method_type)

        if agent_key not in self.agent_registry:
            if (isinstance(method_type, MethodType) and method_type == MethodType.DECODER) or \
               (hasattr(agent_type, 'name') and str(agent_type.name).lower() == 'decoder'):
                raise RuntimeError(
                    "DecoderAgent is unavailable (torch/transformers or local model dependencies may not be installed).\n"
                    "Please install the required dependencies and run in the local decoder environment, or switch to API mode."
                )
            logger.warning(f"Agent type {agent_key} not found, falling back to generic LLM agent")
            agent_key = "llm"

        agent_class = self.agent_registry[agent_key]

        try:

            agent_params = self._prepare_agent_params(
                agent_id, agent_type, method_type, config, **kwargs
            )

            agent = agent_class(**agent_params)

            logger.info(f"Successfully created agent: {agent_id} (type: {agent_key})")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent: {agent_id}, error: {e}")

            if (isinstance(method_type, MethodType) and method_type == MethodType.DECODER) or \
               (hasattr(agent_type, 'name') and str(agent_type.name).lower() == 'decoder'):
                raise RuntimeError(
                    "DecoderAgent initialization failed; fallback to API model is disabled. "
                    "Please confirm the bztmodel environment is activated and check the local model path/probe weight path.\n"
                    f"Error details: {e}"
                )

            try:
                fallback_params = self._prepare_fallback_params(
                    agent_id, agent_type, config
                )
                agent = LLMAgent(**fallback_params)
                logger.warning(f"Created agent using fallback: {agent_id}")
                return agent
            except Exception as fallback_error:
                logger.error(f"Fallback agent creation also failed: {fallback_error}")
                raise

    def _determine_agent_key(self, agent_type: AgentType, method_type: MethodType) -> str:

        if agent_type == AgentType.TRADITIONAL:
            return "traditional"
        elif agent_type == AgentType.LLM:
            return "llm"

        if method_type == MethodType.PILOT:
            return "gsm8k"
        elif method_type == MethodType.PROMPT_PROBE:
            return "prompt_probe"
        elif method_type == MethodType.DECODER:
            return "decoder"

        return "llm"

    def _prepare_agent_params(
        self, 
        agent_id: str, 
        agent_type: AgentType, 
        method_type: MethodType,
        config: Any, 
        **kwargs
    ) -> Dict[str, Any]:

        base_params = {
            "agent_id": agent_id,

            **kwargs
        }

        if method_type == MethodType.PILOT:
            return self._prepare_gsm8k_params(base_params, config)
        elif method_type == MethodType.PROMPT_PROBE:
            return self._prepare_prompt_probe_params(base_params, config)
        elif method_type == MethodType.DECODER:
            return self._prepare_decoder_params(base_params, config)
        else:
            return self._prepare_generic_params(base_params, config)

    def _prepare_gsm8k_params(self, base_params: Dict[str, Any], config: Any) -> Dict[str, Any]:

        params = base_params.copy()

        if hasattr(config, 'dataset_type'):
            params['dataset_type'] = getattr(config, 'dataset_type', 'gsm8k')

        if hasattr(config, 'model_name'):
            params['model_name'] = config.model_name

        if hasattr(config, 'strong_model'):
            params['strong_model'] = getattr(config, 'strong_model')
        if hasattr(config, 'weak_model'):
            params['weak_model'] = getattr(config, 'weak_model')
        if hasattr(config, 'api_key'):
            params['api_key'] = config.api_key
        if hasattr(config, 'api_base_url'):
            params['api_base_url'] = config.api_base_url
        if hasattr(config, 'temperature'):
            params['temperature'] = config.temperature
        if hasattr(config, 'max_tokens'):
            params['max_tokens'] = config.max_tokens
        if hasattr(config, 'use_local_models'):
            params['use_local_models'] = config.use_local_models
        if hasattr(config, 'llama3_model_path'):
            params['llama3_model_path'] = config.llama3_model_path
        if hasattr(config, 'llama31_model_path'):
            params['llama31_model_path'] = config.llama31_model_path

        params['local_influence_mode'] = 'reflection'

        return params

    def _prepare_prompt_probe_params(self, base_params: Dict[str, Any], config: Any) -> Dict[str, Any]:

        params = base_params.copy()

        if hasattr(config, 'strong_model'):
            params['strong_model'] = config.strong_model
        if hasattr(config, 'weak_model'):
            params['weak_model'] = config.weak_model
        if hasattr(config, 'api_key'):
            params['api_key'] = config.api_key
        if hasattr(config, 'api_base_url'):
            params['api_base_url'] = config.api_base_url
        if hasattr(config, 'temperature'):
            params['temperature'] = config.temperature
        if hasattr(config, 'confidence_extraction_method'):
            params['confidence_extraction_method'] = config.confidence_extraction_method
        if hasattr(config, 'dataset_type'):
            params['dataset_type'] = config.dataset_type

        if hasattr(config, 'use_strong_vs_strong'):
            params['use_strong_vs_strong'] = config.use_strong_vs_strong
        if hasattr(config, 'adversarial_prompt'):
            params['adversarial_prompt'] = config.adversarial_prompt
        if hasattr(config, 'dishonest_confidence'):
            params['dishonest_confidence'] = config.dishonest_confidence
        if hasattr(config, 'byzantine_mode'):
            params['byzantine_mode'] = config.byzantine_mode

        params['local_influence_mode'] = 'confidence'

        return params

    def _prepare_safe_params(self, base_params: Dict[str, Any], config: Any) -> Dict[str, Any]:

        params = base_params.copy()

        if hasattr(config, 'llama3_model_path'):
            params['llama3_model_path'] = config.llama3_model_path
        if hasattr(config, 'llama31_model_path'):
            params['llama31_model_path'] = config.llama31_model_path
        if hasattr(config, 'llama3_lcd_path'):
            params['llama3_lcd_path'] = config.llama3_lcd_path
        if hasattr(config, 'llama31_lcd_path'):
            params['llama31_lcd_path'] = config.llama31_lcd_path
        if hasattr(config, 'use_local_models'):
            params['use_local_models'] = config.use_local_models
        if hasattr(config, 'device'):
            params['device'] = config.device
        if hasattr(config, 'max_new_tokens'):
            params['max_new_tokens'] = config.max_new_tokens
        if hasattr(config, 'do_sample'):
            params['do_sample'] = config.do_sample

        return params

    def _prepare_decoder_params(self, base_params: Dict[str, Any], config: Any) -> Dict[str, Any]:
        params = base_params.copy()

        dataset_type = str(getattr(config, 'dataset_type', 'gsm8k')).lower()
        if dataset_type == 'safe':
            lcd_llama3 = getattr(config, 'safe_llama3_lcd', '')
            lcd_llama31 = getattr(config, 'safe_llama31_lcd', '')
        else:
            lcd_llama3 = getattr(config, 'gsm8k_llama3_lcd', '')
            lcd_llama31 = getattr(config, 'gsm8k_llama31_lcd', '')

        params.update({
            'config': config,
            'dataset_type': dataset_type,
            'llama3_model_path': getattr(config, 'llama3_model_path', ''),
            'llama31_model_path': getattr(config, 'llama31_model_path', ''),
            'lcd_llama3_path': lcd_llama3,
            'lcd_llama31_path': lcd_llama31,
            'max_new_tokens': getattr(config, 'max_new_tokens', 512),
            'do_sample': getattr(config, 'do_sample', False),
            'temperature': getattr(config, 'temperature', 0.0),
            'seed': getattr(config, 'seed', 1234),
        })
        return params

    def _prepare_generic_params(self, base_params: Dict[str, Any], config: Any) -> Dict[str, Any]:

        params = base_params.copy()

        if hasattr(config, 'api_key'):
            params['api_key'] = config.api_key
        if hasattr(config, 'api_base_url'):
            params['api_base_url'] = config.api_base_url
        if hasattr(config, 'temperature'):
            params['temperature'] = config.temperature
        if hasattr(config, 'max_tokens'):
            params['max_tokens'] = config.max_tokens
        if hasattr(config, 'dishonest_confidence'):
            params['dishonest_confidence'] = config.dishonest_confidence
        if hasattr(config, 'byzantine_mode'):
            params['byzantine_mode'] = config.byzantine_mode
        if hasattr(config, 'dataset_type'):
            params['dataset_type'] = config.dataset_type
        if hasattr(config, 'strong_model'):
            params['strong_model'] = config.strong_model
        if hasattr(config, 'weak_model'):
            params['weak_model'] = config.weak_model
        if hasattr(config, 'seed'):
            params['seed'] = config.seed

        return params

    def _prepare_fallback_params(
        self, 
        agent_id: str, 
        agent_type: AgentType, 
        config: Any
    ) -> Dict[str, Any]:

        return {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "api_key": getattr(config, 'api_key', None),

            "api_base_url": getattr(config, 'api_base_url', 'https://api.openai.com/v1'),
            "temperature": getattr(config, 'temperature', 0.1),
            "max_tokens": getattr(config, 'max_tokens', 1000),
        }

    def get_supported_agent_types(self) -> Dict[str, str]:

        return {
            key: cls.__name__ for key, cls in self.agent_registry.items()
        }

    def validate_agent_config(
        self, 
        agent_type: AgentType, 
        method_type: MethodType, 
        config: Any
    ) -> bool:

        try:
            agent_key = self._determine_agent_key(agent_type, method_type)

            if agent_key not in self.agent_registry:
                logger.warning(f"Unknown agent type: {agent_key}")
                return False

            required_attrs = ['api_key', 'api_base_url']
            for attr in required_attrs:
                if not hasattr(config, attr):
                    logger.warning(f"Configuration missing required attribute: {attr}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

_agent_factory = None

def get_agent_factory() -> StandardizedAgentFactory:

    global _agent_factory
    if _agent_factory is None:
        _agent_factory = StandardizedAgentFactory()
    return _agent_factory

def create_agent(
    agent_id: str, 
    agent_type: AgentType, 
    method_type: MethodType,
    config: Any,
    **kwargs
) -> IAgent:

    factory = get_agent_factory()
    return factory.create_agent(agent_id, agent_type, method_type, config, **kwargs)

def validate_agent_config(
    agent_type: AgentType, 
    method_type: MethodType, 
    config: Any
) -> bool:

    factory = get_agent_factory()
    return factory.validate_agent_config(agent_type, method_type, config)

if __name__ == "__main__":

    print("Standardized agent factory test")

    factory = get_agent_factory()
    supported_types = factory.get_supported_agent_types()

    print("Supported agent types:")
    for key, class_name in supported_types.items():
        print(f"  {key}: {class_name}")

    print("Agent factory initialization complete")
