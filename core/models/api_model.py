
import os
import logging
from typing import Dict, Any, Optional, List
from .base_model import BaseModel

try:
    from config.unified_config_manager import get_global_config_manager
except ImportError:
    try:
        from config import get_config
    except ImportError:

        def get_config():
            return type('Config', (), {
                'llm_config': {},
                'model_config': {},
                'experiment_config': {}
            })()

logger = logging.getLogger(__name__)

class APIModel(BaseModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.api_key = kwargs.get('api_key') or self._get_api_key()

        self.api_base_url = kwargs.get('api_base_url', 'https://api.openai.com/v1')
        # Per-model base URL override via env vars. Lets us run strong/weak
        # models on different local vLLM servers without touching the agent
        # factory: any model whose name appears in STRONG_MODELS uses
        # STRONG_API_BASE_URL, similarly for WEAK_MODELS / WEAK_API_BASE_URL.
        self.api_base_url = self._resolve_per_model_url(self.api_base_url)
        self.use_camel = kwargs.get('use_camel', False)

        if not self.api_key:
            raise ValueError("API key not provided; please set environment variable or pass api_key parameter")

        if self.use_camel:
            self._prepare_camel_config()

        logger.info(f"API model initialized: {self.model_name}, base URL: {self.api_base_url}")

    def _resolve_per_model_url(self, default_url: str) -> str:
        name = (self.model_name or "").lower()
        strong_models = {m.strip().lower() for m in (os.getenv("STRONG_MODELS", "") or "").split(",") if m.strip()}
        weak_models = {m.strip().lower() for m in (os.getenv("WEAK_MODELS", "") or "").split(",") if m.strip()}
        if name in strong_models:
            url = os.getenv("STRONG_API_BASE_URL")
            if url:
                return url
        if name in weak_models:
            url = os.getenv("WEAK_API_BASE_URL")
            if url:
                return url
        return default_url

    def _get_api_key(self) -> Optional[str]:

        try:
            if hasattr(self.config, 'llm_config') and self.config.llm_config:
                return self.config.llm_config.get('api_key')
            elif isinstance(self.config, dict) and 'api_key' in self.config:
                return self.config['api_key']
        except:
            pass

        api_keys = [
            'OPENAI_API_KEY',
            'OPENAI_COMPATIBILITY_API_KEY', 
            'API_KEY'
        ]

        for key_name in api_keys:
            api_key = os.getenv(key_name)
            if api_key:
                logger.debug(f"Obtained API key from environment variable {key_name}")
                return api_key

        return None

    def _prepare_camel_config(self):

        self._camel_config = {
            'model_name': self.model_name,
            'api_key': self.api_key,
            'api_base_url': self.api_base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout
        }
        logger.debug("Camel framework configuration ready")

    async def _generate_impl(self, prompt: str, **kwargs) -> str:

        if self.use_camel:
            return await self._generate_with_camel(prompt, **kwargs)
        else:
            return await self._generate_direct_api(prompt, **kwargs)

    def _get_effective_model_name(self, raw_model_name: Optional[str] = None) -> str:

        model_name = (raw_model_name or self.model_name or "").strip()
        return model_name

    def _is_reasoning_model(self, model_name: Optional[str] = None) -> bool:
        """OpenAI reasoning models (o1*, o3*, gpt-5*) reject classic chat-completions params."""
        m = (model_name or self.model_name or "").lower()
        return (
            m.startswith("gpt-5")
            or m.startswith("o1")
            or m.startswith("o3")
            or m.startswith("o4")
        )

    def _build_request(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Build a chat-completions request, switching to reasoning-model schema where required."""
        if self._is_reasoning_model(model):
            return {
                "model": model,
                "messages": messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": False,
            }
        req = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", getattr(self, "top_p", 0.9)),
            "frequency_penalty": kwargs.get("frequency_penalty", getattr(self, "frequency_penalty", 0.0)),
            "presence_penalty": kwargs.get("presence_penalty", getattr(self, "presence_penalty", 0.0)),
            "seed": kwargs.get("seed", self.seed),
            "stream": False,
        }
        # For Qwen3 hybrid models, optionally disable the thinking block at
        # serving time. vLLM exposes the chat-template kwarg directly on the
        # request body.
        if "qwen3" in (model or "").lower():
            et = kwargs.get("enable_thinking", None)
            if et is not None:
                # Per-call routing (AIME thinking experiment): honor explicit flag.
                req["chat_template_kwargs"] = {"enable_thinking": bool(et)}
            elif os.getenv("DISABLE_QWEN3_THINKING", "1") != "0":
                req["chat_template_kwargs"] = {"enable_thinking": False}
        return req

    async def _generate_with_camel(self, prompt: str, **kwargs) -> str:

        try:

            from camel.models import ModelFactory
            from camel.messages import BaseMessage
            from camel.types import ModelPlatformType

            if not hasattr(self, '_camel_config'):
                self._prepare_camel_config()

            model_name_lower = self.model_name.lower()

            if 'gpt-4' in model_name_lower:
                model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
                if 'mini' in model_name_lower:
                    model_type = "gpt-4o-mini"
                else:
                    model_type = "gpt-4o"
            elif 'gpt-3.5' in model_name_lower:
                model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
                model_type = "gpt-3.5-turbo"
            elif 'deepseek' in model_name_lower:
                model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
                model_type = self.model_name
            else:

                model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
                model_type = self.model_name

            model_config = {
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'seed': kwargs.get('seed', self.seed),
            }

            model = ModelFactory.create(
                model_platform=model_platform,
                model_type=model_type,
                api_key=self.api_key,
                url=self.api_base_url,
                model_config_dict=model_config
            )

            message = BaseMessage.make_user_message(
                role_name="user",
                content=prompt
            )

            response = model.run([message])

            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'msg'):
                return response.msg.content
            else:
                return str(response)

        except ImportError:
            logger.warning("Camel framework not installed, falling back to direct API call")
            return await self._generate_direct_api(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Camel framework call failed: {e}")

            return await self._generate_direct_api(prompt, **kwargs)

    async def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:

        if self.use_camel:

            return await super().generate_with_messages(messages, **kwargs)
        else:

            return await self._generate_direct_api_with_messages(messages, **kwargs)

    async def _generate_direct_api(self, prompt: str, **kwargs) -> str:

        try:

            effective_model = self._get_effective_model_name()
            request_data = self._build_request(
                effective_model,
                [{"role": "user", "content": prompt}],
                **kwargs,
            )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            url = f"{self.api_base_url.rstrip('/')}/chat/completions"

            response_data = await self._network_client.post_json(
                url=url,
                data=request_data,
                headers=headers,
                timeout=kwargs.get('timeout', self.timeout)
            )

            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']

                if 'usage' in response_data:
                    u = response_data['usage']
                    self._stats['total_tokens'] += u.get('total_tokens', 0)
                    import os as _ousg
                    _ul = _ousg.getenv('USAGE_LOG')
                    if _ul:
                        try:
                            with open(_ul, 'a') as _uf:
                                _uf.write(f"{u.get('prompt_tokens',0)},{u.get('completion_tokens',0)},{u.get('total_tokens',0)}\n")
                        except Exception:
                            pass

                return content
            else:
                raise ValueError("API response format is incorrect")

        except Exception as e:
            logger.error(f"Direct API call failed: {e}")
            raise

    async def _generate_direct_api_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:

        try:

            effective_model = self._get_effective_model_name()
            request_data = self._build_request(effective_model, messages, **kwargs)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            url = f"{self.api_base_url.rstrip('/')}/chat/completions"

            response_data = await self._network_client.post_json(
                url=url,
                data=request_data,
                headers=headers,
                timeout=kwargs.get('timeout', self.timeout)
            )

            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']

                if 'usage' in response_data:
                    u = response_data['usage']
                    self._stats['total_tokens'] += u.get('total_tokens', 0)
                    import os as _ousg
                    _ul = _ousg.getenv('USAGE_LOG')
                    if _ul:
                        try:
                            with open(_ul, 'a') as _uf:
                                _uf.write(f"{u.get('prompt_tokens',0)},{u.get('completion_tokens',0)},{u.get('total_tokens',0)}\n")
                        except Exception:
                            pass

                return content
            else:
                raise ValueError("API response format is incorrect")

        except Exception as e:
            logger.error(f"API call with message list failed: {e}")
            raise

    def get_cost_estimate(self) -> Dict[str, Any]:

        cost_per_1k_tokens = {
            'gpt-4o-mini': 0.0015,
            'gpt-3.5-turbo': 0.002,
            'deepseek-chat': 0.0014,
        }

        base_cost = cost_per_1k_tokens.get(self.model_name, 0.002)
        total_cost = (self._stats['total_tokens'] / 1000) * base_cost

        return {
            'total_tokens': self._stats['total_tokens'],
            'estimated_cost': total_cost,
            'cost_per_1k_tokens': base_cost,
            'currency': 'USD'
        }

    def get_model_info(self) -> Dict[str, Any]:

        base_info = super().get_model_info()

        api_info = {
            'model_type': 'API',
            'api_base_url': self.api_base_url,
            'use_camel': self.use_camel,
            'cost_estimate': self.get_cost_estimate()
        }

        return {**base_info, **api_info} 