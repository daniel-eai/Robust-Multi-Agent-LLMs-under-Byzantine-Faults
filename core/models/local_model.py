
import logging
from typing import Dict, Any, Optional
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

class LocalModel(BaseModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.api_base_url = kwargs.get('api_base_url', 'http://localhost:11434')

        self.model_type = kwargs.get('service_type', kwargs.get('model_type', 'ollama'))
        self.stream = kwargs.get('stream', False)

        self._validate_local_config()

        logger.info(f"Local model initialized: {self.model_name}, service type: {self.model_type}, URL: {self.api_base_url}")

    def _validate_local_config(self):

        if not self.api_base_url:
            raise ValueError("Local model requires an API base URL")

        if self.model_type not in ['ollama', 'lm_studio', 'vllm', 'openai_compatible']:
            logger.warning(f"Unknown model type: {self.model_type}, using generic OpenAI-compatible mode")
            self.model_type = 'openai_compatible'

    async def _generate_impl(self, prompt: str, **kwargs) -> str:

        if self.model_type == 'ollama':
            return await self._generate_with_ollama(prompt, **kwargs)
        else:
            return await self._generate_openai_compatible(prompt, **kwargs)

    async def _generate_with_ollama(self, prompt: str, **kwargs) -> str:

        try:

            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": self.stream,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature),
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 50)
                }
            }

            url = f"{self.api_base_url.rstrip('/')}/api/generate"

            response_data = await self._network_client.post_json(
                url=url,
                data=request_data,
                timeout=kwargs.get('timeout', self.timeout)
            )

            if 'response' in response_data:
                response_text = response_data['response']

                if 'eval_count' in response_data:
                    self._stats['total_tokens'] += response_data.get('eval_count', 0)

                return response_text
            else:
                raise ValueError("Ollama response format is incorrect")

        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")

            await self._check_ollama_model()
            raise

    async def _generate_openai_compatible(self, prompt: str, **kwargs) -> str:

        try:

            request_data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get('temperature', self.temperature),
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "stream": False
            }

            headers = {"Content-Type": "application/json"}

            if self.model_type == 'lm_studio':
                url = f"{self.api_base_url.rstrip('/')}/v1/chat/completions"
            else:
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
                    self._stats['total_tokens'] += response_data['usage'].get('total_tokens', 0)

                return content
            else:
                raise ValueError("API response format is incorrect")

        except Exception as e:
            logger.error(f"Local API call failed: {e}")
            raise

    async def _check_ollama_model(self):

        try:

            url = f"{self.api_base_url.rstrip('/')}/api/tags"
            response_data = await self._network_client.post_json(url, {})

            if 'models' in response_data:
                available_models = [model['name'] for model in response_data['models']]
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} is not in the available list: {available_models}")
                    logger.info(f"Please run: ollama pull {self.model_name}")
                else:
                    logger.info(f"Model {self.model_name} is available")

        except Exception as e:
            logger.error(f"Failed to check Ollama model: {e}")

    async def test_connection(self) -> bool:

        try:
            if self.model_type == 'ollama':

                url = f"{self.api_base_url.rstrip('/')}/api/version"
                await self._network_client.post_json(url, {})
                logger.info("Ollama service connection normal")

                await self._check_ollama_model()

            else:

                test_response = await self.generate("Test", max_tokens=10)
                if not test_response:
                    return False
                logger.info("Local model service connection normal")

            return True

        except Exception as e:
            logger.error(f"Local model connection test failed: {e}")
            return False

    async def list_available_models(self) -> list:

        try:
            if self.model_type == 'ollama':
                url = f"{self.api_base_url.rstrip('/')}/api/tags"
                response_data = await self._network_client.post_json(url, {})

                if 'models' in response_data:
                    return [
                        {
                            'name': model['name'],
                            'size': model.get('size', 0),
                            'modified_at': model.get('modified_at', ''),
                            'digest': model.get('digest', '')
                        }
                        for model in response_data['models']
                    ]

            else:

                logger.warning(f"Model type {self.model_type} does not support listing")
                return []

        except Exception as e:
            logger.error(f"Failed to get model list: {e}")
            return []

    async def pull_model(self, model_name: Optional[str] = None) -> bool:

        if self.model_type != 'ollama':
            logger.warning("Only Ollama supports model pulling")
            return False

        target_model = model_name or self.model_name

        try:
            url = f"{self.api_base_url.rstrip('/')}/api/pull"
            request_data = {"name": target_model}

            logger.info(f"Starting to pull model: {target_model}")
            await self._network_client.post_json(url, request_data, timeout=300)

            logger.info(f"Model {target_model} pulled successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:

        base_info = super().get_model_info()

        local_info = {
            'model_type': 'Local',
            'service_type': self.model_type,
            'api_base_url': self.api_base_url,
            'stream_enabled': self.stream,
            'cost_estimate': {
                'total_tokens': self._stats['total_tokens'],
                'estimated_cost': 0.0,          
                'currency': 'Free'
            }
        }

        return {**base_info, **local_info}

    def get_usage_instructions(self) -> Dict[str, str]:

        instructions = {
            'ollama': '''
Ollama Usage Instructions:
1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
2. Start service: ollama serve
3. Pull model: ollama pull {model_name}
4. Verify model: ollama list
            ''',
            'lm_studio': '''
LM Studio Usage Instructions:
1. Download and install LM Studio
2. Load a model in LM Studio
3. Start local server (usually on port 1234)
4. Ensure API compatibility mode is enabled
            ''',
            'vllm': '''
vLLM Usage Instructions:
1. Install vLLM: pip install vllm
2. Start API server: python -m vllm.entrypoints.openai.api_server --model {model_name}
3. Service typically runs on port 8000
            '''
        }

        return {
            'service_type': self.model_type,
            'instructions': instructions.get(self.model_type, 'Please refer to the documentation for the corresponding service'),
            'api_url': self.api_base_url,
            'current_model': self.model_name
        } 