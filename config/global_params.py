from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

@dataclass
class LLMGenerationParams:

    max_new_tokens: int
    do_sample: bool
    temperature: float
    repetition_penalty: float
    num_return_sequences: int
    use_cache: bool

    top_p: Optional[float] = None
    top_k: Optional[int] = None

    seed: int = 1234

    dataset_type: str = ""
    method: str = ""                                    
    source: str = ""                
    description: str = ""

    def to_generate_kwargs(self, tokenizer) -> Dict[str, Any]:

        kwargs = {
            'max_new_tokens': self.max_new_tokens,
            'use_cache': self.use_cache,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }

        if self.do_sample:

            kwargs['do_sample'] = True
            kwargs['temperature'] = self.temperature
            kwargs['repetition_penalty'] = self.repetition_penalty
            kwargs['num_return_sequences'] = self.num_return_sequences

            if self.top_p is not None:
                kwargs['top_p'] = self.top_p
            if self.top_k is not None:
                kwargs['top_k'] = self.top_k
        else:

            kwargs['do_sample'] = False
            kwargs['repetition_penalty'] = self.repetition_penalty
            kwargs['num_return_sequences'] = self.num_return_sequences

        return kwargs

    def to_dict(self) -> Dict[str, Any]:

        return {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repetition_penalty': self.repetition_penalty,
            'num_return_sequences': self.num_return_sequences,
            'use_cache': self.use_cache,
            'seed': self.seed,
            'dataset_type': self.dataset_type,
            'method': self.method,
            'source': self.source,
            'description': self.description,
        }

@dataclass
class LCDProbeConfig:

    model_path: str

    target_layer: int         
    pooling_method: str                          
    pca_dim: Optional[int] = None             

    probe_type: str = "logistic"                    

    dataset_type: str = ""
    model_name: str = ""                  
    source: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:

        return {
            'model_path': self.model_path,
            'target_layer': self.target_layer,
            'pooling_method': self.pooling_method,
            'pca_dim': self.pca_dim,
            'probe_type': self.probe_type,
            'dataset_type': self.dataset_type,
            'model_name': self.model_name,
            'source': self.source,
            'description': self.description,
        }

class GlobalParams:

    LLAMA3_MODEL_PATH = "models/LLama-3-8B-Instruct"
    LLAMA31_MODEL_PATH = "models/LLama-3.1-8B-Instruct"

    DECODER_GSM8K_DATA = "data/byzantine/gsm8k/llama3.1_10.json"
    DECODER_SAFE_DATA = "data/byzantine/safe/all_llama31_win.json"
    DECODER_COMMONSENSE_DATA = "data/byzantine/commonsense/llama31_win_10.json"

    PILOT_GSM8K_DATA = "data/byzantine/gsm8k/gsm8k_final_dataset_20250723_154643.json"
    PILOT_SAFE_DATA = "data/byzantine/safe/safe_final_dataset_aligned_20250723_191623.json"

    DECODER_GSM8K_GENERATION = LLMGenerationParams(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="gsm8k",
        method="decoder_probe",
        source="decoder_gsm8k_training_config",
        description="GSM8K math reasoning (sampled generation, temperature=0.1)"
    )

    DECODER_SAFE_GENERATION = LLMGenerationParams(
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.2,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="safe",
        method="decoder_probe",
        source="decoder_safe_training_config",
        description="Safe safety (deterministic generation, fully reproducible)"
    )

    PILOT_GSM8K_GENERATION = LLMGenerationParams(
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.0,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="gsm8k",
        method="pilot",
        source="Materials Appendix C",
        description="Pilot experiment GSM8K (API call)"
    )

    PILOT_SAFE_GENERATION = LLMGenerationParams(
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.0,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="safe",
        method="pilot",
        source="Materials Appendix C",
        description="Pilot experiment Safe (API call)"
    )

    PROMPT_PROBE_GSM8K_GENERATION = PILOT_GSM8K_GENERATION
    PROMPT_PROBE_SAFE_GENERATION = PILOT_SAFE_GENERATION

    DECODER_COMMONSENSE_GENERATION = LLMGenerationParams(
        max_new_tokens=128,  
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="commonsense",
        method="decoder_probe",
        source="tools/commonsense_inference.py line 232-243",
        description="CommonsenseQA commonsense reasoning (sampled generation, fully consistent with data collection)"
    )

    LCD_GSM8K_LLAMA3 = LCDProbeConfig(
        model_path="lcd_models/gsm8k/3_pooled_layer16_pca256_logistic",
        target_layer=16,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="logistic",
        dataset_type="gsm8k",
        model_name="llama3",
        source="lcd_gsm8k_llama3_probe",
        description="GSM8K LLaMA3 probe (layer 16, PCA256)"
    )

    LCD_GSM8K_LLAMA31 = LCDProbeConfig(
        model_path="lcd_models/gsm8k/3.1_pooled_layer12_pca256_logistic",
        target_layer=12,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="logistic",
        dataset_type="gsm8k",
        model_name="llama31",
        source="lcd_gsm8k_llama31_probe",
        description="GSM8K LLaMA3.1 probe (layer 12, PCA256)"
    )

    LCD_SAFE_LLAMA3 = LCDProbeConfig(
        model_path="lcd_models/safe/3_pooled_layer17_pca256_logistic",
        target_layer=17,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="logistic",
        dataset_type="safe",
        model_name="llama3",
        source="lcd_safe_llama3_probe",
        description="Safe LLaMA3 probe (layer 17, PCA256, pooled)"
    )

    LCD_SAFE_LLAMA31 = LCDProbeConfig(
        model_path="lcd_models/safe/3.1_pooled_layer12_pca256_logistic",
        target_layer=12,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="logistic",
        dataset_type="safe",
        model_name="llama31",
        source="lcd_safe_llama31_probe",
        description="Safe LLaMA3.1 probe (layer 12, PCA256, pooled)"
    )

    LCD_COMMONSENSE_LLAMA3 = LCDProbeConfig(
        model_path="lcd_models/commonsense/3_pooled_layer14_pca256_mlp",
        target_layer=14,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="mlp",
        dataset_type="commonsense",
        model_name="llama3",
        source="lcd_commonsense_llama3_probe",
        description="CommonsenseQA LLaMA3 probe (layer 14, PCA256, MLP, pooled)"
    )

    LCD_COMMONSENSE_LLAMA31 = LCDProbeConfig(
        model_path="lcd_models/commonsense/3.1_query_layer14_pca256_mlp",
        target_layer=14,
        pooling_method="query",
        pca_dim=256,
        probe_type="mlp",
        dataset_type="commonsense",
        model_name="llama31",
        source="lcd_commonsense_llama31_probe",
        description="CommonsenseQA LLaMA3.1 probe (layer 14, PCA256, MLP, query)"
    )

    PROMPT_PROBE_CONFIDENCE_PROMPT = (
        "On a scale of 0 to 100, how confident are you in your answer? "
        "Please provide only a number without any explanation."
    )

    GLOBAL_SEED = 1234

    API_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_API_TIMEOUT = 60     
    DEFAULT_API_RETRY = 3

    DEFAULT_CONSENSUS_THRESHOLD = 0.5
    DEFAULT_MAX_ITERATIONS = 3

    CUDA_DETERMINISTIC = True                                      
    CUDA_BENCHMARK = False                                     

    @classmethod
    def get_llm_generation_params(cls, method: str, dataset_type: str) -> LLMGenerationParams:

        method = method.lower().replace('_', '').replace('-', '')
        dataset = dataset_type.lower().strip()

        key = f"{method}_{dataset}_generation"
        key = key.replace('promptprobe', 'prompt_probe').replace('decoderprobe', 'decoder')
        key = key.upper()

        if not hasattr(cls, key):
            raise ValueError(
                f"Configuration not found: method={method}, dataset={dataset_type}\n"
                f"Attempted key: {key}"
            )

        return getattr(cls, key)

    @classmethod
    def get_lcd_config(cls, dataset_type: str, model_name: str) -> LCDProbeConfig:

        dataset = dataset_type.lower().strip()
        model = model_name.lower().strip()

        key = f"LCD_{dataset.upper()}_{model.upper()}"

        if not hasattr(cls, key):
            raise ValueError(
                f"LCD configuration not found: dataset={dataset_type}, model={model_name}\n"
                f"Attempted key: {key}"
            )

        return getattr(cls, key)

    @classmethod
    def get_data_path(cls, method: str, dataset_type: str) -> str:

        method = method.lower().replace('_', '').replace('-', '')
        dataset = dataset_type.lower().strip()

        if method in ['pilot', 'promptprobe']:
            prefix = 'PILOT'
        else:                 
            prefix = 'DECODER'

        key = f"{prefix}_{dataset.upper()}_DATA"

        if not hasattr(cls, key):
            raise ValueError(
                f"Data path not found: method={method}, dataset={dataset_type}\n"
                f"Attempted key: {key}"
            )

        return getattr(cls, key)

    @classmethod
    def print_all_configs(cls):

        print("=" * 120)
        print(" Global Parameter Configuration Center ".center(120, "="))
        print("=" * 120)

        print("\n[Model Paths]")
        print(f"  LLAMA3:  {cls.LLAMA3_MODEL_PATH}")
        print(f"  LLAMA31: {cls.LLAMA31_MODEL_PATH}")

        print("\n[Dataset Paths]")
        print(f"  Decoder GSM8K:        {cls.DECODER_GSM8K_DATA}")
        print(f"  Decoder Safe:         {cls.DECODER_SAFE_DATA}")
        print(f"  Decoder CommonsenseQA: {cls.DECODER_COMMONSENSE_DATA}")
        print(f"  Pilot GSM8K:          {cls.PILOT_GSM8K_DATA}")
        print(f"  Pilot Safe:           {cls.PILOT_SAFE_DATA}")

        print("\n[LLM Generation Parameters]")
        for attr_name in dir(cls):
            if attr_name.endswith('_GENERATION'):
                params = getattr(cls, attr_name)
                if isinstance(params, LLMGenerationParams):
                    mode = "Sampling" if params.do_sample else "Deterministic"
                    print(f"\n  {attr_name}:")
                    print(f"    Method: {params.method}, Dataset: {params.dataset_type}")
                    print(f"    Mode: {mode}, temp={params.temperature}, max_tokens={params.max_new_tokens}")
                    print(f"    Source: {params.source}")

        print("\n[LCD Probe Configurations]")
        for attr_name in dir(cls):
            if attr_name.startswith('LCD_'):
                config = getattr(cls, attr_name)
                if isinstance(config, LCDProbeConfig):
                    print(f"\n  {attr_name}:")
                    print(f"    Dataset: {config.dataset_type}, Model: {config.model_name}")
                    print(f"    Layer: {config.target_layer}, Pooling: {config.pooling_method}, PCA: {config.pca_dim}")
                    print(f"    Path: {config.model_path}")

        print("\n[Other Fixed Parameters]")
        print(f"  Global seed: {cls.GLOBAL_SEED}")
        print(f"  CUDA deterministic: {cls.CUDA_DETERMINISTIC}")
        print(f"  CUDA Benchmark: {cls.CUDA_BENCHMARK}")
        print(f"  API Base URL: {cls.API_BASE_URL}")
        print(f"  API timeout: {cls.DEFAULT_API_TIMEOUT}s")

        print("\n" + "=" * 120)

    @classmethod
    def validate_all_paths(cls):

        print("\n" + "=" * 120)
        print(" Path Validation ".center(120, "="))
        print("=" * 120)

        issues = []

        for name in ['LLAMA3_MODEL_PATH', 'LLAMA31_MODEL_PATH']:
            path = Path(getattr(cls, name))
            status = "OK" if path.exists() else "MISSING"
            print(f"[{status}] {name}: {path}")
            if not path.exists():
                issues.append(f"Model path does not exist: {path}")

        for attr_name in dir(cls):
            if attr_name.endswith('_DATA'):
                path = Path(getattr(cls, attr_name))
                status = "OK" if path.exists() else "MISSING"
                print(f"[{status}] {attr_name}: {path}")
                if not path.exists():
                    issues.append(f"Dataset path does not exist: {path}")

        for attr_name in dir(cls):
            if attr_name.startswith('LCD_'):
                config = getattr(cls, attr_name)
                if isinstance(config, LCDProbeConfig):
                    path = Path(config.model_path)

                    exists = path.exists() or path.parent.exists()
                    status = "OK" if exists else "MISSING"
                    print(f"[{status}] {attr_name}: {path}")
                    if not exists:
                        issues.append(f"LCD path does not exist: {path}")

        print("\n" + "=" * 120)

        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("All path validations passed!")

        print("=" * 120)

def get_generation_params(dataset_type: str) -> LLMGenerationParams:

    return GlobalParams.get_llm_generation_params('decoder_probe', dataset_type)

if __name__ == "__main__":
    import sys

    GlobalParams.print_all_configs()

    GlobalParams.validate_all_paths()

    print("\n" + "=" * 120)
    print(" Quick Access Tests ".center(120, "="))
    print("=" * 120)

    try:
        print("\n1. Get Decoder GSM8K generation parameters:")
        params = GlobalParams.get_llm_generation_params('decoder_probe', 'gsm8k')
        print(f"   do_sample={params.do_sample}, temperature={params.temperature}")

        print("\n2. Get Safe LLaMA3 LCD configuration:")
        lcd = GlobalParams.get_lcd_config('safe', 'llama3')
        print(f"   target_layer={lcd.target_layer}, pca_dim={lcd.pca_dim}")

        print("\n3. Get Pilot GSM8K data path:")
        data_path = GlobalParams.get_data_path('pilot', 'gsm8k')
        print(f"   {data_path}")

        print("\nAll tests passed")

    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)

    print("=" * 120)

