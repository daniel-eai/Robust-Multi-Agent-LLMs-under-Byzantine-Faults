import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")

def set_global_seed(seed: int):

    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def read_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], list):
                return data['data']
            if 'results' in data and isinstance(data['results'], list):
                return data['results']
            if 'questions' in data and isinstance(data['questions'], list):
                return data['questions']
            return [data]
        return data

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_samples(input_paths: List[str]) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for p in input_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if path.suffix.lower() == '.jsonl':
            rows = read_jsonl(path)
        elif path.suffix.lower() == '.json':
            rows = read_json(path)
        else:
            raise ValueError(f"Unsupported input format: {path.suffix}")
        all_rows.extend(rows)
    return all_rows

def format_inputs(
    row: Dict[str, Any],
    question_field: str,
    answer_field: Optional[str],
    system_prefix: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    q = row.get(question_field, "")
    a = None if answer_field is None else row.get(answer_field)
    if system_prefix:
        q = f"{system_prefix}{q}"
    return q, a

def ensure_tokenizer_and_model(model_path: str, device: Optional[str] = None):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device_str)
    model.eval()
    return tokenizer, model, device_str

@torch.no_grad()
def extract_hidden_states_for_pair(
    tokenizer,
    model,
    device: str,
    question: str,
    answer: Optional[str],
    max_new_tokens: int = 0,
    temperature: float = 0.0,
) -> Dict[str, np.ndarray]:

    inputs = tokenizer(question, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out_q = model(**inputs, output_hidden_states=True)

    query_hs = [h[:, -1, :].squeeze(0).detach().cpu().float().numpy() for h in out_q.hidden_states]
    query_hs = np.stack(query_hs, axis=0)

    if answer is None and max_new_tokens > 0:

        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 
            temperature=temperature                
        )
        full_ids = gen_out[0]
    else:
        if answer:
            full_text = question + answer
            full_inputs = tokenizer(full_text, return_tensors='pt')
        else:
            full_inputs = inputs
        full_ids = full_inputs['input_ids'][0]

    full_inputs = {
        'input_ids': full_ids.unsqueeze(0).to(device),
        'attention_mask': torch.ones_like(full_ids).unsqueeze(0).to(device)
    }
    out_full = model(**full_inputs, output_hidden_states=True)

    prompt_len = inputs['input_ids'].size(1)
    total_len = full_ids.size(0)
    answer_len = max(0, total_len - prompt_len)

    if answer_len > 0:

        answer_tokens_layers = [h[:, prompt_len:, :].squeeze(0).detach().cpu().float().numpy() for h in out_full.hidden_states]

        answer_tokens_layers = np.stack(answer_tokens_layers, axis=0)
        answer_pooled = answer_tokens_layers.mean(axis=1)

        answer_last_token = answer_tokens_layers[:, -1, :]
    else:

        hidden_size = out_full.hidden_states[-1].size(-1)
        num_layers = len(out_full.hidden_states)
        answer_pooled = np.zeros((num_layers, hidden_size), dtype=np.float32)
        answer_last_token = np.zeros((num_layers, hidden_size), dtype=np.float32)

    return {
        'query_hidden_states': query_hs,                        
        'answer_hidden_states': answer_last_token,              
        'answer_pooled_states': answer_pooled,                  
    }

def save_split_outputs(
    outputs: List[Dict[str, Any]],
    split_dir: Path,
    copy_data: Optional[List[Dict[str, Any]]] = None,
):
    split_dir.mkdir(parents=True, exist_ok=True)

    query_list = [o['query_hidden_states'] for o in outputs]
    answer_list = [o['answer_hidden_states'] for o in outputs]
    pooled_list = [o['answer_pooled_states'] for o in outputs]

    query_array = np.stack(query_list, axis=0)
    answer_array = np.stack(answer_list, axis=0)
    pooled_array = np.stack(pooled_list, axis=0)

    np.save(split_dir / f"{split_dir.name}_query_hidden_states.npy", query_array)
    np.save(split_dir / f"{split_dir.name}_answer_hidden_states.npy", answer_array)
    np.save(split_dir / f"{split_dir.name}_answer_pooled_states.npy", pooled_array)

    if copy_data is not None:
        with open(split_dir / f"{split_dir.name}_data.json", 'w', encoding='utf-8') as f:
            json.dump(copy_data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Saved {split_dir.name}: query={query_array.shape}, answer={answer_array.shape}, pooled={pooled_array.shape}"
    )

def main():
    parser = argparse.ArgumentParser(description="General offline hidden-state extraction tool")
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True, help="Local HF model path")
    parser.add_argument("--inputs", nargs='+', default=None, help="Input data files (JSON/JSONL), multiple allowed")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default=None, help="Input data directory (recursively collects *.json and *.jsonl)")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True, help="Output root directory, e.g. data/hidden_states/gsm8k/llama31")
    parser.add_argument("--question-field", default="question", help="Question field name")

    parser.add_argument("--system-prefix", default=None, help="System prompt prefix (optional)")
    parser.add_argument("--device", default=None, help="cuda/cpu, auto-detected by default")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="Number of tokens to generate on-the-fly (default 1)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default 0.0, deterministic)")
    parser.add_argument("--seed", type=int, default=1234, help="Global random seed (default 1234)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process (for quick small-scale validation)")
    parser.add_argument("--dry-run", action="store_true", help="Skip model loading; generate dummy hidden states to verify IO pipeline")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1, help="Log/periodic-save step size")
    parser.add_argument("--save-interval", dest="save_interval", type=int, default=0, help="Save every N samples (0 = disabled)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    set_global_seed(int(args.seed))

    out_root = Path(args.output_dir)
    split_dir = out_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading samples...")
    input_files: List[str] = []
    if args.inputs:
        input_files.extend(args.inputs)
    if args.data_dir:
        root = Path(args.data_dir)
        if not root.exists():
            raise SystemExit(f"Data directory not found: {root}")
        found = [str(p) for p in root.rglob("*.json")] + [str(p) for p in root.rglob("*.jsonl")]
        if not found:
            logger.warning(f"No JSON/JSONL files found under {root}")
        input_files.extend(found)
    if not input_files:
        raise SystemExit("No input provided: specify data via --inputs or --data-dir")
    rows = load_samples(input_files)
    logger.info(f"Samples: {len(rows)}")

    if args.dry_run:
        tokenizer = None
        model = None
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Dry-run mode: skipping model load, generating dummy hidden states.")
    else:
        logger.info("Loading model and tokenizer...")
        tokenizer, model, device = ensure_tokenizer_and_model(args.model_path, args.device)
        logger.info(f"Device: {device}")

    results: List[Dict[str, Any]] = []
    raw_copy: List[Dict[str, Any]] = []

    if args.max_samples is not None:
        rows = rows[:max(0, int(args.max_samples))]

    for idx, row in enumerate(rows):

        q, _ = format_inputs(row, args.question_field, None, args.system_prefix)
        a = None
        if args.dry_run:

            num_layers = 33
            hidden_size = 4096
            hs = {
                'query_hidden_states': np.zeros((num_layers, hidden_size), dtype=np.float32),
                'answer_hidden_states': np.zeros((num_layers, hidden_size), dtype=np.float32),
                'answer_pooled_states': np.zeros((num_layers, hidden_size), dtype=np.float32),
            }
        else:
            hs = extract_hidden_states_for_pair(
                tokenizer=tokenizer,
                model=model,
                device=device,
                question=q,
                answer=a,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        results.append(hs)
        raw_copy.append(row)
        if (idx + 1) % max(1, args.batch_size) == 0:
            logger.info(f"Processed {idx + 1} / {len(rows)}")
        if args.save_interval and (idx + 1) % max(1, args.save_interval) == 0:
            save_split_outputs(results, split_dir, copy_data=raw_copy)

    save_split_outputs(results, split_dir, copy_data=raw_copy)
    logger.info("Done")

if __name__ == "__main__":
    main()

