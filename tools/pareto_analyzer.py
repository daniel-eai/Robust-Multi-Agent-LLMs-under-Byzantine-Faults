#!/usr/bin/env python3

import os
import sys
import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ParetoAnalyzer:

    def __init__(self, models_dir: str, dataset: Optional[str] = None, 
                 probe_type: Optional[str] = None, model: Optional[str] = None,
                 method: Optional[str] = None):

        self.models_dir = Path(models_dir)
        self.dataset_filter = dataset if dataset != 'all' else None
        self.probe_type_filter = probe_type
        self.model_filter = self._normalize_model(model) if model else None
        self.method_filter = method
        self.pareto_objectives = [
            'test_accuracy', 'test_auc', 'f1_score', 'precision', 'recall', 'specificity'
        ]

        logger.info("Pareto optimal analyzer initialized")
        logger.info(f"Model directory: {self.models_dir}")
        logger.info(f"Dataset filter: {self.dataset_filter or 'all datasets'}")
        logger.info(f"Probe type filter: {self.probe_type_filter or 'all types'}")
        logger.info(f"Model version filter: {self.model_filter or 'all versions'}")
        logger.info(f"Classifier method filter: {self.method_filter or 'all methods'}")
        logger.info(f"Pareto objectives: {self.pareto_objectives}")

    def _normalize_model(self, model: str) -> str:

        model_normalized = model.lower().replace('llama', '').replace('_', '')
        if model_normalized == '3':
            return 'llama3'
        elif model_normalized == '3.1':
            return 'llama31'
        return model_normalized

    def parse_model_name(self, model_name: str) -> Dict[str, Any]:

        config = {}

        if model_name.startswith('3.1_'):
            config['model_version'] = 'llama31'
        elif model_name.startswith('3_'):
            config['model_version'] = 'llama3'
        else:
            config['model_version'] = 'unknown'

        for ptype in ['pooled', 'query', 'answer', 'targeted']:
            if ptype in model_name:
                config['probe_type'] = ptype
                break
        else:
            config['probe_type'] = 'unknown'

        layer_match = re.search(r'layer(\d+)', model_name)
        if layer_match:
            config['layer'] = int(layer_match.group(1))

        pca_match = re.search(r'pca(\d+)', model_name)
        if pca_match:
            config['pca_dim'] = int(pca_match.group(1))

        if 'logistic' in model_name:
            config['method'] = 'logistic'
        elif 'mlp' in model_name:
            config['method'] = 'mlp'
        else:
            config['method'] = 'unknown'

        return config

    def load_metrics_from_file(self, metrics_path: Path) -> Optional[Dict[str, Any]]:

        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            required_metrics = ['test_accuracy', 'test_auc']
            if not all(key in metrics for key in required_metrics):
                logger.warning(f"Incomplete metrics: {metrics_path}")
                return None

            if 'precision' not in metrics:
                metrics['precision'] = 0.0
            if 'recall' not in metrics:
                metrics['recall'] = 0.0
            if 'f1_score' not in metrics:
                metrics['f1_score'] = 0.0
            if 'specificity' not in metrics:
                metrics['specificity'] = 0.0

            return metrics
        except Exception as e:
            logger.error(f"Failed to load metrics {metrics_path}: {e}")
            return None

    def scan_models(self) -> List[Dict[str, Any]]:

        all_results = []

        for metrics_file in self.models_dir.rglob('metrics.json'):

            model_dir = metrics_file.parent
            dataset_name = model_dir.parent.name if model_dir.parent != self.models_dir else 'unknown'
            model_name = model_dir.name

            if self.dataset_filter and self.dataset_filter not in dataset_name.lower():
                continue

            config = self.parse_model_name(model_name)

            if self.probe_type_filter and config.get('probe_type') != self.probe_type_filter:
                continue

            if self.model_filter and config.get('model_version') != self.model_filter:
                continue

            if self.method_filter and config.get('method') != self.method_filter:
                continue

            metrics = self.load_metrics_from_file(metrics_file)
            if metrics is None:
                continue

            result = {
                'model_path': str(model_dir.relative_to(self.models_dir)),
                'dataset': dataset_name,
                'model_name': model_name,
                'config': config,
                'metrics': metrics,
                'pareto_metrics': {
                    'test_accuracy': metrics.get('test_accuracy', 0),
                    'test_auc': metrics.get('test_auc', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'specificity': metrics.get('specificity', 0)
                }
            }

            all_results.append(result)

        logger.info(f"Scanned {len(all_results)} models")
        return all_results

    def _is_pareto_dominated(self, solution_a: Dict[str, float], 
                            solution_b: Dict[str, float]) -> bool:

        all_not_worse = True
        at_least_one_better = False

        for objective in self.pareto_objectives:
            a_value = solution_a.get(objective, 0)
            b_value = solution_b.get(objective, 0)

            if b_value < a_value:             
                all_not_worse = False
                break
            if b_value > a_value:             
                at_least_one_better = True

        return all_not_worse and at_least_one_better

    def _find_pareto_optimal(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        pareto_optimal = []

        for i, candidate in enumerate(all_results):
            is_dominated = False
            candidate_metrics = candidate['pareto_metrics']

            for j, other in enumerate(all_results):
                if i == j:
                    continue

                other_metrics = other['pareto_metrics']
                if self._is_pareto_dominated(candidate_metrics, other_metrics):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_optimal.append(candidate)

        return pareto_optimal

    def generate_comprehensive_ranking(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:

        if not all_results:
            logger.warning("No valid model results")
            return {}

        logger.info(f"Computing Pareto optimal set, candidates: {len(all_results)}")
        pareto_optimal = self._find_pareto_optimal(all_results)
        logger.info(f"Pareto optimal set size: {len(pareto_optimal)}")

        accuracy_ranking = sorted(
            all_results, 
            key=lambda x: x['pareto_metrics']['test_accuracy'], 
            reverse=True
        )[:10]

        pareto_paths = {result['model_path'] for result in pareto_optimal}
        for i, item in enumerate(accuracy_ranking):
            item['rank'] = i + 1
            item['is_pareto_optimal'] = item['model_path'] in pareto_paths

        auc_ranking = sorted(
            all_results,
            key=lambda x: x['pareto_metrics']['test_auc'],
            reverse=True
        )[:10]

        ranking_result = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'total_models_evaluated': len(all_results),
                'pareto_objectives': self.pareto_objectives,
                'dataset_filter': self.dataset_filter,
                'probe_type_filter': self.probe_type_filter
            },
            'pareto_optimal_solutions': pareto_optimal,
            'top_10_accuracy_ranking': accuracy_ranking,
            'top_10_auc_ranking': auc_ranking
        }

        return ranking_result

    def print_ranking_summary(self, ranking_result: Dict[str, Any], top_n: int = 10, pareto_display: int = 15):

        if not ranking_result:
            print("No ranking results to display")
            return

        print("\n" + "="*120)
        print("Pareto Optimal Analysis Report")
        print("="*120)

        eval_info = ranking_result['evaluation_info']
        print(f"\nEvaluation Info:")
        print(f"  Total models evaluated: {eval_info['total_models_evaluated']}")
        print(f"  Evaluation time: {eval_info['timestamp']}")
        print(f"  Dataset filter: {eval_info['dataset_filter'] or 'all datasets'}")
        print(f"  Probe type filter: {eval_info['probe_type_filter'] or 'all types'}")
        print(f"  Pareto objectives: {eval_info['pareto_objectives']}")

        pareto_solutions = ranking_result['pareto_optimal_solutions']
        display_pareto = pareto_solutions[:pareto_display]              

        print(f"\nPareto Optimal Solutions ({len(pareto_solutions)} total, showing first {len(display_pareto)}):")
        print("-"*120)
        print(f"{'Model Path':<60} {'ACC':<8} {'AUC':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Spec':<8}")
        print("-"*120)

        for solution in display_pareto:
            pm = solution['pareto_metrics']
            print(f"{solution['model_path']:<60} "
                  f"{pm['test_accuracy']:<8.4f} "
                  f"{pm['test_auc']:<8.4f} "
                  f"{pm['f1_score']:<8.4f} "
                  f"{pm['precision']:<8.4f} "
                  f"{pm['recall']:<8.4f} "
                  f"{pm['specificity']:<8.4f}")

        acc_ranking = ranking_result['top_10_accuracy_ranking'][:top_n]
        print(f"\nTop-{len(acc_ranking)} Accuracy Ranking:")
        print("-"*120)
        print(f"{'Rank':<6} {'Model Path':<60} {'ACC':<8} {'AUC':<8} {'Pareto':<10}")
        print("-"*120)

        for item in acc_ranking:
            pm = item['pareto_metrics']
            pareto_mark = "PARETO" if item['is_pareto_optimal'] else ""
            print(f"{item['rank']:<6} "
                  f"{item['model_path']:<60} "
                  f"{pm['test_accuracy']:<8.4f} "
                  f"{pm['test_auc']:<8.4f} "
                  f"{pareto_mark:<10}")

        auc_ranking = ranking_result['top_10_auc_ranking'][:top_n]
        print(f"\nTop-{len(auc_ranking)} AUC Ranking:")
        print("-"*120)
        print(f"{'Rank':<6} {'Model Path':<60} {'AUC':<8} {'ACC':<8}")
        print("-"*120)

        for i, item in enumerate(auc_ranking, 1):
            pm = item['pareto_metrics']
            print(f"{i:<6} "
                  f"{item['model_path']:<60} "
                  f"{pm['test_auc']:<8.4f} "
                  f"{pm['test_accuracy']:<8.4f}")

        print("="*120)

    def save_ranking_report(self, ranking_result: Dict[str, Any], output_path: str):

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(ranking_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Ranking report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def run_analysis(self, output_path: Optional[str] = None, top_n: int = 10, pareto_display: int = 15) -> Dict[str, Any]:

        all_results = self.scan_models()

        if not all_results:
            logger.warning("No model results found")
            return {}

        ranking_result = self.generate_comprehensive_ranking(all_results)

        self.print_ranking_summary(ranking_result, top_n=top_n, pareto_display=pareto_display)

        if output_path:
            self.save_ranking_report(ranking_result, output_path)

        return ranking_result

def main():
    parser = argparse.ArgumentParser(
        description='Pareto optimal analyzer - multi-dimensional filtering support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all datasets
  python pareto_analyzer.py --base-dir ../lcd_outputs

  # Analyze only GSM8K
  python pareto_analyzer.py -b ../lcd_outputs -d gsm8k

  # Analyze pooled probe results for GSM8K
  python pareto_analyzer.py -b ../lcd_outputs -d gsm8k -p pooled

  # Analyze LLaMA3.1 on GSM8K
  python pareto_analyzer.py -b ../lcd_outputs -d gsm8k --model 3.1

  # Analyze pooled probe + logistic classifier
  python pareto_analyzer.py -b ../lcd_outputs -p pooled -m logistic

  # Save analysis report
  python pareto_analyzer.py -b ../lcd_outputs -d gsm8k --output gsm8k_pareto.json

  # Show Top 20
  python pareto_analyzer.py -b ../lcd_outputs -t 20

  # Show top 20 Pareto optimal solutions
  python pareto_analyzer.py -b ../lcd_outputs --pareto 20
        """)

    parser.add_argument('--base-dir', '-b', type=str, 
                       default='lcd_outputs',
                       help='LCD output directory path (default: lcd_outputs, relative to cwd or repo root)')
    parser.add_argument('--dataset', '-d', type=str, 
                       choices=['gsm8k', 'safe', 'commonsense', 'all'],
                       default='all',
                       help='Select dataset (default: all)')
    parser.add_argument('--model', type=str,
                       choices=['3', '3.1', 'llama3', 'llama3.1'],
                       help='Filter by model version (LLaMA3 or LLaMA3.1)')
    parser.add_argument('--probe', '-p', type=str, 
                       choices=['answer', 'pooled', 'query'],
                       help='Filter by probe type (hidden state type)')
    parser.add_argument('--method', '-m', type=str,
                       choices=['logistic', 'mlp'],
                       help='Filter by classifier method')
    parser.add_argument('--top', '-t', type=int, default=10,
                       help='Show top N models (default: 10)')
    parser.add_argument('--pareto', type=int, default=15,
                       help='Show top N Pareto optimal solutions (default: 15)')
    parser.add_argument('--output', type=str,
                       help='Output JSON report path')

    args = parser.parse_args()

    analyzer = ParetoAnalyzer(
        models_dir=args.base_dir,
        dataset=args.dataset,
        probe_type=args.probe,
        model=args.model,
        method=args.method
    )

    analyzer.run_analysis(output_path=args.output, top_n=args.top, pareto_display=args.pareto)

if __name__ == "__main__":
    main()
