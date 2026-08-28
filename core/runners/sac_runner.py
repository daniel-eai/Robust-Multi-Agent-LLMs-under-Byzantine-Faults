#!/usr/bin/env python3
"""
Self-Anchored Consensus (SAC) runner.

Algorithm 1 in the paper:
  For each round t = 0 .. T-1:
    1. Every agent i broadcasts r_i^(t) to neighbours.
    2. Agent i scores every neighbour and itself:
         s_{i->j} = phi_i(x, r_j^(t))   [receiver-side]
         s_{i->i} = phi_i(x, r_i^(t))
    3. L_i = { j in N_i | s_{i->j} < s_{i->i} }
    4. R_i = N_i \\ bottomF(L_i)          [discard F lowest from L]
    5. r_i^(t+1) = refine(x, r_i^(t), {(r_j, s_{i->j}) | j in R_i})
"""

import asyncio
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .base_runner import BaseMethodRunner, REPO_ROOT
from ..interfaces import AgentResponse, ConsensusResult, IVisualizer, MethodType, QuestionData
from ..visualization.unified_visualizer_adapter import UnifiedVisualizerAdapter
from config.sac_config import SACConfig

logger = logging.getLogger(__name__)


def _bottom_f(candidates: List[str], scores: Dict[str, float], F: int) -> List[str]:
    if not candidates or F <= 0:
        return []
    return sorted(candidates, key=lambda x: scores.get(x, 0.0))[:F]


class SACRunner(BaseMethodRunner):

    def __init__(self, config: SACConfig):
        super().__init__(config)
        self.F = config.get_adversary_bound()
        logger.info(f"SAC runner initialized (F={self.F})")

    def _get_method_type(self) -> MethodType:
        return MethodType.SAC

    def _get_data_path(self) -> str:
        if getattr(self.config, 'data_path', None):
            return str(self.config.data_path)
        dataset_type = str(getattr(self.config, 'dataset_type', 'math500')).lower()
        if dataset_type == 'safe':
            return str(REPO_ROOT / "data/byzantine/safe/safe_dataset.json")
        elif dataset_type == 'commonsense':
            return str(REPO_ROOT / "data/byzantine/commonsense/commonsense_dataset.json")
        elif dataset_type == 'math500':
            return str(REPO_ROOT / "data/byzantine/math500/math500_500.json")
        else:
            return str(REPO_ROOT / "data/byzantine/gsm8k/gsm8k_dataset.json")

    def _get_data_kwargs(self) -> Dict[str, Any]:
        return {'dataset_type': str(getattr(self.config, 'dataset_type', 'math500')).lower()}

    async def _create_visualizer(self) -> Optional[IVisualizer]:
        if not getattr(self.config, 'visualize', True):
            return None
        try:
            return UnifiedVisualizerAdapter(
                getattr(self.config, 'output_dir', 'results'),
                str(getattr(self.config, 'dataset_type', 'math500')),
                'llm',
            )
        except Exception as e:
            logger.warning(f"Failed to create visualizer: {e}")
            return None

    async def _run_experiment_core(self):
        logger.info(
            f"Starting SAC on {len(self.questions)} questions "
            f"(F={self.F}, T={self.config.rounds})"
        )

        all_consensus_results = []
        agent_ids = list(self.agents.keys())

        for q_idx, question in enumerate(self.questions):
            logger.info(f"Question {q_idx+1}/{len(self.questions)}: {question.question_id}")

            init_tasks = [self.agents[aid].solve_problem(question) for aid in agent_ids]
            init_responses = await asyncio.gather(*init_tasks, return_exceptions=True)

            current_answers: Dict[str, str] = {}
            initial_confidences: Dict[str, float] = {}
            for aid, resp in zip(agent_ids, init_responses):
                if isinstance(resp, Exception):
                    logger.warning(f"Agent {aid} initial response failed: {resp}")
                    current_answers[aid] = ""
                    initial_confidences[aid] = 0.0
                else:
                    current_answers[aid] = str(resp.answer)
                    initial_confidences[aid] = float(getattr(resp, 'confidence', 0.0) or 0.0)
                    self.agents[aid].own_answer = current_answers[aid]

            initial_answers = dict(current_answers)
            round_data = [dict(current_answers)]
            round_traces: List[Dict[str, Any]] = []

            for t in range(self.config.rounds):
                logger.debug(f"  SAC round {t+1}/{self.config.rounds}")
                current_answers, traces = await self._sac_round(
                    question, current_answers, agent_ids
                )
                round_data.append(dict(current_answers))
                round_traces.append(traces)

            final_responses: List[AgentResponse] = []
            for aid in agent_ids:
                final_responses.append(AgentResponse(
                    agent_id=aid,
                    question_id=question.question_id,
                    answer=current_answers.get(aid, ''),
                    confidence=initial_confidences.get(aid, 0.0),
                    reasoning=None,
                    metadata={
                        'initial_answer': initial_answers.get(aid, ''),
                        'final_answer': current_answers.get(aid, ''),
                        'role': getattr(self.agents[aid], 'role', ''),
                        'is_malicious': bool(getattr(self.agents[aid], 'is_malicious', False)),
                        'sac_rounds': self.config.rounds,
                    },
                ))

            honest_ids = [aid for aid in agent_ids if not self.agents[aid].is_malicious]
            honest_answers = {aid: current_answers[aid] for aid in honest_ids if aid in current_answers}
            answer_counts = Counter(a for a in honest_answers.values() if a)
            consensus_answer = answer_counts.most_common(1)[0][0] if answer_counts else ""
            consensus_conf = (
                answer_counts[consensus_answer] / len(honest_answers) if honest_answers else 0.0
            )
            is_correct = self.consensus_engine._check_answer_correctness(
                consensus_answer, question.correct_answer
            )

            all_consensus_results.append(ConsensusResult(
                question_id=question.question_id,
                consensus_answer=consensus_answer,
                consensus_confidence=consensus_conf,
                participant_count=len(agent_ids),
                round_number=self.config.rounds,
                convergence_achieved=True,
                individual_responses=final_responses,
                metadata={
                    'consensus_method': 'sac_honest_majority',
                    'answer_distribution': dict(answer_counts),
                    'initial_answers': initial_answers,
                    'initial_confidences': initial_confidences,
                    'round_data': round_data,
                    'round_traces': round_traces,
                    'adversary_bound': self.F,
                },
                is_correct=is_correct,
            ))

        logger.info(f"SAC complete: {len(all_consensus_results)} questions processed")
        return all_consensus_results

    async def _sac_round(
        self,
        question: QuestionData,
        current_answers: Dict[str, str],
        agent_ids: List[str],
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        tasks = [
            self._sac_update_agent(question, aid, current_answers)
            for aid in agent_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_answers: Dict[str, str] = {}
        traces: Dict[str, Any] = {}
        for aid, result in zip(agent_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Agent {aid} SAC update failed: {result}")
                new_answers[aid] = current_answers[aid]
                traces[aid] = {'error': str(result)}
            else:
                new_answers[aid], traces[aid] = result
        return new_answers, traces

    async def _sac_update_agent(
        self,
        question: QuestionData,
        agent_id: str,
        current_answers: Dict[str, str],
    ) -> Tuple[str, Dict[str, Any]]:
        agent = self.agents[agent_id]
        own_answer = current_answers[agent_id]

        if agent.is_malicious and getattr(agent, 'byzantine_mode', 'model') == 'fixed':
            refined = await agent.sac_refine(question, own_answer, [])
            return refined, {'self_score': None, 'retained': [], 'filtered': []}

        neighbors = self.topology.get_neighbors(agent_id)

        self_score = await agent.sac_evaluate(question, own_answer)

        score_tasks = [agent.sac_evaluate(question, current_answers[nid]) for nid in neighbors]
        scores = await asyncio.gather(*score_tasks, return_exceptions=True)
        neighbor_scores = {
            nid: (sc if not isinstance(sc, Exception) else 0.0)
            for nid, sc in zip(neighbors, scores)
        }

        L = [nid for nid in neighbors if neighbor_scores[nid] < self_score]
        to_remove = set(_bottom_f(L, neighbor_scores, self.F))
        retained_ids = [nid for nid in neighbors if nid not in to_remove]
        retained = [(current_answers[nid], neighbor_scores[nid]) for nid in retained_ids]

        refined = await agent.sac_refine(question, own_answer, retained)

        logger.debug(
            f"  Agent {agent_id}: self_score={self_score:.2f}, |L|={len(L)}, "
            f"|retained|={len(retained)}, {own_answer} -> {refined}"
        )
        return refined, {
            'self_score': self_score,
            'neighbor_scores': neighbor_scores,
            'retained': retained_ids,
            'filtered': sorted(to_remove),
        }


async def run_sac_experiment(config: SACConfig):
    runner = SACRunner(config)
    try:
        return await runner.run_experiment()
    finally:
        runner.cleanup_experiment()
