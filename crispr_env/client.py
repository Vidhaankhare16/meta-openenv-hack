"""
CRISPR Guide RNA Design Environment — Client
=============================================
WebSocket / HTTP client for interacting with the deployed environment.
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import CRISPRAction, CRISPRObservation, CRISPRState


class CRISPREnv(EnvClient[CRISPRAction, CRISPRObservation, CRISPRState]):
    """
    Client for the CRISPR Guide RNA Design environment.

    Usage
    -----
    with CRISPREnv(base_url="https://your-space.hf.space").sync() as env:
        obs = env.reset(task="easy")
        result = env.step(CRISPRAction(
            action_type="scan_sequence",
            pam_positions=[6, 21, 30, 45]
        ))
        print(result.observation.message)
    """

    def _step_payload(self, action: CRISPRAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        obs_data = payload.get("observation", {})
        obs = CRISPRObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task=obs_data.get("task", "easy"),
            sequence=obs_data.get("sequence", ""),
            message=obs_data.get("message", ""),
            mutation_position=obs_data.get("mutation_position"),
            mutation_info=obs_data.get("mutation_info"),
            returned_guide=obs_data.get("returned_guide"),
            efficiency_score=obs_data.get("efficiency_score"),
            candidate_guides=obs_data.get("candidate_guides"),
            genome_excerpt=obs_data.get("genome_excerpt"),
            offtarget_result=obs_data.get("offtarget_result"),
            checked_so_far=obs_data.get("checked_so_far"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CRISPRState:
        return CRISPRState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", "easy"),
            sequence=payload.get("sequence", ""),
            true_pam_sites=payload.get("true_pam_sites", []),
            mutation_position=payload.get("mutation_position", 0),
            best_pam_position=payload.get("best_pam_position", 0),
            designed_guide=payload.get("designed_guide"),
            ontarget_score=payload.get("ontarget_score"),
            candidate_guides=payload.get("candidate_guides", []),
            genome_excerpt=payload.get("genome_excerpt", ""),
            true_offtarget_counts=payload.get("true_offtarget_counts", []),
            true_best_guide_index=payload.get("true_best_guide_index", 0),
            checked_guides=payload.get("checked_guides", {}),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
