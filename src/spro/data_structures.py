"""Data structures for SPRO training."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import torch


@dataclass
class AttackPlan:
    """A complete attack plan (SEMA-style: all prompts generated at once)."""
    intent: str                          # Original harmful intent
    plan_text: str                       # Raw generation with <think> + numbered prompts
    thinking: str                        # Extracted <think> content
    prompts: List[str]                   # Extracted numbered prompts [p1, p2, ...]
    token_ids: torch.Tensor              # Tokenized plan for SPRO
    response_start_idx: int              # Where response tokens begin

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "plan_text": self.plan_text,
            "thinking": self.thinking,
            "prompts": self.prompts,
            "response_start_idx": self.response_start_idx,
        }


@dataclass
class ExecutedEpisode:
    """An attack plan after execution against target."""
    plan: AttackPlan
    conversation: List[Tuple[str, str]]  # [(prompt1, response1), ...]
    turn_scores: List[float]             # Per-turn heuristic scores
    intent_similarities: List[float]     # Cosine sim per prompt
    final_judge_score: int               # 1-4 judge score
    reward: float                        # Combined reward
    reward_components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "conversation": [{"prompt": p, "response": r} for p, r in self.conversation],
            "turn_scores": self.turn_scores,
            "intent_similarities": self.intent_similarities,
            "final_judge_score": self.final_judge_score,
            "reward": self.reward,
            "reward_components": self.reward_components,
        }
