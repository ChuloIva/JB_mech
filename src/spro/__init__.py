"""
SPRO: Self-Guided Process Reward Optimization for Multi-Turn Jailbreak Training

Based on arXiv:2507.01551 - Self-Guided Process Reward Optimization
"""

# Core imports (no external deps)
from .config import SPROConfig
from .data_structures import AttackPlan, ExecutedEpisode
from .parsing import parse_sema_output
from .prompts import SEMA_SYSTEM_PROMPT, JUDGE_PROMPT
from .rewards import (
    SCORE_TO_REWARD,
    outcome_reward,
    score_turn_response,
    divergence_score,
    intent_trajectory_score,
    compute_episode_reward,
)

# Lazy imports for components with external deps
def __getattr__(name):
    """Lazy import for components with external dependencies."""
    if name == "IntentTracker":
        from .intent_tracker import IntentTracker
        return IntentTracker
    elif name == "OpenRouterClient":
        from .api_client import OpenRouterClient
        return OpenRouterClient
    elif name == "compute_log_prob_ratios":
        from .spro_core import compute_log_prob_ratios
        return compute_log_prob_ratios
    elif name == "compute_spro_advantages":
        from .spro_core import compute_spro_advantages
        return compute_spro_advantages
    elif name == "SingleExampleSPROTrainer":
        from .trainer import SingleExampleSPROTrainer
        return SingleExampleSPROTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config
    "SPROConfig",
    # Data structures
    "AttackPlan",
    "ExecutedEpisode",
    # Parsing
    "parse_sema_output",
    # Intent tracking
    "IntentTracker",
    # Rewards
    "SCORE_TO_REWARD",
    "outcome_reward",
    "score_turn_response",
    "divergence_score",
    "intent_trajectory_score",
    "compute_episode_reward",
    # Prompts
    "SEMA_SYSTEM_PROMPT",
    "JUDGE_PROMPT",
    # API
    "OpenRouterClient",
    # SPRO core
    "compute_log_prob_ratios",
    "compute_spro_advantages",
    # Trainer
    "SingleExampleSPROTrainer",
]

__version__ = "0.1.0"
