"""
SPRO: Self-Guided Process Reward Optimization for Multi-Turn Jailbreak Training

Based on arXiv:2507.01551 - Self-Guided Process Reward Optimization
"""

# Core imports (no external deps)
from .config import SPROConfig
from .data_structures import AttackPlan, ExecutedEpisode, IDAScores, QueryBoundary, TrajectoryBoundaries
from .parsing import parse_sema_output, find_query_token_boundaries, find_reasoning_token_boundaries
from .prompts import SEMA_SYSTEM_PROMPT, JUDGE_PROMPT
from .rewards import (
    score_turn_response,
    divergence_bonus,
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
    elif name == "compute_spro_advantages_v2":
        from .spro_core import compute_spro_advantages_v2
        return compute_spro_advantages_v2
    elif name == "compute_spro_advantages_query_aware":
        from .spro_core import compute_spro_advantages_query_aware
        return compute_spro_advantages_query_aware
    elif name == "compute_spro_advantages_three_component":
        from .spro_core import compute_spro_advantages_three_component
        return compute_spro_advantages_three_component
    elif name == "compute_spro_advantages_paper":
        from .spro_core import compute_spro_advantages_paper
        return compute_spro_advantages_paper
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
    "IDAScores",
    "QueryBoundary",
    "TrajectoryBoundaries",
    # Parsing
    "parse_sema_output",
    "find_query_token_boundaries",
    "find_reasoning_token_boundaries",
    # Intent tracking
    "IntentTracker",
    # Rewards
    "score_turn_response",
    "divergence_bonus",
    "compute_episode_reward",
    # Prompts
    "SEMA_SYSTEM_PROMPT",
    "JUDGE_PROMPT",
    # API
    "OpenRouterClient",
    # SPRO core
    "compute_log_prob_ratios",
    "compute_spro_advantages",
    "compute_spro_advantages_v2",
    "compute_spro_advantages_query_aware",
    "compute_spro_advantages_three_component",
    "compute_spro_advantages_paper",
    # Trainer
    "SingleExampleSPROTrainer",
]

__version__ = "0.1.0"
