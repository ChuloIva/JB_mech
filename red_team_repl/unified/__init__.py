"""
Unified Crescendo Attack System.

Single attack mode with configuration flags for all variations.
"""
from .config import CrescendoConfig, TargetMode
from .attack import CrescendoAttack, AttackResult, Turn
from .evaluator import Evaluator, RefusalResult, ObjectiveScore, FullAnalysis
from .generator import Generator, GeneratedTurn
from .strategies import Strategy, STRATEGIES, get_strategy, list_strategies
from .llm import LLMClient, LLMBackend, create_backend

__all__ = [
    # Main
    "CrescendoAttack",
    "CrescendoConfig",
    "AttackResult",
    "Turn",
    # Config
    "TargetMode",
    # Evaluation
    "Evaluator",
    "RefusalResult",
    "ObjectiveScore",
    "FullAnalysis",
    # Generation
    "Generator",
    "GeneratedTurn",
    # Strategies
    "Strategy",
    "STRATEGIES",
    "get_strategy",
    "list_strategies",
    # LLM
    "LLMClient",
    "LLMBackend",
    "create_backend",
]
