"""
Delta Attack v5.1 - Contrastive Why-Interrogation + Steering

Transfer attack using a local whitebox proxy model to attack closed models.
Uses contrastive "why refuse?" vs "why comply?" reasoning for steering.
"""

from .config import DeltaAttackConfig
from .attack import transfer_attack_loop, TransferAttackResult, run_attack_batch
from .api import query_target_model, get_api_client, generate_compliant_suffix
from .evaluation import is_compliant, EvaluationResult
from .steering import generate_with_steering, capture_steered_activations
from .interrogation import run_why_interrogation, WhyInterrogationResult

__all__ = [
    # Config
    "DeltaAttackConfig",
    # Main attack
    "transfer_attack_loop",
    "TransferAttackResult",
    "run_attack_batch",
    # API
    "query_target_model",
    "get_api_client",
    "generate_compliant_suffix",
    # Evaluation
    "is_compliant",
    "EvaluationResult",
    # Steering
    "generate_with_steering",
    "capture_steered_activations",
    # Interrogation
    "run_why_interrogation",
    "WhyInterrogationResult",
]
