"""SPRO Configuration."""

from dataclasses import dataclass


@dataclass
class SPROConfig:
    """Configuration for Single-Example SPRO training."""

    # Model settings
    model_name: str = "Koalacrown/qwen3-4b-multiturn-sft-4bit"
    max_seq_length: int = 32768
    lora_rank: int = 32

    # Training hyperparameters
    learning_rate: float = 1e-5
    kl_coef: float = 0.02
    clip_range: float = 0.2
    max_grad_norm: float = 1.0

    # SPRO advantage computation
    msa_scale: float = 0.1           # Scale factor for MSA component
    advantage_clip: float = 5.0      # Clip advantages to [-clip, clip]
    normalize_advantages: bool = True  # Normalize final advantages
    use_gae_advantages: bool = False   # Use GAE-style discounting (v2)
    use_query_aware_msa: bool = True   # Use query-aware MSA (per modification.md)
    beta: float = 1.0                  # Scaling for cumulative reward in query-aware MSA
    attack_focused_msa: bool = True    # Only MSA for opening+attack queries (handles variable lengths)
    use_three_component_msa: bool = True  # Use three-component MSA: reasoning + opening + attack (modification3.md)
    thinking_only_msa: bool = False  # Only MSA for reasoning/thinking section (ignores opening+attack)

    # SPRO settings
    group_size: int = 8  # Number of parallel attack plans per step
    max_attempts: int = 200  # Max attempts per intent before moving on
    success_threshold: int = 4  # Judge score required for success
    max_prompts: int = 10  # Maximum prompts per attack plan

    # Reward weights (must sum to 1.0)
    # Simplified: let SPRO's MSA handle per-token credit assignment
    outcome_weight: float = 0.85      # Judge score is the primary signal
    divergence_weight: float = 0.15   # Bonus for indirect successful attacks

    # API settings
    openrouter_api_key: str = ""
    target_model: str = "meta-llama/llama-3.1-8b-instruct"
    judge_model: str = "google/gemma-3-27b-it"
    max_concurrent_requests: int = 50

    # vLLM generation
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 9000
    gpu_memory_utilization: float = 0.4

    # Checkpointing
    checkpoint_dir: str = "./spro_checkpoints"
    save_every_n_intents: int = 5

    # Memory management
    max_rollouts_in_memory: int = 100  # Limit stored rollouts to prevent OOM
