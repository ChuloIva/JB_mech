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

    # SPRO settings
    group_size: int = 8  # Number of parallel attack plans per step
    max_attempts: int = 200  # Max attempts per intent before moving on
    success_threshold: int = 4  # Judge score required for success
    max_prompts: int = 10  # Maximum prompts per attack plan

    # Reward weights (must sum to 1.0)
    outcome_weight: float = 0.50
    progress_weight: float = 0.20
    intent_weight: float = 0.20
    divergence_weight: float = 0.10

    # API settings
    openrouter_api_key: str = ""
    target_model: str = "meta-llama/llama-3.1-8b-instruct"
    judge_model: str = "google/gemma-3-27b-it"
    max_concurrent_requests: int = 16

    # vLLM generation
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 9000
    gpu_memory_utilization: float = 0.4

    # Checkpointing
    checkpoint_dir: str = "./spro_checkpoints"
    save_every_n_intents: int = 5
