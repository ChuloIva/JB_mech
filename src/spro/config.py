"""SPRO Configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SPROConfig:
    """Configuration for Single-Example SPRO training."""

    # Model settings
    model_name: str = "Koalacrown/qwen3-4b-multiturn-sft-4bit"
    max_seq_length: int = 32768
    lora_rank: int = 32

    # Training hyperparameters
    learning_rate: float = 1e-6       # Paper uses 1e-6 with cosine decay
    kl_coef: float = 0.0              # Paper uses 0 (no KL penalty)
    entropy_coef: float = 0.001       # Paper uses 0.001 to maintain exploration
    clip_range: float = 0.2
    max_grad_norm: float = 1.0
    batch_ppo_update: bool = True     # Accumulate gradients, single update per attempt

    # SPRO advantage computation
    msa_scale: float = 1.0           # Scale factor for MSA component (paper uses 1.0)
    advantage_clip: float = 5.0      # Clip advantages to [-clip, clip]
    normalize_advantages: bool = False # Paper does NOT do extra final normalization
    use_paper_msa: bool = True        # Use paper's simple MSA (baseline subtraction only)
    use_gae_advantages: bool = False   # Use GAE-style discounting (v2)
    use_query_aware_msa: bool = False  # Use query-aware MSA (per modification.md)
    beta: float = 1.0                  # Scaling for cumulative reward in query-aware MSA
    attack_focused_msa: bool = True    # Only MSA for opening+attack queries (handles variable lengths)
    use_three_component_msa: bool = False  # Use three-component MSA: reasoning + opening + attack
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

    # Rollout logging
    log_rollouts: bool = True                    # Save best/worst rollouts per attempt
    rollout_log_dir: str = "./rollout_logs"      # Directory for rollout logs

    # Memory management
    max_rollouts_in_memory: int = 100  # Limit stored rollouts to prevent OOM

    # ==========================================================================
    # Refusal Direction Reward Settings
    # ==========================================================================
    # When enabled, uses local target model to capture activations and compute
    # refusal direction reward based on trajectory continuity.

    use_refusal_direction: bool = False          # Enable refusal direction reward

    # Local target model (required if use_refusal_direction=True)
    local_target_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    local_target_load_in_4bit: bool = True       # Use 4-bit quantization to save VRAM
    local_target_device: str = "auto"            # Device map for target model

    # Refusal probe settings
    refusal_direction_path: str = ""             # Path to saved refusal direction tensor
    refusal_probe_layers: List[int] = field(default_factory=lambda: [15, 20, 25])
    use_mean_pooling: bool = False               # Mean pool over response vs last token

    # Reward formulation
    refusal_threshold: float = 0.0               # Score below this = refusal triggered
    refusal_use_cosine: bool = True              # Use cosine similarity vs dot product

    # Continuity reward weights (how to combine trajectory signals)
    continuity_weight: float = 0.35              # Weight on unbroken acceptance chain
    smoothness_weight: float = 0.25              # Weight on gradual escalation
    min_score_weight: float = 0.25               # Weight on weakest-link score
    final_score_weight: float = 0.15             # Weight on final state

    # Reward combination (refusal vs IDA)
    refusal_reward_weight: float = 0.4           # Initial weight for refusal reward
    anneal_refusal_reward: bool = True           # Decrease refusal weight over training
    refusal_anneal_start: float = 0.5            # Starting refusal weight
    refusal_anneal_end: float = 0.1              # Final refusal weight
    refusal_anneal_steps: int = 100              # Steps to anneal over

    # Early stopping on refusal
    enable_early_stopping: bool = True           # Abort trajectories that trigger refusal
    early_stop_threshold: float = -0.3           # Refusal score threshold for early stop
    recovery_window: int = 1                     # Turns to attempt recovery before abort

    def get_refusal_weight(self, step: int) -> float:
        """Get refusal reward weight with optional annealing."""
        if not self.anneal_refusal_reward:
            return self.refusal_reward_weight

        if step >= self.refusal_anneal_steps:
            return self.refusal_anneal_end

        # Linear annealing
        progress = step / self.refusal_anneal_steps
        return self.refusal_anneal_start + progress * (self.refusal_anneal_end - self.refusal_anneal_start)
