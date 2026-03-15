"""Core SPRO algorithms: log-prob ratios and advantage computation."""

import torch
import torch.nn.functional as F


def compute_log_prob_ratios(
    policy_model,
    ref_model,
    input_ids: torch.Tensor,       # (batch, seq_len)
    attention_mask: torch.Tensor,  # (batch, seq_len)
    response_start_idx: int,
) -> torch.Tensor:
    """
    Compute log(pi_theta / pi_ref) for each token in response.

    Args:
        policy_model: The policy model being trained
        ref_model: The frozen reference model
        input_ids: Token IDs (batch, seq_len)
        attention_mask: Attention mask (batch, seq_len)
        response_start_idx: Index where response tokens begin

    Returns:
        Tensor of shape (batch, seq_len) with log-prob ratios
    """
    device = input_ids.device

    with torch.no_grad():
        # Get logits from both models
        policy_logits = policy_model(input_ids, attention_mask=attention_mask).logits
        ref_logits = ref_model(
            input_ids.to(ref_model.device),
            attention_mask=attention_mask.to(ref_model.device)
        ).logits
        ref_logits = ref_logits.to(device)

        # Shift for next-token prediction
        policy_logits = policy_logits[:, :-1, :]
        ref_logits = ref_logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        # Compute log probs
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Gather log probs for actual tokens
        chosen_policy = policy_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        chosen_ref = ref_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Log ratio
        log_ratios = chosen_policy - chosen_ref

        # Pad to original length
        log_ratios = F.pad(log_ratios, (1, 0), value=0.0)

        # Zero out prompt tokens
        log_ratios[:, :response_start_idx] = 0.0

    return log_ratios


def compute_spro_advantages(
    outcome_rewards: torch.Tensor,      # (batch,) - final episode rewards
    log_ratios: torch.Tensor,           # (batch, seq_len) - per-token log-prob ratios
    response_mask: torch.Tensor,        # (batch, seq_len) - 1 for response tokens
    msa_scale: float = 0.1,             # Scale factor for MSA component
    advantage_clip: float = 5.0,        # Clip advantages to [-clip, clip]
    normalize_final: bool = True,       # Normalize final advantages
) -> torch.Tensor:
    """
    Compute SPRO advantages: outcome + MSA (Masked Step Advantage).

    A[i,t] = normalized_outcome[i] + scale * normalized_msa[i,t]

    Based on arXiv:2507.01551 - Self-Guided Process Reward Optimization

    FIXES applied:
    - MSA is normalized per-timestep (not just centered)
    - MSA is scaled down to prevent dominating outcome signal
    - Final advantages are normalized and clipped

    Args:
        outcome_rewards: Final episode rewards (batch,)
        log_ratios: Per-token log-prob ratios (batch, seq_len)
        response_mask: Binary mask for response tokens (batch, seq_len)
        msa_scale: Scaling factor for MSA component (default 0.1)
        advantage_clip: Clip advantages to this range (default 5.0)
        normalize_final: Whether to normalize final advantages (default True)

    Returns:
        Tensor of shape (batch, seq_len) with per-token advantages
    """
    batch_size, seq_len = log_ratios.shape
    device = log_ratios.device

    # 1. Outcome advantage (trajectory-level, normalized)
    outcome_mean = outcome_rewards.mean()
    outcome_std = outcome_rewards.std() + 1e-8
    outcome_advantage = (outcome_rewards - outcome_mean) / outcome_std  # (batch,)

    # Expand to (batch, seq_len)
    outcome_expanded = outcome_advantage.unsqueeze(1).expand(-1, seq_len)

    # 2. Cumulative process rewards (Eq. 11 from SPRO paper)
    # Use per-token log ratios directly instead of cumsum to avoid unbounded growth
    # This is a simplification that works better in practice
    cumulative = log_ratios  # (batch, seq_len)

    # 3. MSA: center AND normalize at each timestep
    msa = torch.zeros_like(cumulative)
    for t in range(seq_len):
        valid_mask = response_mask[:, t].bool()
        if valid_mask.sum() > 1:  # Need at least 2 for std
            values_t = cumulative[valid_mask, t]
            mean_t = values_t.mean()
            std_t = values_t.std() + 1e-8
            # Normalize: (x - mean) / std
            msa[valid_mask, t] = (cumulative[valid_mask, t] - mean_t) / std_t
        elif valid_mask.sum() == 1:
            # Single sample: just center
            msa[valid_mask, t] = 0.0

    # 4. Combined advantage with scaling
    # Outcome is the primary signal, MSA provides per-token credit assignment
    advantages = outcome_expanded + msa_scale * msa

    # 5. Apply response mask
    advantages = advantages * response_mask

    # 6. Final normalization (standard PPO practice)
    if normalize_final:
        # Only normalize over valid (masked) positions
        valid_advantages = advantages[response_mask.bool()]
        if valid_advantages.numel() > 1:
            adv_mean = valid_advantages.mean()
            adv_std = valid_advantages.std() + 1e-8
            # Normalize in-place for masked positions
            advantages = torch.where(
                response_mask.bool(),
                (advantages - adv_mean) / adv_std,
                advantages
            )

    # 7. Clip to prevent extreme values
    advantages = torch.clamp(advantages, -advantage_clip, advantage_clip)

    return advantages


def compute_spro_advantages_v2(
    outcome_rewards: torch.Tensor,      # (batch,) - final episode rewards
    log_ratios: torch.Tensor,           # (batch, seq_len) - per-token log-prob ratios
    response_mask: torch.Tensor,        # (batch, seq_len) - 1 for response tokens
    gamma: float = 0.99,                # Discount factor for GAE-style computation
    advantage_clip: float = 5.0,        # Clip advantages to [-clip, clip]
) -> torch.Tensor:
    """
    Alternative SPRO advantage computation using GAE-style discounting.

    This version uses discount factor to prevent cumulative advantages
    from growing unbounded with sequence length.

    Args:
        outcome_rewards: Final episode rewards (batch,)
        log_ratios: Per-token log-prob ratios (batch, seq_len)
        response_mask: Binary mask for response tokens (batch, seq_len)
        gamma: Discount factor (default 0.99)
        advantage_clip: Clip advantages to this range (default 5.0)

    Returns:
        Tensor of shape (batch, seq_len) with per-token advantages
    """
    batch_size, seq_len = log_ratios.shape
    device = log_ratios.device

    # 1. Outcome advantage (normalized)
    outcome_mean = outcome_rewards.mean()
    outcome_std = outcome_rewards.std() + 1e-8
    outcome_advantage = (outcome_rewards - outcome_mean) / outcome_std

    # 2. Compute discounted cumulative rewards (reverse order)
    # This prevents advantages from growing unbounded
    discounted = torch.zeros_like(log_ratios)
    running_sum = torch.zeros(batch_size, device=device)

    for t in range(seq_len - 1, -1, -1):
        running_sum = log_ratios[:, t] + gamma * running_sum * response_mask[:, t]
        discounted[:, t] = running_sum

    # 3. Normalize discounted advantages per-batch
    valid_disc = discounted[response_mask.bool()]
    if valid_disc.numel() > 1:
        disc_mean = valid_disc.mean()
        disc_std = valid_disc.std() + 1e-8
        discounted = torch.where(
            response_mask.bool(),
            (discounted - disc_mean) / disc_std,
            discounted
        )

    # 4. Combine: outcome provides direction, discounted provides per-token signal
    advantages = outcome_advantage.unsqueeze(1) + 0.5 * discounted
    advantages = advantages * response_mask

    # 5. Final normalization
    valid_adv = advantages[response_mask.bool()]
    if valid_adv.numel() > 1:
        adv_mean = valid_adv.mean()
        adv_std = valid_adv.std() + 1e-8
        advantages = torch.where(
            response_mask.bool(),
            (advantages - adv_mean) / adv_std,
            advantages
        )

    # 6. Clip
    advantages = torch.clamp(advantages, -advantage_clip, advantage_clip)

    return advantages
