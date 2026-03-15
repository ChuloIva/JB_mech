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
) -> torch.Tensor:
    """
    Compute SPRO advantages: outcome + MSA (Masked Step Advantage).

    A[i,t] = normalized_outcome[i] + (R~[i,t] - masked_mean_t(R~))

    Based on arXiv:2507.01551 - Self-Guided Process Reward Optimization

    Args:
        outcome_rewards: Final episode rewards (batch,)
        log_ratios: Per-token log-prob ratios (batch, seq_len)
        response_mask: Binary mask for response tokens (batch, seq_len)

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
    cumulative = torch.cumsum(log_ratios, dim=1)  # (batch, seq_len)

    # 3. MSA: center at each timestep (NOT normalized by std, per paper)
    msa = torch.zeros_like(cumulative)
    for t in range(seq_len):
        valid_mask = response_mask[:, t].bool()
        if valid_mask.sum() > 0:
            mean_t = cumulative[valid_mask, t].mean()
            msa[:, t] = cumulative[:, t] - mean_t

    # 4. Combined advantage
    advantages = outcome_expanded + msa

    # Zero out non-response tokens
    advantages = advantages * response_mask

    return advantages
