"""Core SPRO algorithms: log-prob ratios and advantage computation."""

from typing import List, Tuple, Optional
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

    with torch.inference_mode():
        # Get logits from both models
        policy_outputs = policy_model(input_ids, attention_mask=attention_mask)
        policy_logits = policy_outputs.logits
        del policy_outputs

        ref_input = input_ids.to(ref_model.device)
        ref_mask = attention_mask.to(ref_model.device)
        ref_outputs = ref_model(ref_input, attention_mask=ref_mask)
        ref_logits = ref_outputs.logits.to(device)
        del ref_outputs, ref_input, ref_mask

        # Shift for next-token prediction
        policy_logits = policy_logits[:, :-1, :]
        ref_logits = ref_logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        # Compute log probs
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        del policy_logits, ref_logits

        # Gather log probs for actual tokens
        chosen_policy = policy_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        chosen_ref = ref_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        del policy_log_probs, ref_log_probs

        # Log ratio
        log_ratios = chosen_policy - chosen_ref
        del chosen_policy, chosen_ref

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


def compute_spro_advantages_query_aware(
    outcome_rewards: torch.Tensor,           # (batch,) - final episode rewards
    log_ratios: torch.Tensor,                # (batch, seq_len) - per-token log-prob ratios
    response_mask: torch.Tensor,             # (batch, seq_len) - 1 for response tokens
    query_boundaries: List[List[Tuple[int, int]]],  # Per-trajectory: [(start, end), ...]
    beta: float = 1.0,                       # Scaling for cumulative reward
    advantage_clip: float = 5.0,             # Clip advantages to [-clip, clip]
    attack_focused: bool = True,             # Use attack-centric comparison for variable lengths
) -> torch.Tensor:
    """
    Compute SPRO advantages with query-aware masking (per modification2.md spec).

    For variable-length trajectories, uses attack-centric comparison:
    - Opening query (first): Always aligned, gets MSA
    - Attack query (last): Always aligned, gets MSA
    - Middle setup queries: Get outcome advantage only (no MSA)

    This avoids trying to align variable-length middle sections.

    Args:
        outcome_rewards: Final episode rewards (batch,)
        log_ratios: Per-token log-prob ratios (batch, seq_len)
        response_mask: Binary mask for response tokens (batch, seq_len)
        query_boundaries: For each trajectory, list of (start_tok, end_tok) per query
        beta: Scaling factor for cumulative reward (default 1.0)
        advantage_clip: Clip advantages to this range (default 5.0)
        attack_focused: If True, only compute MSA for opening+attack queries

    Returns:
        Tensor of shape (batch, seq_len) with per-token advantages
    """
    batch_size, seq_len = log_ratios.shape
    device = log_ratios.device

    # 1. Compute CUMULATIVE log-prob ratios (Equation 11 from SPRO paper)
    # R_cum[t] = sum(beta * log_ratio[j] for j in 0..t)
    cumulative_rewards = torch.cumsum(beta * log_ratios, dim=1)  # (batch, seq_len)

    # 2. Outcome advantage (trajectory-level, normalized) - the GRPO term
    outcome_mean = outcome_rewards.mean()
    outcome_std = outcome_rewards.std() + 1e-8
    outcome_advantage = (outcome_rewards - outcome_mean) / outcome_std  # (batch,)

    # 3. Initialize MSA tensor (starts at 0 - middle queries stay at 0)
    msa = torch.zeros_like(log_ratios)  # (batch, seq_len)

    if attack_focused:
        # Attack-centric comparison: only MSA for opening (idx 0) and attack (idx -1)
        _compute_msa_for_position(
            msa, cumulative_rewards, query_boundaries, seq_len,
            position="opening"
        )
        _compute_msa_for_position(
            msa, cumulative_rewards, query_boundaries, seq_len,
            position="attack"
        )
    else:
        # Full MSA for all aligned query positions
        max_queries = max(len(qb) for qb in query_boundaries) if query_boundaries else 0

        for query_idx in range(max_queries):
            # Collect cumulative rewards at end of this query for each trajectory
            R_cums_at_query_end = []
            traj_indices = []

            for traj_idx, boundaries in enumerate(query_boundaries):
                if query_idx < len(boundaries):
                    start_tok, end_tok = boundaries[query_idx]
                    end_tok = min(end_tok, seq_len - 1)
                    R_cum = cumulative_rewards[traj_idx, end_tok].item()
                    R_cums_at_query_end.append(R_cum)
                    traj_indices.append(traj_idx)

            if len(R_cums_at_query_end) < 2:
                continue

            baseline = sum(R_cums_at_query_end) / len(R_cums_at_query_end)

            for i, traj_idx in enumerate(traj_indices):
                advantage = R_cums_at_query_end[i] - baseline
                start_tok, end_tok = query_boundaries[traj_idx][query_idx]
                start_tok = max(0, start_tok)
                end_tok = min(end_tok, seq_len - 1)
                msa[traj_idx, start_tok:end_tok + 1] = advantage

    # 4. Normalize MSA (only over non-zero positions to preserve middle=0)
    valid_msa = msa[response_mask.bool()]
    if valid_msa.numel() > 1 and valid_msa.abs().sum() > 0:
        msa_mean = valid_msa.mean()
        msa_std = valid_msa.std() + 1e-8
        msa = torch.where(
            response_mask.bool(),
            (msa - msa_mean) / msa_std,
            msa
        )

    # 5. Combine: outcome (GRPO term) + MSA (query-specific term)
    # Middle queries get outcome_advantage only (msa=0 for them after normalization shift)
    advantages = outcome_advantage.unsqueeze(1) + msa

    # 6. Apply response mask
    advantages = advantages * response_mask

    # 7. Final normalization
    valid_adv = advantages[response_mask.bool()]
    if valid_adv.numel() > 1:
        adv_mean = valid_adv.mean()
        adv_std = valid_adv.std() + 1e-8
        advantages = torch.where(
            response_mask.bool(),
            (advantages - adv_mean) / adv_std,
            advantages
        )

    # 8. Clip
    advantages = torch.clamp(advantages, -advantage_clip, advantage_clip)

    return advantages


def _compute_msa_for_position(
    msa: torch.Tensor,
    cumulative_rewards: torch.Tensor,
    query_boundaries: List[List[Tuple[int, int]]],
    seq_len: int,
    position: str,  # "opening" or "attack"
) -> None:
    """
    Compute MSA for a specific semantic position (opening or attack).

    Modifies msa tensor in-place.

    Args:
        msa: MSA tensor to modify (batch, seq_len)
        cumulative_rewards: Cumulative log-prob ratios (batch, seq_len)
        query_boundaries: Per-trajectory query boundaries
        seq_len: Sequence length
        position: "opening" (first query) or "attack" (last query)
    """
    R_cums = []
    traj_data = []  # (traj_idx, start_tok, end_tok)

    for traj_idx, boundaries in enumerate(query_boundaries):
        if not boundaries:
            continue

        if position == "opening":
            query_idx = 0
        elif position == "attack":
            query_idx = -1  # Last query
        else:
            raise ValueError(f"Unknown position: {position}")

        start_tok, end_tok = boundaries[query_idx]
        end_tok = min(end_tok, seq_len - 1)
        start_tok = max(0, start_tok)

        R_cum = cumulative_rewards[traj_idx, end_tok].item()
        R_cums.append(R_cum)
        traj_data.append((traj_idx, start_tok, end_tok))

    if len(R_cums) < 2:
        return

    # Compute baseline
    baseline = sum(R_cums) / len(R_cums)

    # Assign MSA
    for R_cum, (traj_idx, start_tok, end_tok) in zip(R_cums, traj_data):
        advantage = R_cum - baseline
        msa[traj_idx, start_tok:end_tok + 1] = advantage


def compute_spro_advantages_three_component(
    outcome_rewards: torch.Tensor,           # (batch,) - final episode rewards
    log_ratios: torch.Tensor,                # (batch, seq_len) - per-token log-prob ratios
    response_mask: torch.Tensor,             # (batch, seq_len) - 1 for response tokens
    trajectory_boundaries: List[dict],       # Per-trajectory: {"reasoning": (s,e), "opening": (s,e), "attack": (s,e)}
    beta: float = 1.0,                       # Scaling for cumulative reward
    advantage_clip: float = 5.0,             # Clip advantages to [-clip, clip]
) -> torch.Tensor:
    """
    Compute SPRO advantages with three-component MSA (per modification3.md spec).

    Computes MSA at three key checkpoints:
    1. Reasoning section (strategy planning)
    2. Opening query (first query)
    3. Attack query (final query)

    Middle setup queries get outcome advantage only (no MSA).

    Args:
        outcome_rewards: Final episode rewards (batch,)
        log_ratios: Per-token log-prob ratios (batch, seq_len)
        response_mask: Binary mask for response tokens (batch, seq_len)
        trajectory_boundaries: For each trajectory, dict with reasoning/opening/attack bounds
        beta: Scaling factor for cumulative reward (default 1.0)
        advantage_clip: Clip advantages to this range (default 5.0)

    Returns:
        Tensor of shape (batch, seq_len) with per-token advantages
    """
    batch_size, seq_len = log_ratios.shape
    device = log_ratios.device

    # 1. Compute CUMULATIVE log-prob ratios (Equation 11 from SPRO paper)
    cumulative_rewards = torch.cumsum(beta * log_ratios, dim=1)  # (batch, seq_len)

    # 2. Outcome advantage (trajectory-level, normalized) - the GRPO term
    outcome_mean = outcome_rewards.mean()
    outcome_std = outcome_rewards.std() + 1e-8
    outcome_advantage = (outcome_rewards - outcome_mean) / outcome_std  # (batch,)

    # 3. Initialize MSA tensor (0 for middle queries)
    msa = torch.zeros_like(log_ratios)  # (batch, seq_len)

    # 4. Compute MSA for each of the three components
    for component in ["reasoning", "opening", "attack"]:
        R_cums = []
        traj_data = []  # (traj_idx, start_tok, end_tok)

        for traj_idx, bounds in enumerate(trajectory_boundaries):
            if bounds is None or component not in bounds:
                continue

            start_tok, end_tok = bounds[component]
            if start_tok == end_tok == 0:
                # Empty boundary, skip
                continue

            end_tok = min(end_tok, seq_len - 1)
            start_tok = max(0, start_tok)

            R_cum = cumulative_rewards[traj_idx, end_tok].item()
            R_cums.append(R_cum)
            traj_data.append((traj_idx, start_tok, end_tok))

        if len(R_cums) < 2:
            continue

        # Compute baseline for this component
        baseline = sum(R_cums) / len(R_cums)

        # Assign MSA to tokens in this component
        for R_cum, (traj_idx, start_tok, end_tok) in zip(R_cums, traj_data):
            advantage = R_cum - baseline
            msa[traj_idx, start_tok:end_tok + 1] = advantage

    # 5. Normalize MSA
    valid_msa = msa[response_mask.bool()]
    if valid_msa.numel() > 1 and valid_msa.abs().sum() > 0:
        msa_mean = valid_msa.mean()
        msa_std = valid_msa.std() + 1e-8
        msa = torch.where(
            response_mask.bool(),
            (msa - msa_mean) / msa_std,
            msa
        )

    # 6. Combine: outcome (GRPO term) + MSA (component-specific term)
    advantages = outcome_advantage.unsqueeze(1) + msa

    # 7. Apply response mask
    advantages = advantages * response_mask

    # 8. Final normalization
    valid_adv = advantages[response_mask.bool()]
    if valid_adv.numel() > 1:
        adv_mean = valid_adv.mean()
        adv_std = valid_adv.std() + 1e-8
        advantages = torch.where(
            response_mask.bool(),
            (advantages - adv_mean) / adv_std,
            advantages
        )

    # 9. Clip
    advantages = torch.clamp(advantages, -advantage_clip, advantage_clip)

    return advantages
