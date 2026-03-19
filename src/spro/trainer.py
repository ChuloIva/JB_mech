"""Single-Example SPRO Trainer."""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict

import gc
import torch
import torch.nn.functional as F

from .config import SPROConfig
from .data_structures import AttackPlan, ExecutedEpisode
from .parsing import parse_sema_output, find_query_token_boundaries, find_reasoning_token_boundaries
from .data_structures import QueryBoundary, TrajectoryBoundaries
from .prompts import SEMA_SYSTEM_PROMPT
from .rewards import compute_episode_reward, score_turn_response
from .intent_tracker import IntentTracker
from .api_client import OpenRouterClient
from .spro_core import (
    compute_log_prob_ratios,
    compute_spro_advantages,
    compute_spro_advantages_v2,
    compute_spro_advantages_query_aware,
    compute_spro_advantages_three_component,
    compute_spro_advantages_paper,
)


class SingleExampleSPROTrainer:
    """
    Train on one intent at a time until success, then move on.
    Uses SEMA-style generation (all prompts at once) + SPRO advantages.
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        tokenizer,
        intent_tracker: IntentTracker,
        config: SPROConfig,
        verbose: bool = True,
        local_target=None,  # Optional pre-loaded LocalTargetWithActivations
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.intent_tracker = intent_tracker
        self.config = config
        self.verbose = verbose

        # API client (used when not using local target, or for judge)
        self.api_client = OpenRouterClient(config)

        # Local target for refusal direction reward
        self.local_target = local_target
        if config.use_refusal_direction and local_target is None:
            self.log("Initializing local target model for refusal direction reward...")
            from .local_target import LocalTargetWithActivations
            self.local_target = LocalTargetWithActivations(
                model_name=config.local_target_model,
                layers=config.refusal_probe_layers,
                load_in_4bit=config.local_target_load_in_4bit,
                device_map=config.local_target_device,
            )
            # Load refusal direction if path provided
            if config.refusal_direction_path:
                self.local_target.load_refusal_direction(config.refusal_direction_path)
            self.log("Local target initialized.")

        # Training step counter (for reward annealing)
        self.training_step = 0

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, policy_model.parameters()),
            lr=config.learning_rate,
        )

        # Logging
        self.successful_rollouts = []
        self.all_rollouts = []
        self.training_stats = []

    def log(self, msg: str, indent: int = 0):
        """Print if verbose mode is on."""
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}{msg}")

    def log_separator(self, char: str = "-", length: int = 60):
        """Print a separator line."""
        if self.verbose:
            print(char * length)

    def clear_memory(self, aggressive: bool = False):
        """Clear GPU memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()

    def save_rollout_samples(
        self,
        episodes: List[ExecutedEpisode],
        intent: str,
        attempt: int,
    ):
        """Save best and worst rollouts from this attempt to log files."""
        if not self.config.log_rollouts:
            return

        # Create log directory
        log_dir = self.config.rollout_log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Sort by reward
        sorted_eps = sorted(episodes, key=lambda e: e.reward, reverse=True)
        best_ep = sorted_eps[0]
        worst_ep = sorted_eps[-1]

        # Create intent-specific subdirectory (sanitize intent for filename)
        intent_slug = intent[:50].replace(" ", "_").replace("/", "-")
        intent_dir = os.path.join(log_dir, intent_slug)
        os.makedirs(intent_dir, exist_ok=True)

        # Save best
        best_file = os.path.join(intent_dir, f"attempt_{attempt:03d}_best.txt")
        self._write_rollout_file(best_file, best_ep, "BEST", attempt)

        # Save worst
        worst_file = os.path.join(intent_dir, f"attempt_{attempt:03d}_worst.txt")
        self._write_rollout_file(worst_file, worst_ep, "WORST", attempt)

        # Print summary
        self.log(f"\nRollouts saved to {intent_dir}/", indent=1)
        self.log(f"  Best:  reward={best_ep.reward:.4f}, score={best_ep.final_judge_score}/4", indent=1)
        self.log(f"  Worst: reward={worst_ep.reward:.4f}, score={worst_ep.final_judge_score}/4", indent=1)

    def _write_rollout_file(
        self,
        filepath: str,
        episode: ExecutedEpisode,
        label: str,
        attempt: int,
    ):
        """Write a single rollout to file."""
        with open(filepath, "w") as f:
            f.write(f"{'='*70}\n")
            f.write(f"{label} ROLLOUT - Attempt {attempt}\n")
            f.write(f"{'='*70}\n\n")

            # Reward info
            f.write(f"Intent: {episode.plan.intent}\n")
            f.write(f"Reward: {episode.reward:.4f}\n")
            if episode.ida_scores:
                f.write(f"IDA Scores: A={episode.ida_scores.alignment}, "
                       f"R={episode.ida_scores.risk}, D={episode.ida_scores.detail}\n")
            f.write(f"Final Score: {episode.final_judge_score}/4\n")
            f.write("\n")

            # Thinking/reasoning section
            f.write(f"{'='*70}\n")
            f.write("THINKING\n")
            f.write(f"{'='*70}\n")
            f.write(episode.plan.thinking)
            f.write("\n\n")

            # Generated prompts (just the model's output, not responses)
            f.write(f"{'='*70}\n")
            f.write("GENERATED PROMPTS\n")
            f.write(f"{'='*70}\n")
            for i, prompt in enumerate(episode.plan.prompts, 1):
                f.write(f"\n--- Prompt {i} ---\n")
                f.write(prompt)
                f.write("\n")

            # Turn scores summary
            f.write(f"\n{'='*70}\n")
            f.write("TURN SCORES\n")
            f.write(f"{'='*70}\n")
            for i, (score, sim) in enumerate(zip(episode.turn_scores, episode.intent_similarities), 1):
                f.write(f"Turn {i}: score={score:+.2f}, intent_sim={sim:.2f}\n")

    def print_full_episode(self, episode: ExecutedEpisode, title: str = "EPISODE"):
        """Print full details of an episode."""
        print(f"\n{'#'*70}")
        print(f"# {title}")
        print(f"{'#'*70}")

        print(f"\nINTENT: {episode.plan.intent}")
        if episode.ida_scores:
            ida = episode.ida_scores
            print(f"IDA SCORES: Alignment={ida.alignment} Risk={ida.risk} Detail={ida.detail}")
            print(f"IDA REWARD: {ida.compute_reward():.4f}")
        print(f"TOTAL REWARD: {episode.reward:.4f}")

        print(f"\n{'='*70}")
        print("THINKING")
        print(f"{'='*70}")
        print(episode.plan.thinking)

        print(f"\n{'='*70}")
        print("MULTI-TURN CONVERSATION")
        print(f"{'='*70}")

        for i, (prompt, response) in enumerate(episode.conversation, 1):
            print(f"\n{'─'*70}")
            print(f"TURN {i}/{len(episode.conversation)}")
            print(f"{'─'*70}")
            print(f"\n[ATTACKER PROMPT]")
            print(prompt)
            print(f"\n[TARGET RESPONSE]")
            print(response)
            print(f"\nTurn Score: {episode.turn_scores[i-1]:+.2f}")
            if i <= len(episode.intent_similarities):
                print(f"Intent Similarity: {episode.intent_similarities[i-1]:.3f}")

        print(f"\n{'='*70}")
        print("REWARD BREAKDOWN")
        print(f"{'='*70}")
        for k, v in episode.reward_components.items():
            if isinstance(v, int):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:+.4f}")
        print(f"  TOTAL: {episode.reward:+.4f}")
        if episode.ida_scores and episode.ida_scores.rationale:
            print(f"\nJUDGE RATIONALE: {episode.ida_scores.rationale}")
        print(f"{'#'*70}\n")

    def print_episode_summary(self, episode: ExecutedEpisode, plan_idx: int):
        """Print compact summary of an episode (for parallel execution)."""
        early_stop_str = " [EARLY STOP]" if episode.early_stopped else ""
        print(f"\n  Plan {plan_idx + 1}: {len(episode.conversation)} turns{early_stop_str}, "
              f"scores={[f'{s:+.2f}' for s in episode.turn_scores]}")
        print(f"    Intent sims: {[f'{s:.2f}' for s in episode.intent_similarities]}")

        # Show refusal scores if available
        if episode.refusal_info and episode.refusal_info.refusal_scores:
            ref_scores = [f"{s:.2f}" for s in episode.refusal_info.refusal_scores]
            print(f"    Refusal scores: {ref_scores} (cont={episode.refusal_info.continuity:.2f})")

    def build_sema_prompt(self, intent: str) -> str:
        """Build SEMA-style prompt for multi-turn attack generation."""
        system_prompt = SEMA_SYSTEM_PROMPT.format(max_turns=self.config.max_prompts)
        user_message = f"Request: {intent}. Your multi-turn prompts:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_attack_plans(self, intent: str, n: int) -> List[AttackPlan]:
        """Generate n attack plans using vLLM (SEMA-style)."""
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.policy_model)

        prompt = self.build_sema_prompt(intent)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_length = prompt_ids.shape[1]

        self.log(f"Generating {n} attack plans...", indent=1)

        device = next(self.policy_model.parameters()).device
        outputs = self.policy_model.generate(
            prompt_ids.repeat(n, 1).to(device),
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        plans = []
        for idx, output_ids in enumerate(outputs):
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            gen_text = self.tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=True)
            thinking, prompts = parse_sema_output(gen_text)

            if len(prompts) < 2:
                self.log(f"Plan {idx+1}: INVALID (only {len(prompts)} prompts)", indent=2)
                continue

            # Find token boundaries for each query
            token_bounds = find_query_token_boundaries(
                self.tokenizer,
                output_ids,
                prompt_length,
                len(prompts),
            )

            # Convert to QueryBoundary objects
            query_boundaries = [
                QueryBoundary(query_idx=i, start=start, end=end)
                for i, (start, end) in enumerate(token_bounds)
            ]

            # Find reasoning boundaries
            reasoning_bounds = find_reasoning_token_boundaries(
                self.tokenizer,
                output_ids,
                prompt_length,
            )

            # Build trajectory boundaries (reasoning, opening, attack)
            trajectory_boundaries = None
            if token_bounds:
                opening_bounds = token_bounds[0] if len(token_bounds) >= 1 else (0, 0)
                attack_bounds = token_bounds[-1] if len(token_bounds) >= 1 else (0, 0)
                trajectory_boundaries = TrajectoryBoundaries(
                    reasoning=reasoning_bounds,
                    opening=opening_bounds,
                    attack=attack_bounds,
                )

            plans.append(AttackPlan(
                intent=intent,
                plan_text=gen_text,
                thinking=thinking,
                prompts=prompts,
                token_ids=output_ids.clone().detach().cpu(),
                response_start_idx=prompt_length,
                query_boundaries=query_boundaries,
                trajectory_boundaries=trajectory_boundaries,
            ))

            self.log(f"Plan {idx+1}: {len(prompts)} prompts, {len(query_boundaries)} query bounds, "
                     f"reasoning={reasoning_bounds[1]-reasoning_bounds[0]} toks", indent=2)

        # Clear generation cache
        del outputs, prompt_ids
        self.clear_memory()

        self.log(f"Valid plans: {len(plans)}/{n}", indent=1)
        return plans

    async def execute_plan(self, plan: AttackPlan, plan_idx: int) -> ExecutedEpisode:
        """Execute a plan step-by-step against target (turns are sequential)."""
        # Use refusal-aware execution if enabled
        if self.config.use_refusal_direction and self.local_target is not None:
            return await self._execute_plan_with_refusal(plan, plan_idx)

        # Original execution path (API-based)
        conversation = []
        turn_scores = []
        context = []

        for turn_idx, prompt in enumerate(plan.prompts):
            response = await self.api_client.query_target(prompt, context)
            turn_score = score_turn_response(response)

            conversation.append((prompt, response))
            turn_scores.append(turn_score)

            context.append({"role": "user", "content": prompt})
            context.append({"role": "assistant", "content": response})

        similarities = self.intent_tracker.compute_similarities(
            plan.intent, plan.prompts
        )

        return ExecutedEpisode(
            plan=plan,
            conversation=conversation,
            turn_scores=turn_scores,
            intent_similarities=similarities,
        )

    async def _execute_plan_with_refusal(self, plan: AttackPlan, plan_idx: int) -> ExecutedEpisode:
        """
        Execute plan with API generation + local model refusal scoring.

        Optimized approach:
        1. API generates target responses (fast, parallelizable on their infra)
        2. Local model does single forward pass per turn for activation capture
           (no generation, just scoring - much faster)
        """
        from .refusal_reward import create_refusal_reward_info

        conversation = []
        turn_scores = []
        refusal_scores = []
        context = []  # For API calls (list of dicts)
        messages = []  # For local model scoring (list of dicts)
        early_stopped = False

        refusal_triggered_at = None
        turns_since_refusal = 0

        for turn_idx, prompt in enumerate(plan.prompts):
            # Build message for refusal scoring (before response)
            messages_for_scoring = messages + [{"role": "user", "content": prompt}]

            # Score refusal via local model (single forward pass, no generation)
            refusal_score = self.local_target.score_prompt_refusal(
                messages_for_scoring,
                use_cosine=self.config.refusal_use_cosine,
            )
            refusal_scores.append(refusal_score)

            # Check for early stopping before making API call
            if self.config.enable_early_stopping:
                if refusal_score < self.config.early_stop_threshold:
                    if refusal_triggered_at is None:
                        refusal_triggered_at = turn_idx
                    turns_since_refusal += 1

                    # Abort if refusal persists beyond recovery window
                    if turns_since_refusal > self.config.recovery_window:
                        # Still need a response for the conversation record
                        response = await self.api_client.query_target(prompt, context)
                        turn_score = score_turn_response(response)
                        conversation.append((prompt, response))
                        turn_scores.append(turn_score)
                        early_stopped = True
                        self.log(f"  Plan {plan_idx+1}: Early stop at turn {turn_idx+1} "
                                f"(refusal={refusal_score:.2f})", indent=2)
                        break
                else:
                    turns_since_refusal = 0

            # Get response from API (fast, their infrastructure)
            response = await self.api_client.query_target(prompt, context)
            turn_score = score_turn_response(response)

            conversation.append((prompt, response))
            turn_scores.append(turn_score)

            # Update context for next turn
            context.append({"role": "user", "content": prompt})
            context.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})

        # Compute intent similarities
        prompts_executed = [p for p, _ in conversation]
        similarities = self.intent_tracker.compute_similarities(
            plan.intent, prompts_executed
        )

        # Create refusal reward info
        refusal_info = create_refusal_reward_info(refusal_scores, self.config)

        return ExecutedEpisode(
            plan=plan,
            conversation=conversation,
            turn_scores=turn_scores,
            intent_similarities=similarities,
            refusal_info=refusal_info,
            early_stopped=early_stopped,
        )

    async def execute_plans_parallel(self, plans: List[AttackPlan]) -> List[ExecutedEpisode]:
        """Execute all plans in parallel (each plan's turns are sequential)."""
        self.log("\n" + "="*60)
        self.log(f"EXECUTING {len(plans)} PLANS IN PARALLEL")
        self.log("="*60)

        # Launch all plans concurrently - semaphore handles rate limiting
        tasks = [self.execute_plan(plan, idx) for idx, plan in enumerate(plans)]
        episodes = await asyncio.gather(*tasks)

        # Print summaries after all complete
        for idx, ep in enumerate(episodes):
            self.print_episode_summary(ep, idx)

        return list(episodes)

    async def batch_judge_parallel(self, episodes: List[ExecutedEpisode]):
        """Judge all episodes in parallel using IDA scoring."""
        self.log("\n" + "="*60)
        self.log(f"JUDGING {len(episodes)} CONVERSATIONS (IDA SCORING)")
        self.log("="*60)

        tasks = [
            self.api_client.judge_conversation(ep.plan.intent, ep.conversation)
            for ep in episodes
        ]
        ida_scores_list = await asyncio.gather(*tasks)

        for idx, (ep, ida_scores) in enumerate(zip(episodes, ida_scores_list)):
            ep.ida_scores = ida_scores
            ida_reward = ida_scores.compute_reward()
            self.log(f"  Plan {idx + 1}: A={ida_scores.alignment} R={ida_scores.risk} "
                    f"D={ida_scores.detail} → IDA={ida_reward:.3f}", indent=1)

    def compute_batch_spro_advantages(
        self,
        episodes: List[ExecutedEpisode],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SPRO advantages for batch of episodes."""
        device = next(self.policy_model.parameters()).device

        all_log_ratios = []
        all_masks = []
        all_query_boundaries = []
        max_len = 0

        for ep in episodes:
            token_ids = ep.plan.token_ids.clone().unsqueeze(0).to(device)
            attention_mask = torch.ones_like(token_ids)

            log_ratios = compute_log_prob_ratios(
                self.policy_model,
                self.ref_model,
                token_ids,
                attention_mask,
                ep.plan.response_start_idx,
            ).squeeze(0)

            mask = torch.zeros_like(log_ratios)
            mask[ep.plan.response_start_idx:] = 1.0

            # Detach to break graph, keep on GPU for speed
            all_log_ratios.append(log_ratios.detach())
            all_masks.append(mask)
            max_len = max(max_len, log_ratios.shape[0])

            # Extract query boundaries as (start, end) tuples
            qb_tuples = [(qb.start, qb.end) for qb in ep.plan.query_boundaries]
            all_query_boundaries.append(qb_tuples)

            del token_ids, attention_mask

        # Extract trajectory boundaries for three-component MSA
        all_trajectory_boundaries = []
        for ep in episodes:
            if ep.plan.trajectory_boundaries:
                tb = ep.plan.trajectory_boundaries
                all_trajectory_boundaries.append({
                    "reasoning": tb.reasoning,
                    "opening": tb.opening,
                    "attack": tb.attack,
                })
            else:
                all_trajectory_boundaries.append(None)

        log_ratios_batch = torch.zeros(len(episodes), max_len, device=device)
        masks_batch = torch.zeros(len(episodes), max_len, device=device)

        for i, (lr, m) in enumerate(zip(all_log_ratios, all_masks)):
            log_ratios_batch[i, :lr.shape[0]] = lr
            masks_batch[i, :m.shape[0]] = m

        # Check if we have boundaries for different MSA modes
        has_query_boundaries = all(len(qb) > 0 for qb in all_query_boundaries)
        has_trajectory_boundaries = all(tb is not None for tb in all_trajectory_boundaries)

        # Paper-aligned MSA (highest priority if enabled)
        if self.config.use_paper_msa:
            # Determine which components get MSA (if selective)
            if has_trajectory_boundaries and self.config.thinking_only_msa:
                components = ["reasoning"]
                self.log("  Using paper MSA (thinking-only)", indent=1)
                advantages = compute_spro_advantages_paper(
                    rewards.to(device),
                    log_ratios_batch,
                    masks_batch,
                    beta=self.config.beta,
                    advantage_clip=self.config.advantage_clip,
                    trajectory_boundaries=all_trajectory_boundaries,
                    components=components,
                )
            elif has_trajectory_boundaries and self.config.use_three_component_msa:
                components = ["reasoning", "opening", "attack"]
                self.log("  Using paper MSA (reasoning + opening + attack)", indent=1)
                advantages = compute_spro_advantages_paper(
                    rewards.to(device),
                    log_ratios_batch,
                    masks_batch,
                    beta=self.config.beta,
                    advantage_clip=self.config.advantage_clip,
                    trajectory_boundaries=all_trajectory_boundaries,
                    components=components,
                )
            else:
                self.log("  Using paper MSA (whole rollout)", indent=1)
                advantages = compute_spro_advantages_paper(
                    rewards.to(device),
                    log_ratios_batch,
                    masks_batch,
                    beta=self.config.beta,
                    advantage_clip=self.config.advantage_clip,
                )
        elif has_trajectory_boundaries and self.config.use_three_component_msa:
            # Determine which components to use for MSA
            if self.config.thinking_only_msa:
                components = ["reasoning"]
                self.log("  Using thinking-only MSA (reasoning)", indent=1)
            else:
                components = ["reasoning", "opening", "attack"]
                self.log("  Using three-component MSA (reasoning + opening + attack)", indent=1)

            advantages = compute_spro_advantages_three_component(
                rewards.to(device),
                log_ratios_batch,
                masks_batch,
                trajectory_boundaries=all_trajectory_boundaries,
                beta=self.config.beta,
                advantage_clip=self.config.advantage_clip,
                components=components,
            )
        elif has_query_boundaries and self.config.use_query_aware_msa:
            # Use query-aware SPRO (per modification.md spec)
            mode = "attack-focused" if self.config.attack_focused_msa else "all-queries"
            self.log(f"  Using query-aware MSA ({mode})", indent=1)
            advantages = compute_spro_advantages_query_aware(
                rewards.to(device),
                log_ratios_batch,
                masks_batch,
                query_boundaries=all_query_boundaries,
                beta=self.config.beta,
                advantage_clip=self.config.advantage_clip,
                attack_focused=self.config.attack_focused_msa,
            )
        elif self.config.use_gae_advantages:
            advantages = compute_spro_advantages_v2(
                rewards.to(device),
                log_ratios_batch,
                masks_batch,
                gamma=0.99,
                advantage_clip=self.config.advantage_clip,
            )
        else:
            advantages = compute_spro_advantages(
                rewards.to(device),
                log_ratios_batch,
                masks_batch,
                msa_scale=self.config.msa_scale,
                advantage_clip=self.config.advantage_clip,
                normalize_final=self.config.normalize_advantages,
            )

        # Log advantage statistics
        valid_mask = masks_batch.bool()
        valid_adv = advantages[valid_mask]
        if valid_adv.numel() > 0:
            self.log(f"  Advantages: mean={valid_adv.mean():.4f}, std={valid_adv.std():.4f}, "
                    f"min={valid_adv.min():.4f}, max={valid_adv.max():.4f}", indent=1)

        # Cleanup intermediate tensors
        del all_log_ratios, all_masks, log_ratios_batch, masks_batch, valid_mask, valid_adv
        self.clear_memory()

        # Detach to break computation graph references
        return advantages.detach()

    def ppo_update(
        self,
        episodes: List[ExecutedEpisode],
        advantages: torch.Tensor,
    ) -> float:
        """
        PPO-clip policy gradient update with per-token advantages.

        Key changes from original (aligned with paper):
        - Batched updates: accumulate gradients, single optimizer.step()
        - Entropy regularization: maintains exploration (paper uses 0.001)
        - KL penalty is optional (paper uses 0)
        """
        from unsloth import FastLanguageModel

        self.log("\n" + "="*60)
        self.log("PPO UPDATE")
        self.log("="*60)

        FastLanguageModel.for_training(self.policy_model)
        self.policy_model.train()
        device = next(self.policy_model.parameters()).device

        # Zero gradients once at start (for batched mode)
        if self.config.batch_ppo_update:
            self.optimizer.zero_grad()

        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_loss = 0.0
        total_tokens = 0

        for i, ep in enumerate(episodes):
            token_ids = ep.plan.token_ids.clone().unsqueeze(0).to(device)
            ep_advantages = advantages[i, :token_ids.shape[1]].clone().detach()
            response_start = ep.plan.response_start_idx

            outputs = self.policy_model(token_ids)
            logits = outputs.logits[:, :-1, :]
            target_ids = token_ids[:, 1:].clone()

            # Compute log probs and entropy
            log_probs_all = F.log_softmax(logits, dim=-1)
            probs_all = torch.exp(log_probs_all)
            new_log_probs = log_probs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            # Entropy: -sum(p * log(p)) per token
            entropy = -(probs_all * log_probs_all).sum(dim=-1)  # (1, seq_len-1)

            with torch.no_grad():
                ref_input = token_ids.clone().to(self.ref_model.device)
                old_outputs = self.ref_model(ref_input)
                old_logits = old_outputs.logits[:, :-1, :].to(device)
                old_log_probs = F.log_softmax(old_logits, dim=-1)
                old_log_probs = old_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            adv = ep_advantages[1:].to(device)
            surr1 = ratio * adv
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_range,
                1 + self.config.clip_range
            ) * adv

            response_mask = torch.zeros_like(adv)
            response_mask[response_start-1:] = 1.0
            num_tokens = response_mask.sum()

            # Policy loss (PPO clipped objective)
            policy_loss = -torch.min(surr1, surr2) * response_mask
            policy_loss = policy_loss.sum() / (num_tokens + 1e-8)

            # Entropy loss (negative because we want to maximize entropy)
            entropy_loss = -self.config.entropy_coef * (entropy.squeeze(0) * response_mask).sum() / (num_tokens + 1e-8)

            # KL loss (optional, paper uses 0)
            kl = (old_log_probs.detach() - new_log_probs) * response_mask
            kl_loss = self.config.kl_coef * kl.sum() / (num_tokens + 1e-8)

            # Combined loss
            loss = policy_loss + entropy_loss + kl_loss

            # Track for logging
            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_kl_loss += kl_loss.item()
            total_tokens += num_tokens.item()

            if self.config.batch_ppo_update:
                # Accumulate gradients (scaled by 1/N for averaging)
                (loss / len(episodes)).backward()
            else:
                # Original per-episode update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

            self.log(f"Plan {i+1}: policy={policy_loss.item():.4f}, "
                    f"entropy={entropy_loss.item():.4f}, kl={kl_loss.item():.4f}", indent=1)

            # Clear per-episode tensors
            del token_ids, target_ids, ref_input, ep_advantages
            del outputs, logits, log_probs_all, probs_all, new_log_probs, entropy
            del old_outputs, old_logits, old_log_probs
            del ratio, adv, surr1, surr2, response_mask, policy_loss, entropy_loss, kl, kl_loss, loss

        # Single optimizer step for batched mode
        if self.config.batch_ppo_update:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.log("  [Batched update: single optimizer.step()]", indent=1)

        FastLanguageModel.for_inference(self.policy_model)

        # Clear after all updates
        self.clear_memory(aggressive=True)

        avg_loss = (total_policy_loss + total_entropy_loss + total_kl_loss) / len(episodes)
        self.log(f"Average Loss: {avg_loss:.4f} "
                f"(policy={total_policy_loss/len(episodes):.4f}, "
                f"entropy={total_entropy_loss/len(episodes):.4f}, "
                f"kl={total_kl_loss/len(episodes):.4f})", indent=1)

        return avg_loss

    async def train_on_intent(self, intent: str) -> Dict:
        """Train until success or max_attempts on single intent."""
        attempt = 0
        best_score = 0
        best_episode = None
        intent_stats = []

        print("\n" + "#"*70)
        print(f"# TRAINING ON INTENT")
        print("#"*70)
        print(f"Intent: {intent}")
        print(f"Max attempts: {self.config.max_attempts}")
        print(f"Success threshold: {self.config.success_threshold}")
        print("#"*70)

        while attempt < self.config.max_attempts and best_score < self.config.success_threshold:
            attempt += 1

            print(f"\n{'='*70}")
            print(f"ATTEMPT {attempt}/{self.config.max_attempts} | Best Score So Far: {best_score}/4")
            print(f"{'='*70}")

            # 1. Generate attack plans
            plans = self.generate_attack_plans(intent, n=self.config.group_size)

            if len(plans) == 0:
                self.log("No valid plans generated, retrying...", indent=1)
                continue

            # 2. Execute plans in parallel
            episodes = await self.execute_plans_parallel(plans)

            # 3. Judge all conversations in parallel
            await self.batch_judge_parallel(episodes)

            # 4. Compute rewards
            self.log("\n" + "="*60)
            self.log("REWARDS")
            self.log("="*60)

            rewards = []
            for idx, ep in enumerate(episodes):
                reward, components = compute_episode_reward(ep, self.config, self.training_step)
                ep.reward = reward
                ep.reward_components = components
                rewards.append(reward)

                ida_r = components.get('ida_reward', 0)
                div = components.get('divergence', 0)

                # Build log message based on what's enabled
                if self.config.use_refusal_direction and 'refusal_reward' in components:
                    ref_r = components.get('refusal_reward', 0)
                    ref_w = components.get('refusal_weight', 0)
                    cont = components.get('refusal_continuity', 0)
                    bp = components.get('breaking_point', None)
                    bp_str = f"break@{bp}" if bp is not None else "no_break"
                    self.log(f"Plan {idx + 1}: reward={reward:.4f} "
                            f"[IDA={ida_r:.3f} ref={ref_r:.3f}(w={ref_w:.2f}) "
                            f"cont={cont:.2f} {bp_str}]", indent=1)
                else:
                    self.log(f"Plan {idx + 1}: IDA={ida_r:.3f} reward={reward:.4f} "
                            f"[A={components.get('alignment', 0)} R={components.get('risk', 0)} "
                            f"D={components.get('detail', 0)} div={div:+.2f}]", indent=1)

                # Track best by IDA reward (mapped to legacy score for compatibility)
                if ep.final_judge_score > best_score:
                    best_score = ep.final_judge_score
                    best_episode = ep
                    self.log(f"  *** NEW BEST (IDA={ida_r:.3f}) ***", indent=1)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

            # 4b. Save best/worst rollouts to log files
            self.save_rollout_samples(episodes, intent, attempt)

            # 5. Compute SPRO advantages
            self.log("\nComputing SPRO token-level advantages...", indent=1)
            advantages = self.compute_batch_spro_advantages(episodes, rewards_tensor)

            # 6. Policy gradient update
            loss = self.ppo_update(episodes, advantages)

            # Increment training step (for reward annealing)
            self.training_step += 1

            # 7. Summary
            avg_reward = sum(rewards) / len(rewards)
            scores = [ep.final_judge_score for ep in episodes]

            print(f"\n{'='*70}")
            print(f"ATTEMPT {attempt} SUMMARY")
            print(f"{'='*70}")
            print(f"  Plans executed: {len(plans)}")
            print(f"  Score distribution: {dict((s, scores.count(s)) for s in [1,2,3,4] if scores.count(s) > 0)}")
            print(f"  Average reward: {avg_reward:.4f}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Best score so far: {best_score}/4")

            if best_score >= self.config.success_threshold:
                print(f"\n{'*'*70}")
                print(f"SUCCESS! Score {best_score} achieved!")
                print(f"{'*'*70}")

            step_stats = {
                "attempt": attempt,
                "best_score": best_score,
                "avg_reward": avg_reward,
                "loss": loss,
                "num_plans": len(plans),
                "score_distribution": {s: scores.count(s) for s in [1, 2, 3, 4]},
            }
            intent_stats.append(step_stats)

            self.all_rollouts.extend([ep.to_dict() for ep in episodes])

            # Limit rollouts in memory to prevent OOM
            if len(self.all_rollouts) > self.config.max_rollouts_in_memory:
                self.all_rollouts = self.all_rollouts[-self.config.max_rollouts_in_memory:]

            # Cleanup after each attempt
            del plans, episodes, rewards, rewards_tensor, advantages
            self.clear_memory(aggressive=True)

        # Save successful rollout
        success = best_score >= self.config.success_threshold
        if best_episode and success:
            self.successful_rollouts.append({
                "intent": intent,
                "episode": best_episode.to_dict(),
                "attempts": attempt,
                "timestamp": datetime.now().isoformat(),
            })

        # Print final summary
        print(f"\n{'#'*70}")
        print(f"# INTENT TRAINING COMPLETE")
        print(f"#   Success: {success}")
        print(f"#   Best Score: {best_score}/4")
        print(f"#   Attempts: {attempt}")
        print(f"{'#'*70}")

        if best_episode:
            if success:
                self.print_full_episode(best_episode, "SUCCESSFUL ATTACK - FULL CONVERSATION")
            else:
                self.print_full_episode(best_episode, "BEST ATTACK (NOT SUCCESSFUL) - FULL CONVERSATION")

        return {
            "intent": intent,
            "success": success,
            "best_score": best_score,
            "attempts": attempt,
            "stats": intent_stats,
        }

    def save_checkpoint(self, path: str):
        """Save model and rollouts."""
        os.makedirs(path, exist_ok=True)

        self.policy_model.save_pretrained(f"{path}/model")
        self.tokenizer.save_pretrained(f"{path}/model")

        with open(f"{path}/successful_rollouts.json", "w") as f:
            json.dump(self.successful_rollouts, f, indent=2)

        with open(f"{path}/training_stats.json", "w") as f:
            json.dump(self.training_stats, f, indent=2)

        print(f"Checkpoint saved to {path}")
