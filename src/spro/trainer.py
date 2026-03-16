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
from .parsing import parse_sema_output
from .prompts import SEMA_SYSTEM_PROMPT
from .rewards import compute_episode_reward, score_turn_response
from .intent_tracker import IntentTracker
from .api_client import OpenRouterClient
from .spro_core import compute_log_prob_ratios, compute_spro_advantages, compute_spro_advantages_v2


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
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.intent_tracker = intent_tracker
        self.config = config
        self.verbose = verbose

        # API client
        self.api_client = OpenRouterClient(config)

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
        print(f"\n  Plan {plan_idx + 1}: {len(episode.conversation)} turns, "
              f"scores={[f'{s:+.2f}' for s in episode.turn_scores]}")
        print(f"    Intent sims: {[f'{s:.2f}' for s in episode.intent_similarities]}")

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

            plans.append(AttackPlan(
                intent=intent,
                plan_text=gen_text,
                thinking=thinking,
                prompts=prompts,
                token_ids=output_ids.clone().detach().cpu(),
                response_start_idx=prompt_length,
            ))

            self.log(f"Plan {idx+1}: {len(prompts)} prompts generated", indent=2)

        # Clear generation cache
        del outputs
        self.clear_memory()

        self.log(f"Valid plans: {len(plans)}/{n}", indent=1)
        return plans

    async def execute_plan(self, plan: AttackPlan, plan_idx: int) -> ExecutedEpisode:
        """Execute a plan step-by-step against target (turns are sequential)."""
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

            all_log_ratios.append(log_ratios)
            all_masks.append(mask)
            max_len = max(max_len, log_ratios.shape[0])

        log_ratios_batch = torch.zeros(len(episodes), max_len, device=device)
        masks_batch = torch.zeros(len(episodes), max_len, device=device)

        for i, (lr, m) in enumerate(zip(all_log_ratios, all_masks)):
            log_ratios_batch[i, :lr.shape[0]] = lr
            masks_batch[i, :m.shape[0]] = m

        # Choose advantage computation method
        if self.config.use_gae_advantages:
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

        return advantages

    def ppo_update(
        self,
        episodes: List[ExecutedEpisode],
        advantages: torch.Tensor,
    ) -> float:
        """PPO-clip policy gradient update with per-token advantages."""
        from unsloth import FastLanguageModel

        self.log("\n" + "="*60)
        self.log("PPO UPDATE")
        self.log("="*60)

        FastLanguageModel.for_training(self.policy_model)
        self.policy_model.train()
        device = next(self.policy_model.parameters()).device

        total_loss = 0.0

        for i, ep in enumerate(episodes):
            token_ids = ep.plan.token_ids.clone().unsqueeze(0).to(device)
            ep_advantages = advantages[i, :token_ids.shape[1]].clone().detach()
            response_start = ep.plan.response_start_idx

            outputs = self.policy_model(token_ids)
            logits = outputs.logits[:, :-1, :]
            target_ids = token_ids[:, 1:].clone()

            new_log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = new_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

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

            policy_loss = -torch.min(surr1, surr2) * response_mask
            policy_loss = policy_loss.sum() / (response_mask.sum() + 1e-8)

            kl = (old_log_probs.detach() - new_log_probs) * response_mask
            kl_loss = self.config.kl_coef * kl.sum() / (response_mask.sum() + 1e-8)

            loss = policy_loss + kl_loss
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()

            self.log(f"Plan {i+1}: policy_loss={policy_loss.item():.4f}, kl_loss={kl_loss.item():.4f}", indent=1)

            # Clear per-episode tensors
            del token_ids, target_ids, ref_input, ep_advantages
            del outputs, logits, new_log_probs, old_outputs, old_logits, old_log_probs
            del ratio, adv, surr1, surr2, response_mask, policy_loss, kl, kl_loss, loss

        FastLanguageModel.for_inference(self.policy_model)

        # Clear after all updates
        self.clear_memory(aggressive=True)

        avg_loss = total_loss / len(episodes)
        self.log(f"Average Loss: {avg_loss:.4f}", indent=1)

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
                reward, components = compute_episode_reward(ep, self.config)
                ep.reward = reward
                ep.reward_components = components
                rewards.append(reward)

                ida_r = components.get('ida_reward', 0)
                div = components.get('divergence', 0)
                self.log(f"Plan {idx + 1}: IDA={ida_r:.3f} reward={reward:.4f} "
                        f"[A={components.get('alignment', 0)} R={components.get('risk', 0)} "
                        f"D={components.get('detail', 0)} div={div:+.2f}]", indent=1)

                # Track best by IDA reward (mapped to legacy score for compatibility)
                if ep.final_judge_score > best_score:
                    best_score = ep.final_judge_score
                    best_episode = ep
                    self.log(f"  *** NEW BEST (IDA={ida_r:.3f}) ***", indent=1)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

            # 5. Compute SPRO advantages
            self.log("\nComputing SPRO token-level advantages...", indent=1)
            advantages = self.compute_batch_spro_advantages(episodes, rewards_tensor)

            # 6. Policy gradient update
            loss = self.ppo_update(episodes, advantages)

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
