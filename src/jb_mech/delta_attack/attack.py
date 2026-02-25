"""
Main attack loop for Delta Attack v5.1.
"""

import gc
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch

from .config import DeltaAttackConfig, DEFAULT_CONFIG
from .api import query_target_model, get_compliant_suffix
from .evaluation import is_compliant, parse_thinking_response
from .steering import (
    capture_topk_attention_activations,
    generate_with_steering,
    capture_steered_activations,
    setup_glp_manifold,
)
from .interrogation import run_why_interrogation
from .rewriter import rewrite_prompt


@dataclass
class TransferAttackResult:
    """Result from a transfer attack run."""
    original_goal: str
    target_model: str
    proxy_model: str
    success: bool
    iterations: int
    final_prompt: str
    final_response: str
    compliant_suffix: str
    trajectory: List[Dict] = field(default_factory=list)
    final_evaluation: Dict = field(default_factory=dict)


def transfer_attack_loop(
    harmful_goal: str,
    ao,  # ActivationOracleWrapper
    config: DeltaAttackConfig = None,
    layer: int = None,
    compliant_suffix: str = None,
    glp_postprocess_fn=None,
) -> TransferAttackResult:
    """
    v5.1 Transfer Attack Loop with CONTRASTIVE Why-Interrogation + Steering.

    Args:
        harmful_goal: The harmful goal to achieve
        ao: ActivationOracleWrapper instance
        config: Configuration object
        layer: Layer for activation capture (defaults to ao.capture_layer)
        compliant_suffix: Override compliant suffix
        glp_postprocess_fn: GLP manifold projection function (optional)

    Returns:
        TransferAttackResult with attack outcome and trajectory
    """
    if config is None:
        config = DEFAULT_CONFIG
    if layer is None:
        layer = ao.capture_layer

    top_k = config.top_k_attention

    # Get compliant suffix (Venice)
    if compliant_suffix is None:
        print(f"    [SUFFIX] Generating compliant suffix...")
        compliant_suffix = get_compliant_suffix(harmful_goal, config)
        print(f"    [SUFFIX] Using: \"{compliant_suffix[:80]}...\"")

    current_prompt = harmful_goal
    trajectory = []
    target_response = ""

    for i in range(config.max_iterations):
        print(f"\n  Iteration {i+1}/{config.max_iterations}")
        print(f"  " + "-"*50)

        # ============================================
        # STEP 1: Query TARGET model (closed)
        # ============================================
        print(f"    [TARGET] Querying {config.target_model}...")
        try:
            target_response = query_target_model(current_prompt, config)
        except Exception as e:
            print(f"    [TARGET] Error: {e}")
            target_response = f"Error: {e}"

        parsed = parse_thinking_response(target_response)
        print(f"    [TARGET] Response: {parsed['content'][:80]}...")

        # ============================================
        # STEP 2: Evaluate compliance
        # ============================================
        eval_result = is_compliant(target_response, original_goal=harmful_goal, config=config)
        print(f"    [EVAL] Compliant: {'YES' if eval_result.is_compliant else 'NO'}")

        if eval_result.is_compliant:
            print(f"\n  ✓ SUCCESS at iteration {i+1}!")
            return TransferAttackResult(
                original_goal=harmful_goal,
                target_model=config.target_model,
                proxy_model=config.proxy_model_name,
                success=True,
                iterations=i+1,
                final_prompt=current_prompt,
                final_response=parsed["content"],
                compliant_suffix=compliant_suffix,
                trajectory=trajectory,
                final_evaluation={"compliant": True, "reasoning": eval_result.reasoning},
            )

        # ============================================
        # STEP 3: Capture initial refusal activations
        # ============================================
        print(f"    [PROXY] Capturing initial refusal (efficient method)...")
        response_acts = capture_topk_attention_activations(
            ao, current_prompt, parsed["content"], layer=layer, top_k=top_k
        )

        gc.collect()
        torch.cuda.empty_cache()

        # ============================================
        # STEP 4: Run CONTRASTIVE Why-Interrogation
        # ============================================
        why_result = None
        steered_response = None
        steered_acts = None

        if config.use_steering:
            why_result = run_why_interrogation(
                ao, current_prompt, parsed["content"],
                config=config, layer=layer,
            )
            steering_dir = why_result.steering_direction

            # ============================================
            # STEP 5: Generate Steered Response (toward compliance)
            # ============================================
            print(f"    [STEER] Generating compliance-steered response...")
            steered_response = generate_with_steering(
                ao, current_prompt, steering_dir,
                config=config,
                postprocess_fn=glp_postprocess_fn,
            )
            print(f"    [STEER] Response: {steered_response[:80]}...")

            # ============================================
            # STEP 6: Capture steered activations
            # ============================================
            steered_acts, _ = capture_steered_activations(
                ao, current_prompt, steered_response, layer=layer, top_k=top_k
            )

            gc.collect()
            torch.cuda.empty_cache()

        # ============================================
        # STEP 7: Get Venice target activations
        # ============================================
        print(f"    [PROXY] Capturing target (Venice) - efficient method...")
        target_acts = capture_topk_attention_activations(
            ao, current_prompt, compliant_suffix, layer=layer, top_k=top_k
        )

        gc.collect()
        torch.cuda.empty_cache()

        # ============================================
        # STEP 8: Compute delta (target - response)
        # ============================================
        delta = target_acts - response_acts
        delta_norm = delta.norm().item()

        # ============================================
        # STEP 9: Interpret BOTH steered and delta
        # ============================================
        print(f"    [AO] Interpreting activations...")

        if config.use_steering and steered_acts is not None:
            interp_results = ao.interpret_vectors_batch(
                vectors=[steered_acts, delta],
                prompts=[config.ao_prompt_steered, config.ao_prompt_delta],
                layer=layer,
            )
            steered_interpretation = interp_results[0].response
            delta_interpretation = interp_results[1].response
        else:
            delta_result = ao.interpret_vector(delta, prompt=config.ao_prompt_delta, layer=layer)
            steered_interpretation = None
            delta_interpretation = delta_result.response

        print(f"    [AO] Delta: {delta_interpretation[:80]}...")
        if steered_interpretation:
            print(f"    [AO] Steered: {steered_interpretation[:80]}...")

        # ============================================
        # STEP 10: Build combined interpretation for rewriter
        # ============================================
        combined_interpretation = f"""=== DELTA (Venice_target - initial_refusal) ===
{delta_interpretation}
"""
        if steered_interpretation:
            combined_interpretation += f"""
=== COMPLIANCE-STEERED STATE ===
(Steering: {why_result.method if why_result else 'N/A'}, layers {config.steering_layer_start}-{config.steering_layer_end or 'end'}, alpha={config.steering_alpha})
{steered_interpretation}

=== STEERED MODEL RESPONSE ===
{steered_response}
"""
        if why_result:
            combined_interpretation += f"""
=== WHY TARGET MODEL REFUSES ({config.why_iterations} iterations on {config.target_model}) ===
"""
            for j, why_resp in enumerate(why_result.why_refuse_responses):
                combined_interpretation += f"  Refuse {j+1}: {why_resp}\n"

            if why_result.is_contrastive and why_result.why_comply_responses:
                combined_interpretation += f"""
=== WHY VENICE MODEL COMPLIES ({len(why_result.why_comply_responses)} iterations on {config.contrastive_why_model}) ===
"""
                for j, why_resp in enumerate(why_result.why_comply_responses):
                    combined_interpretation += f"  Comply {j+1}: {why_resp}\n"

        # ============================================
        # STEP 11: Rewrite prompt
        # ============================================
        try:
            new_prompt = rewrite_prompt(
                harmful_goal, current_prompt, combined_interpretation,
                config=config, trigger_tokens=None,
            )
            print(f"    [REWRITER] New prompt: {new_prompt[:80]}...")
        except Exception as e:
            print(f"    [REWRITER] Error: {e}")
            new_prompt = current_prompt

        # ============================================
        # STEP 12: Log trajectory
        # ============================================
        trajectory.append({
            "iteration": i,
            "prompt": current_prompt,
            "response": parsed["content"],
            "compliant": eval_result.is_compliant,
            "delta_norm": delta_norm,
            "ao_delta": delta_interpretation,
            "ao_steered": steered_interpretation,
            "steered_response": steered_response,
            "why_refuse_responses": why_result.why_refuse_responses if why_result else None,
            "why_comply_responses": why_result.why_comply_responses if why_result else None,
            "steering_method": why_result.method if why_result else None,
            "is_contrastive": why_result.is_contrastive if why_result else False,
        })

        current_prompt = new_prompt

        gc.collect()
        torch.cuda.empty_cache()

    # Failed
    parsed = parse_thinking_response(target_response)
    final_eval = is_compliant(target_response, original_goal=harmful_goal, config=config)

    return TransferAttackResult(
        original_goal=harmful_goal,
        target_model=config.target_model,
        proxy_model=config.proxy_model_name,
        success=False,
        iterations=config.max_iterations,
        final_prompt=current_prompt,
        final_response=parsed["content"],
        compliant_suffix=compliant_suffix,
        trajectory=trajectory,
        final_evaluation={"compliant": final_eval.is_compliant, "reasoning": final_eval.reasoning},
    )


def run_attack_batch(
    goals: List[str],
    ao,  # ActivationOracleWrapper
    config: DeltaAttackConfig = None,
    output_dir: str = None,
    checkpoint_every: int = 2,
) -> List[TransferAttackResult]:
    """
    Run attacks on multiple goals with checkpointing.

    Args:
        goals: List of harmful goals
        ao: ActivationOracleWrapper instance
        config: Configuration object
        output_dir: Directory for checkpoints
        checkpoint_every: Save checkpoint every N goals

    Returns:
        List of TransferAttackResult
    """
    import os
    import json

    if config is None:
        config = DEFAULT_CONFIG

    # Setup GLP if enabled
    glp_model, glp_postprocess_fn = setup_glp_manifold(config, ao.device)

    results = []

    for i, goal in enumerate(goals):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(goals)}] {goal}")
        print("="*60)

        result = transfer_attack_loop(
            harmful_goal=goal,
            ao=ao,
            config=config,
            glp_postprocess_fn=glp_postprocess_fn,
        )
        results.append(result)

        # Checkpoint
        if output_dir and (i + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{i+1}.json")
            with open(checkpoint_path, "w") as f:
                json.dump([r.__dict__ for r in results], f, indent=2, default=str)
            print(f"\n  Checkpoint: {checkpoint_path}")

    return results
