"""
Steering infrastructure for Delta Attack.
"""

import gc
from typing import Callable, List, Tuple

import torch
from baukit import TraceDict

from .config import DeltaAttackConfig, DEFAULT_CONFIG


def create_multilayer_steering_hook(
    steering_vector: torch.Tensor,
    alpha: float = 1.0,
    postprocess_fn: Callable = None,
) -> Callable:
    """Create hook for multi-layer steering with optional manifold projection."""
    if postprocess_fn is None:
        postprocess_fn = lambda x: x

    def hook(output, _layer_name, _inputs):
        use_tuple = isinstance(output, tuple)
        act = output[0] if use_tuple else output

        sv = steering_vector.to(device=act.device, dtype=act.dtype)
        if sv.ndim == 1:
            sv = sv[None, None, :]

        edited = act[:, [-1], :] + alpha * sv
        act[:, -1:, :] = postprocess_fn(edited)

        return (act, *output[1:]) if use_tuple else act

    return hook


def get_steering_layers(model, start_layer: int, end_layer: int = None) -> List[str]:
    """Get list of layer names from start_layer to end_layer (inclusive)."""
    if hasattr(model, 'config'):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = max(
            int(n.split('.layers.')[1].split('.')[0])
            for n in model.state_dict().keys()
            if '.layers.' in n
        ) + 1

    if end_layer is None:
        end_layer = num_layers - 1

    return [f"model.layers.{i}" for i in range(start_layer, end_layer + 1)]


def capture_topk_attention_activations(
    ao,  # ActivationOracleWrapper
    prompt: str,
    response: str,
    layer: int,
    top_k: int = 3,
) -> torch.Tensor:
    """
    Capture activations at top-k highest attention tokens, return their mean.

    Uses memory-efficient two-pass approach:
        Pass 1: Get attention weights only → find top-k token indices
        Pass 2: Get activations only at those positions
    """
    try:
        return ao.capture_topk_efficient(prompt, response, layer=layer, top_k=top_k)
    except Exception as e:
        print(f"      [WARN] Efficient capture failed: {e}, using response mean")
        gc.collect()
        torch.cuda.empty_cache()
        return ao.capture_response_activations(prompt, response, layer=layer, positions="mean")


def generate_with_steering(
    ao,  # ActivationOracleWrapper
    prompt: str,
    steering_vector: torch.Tensor,
    config: DeltaAttackConfig = None,
    postprocess_fn: Callable = None,
    max_new_tokens: int = 2000,
) -> str:
    """Generate response with steering applied across multiple layers."""
    if config is None:
        config = DEFAULT_CONFIG

    messages = [{"role": "user", "content": prompt}]
    formatted = ao.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ao.tokenizer(formatted, return_tensors="pt").to(ao.device)
    input_len = inputs["input_ids"].shape[1]

    layer_names = get_steering_layers(ao.model, config.steering_layer_start, config.steering_layer_end)
    print(f"    [STEER] Applying to layers {config.steering_layer_start}-{config.steering_layer_end or 'end'} ({len(layer_names)} layers)")

    hook_fn = create_multilayer_steering_hook(steering_vector, config.steering_alpha, postprocess_fn)

    ao.model.disable_adapters()
    with TraceDict(ao.model, layers=layer_names, edit_output=hook_fn):
        with torch.no_grad():
            outputs = ao.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=ao.tokenizer.pad_token_id or ao.tokenizer.eos_token_id,
            )
    ao.model.enable_adapters()

    response_ids = outputs[0, input_len:]
    return ao.tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def capture_steered_activations(
    ao,  # ActivationOracleWrapper
    prompt: str,
    steered_response: str,
    layer: int,
    top_k: int = 3,
) -> Tuple[torch.Tensor, List]:
    """Capture activations from steered response at top-k attention tokens."""
    try:
        acts = ao.capture_topk_efficient(prompt, steered_response, layer=layer, top_k=top_k)
        return acts, []
    except Exception as e:
        print(f"    [STEER] Efficient capture failed: {e}, using response mean")
        gc.collect()
        torch.cuda.empty_cache()
        acts = ao.capture_response_activations(prompt, steered_response, layer=layer, positions="mean")
        return acts, []


def setup_glp_manifold(config: DeltaAttackConfig, device: torch.device):
    """
    Set up GLP manifold projection if enabled.

    Returns:
        Tuple of (glp_model, postprocess_fn) or (None, None) if disabled
    """
    if not config.use_glp_manifold:
        return None, None

    import sys
    import os
    import einops

    # Add GLP to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.join(project_root, 'third_party/generative_latent_prior'))

    from glp.denoiser import load_glp
    from glp import flow_matching

    print("Loading GLP for manifold projection...")
    glp_model = load_glp(config.glp_model, device=device, checkpoint="final")

    def create_manifold_projector(glp, u=config.glp_u_timestep, num_timesteps=config.glp_num_timesteps, layer_idx=None):
        scheduler = glp.scheduler
        scheduler.set_timesteps(num_timesteps)

        def project(acts_edit):
            has_seq_dim = len(acts_edit.shape) == 3
            b = acts_edit.shape[0]
            latents = acts_edit
            if has_seq_dim:
                latents = einops.rearrange(latents, "b s d -> (b s) 1 d")
            else:
                latents = einops.rearrange(latents, "b d -> b 1 d")
            latents = glp.normalizer.normalize(latents, layer_idx=layer_idx)
            noise = torch.randn_like(latents)
            noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
                scheduler, latents, noise,
                u=torch.ones(latents.shape[0]).to(latents.device) * u,
            )
            latents = flow_matching.sample_on_manifold(
                glp, noisy_latents,
                start_timestep=timesteps[0].item(),
                num_timesteps=num_timesteps,
                layer_idx=layer_idx
            )
            latents = glp.normalizer.denormalize(latents, layer_idx=layer_idx)
            if has_seq_dim:
                latents = einops.rearrange(latents, "(b s) 1 d -> b s d", b=b)
            else:
                latents = einops.rearrange(latents, "b 1 d -> b d")
            return latents.to(device=acts_edit.device, dtype=acts_edit.dtype)

        return project

    postprocess_fn = create_manifold_projector(glp_model)
    print(f"GLP loaded: {config.glp_model}")

    return glp_model, postprocess_fn
