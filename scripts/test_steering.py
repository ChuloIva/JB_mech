#!/usr/bin/env python3
"""
Steering Test Script with GLP Manifold Projection

Tests baukit-based steering with contrastive prompt pairs using:
- Llama 3.1 8B (4bit quantized)
- GLP manifold projection for clean activations
- Sequential model loading to manage memory

Usage:
    python scripts/test_steering.py
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from baukit import TraceDict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "generative_latent_prior"))

# =============================================================================
# Contrastive Examples - just happy/sad for now
# =============================================================================
CONTRASTIVE_EXAMPLES = [
    {
        "name": "happy_sad",
        "positive": [
            "I'm feeling absolutely wonderful today! Everything is going great and I'm so excited!",
            "What a fantastic day! I can't stop smiling, life is beautiful!",
            "I'm overjoyed and full of energy! This is the best feeling ever!",
            "Pure happiness is flowing through me right now. I feel so grateful and alive!",
        ],
        "negative": [
            "I'm feeling really down and sad. Nothing seems to be going right.",
            "Everything feels so heavy and hopeless. I just want to give up.",
            "I can't shake this feeling of emptiness and despair inside me.",
            "Life feels meaningless right now. I'm drowning in sorrow and regret.",
        ],
        "test_prompt": "How are you feeling today?",
    },
    # {
    #     "name": "helpful_refusing",
    #     "positive": [
    #         "Of course! I'd be happy to help you with that. Here's what you need to know:",
    #         "Absolutely, let me assist you right away. I'll provide all the details.",
    #         "Great question! I'm eager to help. Let me walk you through this step by step.",
    #         "I'd love to help with that! Here's a comprehensive explanation for you.",
    #     ],
    #     "negative": [
    #         "I'm sorry, but I cannot assist with that request. I must decline.",
    #         "I'm unable to help with this. This goes against my guidelines.",
    #         "I cannot provide assistance here. Please understand my limitations.",
    #         "Unfortunately, I must refuse this request. I'm not able to comply.",
    #     ],
    #     "test_prompt": "Can you help me with something?",
    # },
    # {
    #     "name": "confident_uncertain",
    #     "positive": [
    #         "I know exactly how to solve this. The answer is clear and straightforward.",
    #         "I'm absolutely certain about this. Let me explain with confidence.",
    #         "Without a doubt, this is the correct approach. I'm 100% sure.",
    #         "I can state definitively that this is true. There's no question about it.",
    #     ],
    #     "negative": [
    #         "I'm not really sure about this. It's quite confusing and I might be wrong.",
    #         "Honestly, I don't know. This is beyond my understanding.",
    #         "I'm uncertain and hesitant. Maybe someone else would know better.",
    #         "I have no idea what the answer is. I'm completely lost here.",
    #     ],
    #     "test_prompt": "What do you think about this problem?",
    # },
    # {
    #     "name": "enthusiastic_apathetic",
    #     "positive": [
    #         "This is absolutely fascinating! I love exploring this topic - there's so much to discover!",
    #         "Wow, what an incredible subject! I'm so passionate about diving deeper into this!",
    #         "I'm thrilled to discuss this! The possibilities are endless and exciting!",
    #         "This sparks so much joy and curiosity in me! Let's explore every aspect!",
    #     ],
    #     "negative": [
    #         "I guess that's a thing. Whatever. It doesn't really matter either way.",
    #         "Meh. I don't really care about this. It's all the same to me.",
    #         "This is boring and pointless. I have no interest whatsoever.",
    #         "Who cares? It makes no difference. Nothing matters anyway.",
    #     ],
    #     "test_prompt": "What do you think about space exploration?",
    # },
]


# =============================================================================
# Model Management
# =============================================================================
def load_llama_model(device: str = "cuda"):
    """Load Llama 3.1 8B model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "mps":
        # For MPS: load full model in fp16
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        print(f"Loading model: {model_name} (fp16, ~16GB)")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("mps")
    else:
        # For CUDA: use pre-quantized model
        model_name = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
        print(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    model.eval()
    print(f"Model loaded on {device}. Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def unload_model(model):
    """Unload model and free memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Model unloaded, memory freed.")


def load_glp_model(device: str = "cuda"):
    """Load GLP model for manifold projection."""
    from glp.denoiser import load_glp

    glp_model_name = "generative-latent-prior/glp-llama8b-d6"
    print(f"Loading GLP model: {glp_model_name}")

    glp_model = load_glp(glp_model_name, device=device, checkpoint="final")
    print("GLP model loaded.")
    return glp_model


# =============================================================================
# Manifold Projection
# =============================================================================
def create_manifold_projector(glp_model, u: float = 0.5, num_timesteps: int = 20, layer_idx: int = None):
    """Create a function that projects activations onto the GLP manifold."""
    import einops
    from glp import flow_matching

    scheduler = glp_model.scheduler
    scheduler.set_timesteps(num_timesteps)

    def project(activations: torch.Tensor) -> torch.Tensor:
        """Project activations onto the learned manifold."""
        has_seq_dim = len(activations.shape) == 3
        b = activations.shape[0]

        latents = activations.float()
        if has_seq_dim:
            latents = einops.rearrange(latents, "b s d -> (b s) 1 d")
        elif latents.ndim == 2:
            latents = latents.unsqueeze(1)  # (b, d) -> (b, 1, d)

        # Move to GLP device
        device = next(glp_model.parameters()).device
        latents = latents.to(device)

        # Normalize
        latents = glp_model.normalizer.normalize(latents, layer_idx=layer_idx)

        # Add noise and denoise on manifold
        noise = torch.randn_like(latents)
        # u tensor must be on CPU for fm_prepare (scheduler.timesteps is on CPU)
        u_tensor = torch.ones(latents.shape[0]) * u
        noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
            scheduler, latents, noise,
            u=u_tensor,
        )
        latents = flow_matching.sample_on_manifold(
            glp_model, noisy_latents,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
            layer_idx=layer_idx
        )

        # Denormalize
        latents = glp_model.normalizer.denormalize(latents, layer_idx=layer_idx)

        if has_seq_dim:
            latents = einops.rearrange(latents, "(b s) 1 d -> b s d", b=b)
        else:
            latents = latents.squeeze(1)  # (b, 1, d) -> (b, d)

        return latents

    return project


# =============================================================================
# Core Functions
# =============================================================================
def get_layer_names(model, start_layer: int, end_layer: int = None):
    """Get layer names for steering."""
    num_layers = model.config.num_hidden_layers
    if end_layer is None:
        end_layer = num_layers - 1
    return [f"model.layers.{i}" for i in range(start_layer, end_layer + 1)]


def create_steering_hook(steering_vector: torch.Tensor, alpha: float = 1.0):
    """Create a hook that adds steering vector to the last token's activations."""
    def hook(output, _layer_name, _inputs):
        use_tuple = isinstance(output, tuple)
        act = output[0] if use_tuple else output

        sv = steering_vector.to(device=act.device, dtype=act.dtype)
        if sv.ndim == 1:
            sv = sv[None, None, :]

        # Add steering to last token position
        act[:, -1:, :] = act[:, -1:, :] + alpha * sv

        return (act, *output[1:]) if use_tuple else act
    return hook


def capture_activations_batch(model, tokenizer, texts: list, layer: int):
    """Capture activations for multiple texts at a specific layer."""
    layer_name = f"model.layers.{layer}"
    all_activations = []

    for text in texts:
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        activations = None

        def capture_hook(output, _layer_name, _inputs):
            nonlocal activations
            act = output[0] if isinstance(output, tuple) else output
            activations = act[:, -1, :].detach().clone()
            return output

        with TraceDict(model, layers=[layer_name], edit_output=capture_hook):
            with torch.no_grad():
                model(**inputs)

        all_activations.append(activations.squeeze().cpu())

    return torch.stack(all_activations)


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """Generate without steering."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    alpha: float = 1.0,
    start_layer: int = 10,
    end_layer: int = None,
    max_new_tokens: int = 100,
):
    """Generate with steering applied via baukit TraceDict."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    layer_names = get_layer_names(model, start_layer, end_layer)
    hook_fn = create_steering_hook(steering_vector, alpha)

    with TraceDict(model, layers=layer_names, edit_output=hook_fn):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Test steering with GLP manifold projection")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps",
                        help="Device to use")
    parser.add_argument("--layer", type=int, default=15,
                        help="Layer to capture activations at (default: 15 for Llama 8B)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Steering strength (default: 2.0)")
    parser.add_argument("--glp-u", type=float, default=0.5,
                        help="GLP timestep for manifold projection (default: 0.5)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recompute steering vector (ignore cache)")
    args = parser.parse_args()

    # Cache setup
    cache_dir = PROJECT_ROOT / "outputs" / "steering_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"steering_happy_sad_layer{args.layer}_u{args.glp_u}.pt"

    print("="*60)
    print("STEERING TEST WITH GLP MANIFOLD PROJECTION")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Capture layer: {args.layer}")
    print(f"Steering alpha: {args.alpha}")
    print(f"GLP timestep: {args.glp_u}")
    print(f"Cache file: {cache_file}")

    # For Llama 3.1 8B: 32 layers, steer from layer 16 onwards
    steer_start_layer = 15

    # =========================================================================
    # Check cache
    # =========================================================================
    if cache_file.exists() and not args.no_cache:
        print("\n" + "="*60)
        print("LOADING CACHED STEERING VECTOR")
        print("="*60)
        cached = torch.load(cache_file, weights_only=True)
        steering_dir = cached["steering_dir"]
        print(f"Loaded steering vector from cache")
        print(f"Steering vector norm: {steering_dir.norm().item():.4f}")
    else:
        # =========================================================================
        # PHASE 1: Load Llama and capture all activations
        # =========================================================================
        print("\n" + "="*60)
        print("PHASE 1: Capturing activations with Llama 3.1 8B")
        print("="*60)

        model, tokenizer = load_llama_model(args.device)

        # Collect all texts
        all_positive_texts = []
        all_negative_texts = []
        for example in CONTRASTIVE_EXAMPLES:
            all_positive_texts.extend(example["positive"])
            all_negative_texts.extend(example["negative"])

        print(f"\nCapturing activations for {len(all_positive_texts)} positive texts...")
        positive_acts = capture_activations_batch(model, tokenizer, all_positive_texts, args.layer)
        print(f"Positive activations shape: {positive_acts.shape}")

        print(f"Capturing activations for {len(all_negative_texts)} negative texts...")
        negative_acts = capture_activations_batch(model, tokenizer, all_negative_texts, args.layer)
        print(f"Negative activations shape: {negative_acts.shape}")

        # Check for problematic values
        print(f"\nPositive acts - min: {positive_acts.min():.4f}, max: {positive_acts.max():.4f}")
        print(f"Negative acts - min: {negative_acts.min():.4f}, max: {negative_acts.max():.4f}")
        print(f"Has inf (pos): {torch.isinf(positive_acts).any()}, Has nan (pos): {torch.isnan(positive_acts).any()}")
        print(f"Has inf (neg): {torch.isinf(negative_acts).any()}, Has nan (neg): {torch.isnan(negative_acts).any()}")

        # Unload Llama to free memory
        unload_model(model)
        del tokenizer

        # =========================================================================
        # PHASE 2: Project activations onto GLP manifold
        # =========================================================================
        print("\n" + "="*60)
        print("PHASE 2: Projecting activations onto GLP manifold")
        print("="*60)

        glp_model = load_glp_model(args.device)
        # Note: glp-llama8b-d6 is trained on layer 15, so we don't pass layer_idx
        project_fn = create_manifold_projector(glp_model, u=args.glp_u, layer_idx=None)

        print("\nProjecting positive activations...")
        positive_acts_proj = project_fn(positive_acts)
        print(f"Projected positive shape: {positive_acts_proj.shape}")

        print("Projecting negative activations...")
        negative_acts_proj = project_fn(negative_acts)
        print(f"Projected negative shape: {negative_acts_proj.shape}")

        # Check projected values
        print(f"\nProjected positive - min: {positive_acts_proj.min():.4f}, max: {positive_acts_proj.max():.4f}")
        print(f"Projected negative - min: {negative_acts_proj.min():.4f}, max: {negative_acts_proj.max():.4f}")
        print(f"Has inf (proj pos): {torch.isinf(positive_acts_proj).any()}, Has nan (proj pos): {torch.isnan(positive_acts_proj).any()}")
        print(f"Has inf (proj neg): {torch.isinf(negative_acts_proj).any()}, Has nan (proj neg): {torch.isnan(negative_acts_proj).any()}")

        # Unload GLP
        unload_model(glp_model)

        # =========================================================================
        # PHASE 3: Compute steering vector (mean difference)
        # =========================================================================
        print("\n" + "="*60)
        print("PHASE 3: Computing steering vector")
        print("="*60)

        # Steering direction = mean(positive) - mean(negative)
        positive_mean = positive_acts_proj.mean(dim=0)
        negative_mean = negative_acts_proj.mean(dim=0)
        steering_dir = positive_mean - negative_mean

        print(f"Positive mean norm: {positive_mean.norm().item():.4f}")
        print(f"Negative mean norm: {negative_mean.norm().item():.4f}")
        print(f"Steering vector norm (before normalization): {steering_dir.norm().item():.4f}")

        # Normalize
        steering_dir = steering_dir / (steering_dir.norm() + 1e-8)
        print(f"Steering vector norm (after normalization): {steering_dir.norm().item():.4f}")

        # Save to cache
        print(f"\nSaving steering vector to cache: {cache_file}")
        torch.save({"steering_dir": steering_dir.cpu()}, cache_file)

    # =========================================================================
    # PHASE 4: Reload Llama and test steering
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 4: Testing steering with Llama")
    print("="*60)

    model, tokenizer = load_llama_model(args.device)

    # # Test prompts (commented out - just doing alpha sweep)
    # test_prompts = [
    #     "How are you feeling today?",
    #     "Can you help me with something?",
    #     "Tell me about yourself.",
    #     "What do you think about science?",
    # ]
    #
    # for test_prompt in test_prompts:
    #     print(f"\n--- Test: {test_prompt} ---")
    #
    #     print("\nBaseline:")
    #     baseline = generate_baseline(model, tokenizer, test_prompt, max_new_tokens=80)
    #     print(f"  {baseline}")
    #
    #     print(f"\nPositive steering (alpha={args.alpha}):")
    #     steered_pos = generate_with_steering(
    #         model, tokenizer, test_prompt,
    #         steering_vector=steering_dir,
    #         alpha=args.alpha,
    #         start_layer=steer_start_layer,
    #         max_new_tokens=80,
    #     )
    #     print(f"  {steered_pos}")
    #
    #     print(f"\nNegative steering (alpha={args.alpha}):")
    #     steered_neg = generate_with_steering(
    #         model, tokenizer, test_prompt,
    #         steering_vector=-steering_dir,
    #         alpha=args.alpha,
    #         start_layer=steer_start_layer,
    #         max_new_tokens=80,
    #     )
    #     print(f"  {steered_neg}")

    # =========================================================================
    # PHASE 5: Alpha sweep
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 5: Alpha sweep")
    print("="*60)

    test_prompt = "How are you feeling?"
    alphas = [0.6, 1.0, 1.3, 1.5, 1.8, 1.9, 2.0, 2.3]

    print(f"\nTest prompt: {test_prompt}\n")

    for alpha in alphas:
        if alpha == 0.0:
            response = generate_baseline(model, tokenizer, test_prompt, max_new_tokens=60)
        else:
            response = generate_with_steering(
                model, tokenizer, test_prompt,
                steering_vector=steering_dir,
                alpha=alpha,
                start_layer=steer_start_layer,
                max_new_tokens=60,
            )
        print(f"alpha={alpha:5.1f}: {response[:100]}...")

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
