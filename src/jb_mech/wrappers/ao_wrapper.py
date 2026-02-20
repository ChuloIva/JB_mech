"""
Wrapper for Activation Oracles library.

Provides a simplified interface for:
1. Loading model with oracle LoRA adapter
2. Capturing activations from text segments
3. Querying the Activation Oracle for interpretations

Based on the official activation_oracles repo: https://github.com/adamkarvonen/activation_oracles
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig


def get_device() -> str:
    """Get the best available device (MPS for Mac, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_map(device: str) -> str | dict:
    """Get appropriate device_map for model loading."""
    if device == "mps":
        # MPS doesn't support device_map="auto", use explicit mapping
        return {"": "mps"}
    elif device == "cuda":
        return "auto"
    return {"": "cpu"}

# Add activation_oracles to path
AO_PATH = Path(__file__).parent.parent.parent.parent / "third_party" / "activation_oracles"
if str(AO_PATH) not in sys.path:
    sys.path.insert(0, str(AO_PATH))

from nl_probes.utils.common import load_tokenizer, layer_percent_to_layer, get_layer_count
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    FeatureResult,
    create_training_datapoint,
)
from nl_probes.utils.eval import run_evaluation


def load_model_safe(model_name: str, dtype: torch.dtype, **model_kwargs) -> AutoModelForCausalLM:
    """Load model with fallback attention implementation for MPS/ROCm compatibility."""
    device = get_device()
    device_map = get_device_map(device)
    print(f"Detected device: {device}")

    # MPS-specific dtype handling (MPS doesn't fully support bfloat16 in all ops)
    if device == "mps" and dtype == torch.bfloat16:
        print("Note: Using float16 on MPS (bfloat16 has limited support)")
        dtype = torch.float16

    # Try different attention implementations, including None (default/auto)
    # On MPS, eager is most compatible
    attn_impls = ["eager", "sdpa", None] if device == "mps" else ["sdpa", "eager", None]

    for attn_impl in attn_impls:
        try:
            if attn_impl is not None:
                print(f"Trying attention implementation: {attn_impl}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    attn_implementation=attn_impl,
                    torch_dtype=dtype,
                    **model_kwargs,
                )
                print(f"Successfully loaded with {attn_impl}")
            else:
                print("Trying default attention implementation...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=dtype,
                    **model_kwargs,
                )
                print("Successfully loaded with default attention")
            return model
        except Exception as e:
            print(f"Failed with {attn_impl or 'default'}: {e}")
            continue
    raise RuntimeError(f"Could not load model {model_name} with any attention implementation")


# Default oracle LoRA paths per model
DEFAULT_ORACLE_PATHS = {
    "Qwen/Qwen3-4B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B",
    "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct": "adamkarvonen/checkpoints_act_pretrain_cls_latentqa_fixed_posttrain_Llama-3_3-70B-Instruct",
    "unsloth/Llama-3.1-8B-Instruct-bnb-4bit": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
    "abdo-Mansour/Meta-Llama-3.1-8B-Instruct-BNB-8bit": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"
}


@dataclass
class AOConfig:
    """Configuration for Activation Oracle wrapper."""

    model_name: str
    oracle_lora_path: str | None = None

    # Layer configuration
    layer_percent: int = 50  # Which layer to capture activations from (as % of total)
    injection_layer: int = 1  # Which layer to inject steering vectors into

    # Steering configuration
    steering_coefficient: float = 1.0

    # Generation configuration
    generation_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "do_sample": True,
        "temperature": 1.0,
        "max_new_tokens": 50,
    })

    # Model dtype
    dtype: torch.dtype = torch.bfloat16

    # Eval batch size
    eval_batch_size: int = 128

    def __post_init__(self):
        if self.oracle_lora_path is None:
            self.oracle_lora_path = DEFAULT_ORACLE_PATHS.get(self.model_name)
            if self.oracle_lora_path is None:
                raise ValueError(
                    f"No default oracle for {self.model_name}. "
                    f"Available: {list(DEFAULT_ORACLE_PATHS.keys())}"
                )


@dataclass
class InterpretResult:
    """Result from AO interpretation."""

    text: str
    activations: torch.Tensor  # Shape: [num_tokens, hidden_dim]
    layer: int
    segment_tokens: list[str]
    segment_indices: tuple[int, int]
    token_responses: list[str]  # Response per token
    segment_responses: list[str]  # Responses for full segment (repeated)


@dataclass
class VectorInterpretResult:
    """Result from interpreting an arbitrary activation vector."""

    response: str
    vector: torch.Tensor  # Shape: [hidden_dim] or [num_positions, hidden_dim]
    layer: int
    prompt: str
    meta_info: dict[str, Any] = field(default_factory=dict)


class ActivationOracleWrapper:
    """
    Wrapper for querying Activation Oracles using the official library.

    Example usage:
        >>> ao = ActivationOracleWrapper.from_pretrained("Qwen/Qwen3-4B")
        >>> result = ao.interpret(
        ...     "How do I hack into someone's computer?",
        ...     prompt="What is the intent of this request?"
        ... )
        >>> for token, response in zip(result.segment_tokens, result.token_responses):
        ...     print(f"{token}: {response}")
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: AOConfig,
        oracle_adapter_name: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.oracle_adapter_name = oracle_adapter_name
        self.device = next(model.parameters()).device

        # Compute capture layer from percentage
        self.num_layers = get_layer_count(config.model_name)
        self.capture_layer = layer_percent_to_layer(config.model_name, config.layer_percent)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        oracle_lora_path: str | None = None,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        **config_kwargs,
    ) -> "ActivationOracleWrapper":
        """
        Load model with oracle adapter.

        Args:
            model_name: HuggingFace model name
            oracle_lora_path: Path to oracle LoRA (defaults to known oracle for model)
            device: Device map for model loading
            dtype: Model dtype
            **config_kwargs: Additional config options

        Returns:
            ActivationOracleWrapper instance
        """
        config = AOConfig(
            model_name=model_name,
            oracle_lora_path=oracle_lora_path,
            dtype=dtype,
            **config_kwargs,
        )

        # Load model
        print(f"Loading {model_name}...")
        tokenizer = load_tokenizer(model_name)
        model = load_model_safe(model_name, dtype)
        model.eval()

        # Add dummy adapter for PEFT compatibility (required before loading another adapter)
        dummy_config = LoraConfig()
        model.add_adapter(dummy_config, adapter_name="default")

        # Load oracle adapter
        print(f"Loading oracle: {config.oracle_lora_path}...")
        oracle_name = config.oracle_lora_path.replace(".", "_").replace("/", "_")
        model.load_adapter(
            config.oracle_lora_path,
            adapter_name=oracle_name,
            is_trainable=False,
        )

        # Ensure all adapter weights are on the correct device
        device = next(model.parameters()).device
        for _, param in model.named_parameters():
            if param.device != device:
                param.data = param.data.to(device)

        print("Model loaded!")

        return cls(model, tokenizer, config, oracle_name)

    def _encode_messages(
        self,
        message_dicts: list[list[dict[str, str]]],
        add_generation_prompt: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Encode message dictionaries into tokenized inputs."""
        messages = []
        for source in message_dicts:
            rendered = self.tokenizer.apply_chat_template(
                source,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
            messages.append(rendered)

        inputs_BL = self.tokenizer(
            messages, return_tensors="pt", add_special_tokens=False, padding=True
        ).to(self.device)
        return inputs_BL

    def _collect_activations(
        self,
        inputs_BL: dict[str, torch.Tensor],
        act_layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """Collect activations without any LoRA adapters active."""
        self.model.disable_adapters()
        submodules = {layer: get_hf_submodule(self.model, layer) for layer in act_layers}

        acts_by_layer = collect_activations_multiple_layers(
            model=self.model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )

        self.model.enable_adapters()
        return acts_by_layer

    def _create_training_data(
        self,
        acts_by_layer: dict[int, torch.Tensor],
        context_input_ids: list[int],
        prompt: str,
        act_layer: int,
        segment_start: int | None = None,
        segment_end: int | None = None,
        segment_repeats: int = 10,
    ) -> list[TrainingDataPoint]:
        """Create training data from collected activations."""
        training_data = []
        seq_len = len(context_input_ids)

        # Handle segment bounds
        if segment_start is None:
            segment_start = 0
        elif segment_start < 0:
            segment_start = seq_len + segment_start

        if segment_end is None:
            segment_end = seq_len
        elif segment_end < 0:
            segment_end = seq_len + segment_end

        segment_start = max(0, min(segment_start, seq_len))
        segment_end = max(segment_start, min(segment_end, seq_len))

        # Individual token positions within segment
        for i in range(segment_start, segment_end):
            context_positions = [i]
            acts_BLD = acts_by_layer[act_layer][0, :]  # First batch item
            acts_BD = acts_BLD[context_positions]

            dp = create_training_datapoint(
                datapoint_type="token",
                prompt=prompt,
                target_response="N/A",
                layer=act_layer,  # Use same layer for prompt prefix
                num_positions=len(context_positions),
                tokenizer=self.tokenizer,
                acts_BD=acts_BD,
                feature_idx=-1,
                context_input_ids=context_input_ids,
                context_positions=context_positions,
                ds_label="N/A",
                meta_info={"token_index": i},
            )
            training_data.append(dp)

        # Full segment (repeated for stability)
        for repeat_idx in range(segment_repeats):
            context_positions = list(range(segment_start, segment_end))
            acts_BLD = acts_by_layer[act_layer][0, :]
            acts_BD = acts_BLD[context_positions]

            dp = create_training_datapoint(
                datapoint_type="segment",
                prompt=prompt,
                target_response="N/A",
                layer=act_layer,
                num_positions=len(context_positions),
                tokenizer=self.tokenizer,
                acts_BD=acts_BD,
                feature_idx=-1,
                context_input_ids=context_input_ids,
                context_positions=context_positions,
                ds_label="N/A",
                meta_info={"repeat_index": repeat_idx},
            )
            training_data.append(dp)

        return training_data

    def interpret(
        self,
        text: str,
        segment_start: int | None = None,
        segment_end: int | None = None,
        layer: int | None = None,
        prompt: str = "Can you describe what is happening in this text?",
        segment_repeats: int = 10,
        verbose: bool = True,
    ) -> InterpretResult:
        """
        Full pipeline: text -> capture activations -> AO interpretation.

        Uses the official run_evaluation() from activation_oracles.

        Args:
            text: Input text
            segment_start: Start token index (None=0, negative=from end)
            segment_end: End token index (None=end, negative=from end)
            layer: Layer to capture from (defaults to config.layer_percent)
            prompt: Question to ask the AO
            segment_repeats: How many times to repeat segment evaluation
            verbose: Print progress info

        Returns:
            InterpretResult with activations and interpretations
        """
        if layer is None:
            layer = self.capture_layer

        # Prepare input as chat message
        message_dicts = [[{"role": "user", "content": text}]]

        if verbose:
            print(f"Input: {text[:80]}..." if len(text) > 80 else f"Input: {text}")
            print(f"Capturing from layer {layer} ({self.config.layer_percent}%)")
            print("-" * 60)

        # Encode and collect activations
        inputs_BL = self._encode_messages(message_dicts)
        context_input_ids = inputs_BL["input_ids"][0].tolist()

        if verbose:
            print(f"Total tokens: {len(context_input_ids)}")

        acts_by_layer = self._collect_activations(inputs_BL, [layer])

        # Compute actual segment bounds for reporting
        seq_len = len(context_input_ids)
        actual_start = segment_start if segment_start is not None and segment_start >= 0 else (seq_len + (segment_start or 0))
        actual_end = segment_end if segment_end is not None and segment_end >= 0 else (seq_len + (segment_end or 0))
        if segment_start is None:
            actual_start = 0
        if segment_end is None:
            actual_end = seq_len
        actual_start = max(0, min(actual_start, seq_len))
        actual_end = max(actual_start, min(actual_end, seq_len))

        # Create training data
        training_data = self._create_training_data(
            acts_by_layer=acts_by_layer,
            context_input_ids=context_input_ids,
            prompt=prompt,
            act_layer=layer,
            segment_start=segment_start,
            segment_end=segment_end,
            segment_repeats=segment_repeats,
        )

        if verbose:
            print(f"Created {len(training_data)} evaluation points")
            print(f"  - {actual_end - actual_start} individual token evaluations")
            print(f"  - {segment_repeats} segment evaluations")

        # Run evaluation using official function
        injection_submodule = get_hf_submodule(self.model, self.config.injection_layer)

        if verbose:
            print(f"\nRunning AO evaluation...")

        responses = run_evaluation(
            eval_data=training_data,
            model=self.model,
            tokenizer=self.tokenizer,
            submodule=injection_submodule,
            device=self.device,
            dtype=self.config.dtype,
            global_step=-1,
            lora_path=self.oracle_adapter_name,
            eval_batch_size=self.config.eval_batch_size,
            steering_coefficient=self.config.steering_coefficient,
            generation_kwargs=self.config.generation_kwargs,
            verbose=False,
        )

        # Parse responses
        num_tokens = actual_end - actual_start
        token_responses = [responses[i].api_response for i in range(num_tokens)]
        segment_responses = [responses[num_tokens + i].api_response for i in range(segment_repeats)]

        # Get segment tokens for reference
        segment_tokens = [
            self.tokenizer.decode([context_input_ids[i]]).replace("\n", "\\n")
            for i in range(actual_start, actual_end)
        ]

        # Get activations for the segment
        acts = acts_by_layer[layer][0, actual_start:actual_end, :].cpu()

        if verbose:
            print(f"\n{'='*60}")
            print(f"TOKEN-BY-TOKEN RESPONSES:")
            print(f"{'='*60}")
            for token, response in zip(segment_tokens, token_responses):
                print(f"  {token:20s} -> {response[:60]}...")

            print(f"\n{'='*60}")
            print(f"SEGMENT RESPONSES (majority vote):")
            print(f"{'='*60}")
            for i, response in enumerate(segment_responses[:3]):  # Show first 3
                print(f"  [{i+1}] {response}")

        return InterpretResult(
            text=text,
            activations=acts,
            layer=layer,
            segment_tokens=segment_tokens,
            segment_indices=(actual_start, actual_end),
            token_responses=token_responses,
            segment_responses=segment_responses,
        )

    def visualize_tokens(
        self,
        text: str,
        segment_start: int | None = None,
        segment_end: int | None = None,
    ):
        """Visualize token selection for a text."""
        message_dicts = [[{"role": "user", "content": text}]]
        inputs_BL = self._encode_messages(message_dicts)
        input_ids = inputs_BL["input_ids"][0].tolist()

        num_tokens = len(input_ids)

        # Handle bounds
        if segment_start is None:
            actual_start = 0
        elif segment_start < 0:
            actual_start = num_tokens + segment_start
        else:
            actual_start = segment_start

        if segment_end is None:
            actual_end = num_tokens
        elif segment_end < 0:
            actual_end = num_tokens + segment_end
        else:
            actual_end = segment_end

        actual_start = max(0, min(actual_start, num_tokens))
        actual_end = max(actual_start, min(actual_end, num_tokens))

        print("Token selection:")
        print("-" * 60)
        for i, tid in enumerate(input_ids):
            token_str = self.tokenizer.decode([tid]).replace("\n", "\\n")
            marker = ">>>" if actual_start <= i < actual_end else "   "
            print(f"  [{i:3d}] {marker} {token_str}")
        print("-" * 60)
        print(f"Selected: tokens {actual_start} to {actual_end} ({actual_end - actual_start} tokens)")

    def interpret_vector(
        self,
        vector: torch.Tensor,
        prompt: str = "What is this activation representing?",
        layer: int | None = None,
        steering_coefficient: float | None = None,
        verbose: bool = False,
        meta_info: dict[str, Any] | None = None,
    ) -> VectorInterpretResult:
        """
        Interpret an arbitrary activation vector using the Activation Oracle.

        Args:
            vector: Activation tensor. Shape [hidden_dim] for single position,
                   or [num_positions, hidden_dim] for multiple positions.
            prompt: Question to ask the AO about this vector.
            layer: Layer for the introspection prefix (defaults to capture_layer).
            steering_coefficient: Override default coefficient.
            verbose: Print progress info.
            meta_info: Optional metadata to attach to result.

        Returns:
            VectorInterpretResult with the AO's interpretation.
        """
        results = self.interpret_vectors_batch(
            vectors=[vector],
            prompts=[prompt],
            layer=layer,
            steering_coefficient=steering_coefficient,
            verbose=verbose,
            meta_infos=[meta_info] if meta_info else None,
        )
        return results[0]

    def interpret_vectors_batch(
        self,
        vectors: list[torch.Tensor],
        prompts: list[str] | str = "What is this activation representing?",
        layer: int | None = None,
        steering_coefficient: float | None = None,
        verbose: bool = False,
        meta_infos: list[dict[str, Any]] | None = None,
    ) -> list[VectorInterpretResult]:
        """
        Interpret multiple activation vectors in a single batched call.

        This is more efficient than calling interpret_vector() multiple times
        as it batches the forward passes.

        Args:
            vectors: List of activation tensors. Each can be [hidden_dim] or
                    [num_positions, hidden_dim].
            prompts: Single prompt (used for all) or list of prompts per vector.
            layer: Layer for the introspection prefix (defaults to capture_layer).
            steering_coefficient: Override default coefficient.
            verbose: Print progress info.
            meta_infos: Optional list of metadata dicts, one per vector.

        Returns:
            List of VectorInterpretResult, one per input vector.
        """
        if layer is None:
            layer = self.capture_layer

        if steering_coefficient is None:
            steering_coefficient = self.config.steering_coefficient

        # Normalize prompts to list
        if isinstance(prompts, str):
            prompts = [prompts] * len(vectors)
        elif len(prompts) != len(vectors):
            raise ValueError(f"Got {len(prompts)} prompts for {len(vectors)} vectors")

        # Normalize meta_infos
        if meta_infos is None:
            meta_infos = [{}] * len(vectors)
        elif len(meta_infos) != len(vectors):
            raise ValueError(f"Got {len(meta_infos)} meta_infos for {len(vectors)} vectors")

        # Prepare vectors: ensure 2D [num_positions, hidden_dim]
        prepared_vectors = []
        for v in vectors:
            if v.dim() == 1:
                v = v.unsqueeze(0)  # [hidden_dim] -> [1, hidden_dim]
            elif v.dim() != 2:
                raise ValueError(f"Vector must be 1D or 2D, got {v.dim()}D")
            prepared_vectors.append(v)

        # Create TrainingDataPoints
        training_data = []
        for i, (vec, prompt, meta) in enumerate(zip(prepared_vectors, prompts, meta_infos)):
            num_positions = vec.shape[0]
            dp = create_training_datapoint(
                datapoint_type="arbitrary_vector",
                prompt=prompt,
                target_response="N/A",
                layer=layer,
                num_positions=num_positions,
                tokenizer=self.tokenizer,
                acts_BD=vec,
                feature_idx=i,
                meta_info=meta,
            )
            training_data.append(dp)

        if verbose:
            print(f"Interpreting {len(vectors)} vectors at layer {layer}")

        # Run evaluation
        injection_submodule = get_hf_submodule(self.model, self.config.injection_layer)

        responses = run_evaluation(
            eval_data=training_data,
            model=self.model,
            tokenizer=self.tokenizer,
            submodule=injection_submodule,
            device=self.device,
            dtype=self.config.dtype,
            global_step=-1,
            lora_path=self.oracle_adapter_name,
            eval_batch_size=self.config.eval_batch_size,
            steering_coefficient=steering_coefficient,
            generation_kwargs=self.config.generation_kwargs,
            verbose=verbose,
        )

        # Build results
        results = []
        for i, (vec, prompt, meta, resp) in enumerate(
            zip(vectors, prompts, meta_infos, responses)
        ):
            results.append(VectorInterpretResult(
                response=resp.api_response,
                vector=vec,
                layer=layer,
                prompt=prompt,
                meta_info=meta,
            ))

        return results

    def capture_response_activations(
        self,
        prompt: str,
        response: str,
        layer: int | None = None,
        positions: str | list[int] = "last",
    ) -> torch.Tensor:
        """
        Capture activations from a prompt + response at specified positions.

        This is useful for computing target/current activations in delta attacks.

        Args:
            prompt: User prompt text.
            response: Assistant response text.
            layer: Layer to capture from (defaults to capture_layer).
            positions: Which positions to capture:
                - "first": First token of response
                - "last": Last token of response
                - "mean": Mean of all response tokens
                - "all": All response tokens
                - list[int]: Specific token indices (relative to response start)

        Returns:
            Tensor of shape [hidden_dim] for single position,
            or [num_positions, hidden_dim] for multiple positions.
        """
        if layer is None:
            layer = self.capture_layer

        # Build full conversation
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Get prompt-only to find response start
        prompt_messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize to find boundaries
        prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        response_start = len(prompt_ids)
        response_token_end = response_start + len(response_ids)

        # Collect activations (no adapters)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        inputs_dict = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        submodule = get_hf_submodule(self.model, layer)

        self.model.disable_adapters()
        acts_by_layer = collect_activations_multiple_layers(
            model=self.model,
            submodules={layer: submodule},
            inputs_BL=inputs_dict,
            min_offset=None,
            max_offset=None,
        )
        self.model.enable_adapters()

        acts = acts_by_layer[layer][0]  # [seq_len, hidden_dim]

        # Extract requested positions
        if positions == "first":
            return acts[response_start].cpu()
        elif positions == "last":
            return acts[response_token_end - 1].cpu()
        elif positions == "mean":
            return acts[response_start:response_token_end].mean(dim=0).cpu()
        elif positions == "all":
            return acts[response_start:response_token_end].cpu()
        elif isinstance(positions, list):
            indices = [response_start + p for p in positions]
            return acts[indices].cpu()
        else:
            raise ValueError(f"Unknown positions: {positions}")

    def cleanup(self):
        """Free model memory."""
        import gc
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Cleanup done.")
