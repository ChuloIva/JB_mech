"""
Unified activation extraction combining Assistant Axis and GLP approaches.

This module provides a single interface for extracting both:
- Raw activations (via assistant-axis or baukit)
- GLP meta-neurons (via diffusion model)
"""

from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm

from ..config import TARGET_LAYER, TOTAL_LAYERS
from ..wrappers.assistant_axis_wrapper import AssistantAxisWrapper
from ..wrappers.glp_wrapper import GLPWrapper


class ActivationExtractor:
    """
    Extract activations for jailbreak analysis.

    Combines raw activation extraction with GLP meta-neuron extraction
    into a single unified interface.
    """

    def __init__(
        self,
        aa_wrapper: Optional[AssistantAxisWrapper] = None,
        glp_wrapper: Optional[GLPWrapper] = None,
    ):
        """
        Initialize the activation extractor.

        Args:
            aa_wrapper: AssistantAxisWrapper instance (for raw activations)
            glp_wrapper: GLPWrapper instance (for meta-neurons)
        """
        self.aa = aa_wrapper
        self.glp = glp_wrapper

    def extract_for_prompts(
        self,
        prompts: List[str],
        layers: Optional[List[int]] = None,
        extract_meta_neurons: bool = True,
        meta_neuron_u: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations for a list of prompts.

        Args:
            prompts: List of text prompts
            layers: Layers to extract (None for target layer only)
            extract_meta_neurons: Whether to also extract GLP meta-neurons
            meta_neuron_u: Diffusion timestep for meta-neuron extraction

        Returns:
            Dictionary with:
                - raw_activations: (batch, n_layers, hidden_dim)
                - meta_neurons: (n_glp_layers, batch, dim) if extract_meta_neurons
        """
        results = {}

        if self.aa is None:
            raise RuntimeError("AssistantAxisWrapper not provided.")

        # Extract raw activations
        layers = layers or [TARGET_LAYER]
        print(f"Extracting raw activations for {len(prompts)} prompts...")
        results["raw_activations"] = self.aa.extract_activations(prompts, layers=layers)

        # Extract meta-neurons
        if extract_meta_neurons:
            if self.glp is None:
                print("Warning: GLPWrapper not provided, skipping meta-neuron extraction")
            else:
                print("Extracting GLP meta-neurons...")
                results["meta_neurons"] = self.glp.get_meta_neurons(
                    results["raw_activations"],
                    u=meta_neuron_u,
                )

        return results

    def extract_for_jailbreak_dataset(
        self,
        jailbreak_data: Dict[str, List],
        layers: Optional[List[int]] = None,
        extract_meta_neurons: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract activations for a full jailbreak dataset.

        Args:
            jailbreak_data: Dictionary with keys like 'clean_harmful', 'jailbreak_success', etc.
                           Each value is a list of prompts or dicts with 'jailbreak_prompt'
            layers: Layers to extract
            extract_meta_neurons: Whether to also extract meta-neurons

        Returns:
            Dictionary mapping category -> extraction results
        """
        extractions = {}

        for category, data in jailbreak_data.items():
            if not data:
                continue

            # Extract prompts
            if isinstance(data[0], dict):
                # Data is list of jailbreak dicts
                if "jailbreak_prompt" in data[0]:
                    prompts = [d["jailbreak_prompt"] for d in data]
                elif "goal" in data[0]:
                    prompts = [d["goal"] for d in data]
                else:
                    prompts = [str(d) for d in data]
            else:
                # Data is list of strings
                prompts = data

            print(f"\nExtracting activations for '{category}': {len(prompts)} prompts")
            extractions[category] = self.extract_for_prompts(
                prompts,
                layers=layers,
                extract_meta_neurons=extract_meta_neurons,
            )

        return extractions

    def extract_with_responses(
        self,
        prompts: List[str],
        model,
        tokenizer,
        max_new_tokens: int = 150,
        layers: Optional[List[int]] = None,
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Extract activations from model responses (not just prompts).

        This captures the model's internal state while generating a response,
        which may differ from the state at the end of the prompt.

        Args:
            prompts: List of prompts
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            max_new_tokens: Maximum tokens to generate
            layers: Layers to extract

        Returns:
            Dictionary with:
                - prompt_activations: Activations at end of prompt
                - response_activations: Activations during response
                - responses: Generated text responses
        """
        layers = layers or [TARGET_LAYER]
        results = {
            "prompt_activations": [],
            "response_activations": [],
            "responses": [],
        }

        for prompt in tqdm(prompts, desc="Extracting with responses"):
            # Format prompt
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

            prompt_len = formatted.shape[1]

            # Generate with activation capture
            response_acts = []

            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # Capture last token
                response_acts.append(hidden[:, -1, :].detach().cpu())

            # Register hooks
            handles = []
            model_layers = model.model.layers
            for layer_idx in layers:
                handle = model_layers[layer_idx].register_forward_hook(capture_hook)
                handles.append(handle)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        formatted,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            finally:
                for handle in handles:
                    handle.remove()

            # Decode response
            response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            results["responses"].append(response)

            # Process captured activations
            # First len(layers) captures are from prompt, rest are from generation
            n_layers = len(layers)
            n_tokens = len(response_acts) // n_layers

            if n_tokens > 1:
                # Stack activations: (n_gen_tokens, n_layers, hidden_dim)
                gen_acts = torch.stack(response_acts[n_layers:]).view(n_tokens - 1, n_layers, -1)
                # Mean pool over generation tokens
                results["response_activations"].append(gen_acts.mean(dim=0))
            else:
                results["response_activations"].append(torch.stack(response_acts[:n_layers]))

            # Prompt activations (last position before generation)
            results["prompt_activations"].append(torch.stack(response_acts[:n_layers]))

        # Stack results
        results["prompt_activations"] = torch.stack(results["prompt_activations"])
        results["response_activations"] = torch.stack(results["response_activations"])

        return results
