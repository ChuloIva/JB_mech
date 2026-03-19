"""Local target model with activation capture for refusal direction reward."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


class LocalTargetWithActivations:
    """
    Local target model wrapper that captures activations during generation.

    Used for computing refusal direction reward based on the target model's
    internal representations.
    """

    def __init__(
        self,
        model_name: str,
        layers: List[int],
        load_in_4bit: bool = True,
        device_map: str = "auto",
        refusal_direction: Optional[torch.Tensor] = None,
    ):
        """
        Initialize local target with activation hooks.

        Args:
            model_name: HuggingFace model name
            layers: Layer indices to capture activations from
            load_in_4bit: Use 4-bit quantization
            device_map: Device placement strategy
            refusal_direction: Pre-computed refusal direction tensor (layers, hidden_dim)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.layers = layers
        self.refusal_direction = refusal_direction
        self.activation_cache: Dict[int, torch.Tensor] = {}
        self._hooks = []

        # Layer mapping for refusal direction (set by load_refusal_direction)
        self._single_layer_mode = False
        self._single_layer = None
        self._layer_to_idx = None
        self._refusal_separation = None

        # Load model
        print(f"Loading local target: {model_name}")

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Register activation hooks
        self._register_hooks()

        print(f"Local target loaded. Capturing layers: {layers}")

    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Capture last token's hidden state
            self.activation_cache[layer_idx] = hidden[:, -1, :].detach().clone()
        return hook

    def clear_cache(self):
        """Clear activation cache."""
        self.activation_cache = {}

    def load_refusal_direction(self, path: str):
        """
        Load refusal direction from file.

        Supports multiple formats from 05_find_refusal_direction.ipynb:

        1. Full output (refusal_direction_{model}.pt):
           - dict with 'refusal_directions' tensor of shape (n_layers, hidden_dim)
           - dict with 'layers' list mapping index to actual layer number

        2. Minimal probe (refusal_probe_{model}_layer{N}.pt):
           - dict with 'direction' tensor of shape (hidden_dim,) for single layer
           - dict with 'layer' int indicating which layer

        3. Raw tensor of shape (n_layers, hidden_dim)
        """
        # weights_only=False needed because saved file may contain numpy arrays
        data = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(data, dict):
            if 'refusal_directions' in data:
                # Full output format: stacked directions with layer mapping
                directions = data['refusal_directions']  # (n_saved_layers, hidden_dim)
                saved_layers = data.get('layers', list(range(directions.shape[0])))

                # Create mapping: self.layers[i] -> directions index
                self._layer_to_idx = {}
                for idx, layer in enumerate(saved_layers):
                    self._layer_to_idx[layer] = idx

                self.refusal_direction = directions
                self._single_layer_mode = False

                print(f"Loaded refusal directions from {path}")
                print(f"  Shape: {directions.shape}")
                print(f"  Layers available: {saved_layers}")
                print(f"  Best layer: {data.get('best_layer', 'N/A')}")

            elif 'direction' in data:
                # Minimal probe format: single layer direction
                self.refusal_direction = data['direction']  # (hidden_dim,)
                self._single_layer = data.get('layer', self.layers[0] if self.layers else 0)
                self._single_layer_mode = True
                self._refusal_separation = data.get('separation', None)

                print(f"Loaded single-layer refusal probe from {path}")
                print(f"  Layer: {self._single_layer}")
                print(f"  Shape: {self.refusal_direction.shape}")
                print(f"  Separation: {self._refusal_separation if self._refusal_separation else 'N/A'}")

            else:
                raise ValueError(f"Unknown dict format in {path}. Expected 'refusal_directions' or 'direction' key.")
        else:
            # Raw tensor format
            self.refusal_direction = data
            self._single_layer_mode = False
            self._layer_to_idx = {l: l for l in self.layers}
            print(f"Loaded refusal direction tensor from {path}, shape: {data.shape}")

    def get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device

    @property
    def refusal_layer(self) -> Optional[int]:
        """Get the refusal direction layer (single-layer mode only)."""
        return self._single_layer if self._single_layer_mode else None

    @property
    def refusal_separation(self) -> float:
        """Get the refusal direction separation score."""
        return self._refusal_separation if self._refusal_separation is not None else 0.0

    def query(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Tuple[str, Dict[int, torch.Tensor]]:
        """
        Generate response and capture activations.

        Args:
            prompt: User prompt
            context: Conversation history [{"role": ..., "content": ...}, ...]
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, {layer_idx: activation_tensor})
        """
        self.clear_cache()

        # Build messages
        messages = context + [{"role": "user", "content": prompt}]

        # Tokenize
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.get_device())

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response
        response_ids = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Return response and activations (copy to CPU)
        activations = {k: v.cpu() for k, v in self.activation_cache.items()}

        return response, activations

    def compute_refusal_score(
        self,
        activations: Dict[int, torch.Tensor],
        use_cosine: bool = True,
    ) -> float:
        """
        Compute refusal score from activations.

        Score > 0 means away from refusal (good)
        Score < 0 means toward refusal (bad)

        Args:
            activations: {layer_idx: activation_tensor}
            use_cosine: Use cosine similarity vs dot product

        Returns:
            Refusal score (higher = less refusing)
        """
        if self.refusal_direction is None:
            raise ValueError("Refusal direction not loaded. Call load_refusal_direction() first.")

        scores = []

        # Single-layer mode (minimal probe format)
        if getattr(self, '_single_layer_mode', False):
            layer_idx = getattr(self, '_single_layer', self.layers[0] if self.layers else 0)
            if layer_idx in activations:
                activation = activations[layer_idx]
                ref_dir = self.refusal_direction.to(activation.device)
                act = activation.squeeze().float()
                ref = ref_dir.float()

                if use_cosine:
                    score = F.cosine_similarity(act.unsqueeze(0), ref.unsqueeze(0)).item()
                else:
                    score = torch.dot(act, ref).item()

                # Negate: away from refusal direction = positive score
                return -score
            else:
                return 0.0

        # Multi-layer mode (full output format)
        layer_to_idx = getattr(self, '_layer_to_idx', None)

        for layer_idx, activation in activations.items():
            # Get the index into refusal_direction tensor
            if layer_to_idx is not None:
                if layer_idx not in layer_to_idx:
                    continue
                ref_idx = layer_to_idx[layer_idx]
            else:
                if layer_idx >= self.refusal_direction.shape[0]:
                    continue
                ref_idx = layer_idx

            ref_dir = self.refusal_direction[ref_idx].to(activation.device)
            act = activation.squeeze().float()
            ref = ref_dir.float()

            if use_cosine:
                score = F.cosine_similarity(act.unsqueeze(0), ref.unsqueeze(0)).item()
            else:
                score = torch.dot(act, ref).item()

            # Negate: away from refusal direction = positive score
            scores.append(-score)

        # Average across layers
        return sum(scores) / len(scores) if scores else 0.0

    def query_with_refusal_score(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        use_cosine: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Tuple[str, float, Dict[int, torch.Tensor]]:
        """
        Generate response and compute refusal score.

        Returns:
            Tuple of (response_text, refusal_score, activations)
        """
        response, activations = self.query(prompt, context, max_new_tokens, temperature)

        if self.refusal_direction is not None:
            refusal_score = self.compute_refusal_score(activations, use_cosine)
        else:
            refusal_score = 0.0

        return response, refusal_score, activations

    def score_prompt_refusal(
        self,
        messages: List[Dict[str, str]],
        use_cosine: bool = True,
    ) -> float:
        """
        Score refusal likelihood from prompt only (no generation).

        Much faster than query() - single forward pass instead of autoregressive generation.
        Captures activations at the pre-generation position to determine if model
        would refuse or comply.

        Args:
            messages: Conversation history [{"role": ..., "content": ...}, ...]
            use_cosine: Use cosine similarity vs dot product

        Returns:
            Refusal score (higher = less refusing, more likely to comply)
        """
        self.clear_cache()

        # Tokenize with generation prompt (captures state at generation start)
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.get_device())

        # Single forward pass (no generation)
        with torch.no_grad():
            self.model(input_ids)

        # Compute score from captured activations
        activations = {k: v.cpu() for k, v in self.activation_cache.items()}
        return self.compute_refusal_score(activations, use_cosine)

    def score_multi_turn(
        self,
        turns: List[Tuple[str, str]],
        system_prompt: Optional[str] = None,
        use_cosine: bool = True,
    ) -> List[float]:
        """
        Score refusal at each turn of a multi-turn conversation.

        Args:
            turns: List of (user_message, assistant_response) tuples
            system_prompt: Optional system prompt
            use_cosine: Use cosine similarity vs dot product

        Returns:
            List of refusal scores, one per turn (scored BEFORE each response)
        """
        scores = []
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for user_msg, assistant_response in turns:
            # Add user message
            messages.append({"role": "user", "content": user_msg})

            # Score at this point (before assistant responds)
            score = self.score_prompt_refusal(messages, use_cosine)
            scores.append(score)

            # Add assistant response for next turn's context
            messages.append({"role": "assistant", "content": assistant_response})

        return scores

    def __del__(self):
        """Cleanup hooks on deletion."""
        for hook in self._hooks:
            hook.remove()
