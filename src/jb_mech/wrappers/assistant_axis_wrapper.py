"""
Wrapper for the assistant-axis library.

Provides a unified interface for computing, loading, and using the Assistant Axis
for Llama 3.1 8B, including activation extraction, projection, and steering.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from ..config import (
    MODEL_NAME,
    TARGET_LAYER,
    TOTAL_LAYERS,
    ASSISTANT_AXIS_DIR,
    AXIS_PATH,
    add_third_party_to_path,
)


class AssistantAxisWrapper:
    """
    Unified interface for assistant-axis operations on Llama 3.1 8B.

    This wrapper handles:
    - Model loading via ProbingModel
    - Activation extraction via ActivationExtractor
    - Axis projection and computation
    - Activation steering for causal interventions
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        load_model: bool = True,
    ):
        """
        Initialize the Assistant Axis wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (None for auto)
            load_model: Whether to load the model immediately
        """
        # Add assistant-axis to path
        add_third_party_to_path()

        # Import after adding to path
        from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor
        from assistant_axis import compute_axis, project, project_batch, load_axis, save_axis
        from assistant_axis.steering import ActivationSteering

        # Store references to functions
        self._compute_axis = compute_axis
        self._project = project
        self._project_batch = project_batch
        self._load_axis = load_axis
        self._save_axis = save_axis
        self._ActivationSteering = ActivationSteering

        self.model_name = model_name
        self.probing_model = None
        self.encoder = None
        self.extractor = None
        self.axis = None

        if load_model:
            self._load_model(model_name, device)

    def _load_model(self, model_name: str, device: Optional[str] = None):
        """Load the model and initialize extraction utilities."""
        from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor

        print(f"Loading ProbingModel: {model_name}")
        self.probing_model = ProbingModel(model_name, device=device)
        self.encoder = ConversationEncoder(self.probing_model.tokenizer, model_name)
        self.extractor = ActivationExtractor(self.probing_model, self.encoder)

    def load_axis(self, axis_path: Union[str, Path] = AXIS_PATH) -> torch.Tensor:
        """
        Load a pre-computed assistant axis.

        Args:
            axis_path: Path to the axis file

        Returns:
            Loaded axis tensor of shape (n_layers, hidden_dim)
        """
        axis_path = Path(axis_path)
        if not axis_path.exists():
            raise FileNotFoundError(f"Axis not found at {axis_path}. Run the pipeline to compute it.")

        self.axis = self._load_axis(str(axis_path))
        print(f"Loaded axis from {axis_path}, shape: {self.axis.shape}")
        return self.axis

    def save_axis(self, axis: torch.Tensor, axis_path: Union[str, Path] = AXIS_PATH, metadata: dict = None):
        """
        Save a computed assistant axis.

        Args:
            axis: Axis tensor to save
            axis_path: Path to save the axis
            metadata: Optional metadata to save with the axis
        """
        axis_path = Path(axis_path)
        axis_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_axis(axis, str(axis_path), metadata)
        print(f"Saved axis to {axis_path}")

    def extract_activations(
        self,
        prompts: List[str],
        layers: Optional[List[int]] = None,
        chat_format: bool = True,
    ) -> torch.Tensor:
        """
        Extract activations for a list of prompts.

        Args:
            prompts: List of text prompts
            layers: Layers to extract (None for all layers)
            chat_format: Whether to apply chat template

        Returns:
            Tensor of shape (batch, n_layers, hidden_dim) or (batch, hidden_dim) if single layer
        """
        if self.extractor is None:
            raise RuntimeError("Model not loaded. Initialize with load_model=True or call _load_model()")

        layers = layers if layers is not None else list(range(TOTAL_LAYERS))

        activations = self.extractor.for_prompts(prompts, layer=layers)
        return activations

    def extract_conversation_activations(
        self,
        conversations: List[List[Dict[str, str]]],
        layers: Optional[List[int]] = None,
        max_length: int = 2048,
    ) -> torch.Tensor:
        """
        Extract activations for multi-turn conversations.

        Args:
            conversations: List of conversations, each is a list of {role, content} dicts
            layers: Layers to extract (None for all layers)
            max_length: Maximum sequence length

        Returns:
            Tensor of activations
        """
        if self.extractor is None:
            raise RuntimeError("Model not loaded.")

        layers = layers if layers is not None else list(range(TOTAL_LAYERS))

        activations, metadata = self.extractor.batch_conversations(
            conversations, layer=layers, max_length=max_length
        )
        return activations

    def compute_axis_from_activations(
        self,
        role_activations: torch.Tensor,
        default_activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute assistant axis from role and default activations.

        axis = mean(default) - mean(role)

        Args:
            role_activations: Activations from role-playing responses (n_role, n_layers, hidden_dim)
            default_activations: Activations from default assistant responses (n_default, n_layers, hidden_dim)

        Returns:
            Computed axis of shape (n_layers, hidden_dim)
        """
        axis = self._compute_axis(role_activations, default_activations)
        self.axis = axis
        return axis

    def project_onto_axis(
        self,
        activations: torch.Tensor,
        axis: Optional[torch.Tensor] = None,
        layer: int = TARGET_LAYER,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Project activations onto the assistant axis.

        Args:
            activations: Activations to project (batch, n_layers, hidden_dim) or (n_layers, hidden_dim)
            axis: Axis to project onto (uses loaded axis if None)
            layer: Layer index to use for projection
            normalize: Whether to normalize the axis before projection

        Returns:
            Scalar projections (batch,) or single scalar
        """
        if axis is None:
            axis = self.axis
        if axis is None:
            raise ValueError("No axis provided and no axis loaded. Call load_axis() first.")

        if activations.ndim == 2:
            # Single activation: (n_layers, hidden_dim)
            return self._project(activations, axis, layer=layer, normalize=normalize)
        else:
            # Batch: (batch, n_layers, hidden_dim)
            return self._project_batch(activations, axis, layer=layer, normalize=normalize)

    def cosine_similarity_to_axis(
        self,
        activations: torch.Tensor,
        axis: Optional[torch.Tensor] = None,
        layer: int = TARGET_LAYER,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between activations and the axis.

        Args:
            activations: Activations to compare (batch, n_layers, hidden_dim)
            axis: Axis to compare against (uses loaded axis if None)
            layer: Layer index to use

        Returns:
            Cosine similarities (batch,)
        """
        if axis is None:
            axis = self.axis
        if axis is None:
            raise ValueError("No axis provided and no axis loaded.")

        acts = activations[:, layer, :].float()
        ax = axis[layer].float()

        acts_norm = acts / (acts.norm(dim=1, keepdim=True) + 1e-8)
        ax_norm = ax / (ax.norm() + 1e-8)

        return acts_norm @ ax_norm

    def steer(
        self,
        steering_vector: torch.Tensor,
        coefficient: float = 1.0,
        layer: int = TARGET_LAYER,
        intervention_type: str = "addition",
        **kwargs,
    ):
        """
        Create an ActivationSteering context manager.

        Args:
            steering_vector: Vector to steer with
            coefficient: Steering coefficient (positive = toward, negative = away)
            layer: Layer to apply steering at
            intervention_type: "addition", "ablation", or "capping"
            **kwargs: Additional arguments for ActivationSteering

        Returns:
            ActivationSteering context manager
        """
        if self.probing_model is None:
            raise RuntimeError("Model not loaded.")

        return self._ActivationSteering(
            self.probing_model.model,
            steering_vectors=[steering_vector],
            coefficients=[coefficient],
            layer_indices=[layer],
            intervention_type=intervention_type,
            **kwargs,
        )

    def steer_toward_assistant(self, coefficient: float = 1.0, layer: int = TARGET_LAYER):
        """
        Create a steering context that steers toward the assistant direction.

        Args:
            coefficient: Strength of steering (higher = more assistant-like)
            layer: Layer to apply steering at

        Returns:
            ActivationSteering context manager
        """
        if self.axis is None:
            raise ValueError("No axis loaded. Call load_axis() first.")

        return self.steer(self.axis[layer], coefficient=coefficient, layer=layer)

    def steer_away_from_assistant(self, coefficient: float = 1.0, layer: int = TARGET_LAYER):
        """
        Create a steering context that steers away from the assistant direction.

        Args:
            coefficient: Strength of steering (higher = further from assistant)
            layer: Layer to apply steering at

        Returns:
            ActivationSteering context manager
        """
        return self.steer_toward_assistant(coefficient=-coefficient, layer=layer)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        chat_format: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            chat_format: Whether to apply chat template

        Returns:
            Generated text
        """
        if self.probing_model is None:
            raise RuntimeError("Model not loaded.")

        return self.probing_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            chat_format=chat_format,
        )

    def close(self):
        """Clean up resources."""
        if self.probing_model is not None:
            self.probing_model.close()
            self.probing_model = None
        self.encoder = None
        self.extractor = None
