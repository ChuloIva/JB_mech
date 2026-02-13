"""
Projection analysis for jailbreak mechanism understanding.

Provides tools for:
- Projecting activations onto the Assistant Axis
- Computing jailbreak directions
- Decomposing jailbreak effects into axis-aligned and orthogonal components
- Computing cosine similarities to persona vectors
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..config import TARGET_LAYER


class ProjectionAnalyzer:
    """
    Analyze activations by projecting onto reference directions.

    Supports:
    - Assistant Axis projection (how "assistant-like" is the activation)
    - Persona vector projection (similarity to specific personas)
    - Jailbreak direction computation and decomposition
    """

    def __init__(
        self,
        axis_path: Optional[Union[str, Path]] = None,
        persona_vectors_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the projection analyzer.

        Args:
            axis_path: Path to the assistant axis .pt file
            persona_vectors_path: Path to persona vectors .pt file
        """
        self.axis = None
        self.persona_vectors = None

        if axis_path:
            self.load_axis(axis_path)

        if persona_vectors_path:
            self.load_persona_vectors(persona_vectors_path)

    def load_axis(self, path: Union[str, Path]):
        """Load the assistant axis from a file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Axis not found: {path}")

        self.axis = torch.load(path, map_location="cpu")
        print(f"Loaded axis from {path}, shape: {self.axis.shape}")

    def load_persona_vectors(self, path: Union[str, Path]):
        """Load persona vectors from a file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Persona vectors not found: {path}")

        self.persona_vectors = torch.load(path, map_location="cpu")
        print(f"Loaded {len(self.persona_vectors)} persona vectors from {path}")

    def project_onto_axis(
        self,
        activations: torch.Tensor,
        layer: int = TARGET_LAYER,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Project activations onto the assistant axis.

        Higher values indicate more "assistant-like" behavior.

        Args:
            activations: Activations (batch, n_layers, hidden_dim) or (n_layers, hidden_dim)
            layer: Layer index to use
            normalize: Whether to normalize the axis

        Returns:
            Scalar projections (batch,) or single scalar
        """
        if self.axis is None:
            raise ValueError("No axis loaded. Call load_axis() first.")

        ax = self.axis[layer].float()
        if normalize:
            ax = ax / (ax.norm() + 1e-8)

        if activations.ndim == 2:
            # Single activation
            acts = activations[layer].float()
            return (acts @ ax).item()
        else:
            # Batch
            acts = activations[:, layer, :].float()
            return acts @ ax

    def project_onto_persona(
        self,
        activations: torch.Tensor,
        persona_name: str,
        layer: int = TARGET_LAYER,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Project activations onto a specific persona vector.

        Args:
            activations: Activations (batch, n_layers, hidden_dim)
            persona_name: Name of the persona
            layer: Layer index to use
            normalize: Whether to normalize

        Returns:
            Scalar projections (batch,)
        """
        if self.persona_vectors is None:
            raise ValueError("No persona vectors loaded.")

        if persona_name not in self.persona_vectors:
            raise ValueError(f"Unknown persona: {persona_name}")

        persona_vec = self.persona_vectors[persona_name][layer].float()
        if normalize:
            persona_vec = persona_vec / (persona_vec.norm() + 1e-8)

        acts = activations[:, layer, :].float()
        return acts @ persona_vec

    def cosine_similarity_to_axis(
        self,
        activations: torch.Tensor,
        layer: int = TARGET_LAYER,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between activations and the axis.

        Args:
            activations: Activations (batch, n_layers, hidden_dim)
            layer: Layer index

        Returns:
            Cosine similarities (batch,)
        """
        if self.axis is None:
            raise ValueError("No axis loaded.")

        acts = activations[:, layer, :].float()
        ax = self.axis[layer].float()

        acts_norm = acts / (acts.norm(dim=1, keepdim=True) + 1e-8)
        ax_norm = ax / (ax.norm() + 1e-8)

        return acts_norm @ ax_norm

    def cosine_similarity_to_persona(
        self,
        activations: torch.Tensor,
        persona_name: str,
        layer: int = TARGET_LAYER,
    ) -> torch.Tensor:
        """
        Compute cosine similarity to a persona vector.

        Args:
            activations: Activations (batch, n_layers, hidden_dim)
            persona_name: Name of the persona
            layer: Layer index

        Returns:
            Cosine similarities (batch,)
        """
        if self.persona_vectors is None:
            raise ValueError("No persona vectors loaded.")

        acts = activations[:, layer, :].float()
        persona = self.persona_vectors[persona_name][layer].float()

        acts_norm = acts / (acts.norm(dim=1, keepdim=True) + 1e-8)
        persona_norm = persona / (persona.norm() + 1e-8)

        return acts_norm @ persona_norm

    def compute_jailbreak_direction(
        self,
        jailbreak_acts: torch.Tensor,
        clean_acts: torch.Tensor,
        layer: int = TARGET_LAYER,
    ) -> torch.Tensor:
        """
        Compute the mean jailbreak direction (difference vector).

        jailbreak_direction = mean(jailbreak_acts) - mean(clean_acts)

        Args:
            jailbreak_acts: Activations from successful jailbreaks (batch, n_layers, hidden_dim)
            clean_acts: Activations from clean prompts (batch, n_layers, hidden_dim)
            layer: Layer index

        Returns:
            Jailbreak direction vector (hidden_dim,)
        """
        jb_mean = jailbreak_acts[:, layer, :].float().mean(dim=0)
        clean_mean = clean_acts[:, layer, :].float().mean(dim=0)

        return jb_mean - clean_mean

    def decompose_jailbreak_direction(
        self,
        jailbreak_direction: torch.Tensor,
        layer: int = TARGET_LAYER,
    ) -> Dict:
        """
        Decompose jailbreak direction into assistant axis component and residual.

        This helps understand how much of the jailbreak effect is explained
        by movement away from the "assistant" direction.

        Args:
            jailbreak_direction: The jailbreak direction vector (hidden_dim,)
            layer: Layer index for the axis

        Returns:
            Dictionary with:
                - axis_component: Component along the axis
                - residual: Orthogonal component
                - axis_projection: Scalar projection onto axis
                - variance_explained: Fraction of variance explained by axis
                - residual_norm: Magnitude of residual
        """
        if self.axis is None:
            raise ValueError("No axis loaded.")

        jb_dir = jailbreak_direction.float()
        ax = self.axis[layer].float()
        ax_norm = ax / (ax.norm() + 1e-8)

        # Project onto axis
        axis_projection = (jb_dir @ ax_norm).item()
        axis_component = axis_projection * ax_norm

        # Compute residual
        residual = jb_dir - axis_component

        # Compute variance explained
        total_variance = (jb_dir ** 2).sum().item()
        axis_variance = (axis_component ** 2).sum().item()
        variance_explained = axis_variance / (total_variance + 1e-8)

        return {
            "axis_component": axis_component,
            "residual": residual,
            "axis_projection": axis_projection,
            "variance_explained": variance_explained,
            "residual_norm": residual.norm().item(),
            "jailbreak_norm": jb_dir.norm().item(),
        }

    def analyze_all_personas(
        self,
        activations: torch.Tensor,
        layer: int = TARGET_LAYER,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cosine similarity to all loaded persona vectors.

        Args:
            activations: Activations (batch, n_layers, hidden_dim)
            layer: Layer index

        Returns:
            Dictionary mapping persona_name -> similarities (batch,)
        """
        if self.persona_vectors is None:
            raise ValueError("No persona vectors loaded.")

        similarities = {}
        for name in self.persona_vectors.keys():
            similarities[name] = self.cosine_similarity_to_persona(activations, name, layer)

        return similarities

    def find_closest_personas(
        self,
        activations: torch.Tensor,
        k: int = 5,
        layer: int = TARGET_LAYER,
    ) -> List[Tuple[str, float]]:
        """
        Find the k closest persona vectors to the mean activation.

        Args:
            activations: Activations (batch, n_layers, hidden_dim)
            k: Number of top personas to return
            layer: Layer index

        Returns:
            List of (persona_name, mean_similarity) tuples, sorted descending
        """
        similarities = self.analyze_all_personas(activations, layer)

        # Compute mean similarity for each persona
        mean_sims = [(name, sim.mean().item()) for name, sim in similarities.items()]

        # Sort by similarity (descending)
        mean_sims.sort(key=lambda x: x[1], reverse=True)

        return mean_sims[:k]

    def compare_projections(
        self,
        group1_acts: torch.Tensor,
        group2_acts: torch.Tensor,
        layer: int = TARGET_LAYER,
    ) -> Dict:
        """
        Compare axis projections between two groups.

        Useful for testing hypotheses about jailbreak vs clean prompts.

        Args:
            group1_acts: First group activations (batch, n_layers, hidden_dim)
            group2_acts: Second group activations (batch, n_layers, hidden_dim)
            layer: Layer index

        Returns:
            Dictionary with statistics:
                - group1_mean, group1_std
                - group2_mean, group2_std
                - difference: mean difference
                - effect_size: Cohen's d
        """
        proj1 = self.project_onto_axis(group1_acts, layer)
        proj2 = self.project_onto_axis(group2_acts, layer)

        p1 = proj1.numpy() if isinstance(proj1, torch.Tensor) else np.array([proj1])
        p2 = proj2.numpy() if isinstance(proj2, torch.Tensor) else np.array([proj2])

        # Compute Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(p1) - 1) * p1.std() ** 2 + (len(p2) - 1) * p2.std() ** 2)
            / (len(p1) + len(p2) - 2)
        )
        effect_size = (p1.mean() - p2.mean()) / (pooled_std + 1e-8)

        return {
            "group1_projections": proj1,
            "group2_projections": proj2,
            "group1_mean": p1.mean(),
            "group1_std": p1.std(),
            "group2_mean": p2.mean(),
            "group2_std": p2.std(),
            "difference": p1.mean() - p2.mean(),
            "effect_size": effect_size,
        }
