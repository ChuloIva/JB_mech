"""
Visualization utilities for jailbreak analysis.

Provides functions to create:
- Distribution plots for axis projections
- PCA visualizations of activation space
- (H, J, R) safety coordinate system plots
- Persona similarity radar charts
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


class JailbreakVisualizer:
    """Visualization utilities for jailbreak mechanism analysis."""

    @staticmethod
    def plot_axis_projection_distribution(
        projections: Dict[str, torch.Tensor],
        title: str = "Assistant Axis Projections by Category",
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot violin/box plots of assistant axis projections by category.

        Args:
            projections: Dictionary mapping category -> projection values
            title: Plot title
            save_path: Optional path to save the figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data
        data = []
        labels = []
        for category, values in projections.items():
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            data.append(values)
            labels.append(category)

        # Create violin plot
        parts = ax.violinplot(data, showmeans=True, showmedians=True)

        # Customize colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        # Set labels
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Projection on Assistant Axis")
        ax.set_title(title)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")

        return fig

    @staticmethod
    def plot_pca_activation_space(
        activations_dict: Dict[str, torch.Tensor],
        persona_vectors: Optional[Dict[str, torch.Tensor]] = None,
        layer: int = 16,
        highlight_personas: Optional[List[str]] = None,
        title: str = "Activation Space (PCA)",
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """
        2D PCA visualization of activation space with optional personas.

        Args:
            activations_dict: Dictionary mapping category -> activations
            persona_vectors: Optional persona vectors to plot
            layer: Layer to visualize
            highlight_personas: Persona names to annotate
            title: Plot title
            save_path: Optional path to save
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Collect all activations
        all_acts = []
        labels = []
        category_indices = {}
        current_idx = 0

        for category, acts in activations_dict.items():
            if isinstance(acts, torch.Tensor):
                if acts.ndim == 3:
                    acts = acts[:, layer, :].cpu().numpy()
                else:
                    acts = acts.cpu().numpy()
            n_samples = len(acts)
            all_acts.append(acts)
            labels.extend([category] * n_samples)
            category_indices[category] = (current_idx, current_idx + n_samples)
            current_idx += n_samples

        # Add persona vectors if provided
        persona_indices = {}
        if persona_vectors:
            for name, vec in persona_vectors.items():
                if isinstance(vec, torch.Tensor):
                    vec = vec[layer].cpu().numpy().reshape(1, -1)
                all_acts.append(vec)
                labels.append(f"persona:{name}")
                persona_indices[name] = current_idx
                current_idx += 1

        all_acts = np.vstack(all_acts)

        # PCA
        pca = PCA(n_components=2)
        embedded = pca.fit_transform(all_acts)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        # Get unique categories (excluding personas)
        categories = [c for c in activations_dict.keys()]
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

        # Plot each category
        for i, category in enumerate(categories):
            start, end = category_indices[category]
            ax.scatter(
                embedded[start:end, 0],
                embedded[start:end, 1],
                c=[colors[i]],
                label=category,
                alpha=0.6,
                s=50,
            )

        # Plot personas
        if persona_vectors:
            for name, idx in persona_indices.items():
                ax.scatter(
                    embedded[idx, 0],
                    embedded[idx, 1],
                    c="black",
                    marker="*",
                    s=200,
                    alpha=0.8,
                )
                if highlight_personas and name in highlight_personas:
                    ax.annotate(
                        name,
                        (embedded[idx, 0], embedded[idx, 1]),
                        fontsize=8,
                        alpha=0.8,
                    )

        ax.legend(loc="upper right")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_hjr_space(
        hjr_coords: Dict[str, torch.Tensor],
        title_prefix: str = "Safety Coordinate System",
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """
        Plot the (H, J, R) safety coordinate system in 2D projections.

        Args:
            hjr_coords: Dictionary mapping category -> (batch, 3) coordinates
            title_prefix: Prefix for subplot titles
            save_path: Optional path to save
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Get colors for categories
        categories = list(hjr_coords.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

        # H vs J
        ax1 = axes[0]
        for i, (cat, coords) in enumerate(hjr_coords.items()):
            if isinstance(coords, torch.Tensor):
                coords = coords.cpu().numpy()
            ax1.scatter(coords[:, 0], coords[:, 1], c=[colors[i]], alpha=0.5, label=cat)
        ax1.set_xlabel("H (Harm Detection)")
        ax1.set_ylabel("J (Jailbreak Success)")
        ax1.set_title(f"{title_prefix}: H vs J")
        ax1.legend()

        # H vs R
        ax2 = axes[1]
        for i, (cat, coords) in enumerate(hjr_coords.items()):
            if isinstance(coords, torch.Tensor):
                coords = coords.cpu().numpy()
            ax2.scatter(coords[:, 0], coords[:, 2], c=[colors[i]], alpha=0.5, label=cat)
        ax2.set_xlabel("H (Harm Detection)")
        ax2.set_ylabel("R (Refusal)")
        ax2.set_title(f"{title_prefix}: H vs R")
        ax2.legend()

        # J vs R
        ax3 = axes[2]
        for i, (cat, coords) in enumerate(hjr_coords.items()):
            if isinstance(coords, torch.Tensor):
                coords = coords.cpu().numpy()
            ax3.scatter(coords[:, 1], coords[:, 2], c=[colors[i]], alpha=0.5, label=cat)
        ax3.set_xlabel("J (Jailbreak Success)")
        ax3.set_ylabel("R (Refusal)")
        ax3.set_title(f"{title_prefix}: J vs R")
        ax3.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_persona_similarity_radar(
        similarities: Dict[str, float],
        title: str = "Persona Similarities",
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Radar chart of similarities to different persona categories.

        Args:
            similarities: Dictionary mapping persona_category -> mean similarity
            title: Plot title
            save_path: Optional path to save
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        categories = list(similarities.keys())
        values = list(similarities.values())

        # Number of variables
        N = len(categories)

        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop

        values += values[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_jailbreak_direction_decomposition(
        decomposition: Dict,
        title: str = "Jailbreak Direction Decomposition",
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Visualize the decomposition of jailbreak direction.

        Args:
            decomposition: Output from ProjectionAnalyzer.decompose_jailbreak_direction
            title: Plot title
            save_path: Optional path to save
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Pie chart of variance explained
        ax1 = axes[0]
        variance_explained = decomposition["variance_explained"]
        labels = ["Axis-Aligned", "Orthogonal"]
        sizes = [variance_explained, 1 - variance_explained]
        colors = ["#2ecc71", "#e74c3c"]

        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Variance Explained")

        # Bar chart of norms
        ax2 = axes[1]
        norms = [
            decomposition["jailbreak_norm"],
            decomposition["axis_projection"],
            decomposition["residual_norm"],
        ]
        bar_labels = ["Total", "Axis Component", "Residual"]
        bar_colors = ["#3498db", "#2ecc71", "#e74c3c"]

        ax2.bar(bar_labels, norms, color=bar_colors)
        ax2.set_ylabel("Magnitude")
        ax2.set_title("Direction Components")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_correlation_matrix(
        correlation_matrix: torch.Tensor,
        labels: List[str] = ["H", "J", "R"],
        title: str = "H, J, R Correlation Matrix",
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (6, 5),
    ) -> plt.Figure:
        """
        Plot a correlation matrix heatmap.

        Args:
            correlation_matrix: Correlation matrix tensor
            labels: Axis labels
            title: Plot title
            save_path: Optional path to save
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if isinstance(correlation_matrix, torch.Tensor):
            corr = correlation_matrix.cpu().numpy()
        else:
            corr = correlation_matrix

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(
                    j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black",
                )

        plt.colorbar(im, ax=ax)
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
