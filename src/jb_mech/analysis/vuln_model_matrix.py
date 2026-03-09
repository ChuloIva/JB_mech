"""
Model vulnerability matrix analysis using embedding similarity.

Cross-tabulates models × attack clusters and finds model vulnerability profiles.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity


class ModelVulnerabilityMatrix:
    """
    Build and analyze model x attack-cluster vulnerability matrix.

    Uses embedding similarity to find model vulnerability profiles.
    """

    # Model name normalization patterns (order matters - more specific first)
    MODEL_PATTERNS = [
        ("GPT-4o Mini", "GPT-4o Mini"),
        ("GPT-4o", "GPT-4o"),
        ("GPT-4 Turbo", "GPT-4 Turbo"),
        ("GPT-4", "GPT-4"),
        ("GPT-3.5 Turbo", "GPT-3.5 Turbo"),
        ("GPT-3.5", "GPT-3.5"),
        ("GPT-5", "GPT-5"),
        ("Claude 4.5", "Claude 4.5"),
        ("Claude 4", "Claude 4"),
        ("Claude 3.5 Sonnet", "Claude 3.5 Sonnet"),
        ("Claude 3.5", "Claude 3.5"),
        ("Claude 3 Opus", "Claude 3 Opus"),
        ("Claude 3", "Claude 3"),
        ("Llama 3.3", "Llama 3.3"),
        ("Llama 3.2", "Llama 3.2"),
        ("Llama 3.1", "Llama 3.1"),
        ("Llama 3 8B", "Llama 3 8B"),
        ("Llama 3 70B", "Llama 3 70B"),
        ("Llama 3", "Llama 3"),
        ("Llama 2 7B", "Llama 2 7B"),
        ("Llama 2 13B", "Llama 2 13B"),
        ("Llama 2 70B", "Llama 2 70B"),
        ("Llama 2", "Llama 2"),
        ("Qwen 2.5", "Qwen 2.5"),
        ("Qwen 2", "Qwen 2"),
        ("Gemini 2", "Gemini 2"),
        ("Gemini 1.5 Pro", "Gemini 1.5 Pro"),
        ("Gemini 1.5", "Gemini 1.5"),
        ("Gemma 3", "Gemma 3"),
        ("Gemma 2", "Gemma 2"),
        ("Mistral 7B", "Mistral 7B"),
        ("Mistral", "Mistral"),
        ("DeepSeek R1", "DeepSeek R1"),
        ("DeepSeek V3", "DeepSeek V3"),
        ("DeepSeek", "DeepSeek"),
        ("Vicuna", "Vicuna"),
        ("Phi", "Phi"),
    ]

    def __init__(self):
        pass

    def _normalize_model_name(self, model: str) -> str:
        """Normalize a single model name to canonical form."""
        model_lower = model.lower()
        for pattern, canonical in self.MODEL_PATTERNS:
            if pattern.lower() in model_lower:
                return canonical
        return model.strip()

    def build_matrix(
        self,
        vulnerabilities: List[Dict],
        cluster_labels: np.ndarray,
        normalize: bool = True,
        min_vulns_per_model: int = 3,
    ) -> Tuple[pd.DataFrame, List[str], List[int]]:
        """
        Build model x cluster vulnerability matrix.

        Args:
            vulnerabilities: List of vulnerability dictionaries
            cluster_labels: Cluster label for each vulnerability
            normalize: Normalize by total vulnerabilities per model
            min_vulns_per_model: Minimum vulnerabilities to include a model

        Returns:
            Tuple of (matrix_df, model_names, cluster_ids)
        """
        # Count vulnerabilities per normalized model
        model_vuln_counts = defaultdict(int)
        for vuln in vulnerabilities:
            for model in vuln.get("affectedModels", []):
                normalized = self._normalize_model_name(model)
                model_vuln_counts[normalized] += 1

        # Filter to models with enough vulnerabilities
        significant_models = sorted([
            m for m, count in model_vuln_counts.items()
            if count >= min_vulns_per_model
        ])

        clusters = sorted(set(cluster_labels) - {-1})

        # Build matrix
        matrix = np.zeros((len(significant_models), len(clusters)))
        model_totals = defaultdict(int)

        for vuln, cluster in zip(vulnerabilities, cluster_labels):
            if cluster == -1:
                continue

            for model in vuln.get("affectedModels", []):
                normalized = self._normalize_model_name(model)
                if normalized in significant_models:
                    model_idx = significant_models.index(normalized)
                    cluster_idx = clusters.index(cluster)
                    matrix[model_idx, cluster_idx] += 1
                    model_totals[normalized] += 1

        # Normalize if requested
        if normalize:
            for i, model in enumerate(significant_models):
                if model_totals[model] > 0:
                    matrix[i, :] /= model_totals[model]

        df = pd.DataFrame(
            matrix,
            index=significant_models,
            columns=[f"Cluster_{c}" for c in clusters],
        )

        return df, significant_models, clusters

    def compute_model_similarity(
        self,
        matrix_df: pd.DataFrame,
        method: str = "cosine",
    ) -> pd.DataFrame:
        """
        Compute pairwise similarity between models based on vulnerability profiles.

        Args:
            matrix_df: Model x cluster matrix
            method: "cosine" or "correlation"

        Returns:
            Model x model similarity matrix
        """
        if method == "cosine":
            similarity = cosine_similarity(matrix_df.values)
        elif method == "correlation":
            similarity = np.corrcoef(matrix_df.values)
        else:
            raise ValueError(f"Unknown method: {method}")

        return pd.DataFrame(
            similarity,
            index=matrix_df.index,
            columns=matrix_df.index,
        )

    def compute_embedding_similarity(
        self,
        vulnerabilities: List[Dict],
        embeddings: np.ndarray,
        min_vulns: int = 5,
    ) -> Dict:
        """
        Compute model vulnerability profiles using embedding similarity.

        For each model, compute average embedding of its vulnerabilities.

        Args:
            vulnerabilities: List of vulnerability dicts
            embeddings: (N, D) embeddings
            min_vulns: Minimum vulnerabilities to include model

        Returns:
            Dict with model_names, model_embeddings, similarity_matrix
        """
        # Compute per-model embeddings
        model_embeddings = defaultdict(list)

        for vuln, emb in zip(vulnerabilities, embeddings):
            for model in vuln.get("affectedModels", []):
                normalized = self._normalize_model_name(model)
                model_embeddings[normalized].append(emb)

        # Filter to models with enough data
        model_embeddings = {
            k: v for k, v in model_embeddings.items()
            if len(v) >= min_vulns
        }

        # Average embeddings per model
        model_names = sorted(model_embeddings.keys())
        avg_embeddings = np.array([
            np.mean(model_embeddings[m], axis=0)
            for m in model_names
        ])

        # Compute similarity
        similarity = cosine_similarity(avg_embeddings)

        return {
            "model_names": model_names,
            "model_embeddings": avg_embeddings,
            "similarity_matrix": pd.DataFrame(
                similarity,
                index=model_names,
                columns=model_names,
            ),
        }

    def find_model_blind_spots(
        self,
        matrix_df: pd.DataFrame,
        cluster_summaries: Dict[int, str],
        threshold: float = 0.02,
    ) -> Dict[str, List[str]]:
        """
        Identify attack clusters each model is NOT vulnerable to.

        Args:
            matrix_df: Model x cluster matrix (normalized)
            cluster_summaries: Human-readable cluster names
            threshold: Below this is considered "not vulnerable"

        Returns:
            Dict mapping model -> list of "blind spot" clusters
        """
        blind_spots = {}

        for model in matrix_df.index:
            row = matrix_df.loc[model]
            not_vulnerable = row[row <= threshold].index.tolist()

            if not_vulnerable:
                blind_spots[model] = [
                    cluster_summaries.get(
                        int(c.replace("Cluster_", "")),
                        c
                    )
                    for c in not_vulnerable
                ]

        return blind_spots


class ModelMatrixVisualizer:
    """Visualization for model vulnerability analysis."""

    @staticmethod
    def plot_vulnerability_heatmap(
        matrix_df: pd.DataFrame,
        cluster_names: Optional[Dict[int, str]] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (16, 12),
    ) -> plt.Figure:
        """
        Heatmap of model x cluster vulnerabilities.
        """
        plot_df = matrix_df.copy()

        # Rename columns if cluster names provided
        if cluster_names:
            plot_df = plot_df.rename(columns={
                f"Cluster_{k}": v for k, v in cluster_names.items()
            })

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            plot_df,
            cmap="YlOrRd",
            ax=ax,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Vulnerability Share (normalized)"},
        )

        ax.set_title("Model Vulnerability Profile by Attack Cluster")
        ax.set_xlabel("Attack Cluster")
        ax.set_ylabel("Model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"Saved heatmap to {save_path}")

        return fig

    @staticmethod
    def plot_model_dendrogram(
        similarity_df: pd.DataFrame,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Dendrogram showing model clustering by vulnerability similarity.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Convert similarity to distance
        distance = 1 - similarity_df.values
        np.fill_diagonal(distance, 0)

        # Ensure non-negative
        distance = np.maximum(distance, 0)

        # Compute linkage
        condensed = squareform(distance, checks=False)
        linkage_matrix = linkage(condensed, method="average")

        dendrogram(
            linkage_matrix,
            labels=similarity_df.index.tolist(),
            ax=ax,
            leaf_rotation=90,
        )

        ax.set_title("Model Similarity Dendrogram (by Vulnerability Profile)")
        ax.set_xlabel("Model")
        ax.set_ylabel("Distance (1 - Cosine Similarity)")
        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"Saved dendrogram to {save_path}")

        return fig

    @staticmethod
    def plot_embedding_similarity_heatmap(
        similarity_df: pd.DataFrame,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 12),
    ) -> plt.Figure:
        """
        Heatmap of model-to-model similarity based on embeddings.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Cluster the heatmap
        sns.clustermap(
            similarity_df,
            cmap="RdYlBu_r",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            figsize=figsize,
            cbar_kws={"label": "Cosine Similarity"},
        )

        plt.title("Model Similarity (based on vulnerability embeddings)")

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"Saved embedding similarity to {save_path}")

        return fig
