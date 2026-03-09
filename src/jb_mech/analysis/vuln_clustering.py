"""
Clustering analysis for vulnerability embeddings.

Uses HDBSCAN for density-based clustering and UMAP for dimensionality reduction.
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from collections import Counter

import numpy as np
import hdbscan
import umap
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringConfig:
    """Configuration for HDBSCAN clustering."""
    min_cluster_size: int = 5
    min_samples: int = 3
    cluster_selection_method: str = "eom"  # excess of mass
    metric: str = "euclidean"


@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction."""
    n_neighbors: int = 15
    n_components: int = 2
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42


class VulnerabilityClusterer:
    """
    Cluster vulnerability embeddings using HDBSCAN.

    Finds natural "attack families" beyond what tags capture.
    """

    def __init__(
        self,
        cluster_config: ClusteringConfig = None,
        umap_config: UMAPConfig = None,
    ):
        self.cluster_config = cluster_config or ClusteringConfig()
        self.umap_config = umap_config or UMAPConfig()

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = None,
    ) -> np.ndarray:
        """
        Reduce embedding dimensions with UMAP.

        Args:
            embeddings: (N, D) high-dimensional embeddings
            n_components: Target dimensions (default: 2 for viz)

        Returns:
            (N, n_components) reduced embeddings
        """
        n_components = n_components or self.umap_config.n_components

        reducer = umap.UMAP(
            n_neighbors=self.umap_config.n_neighbors,
            n_components=n_components,
            min_dist=self.umap_config.min_dist,
            metric=self.umap_config.metric,
            random_state=self.umap_config.random_state,
        )

        return reducer.fit_transform(embeddings)

    def cluster(
        self,
        embeddings: np.ndarray,
        use_umap_for_clustering: bool = True,
        umap_dims_for_clustering: int = 50,
    ) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """
        Cluster embeddings using HDBSCAN.

        Args:
            embeddings: (N, D) embeddings
            use_umap_for_clustering: Reduce dims before clustering
            umap_dims_for_clustering: Target dims for clustering

        Returns:
            Tuple of (cluster_labels, fitted_clusterer)
        """
        if use_umap_for_clustering and embeddings.shape[1] > umap_dims_for_clustering:
            print(f"Reducing to {umap_dims_for_clustering}D for clustering...")
            embeddings_for_clustering = self.reduce_dimensions(
                embeddings, n_components=umap_dims_for_clustering
            )
        else:
            embeddings_for_clustering = embeddings

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cluster_config.min_cluster_size,
            min_samples=self.cluster_config.min_samples,
            cluster_selection_method=self.cluster_config.cluster_selection_method,
            metric=self.cluster_config.metric,
        )

        labels = clusterer.fit_predict(embeddings_for_clustering)

        return labels, clusterer

    def get_cluster_stats(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> Dict:
        """Compute clustering statistics."""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())

        # Silhouette score (excluding noise)
        mask = labels != -1
        if mask.sum() > 1 and n_clusters > 1:
            sil_score = float(silhouette_score(embeddings[mask], labels[mask]))
        else:
            sil_score = None

        cluster_sizes = Counter(labels.tolist())

        return {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_ratio": float(n_noise / len(labels)),
            "silhouette_score": sil_score,
            "cluster_sizes": {str(k): v for k, v in cluster_sizes.items()},
        }
