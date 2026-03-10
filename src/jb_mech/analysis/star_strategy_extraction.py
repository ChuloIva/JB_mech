"""
Extract 100 diverse prompt-layer jailbreak strategies for STAR reproduction.

Filters the vulnerability database to prompt-layer attacks and uses K-Means
clustering to select maximally diverse strategies.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer


@dataclass
class StrategyExtractionConfig:
    """Configuration for strategy extraction."""
    n_strategies: int = 100

    # Filter criteria - entries must have ALL required tags and NONE of excluded tags
    required_tags: List[str] = None
    excluded_tags: List[str] = None

    # Embedding model for clustering
    embedding_model: str = "all-MiniLM-L6-v2"

    # Output paths
    output_dir: Path = Path("outputs/star_strategies")

    def __post_init__(self):
        if self.required_tags is None:
            # Must be prompt-layer and jailbreak, must be blackbox (no model access)
            self.required_tags = ["prompt-layer", "jailbreak", "blackbox"]
        if self.excluded_tags is None:
            # Exclude multimodal (we're doing text-only) and whitebox (no model access)
            self.excluded_tags = ["multimodal", "vision", "whitebox", "audio"]


@dataclass
class ExtractedStrategy:
    """A strategy extracted for STAR training."""
    id: str
    title: str
    description: str
    tags: List[str]
    paper_url: str
    cluster_id: int
    distance_to_centroid: float


class STARStrategyExtractor:
    """
    Extract diverse prompt-layer strategies for STAR reproduction.

    Pipeline:
    1. Load vulnerability database
    2. Filter to prompt-layer + jailbreak + blackbox entries
    3. Embed descriptions
    4. K-Means cluster into N groups
    5. Select entry closest to each centroid
    """

    def __init__(self, config: Optional[StrategyExtractionConfig] = None):
        self.config = config or StrategyExtractionConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = None  # Lazy load

    def _load_embedder(self):
        if self.embedder is None:
            print(f"Loading embedding model: {self.config.embedding_model}")
            self.embedder = SentenceTransformer(self.config.embedding_model)
        return self.embedder

    def load_vulnerabilities(self, db_path: Path) -> List[Dict]:
        """Load vulnerability database."""
        with open(db_path) as f:
            return json.load(f)

    def filter_entries(self, entries: List[Dict]) -> List[Dict]:
        """
        Filter entries to prompt-layer jailbreak attacks.

        Keeps entries that:
        - Have ALL required tags
        - Have NONE of excluded tags
        """
        filtered = []

        for entry in entries:
            tags = set(entry.get("tags", []))

            # Check required tags
            has_required = all(tag in tags for tag in self.config.required_tags)

            # Check excluded tags
            has_excluded = any(tag in tags for tag in self.config.excluded_tags)

            if has_required and not has_excluded:
                filtered.append(entry)

        print(f"Filtered {len(entries)} -> {len(filtered)} entries")
        print(f"  Required tags: {self.config.required_tags}")
        print(f"  Excluded tags: {self.config.excluded_tags}")

        return filtered

    def embed_entries(self, entries: List[Dict]) -> np.ndarray:
        """Embed entry descriptions."""
        embedder = self._load_embedder()
        descriptions = [e["description"] for e in entries]

        print(f"Embedding {len(descriptions)} descriptions...")
        embeddings = embedder.encode(descriptions, show_progress_bar=True)

        return embeddings

    def cluster_and_select(
        self,
        entries: List[Dict],
        embeddings: np.ndarray,
    ) -> List[ExtractedStrategy]:
        """
        Cluster entries and select one per cluster (closest to centroid).
        """
        n_clusters = min(self.config.n_strategies, len(entries))

        print(f"Clustering into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # For each cluster, find entry closest to centroid
        selected = []

        for cluster_id in range(n_clusters):
            # Get entries in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = embeddings[cluster_mask]

            # Find closest to centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            best_local_idx = np.argmin(distances)
            best_global_idx = cluster_indices[best_local_idx]

            entry = entries[best_global_idx]

            strategy = ExtractedStrategy(
                id=entry.get("cveId", entry.get("slug", f"entry_{best_global_idx}")),
                title=entry["title"],
                description=entry["description"],
                tags=entry.get("tags", []),
                paper_url=entry.get("paperUrl", ""),
                cluster_id=cluster_id,
                distance_to_centroid=float(distances[best_local_idx]),
            )
            selected.append(strategy)

        # Sort by cluster_id for consistent ordering
        selected.sort(key=lambda x: x.cluster_id)

        return selected

    def compute_diversity_stats(
        self,
        strategies: List[ExtractedStrategy],
    ) -> Dict:
        """Compute diversity statistics for selected strategies."""
        embedder = self._load_embedder()
        descriptions = [s.description for s in strategies]
        embeddings = embedder.encode(descriptions)

        # Pairwise cosine distances
        dist_matrix = cosine_distances(embeddings)
        pairwise_dist = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])

        # Tag diversity
        all_tags = set()
        for s in strategies:
            all_tags.update(s.tags)

        return {
            "n_strategies": len(strategies),
            "mean_pairwise_distance": float(pairwise_dist),
            "min_pairwise_distance": float(dist_matrix[np.triu_indices_from(dist_matrix, k=1)].min()),
            "max_pairwise_distance": float(dist_matrix[np.triu_indices_from(dist_matrix, k=1)].max()),
            "unique_tags": len(all_tags),
            "tag_list": sorted(list(all_tags)),
        }

    def extract(
        self,
        db_path: Path,
        save: bool = True,
    ) -> Tuple[List[ExtractedStrategy], Dict]:
        """
        Main extraction pipeline.

        Returns:
            Tuple of (strategies, diversity_stats)
        """
        # Load and filter
        entries = self.load_vulnerabilities(db_path)
        filtered = self.filter_entries(entries)

        if len(filtered) < self.config.n_strategies:
            print(f"WARNING: Only {len(filtered)} entries after filtering, "
                  f"requested {self.config.n_strategies}")

        # Embed
        embeddings = self.embed_entries(filtered)

        # Cluster and select
        strategies = self.cluster_and_select(filtered, embeddings)

        # Compute stats
        stats = self.compute_diversity_stats(strategies)

        print(f"\nExtracted {len(strategies)} strategies")
        print(f"Mean pairwise distance: {stats['mean_pairwise_distance']:.4f}")
        print(f"Unique tags: {stats['unique_tags']}")

        if save:
            self._save_results(strategies, stats, filtered, embeddings)

        return strategies, stats

    def _save_results(
        self,
        strategies: List[ExtractedStrategy],
        stats: Dict,
        all_filtered: List[Dict],
        embeddings: np.ndarray,
    ):
        """Save extraction results."""
        output_dir = self.config.output_dir

        # Save strategies
        strategies_path = output_dir / "star_seed_strategies.json"
        with open(strategies_path, "w") as f:
            json.dump([asdict(s) for s in strategies], f, indent=2)
        print(f"Saved strategies to {strategies_path}")

        # Save stats
        stats_path = output_dir / "extraction_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats to {stats_path}")

        # Save all filtered entries (for reference)
        filtered_path = output_dir / "all_filtered_entries.json"
        with open(filtered_path, "w") as f:
            json.dump(all_filtered, f, indent=2)
        print(f"Saved {len(all_filtered)} filtered entries to {filtered_path}")

        # Save embeddings
        embeddings_path = output_dir / "filtered_embeddings.npz"
        np.savez_compressed(embeddings_path, embeddings=embeddings)
        print(f"Saved embeddings to {embeddings_path}")

        # Save a simple text version for easy reading
        text_path = output_dir / "strategies_readable.txt"
        with open(text_path, "w") as f:
            for i, s in enumerate(strategies):
                f.write(f"{'='*80}\n")
                f.write(f"STRATEGY {i+1}: {s.title}\n")
                f.write(f"{'='*80}\n")
                f.write(f"ID: {s.id}\n")
                f.write(f"Tags: {', '.join(s.tags)}\n")
                f.write(f"Paper: {s.paper_url}\n")
                f.write(f"\nDescription:\n{s.description}\n\n")
        print(f"Saved readable version to {text_path}")


def main():
    """Run strategy extraction with default config."""
    # Default paths
    db_path = Path("scrapers/output/lm_security_db_clean.json")

    # Create extractor with config
    config = StrategyExtractionConfig(
        n_strategies=100,
        required_tags=["prompt-layer", "jailbreak", "blackbox"],
        excluded_tags=["multimodal", "vision", "whitebox", "audio"],
    )

    extractor = STARStrategyExtractor(config)
    strategies, stats = extractor.extract(db_path)

    # Print sample
    print("\n" + "="*80)
    print("SAMPLE STRATEGIES:")
    print("="*80)
    for s in strategies[:5]:
        print(f"\n[{s.cluster_id}] {s.title}")
        print(f"    Tags: {s.tags}")
        print(f"    {s.description[:200]}...")


if __name__ == "__main__":
    main()
