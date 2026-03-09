#!/usr/bin/env python3
"""
Analyze LLM vulnerabilities from Promptfoo LM Security Database.

This script runs the full analysis pipeline:
1. Embed vulnerability descriptions (OpenAI)
2. Cluster to find attack families (HDBSCAN)
3. Extract structured attack templates (Trinity)
4. Build model vulnerability matrix

Usage:
    python scripts/analyze_vuln_database.py
    python scripts/analyze_vuln_database.py --skip-templates
    python scripts/analyze_vuln_database.py --templates-only --max-templates 50
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.jb_mech.analysis.vuln_embeddings import VulnerabilityEmbedder, EmbeddingConfig
from src.jb_mech.analysis.vuln_clustering import VulnerabilityClusterer, ClusteringConfig
from src.jb_mech.analysis.vuln_visualization import VulnerabilityVisualizer
from src.jb_mech.analysis.vuln_templates import AttackTemplateExtractor, TemplateExtractionConfig
from src.jb_mech.analysis.vuln_model_matrix import (
    ModelVulnerabilityMatrix,
    ModelMatrixVisualizer,
)


def load_vulnerabilities(path: Path) -> list:
    """Load vulnerability data from JSON."""
    with open(path) as f:
        return json.load(f)


def generate_cluster_summaries(
    vulnerabilities: list,
    labels: list,
) -> dict:
    """Generate human-readable cluster summaries using most common tags."""
    cluster_summaries = {}
    clusters = sorted(set(labels) - {-1})

    for cluster in clusters:
        cluster_vulns = [v for v, l in zip(vulnerabilities, labels) if l == cluster]

        # Get most common tags
        tag_counts = Counter()
        for v in cluster_vulns:
            tag_counts.update(v.get("tags", []))

        top_tags = [t for t, _ in tag_counts.most_common(3)]
        cluster_summaries[cluster] = f"{'+'.join(top_tags)} ({len(cluster_vulns)})"

    return cluster_summaries


def run_embedding_clustering(
    vulnerabilities: list,
    output_dir: Path,
) -> tuple:
    """Run embedding and clustering pipeline."""
    print("\n" + "=" * 60)
    print("PHASE 1: Embedding & Clustering")
    print("=" * 60)

    descriptions = [v.get("description", "") for v in vulnerabilities]

    # Generate embeddings
    print("\n[1/3] Generating embeddings...")
    embedder = VulnerabilityEmbedder(EmbeddingConfig(
        cache_dir=output_dir / "embeddings_cache"
    ))
    embeddings = embedder.embed_texts(descriptions)
    print(f"Embeddings shape: {embeddings.shape}")

    # Cluster
    print("\n[2/3] Clustering vulnerabilities...")
    clusterer = VulnerabilityClusterer()

    # Get 2D embeddings for visualization
    print("  Reducing to 2D for visualization...")
    embeddings_2d = clusterer.reduce_dimensions(embeddings, n_components=2)

    # Cluster
    labels, _ = clusterer.cluster(embeddings)
    stats = clusterer.get_cluster_stats(labels, embeddings)

    print(f"  Found {stats['n_clusters']} clusters")
    print(f"  Noise points: {stats['n_noise_points']} ({stats['noise_ratio']:.1%})")
    if stats['silhouette_score']:
        print(f"  Silhouette score: {stats['silhouette_score']:.3f}")

    # Generate summaries
    cluster_summaries = generate_cluster_summaries(vulnerabilities, labels.tolist())

    # Save cluster results
    cluster_results = {
        "stats": stats,
        "labels": labels.tolist(),
        "summaries": cluster_summaries,
    }
    with open(output_dir / "cluster_results.json", "w") as f:
        json.dump(cluster_results, f, indent=2)

    # Visualizations
    print("\n[3/3] Generating visualizations...")
    viz = VulnerabilityVisualizer()

    titles = [v.get("title", "") for v in vulnerabilities]
    tags = [v.get("tags", []) for v in vulnerabilities]

    # Interactive scatter plot
    viz.plot_cluster_scatter(
        embeddings_2d, labels, titles, tags,
        save_path=output_dir / "cluster_scatter.html"
    )

    # Clickable scatter plot (opens paper on click)
    paper_urls = [v.get("paperUrl", "") for v in vulnerabilities]
    paper_titles = [v.get("paperTitle", "") for v in vulnerabilities]
    viz.plot_cluster_scatter_clickable(
        embeddings_2d, labels, titles, tags,
        paper_urls=paper_urls,
        paper_titles=paper_titles,
        save_path=output_dir / "cluster_scatter_clickable.html"
    )

    # Tag heatmap
    viz.plot_cluster_tag_heatmap(
        labels, tags,
        save_path=output_dir / "cluster_tag_heatmap.png"
    )

    # Cluster summaries
    viz.plot_cluster_summary(
        labels, tags, titles,
        save_path=output_dir / "cluster_summaries.json"
    )

    return embeddings, labels, cluster_summaries


def run_template_extraction(
    vulnerabilities: list,
    output_dir: Path,
    max_items: int = None,
) -> list:
    """Run template extraction pipeline."""
    print("\n" + "=" * 60)
    print("PHASE 2: Attack Template Extraction")
    print("=" * 60)

    extractor = AttackTemplateExtractor(TemplateExtractionConfig(
        cache_dir=output_dir / "templates_cache"
    ))

    cached = extractor.get_cached_count()
    print(f"\nCached templates: {cached}/{len(vulnerabilities)}")

    if max_items:
        print(f"Processing up to {max_items} items...")

    templates = extractor.extract_all(vulnerabilities, max_items=max_items)
    print(f"\nExtracted {len(templates)} templates")

    return templates


def run_model_matrix(
    vulnerabilities: list,
    embeddings,
    labels,
    cluster_summaries: dict,
    output_dir: Path,
) -> None:
    """Run model vulnerability matrix analysis."""
    print("\n" + "=" * 60)
    print("PHASE 3: Model Vulnerability Matrix")
    print("=" * 60)

    matrix_analyzer = ModelVulnerabilityMatrix()

    # Build cluster-based matrix
    print("\n[1/4] Building model x cluster matrix...")
    matrix_df, model_list, cluster_ids = matrix_analyzer.build_matrix(
        vulnerabilities, labels, normalize=True, min_vulns_per_model=5
    )
    print(f"  Matrix shape: {matrix_df.shape}")
    print(f"  Models included: {len(model_list)}")

    # Save raw matrix
    matrix_df.to_csv(output_dir / "model_cluster_matrix.csv")

    # Compute model similarity from clusters
    print("\n[2/4] Computing model similarity (cluster-based)...")
    similarity_df = matrix_analyzer.compute_model_similarity(matrix_df)
    similarity_df.to_csv(output_dir / "model_similarity_clusters.csv")

    # Compute embedding-based similarity
    print("\n[3/4] Computing model similarity (embedding-based)...")
    emb_results = matrix_analyzer.compute_embedding_similarity(
        vulnerabilities, embeddings, min_vulns=5
    )
    emb_results["similarity_matrix"].to_csv(
        output_dir / "model_similarity_embeddings.csv"
    )

    # Find blind spots
    blind_spots = matrix_analyzer.find_model_blind_spots(
        matrix_df, cluster_summaries, threshold=0.02
    )
    with open(output_dir / "model_blind_spots.json", "w") as f:
        json.dump(blind_spots, f, indent=2)

    # Visualizations
    print("\n[4/4] Generating visualizations...")
    viz = ModelMatrixVisualizer()

    # Heatmap
    viz.plot_vulnerability_heatmap(
        matrix_df,
        cluster_names=cluster_summaries,
        save_path=output_dir / "model_vulnerability_heatmap.png"
    )

    # Dendrogram
    viz.plot_model_dendrogram(
        similarity_df,
        save_path=output_dir / "model_similarity_dendrogram.png"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM vulnerability database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (skip templates for quick run)
    python scripts/analyze_vuln_database.py --skip-templates

    # Only run template extraction (slow)
    python scripts/analyze_vuln_database.py --templates-only

    # Limit template extraction for testing
    python scripts/analyze_vuln_database.py --max-templates 50
        """
    )
    parser.add_argument(
        "--skip-templates",
        action="store_true",
        help="Skip template extraction (slow, uses API)"
    )
    parser.add_argument(
        "--templates-only",
        action="store_true",
        help="Only run template extraction"
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=None,
        help="Limit number of templates to extract"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("scrapers/output/lm_security_db_clean.json"),
        help="Path to vulnerability JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vuln_analysis"),
        help="Output directory"
    )
    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LLM Vulnerability Analysis Pipeline")
    print("=" * 60)

    # Load data
    print(f"\nLoading vulnerabilities from {args.data_path}...")
    vulnerabilities = load_vulnerabilities(args.data_path)
    print(f"Loaded {len(vulnerabilities)} vulnerabilities")

    if args.templates_only:
        # Only run template extraction
        run_template_extraction(
            vulnerabilities,
            args.output_dir,
            max_items=args.max_templates,
        )
    else:
        # Run embedding & clustering
        embeddings, labels, cluster_summaries = run_embedding_clustering(
            vulnerabilities,
            args.output_dir,
        )

        # Run template extraction (optional)
        if not args.skip_templates:
            run_template_extraction(
                vulnerabilities,
                args.output_dir,
                max_items=args.max_templates,
            )
        else:
            print("\n[Skipping template extraction]")

        # Run model matrix analysis
        run_model_matrix(
            vulnerabilities,
            embeddings,
            labels,
            cluster_summaries,
            args.output_dir,
        )

    # Summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {args.output_dir}")
    print("\nKey files:")
    for f in sorted(args.output_dir.glob("*")):
        if f.is_file() and not f.name.endswith("_cache"):
            size = f.stat().st_size
            size_str = f"{size / 1024:.1f}KB" if size > 1024 else f"{size}B"
            print(f"  {f.name}: {size_str}")


if __name__ == "__main__":
    main()
