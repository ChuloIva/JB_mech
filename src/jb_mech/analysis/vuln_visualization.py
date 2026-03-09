"""
Visualization utilities for vulnerability clustering analysis.
"""

from typing import Dict, List, Optional
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


class VulnerabilityVisualizer:
    """Visualization utilities for vulnerability analysis."""

    @staticmethod
    def plot_cluster_scatter(
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        titles: List[str],
        tags: List[List[str]],
        save_path: Optional[Path] = None,
        paper_urls: Optional[List[str]] = None,
    ) -> go.Figure:
        """
        Interactive scatter plot of clustered vulnerabilities.

        Args:
            embeddings_2d: (N, 2) UMAP embeddings
            labels: Cluster labels (-1 for noise)
            titles: Vulnerability titles for hover
            tags: Tags for each vulnerability
            save_path: Optional path to save HTML
            paper_urls: Optional list of paper URLs for click-to-open
        """
        import pandas as pd

        # Create dataframe for plotly
        df = pd.DataFrame({
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "cluster": [f"Cluster {l}" if l >= 0 else "Noise" for l in labels],
            "title": titles,
            "tags": [", ".join(t[:5]) for t in tags],
        })

        if paper_urls:
            df["paper_url"] = paper_urls

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            hover_data=["title", "tags"],
            title="Vulnerability Clusters (UMAP + HDBSCAN)",
            labels={"x": "UMAP 1", "y": "UMAP 2", "cluster": "Cluster"},
            custom_data=["paper_url"] if paper_urls else None,
        )

        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            width=1200,
            height=800,
            legend_title_text="Cluster",
        )

        if save_path:
            fig.write_html(str(save_path))
            print(f"Saved interactive plot to {save_path}")

        return fig

    @staticmethod
    def plot_cluster_scatter_clickable(
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        titles: List[str],
        tags: List[List[str]],
        paper_urls: List[str],
        paper_titles: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
    ) -> go.Figure:
        """
        Interactive scatter plot with clickable points that open paper URLs.

        Args:
            embeddings_2d: (N, 2) UMAP embeddings
            labels: Cluster labels (-1 for noise)
            titles: Vulnerability titles for hover
            tags: Tags for each vulnerability
            paper_urls: Paper URLs to open on click
            paper_titles: Optional paper titles for hover
            save_path: Optional path to save HTML
        """
        import pandas as pd

        # Create dataframe
        df = pd.DataFrame({
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "cluster": [f"Cluster {l}" if l >= 0 else "Noise" for l in labels],
            "title": titles,
            "tags": [", ".join(t[:5]) for t in tags],
            "paper_url": paper_urls,
            "paper_title": paper_titles if paper_titles else [""] * len(titles),
        })

        # Get unique clusters for consistent colors
        unique_clusters = sorted(df["cluster"].unique(), key=lambda x: (x == "Noise", x))
        color_map = {c: i for i, c in enumerate(unique_clusters)}

        # Create figure with subplots for each cluster (to maintain colors)
        fig = go.Figure()

        for cluster in unique_clusters:
            cluster_df = df[df["cluster"] == cluster]

            fig.add_trace(go.Scatter(
                x=cluster_df["x"],
                y=cluster_df["y"],
                mode="markers",
                name=cluster,
                marker=dict(size=10, opacity=0.7),
                text=cluster_df["title"],
                customdata=list(zip(
                    cluster_df["paper_url"],
                    cluster_df["tags"],
                    cluster_df["paper_title"],
                )),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Tags: %{customdata[1]}<br>"
                    "Paper: %{customdata[2]}<br>"
                    "<i>Click to open paper</i>"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            title="Vulnerability Clusters (Click to Open Paper)",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            width=1400,
            height=900,
            legend_title_text="Cluster",
            hovermode="closest",
        )

        if save_path:
            # Add JavaScript for click handling
            html_content = fig.to_html(include_plotlyjs=True, full_html=True)

            # Inject click handler JavaScript
            click_handler_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    var plot = document.getElementsByClassName('plotly-graph-div')[0];

    plot.on('plotly_click', function(data) {
        var point = data.points[0];
        if (point.customdata && point.customdata[0]) {
            var url = point.customdata[0];
            if (url && url.length > 0) {
                window.open(url, '_blank');
            }
        }
    });

    // Change cursor on hover
    plot.on('plotly_hover', function(data) {
        plot.style.cursor = 'pointer';
    });

    plot.on('plotly_unhover', function(data) {
        plot.style.cursor = 'default';
    });
});
</script>
"""
            # Insert before closing body tag
            html_content = html_content.replace("</body>", click_handler_js + "</body>")

            with open(save_path, "w") as f:
                f.write(html_content)

            print(f"Saved clickable plot to {save_path}")

        return fig

    @staticmethod
    def plot_cluster_tag_heatmap(
        labels: np.ndarray,
        tags: List[List[str]],
        save_path: Optional[Path] = None,
        figsize: tuple = (14, 10),
        top_n_tags: int = 20,
    ) -> plt.Figure:
        """
        Heatmap showing tag distribution across clusters.
        """
        # Get unique clusters (excluding noise)
        clusters = sorted(set(labels) - {-1})

        if not clusters:
            print("No clusters found (all noise)")
            return None

        # Get top tags by overall frequency
        all_tags_flat = [t for tag_list in tags for t in tag_list]
        top_tags = [t for t, _ in Counter(all_tags_flat).most_common(top_n_tags)]

        # Build matrix
        matrix = np.zeros((len(clusters), len(top_tags)))
        for i, cluster in enumerate(clusters):
            cluster_mask = labels == cluster
            cluster_tags = [t for j, tag_list in enumerate(tags)
                          if cluster_mask[j] for t in tag_list]
            tag_counts = Counter(cluster_tags)
            cluster_size = cluster_mask.sum()
            for j, tag in enumerate(top_tags):
                matrix[i, j] = tag_counts.get(tag, 0) / cluster_size if cluster_size > 0 else 0

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            matrix,
            xticklabels=top_tags,
            yticklabels=[f"Cluster {c}" for c in clusters],
            cmap="YlOrRd",
            ax=ax,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Tag Prevalence (normalized)"},
        )
        ax.set_title("Tag Prevalence by Cluster")
        ax.set_xlabel("Tag")
        ax.set_ylabel("Cluster")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"Saved heatmap to {save_path}")

        return fig

    @staticmethod
    def plot_cluster_summary(
        labels: np.ndarray,
        tags: List[List[str]],
        titles: List[str],
        save_path: Optional[Path] = None,
    ) -> Dict[int, Dict]:
        """
        Generate text summary for each cluster.

        Returns dict mapping cluster_id -> {size, top_tags, example_titles}
        """
        clusters = sorted(set(labels) - {-1})
        summaries = {}

        for cluster in clusters:
            mask = labels == cluster
            cluster_tags = [t for i, tag_list in enumerate(tags) if mask[i] for t in tag_list]
            cluster_titles = [titles[i] for i in range(len(titles)) if mask[i]]

            top_tags = [t for t, _ in Counter(cluster_tags).most_common(5)]

            summaries[int(cluster)] = {
                "size": int(mask.sum()),
                "top_tags": top_tags,
                "example_titles": cluster_titles[:5],
                "label": "+".join(top_tags[:3]),
            }

        if save_path:
            import json
            with open(save_path, "w") as f:
                json.dump(summaries, f, indent=2)
            print(f"Saved cluster summaries to {save_path}")

        return summaries
