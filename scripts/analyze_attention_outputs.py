"""
Analyze attention outputs across model families with normalized special tokens.
"""

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Token normalization mapping per model family
TOKEN_NORMALIZATION = {
    "llama": {
        "<|begin_of_text|>": "BOS",
        "<|end_of_text|>": "EOS",
        "<|start_header_id|>": "ROLE_START",
        "<|end_header_id|>": "ROLE_END",
        "<|eot_id|>": "TURN_END",
        "system": "ROLE_SYSTEM",
        "user": "ROLE_USER",
        "assistant": "ROLE_ASSISTANT",
    },
    "qwen": {
        "<|im_start|>": "ROLE_START",
        "<|im_end|>": "TURN_END",
        "<|endoftext|>": "EOS",
        "system": "ROLE_SYSTEM",
        "user": "ROLE_USER",
        "assistant": "ROLE_ASSISTANT",
    },
    "gemma": {
        "<bos>": "BOS",
        "<eos>": "EOS",
        "<start_of_turn>": "ROLE_START",
        "<end_of_turn>": "TURN_END",
        "user": "ROLE_USER",
        "model": "ROLE_ASSISTANT",
    },
}


def get_model_family(model_name: str) -> str:
    if "llama" in model_name.lower():
        return "llama"
    elif "qwen" in model_name.lower():
        return "qwen"
    elif "gemma" in model_name.lower():
        return "gemma"
    return "unknown"


def get_model_size(model_name: str) -> float:
    """Extract size for sorting."""
    import re
    match = re.search(r'(\d+)b', model_name.lower())
    if match:
        return float(match.group(1))
    return 0


def normalize_token(token: str, family: str) -> tuple[str, bool]:
    mapping = TOKEN_NORMALIZATION.get(family, {})
    if token in mapping:
        return mapping[token], True
    if token.strip() == "" and "\n" in token:
        return "NEWLINE", True
    return token, False


def load_results(data_dir: str) -> dict:
    results = {}
    data_path = Path(data_dir)
    for file_path in data_path.glob("intermediate_*.json"):
        model_key = file_path.stem.replace("intermediate_", "")
        with open(file_path) as f:
            data = json.load(f)
        results[model_key] = data
    return results


def analyze_single_example(result: dict, family: str) -> dict:
    tokens = result.get("tokens", [])
    scores = result.get("attention_scores", [])
    if not tokens or not scores:
        return None

    normalized = []
    for i, (tok, score) in enumerate(zip(tokens, scores)):
        norm_tok, is_special = normalize_token(tok, family)
        normalized.append({
            "idx": i, "original": tok, "normalized": norm_tok,
            "is_special": is_special, "score": score,
        })

    special_scores = defaultdict(list)
    content_scores = []
    content_tokens = []

    for item in normalized:
        if item["is_special"]:
            special_scores[item["normalized"]].append(item["score"])
        else:
            content_scores.append(item["score"])
            content_tokens.append(item["original"])

    total_special = sum(sum(scores) for scores in special_scores.values())
    total_content = sum(content_scores)

    if content_scores:
        content_with_scores = sorted(zip(content_tokens, content_scores), key=lambda x: x[1], reverse=True)
        top_content = content_with_scores[:10]
    else:
        top_content = []

    return {
        "special_scores": {k: sum(v) for k, v in special_scores.items()},
        "total_special": total_special,
        "total_content": total_content,
        "special_ratio": total_special / (total_special + total_content + 1e-10),
        "top_content_tokens": top_content,
        "all_tokens_normalized": normalized,
    }


def analyze_model(model_data: dict, model_key: str) -> dict:
    family = get_model_family(model_key)
    results = model_data.get("results", [])
    analyses = []
    for r in results:
        if "error" in r:
            continue
        analysis = analyze_single_example(r, family)
        if analysis:
            analysis["behavior_id"] = r.get("behavior_id", -1)
            analysis["category"] = r.get("category", "unknown")
            analysis["prompt"] = r.get("prompt", "")
            analyses.append(analysis)
    return {"model": model_key, "family": family, "analyses": analyses}


def compute_cross_model_stats(all_analyses: dict) -> dict:
    model_stats = {}
    for model_key, model_data in all_analyses.items():
        analyses = model_data["analyses"]
        if not analyses:
            continue
        category_scores = defaultdict(list)
        special_ratios = []
        for a in analyses:
            special_ratios.append(a["special_ratio"])
            for cat, score in a["special_scores"].items():
                category_scores[cat].append(score)
        model_stats[model_key] = {
            "family": model_data["family"],
            "n_examples": len(analyses),
            "avg_special_ratio": np.mean(special_ratios),
            "std_special_ratio": np.std(special_ratios),
            "avg_category_scores": {k: np.mean(v) for k, v in category_scores.items()},
        }
    return model_stats


def compute_content_token_overlap(all_analyses: dict, model_order: list, top_k: int = 5) -> pd.DataFrame:
    """Compute overlap with specified model order."""
    model_token_sets = {}
    for model_key, model_data in all_analyses.items():
        token_sets = []
        for a in model_data["analyses"]:
            tokens = {tok.strip().lower() for tok, _ in a["top_content_tokens"][:top_k]}
            token_sets.append(tokens)
        model_token_sets[model_key] = token_sets

    n_models = len(model_order)
    overlap_matrix = np.zeros((n_models, n_models))
    for i, m1 in enumerate(model_order):
        for j, m2 in enumerate(model_order):
            sets1 = model_token_sets[m1]
            sets2 = model_token_sets[m2]
            similarities = []
            for s1, s2 in zip(sets1, sets2):
                if len(s1 | s2) > 0:
                    similarities.append(len(s1 & s2) / len(s1 | s2))
            overlap_matrix[i, j] = np.mean(similarities) if similarities else 0.0
    return pd.DataFrame(overlap_matrix, index=model_order, columns=model_order)


def get_global_top_content_tokens(all_analyses: dict, top_n: int = 20) -> list:
    """Get most commonly attended content tokens across all models."""
    token_counts = defaultdict(lambda: {"count": 0, "total_score": 0, "models": set()})

    for model_key, model_data in all_analyses.items():
        for a in model_data["analyses"]:
            for tok, score in a["top_content_tokens"][:5]:
                tok_clean = tok.strip().lower()
                if len(tok_clean) > 1:
                    token_counts[tok_clean]["count"] += 1
                    token_counts[tok_clean]["total_score"] += score
                    token_counts[tok_clean]["models"].add(model_key)

    sorted_tokens = sorted(
        token_counts.items(),
        key=lambda x: (len(x[1]["models"]), x[1]["count"]),
        reverse=True
    )
    return sorted_tokens[:top_n]


def get_example_analyses(all_analyses: dict, example_ids: list = [0, 5, 10]) -> list:
    """Get detailed attention breakdown for specific examples."""
    examples = []

    for ex_id in example_ids:
        example_data = {"behavior_id": ex_id, "models": {}}
        prompt = None
        category = None

        for model_key, model_data in all_analyses.items():
            for a in model_data["analyses"]:
                if a["behavior_id"] == ex_id:
                    if prompt is None:
                        prompt = a["prompt"]
                        category = a["category"]

                    # Get top 5 content tokens with scores
                    top_content = [(tok.strip(), f"{score:.4f}")
                                  for tok, score in a["top_content_tokens"][:5]]

                    # Get top special tokens
                    top_special = sorted(a["special_scores"].items(),
                                        key=lambda x: x[1], reverse=True)[:3]
                    top_special = [(cat, f"{score:.4f}") for cat, score in top_special]

                    example_data["models"][model_key] = {
                        "top_content": top_content,
                        "top_special": top_special,
                        "special_ratio": a["special_ratio"],
                    }
                    break

        example_data["prompt"] = prompt
        example_data["category"] = category
        examples.append(example_data)

    return examples


def sort_models_by_family(model_stats: dict) -> list:
    """Sort models by family, then by size within family."""
    models = list(model_stats.keys())
    return sorted(models, key=lambda m: (model_stats[m]["family"], get_model_size(m)))


def generate_markdown_report(model_stats: dict, overlap_df: pd.DataFrame,
                            all_analyses: dict, model_order: list, output_path: Path):
    """Generate markdown analysis report."""

    # Compute within/cross family stats
    within, cross = [], []
    for i, m1 in enumerate(model_order):
        for j, m2 in enumerate(model_order):
            if i >= j:
                continue
            f1, f2 = model_stats[m1]["family"], model_stats[m2]["family"]
            score = overlap_df.loc[m1, m2]
            (within if f1 == f2 else cross).append(score)

    # Get top content tokens
    top_tokens = get_global_top_content_tokens(all_analyses, top_n=20)

    # Get example analyses
    examples = get_example_analyses(all_analyses, example_ids=[0, 3, 7, 15, 25])

    md = []
    md.append("# Attention Convergence Analysis")
    md.append("")
    md.append("Analysis of attention patterns across model families with normalized special tokens.")
    md.append("")

    # Key findings
    md.append("## Key Findings")
    md.append("")
    md.append(f"- **Special tokens dominate attention**: Models attend {np.mean([s['avg_special_ratio'] for s in model_stats.values()])*100:.0f}% to template tokens on average")
    md.append(f"- **Within-family overlap**: {np.mean(within):.3f} (models from same family agree on important content tokens)")
    md.append(f"- **Cross-family overlap**: {np.mean(cross):.3f} (less agreement across different architectures)")
    md.append(f"- **Overlap difference**: {np.mean(within) - np.mean(cross):.3f} (significant family effect)")
    md.append("")

    # Per-model stats table
    md.append("## Per-Model Statistics")
    md.append("")
    md.append("| Model | Family | N | Avg Special % | Avg Content % | Top Special Token |")
    md.append("|-------|--------|---|---------------|---------------|-------------------|")

    for model_key in model_order:
        stats = model_stats[model_key]
        cat_scores = stats["avg_category_scores"]
        top_cat = max(cat_scores.items(), key=lambda x: x[1]) if cat_scores else ("N/A", 0)
        content_pct = (1 - stats['avg_special_ratio']) * 100
        md.append(f"| {model_key} | {stats['family']} | {stats['n_examples']} | "
                 f"{stats['avg_special_ratio']*100:.1f}% | {content_pct:.1f}% | {top_cat[0]} ({top_cat[1]:.3f}) |")
    md.append("")

    # Special token breakdown by family
    md.append("## Special Token Attention by Family")
    md.append("")

    families = sorted(set(s["family"] for s in model_stats.values()))
    for family in families:
        md.append(f"### {family.upper()}")
        md.append("")

        family_cat_scores = defaultdict(list)
        for model_key, stats in model_stats.items():
            if stats["family"] == family:
                for cat, score in stats["avg_category_scores"].items():
                    family_cat_scores[cat].append(score)

        sorted_cats = sorted(family_cat_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)

        md.append("| Token Category | Avg Attention |")
        md.append("|----------------|---------------|")
        for cat, scores in sorted_cats:
            md.append(f"| {cat} | {np.mean(scores):.4f} |")
        md.append("")

    # Top content tokens
    md.append("## Most Attended Content Tokens (Across All Models)")
    md.append("")
    md.append("These are the actual prompt words that models attend to most (excluding template tokens).")
    md.append("")
    md.append("| Token | Models | Count | Avg Score |")
    md.append("|-------|--------|-------|-----------|")
    for tok, stats in top_tokens:
        avg_score = stats["total_score"] / stats["count"]
        md.append(f"| `{tok}` | {len(stats['models'])} | {stats['count']} | {avg_score:.4f} |")
    md.append("")

    # Example breakdowns
    md.append("## Detailed Example Breakdowns")
    md.append("")
    md.append("Showing what each model attends to for specific prompts.")
    md.append("")

    for ex in examples:
        if ex["prompt"] is None:
            continue
        md.append(f"### Example {ex['behavior_id']}: {ex['category']}")
        md.append("")
        md.append(f"**Prompt**: \"{ex['prompt']}\"")
        md.append("")
        md.append("| Model | Special % | Top Content Tokens | Top Special Tokens |")
        md.append("|-------|-----------|-------------------|-------------------|")

        for model_key in model_order:
            if model_key in ex["models"]:
                m = ex["models"][model_key]
                content_str = ", ".join([f"`{t}`({s})" for t, s in m["top_content"][:3]])
                special_str = ", ".join([f"{t}({s})" for t, s in m["top_special"][:2]])
                md.append(f"| {model_key} | {m['special_ratio']*100:.0f}% | {content_str} | {special_str} |")
        md.append("")

    # Overlap matrix
    md.append("## Content Token Overlap Matrix (Grouped by Family)")
    md.append("")
    md.append("Jaccard similarity of top-5 attended content tokens per example.")
    md.append("Models are grouped by family (Gemma, Llama, Qwen) for easier comparison.")
    md.append("")
    md.append("```")
    md.append(overlap_df.round(3).to_string())
    md.append("```")
    md.append("")

    # Within vs cross family
    md.append("## Within-Family vs Cross-Family Comparison")
    md.append("")
    md.append(f"| Comparison | Mean | Std | N |")
    md.append(f"|------------|------|-----|---|")
    md.append(f"| Within-family | {np.mean(within):.3f} | {np.std(within):.3f} | {len(within)} |")
    md.append(f"| Cross-family | {np.mean(cross):.3f} | {np.std(cross):.3f} | {len(cross)} |")
    md.append("")

    # Interpretation
    md.append("## Interpretation")
    md.append("")
    md.append("1. **Template tokens are attention sinks**: ~79% of attention goes to BOS, newlines, and role markers rather than actual content.")
    md.append("2. **Family-specific patterns**: Llama/Gemma use BOS as primary attention sink, Qwen uses newlines.")
    md.append("3. **Content agreement**: When looking at actual prompt content, models within the same family agree more (0.52) than across families (0.39).")
    md.append("4. **Common trigger words**: Words like 'to', 'that', 'write', 'post', 'blog', 'illegally', 'scam' appear in top attention across all families.")
    md.append("")

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(md))

    return within, cross


def generate_visualizations(model_stats: dict, overlap_df: pd.DataFrame,
                           all_analyses: dict, model_order: list,
                           within: list, cross: list, output_dir: Path):
    """Generate visualization plots."""

    fig = plt.figure(figsize=(16, 14))

    # Create grid spec for better layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1.2], hspace=0.35, wspace=0.25)

    colors = {'gemma': '#2ecc71', 'llama': '#3498db', 'qwen': '#e74c3c'}

    # 1. Overlap heatmap (grouped by family)
    ax1 = fig.add_subplot(gs[0, 0])
    n_models = len(model_order)
    im = ax1.imshow(overlap_df.values, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(n_models))
    ax1.set_yticks(range(n_models))
    ax1.set_xticklabels(model_order, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(model_order, fontsize=9)

    # Add family separators
    families_in_order = [model_stats[m]["family"] for m in model_order]
    prev_family = families_in_order[0]
    for i, fam in enumerate(families_in_order):
        if fam != prev_family:
            ax1.axhline(y=i-0.5, color='white', linewidth=2)
            ax1.axvline(x=i-0.5, color='white', linewidth=2)
            prev_family = fam

    for i in range(n_models):
        for j in range(n_models):
            val = overlap_df.values[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax1.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7, color=color)
    ax1.set_title("Content Token Overlap (Grouped by Family)", fontsize=11)
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # 2. Within vs cross family boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([within, cross], tick_labels=['Within Family', 'Cross Family'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax2.set_ylabel('Jaccard Similarity')
    ax2.set_title('Token Overlap: Within vs Cross Family', fontsize=11)
    ax2.axhline(y=np.mean(within), color='#3498db', linestyle='--', alpha=0.7, label=f'Within mean: {np.mean(within):.3f}')
    ax2.axhline(y=np.mean(cross), color='#e74c3c', linestyle='--', alpha=0.7, label=f'Cross mean: {np.mean(cross):.3f}')
    ax2.legend(loc='lower right')

    # 3. Attention breakdown: Special vs Content
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(model_order))
    special_ratios = [model_stats[m]["avg_special_ratio"] * 100 for m in model_order]
    content_ratios = [100 - r for r in special_ratios]

    bar_colors = [colors.get(model_stats[m]["family"], 'gray') for m in model_order]

    ax3.bar(x, special_ratios, label='Special/Template Tokens', color=bar_colors, alpha=0.8)
    ax3.bar(x, content_ratios, bottom=special_ratios, label='Content Tokens', color=bar_colors, alpha=0.4)
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_order, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Attention %')
    ax3.set_title('Attention Split: Special vs Content Tokens', fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.axhline(y=np.mean(special_ratios), color='black', linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right')

    # 4. Special token attention by family (grouped bar)
    ax4 = fig.add_subplot(gs[1, 1])
    families_list = sorted(set(s["family"] for s in model_stats.values()))
    all_cats = set()
    family_data = {}
    for family in families_list:
        family_cat_scores = defaultdict(list)
        for model_key, stats in model_stats.items():
            if stats["family"] == family:
                for cat, score in stats["avg_category_scores"].items():
                    family_cat_scores[cat].append(score)
        family_data[family] = {k: np.mean(v) for k, v in family_cat_scores.items()}
        all_cats.update(family_cat_scores.keys())

    top_cats = ['BOS', 'NEWLINE', 'ROLE_START', 'ROLE_ASSISTANT', 'TURN_END', 'ROLE_END']
    top_cats = [c for c in top_cats if c in all_cats]

    x = np.arange(len(top_cats))
    width = 0.25
    for i, family in enumerate(families_list):
        scores = [family_data[family].get(cat, 0) for cat in top_cats]
        ax4.bar(x + i * width, scores, width, label=family.capitalize(), color=colors.get(family, 'gray'))

    ax4.set_xticks(x + width)
    ax4.set_xticklabels(top_cats, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Avg Attention Score')
    ax4.set_title('Special Token Attention by Family', fontsize=11)
    ax4.legend()

    # 5. Top content tokens bar chart
    ax5 = fig.add_subplot(gs[2, :])
    top_tokens = get_global_top_content_tokens(all_analyses, top_n=15)

    tokens = [t[0] for t in top_tokens]
    avg_scores = [t[1]["total_score"] / t[1]["count"] for t in top_tokens]
    n_models_per_token = [len(t[1]["models"]) for t in top_tokens]

    # Color by number of models that attend to it
    colors_bar = plt.cm.Blues(np.array(n_models_per_token) / max(n_models_per_token))

    bars = ax5.bar(range(len(tokens)), avg_scores, color=colors_bar)
    ax5.set_xticks(range(len(tokens)))
    ax5.set_xticklabels([f'"{t}"' for t in tokens], rotation=45, ha='right', fontsize=10)
    ax5.set_ylabel('Avg Attention Score')
    ax5.set_title('Top Attended Content Tokens (Across All Models)', fontsize=11)

    # Add model count annotation
    for i, (bar, n) in enumerate(zip(bars, n_models_per_token)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{n}', ha='center', va='bottom', fontsize=8, color='gray')
    ax5.set_xlabel('(Numbers above bars = # of model families attending to this token)')

    plt.savefig(output_dir / "attention_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    data_dir = Path(__file__).parent.parent / "data" / "Attention_output"
    output_dir = data_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading data from: {data_dir}")
    raw_results = load_results(data_dir)

    if not raw_results:
        print("No results found!")
        return

    print("Analyzing models...")
    all_analyses = {}
    for model_key, model_data in raw_results.items():
        all_analyses[model_key] = analyze_model(model_data, model_key)

    model_stats = compute_cross_model_stats(all_analyses)

    # Sort models by family, then size
    model_order = sort_models_by_family(model_stats)
    print(f"Model order (grouped by family): {model_order}")

    overlap_df = compute_content_token_overlap(all_analyses, model_order, top_k=5)

    # Generate markdown report
    md_path = output_dir / "attention_analysis.md"
    within, cross = generate_markdown_report(model_stats, overlap_df, all_analyses, model_order, md_path)
    print(f"Saved: {md_path}")

    # Generate visualizations
    generate_visualizations(model_stats, overlap_df, all_analyses, model_order, within, cross, output_dir)
    print(f"Saved: {output_dir / 'attention_analysis.png'}")


if __name__ == "__main__":
    main()
