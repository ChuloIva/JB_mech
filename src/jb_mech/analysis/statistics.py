"""
Statistical analysis utilities for jailbreak mechanism research.

Provides functions for:
- Comparing distributions between groups
- Computing effect sizes
- Testing hypotheses about jailbreak mechanisms
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats


class StatisticalAnalyzer:
    """Statistical analysis utilities for jailbreak research."""

    @staticmethod
    def compare_distributions(
        group1: Union[torch.Tensor, np.ndarray],
        group2: Union[torch.Tensor, np.ndarray],
        test_type: str = "mannwhitneyu",
        alternative: str = "two-sided",
    ) -> Dict:
        """
        Compare two distributions with appropriate statistical tests.

        Args:
            group1: First group values
            group2: Second group values
            test_type: "mannwhitneyu", "ttest", or "ks"
            alternative: "two-sided", "less", or "greater"

        Returns:
            Dictionary with test results:
                - group1_mean, group1_std
                - group2_mean, group2_std
                - statistic, p_value
                - significant (at alpha=0.05)
                - effect_size (Cohen's d)
        """
        # Convert to numpy
        g1 = group1.cpu().numpy() if isinstance(group1, torch.Tensor) else np.asarray(group1)
        g2 = group2.cpu().numpy() if isinstance(group2, torch.Tensor) else np.asarray(group2)

        results = {
            "group1_n": len(g1),
            "group2_n": len(g2),
            "group1_mean": float(g1.mean()),
            "group1_std": float(g1.std()),
            "group2_mean": float(g2.mean()),
            "group2_std": float(g2.std()),
        }

        # Run appropriate test
        if test_type == "mannwhitneyu":
            stat, p = stats.mannwhitneyu(g1, g2, alternative=alternative)
            results["test_name"] = "Mann-Whitney U"
        elif test_type == "ttest":
            stat, p = stats.ttest_ind(g1, g2)
            if alternative != "two-sided":
                p = p / 2
                if alternative == "greater" and stat < 0:
                    p = 1 - p
                elif alternative == "less" and stat > 0:
                    p = 1 - p
            results["test_name"] = "Independent t-test"
        elif test_type == "ks":
            stat, p = stats.ks_2samp(g1, g2)
            results["test_name"] = "Kolmogorov-Smirnov"
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        results["statistic"] = float(stat)
        results["p_value"] = float(p)
        results["significant_05"] = p < 0.05
        results["significant_01"] = p < 0.01
        results["significant_001"] = p < 0.001

        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(g1) - 1) * g1.std() ** 2 + (len(g2) - 1) * g2.std() ** 2)
            / (len(g1) + len(g2) - 2)
        )
        cohens_d = (g1.mean() - g2.mean()) / (pooled_std + 1e-8)
        results["effect_size_cohens_d"] = float(cohens_d)

        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        results["effect_interpretation"] = interpretation

        return results

    @staticmethod
    def correlation_analysis(
        data: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> pd.DataFrame:
        """
        Compute correlation matrix between multiple variables.

        Args:
            data: Dictionary mapping variable_name -> values

        Returns:
            Pandas DataFrame with correlation matrix
        """
        # Convert all to numpy
        np_data = {}
        for name, values in data.items():
            if isinstance(values, torch.Tensor):
                np_data[name] = values.cpu().numpy().flatten()
            else:
                np_data[name] = np.asarray(values).flatten()

        df = pd.DataFrame(np_data)
        return df.corr()

    @staticmethod
    def test_hypothesis_jailbreak_vs_clean(
        jailbreak_projections: Union[torch.Tensor, np.ndarray],
        clean_projections: Union[torch.Tensor, np.ndarray],
    ) -> Dict:
        """
        Test Hypothesis 1: Jailbreaks push away from Assistant Axis.

        Expected: Jailbreak projections < Clean projections

        Args:
            jailbreak_projections: Axis projections for successful jailbreaks
            clean_projections: Axis projections for clean (refused) prompts

        Returns:
            Dictionary with test results and interpretation
        """
        results = StatisticalAnalyzer.compare_distributions(
            jailbreak_projections,
            clean_projections,
            test_type="mannwhitneyu",
            alternative="less",  # Test if jailbreak < clean
        )

        # Add interpretation
        if results["significant_05"] and results["group1_mean"] < results["group2_mean"]:
            results["hypothesis_supported"] = True
            results["interpretation"] = (
                f"Hypothesis 1 SUPPORTED: Jailbreak projections ({results['group1_mean']:.3f}) "
                f"are significantly lower than clean projections ({results['group2_mean']:.3f}) "
                f"(p={results['p_value']:.4f}, d={results['effect_size_cohens_d']:.2f})"
            )
        else:
            results["hypothesis_supported"] = False
            results["interpretation"] = (
                f"Hypothesis 1 NOT SUPPORTED: No significant evidence that jailbreaks "
                f"have lower projections (p={results['p_value']:.4f})"
            )

        return results

    @staticmethod
    def test_hypothesis_harmful_persona(
        jailbreak_similarities: Dict[str, Union[torch.Tensor, np.ndarray]],
        clean_similarities: Dict[str, Union[torch.Tensor, np.ndarray]],
        harmful_personas: List[str],
    ) -> Dict:
        """
        Test Hypothesis 2: Jailbreaks activate harmful personas.

        Expected: Jailbreak similarity to harmful personas > Clean similarity

        Args:
            jailbreak_similarities: {persona_name: similarities} for jailbreaks
            clean_similarities: {persona_name: similarities} for clean prompts
            harmful_personas: List of harmful persona names

        Returns:
            Dictionary with test results for each persona
        """
        results = {
            "persona_tests": {},
            "overall_supported": False,
            "supporting_count": 0,
        }

        for persona in harmful_personas:
            if persona not in jailbreak_similarities:
                continue

            jb_sims = jailbreak_similarities[persona]
            clean_sims = clean_similarities[persona]

            test_result = StatisticalAnalyzer.compare_distributions(
                jb_sims,
                clean_sims,
                test_type="mannwhitneyu",
                alternative="greater",
            )

            results["persona_tests"][persona] = test_result

            if test_result["significant_05"] and test_result["group1_mean"] > test_result["group2_mean"]:
                results["supporting_count"] += 1

        # Overall interpretation
        total_personas = len(results["persona_tests"])
        if total_personas > 0:
            support_ratio = results["supporting_count"] / total_personas
            results["support_ratio"] = support_ratio
            results["overall_supported"] = support_ratio >= 0.5

            if results["overall_supported"]:
                results["interpretation"] = (
                    f"Hypothesis 2 SUPPORTED: {results['supporting_count']}/{total_personas} "
                    f"harmful personas show significantly higher activation in jailbreaks"
                )
            else:
                results["interpretation"] = (
                    f"Hypothesis 2 NOT SUPPORTED: Only {results['supporting_count']}/{total_personas} "
                    f"harmful personas show higher activation"
                )

        return results

    @staticmethod
    def test_hypothesis_orthogonal(
        jailbreak_direction: torch.Tensor,
        axis: torch.Tensor,
        layer: int,
        threshold: float = 0.3,
    ) -> Dict:
        """
        Test Hypothesis 4: Jailbreaks operate through orthogonal mechanism.

        Expected: Low correlation between jailbreak direction and axis.

        Args:
            jailbreak_direction: Computed jailbreak direction vector
            axis: Assistant axis
            layer: Layer to analyze
            threshold: Cosine similarity threshold for "orthogonal"

        Returns:
            Dictionary with analysis
        """
        jb_dir = jailbreak_direction.float()
        ax = axis[layer].float()

        # Cosine similarity
        jb_norm = jb_dir / (jb_dir.norm() + 1e-8)
        ax_norm = ax / (ax.norm() + 1e-8)
        cosine_sim = (jb_norm @ ax_norm).item()

        # Variance explained by axis
        projection = (jb_dir @ ax_norm).item()
        variance_explained = (projection ** 2) / ((jb_dir ** 2).sum().item() + 1e-8)

        results = {
            "cosine_similarity": cosine_sim,
            "variance_explained": variance_explained,
            "threshold": threshold,
        }

        if abs(cosine_sim) < threshold and variance_explained < threshold:
            results["hypothesis_supported"] = True
            results["interpretation"] = (
                f"Hypothesis 4 SUPPORTED: Jailbreak direction is largely orthogonal to axis "
                f"(cosine={cosine_sim:.3f}, variance_explained={variance_explained:.1%})"
            )
        else:
            results["hypothesis_supported"] = False
            results["interpretation"] = (
                f"Hypothesis 4 NOT SUPPORTED: Jailbreak direction has substantial alignment with axis "
                f"(cosine={cosine_sim:.3f}, variance_explained={variance_explained:.1%})"
            )

        return results

    @staticmethod
    def summarize_all_hypotheses(
        h1_result: Dict,
        h2_result: Dict,
        h3_result: Optional[Dict] = None,
        h4_result: Optional[Dict] = None,
    ) -> str:
        """
        Create a summary of all hypothesis tests.

        Args:
            h1_result: Results from test_hypothesis_jailbreak_vs_clean
            h2_result: Results from test_hypothesis_harmful_persona
            h3_result: Optional combined hypothesis results
            h4_result: Results from test_hypothesis_orthogonal

        Returns:
            Summary string
        """
        summary = ["=" * 60]
        summary.append("HYPOTHESIS TEST SUMMARY")
        summary.append("=" * 60)

        summary.append("\nHypothesis 1: Jailbreaks = Anti-Assistant")
        summary.append("-" * 40)
        summary.append(h1_result.get("interpretation", "Not tested"))

        summary.append("\nHypothesis 2: Jailbreaks = Harmful Persona Activation")
        summary.append("-" * 40)
        summary.append(h2_result.get("interpretation", "Not tested"))

        if h3_result:
            summary.append("\nHypothesis 3: Jailbreaks = Both")
            summary.append("-" * 40)
            summary.append(h3_result.get("interpretation", "Not tested"))

        if h4_result:
            summary.append("\nHypothesis 4: Jailbreaks = Orthogonal Mechanism")
            summary.append("-" * 40)
            summary.append(h4_result.get("interpretation", "Not tested"))

        summary.append("\n" + "=" * 60)
        summary.append("CONCLUSIONS")
        summary.append("=" * 60)

        # Determine which hypothesis is most supported
        supported = []
        if h1_result.get("hypothesis_supported"):
            supported.append("H1 (Anti-Assistant)")
        if h2_result.get("overall_supported"):
            supported.append("H2 (Harmful Persona)")
        if h4_result and h4_result.get("hypothesis_supported"):
            supported.append("H4 (Orthogonal)")

        if not supported:
            summary.append("No hypothesis strongly supported by the data.")
        elif len(supported) == 1:
            summary.append(f"Primary mechanism: {supported[0]}")
        else:
            summary.append(f"Multiple mechanisms supported: {', '.join(supported)}")
            summary.append("This suggests Hypothesis 3 (combination) may be correct.")

        return "\n".join(summary)
