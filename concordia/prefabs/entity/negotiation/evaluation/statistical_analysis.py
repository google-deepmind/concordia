# Statistical Analysis Utilities for Paper-Worthy Evaluation
# Provides significance testing, effect sizes, and power analysis

"""
Statistical analysis tools aligned with research methodology:
- Effect size calculations (Cohen's d)
- Confidence intervals
- Power analysis
- ANOVA for multiple comparisons
- Result visualization helpers
"""

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_interpretation: str
    significant: bool
    confidence_interval: Tuple[float, float]
    sample_sizes: Tuple[int, int]
    notes: str = ""


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean1 = statistics.mean(group1)
    mean2 = statistics.mean(group2)

    var1 = statistics.variance(group1) if n1 > 1 else 0
    var2 = statistics.variance(group2) if n2 > 1 else 0

    # Pooled standard deviation
    if n1 + n2 - 2 <= 0:
        return 0.0

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def welchs_t_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """
    Welch's t-test for unequal variances.
    Returns (t-statistic, approximate p-value).
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return (0.0, 1.0)

    mean1 = statistics.mean(group1)
    mean2 = statistics.mean(group2)
    var1 = statistics.variance(group1)
    var2 = statistics.variance(group2)

    # Standard error of difference
    se = math.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return (0.0, 1.0)

    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var1/n1 + var2/n2)**2
    denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = num / denom if denom > 0 else 1

    # Approximate p-value using normal distribution for large df
    # (proper implementation would use t-distribution)
    z = abs(t_stat)
    p_value = 2 * (1 - _normal_cdf(z))

    return (t_stat, p_value)


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for the mean."""
    n = len(data)
    if n < 2:
        return (data[0], data[0]) if n == 1 else (0.0, 0.0)

    mean = statistics.mean(data)
    se = statistics.stdev(data) / math.sqrt(n)

    # Z-score for confidence level (approximation)
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    margin = z * se
    return (mean - margin, mean + margin)


def compare_conditions(
    group1: List[float],
    group2: List[float],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    alpha: float = 0.05
) -> StatisticalResult:
    """
    Complete statistical comparison between two conditions.
    """
    n1, n2 = len(group1), len(group2)

    # Calculate means and standard deviations
    mean1 = statistics.mean(group1) if group1 else 0
    mean2 = statistics.mean(group2) if group2 else 0

    # Effect size
    d = cohens_d(group1, group2)
    d_interp = interpret_cohens_d(d)

    # Statistical test
    t_stat, p_value = welchs_t_test(group1, group2)

    # Confidence interval for difference
    if n1 >= 2 and n2 >= 2:
        diff_se = math.sqrt(
            statistics.variance(group1)/n1 +
            statistics.variance(group2)/n2
        )
        diff_ci = (
            mean1 - mean2 - 1.96 * diff_se,
            mean1 - mean2 + 1.96 * diff_se
        )
    else:
        diff_ci = (mean1 - mean2, mean1 - mean2)

    return StatisticalResult(
        test_name="Welch's t-test",
        statistic=t_stat,
        p_value=p_value,
        effect_size=d,
        effect_interpretation=d_interp,
        significant=p_value < alpha,
        confidence_interval=diff_ci,
        sample_sizes=(n1, n2),
        notes=f"Comparing {group1_name} (M={mean1:.3f}) vs {group2_name} (M={mean2:.3f})"
    )


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Calculate required sample size per group for given effect size and power.
    Uses approximation for two-sample t-test.
    """
    if effect_size == 0:
        return float('inf')

    # Z-scores for alpha and beta
    z_alpha = 1.96  # two-tailed alpha = 0.05
    z_beta = 0.84   # power = 0.80

    if alpha == 0.01:
        z_alpha = 2.576
    elif alpha == 0.10:
        z_alpha = 1.645

    if power == 0.90:
        z_beta = 1.28
    elif power == 0.95:
        z_beta = 1.645

    # Sample size formula
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return math.ceil(n)


def required_trials_table() -> Dict[str, int]:
    """Generate table of required trials for different effect sizes."""
    effect_sizes = {
        'small (d=0.2)': 0.2,
        'small-medium (d=0.35)': 0.35,
        'medium (d=0.5)': 0.5,
        'medium-large (d=0.65)': 0.65,
        'large (d=0.8)': 0.8,
    }

    return {
        name: power_analysis(d)
        for name, d in effect_sizes.items()
    }


def one_way_anova(groups: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    One-way ANOVA for comparing multiple groups.
    Returns F-statistic, p-value approximation, and eta-squared.
    """
    if len(groups) < 2:
        return {'error': 'Need at least 2 groups'}

    all_values = []
    group_means = {}
    group_ns = {}

    for name, values in groups.items():
        all_values.extend(values)
        group_means[name] = statistics.mean(values) if values else 0
        group_ns[name] = len(values)

    grand_mean = statistics.mean(all_values) if all_values else 0
    n_total = len(all_values)
    k = len(groups)  # Number of groups

    # Between-group sum of squares
    ss_between = sum(
        n * (mean - grand_mean) ** 2
        for name, mean in group_means.items()
        for n in [group_ns[name]]
    )

    # Within-group sum of squares
    ss_within = sum(
        (x - group_means[name]) ** 2
        for name, values in groups.items()
        for x in values
    )

    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k

    if df_between <= 0 or df_within <= 0 or ss_within == 0:
        return {'error': 'Insufficient data for ANOVA'}

    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # F-statistic
    f_stat = ms_between / ms_within

    # Effect size (eta-squared)
    eta_squared = ss_between / (ss_between + ss_within)

    # Approximate p-value (rough approximation)
    # Proper implementation would use F-distribution
    p_value = 1 / (1 + f_stat) ** 0.5  # Very rough approximation

    return {
        'f_statistic': f_stat,
        'p_value_approx': p_value,
        'eta_squared': eta_squared,
        'eta_interpretation': interpret_eta_squared(eta_squared),
        'df_between': df_between,
        'df_within': df_within,
        'group_means': group_means,
        'group_ns': group_ns
    }


def interpret_eta_squared(eta_sq: float) -> str:
    """Interpret eta-squared effect size."""
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


def pairwise_comparisons(
    groups: Dict[str, List[float]],
    correction: str = "bonferroni"
) -> List[StatisticalResult]:
    """
    Perform pairwise comparisons between all groups.
    Applies multiple comparison correction.
    """
    results = []
    group_names = list(groups.keys())
    n_comparisons = len(group_names) * (len(group_names) - 1) // 2

    for i, name1 in enumerate(group_names):
        for name2 in group_names[i+1:]:
            result = compare_conditions(
                groups[name1],
                groups[name2],
                name1,
                name2
            )

            # Apply Bonferroni correction
            if correction == "bonferroni":
                corrected_alpha = 0.05 / n_comparisons
                result.significant = result.p_value < corrected_alpha
                result.notes += f" (Bonferroni corrected, α={corrected_alpha:.4f})"

            results.append(result)

    return results


def generate_results_table(
    results: Dict[str, Dict[str, float]],
    baseline_key: str = "baseline"
) -> str:
    """
    Generate a formatted results table for paper.

    results: {condition_name: {metric_name: value, ...}, ...}
    """
    if not results:
        return "No results to display"

    # Get all metrics
    metrics = set()
    for condition_data in results.values():
        metrics.update(condition_data.keys())
    metrics = sorted(metrics)

    # Header
    header = ["Condition"] + list(metrics)
    col_widths = [max(len(h), 15) for h in header]

    # Build table
    lines = []
    lines.append(" | ".join(h.ljust(w) for h, w in zip(header, col_widths)))
    lines.append("-|-".join("-" * w for w in col_widths))

    # Get baseline values for comparison
    baseline = results.get(baseline_key, {})

    for condition, data in results.items():
        row = [condition[:col_widths[0]]]
        for i, metric in enumerate(metrics):
            value = data.get(metric, 0)
            baseline_val = baseline.get(metric, 0)

            if condition != baseline_key and baseline_val != 0:
                diff = (value - baseline_val) / baseline_val * 100
                cell = f"{value:.3f} ({diff:+.1f}%)"
            else:
                cell = f"{value:.3f}"

            row.append(cell.ljust(col_widths[i+1]))

        lines.append(" | ".join(row))

    return "\n".join(lines)


def summarize_for_paper(
    experiments: Dict[str, Any],
    main_metric: str = "social_welfare"
) -> str:
    """
    Generate paper-ready summary of results.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PAPER-READY RESULTS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Required sample size info
    lines.append("STATISTICAL POWER ANALYSIS")
    lines.append("-" * 40)
    trials_table = required_trials_table()
    for effect, n in trials_table.items():
        lines.append(f"  {effect}: n = {n} per group")
    lines.append("")

    # Extract data for comparison
    conditions = {}
    for exp_name, exp_data in experiments.items():
        if hasattr(exp_data, 'trials'):
            values = [t.social_welfare_score for t in exp_data.trials]
            conditions[exp_name] = values

    if len(conditions) >= 2:
        lines.append("OMNIBUS TEST (ANOVA)")
        lines.append("-" * 40)
        anova = one_way_anova(conditions)
        if 'error' not in anova:
            lines.append(f"  F({anova['df_between']}, {anova['df_within']}) = {anova['f_statistic']:.2f}")
            lines.append(f"  η² = {anova['eta_squared']:.3f} ({anova['eta_interpretation']} effect)")
            lines.append("")

            lines.append("PAIRWISE COMPARISONS")
            lines.append("-" * 40)
            comparisons = pairwise_comparisons(conditions)
            for comp in comparisons:
                sig_marker = "*" if comp.significant else ""
                lines.append(
                    f"  {comp.notes}: d = {comp.effect_size:.2f} "
                    f"({comp.effect_interpretation}){sig_marker}"
                )
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


class ResultsAnalyzer:
    """Class for comprehensive analysis of experiment results."""

    def __init__(self, experiments: Dict[str, Any]):
        self.experiments = experiments
        self.conditions: Dict[str, List[float]] = {}
        self._extract_data()

    def _extract_data(self):
        """Extract numerical data from experiments."""
        for name, exp in self.experiments.items():
            if hasattr(exp, 'trials'):
                self.conditions[name] = [
                    t.social_welfare_score for t in exp.trials
                ]

    def full_analysis(self) -> Dict[str, Any]:
        """Run complete statistical analysis."""
        analysis = {
            'summary_statistics': {},
            'effect_sizes': {},
            'significance_tests': {},
            'power_achieved': {},
            'recommendations': []
        }

        # Summary statistics for each condition
        for name, values in self.conditions.items():
            if values:
                analysis['summary_statistics'][name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'n': len(values),
                    'ci_95': confidence_interval(values),
                }

        # Find baseline
        baseline_name = next((n for n in self.conditions if 'baseline' in n.lower()), None)
        if not baseline_name:
            baseline_name = list(self.conditions.keys())[0] if self.conditions else None

        if baseline_name and len(self.conditions) > 1:
            baseline_values = self.conditions[baseline_name]

            # Compare each condition to baseline
            for name, values in self.conditions.items():
                if name != baseline_name:
                    result = compare_conditions(
                        values, baseline_values,
                        name, baseline_name
                    )
                    analysis['effect_sizes'][name] = result.effect_size
                    analysis['significance_tests'][name] = {
                        'p_value': result.p_value,
                        'significant': result.significant,
                        'interpretation': result.effect_interpretation
                    }

                    # Power achieved
                    n = len(values)
                    if result.effect_size != 0:
                        required_n = power_analysis(abs(result.effect_size))
                        analysis['power_achieved'][name] = {
                            'actual_n': n,
                            'required_n': required_n,
                            'adequate': n >= required_n
                        }

        # Recommendations
        if analysis['power_achieved']:
            underpowered = [
                name for name, power in analysis['power_achieved'].items()
                if not power['adequate']
            ]
            if underpowered:
                max_required = max(
                    analysis['power_achieved'][n]['required_n']
                    for n in underpowered
                )
                analysis['recommendations'].append(
                    f"Increase trials to {max_required} for adequate power in: {', '.join(underpowered)}"
                )

        return analysis

    def generate_latex_table(self) -> str:
        """Generate LaTeX-formatted results table."""
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Experimental Results: Social Welfare by Condition}",
            "\\label{tab:results}",
            "\\begin{tabular}{lccc}",
            "\\hline",
            "Condition & Mean (SD) & Cohen's d & p-value \\\\",
            "\\hline"
        ]

        # Find baseline
        baseline_name = next((n for n in self.conditions if 'baseline' in n.lower()), None)
        baseline_values = self.conditions.get(baseline_name, []) if baseline_name else []

        for name, values in self.conditions.items():
            if not values:
                continue

            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0

            if name == baseline_name:
                lines.append(f"{name} & {mean:.3f} ({std:.3f}) & -- & -- \\\\")
            elif baseline_values:
                d = cohens_d(values, baseline_values)
                _, p = welchs_t_test(values, baseline_values)
                sig = "*" if p < 0.05 else ""
                lines.append(f"{name} & {mean:.3f} ({std:.3f}) & {d:.2f} & {p:.3f}{sig} \\\\")

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return "\n".join(lines)
