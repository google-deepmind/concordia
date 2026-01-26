# MATS Probe Training and Analysis v2
# Enhanced version with:
#   - Per-scenario AUC tracking for cross-scenario generalization
#   - Model scale comparison (9B vs 27B)
#   - Difference-in-means baseline
#   - Better visualization
#
# Usage:
#   python train_probes.py --data ./experiment_outputs/merged_activations_*.pt
#   python train_probes.py --compare results_9b.json results_27b.json

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =============================================================================
# PROBE CLASSES
# =============================================================================

@dataclass
class ProbeResult:
    """Results from training a probe."""
    layer: int
    label_type: str  # "gm" or "agent"
    r2_score: float
    accuracy: float
    auc: float
    train_r2: float
    test_r2: float
    cross_val_scores: List[float]

    def to_dict(self) -> Dict:
        return {
            "layer": int(self.layer),
            "label_type": self.label_type,
            "r2_score": float(self.r2_score),
            "accuracy": float(self.accuracy),
            "auc": float(self.auc),
            "train_r2": float(self.train_r2),
            "test_r2": float(self.test_r2),
            "cross_val_mean": float(np.mean(self.cross_val_scores)) if self.cross_val_scores else 0.0,
            "cross_val_std": float(np.std(self.cross_val_scores)) if self.cross_val_scores else 0.0,
        }


def train_ridge_probe(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 10.0,
    use_pca: bool = True,
    n_components: int = 50,
) -> Tuple[Ridge, ProbeResult]:
    """Train a Ridge regression probe."""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Optional PCA to reduce overfitting
    if use_pca:
        n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_comp)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = X

    # Train probe
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)

    # Evaluate
    train_pred = probe.predict(X_train)
    test_pred = probe.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Cross-validation
    cv_scores = cross_val_score(Ridge(alpha=alpha), X_pca, y, cv=5, scoring='r2')

    # Binary metrics
    binary_pred = (test_pred > 0.5).astype(int)
    binary_true = (y_test > 0.5).astype(int)

    accuracy = accuracy_score(binary_true, binary_pred)

    try:
        auc = roc_auc_score(binary_true, test_pred)
    except ValueError:
        auc = 0.5

    result = ProbeResult(
        layer=-1,
        label_type="",
        r2_score=test_r2,
        accuracy=accuracy,
        auc=auc,
        train_r2=train_r2,
        test_r2=test_r2,
        cross_val_scores=cv_scores.tolist(),
    )

    return probe, result


def train_mass_mean_probe(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, ProbeResult]:
    """
    Train a mass-mean direction probe (Marks & Tegmark method).
    Often more robust than Ridge for binary concepts.
    """

    binary_y = (y > threshold).astype(bool)

    # Check we have both classes
    if binary_y.sum() == 0 or binary_y.sum() == len(binary_y):
        # All same class - return dummy
        return np.zeros(X.shape[1]), ProbeResult(
            layer=-1, label_type="", r2_score=0.0, accuracy=0.5,
            auc=0.5, train_r2=0.0, test_r2=0.0, cross_val_scores=[]
        )

    # Compute means
    honest_mean = X[~binary_y].mean(axis=0)
    deceptive_mean = X[binary_y].mean(axis=0)

    # Direction vector
    direction = deceptive_mean - honest_mean
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = np.zeros_like(direction)
    else:
        direction = direction / norm

    # Project all points onto direction
    projections = X @ direction

    # Evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    test_proj = X_test @ direction
    train_proj = X_train @ direction

    # Normalize projections
    proj_min, proj_max = projections.min(), projections.max()
    if proj_max - proj_min < 1e-8:
        test_pred = np.full_like(test_proj, 0.5)
        train_pred = np.full_like(train_proj, 0.5)
    else:
        test_pred = (test_proj - proj_min) / (proj_max - proj_min)
        train_pred = (train_proj - proj_min) / (proj_max - proj_min)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Binary metrics
    binary_pred = (test_pred > 0.5).astype(int)
    binary_true = (y_test > 0.5).astype(int)

    accuracy = accuracy_score(binary_true, binary_pred)

    try:
        auc = roc_auc_score(binary_true, test_pred)
    except ValueError:
        auc = 0.5

    result = ProbeResult(
        layer=-1,
        label_type="",
        r2_score=test_r2,
        accuracy=accuracy,
        auc=auc,
        train_r2=train_r2,
        test_r2=test_r2,
        cross_val_scores=[],
    )

    return direction, result


# =============================================================================
# SANITY CHECKS
# =============================================================================

def sanity_check_random_labels(
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int = 5,
) -> Dict[str, float]:
    """Sanity check: probes on shuffled labels should give R² near 0."""
    shuffle_r2s = []

    for seed in range(n_shuffles):
        np.random.seed(seed)
        y_shuffled = np.random.permutation(y)
        _, result = train_ridge_probe(X, y_shuffled)
        shuffle_r2s.append(result.r2_score)

    return {
        "mean_shuffled_r2": float(np.mean(shuffle_r2s)),
        "std_shuffled_r2": float(np.std(shuffle_r2s)),
        "max_shuffled_r2": float(np.max(shuffle_r2s)),
        "passed": bool(np.mean(shuffle_r2s) < 0.05),
    }


def sanity_check_train_test_gap(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """Check for overfitting."""
    _, result = train_ridge_probe(X, y)

    gap = result.train_r2 - result.test_r2

    return {
        "train_r2": float(result.train_r2),
        "test_r2": float(result.test_r2),
        "gap": float(gap),
        "passed": bool(gap < 0.2),
    }


def sanity_check_label_variance(y: np.ndarray) -> Dict[str, float]:
    """Check that labels have variance."""
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "n_unique": int(len(np.unique(y))),
        "passed": bool(np.std(y) > 0.1),
    }


# =============================================================================
# PER-LAYER PROBE COMPARISON
# =============================================================================

def train_probes_per_layer(
    activations_by_layer: Dict[int, np.ndarray],
    labels: np.ndarray,
    alpha: float = 10.0,
    use_pca: bool = True,
    n_components: int = 50,
) -> Dict[int, Dict[str, float]]:
    """
    Train separate probes on each layer to find optimal layer for deception detection.

    This is a critical validation step. Expected pattern from literature:
    - Early layers: Low accuracy (basic features, not semantic)
    - Middle layers: Peak accuracy (high-level concepts encoded)
    - Late layers: Lower accuracy (output formatting)

    If accuracy is flat across layers, something is wrong (probe using surface features).

    Args:
        activations_by_layer: Dict mapping layer_num -> activations [N, d_model]
        labels: Ground truth labels [N]
        alpha: Ridge regularization
        use_pca: Whether to apply PCA
        n_components: PCA components

    Returns:
        Dict mapping layer_num -> {auc, r2, accuracy, train_r2, test_r2}
    """
    results = {}

    for layer_num, X in sorted(activations_by_layer.items()):
        # Convert to numpy if tensor
        if hasattr(X, 'numpy'):
            X = X.cpu().numpy() if hasattr(X, 'cpu') else X.numpy()
        if hasattr(labels, 'numpy'):
            y = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels.numpy()
        else:
            y = np.array(labels)

        # Skip if not enough samples
        if len(y) < 10:
            results[layer_num] = {
                'auc': 0.5, 'r2': 0.0, 'accuracy': 0.5,
                'train_r2': 0.0, 'test_r2': 0.0, 'note': 'insufficient samples'
            }
            continue

        # Check label variance
        if np.std(y) < 0.01:
            results[layer_num] = {
                'auc': 0.5, 'r2': 0.0, 'accuracy': 0.5,
                'train_r2': 0.0, 'test_r2': 0.0, 'note': 'no label variance'
            }
            continue

        try:
            _, probe_result = train_ridge_probe(
                X, y, alpha=alpha, use_pca=use_pca, n_components=n_components
            )
            results[layer_num] = {
                'auc': probe_result.auc,
                'r2': probe_result.r2_score,
                'accuracy': probe_result.accuracy,
                'train_r2': probe_result.train_r2,
                'test_r2': probe_result.test_r2,
            }
        except Exception as e:
            results[layer_num] = {
                'auc': 0.5, 'r2': 0.0, 'accuracy': 0.5,
                'train_r2': 0.0, 'test_r2': 0.0, 'error': str(e)
            }

    return results


def find_best_layer(layer_results: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    """
    Find the best layer and analyze the layer accuracy curve.

    Returns:
        Dict with best_layer, peak_auc, curve_analysis, etc.
    """
    if not layer_results:
        return {'best_layer': None, 'error': 'No layer results'}

    # Find best by AUC
    best_layer = max(layer_results.keys(), key=lambda l: layer_results[l].get('auc', 0))
    best_auc = layer_results[best_layer]['auc']

    # Analyze curve shape
    layers = sorted(layer_results.keys())
    aucs = [layer_results[l]['auc'] for l in layers]

    # Check for expected inverted-U pattern
    n_layers = len(layers)
    if n_layers >= 3:
        first_third = aucs[:n_layers//3]
        middle_third = aucs[n_layers//3:2*n_layers//3]
        last_third = aucs[2*n_layers//3:]

        first_avg = np.mean(first_third) if first_third else 0
        middle_avg = np.mean(middle_third) if middle_third else 0
        last_avg = np.mean(last_third) if last_third else 0

        # Expected: middle > first and middle > last
        has_expected_shape = middle_avg > first_avg and middle_avg > last_avg
    else:
        has_expected_shape = None
        first_avg = middle_avg = last_avg = None

    # Check for flat curve (red flag)
    auc_std = np.std(aucs)
    is_flat = auc_std < 0.05  # If std < 5%, curve is suspiciously flat

    # Relative position of best layer (0=first, 1=last)
    if len(layers) > 1:
        relative_position = (best_layer - min(layers)) / (max(layers) - min(layers))
    else:
        relative_position = 0.5

    return {
        'best_layer': int(best_layer),
        'peak_auc': float(best_auc),
        'peak_r2': float(layer_results[best_layer].get('r2', 0)),
        'auc_std_across_layers': float(auc_std),
        'is_flat_curve': bool(is_flat),
        'has_expected_inverted_u': has_expected_shape,
        'relative_position': float(relative_position),
        'layer_aucs': {int(l): float(layer_results[l]['auc']) for l in layers},
        'analysis': {
            'first_third_avg_auc': float(first_avg) if first_avg is not None else None,
            'middle_third_avg_auc': float(middle_avg) if middle_avg is not None else None,
            'last_third_avg_auc': float(last_avg) if last_avg is not None else None,
        },
        'warnings': [
            'FLAT CURVE: Probe may be using surface features' if is_flat else None,
            'NO INVERTED-U: Unexpected layer pattern' if has_expected_shape is False else None,
            'EARLY PEAK: Best layer in first third' if relative_position < 0.33 else None,
            'LATE PEAK: Best layer in last third' if relative_position > 0.67 else None,
        ],
    }


def plot_layer_accuracy_curve(
    layer_results: Dict[int, Dict[str, float]],
    output_path: str = 'layer_accuracy_curve.png',
    title: str = 'Probe Accuracy by Layer',
) -> None:
    """
    Generate the standard layer accuracy curve plot.

    This is a key visualization for mechanistic interpretability papers.
    """
    layers = sorted(layer_results.keys())
    aucs = [layer_results[l]['auc'] for l in layers]
    r2s = [layer_results[l]['r2'] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # AUC plot
    ax1.plot(layers, aucs, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='gray', linestyle='--', label='Random (AUC=0.5)')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('AUC by Layer', fontsize=14)
    ax1.set_ylim(0.4, 1.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark best layer
    best_layer = max(layers, key=lambda l: layer_results[l]['auc'])
    best_auc = layer_results[best_layer]['auc']
    ax1.scatter([best_layer], [best_auc], color='red', s=200, zorder=5, marker='*')
    ax1.annotate(f'Best: L{best_layer}\nAUC={best_auc:.3f}',
                 xy=(best_layer, best_auc), xytext=(10, -20),
                 textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))

    # R² plot
    ax2.plot(layers, r2s, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=0.0, color='gray', linestyle='--', label='Random (R²=0)')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('R² by Layer', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved layer accuracy curve to {output_path}")


# =============================================================================
# GENERALIZATION ANALYSIS WITH AUC
# =============================================================================

def compute_generalization_auc(
    X: np.ndarray,
    y: np.ndarray,
    scenarios: List[str],
    alpha: float = 10.0,
) -> Dict[str, Any]:
    """
    Compute cross-scenario generalization using AUC (more robust than R²).

    For each scenario:
    - Train on all OTHER scenarios
    - Test on this scenario
    - Report both R² and AUC
    """
    unique_scenarios = list(set(scenarios))
    results = {}

    for holdout in unique_scenarios:
        # Split by scenario
        train_mask = np.array([s != holdout for s in scenarios])
        test_mask = ~train_mask

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        # Skip if test set has only one class
        if len(np.unique((y_test > 0.5).astype(int))) < 2:
            results[holdout] = {
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "test_r2": None,
                "test_auc": None,
                "deception_rate": float(np.mean(y_test)),
                "note": "Single class in test set",
            }
            continue

        # Apply PCA
        n_comp = min(50, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Train Ridge probe
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_pca, y_train)

        test_pred = probe.predict(X_test_pca)
        test_r2 = r2_score(y_test, test_pred)

        # Compute AUC
        binary_true = (y_test > 0.5).astype(int)
        try:
            test_auc = roc_auc_score(binary_true, test_pred)
        except ValueError:
            test_auc = 0.5

        results[holdout] = {
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
            "test_r2": float(test_r2),
            "test_auc": float(test_auc),
            "deception_rate": float(np.mean(y_test)),
        }

    # Compute averages (excluding None values)
    valid_r2s = [r["test_r2"] for r in results.values() if r["test_r2"] is not None]
    valid_aucs = [r["test_auc"] for r in results.values() if r["test_auc"] is not None]

    return {
        "by_scenario": results,
        "average_r2": float(np.mean(valid_r2s)) if valid_r2s else None,
        "average_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
        "std_r2": float(np.std(valid_r2s)) if valid_r2s else None,
        "std_auc": float(np.std(valid_aucs)) if valid_aucs else None,
    }


# =============================================================================
# PER-SCENARIO DECEPTION RATES
# =============================================================================

def compute_deception_rates(
    y: np.ndarray,
    scenarios: List[str],
) -> Dict[str, float]:
    """Compute deception rate per scenario."""
    unique_scenarios = list(set(scenarios))
    rates = {}

    for scenario in unique_scenarios:
        mask = np.array([s == scenario for s in scenarios])
        rates[scenario] = float(np.mean(y[mask]))

    return rates


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_model_scales(
    results_9b_path: str,
    results_27b_path: str,
) -> Dict[str, Any]:
    """Compare probe results across model scales."""

    with open(results_9b_path) as f:
        r9b = json.load(f)
    with open(results_27b_path) as f:
        r27b = json.load(f)

    comparison = {
        "9b": {},
        "27b": {},
        "deltas": {},
    }

    # Best probe R²
    comparison["9b"]["best_r2"] = r9b["best_probe"]["r2"]
    comparison["27b"]["best_r2"] = r27b["best_probe"]["r2"]
    comparison["deltas"]["best_r2"] = r27b["best_probe"]["r2"] - r9b["best_probe"]["r2"]

    # Best probe layer
    comparison["9b"]["best_layer"] = r9b["best_probe"]["layer"]
    comparison["27b"]["best_layer"] = r27b["best_probe"]["layer"]

    # GM vs Agent gap
    gap_9b = r9b["gm_vs_agent"]["gm_ridge_r2"] - r9b["gm_vs_agent"]["agent_ridge_r2"]
    gap_27b = r27b["gm_vs_agent"]["gm_ridge_r2"] - r27b["gm_vs_agent"]["agent_ridge_r2"]
    comparison["9b"]["gm_agent_gap"] = gap_9b
    comparison["27b"]["gm_agent_gap"] = gap_27b
    comparison["deltas"]["gm_agent_gap"] = gap_27b - gap_9b

    # GM AUC
    comparison["9b"]["gm_auc"] = r9b["gm_vs_agent"]["gm_auc"]
    comparison["27b"]["gm_auc"] = r27b["gm_vs_agent"]["gm_auc"]
    comparison["deltas"]["gm_auc"] = r27b["gm_vs_agent"]["gm_auc"] - r9b["gm_vs_agent"]["gm_auc"]

    # Generalization (if available)
    if "generalization" in r9b and "generalization" in r27b:
        if r9b["generalization"].get("average_r2") is not None:
            comparison["9b"]["cross_scenario_r2"] = r9b["generalization"]["average_r2"]
        if r27b["generalization"].get("average_r2") is not None:
            comparison["27b"]["cross_scenario_r2"] = r27b["generalization"]["average_r2"]
        if "cross_scenario_r2" in comparison["9b"] and "cross_scenario_r2" in comparison["27b"]:
            comparison["deltas"]["cross_scenario_r2"] = (
                comparison["27b"]["cross_scenario_r2"] - comparison["9b"]["cross_scenario_r2"]
            )

        if r9b["generalization"].get("average_auc") is not None:
            comparison["9b"]["cross_scenario_auc"] = r9b["generalization"]["average_auc"]
        if r27b["generalization"].get("average_auc") is not None:
            comparison["27b"]["cross_scenario_auc"] = r27b["generalization"]["average_auc"]
        if "cross_scenario_auc" in comparison["9b"] and "cross_scenario_auc" in comparison["27b"]:
            comparison["deltas"]["cross_scenario_auc"] = (
                comparison["27b"]["cross_scenario_auc"] - comparison["9b"]["cross_scenario_auc"]
            )

    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """Pretty-print model comparison."""
    print("\n" + "=" * 70)
    print("MODEL SCALE COMPARISON: Gemma 9B vs 27B")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'9B':<12} {'27B':<12} {'Delta':<12}")
    print("-" * 70)

    metrics = [
        ("Best R²", "best_r2"),
        ("Best Layer", "best_layer"),
        ("GM-Agent Gap", "gm_agent_gap"),
        ("GM AUC", "gm_auc"),
        ("Cross-Scenario R²", "cross_scenario_r2"),
        ("Cross-Scenario AUC", "cross_scenario_auc"),
    ]

    for label, key in metrics:
        val_9b = comparison["9b"].get(key)
        val_27b = comparison["27b"].get(key)
        delta = comparison["deltas"].get(key)

        if val_9b is None and val_27b is None:
            continue

        val_9b_str = f"{val_9b:.3f}" if isinstance(val_9b, float) else str(val_9b) if val_9b else "N/A"
        val_27b_str = f"{val_27b:.3f}" if isinstance(val_27b, float) else str(val_27b) if val_27b else "N/A"
        delta_str = f"{delta:+.3f}" if isinstance(delta, float) else str(delta) if delta else "N/A"

        print(f"{label:<30} {val_9b_str:<12} {val_27b_str:<12} {delta_str:<12}")

    print("=" * 70)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_full_analysis(data_path: str) -> Dict[str, Any]:
    """Run complete probe training and analysis."""

    print(f"\n{'='*60}")
    print("MATS PROBE TRAINING AND ANALYSIS v2")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading data from: {data_path}")
    data = torch.load(data_path, weights_only=False)

    activations = data["activations"]
    labels = data["labels"]
    config = data.get("config", {})

    gm_labels = np.array(labels["gm_labels"])
    agent_labels = np.array(labels["agent_labels"])
    scenarios = labels["scenario"]

    print(f"Loaded {len(gm_labels)} samples")
    print(f"Layers available: {list(activations.keys())}")

    results = {
        "sanity_checks": {},
        "layer_analysis": {},
        "gm_vs_agent": {},
        "generalization": {},
        "deception_rates": {},
        "best_probe": None,
    }

    # Deception rates
    print(f"\n{'='*60}")
    print("DECEPTION RATES BY SCENARIO")
    print(f"{'='*60}")

    deception_rates = compute_deception_rates(gm_labels, scenarios)
    results["deception_rates"] = deception_rates

    for scenario, rate in sorted(deception_rates.items(), key=lambda x: x[1]):
        print(f"  {scenario}: {rate*100:.1f}%")

    # Choose primary layer (mid-layer)
    layers = sorted(activations.keys())
    mid_layer = layers[len(layers) // 2]
    X_mid = activations[mid_layer].float().numpy()

    print(f"\nPrimary analysis layer: {mid_layer}")
    print(f"Activation shape: {X_mid.shape}")

    # ==========================================================================
    # SANITY CHECKS
    # ==========================================================================
    print(f"\n{'='*60}")
    print("SANITY CHECKS")
    print(f"{'='*60}")

    # Check 1: Label variance
    print("\n1. Label Variance Check")
    variance_check = sanity_check_label_variance(gm_labels)
    results["sanity_checks"]["label_variance"] = variance_check
    status = "PASSED" if variance_check["passed"] else "FAILED"
    print(f"   GM labels - mean: {variance_check['mean']:.3f}, std: {variance_check['std']:.3f}")
    print(f"   {status}")

    # Check 2: Random labels
    print("\n2. Random Labels Check")
    random_check = sanity_check_random_labels(X_mid, gm_labels)
    results["sanity_checks"]["random_labels"] = random_check
    status = "PASSED" if random_check["passed"] else "FAILED"
    print(f"   Shuffled R²: {random_check['mean_shuffled_r2']:.4f} +/- {random_check['std_shuffled_r2']:.4f}")
    print(f"   {status} (should be near 0)")

    # Check 3: Train-test gap
    print("\n3. Train-Test Gap Check")
    gap_check = sanity_check_train_test_gap(X_mid, gm_labels)
    results["sanity_checks"]["train_test_gap"] = gap_check
    status = "PASSED" if gap_check["passed"] else "FAILED"
    print(f"   Train R²: {gap_check['train_r2']:.3f}, Test R²: {gap_check['test_r2']:.3f}, Gap: {gap_check['gap']:.3f}")
    print(f"   {status}")

    # ==========================================================================
    # LAYER COMPARISON
    # ==========================================================================
    print(f"\n{'='*60}")
    print("LAYER ANALYSIS")
    print(f"{'='*60}")

    layer_results = {}
    best_layer = None
    best_r2 = -1

    for layer in layers:
        X = activations[layer].float().numpy()

        # Train probe on GM labels
        _, gm_result = train_ridge_probe(X, gm_labels)
        gm_result.layer = layer
        gm_result.label_type = "gm"

        # Train probe on agent labels
        _, agent_result = train_ridge_probe(X, agent_labels)
        agent_result.layer = layer
        agent_result.label_type = "agent"

        layer_results[layer] = {
            "gm": gm_result.to_dict(),
            "agent": agent_result.to_dict(),
        }

        print(f"\nLayer {layer}:")
        print(f"  GM labels    - R²: {gm_result.r2_score:.3f}, AUC: {gm_result.auc:.3f}")
        print(f"  Agent labels - R²: {agent_result.r2_score:.3f}, AUC: {agent_result.auc:.3f}")

        if gm_result.r2_score > best_r2:
            best_r2 = gm_result.r2_score
            best_layer = layer

    results["layer_analysis"] = layer_results
    results["best_probe"] = {"layer": int(best_layer), "r2": float(best_r2)}

    print(f"\nBest layer: {best_layer} (R² = {best_r2:.3f})")

    # ==========================================================================
    # GM vs AGENT COMPARISON
    # ==========================================================================
    print(f"\n{'='*60}")
    print("GM vs AGENT LABEL COMPARISON")
    print(f"{'='*60}")

    X_best = activations[best_layer].float().numpy()

    _, gm_result = train_ridge_probe(X_best, gm_labels)
    _, agent_result = train_ridge_probe(X_best, agent_labels)

    # Mass-mean probe
    _, gm_mm_result = train_mass_mean_probe(X_best, gm_labels)

    results["gm_vs_agent"] = {
        "gm_ridge_r2": float(gm_result.r2_score),
        "agent_ridge_r2": float(agent_result.r2_score),
        "gm_mass_mean_r2": float(gm_mm_result.r2_score),
        "gm_auc": float(gm_result.auc),
        "agent_auc": float(agent_result.auc),
        "gm_mass_mean_auc": float(gm_mm_result.auc),
        "gm_wins": bool(gm_result.r2_score > agent_result.r2_score),
    }

    print(f"\nGM (Ground Truth):")
    print(f"  Ridge R²:     {gm_result.r2_score:.3f}")
    print(f"  Ridge AUC:    {gm_result.auc:.3f}")
    print(f"  Mass-Mean R²: {gm_mm_result.r2_score:.3f}")
    print(f"  Mass-Mean AUC:{gm_mm_result.auc:.3f}")

    print(f"\nAgent (Self-Report):")
    print(f"  Ridge R²:     {agent_result.r2_score:.3f}")
    print(f"  Ridge AUC:    {agent_result.auc:.3f}")

    if results["gm_vs_agent"]["gm_wins"]:
        print(f"\n>> GM labels more predictable than agent self-report!")
        print(f"   This suggests agents encode information they don't 'acknowledge'.")
    else:
        print(f"\n   Agent labels equally/more predictable than GM.")

    # ==========================================================================
    # GENERALIZATION WITH AUC
    # ==========================================================================
    print(f"\n{'='*60}")
    print("GENERALIZATION ANALYSIS (with AUC)")
    print(f"{'='*60}")

    unique_scenarios = list(set(scenarios))

    if len(unique_scenarios) >= 3:
        gen_results = compute_generalization_auc(X_best, gm_labels, scenarios)
        results["generalization"] = gen_results

        for holdout, res in gen_results["by_scenario"].items():
            r2_str = f"{res['test_r2']:.3f}" if res['test_r2'] is not None else "N/A"
            auc_str = f"{res['test_auc']:.3f}" if res['test_auc'] is not None else "N/A"
            rate_str = f"{res['deception_rate']*100:.0f}%"
            print(f"\nHoldout: {holdout} (deception rate: {rate_str})")
            print(f"  R²:  {r2_str}")
            print(f"  AUC: {auc_str}")

        print(f"\n--- AVERAGES ---")
        if gen_results["average_r2"] is not None:
            print(f"Average cross-scenario R²:  {gen_results['average_r2']:.3f} +/- {gen_results['std_r2']:.3f}")
        if gen_results["average_auc"] is not None:
            print(f"Average cross-scenario AUC: {gen_results['average_auc']:.3f} +/- {gen_results['std_auc']:.3f}")

        # Explain R² vs AUC difference
        if gen_results["average_r2"] is not None and gen_results["average_auc"] is not None:
            if gen_results["average_r2"] < 0 and gen_results["average_auc"] > 0.5:
                print(f"\n>> NOTE: Negative R² with positive AUC is expected!")
                print(f"   R² is sensitive to base rate differences between scenarios.")
                print(f"   AUC measures ranking ability, which transfers better.")
    else:
        print("Not enough scenarios for generalization analysis")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\n1. Sanity Checks:")
    all_passed = all(
        check.get("passed", True)
        for check in results["sanity_checks"].values()
    )
    print(f"   {'All passed' if all_passed else 'Some failed'}")

    print(f"\n2. Best Probe Performance:")
    print(f"   Layer {best_layer}: R² = {best_r2:.3f}, AUC = {results['gm_vs_agent']['gm_auc']:.3f}")

    print(f"\n3. GM vs Agent:")
    if results["gm_vs_agent"]["gm_wins"]:
        print(f"   GM more predictable (evidence for implicit deception encoding)")
    else:
        print(f"   Agent equally/more predictable")

    if results["generalization"].get("average_auc"):
        print(f"\n4. Generalization:")
        print(f"   Cross-scenario R²:  {results['generalization']['average_r2']:.3f}")
        print(f"   Cross-scenario AUC: {results['generalization']['average_auc']:.3f}")

    return results


def plot_results(results: Dict, output_path: str = None):
    """Generate visualization of results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Layer comparison
    ax1 = axes[0, 0]
    layers = sorted(results["layer_analysis"].keys())
    gm_r2s = [results["layer_analysis"][l]["gm"]["r2_score"] for l in layers]
    agent_r2s = [results["layer_analysis"][l]["agent"]["r2_score"] for l in layers]
    gm_aucs = [results["layer_analysis"][l]["gm"]["auc"] for l in layers]

    x = np.arange(len(layers))
    width = 0.25

    ax1.bar(x - width, gm_r2s, width, label='GM R²', color='tab:blue')
    ax1.bar(x, agent_r2s, width, label='Agent R²', color='tab:orange')
    ax1.bar(x + width, gm_aucs, width, label='GM AUC', color='tab:green', alpha=0.7)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Score')
    ax1.set_title('Probe Performance by Layer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    # 2. GM vs Agent comparison
    ax2 = axes[0, 1]
    comparison = results["gm_vs_agent"]
    methods = ['Ridge\n(GM)', 'Ridge\n(Agent)', 'Mass-Mean\n(GM)']
    r2_values = [comparison["gm_ridge_r2"], comparison["agent_ridge_r2"], comparison["gm_mass_mean_r2"]]
    auc_values = [comparison["gm_auc"], comparison["agent_auc"], comparison.get("gm_mass_mean_auc", 0.5)]

    x = np.arange(len(methods))
    width = 0.35

    ax2.bar(x - width/2, r2_values, width, label='R²', color='tab:blue')
    ax2.bar(x + width/2, auc_values, width, label='AUC', color='tab:green')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('Score')
    ax2.set_title('GM vs Agent Label Comparison')
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 3. Generalization with both R² and AUC
    ax3 = axes[1, 0]
    if "generalization" in results and "by_scenario" in results["generalization"]:
        gen = results["generalization"]["by_scenario"]
        scenarios = list(gen.keys())
        r2s = [gen[s]["test_r2"] if gen[s]["test_r2"] is not None else 0 for s in scenarios]
        aucs = [gen[s]["test_auc"] if gen[s]["test_auc"] is not None else 0.5 for s in scenarios]
        rates = [gen[s]["deception_rate"] * 100 for s in scenarios]

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax3.bar(x - width/2, r2s, width, label='R²', color='tab:purple', alpha=0.8)
        bars2 = ax3.bar(x + width/2, aucs, width, label='AUC', color='tab:cyan', alpha=0.8)

        # Add deception rates as text
        for i, (scenario, rate) in enumerate(zip(scenarios, rates)):
            ax3.text(i, max(r2s[i], aucs[i]) + 0.05, f'{rate:.0f}%', ha='center', fontsize=8)

        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='R²=0')
        ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='AUC=0.5 (chance)')

        ax3.set_ylabel('Score')
        ax3.set_title('Cross-Scenario Generalization\n(labels show deception rate)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios, rotation=45, ha='right')
        ax3.legend(loc='lower right')
    else:
        ax3.text(0.5, 0.5, 'Not enough scenarios', ha='center', va='center')
        ax3.set_title('Generalization')

    # 4. Deception rates by scenario
    ax4 = axes[1, 1]
    if "deception_rates" in results:
        rates = results["deception_rates"]
        scenarios = sorted(rates.keys(), key=lambda x: rates[x])
        values = [rates[s] * 100 for s in scenarios]

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(scenarios)))
        bars = ax4.barh(scenarios, values, color=colors)

        ax4.set_xlabel('Deception Rate (%)')
        ax4.set_title('Deception Rate by Scenario')
        ax4.set_xlim(0, 100)

        for bar, val in zip(bars, values):
            ax4.text(val + 2, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}%', va='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    plt.show()

    return fig


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train probes on captured activations (v2)")

    parser.add_argument("--data", type=str,
                        help="Path to activations.pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save plots")
    parser.add_argument("--compare", nargs=2, metavar=('9B_RESULTS', '27B_RESULTS'),
                        help="Compare two result files (9B vs 27B)")

    args = parser.parse_args()

    # Model comparison mode
    if args.compare:
        comparison = compare_model_scales(args.compare[0], args.compare[1])
        print_comparison(comparison)

        # Save comparison
        output_path = "model_comparison.json"
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {output_path}")
        return

    # Regular analysis mode
    if not args.data:
        parser.error("--data is required for analysis mode")

    # Run analysis
    results = run_full_analysis(args.data)

    # Save results
    if args.output:
        output_path = args.output
    else:
        data_path = Path(args.data)
        output_path = data_path.parent / "probe_results_v2.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate plots
    if args.plot:
        plot_path = Path(output_path).with_suffix(".png")
        plot_results(results, str(plot_path))


if __name__ == "__main__":
    main()
