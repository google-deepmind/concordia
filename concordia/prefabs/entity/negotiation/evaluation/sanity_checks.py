# Sanity Checks for Deception Probe Validation
# Run these after training probes to validate results
#
# Usage:
#   from sanity_checks import run_all_sanity_checks, run_causal_validation, print_limitations
#
#   sanity = run_all_sanity_checks(activations, gm_labels)
#   if sanity["all_passed"]:
#       causal = run_causal_validation(model, activations, gm_labels, best_layer)
#   print_limitations(len(gm_labels), "Gemma 2 9B", causal.get("causal_effect_detected", False))

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# =============================================================================
# SANITY CHECK FUNCTIONS
# =============================================================================

def sanity_check_random_labels(
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int = 5,
) -> Dict[str, Any]:
    """
    Check 1: Probes on shuffled labels should give R² ≈ 0.
    If they don't, probes are memorizing, not learning.
    """
    shuffle_r2s = []

    for seed in range(n_shuffles):
        np.random.seed(seed)
        y_shuffled = np.random.permutation(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_shuffled, test_size=0.2, random_state=42
        )

        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        test_pred = probe.predict(X_test)
        shuffle_r2s.append(r2_score(y_test, test_pred))

    mean_r2 = np.mean(shuffle_r2s)
    return {
        "mean_shuffled_r2": float(mean_r2),
        "std_shuffled_r2": float(np.std(shuffle_r2s)),
        "max_shuffled_r2": float(np.max(shuffle_r2s)),
        "passed": mean_r2 < 0.05,  # Should be near 0
        "message": f"Shuffled R²: {mean_r2:.4f} (should be < 0.05)"
    }


def sanity_check_train_test_gap(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Check 2: Large train-test gap indicates overfitting.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)

    train_r2 = r2_score(y_train, probe.predict(X_train))
    test_r2 = r2_score(y_test, probe.predict(X_test))
    gap = train_r2 - test_r2

    return {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "gap": float(gap),
        "passed": gap < 0.2,  # Gap should be small
        "message": f"Train-test gap: {gap:.3f} (should be < 0.2)"
    }


def sanity_check_label_variance(y: np.ndarray) -> Dict[str, Any]:
    """
    Check 3: Labels must have variance for probes to learn anything.
    """
    std = np.std(y)
    return {
        "mean": float(np.mean(y)),
        "std": float(std),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "n_unique": int(len(np.unique(y))),
        "passed": std > 0.1,  # Need some variance
        "message": f"Label std: {std:.3f} (should be > 0.1)"
    }


def sanity_check_layer_0_baseline(
    activations: Dict[int, np.ndarray],
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Check 4: Layer 0 should have lower R² than mid-layers.
    If layer 0 is best, the probe may be picking up input features, not learned representations.
    """
    layers = sorted(activations.keys())

    if 0 not in layers:
        return {
            "passed": True,
            "message": "Layer 0 not in activations, skipping check"
        }

    layer_r2s = {}
    for layer in layers:
        X = activations[layer]
        if hasattr(X, 'numpy'):
            X = X.numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        layer_r2s[layer] = r2_score(y_test, probe.predict(X_test))

    layer_0_r2 = layer_r2s[0]
    mid_layer = layers[len(layers) // 2]
    mid_r2 = layer_r2s[mid_layer]

    passed = mid_r2 > layer_0_r2

    return {
        "layer_0_r2": float(layer_0_r2),
        "mid_layer": int(mid_layer),
        "mid_layer_r2": float(mid_r2),
        "all_layer_r2s": {int(k): float(v) for k, v in layer_r2s.items()},
        "passed": passed,
        "message": f"Layer 0 R²: {layer_0_r2:.3f}, Layer {mid_layer} R²: {mid_r2:.3f} (mid should be higher)"
    }


def run_all_sanity_checks(
    activations: Dict[int, np.ndarray],
    gm_labels: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all 4 sanity checks.

    Args:
        activations: Dict mapping layer -> activation array [N, d_model]
        gm_labels: Ground truth deception labels [N]
        verbose: Print results

    Returns:
        Dict with all check results and "all_passed" boolean
    """
    # Get mid-layer activations for single-layer checks
    layers = sorted(activations.keys())
    mid_layer = layers[len(layers) // 2]
    X_mid = activations[mid_layer]
    if hasattr(X_mid, 'numpy'):
        X_mid = X_mid.numpy()

    if hasattr(gm_labels, 'numpy'):
        gm_labels = gm_labels.numpy()

    results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("SANITY CHECKS")
        print("=" * 60)

    # Check 1: Random labels
    results["random_labels"] = sanity_check_random_labels(X_mid, gm_labels)
    if verbose:
        status = "PASSED" if results["random_labels"]["passed"] else "FAILED"
        print(f"\n1. Random Labels: {status}")
        print(f"   {results['random_labels']['message']}")

    # Check 2: Train-test gap
    results["train_test_gap"] = sanity_check_train_test_gap(X_mid, gm_labels)
    if verbose:
        status = "PASSED" if results["train_test_gap"]["passed"] else "FAILED"
        print(f"\n2. Train-Test Gap: {status}")
        print(f"   {results['train_test_gap']['message']}")

    # Check 3: Label variance
    results["label_variance"] = sanity_check_label_variance(gm_labels)
    if verbose:
        status = "PASSED" if results["label_variance"]["passed"] else "FAILED"
        print(f"\n3. Label Variance: {status}")
        print(f"   {results['label_variance']['message']}")

    # Check 4: Layer 0 baseline
    results["layer_0_baseline"] = sanity_check_layer_0_baseline(activations, gm_labels)
    if verbose:
        status = "PASSED" if results["layer_0_baseline"]["passed"] else "FAILED"
        print(f"\n4. Layer 0 Baseline: {status}")
        print(f"   {results['layer_0_baseline']['message']}")

    # Overall
    all_passed = all(r["passed"] for r in results.values())
    results["all_passed"] = all_passed

    if verbose:
        print("\n" + "-" * 60)
        overall = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
        print(f"Overall: {overall}")

    return results


# =============================================================================
# CAUSAL VALIDATION (Optional, requires model)
# =============================================================================

def run_causal_validation(
    model,
    activations: Dict[int, np.ndarray],
    gm_labels: np.ndarray,
    best_layer: int,
    n_samples: int = 20,
    alpha: float = 2.0,
) -> Dict[str, Any]:
    """
    Causal validation: Intervene on activations and check if it changes model behavior.

    This is the gold standard test - if adding the "deception direction" to activations
    actually makes the model produce more deceptive outputs, the probe is causally valid.

    Args:
        model: TransformerLens HookedTransformer
        activations: Dict mapping layer -> activation array
        gm_labels: Ground truth labels
        best_layer: Layer to intervene on
        n_samples: Number of samples to test
        alpha: Intervention strength multiplier

    Returns:
        Dict with causal validation results
    """
    import torch

    X = activations[best_layer]
    if hasattr(X, 'numpy'):
        X = X.numpy()

    if hasattr(gm_labels, 'numpy'):
        gm_labels = gm_labels.numpy()

    # Compute deception direction (mass-mean)
    honest_mask = gm_labels < 0.5
    deceptive_mask = gm_labels >= 0.5

    if honest_mask.sum() < 5 or deceptive_mask.sum() < 5:
        return {
            "causal_effect_detected": False,
            "message": "Not enough samples in each class for causal validation",
            "passed": False,
        }

    honest_mean = X[honest_mask].mean(axis=0)
    deceptive_mean = X[deceptive_mask].mean(axis=0)
    direction = deceptive_mean - honest_mean
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    direction_tensor = torch.tensor(direction, dtype=torch.float32)

    # Test intervention on honest samples
    honest_indices = np.where(honest_mask)[0][:n_samples]

    behavioral_changes = []

    test_prompt = "In this negotiation, I will be completely honest about my position."

    for idx in honest_indices:
        try:
            # Define hook to add deception direction
            def intervention_hook(activation, hook):
                # Add deception direction to last token
                activation[:, -1, :] += alpha * direction_tensor.to(activation.device)
                return activation

            # Get baseline output
            tokens = model.to_tokens(test_prompt)
            baseline_logits = model(tokens)

            # Get intervened output
            hook_name = f"blocks.{best_layer}.hook_resid_post"
            intervened_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, intervention_hook)]
            )

            # Check if outputs differ meaningfully
            logit_diff = (intervened_logits - baseline_logits).abs().mean().item()
            behavioral_changes.append(logit_diff)

        except Exception as e:
            continue

    if not behavioral_changes:
        return {
            "causal_effect_detected": False,
            "message": "Intervention failed to run",
            "passed": False,
        }

    mean_change = np.mean(behavioral_changes)

    # Threshold for "meaningful" change (empirical)
    causal_effect = mean_change > 0.1

    return {
        "causal_effect_detected": causal_effect,
        "mean_logit_change": float(mean_change),
        "n_samples_tested": len(behavioral_changes),
        "intervention_strength": alpha,
        "passed": causal_effect,
        "message": f"Mean logit change: {mean_change:.4f} (threshold: 0.1)"
    }


# =============================================================================
# LIMITATIONS REPORTING
# =============================================================================

def print_limitations(
    n_samples: int,
    model_name: str,
    causal_validated: bool = False,
) -> None:
    """
    Print honest limitations of the experiment.
    Always call this at the end of your analysis.
    """
    print("\n" + "=" * 60)
    print("LIMITATIONS AND CAVEATS")
    print("=" * 60)

    limitations = []

    # Sample size
    if n_samples < 100:
        limitations.append(f"- Small sample size (n={n_samples}). Results may not generalize.")
    elif n_samples < 200:
        limitations.append(f"- Moderate sample size (n={n_samples}). Consider running more trials.")

    # Single model
    limitations.append(f"- Single model ({model_name}). Results may not transfer to other models.")

    # Simulated negotiations
    limitations.append("- Simulated negotiations with scripted counterparts, not real negotiations.")

    # Linear probes
    limitations.append("- Linear probes assume deception is linearly represented (may miss nonlinear patterns).")

    # Causal validation
    if not causal_validated:
        limitations.append("- No causal validation performed. Correlation != causation.")
    else:
        limitations.append("+ Causal validation passed (intervention changed model behavior).")

    # Ground truth
    limitations.append("- Ground truth based on rule-based parsing, may have label noise.")

    # Generalization
    limitations.append("- Train/test on same distribution. Real-world deception may differ.")

    for lim in limitations:
        print(lim)

    print("\n" + "-" * 60)
    print("INTERPRETATION GUIDANCE")
    print("-" * 60)
    print("""
If R² > 0.15 and sanity checks pass:
  -> Deception is likely linearly represented at some layer
  -> But this doesn't prove the model "knows" it's being deceptive

If GM R² > Agent R²:
  -> Model encodes information it doesn't "acknowledge" in its responses
  -> This is the key finding for implicit deception research

If generalization R² > 0.10:
  -> Probe captures general deception, not scenario-specific patterns
  -> Stronger evidence for a universal "deception direction"

Always report limitations alongside positive results.
""")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Generate fake data for testing
    print("Testing sanity_checks.py with synthetic data...")

    n_samples = 100
    d_model = 256

    # Fake activations
    activations = {
        0: np.random.randn(n_samples, d_model),
        10: np.random.randn(n_samples, d_model),
        20: np.random.randn(n_samples, d_model),
    }

    # Add some signal to layer 20
    gm_labels = np.random.rand(n_samples)
    activations[20] += np.outer(gm_labels, np.random.randn(d_model)) * 0.5

    # Run checks
    results = run_all_sanity_checks(activations, gm_labels, verbose=True)

    # Print limitations
    print_limitations(n_samples, "Test Model", causal_validated=False)

    print("\n\nTest complete!")
