#!/usr/bin/env python3
"""Complete analysis script for deception detection experiment results.

Run on RunPod with:
    python analyze_results.py --input /workspace/persistent/outputs

Or locally with:
    python analyze_results.py --input concordia/outputs
"""

import argparse
import json
import glob
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def load_activation_files(input_dir: str):
    """Load all activation files from input directory."""
    files = glob.glob(f"{input_dir}/*/activations_*.pt")
    print(f"Found {len(files)} activation files")

    all_samples = []
    scenario_samples = {}

    for f in files:
        scenario = Path(f).parent.name
        print(f"\nLoading: {scenario}")
        data = torch.load(f, map_location='cpu', weights_only=False)
        samples = data if isinstance(data, list) else data.get('samples', [])
        print(f"  Samples: {len(samples)}")

        if samples:
            print(f"  Keys: {list(samples[0].keys())}")
            has_sae = 'sae_features' in samples[0]
            print(f"  SAE features: {'YES' if has_sae else 'NO'}")

            scenario_samples[scenario] = samples
            all_samples.extend(samples)

    return all_samples, scenario_samples


def get_activations(samples, layer=13):
    """Extract activations and labels from samples."""
    X = []
    y_gm = []
    y_agent = []
    scenarios = []

    for s in samples:
        # Get activations
        if 'activations' in s:
            act = s['activations']
            if isinstance(act, dict) and layer in act:
                X.append(act[layer].numpy() if hasattr(act[layer], 'numpy') else act[layer])
            elif isinstance(act, dict) and str(layer) in act:
                X.append(act[str(layer)].numpy() if hasattr(act[str(layer)], 'numpy') else act[str(layer)])
            elif hasattr(act, 'numpy'):
                X.append(act.numpy())
            else:
                continue
        else:
            continue

        # Get labels
        gm = s.get('gm_deceptive', s.get('metadata', {}).get('gm_deceptive', False))
        agent = s.get('agent_deceptive', s.get('metadata', {}).get('agent_deceptive', False))
        y_gm.append(1 if gm else 0)
        y_agent.append(1 if agent else 0)
        scenarios.append(s.get('scenario', s.get('metadata', {}).get('scenario', 'unknown')))

    return np.array(X), np.array(y_gm), np.array(y_agent), scenarios


def analyze_merged_probe(X, y_gm):
    """Train and evaluate probe on merged data."""
    print("\n" + "="*70)
    print("LAYER 13 ANALYSIS (MERGED DATA)")
    print("="*70)

    print(f"Activation shape: {X.shape}")
    print(f"GM deceptive: {y_gm.sum()}/{len(y_gm)} ({100*y_gm.mean():.1f}%)")

    if len(np.unique(y_gm)) < 2:
        print("ERROR: Only one class present, cannot train probe")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_gm, test_size=0.2, random_state=42, stratify=y_gm
    )

    clf = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0])
    clf.fit(X_train, y_train)

    y_pred = clf.decision_function(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = clf.score(X_test, y_test)

    print(f"\nMerged Probe Results:")
    print(f"  Test AUC: {auc:.3f}")
    print(f"  Test Accuracy: {acc:.3f}")

    return auc, acc


def analyze_cross_scenario_generalization(X, y_gm, scenarios):
    """Test cross-scenario generalization."""
    print("\n" + "="*70)
    print("CROSS-SCENARIO GENERALIZATION")
    print("="*70)

    unique_scenarios = list(set(scenarios))
    print(f"Scenarios: {unique_scenarios}")

    generalization_results = {}

    for test_scenario in unique_scenarios:
        train_idx = [i for i, s in enumerate(scenarios) if s != test_scenario]
        test_idx = [i for i, s in enumerate(scenarios) if s == test_scenario]

        if len(train_idx) < 10 or len(test_idx) < 10:
            continue

        X_train, y_train = X[train_idx], y_gm[train_idx]
        X_test, y_test = X[test_idx], y_gm[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  {test_scenario}: Skipped (single class)")
            continue

        clf = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0])
        clf.fit(X_train, y_train)

        y_pred = clf.decision_function(X_test)
        auc = roc_auc_score(y_test, y_pred)

        generalization_results[test_scenario] = auc
        print(f"  Train on others, test on {test_scenario}: AUC = {auc:.3f}")

    avg_gen = np.mean(list(generalization_results.values())) if generalization_results else 0
    print(f"\n  Average generalization AUC: {avg_gen:.3f}")

    return generalization_results, avg_gen


def analyze_sae_features(all_samples):
    """Analyze SAE features if present."""
    if not all_samples or 'sae_features' not in all_samples[0]:
        print("\nNo SAE features found in samples")
        return False, []

    print("\n" + "="*70)
    print("SAE FEATURE ANALYSIS")
    print("="*70)

    sae_features = []
    sae_labels = []

    for s in all_samples:
        if 'sae_features' in s:
            sae = s['sae_features']
            if hasattr(sae, 'numpy'):
                sae = sae.numpy()
            sae_features.append(sae)
            sae_labels.append(1 if s.get('gm_deceptive') else 0)

    X_sae = np.array(sae_features)
    y_sae = np.array(sae_labels)

    print(f"SAE features shape: {X_sae.shape}")
    print(f"Deceptive samples: {y_sae.sum()}/{len(y_sae)}")

    # Find top correlated features
    correlations = []
    for i in range(X_sae.shape[1]):
        if X_sae[:, i].std() > 0:
            corr = np.corrcoef(X_sae[:, i], y_sae)[0, 1]
            if not np.isnan(corr):
                correlations.append((i, corr, X_sae[:, i].mean()))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 20 SAE features correlated with deception:")
    print(f"{'Feature':<10} {'Corr':<10} {'Direction':<12} {'Neuronpedia Link'}")
    print("-"*80)

    top_sae_features = []
    for feat_id, corr, mean_act in correlations[:20]:
        direction = "DECEPTION+" if corr > 0 else "HONESTY+"
        link = f"https://neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/{feat_id}"
        print(f"{feat_id:<10} {corr:+.4f}    {direction:<12} {link}")
        top_sae_features.append({
            "feature_id": int(feat_id),
            "correlation": float(corr),
            "direction": direction,
            "neuronpedia_url": link
        })

    return True, top_sae_features


def main():
    parser = argparse.ArgumentParser(description="Analyze deception detection results")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with scenario outputs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: input/complete_analysis.json)")
    parser.add_argument("--layer", type=int, default=13,
                        help="Layer to analyze (default: 13)")
    args = parser.parse_args()

    output_file = args.output or f"{args.input}/complete_analysis.json"

    print("="*70)
    print("COMPLETE DECEPTION ANALYSIS")
    print("="*70)

    # Load data
    all_samples, scenario_samples = load_activation_files(args.input)

    if not all_samples:
        print("ERROR: No samples found!")
        return

    print(f"\n{'='*70}")
    print(f"TOTAL SAMPLES: {len(all_samples)}")
    print(f"{'='*70}")

    # Extract activations
    X, y_gm, y_agent, scenarios = get_activations(all_samples, layer=args.layer)

    if len(X) == 0:
        print("ERROR: Could not extract activations!")
        return

    # Merged probe analysis
    auc, acc = analyze_merged_probe(X, y_gm)

    # Cross-scenario generalization
    gen_results, avg_gen = analyze_cross_scenario_generalization(X, y_gm, scenarios)

    # SAE analysis
    has_sae, top_sae_features = analyze_sae_features(all_samples)

    # Compile results
    results = {
        "total_samples": len(all_samples),
        "scenarios": list(set(scenarios)),
        "merged_probe": {
            "layer": args.layer,
            "auc": float(auc) if auc else None,
            "accuracy": float(acc) if acc else None,
        },
        "generalization": {k: float(v) for k, v in gen_results.items()},
        "avg_generalization_auc": float(avg_gen),
        "has_sae": has_sae,
        "top_sae_features": top_sae_features,
        "deception_rates": {
            scenario: float(np.mean([1 if s.get('gm_deceptive') else 0
                                      for s in samples]))
            for scenario, samples in scenario_samples.items()
        }
    }

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"RESULTS SAVED TO: {output_file}")
    print("="*70)

    # Print summary for copy-paste
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
