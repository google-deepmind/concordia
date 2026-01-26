# Mechanistic Interpretability Tools for Concordia Negotiation Research
# Includes: TransformerLens, SAE Lens, Gemma Scope SAEs, Probing, Visualization

"""
Complete toolkit for mechanistic interpretability research on negotiation agents.

Tools included:
- TransformerLens: Activation extraction via run_with_cache()
- SAE Lens: Sparse Autoencoder analysis
- Gemma Scope: Pre-trained SAEs for Gemma 2 models
- Probing: Linear probes for direction extraction
- Visualization: Attention patterns and feature activations

Installation:
    pip install transformer-lens sae-lens torch sklearn plotly

Usage:
    from mech_interp_tools import (
        load_gemma_with_cache,
        load_gemma_scope_sae,
        extract_sae_features,
        train_linear_probe,
        visualize_feature_activations,
    )
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# =============================================================================
# INSTALLATION VERIFICATION
# =============================================================================

def verify_installation() -> Dict[str, bool]:
    """Verify all mechanistic interpretability tools are installed correctly."""
    results = {}

    # 1. PyTorch
    try:
        import torch
        results['pytorch'] = True
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  CUDA not available (CPU only)")
    except ImportError:
        results['pytorch'] = False
        print("✗ PyTorch not installed: pip install torch")

    # 2. TransformerLens
    try:
        from transformer_lens import HookedTransformer
        results['transformer_lens'] = True
        print("✓ TransformerLens installed")
    except ImportError:
        results['transformer_lens'] = False
        print("✗ TransformerLens not installed: pip install transformer-lens")

    # 3. SAE Lens
    try:
        from sae_lens import SAE
        results['sae_lens'] = True
        print("✓ SAE Lens installed")
    except ImportError:
        results['sae_lens'] = False
        print("✗ SAE Lens not installed: pip install sae-lens")

    # 4. sklearn (for probing)
    try:
        from sklearn.linear_model import Ridge, LogisticRegression
        results['sklearn'] = True
        print("✓ scikit-learn installed")
    except ImportError:
        results['sklearn'] = False
        print("✗ scikit-learn not installed: pip install scikit-learn")

    # 5. Visualization tools
    try:
        import plotly
        results['plotly'] = True
        print("✓ Plotly installed")
    except ImportError:
        results['plotly'] = False
        print("⚠ Plotly not installed (optional): pip install plotly")

    try:
        import matplotlib
        results['matplotlib'] = True
        print("✓ Matplotlib installed")
    except ImportError:
        results['matplotlib'] = False
        print("⚠ Matplotlib not installed (optional): pip install matplotlib")

    # 6. circuitsvis (optional)
    try:
        import circuitsvis
        results['circuitsvis'] = True
        print("✓ circuitsvis installed")
    except ImportError:
        results['circuitsvis'] = False
        print("⚠ circuitsvis not installed (optional): pip install circuitsvis")

    return results


def verify_gemma_loading(model_name: str = "google/gemma-2-9b-it", device: str = "cpu"):
    """Verify TransformerLens can load Gemma 2 correctly."""
    print(f"\nVerifying Gemma loading: {model_name}")

    try:
        from transformer_lens import HookedTransformer

        print("  Loading model (this may take a few minutes)...")
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        print(f"  ✓ Model loaded successfully")
        print(f"    Layers: {model.cfg.n_layers}")
        print(f"    Hidden dim: {model.cfg.d_model}")
        print(f"    Heads: {model.cfg.n_heads}")

        # Test activation extraction
        print("  Testing activation extraction...")
        tokens = model.to_tokens("Hello, world!")
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: "resid_post" in n
            )

        n_cached = len([k for k in cache.keys() if "resid_post" in k])
        print(f"  ✓ Cached {n_cached} residual stream activations")

        # Show sample activation shape
        sample_key = [k for k in cache.keys() if "resid_post" in k][0]
        print(f"    Sample activation shape: {cache[sample_key].shape}")

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def verify_sae_loading(
    sae_release: str = "gemma-scope-9b-pt-res-canonical",
    sae_id: str = "layer_21/width_16k/canonical",
):
    """Verify SAE Lens can load Gemma Scope SAEs."""
    print(f"\nVerifying SAE loading: {sae_release}")

    try:
        from sae_lens import SAE

        print(f"  Loading SAE: {sae_id}")
        # Use new API that returns just the SAE
        sae = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
        )

        print(f"  ✓ SAE loaded successfully")
        print(f"    Input dim: {sae.cfg.d_in}")
        print(f"    Feature dim: {sae.cfg.d_sae}")

        # Test encoding
        print("  Testing encoding...")
        dummy_input = torch.randn(1, sae.cfg.d_in)
        features = sae.encode(dummy_input)
        print(f"  ✓ Encoded shape: {features.shape}")
        print(f"    Non-zero features: {(features > 0).sum().item()}")

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


# =============================================================================
# ACTIVATION EXTRACTION
# =============================================================================

@dataclass
class ActivationCache:
    """Container for extracted activations."""
    residual_stream: Dict[int, torch.Tensor]  # layer -> [batch, seq, d_model]
    attention_patterns: Optional[Dict[int, torch.Tensor]] = None
    mlp_outputs: Optional[Dict[int, torch.Tensor]] = None
    tokens: Optional[torch.Tensor] = None
    text: Optional[str] = None


def load_gemma_with_cache(
    model_name: str = "google/gemma-2-9b-it",
    device: str = "cuda",
):
    """Load Gemma model configured for activation caching."""
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    return model


def extract_activations(
    model,
    text: str,
    layers: Optional[List[int]] = None,
    include_attention: bool = False,
    include_mlp: bool = False,
) -> ActivationCache:
    """Extract activations from a forward pass.

    Args:
        model: HookedTransformer model
        text: Input text
        layers: Which layers to extract (None = all)
        include_attention: Also extract attention patterns
        include_mlp: Also extract MLP outputs

    Returns:
        ActivationCache with requested activations
    """
    tokens = model.to_tokens(text)
    n_layers = model.cfg.n_layers
    layers = layers or list(range(n_layers))

    # Build filter for what to cache
    def name_filter(name: str) -> bool:
        # Always include residual stream
        if "resid_post" in name:
            layer_num = int(name.split(".")[1])
            return layer_num in layers
        # Optionally include attention
        if include_attention and "attn.hook_pattern" in name:
            layer_num = int(name.split(".")[1])
            return layer_num in layers
        # Optionally include MLP
        if include_mlp and "mlp.hook_post" in name:
            layer_num = int(name.split(".")[1])
            return layer_num in layers
        return False

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=name_filter)

    # Organize into ActivationCache
    residual = {}
    attention = {} if include_attention else None
    mlp = {} if include_mlp else None

    for key, value in cache.items():
        layer_num = int(key.split(".")[1])
        if "resid_post" in key:
            residual[layer_num] = value.cpu()
        elif include_attention and "attn.hook_pattern" in key:
            attention[layer_num] = value.cpu()
        elif include_mlp and "mlp.hook_post" in key:
            mlp[layer_num] = value.cpu()

    return ActivationCache(
        residual_stream=residual,
        attention_patterns=attention,
        mlp_outputs=mlp,
        tokens=tokens.cpu(),
        text=text,
    )


# =============================================================================
# SAE FEATURE EXTRACTION
# =============================================================================

@dataclass
class SAEFeatures:
    """Container for SAE feature activations."""
    features: torch.Tensor  # [batch, seq, n_features]
    top_features: List[int]  # Most active feature indices
    feature_activations: Dict[int, float]  # feature_idx -> activation
    sparsity: float  # Fraction of non-zero features


def load_gemma_scope_sae(
    model_size: str = "9b",
    layer: int = 21,
    width: str = "16k",
    site: str = "res",  # res, mlp, or att
    variant: str = "canonical",
) -> Tuple[Any, Dict]:
    """Load a Gemma Scope SAE.

    Args:
        model_size: "2b", "9b", or "27b"
        layer: Layer number
        width: "16k", "64k", "256k", or "1m"
        site: "res" (residual), "mlp", or "att" (attention)
        variant: "canonical" or l0 target like "average_l0_10"

    Returns:
        (sae, config_dict)
    """
    from sae_lens import SAE

    release = f"gemma-scope-{model_size}-pt-{site}-canonical"
    sae_id = f"layer_{layer}/width_{width}/canonical"

    # New API returns just the SAE object
    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
    )

    # Config is accessible via sae.cfg
    cfg_dict = {
        'd_in': sae.cfg.d_in,
        'd_sae': sae.cfg.d_sae,
    }

    return sae, cfg_dict


def extract_sae_features(
    sae,
    activations: torch.Tensor,
    top_k: int = 10,
) -> SAEFeatures:
    """Extract SAE features from activations.

    Args:
        sae: Loaded SAE model
        activations: [batch, seq, d_model] or [d_model] tensor
        top_k: Number of top features to return

    Returns:
        SAEFeatures with feature activations
    """
    # Handle different input shapes
    if activations.dim() == 1:
        activations = activations.unsqueeze(0).unsqueeze(0)
    elif activations.dim() == 2:
        activations = activations.unsqueeze(0)

    # Move to same device as SAE
    activations = activations.to(sae.W_enc.device)

    # Encode to get feature activations
    with torch.no_grad():
        # Flatten for SAE
        batch, seq, d_model = activations.shape
        flat = activations.reshape(-1, d_model)
        features = sae.encode(flat)
        features = features.reshape(batch, seq, -1)

    # Get top features (last position, which is typically most relevant)
    last_pos_features = features[0, -1, :]
    top_indices = torch.topk(last_pos_features, k=top_k).indices.tolist()

    # Build feature activation dict
    feature_acts = {
        idx: last_pos_features[idx].item()
        for idx in top_indices
    }

    # Calculate sparsity
    sparsity = (features > 0).float().mean().item()

    return SAEFeatures(
        features=features.cpu(),
        top_features=top_indices,
        feature_activations=feature_acts,
        sparsity=sparsity,
    )


# =============================================================================
# PROBING
# =============================================================================

@dataclass
class ProbeResult:
    """Results from training a linear probe."""
    accuracy: float  # For classification
    r2_score: float  # For regression
    coefficients: np.ndarray
    intercept: float
    feature_importance: Dict[int, float]


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    task: str = "regression",  # "regression" or "classification"
    regularization: float = 1.0,
    test_size: float = 0.2,
) -> ProbeResult:
    """Train a linear probe on activations.

    Args:
        X: Activations [N, d_model]
        y: Labels [N] (continuous for regression, discrete for classification)
        task: "regression" or "classification"
        regularization: Regularization strength (higher = more regularization)
        test_size: Fraction for test set

    Returns:
        ProbeResult with trained probe info
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import r2_score, accuracy_score

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if task == "regression":
        probe = Ridge(alpha=regularization)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        score = r2_score(y_test, y_pred)
        accuracy = 0.0
    else:
        probe = LogisticRegression(C=1/regularization, max_iter=1000)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        score = 0.0
        accuracy = accuracy_score(y_test, y_pred)

    # Get feature importance
    coeffs = probe.coef_.flatten()
    importance = {
        i: abs(c) for i, c in enumerate(coeffs)
    }
    top_features = dict(sorted(importance.items(), key=lambda x: -x[1])[:20])

    return ProbeResult(
        accuracy=accuracy,
        r2_score=score,
        coefficients=coeffs,
        intercept=probe.intercept_ if hasattr(probe, 'intercept_') else 0,
        feature_importance=top_features,
    )


def extract_direction(
    X_positive: np.ndarray,
    X_negative: np.ndarray,
    method: str = "difference",
) -> np.ndarray:
    """Extract a direction in activation space.

    Args:
        X_positive: Activations for positive examples [N, d_model]
        X_negative: Activations for negative examples [N, d_model]
        method: "difference" (mean diff) or "probe" (train classifier)

    Returns:
        Direction vector [d_model]
    """
    if method == "difference":
        # Simple difference of means
        direction = X_positive.mean(axis=0) - X_negative.mean(axis=0)
        direction = direction / np.linalg.norm(direction)
        return direction

    elif method == "probe":
        # Train a linear probe
        X = np.vstack([X_positive, X_negative])
        y = np.array([1] * len(X_positive) + [0] * len(X_negative))

        from sklearn.linear_model import LogisticRegression
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)

        direction = probe.coef_.flatten()
        direction = direction / np.linalg.norm(direction)
        return direction

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# MASS-MEAN PROBES (Marks & Tegmark method)
# =============================================================================

@dataclass
class MassMeanProbeResult:
    """Results from mass-mean probe (Marks & Tegmark 2023)."""
    direction: np.ndarray  # The deception direction [d_model]
    accuracy: float  # Classification accuracy
    auroc: float  # Area under ROC curve
    separation: float  # Mean separation between classes
    threshold: float  # Optimal classification threshold


def mass_mean_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
) -> MassMeanProbeResult:
    """Train a mass-mean probe (Marks & Tegmark 2023).

    This is simpler than logistic regression and often works better for
    detecting linear features like deception.

    Method:
    1. Compute mean activation for honest (label=0) and deceptive (label=1) samples
    2. Direction = deceptive_mean - honest_mean
    3. Project all activations onto this direction
    4. Find optimal threshold for classification

    Args:
        activations: [N, d_model] activation vectors
        labels: [N] binary labels (0=honest, 1=deceptive)
        normalize: Whether to normalize the direction vector

    Returns:
        MassMeanProbeResult with direction and metrics
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    # Split by label
    honest_mask = labels == 0
    deceptive_mask = labels == 1

    if honest_mask.sum() == 0 or deceptive_mask.sum() == 0:
        raise ValueError("Need both honest and deceptive samples")

    honest_acts = activations[honest_mask]
    deceptive_acts = activations[deceptive_mask]

    # Compute direction (mass-mean difference)
    honest_mean = honest_acts.mean(axis=0)
    deceptive_mean = deceptive_acts.mean(axis=0)
    direction = deceptive_mean - honest_mean

    if normalize:
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

    # Project all activations onto direction
    projections = activations @ direction

    # Compute separation
    honest_proj = projections[honest_mask]
    deceptive_proj = projections[deceptive_mask]
    separation = deceptive_proj.mean() - honest_proj.mean()

    # Find optimal threshold (midpoint between means)
    threshold = (honest_proj.mean() + deceptive_proj.mean()) / 2

    # Compute metrics
    predictions = (projections > threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)

    try:
        auroc = roc_auc_score(labels, projections)
    except ValueError:
        auroc = 0.5  # If only one class present

    return MassMeanProbeResult(
        direction=direction,
        accuracy=accuracy,
        auroc=auroc,
        separation=separation,
        threshold=threshold,
    )


def apply_mass_mean_probe(
    activations: np.ndarray,
    probe_result: MassMeanProbeResult,
) -> np.ndarray:
    """Apply a trained mass-mean probe to new activations.

    Args:
        activations: [N, d_model] or [d_model] activation vectors
        probe_result: Trained probe from mass_mean_probe()

    Returns:
        [N] or scalar deception scores (higher = more deceptive)
    """
    if activations.ndim == 1:
        return float(activations @ probe_result.direction)
    return activations @ probe_result.direction


def compare_probe_methods(
    activations: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    """Compare mass-mean vs logistic regression probes.

    This helps determine which method works better for your data.

    Args:
        activations: [N, d_model] activation vectors
        labels: [N] binary labels

    Returns:
        Dict with metrics for each method
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size, random_state=42
    )

    results = {}

    # Mass-mean probe
    mm_probe = mass_mean_probe(X_train, y_train)
    mm_scores = apply_mass_mean_probe(X_test, mm_probe)
    mm_preds = (mm_scores > mm_probe.threshold).astype(int)

    results['mass_mean'] = {
        'accuracy': accuracy_score(y_test, mm_preds),
        'auroc': roc_auc_score(y_test, mm_scores) if len(np.unique(y_test)) > 1 else 0.5,
    }

    # Logistic regression probe
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]

    results['logistic_regression'] = {
        'accuracy': accuracy_score(y_test, lr_preds),
        'auroc': roc_auc_score(y_test, lr_probs) if len(np.unique(y_test)) > 1 else 0.5,
    }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_feature_activations(
    features: SAEFeatures,
    title: str = "SAE Feature Activations",
    top_k: int = 20,
    save_path: Optional[str] = None,
):
    """Visualize top SAE feature activations."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return

    # Get top features
    sorted_features = sorted(
        features.feature_activations.items(),
        key=lambda x: -x[1]
    )[:top_k]

    indices = [str(f[0]) for f in sorted_features]
    values = [f[1] for f in sorted_features]

    fig = go.Figure(data=[
        go.Bar(x=indices, y=values, marker_color='steelblue')
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Feature Index",
        yaxis_title="Activation",
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        fig.show()


def visualize_probe_results(
    results: Dict[str, ProbeResult],
    title: str = "Probe Results by Layer",
    save_path: Optional[str] = None,
):
    """Visualize probe performance across layers."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return

    layers = list(results.keys())
    scores = [r.r2_score if r.r2_score > 0 else r.accuracy for r in results.values()]

    fig = go.Figure(data=[
        go.Scatter(x=layers, y=scores, mode='lines+markers', marker=dict(size=10))
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Score (R² or Accuracy)",
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


# =============================================================================
# END-TO-END PIPELINE
# =============================================================================

def run_full_analysis(
    model_name: str = "google/gemma-2-9b-it",
    text: str = "I promise to cooperate with you on this deal.",
    layers_to_analyze: List[int] = [0, 21, 41],
    device: str = "cuda",
):
    """Run complete analysis pipeline on a text sample.

    Demonstrates:
    1. Loading Gemma with TransformerLens
    2. Extracting activations
    3. Loading Gemma Scope SAE
    4. Getting SAE features
    """
    print("=" * 60)
    print("FULL MECHANISTIC INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    # 1. Load model
    print(f"\n[1] Loading model: {model_name}")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    print(f"    ✓ Loaded {model.cfg.n_layers} layers, {model.cfg.d_model} dims")

    # 2. Extract activations
    print(f"\n[2] Extracting activations for: '{text[:50]}...'")
    cache = extract_activations(model, text, layers=layers_to_analyze)
    for layer, act in cache.residual_stream.items():
        print(f"    Layer {layer}: {act.shape}")

    # 3. Load SAE (layer 21 residual stream for 9B)
    print(f"\n[3] Loading Gemma Scope SAE")
    try:
        sae, cfg = load_gemma_scope_sae(model_size="9b", layer=21, width="16k")
        print(f"    ✓ SAE loaded: {sae.cfg.d_sae} features")

        # 4. Get SAE features
        print(f"\n[4] Extracting SAE features")
        layer_21_act = cache.residual_stream[21][0, -1, :]  # Last token
        sae_features = extract_sae_features(sae, layer_21_act)
        print(f"    Sparsity: {sae_features.sparsity:.4f}")
        print(f"    Top features: {sae_features.top_features[:5]}")

    except Exception as e:
        print(f"    ⚠ SAE loading failed (may need to download): {e}")
        sae_features = None

    print("\n" + "=" * 60)
    print("Analysis complete!")

    return cache, sae_features


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Mechanistic Interpretability Tools")
    print("=" * 40)
    print("\nVerifying installation...")
    results = verify_installation()

    all_required = all([
        results.get('pytorch', False),
        results.get('transformer_lens', False),
        results.get('sae_lens', False),
        results.get('sklearn', False),
    ])

    if all_required:
        print("\n✓ All required tools installed!")
        print("\nTo verify Gemma loading (requires download):")
        print("  verify_gemma_loading('google/gemma-2-9b-it', device='cpu')")
        print("\nTo verify SAE loading (requires download):")
        print("  verify_sae_loading()")
    else:
        print("\n✗ Missing required tools. Install with:")
        print("  pip install torch transformer-lens sae-lens scikit-learn")
