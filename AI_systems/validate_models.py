"""
Model Validation Suite
AI_systems/validate_models.py

Validates all AI models and components used in the Generative Data Visualisation project.
Covers Phase 7 (Evaluation) from the project roadmap:
  - VAE reconstruction loss + latent space quality
  - PCA variance explained
  - KMeans clustering (silhouette score + elbow method)
  - MLP mapping network output range validation
  - Information entropy of generated visual parameters
  - Latent space distribution check

Usage:
    python AI_systems/validate_models.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
VAE_DIR     = ROOT / "AI_systems" / "vae_artifacts"
MAPPING_DIR = ROOT / "AI_systems" / "mapping_network_artifacts"

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def passed(msg: str):
    print(f"  [✓] {msg}")

def warning(msg: str):
    print(f"  [!] {msg}")

def failed(msg: str):
    print(f"  [✗] {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VAE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_vae():
    section("VAE — Latent Space & Reconstruction")

    # Load latent vectors
    latent_path = VAE_DIR / "latent_vectors.npy"
    if not latent_path.exists():
        failed(f"latent_vectors.npy not found at {latent_path}")
        return None

    Z = np.load(str(latent_path))
    passed(f"Latent vectors loaded: shape {Z.shape}")

    # Check latent dimensionality
    latent_dim = Z.shape[1]
    passed(f"Latent dimension: {latent_dim}")

    # Check for NaN / Inf
    if np.isnan(Z).any() or np.isinf(Z).any():
        failed("Latent vectors contain NaN or Inf values")
    else:
        passed("No NaN or Inf in latent vectors")

    # Latent space distribution — should be roughly standard normal
    mean = Z.mean()
    std  = Z.std()
    print(f"  [i] Latent mean: {mean:.4f}  (ideal ≈ 0)")
    print(f"  [i] Latent std:  {std:.4f}  (ideal ≈ 1)")
    if abs(mean) < 1.0 and 0.3 < std < 3.0:
        passed("Latent distribution looks reasonable")
    else:
        warning("Latent distribution may indicate posterior collapse or poor training")

    # Load training summary
    summary_path = VAE_DIR / "vae_training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        recon_loss = summary.get("final_reconstruction_loss") or summary.get("reconstruction_loss")
        kl_loss    = summary.get("final_kl_loss") or summary.get("kl_loss")
        total_loss = summary.get("final_loss") or summary.get("loss")

        if recon_loss is not None:
            print(f"  [i] Reconstruction loss: {recon_loss:.4f}")
            if recon_loss < 50:
                passed("Reconstruction loss is acceptable")
            else:
                warning("Reconstruction loss is high — VAE may need more training")

        if kl_loss is not None:
            print(f"  [i] KL divergence loss:  {kl_loss:.4f}")
            if kl_loss > 0:
                passed("KL divergence is positive (encoder is active)")
            else:
                warning("KL divergence is 0 — possible posterior collapse")

        if total_loss is not None:
            print(f"  [i] Total loss:          {total_loss:.4f}")
    else:
        warning("vae_training_summary.json not found — skipping loss validation")

    return Z


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PCA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_pca(Z: np.ndarray):
    section("PCA — Dimensionality Reduction Quality")

    if Z is None:
        failed("No latent vectors available — skipping PCA validation")
        return

    n_components = min(3, Z.shape[1], Z.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(Z)

    variance_ratio = pca.explained_variance_ratio_
    cumulative_var = float(np.cumsum(variance_ratio)[-1])

    for i, v in enumerate(variance_ratio):
        print(f"  [i] PC{i+1} variance explained: {v:.1%}")

    print(f"  [i] Total variance captured: {cumulative_var:.1%}")

    if cumulative_var >= 0.30:
        passed(f"PCA captures {cumulative_var:.1%} of variance — acceptable for 3 components")
    else:
        warning(f"PCA only captures {cumulative_var:.1%} — latent space may be high-entropy (not necessarily bad)")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLUSTERING VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_clustering(Z: np.ndarray):
    section("KMeans Clustering — Silhouette + Elbow Method")

    if Z is None:
        failed("No latent vectors available — skipping clustering validation")
        return

    # Elbow method
    inertias = []
    k_range  = range(2, 9)
    print("  [→] Running elbow method (k=2..8) …")

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Z)
        inertias.append(km.inertia_)

    # Find elbow — largest drop in inertia
    drops = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    optimal_k = list(k_range)[np.argmax(drops) + 1]
    print(f"  [i] Inertias: {[round(v,1) for v in inertias]}")
    passed(f"Elbow method suggests optimal k = {optimal_k}")

    # Silhouette scores for each k
    print("  [→] Computing silhouette scores …")
    best_k, best_sil = 2, -1
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Z)
        sil    = silhouette_score(Z, labels)
        print(f"  [i] k={k}  silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, k

    passed(f"Best silhouette score: {best_sil:.4f} at k={best_k}")

    if best_sil >= 0.25:
        passed("Clustering quality is acceptable (silhouette ≥ 0.25)")
    else:
        warning("Low silhouette score — clusters may overlap in latent space")

    # Final cluster stats with best k
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels   = km_final.fit_predict(Z)
    counts   = np.bincount(labels)
    print(f"  [i] Cluster sizes (k={best_k}): {counts.tolist()}")

    # Entropy of cluster distribution
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * np.log2(probs)).sum()
    print(f"  [i] Cluster distribution entropy: {entropy:.4f}")
    passed(f"Entropy computed: {entropy:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAPPING NETWORK VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_mapping_network():
    section("MLP Mapping Network — θ Output Validation")

    theta_path = MAPPING_DIR / "theta_pred.npy"
    if not theta_path.exists():
        failed(f"theta_pred.npy not found at {theta_path}")
        return

    theta = np.load(str(theta_path))
    passed(f"Theta predictions loaded: shape {theta.shape}")

    # Check value range — sigmoid output should be in [0, 1]
    t_min, t_max = theta.min(), theta.max()
    print(f"  [i] θ value range: [{t_min:.4f}, {t_max:.4f}]  (expected [0, 1])")

    if t_min >= 0.0 and t_max <= 1.0:
        passed("All θ values are within [0, 1] — sigmoid output is correct")
    else:
        warning("Some θ values are outside [0, 1] — check activation function")

    # Check for NaN / Inf
    if np.isnan(theta).any() or np.isinf(theta).any():
        failed("θ contains NaN or Inf values")
    else:
        passed("No NaN or Inf in θ predictions")

    # Per-parameter stats
    param_names = ["turbulence", "colour", "noise", "fragmentation", "velocity"]
    n_params    = min(theta.shape[1], len(param_names))
    print(f"  [i] Per-parameter means:")
    for i in range(n_params):
        print(f"      θ[{i}] {param_names[i]:<15} mean={theta[:,i].mean():.4f}  std={theta[:,i].std():.4f}")

    # Information entropy of θ outputs (per parameter)
    section_entropy = []
    for i in range(n_params):
        col = theta[:, i]
        # Bin into 10 buckets
        hist, _ = np.histogram(col, bins=10, range=(0, 1), density=True)
        hist    = hist / hist.sum()
        hist    = hist[hist > 0]
        ent     = -(hist * np.log2(hist)).sum()
        section_entropy.append(ent)

    avg_entropy = np.mean(section_entropy)
    print(f"  [i] Average θ entropy across parameters: {avg_entropy:.4f}")
    if avg_entropy > 1.0:
        passed("θ outputs show good diversity (entropy > 1.0)")
    else:
        warning("Low θ entropy — visual parameters may be too uniform (possible mode collapse)")

    # Load training summary
    summary_path = MAPPING_DIR / "mapping_training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        final_loss = summary.get("final_loss") or summary.get("loss")
        if final_loss is not None:
            print(f"  [i] Mapping network final loss: {final_loss:.6f}")
            passed("Training summary loaded successfully")
    else:
        warning("mapping_training_summary.json not found — skipping loss check")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OVERALL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary():
    section("VALIDATION COMPLETE")
    print("""
  Components validated:
    [1] VAE — latent vectors, distribution, reconstruction loss, KL divergence
    [2] PCA — variance explained across 3 components
    [3] KMeans — elbow method, silhouette scores (k=2..8), cluster entropy
    [4] MLP Mapping Network — θ range, NaN check, entropy, per-parameter stats

  Metrics produced (per project spec Phase 7):
    ✓ Silhouette Score
    ✓ Reconstruction Loss
    ✓ Information Entropy (cluster distribution + θ output)
    ✓ KL Divergence
    ✓ Elbow Method (optimal k)
    ✓ Latent space distribution check
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  GENERATIVE DATA VISUALISATION — MODEL VALIDATION")
    print("="*60)

    start = time.time()

    Z = validate_vae()
    validate_pca(Z)
    validate_clustering(Z)
    validate_mapping_network()
    print_summary()

    elapsed = time.time() - start
    print(f"  Total validation time: {elapsed:.1f}s\n")