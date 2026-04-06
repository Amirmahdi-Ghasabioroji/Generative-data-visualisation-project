"""
Simple and effective model validation for this project.

Covers:
- VAE
- PCA (on VAE latent vectors)
- Scraper categorisation model
- Mapping network
- Latent visual mapper

Design goals:
- Fast defaults
- Clear pass/warn/fail output
- Real metrics without over-engineering
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

TF_IMPORT_ERROR: str | None = None
try:
    import tensorflow as tf
except Exception as exc:
    tf = None
    TF_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import joblib
except Exception:
    joblib = None

from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AI_DIR = ROOT / "AI_systems"
VAE_DIR = AI_DIR / "vae_artifacts"
MAPPING_DIR = AI_DIR / "mapping_network_artifacts"
SCRAPER_DIR = AI_DIR / "scraper_model_artifacts"
LATENT_MAPPER_DIR = AI_DIR / "latent_mapper_artifacts"
DATASET_DIR = ROOT / "Data_Pipeline" / "datasets"

COUNTS = {"pass": 0, "warn": 0, "fail": 0}


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def passed(msg: str) -> None:
    COUNTS["pass"] += 1
    print(f"[PASS] {msg}")


def warning(msg: str) -> None:
    COUNTS["warn"] += 1
    print(f"[WARN] {msg}")


def failed(msg: str) -> None:
    COUNTS["fail"] += 1
    print(f"[FAIL] {msg}")


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        warning(f"Could not parse JSON {path.name}: {exc}")
        return None


def load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(str(path))
    except Exception as exc:
        warning(f"Could not load {path.name}: {exc}")
        return None


def sample_idx(n: int, max_rows: int, seed: int) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_rows, replace=False))


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    s = counts.sum()
    if s <= 0:
        return 0.0
    probs = counts / s
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def tf_unavailable_reason() -> str:
    if TF_IMPORT_ERROR:
        return f"TensorFlow unavailable ({TF_IMPORT_ERROR})"
    return "TensorFlow unavailable"


def latest_bluesky_dataset() -> Path | None:
    files = sorted(DATASET_DIR.glob("bitcoin_bluesky_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def validate_assets(report: dict[str, Any]) -> None:
    section("0) Quick Asset Check")
    metrics: dict[str, Any] = {"historical": {}, "live": {}}

    historical_assets = {
        "vae_latents": VAE_DIR / "latent_vectors.npy",
        "vae_weights": VAE_DIR / "vae_weights.weights.h5",
        "mapping_theta_pred": MAPPING_DIR / "theta_pred.npy",
        "mapping_theta_targets": MAPPING_DIR / "theta_targets.npy",
        "scraper_model": SCRAPER_DIR / "model.keras",
        "scraper_kmeans": SCRAPER_DIR / "kmeans.joblib",
    }
    live_assets = {
        "latent_mapper_weights": LATENT_MAPPER_DIR / "latent_mapper.weights.h5",
        "latent_mapper_config": LATENT_MAPPER_DIR / "latent_mapper_config.json",
    }

    hist_ok = True
    for key, path in historical_assets.items():
        ok = path.exists()
        metrics["historical"][key] = ok
        if ok:
            passed(f"{key} found")
        else:
            warning(f"{key} missing")
            hist_ok = False

    live_ok = True
    for key, path in live_assets.items():
        ok = path.exists()
        metrics["live"][key] = ok
        if ok:
            passed(f"{key} found")
        else:
            warning(f"{key} missing")
            live_ok = False

    metrics["historical_ready"] = hist_ok
    metrics["live_ready"] = live_ok
    report["setup_assets"] = metrics


def validate_vae(report: dict[str, Any]) -> np.ndarray | None:
    section("1) VAE")
    metrics: dict[str, Any] = {}

    z = load_npy(VAE_DIR / "latent_vectors.npy")
    if z is None:
        failed("latent_vectors.npy missing")
        report["vae"] = metrics
        return None

    if z.ndim != 2:
        failed(f"Latent vectors should be 2D, got {z.shape}")
        report["vae"] = metrics
        return None

    nan_count = int(np.isnan(z).sum())
    inf_count = int(np.isinf(z).sum())

    metrics["latent_shape"] = [int(z.shape[0]), int(z.shape[1])]
    metrics["latent_mean"] = float(np.mean(z))
    metrics["latent_std"] = float(np.std(z))
    metrics["nan_count"] = nan_count
    metrics["inf_count"] = inf_count

    info(f"shape={z.shape}, mean={metrics['latent_mean']:.6f}, std={metrics['latent_std']:.6f}")

    if nan_count == 0 and inf_count == 0:
        passed("Latent vectors have no NaN/Inf")
    else:
        failed("Latent vectors contain NaN/Inf")

    summary = load_json(VAE_DIR / "vae_training_summary.json") or load_json(AI_DIR / "vae_training_summary.json")
    metrics["has_training_summary"] = summary is not None
    if summary is not None:
        passed("VAE training summary found")
    else:
        warning("VAE training summary not found")

    report["vae"] = metrics
    return z


def validate_pca(report: dict[str, Any], z: np.ndarray | None, seed: int, max_rows: int) -> None:
    section("2) PCA")
    metrics: dict[str, Any] = {}

    if z is None:
        failed("No latent vectors available")
        report["pca"] = metrics
        return

    idx = sample_idx(len(z), max_rows=max_rows, seed=seed)
    z_eval = z[idx].astype(np.float32)

    n_components = min(3, z_eval.shape[1], z_eval.shape[0])
    if n_components < 2:
        failed("Not enough latent dimensions/rows for PCA")
        report["pca"] = metrics
        return

    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(z_eval)
    z_recon = pca.inverse_transform(z_pca)

    var_ratio = pca.explained_variance_ratio_
    cumulative = float(np.sum(var_ratio))
    recon_mse = float(np.mean((z_eval - z_recon) ** 2))

    metrics["eval_rows"] = int(len(z_eval))
    metrics["explained_variance_ratio"] = [float(v) for v in var_ratio]
    metrics["cumulative_variance"] = cumulative
    metrics["reconstruction_mse"] = recon_mse

    info(f"cumulative_variance={cumulative:.4f}, reconstruction_mse={recon_mse:.6f}")

    if cumulative >= 0.50:
        passed("PCA variance capture is good")
    else:
        warning("PCA variance capture is low")

    report["pca"] = metrics


def validate_scraper(report: dict[str, Any], seed: int, max_texts: int) -> None:
    section("3) Scraper Categorisation Model")
    metrics: dict[str, Any] = {}

    profiles = load_json(SCRAPER_DIR / "cluster_profiles.json")
    thresholds = load_json(SCRAPER_DIR / "thresholds.json")
    metadata = load_json(SCRAPER_DIR / "model_metadata.json")

    model_path = SCRAPER_DIR / "model.keras"
    kmeans_path = SCRAPER_DIR / "kmeans.joblib"

    required_ok = model_path.exists() and kmeans_path.exists() and profiles is not None and thresholds is not None
    metrics["artifacts_ready"] = bool(required_ok)

    if not required_ok:
        failed("Scraper artifacts incomplete")
        report["scraper_model"] = metrics
        return

    passed("Core scraper artifacts are present")

    sizes = []
    for v in (profiles or {}).values():
        if isinstance(v, dict):
            sizes.append(int(v.get("size", 0)))
    sizes = [s for s in sizes if s > 0]

    if sizes:
        ent = entropy_from_counts(np.asarray(sizes))
        norm_ent = ent / max(math.log2(len(sizes)), 1e-9)
        metrics["cluster_count"] = int(len(sizes))
        metrics["cluster_entropy"] = float(ent)
        metrics["cluster_entropy_normalized"] = float(norm_ent)
        info(f"clusters={len(sizes)}, entropy_norm={norm_ent:.4f}")
        if norm_ent > 0.60:
            passed("Cluster distribution looks balanced")
        else:
            warning("Cluster distribution is concentrated")

    # Optional quick runtime check.
    if tf is None:
        warning(tf_unavailable_reason() + "; skipping scraper runtime check")
        report["scraper_model"] = metrics
        return
    if joblib is None:
        warning("joblib unavailable; skipping scraper runtime check")
        report["scraper_model"] = metrics
        return

    try:
        keras_model = tf.keras.models.load_model(model_path)
        extractor = tf.keras.Model(keras_model.input, keras_model.get_layer("latent").output)
        kmeans = joblib.load(kmeans_path)
    except Exception as exc:
        warning(f"Could not load scraper runtime artifacts: {exc}")
        report["scraper_model"] = metrics
        return

    dataset = latest_bluesky_dataset()
    if dataset is None:
        warning("No Bluesky dataset found for runtime check")
        report["scraper_model"] = metrics
        return

    try:
        data = json.loads(dataset.read_text(encoding="utf-8"))
        texts = [str(x.get("text", "")).strip() for x in data if isinstance(x, dict)]
        texts = [t for t in texts if len(t) >= 5]
    except Exception as exc:
        warning(f"Could not read dataset {dataset.name}: {exc}")
        report["scraper_model"] = metrics
        return

    if not texts:
        warning("No valid text rows for scraper runtime check")
        report["scraper_model"] = metrics
        return

    rng = np.random.default_rng(seed)
    n_take = min(max_texts, len(texts))
    if len(texts) > n_take:
        rows = np.sort(rng.choice(len(texts), size=n_take, replace=False))
        sample_texts = [texts[int(i)] for i in rows]
    else:
        sample_texts = texts

    t0 = time.perf_counter()
    emb = extractor.predict(np.asarray(sample_texts, dtype=object), batch_size=128, verbose=0)
    labels = kmeans.predict(emb)
    elapsed = time.perf_counter() - t0

    ms_per_post = float(1000.0 * elapsed / max(len(sample_texts), 1))
    cluster_util = float(len(np.unique(labels)) / max(int(getattr(kmeans, "n_clusters", 1)), 1))

    metrics["runtime_rows"] = int(len(sample_texts))
    metrics["runtime_ms_per_post"] = ms_per_post
    metrics["runtime_cluster_utilization"] = cluster_util
    metrics["thresholds"] = thresholds
    if metadata is not None:
        metrics["metadata"] = {
            "trained_at": metadata.get("trained_at"),
            "num_posts": metadata.get("num_posts"),
            "num_clusters": metadata.get("num_clusters"),
        }

    info(f"runtime_ms_per_post={ms_per_post:.3f}, cluster_utilization={cluster_util:.3f}")

    if ms_per_post <= 20.0:
        passed("Scraper runtime latency is good")
    else:
        warning("Scraper runtime latency is high")

    report["scraper_model"] = metrics


def validate_mapping(report: dict[str, Any]) -> None:
    section("4) Mapping Network")
    metrics: dict[str, Any] = {}

    pred = load_npy(MAPPING_DIR / "theta_pred.npy")
    tgt = load_npy(MAPPING_DIR / "theta_targets.npy")
    summary = load_json(MAPPING_DIR / "mapping_training_summary.json")

    if pred is None or tgt is None:
        failed("theta_pred/theta_targets missing")
        report["mapping_network"] = metrics
        return

    if pred.ndim != 2 or tgt.ndim != 2:
        failed(f"Invalid theta shapes: pred={pred.shape}, target={tgt.shape}")
        report["mapping_network"] = metrics
        return

    n = min(len(pred), len(tgt))
    pred = pred[-n:]
    tgt = tgt[-n:]

    mse = float(np.mean((pred - tgt) ** 2))
    mae = float(np.mean(np.abs(pred - tgt)))
    vmin = float(np.min(pred))
    vmax = float(np.max(pred))

    metrics["rows"] = int(n)
    metrics["mse"] = mse
    metrics["mae"] = mae
    metrics["pred_range"] = [vmin, vmax]
    if summary is not None:
        metrics["train_mse_summary"] = float(((summary.get("training") or {}).get("train_mse", np.nan)))

    info(f"mse={mse:.6f}, mae={mae:.6f}, range=[{vmin:.4f}, {vmax:.4f}]")

    if mse <= 0.02:
        passed("Mapping error is within expected range")
    else:
        warning("Mapping MSE is higher than expected")

    if vmin >= -1e-6 and vmax <= 1.0 + 1e-6:
        passed("Mapping outputs are in [0, 1]")
    else:
        warning("Mapping outputs exceed [0, 1]")

    report["mapping_network"] = metrics


def validate_latent_mapper(report: dict[str, Any], max_steps: int) -> None:
    section("5) Latent Visual Mapper")
    metrics: dict[str, Any] = {}

    cfg = load_json(LATENT_MAPPER_DIR / "latent_mapper_config.json")
    w = LATENT_MAPPER_DIR / "latent_mapper.weights.h5"

    if cfg is None or not w.exists():
        failed("Latent mapper config/weights missing")
        report["latent_visual_mapper"] = metrics
        return

    metrics["config"] = {
        "pca_dim": cfg.get("pca_dim"),
        "param_names": cfg.get("param_names"),
    }
    passed("Latent mapper artifacts found")

    if tf is None:
        warning(tf_unavailable_reason() + "; skipping latent mapper runtime check")
        report["latent_visual_mapper"] = metrics
        return

    try:
        from AI_systems.latent_visual_mapper import StreamingLatentVisualMapper
    except Exception as exc:
        warning(f"Could not import latent_visual_mapper: {exc}")
        report["latent_visual_mapper"] = metrics
        return

    z = load_npy(VAE_DIR / "latent_vectors.npy")
    if z is None or z.ndim != 2 or len(z) < 20:
        warning("Insufficient latent vectors for mapper runtime check")
        report["latent_visual_mapper"] = metrics
        return

    pca_dim = int(cfg.get("pca_dim", 3))
    if z.shape[1] < pca_dim:
        warning("Latent dim is smaller than mapper pca_dim")
        report["latent_visual_mapper"] = metrics
        return

    z_stream = PCA(n_components=pca_dim).fit_transform(z[-max(200, max_steps):]).astype(np.float32)

    mapper = StreamingLatentVisualMapper(model_dir=LATENT_MAPPER_DIR, pca_dim=pca_dim, param_names=cfg.get("param_names"))
    if not mapper.load():
        warning("Latent mapper failed to load")
        report["latent_visual_mapper"] = metrics
        return

    steps = min(max_steps, len(z_stream))
    t0 = time.perf_counter()
    out_ranges = []
    for row in z_stream[-steps:]:
        params = mapper.process_stream_step(row)
        vals = np.asarray(list(params.values()), dtype=np.float32)
        out_ranges.append([float(np.min(vals)), float(np.max(vals))])
    elapsed = time.perf_counter() - t0

    mins = [x[0] for x in out_ranges]
    maxs = [x[1] for x in out_ranges]

    ms_per_step = float(1000.0 * elapsed / max(steps, 1))
    metrics["runtime_steps"] = int(steps)
    metrics["runtime_ms_per_step"] = ms_per_step
    metrics["output_min"] = float(np.min(mins))
    metrics["output_max"] = float(np.max(maxs))

    info(
        f"runtime_ms_per_step={ms_per_step:.3f}, "
        f"output_range=[{metrics['output_min']:.4f}, {metrics['output_max']:.4f}]"
    )

    if ms_per_step <= 20.0:
        passed("Latent mapper step latency is good")
    else:
        warning("Latent mapper step latency is high")

    if metrics["output_min"] >= -1e-6 and metrics["output_max"] <= 1.0 + 1e-6:
        passed("Latent mapper outputs are in [0, 1]")
    else:
        warning("Latent mapper outputs exceed [0, 1]")

    report["latent_visual_mapper"] = metrics


def write_summary(report: dict[str, Any], report_out: Path, started_at: float) -> None:
    section("Summary")
    elapsed = time.time() - started_at
    print(f"Pass={COUNTS['pass']}  Warn={COUNTS['warn']}  Fail={COUNTS['fail']}  Elapsed={elapsed:.2f}s")
    print(f"Report: {report_out}")

    report["summary"] = {
        "pass": COUNTS["pass"],
        "warn": COUNTS["warn"],
        "fail": COUNTS["fail"],
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple, fast, and effective model validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pca-rows", type=int, default=8000)
    parser.add_argument("--max-texts", type=int, default=120)
    parser.add_argument("--max-mapper-steps", type=int, default=120)
    parser.add_argument("--report-out", default="AI_systems/model_validation_report.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    if tf is not None:
        try:
            tf.random.set_seed(args.seed)
        except Exception:
            pass

    started_at = time.time()

    section("Generative Data Visualisation - Simple Validation")
    info(f"UTC time: {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}")

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "validation_mode": "simple",
        "models": [
            "vae",
            "pca",
            "scraper_model",
            "mapping_network",
            "latent_visual_mapper",
        ],
    }

    validate_assets(report)
    z = validate_vae(report)
    validate_pca(report, z=z, seed=args.seed, max_rows=args.max_pca_rows)
    validate_scraper(report, seed=args.seed, max_texts=args.max_texts)
    validate_mapping(report)
    validate_latent_mapper(report, max_steps=args.max_mapper_steps)

    report_out = Path(args.report_out)
    if not report_out.is_absolute():
        report_out = ROOT / report_out
    report_out.parent.mkdir(parents=True, exist_ok=True)

    write_summary(report, report_out=report_out, started_at=started_at)

    with report_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
