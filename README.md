# Generative Data Visualisation Project

## Overview
This project builds a multimodal Bitcoin intelligence-to-visualisation system.

It combines:
- Market microstructure from Binance.
- Social signal from Bluesky posts.
- Representation learning (VAE + PCA + online mapper).
- A live generative renderer driven by interpretable market semantics.

The core outcome is a real-time and timeline-based visual experience where model outputs are mapped to visual controls (`motion_intensity`, `particle_density`, `distortion_strength`, `noise_scale`, `color_dynamics`).

## Project Aim
The aim is to turn high-frequency market and social dynamics into an interpretable latent space, then render that latent behavior as a generative visual narrative.

In practical terms, the system is designed to:
- Learn structure from BTC + social data without manual labels.
- Track regime behavior online.
- Keep visual outputs semantically anchored to market condition factors.
- Support both historical playback and live streaming.

## What Is Implemented

### 1) Data acquisition and preprocessing
- Historical Binance BTC candles (30m shards) via `Data_Pipeline/BTC_datapipeline.py`.
- Bluesky static snapshot pipeline via `Data_Pipeline/Static_Bluesky.py`.
- Longer-range Bluesky historical scraper with AI enrichment via `AI_systems/bitcoin_blusky_pipeline.py`.
- Real-time Binance stream ingestion via `Data_Pipeline/Real_time_Crypto.py`.

### 2) Feature engineering
- `Data_Pipeline/feature_matrix.py` aligns market and social windows and exports:
	- `market_features.npy`
	- `social_features.npy`
	- `cross_features.npy`
	- `full_features.npy`
	- optional temporal context variants (`full_features_ctx_w3.npy`, `full_features_ctx_w6.npy`, ...)

### 3) Learning stack
- Unsupervised scraper model (`AI_systems/train_unsupervised_scraper_model.py`): text CNN autoencoder-style latent + KMeans clusters + cluster profiles.
- VAE (`AI_systems/vae_model.py`): compresses multimodal features to latent vectors.
- PCA (`AI_systems/pca_model.py` and live runner): projects rolling features to low-dimensional online latent state.
- Mapping network (`AI_systems/mapping_network.py`): maps VAE latent to interpretable theta targets.
- Streaming latent mapper (`AI_systems/latent_visual_mapper.py`): online adaptation from live PCA latent to visual parameters with regime tracking.

### 4) Visual systems
- Live integrated pipeline: `Generative_visualisation/live_btc_visual_pipeline.py`.
- Optional live PCA viewer: `Generative_visualisation/live_btc_pca_visual.py`.
- Historical timeline viewer: `Generative_visualisation/latent_timeline_visual_engine.py`.
- Real-time renderer core: `Generative_visualisation/visual_engine.py`.

### 5) Validation
- Unified model checks and report generation via `AI_systems/validate_models.py`.

## Repository Layout
- `AI_systems/`: model training, mapping, latent online adaptation, validation.
- `Data_Pipeline/`: data ingestion and feature matrix generation.
- `Generative_visualisation/`: live and timeline visualisation engines.
- `vae_model/data*`: exported aligned feature tensors and timestamps.
- `AI_systems/*_artifacts/`: trained model weights and metadata.

## Environment Setup

### Prerequisites
- Python 3.10+
- Windows/macOS/Linux

### Create environment
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### Install dependencies
There is no pinned requirements file in this repo yet, so install the core packages directly:

```bash
pip install numpy matplotlib scikit-learn joblib tensorflow python-binance atproto sentence-transformers
```

## Credentials
For authenticated Bluesky scraping, set:
- `BLUESKY_HANDLE`
- `BLUESKY_APP_PASSWORD`

If unset, some scripts fall back to public mode or interactive prompts.

## Recommended End-to-End Workflow

### Step 1: Train scraper artifacts (optional if already present)
```bash
python AI_systems/train_unsupervised_scraper_model.py --input-json Data_Pipeline/datasets/bitcoin_bluesky_jan2025_jun2025.json
```

### Step 2: Build aligned feature tensors
```bash
python Data_Pipeline/feature_matrix.py
```

### Step 3: Train VAE and export latent vectors
```bash
python AI_systems/vae_model.py --data-dir vae_model/data --epochs 30 --batch-size 128
```

### Step 4: Train latent-to-theta mapping network
```bash
python AI_systems/mapping_network.py --latent-path AI_systems/vae_artifacts/latent_vectors.npy --features-dir vae_model/data
```

### Step 5: Run validation report
```bash
python AI_systems/validate_models.py --report-out AI_systems/model_validation_report.json
```

### Step 6: Run live visual pipeline
```bash
python Generative_visualisation/live_btc_visual_pipeline.py
```

## Main Run Modes

### Live integrated visual (recommended)
```bash
python Generative_visualisation/live_btc_visual_pipeline.py
```
Flow: Binance stream -> feature extraction -> PCA(3D) -> streaming latent mapper -> generative renderer.

### Live PCA-only mode
```bash
python Generative_visualisation/live_btc_pca_visual.py
```

### Historical timeline playback
```bash
python Generative_visualisation/latent_timeline_visual_engine.py \
	--latent AI_systems/vae_artifacts/latent_vectors.npy \
	--theta AI_systems/mapping_network_artifacts/theta_pred.npy \
	--timestamps vae_model/data_full_2023_2025_check/timestamps.npy
```

## Data and Model Contracts

### Feature dimensions
- Market features: 12
- Social features: 27
- Cross features: 3
- Full input to VAE: 42

### Theta semantics (5 outputs)
- `theta[0]`: turbulence
- `theta[1]`: trend/fear-greed color axis
- `theta[2]`: distortion / regime irregularity
- `theta[3]`: fragmentation / structural dispersion
- `theta[4]`: velocity / state-change speed

### Visual parameter keys
- `motion_intensity`
- `particle_density`
- `distortion_strength`
- `noise_scale`
- `color_dynamics`

## Artifacts Produced

### Scraper artifacts
- `AI_systems/scraper_model_artifacts/model.keras`
- `AI_systems/scraper_model_artifacts/kmeans.joblib`
- `AI_systems/scraper_model_artifacts/cluster_profiles.json`

### VAE artifacts
- `AI_systems/vae_artifacts/vae_weights.weights.h5`
- `AI_systems/vae_artifacts/latent_vectors.npy`

### Mapping artifacts
- `AI_systems/mapping_network_artifacts/mapping_network.weights.h5`
- `AI_systems/mapping_network_artifacts/theta_targets.npy`
- `AI_systems/mapping_network_artifacts/theta_pred.npy`

### Online mapper artifacts
- `AI_systems/latent_mapper_artifacts/latent_mapper.weights.h5`
- `AI_systems/latent_mapper_artifacts/latent_mapper_config.json`

### Validation output
- `AI_systems/model_validation_report.json`

## Troubleshooting
- If `tensorflow` import fails, validation will skip TF-dependent checks.
- If live stream drops, reconnect wrappers in live scripts retry automatically.
- If Bluesky auth fails, verify app password (not account password).
- If you see shape mismatch warnings, ensure feature exports and latent vectors come from the same run window.

## Current Status
This repository already includes trained artifacts and multi-period datasets (2023-2025) so you can run timeline visualisation and validation immediately, then retrain components incrementally as needed.