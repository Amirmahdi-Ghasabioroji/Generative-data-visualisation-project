"""
Streaming unsupervised mapper from 3D PCA latent vectors to generative visual parameters.

This module is designed for live Bitcoin PCA streams and learns online without labels.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

keras = tf.keras
layers = keras.layers
regularizers = keras.regularizers


PARAMETER_NAMES = [
    "motion_intensity",
    "particle_density",
    "distortion_strength",
    "noise_scale",
    "color_dynamics",
]


@dataclass
class AutoConfig:
    activation: str
    optimizer_name: str
    learning_rate: float
    l2_strength: float
    hidden_units: Tuple[int, int]
    bottleneck_dim: int


class UnsupervisedLatentMapper(keras.Model):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: AutoConfig,
        smoothness_weight: float = 0.10,
        variance_weight: float = 0.08,
        target_variance: float = 0.020,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.smoothness_weight = smoothness_weight
        self.variance_weight = variance_weight
        self.target_variance = target_variance

        reg = regularizers.l2(config.l2_strength)

        self.encoder = keras.Sequential(
            [
                layers.Input(shape=(input_dim,)),
                layers.Dense(config.hidden_units[0], activation=config.activation, kernel_regularizer=reg),
                layers.Dense(config.hidden_units[1], activation=config.activation, kernel_regularizer=reg),
                layers.Dense(config.bottleneck_dim, activation="linear", name="bottleneck"),
            ],
            name="encoder",
        )

        self.decoder = keras.Sequential(
            [
                layers.Input(shape=(config.bottleneck_dim,)),
                layers.Dense(config.hidden_units[1], activation=config.activation, kernel_regularizer=reg),
                layers.Dense(config.hidden_units[0], activation=config.activation, kernel_regularizer=reg),
                layers.Dense(input_dim, activation="linear", name="z_reconstruction"),
            ],
            name="decoder",
        )

        self.visual_head = keras.Sequential(
            [
                layers.Input(shape=(config.bottleneck_dim,)),
                layers.Dense(config.hidden_units[1], activation=config.activation, kernel_regularizer=reg),
                layers.Dense(output_dim, activation="sigmoid", name="visual_parameters_0_1"),
            ],
            name="visual_head",
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.smooth_tracker = keras.metrics.Mean(name="smoothness_loss")
        self.var_tracker = keras.metrics.Mean(name="variance_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.smooth_tracker, self.var_tracker]

    def call(self, inputs, training=False):
        h = self.encoder(inputs, training=training)
        z_hat = self.decoder(h, training=training)
        params = self.visual_head(h, training=training)
        return z_hat, params

    def train_step(self, data):
        x = tf.cast(data, tf.float32)
        if len(x.shape) != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input shape (batch, {self.input_dim}), got {x.shape}")

        noise_scale = 0.02
        x_noisy = x + tf.random.normal(tf.shape(x), stddev=noise_scale)

        with tf.GradientTape() as tape:
            z_hat, params = self(x_noisy, training=True)
            recon_loss = tf.reduce_mean(tf.square(x - z_hat))

            diff_params = params[1:] - params[:-1]
            diff_input = x[1:] - x[:-1]
            input_speed = tf.norm(diff_input, axis=1, keepdims=True) + 1e-4
            smoothness_loss = tf.reduce_mean(tf.square(diff_params) / input_speed)

            param_var = tf.math.reduce_variance(params, axis=0)
            variance_loss = tf.reduce_mean(tf.nn.relu(self.target_variance - param_var))

            total_loss = recon_loss + self.smoothness_weight * smoothness_loss + self.variance_weight * variance_loss
            total_loss += tf.add_n(self.losses) if self.losses else 0.0

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.recon_tracker.update_state(recon_loss)
        self.smooth_tracker.update_state(smoothness_loss)
        self.var_tracker.update_state(variance_loss)

        return {
            "loss": self.loss_tracker.result(),
            "reconstruction_loss": self.recon_tracker.result(),
            "smoothness_loss": self.smooth_tracker.result(),
            "variance_loss": self.var_tracker.result(),
        }


class StreamingLatentVisualMapper:
    def __init__(
        self,
        model_dir: str | Path,
        pca_dim: int = 3,
        param_names: Optional[List[str]] = None,
        n_regimes: int = 3,
        stream_buffer_size: int = 512,
        train_window: int = 128,
        train_every: int = 8,
        traversal_steps: int = 6,
    ):
        self.pca_dim = pca_dim
        self.param_names = param_names or PARAMETER_NAMES
        self.output_dim = len(self.param_names)

        self.stream_buffer: Deque[np.ndarray] = deque(maxlen=stream_buffer_size)
        self.train_window = train_window
        self.train_every = train_every
        self.traversal_steps = traversal_steps
        self.step_count = 0
        self.n_regimes = max(2, int(n_regimes))

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.model_dir / "latent_mapper.weights.h5"
        self.config_path = self.model_dir / "latent_mapper_config.json"

        self.model: Optional[UnsupervisedLatentMapper] = None
        self.auto_config: Optional[AutoConfig] = None
        self.regime_centroids: Optional[np.ndarray] = None
        self.regime_counts: Optional[np.ndarray] = None
        self.latest_regime_id: Optional[int] = None
        self.latest_regime_confidence: float = 0.0
        self.smoothed_regime_confidence: float = 0.0

        # Streaming z normalization state (persisted).
        self.z_mean = np.zeros((self.pca_dim,), dtype=np.float32)
        self.z_var = np.ones((self.pca_dim,), dtype=np.float32)
        self.z_count = 0
        self.z_ema_alpha = 0.02

        # Regime stability controls.
        self.regime_switch_margin = 0.03
        self.confidence_ema_alpha = 0.20

    def _normalize_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        std = np.sqrt(np.maximum(self.z_var, 1e-6)).astype(np.float32)
        return ((z - self.z_mean) / std).astype(np.float32)

    def _update_z_stats(self, z_t: np.ndarray):
        z_t = np.asarray(z_t, dtype=np.float32).reshape(-1)
        if z_t.shape[0] != self.pca_dim:
            return

        if self.z_count == 0:
            self.z_mean = z_t.astype(np.float32)
            self.z_var = np.full((self.pca_dim,), 1e-2, dtype=np.float32)
            self.z_count = 1
            return

        alpha = float(self.z_ema_alpha)
        delta = z_t - self.z_mean
        self.z_mean = (1.0 - alpha) * self.z_mean + alpha * z_t
        self.z_var = (1.0 - alpha) * self.z_var + alpha * np.square(delta)
        self.z_var = np.maximum(self.z_var, 1e-6).astype(np.float32)
        self.z_count += 1

    def _bottleneck_batch(self, z_batch: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Run warmup_train(...) or load().")
        z_batch = np.asarray(z_batch, dtype=np.float32)
        if z_batch.ndim != 2 or z_batch.shape[1] != self.pca_dim:
            raise ValueError(f"Expected z_batch shape (N, {self.pca_dim}).")
        z_batch_n = self._normalize_z(z_batch)
        h = self.model.encoder(z_batch_n, training=False)
        return np.asarray(h.numpy(), dtype=np.float32)

    def _initialize_regimes(self, z_sequence: np.ndarray):
        h = self._bottleneck_batch(z_sequence)
        if h.shape[0] == 0:
            return

        order = np.argsort(h[:, 0])
        ordered = h[order]
        pick_idx = np.linspace(0, max(0, ordered.shape[0] - 1), self.n_regimes).astype(int)
        centroids = ordered[pick_idx]

        if centroids.shape[0] < self.n_regimes:
            repeats = self.n_regimes - centroids.shape[0]
            centroids = np.vstack([centroids, np.repeat(centroids[-1:], repeats, axis=0)])

        self.regime_centroids = np.asarray(centroids, dtype=np.float32)
        self.regime_counts = np.ones((self.n_regimes,), dtype=np.float32)

    def _assign_regime(self, h_t: np.ndarray, current_regime_id: Optional[int] = None) -> tuple[int, float]:
        if self.regime_centroids is None:
            raise RuntimeError("Regime centroids are not initialized.")

        distances = np.linalg.norm(self.regime_centroids - h_t[None, :], axis=1)
        sorted_idx = np.argsort(distances)
        best_id = int(sorted_idx[0])
        best_dist = float(distances[best_id])
        second_dist = float(distances[sorted_idx[1]]) if len(sorted_idx) > 1 else best_dist + 1e-6

        # Separation-aware confidence scaled by overall local distance spread.
        # This avoids chronically tiny values when clusters are close in absolute terms.
        separation = second_dist - best_dist
        spread_scale = float(np.mean(distances) + np.std(distances) + 1e-6)
        sep_score = separation / spread_scale
        confidence = float(1.0 / (1.0 + np.exp(-8.0 * sep_score)))
        confidence = float(np.clip((confidence - 0.5) * 2.0, 0.0, 1.0))

        return best_id, confidence

    def _update_regime_centroid(self, regime_id: int, h_t: np.ndarray):
        if self.regime_centroids is None or self.regime_counts is None:
            return

        self.regime_counts[regime_id] += 1.0
        adaptive_lr = max(0.02, 1.0 / float(self.regime_counts[regime_id]))
        self.regime_centroids[regime_id] = (
            (1.0 - adaptive_lr) * self.regime_centroids[regime_id] + adaptive_lr * h_t
        )

    def _online_regime_step(self, z_t: np.ndarray):
        if self.model is None:
            return

        if self.regime_centroids is None:
            if len(self.stream_buffer) >= 1:
                boot = np.asarray(list(self.stream_buffer), dtype=np.float32)
                self._initialize_regimes(boot)
            else:
                self.latest_regime_id = None
                self.latest_regime_confidence = 0.0
                return

        h_t = self._bottleneck_batch(np.asarray(z_t, dtype=np.float32).reshape(1, -1))[0]
        regime_id, confidence = self._assign_regime(h_t, current_regime_id=self.latest_regime_id)

        # Hysteresis: avoid rapid flips unless confidence margin is meaningful.
        if self.latest_regime_id is not None and regime_id != self.latest_regime_id and confidence < self.regime_switch_margin:
            regime_id = int(self.latest_regime_id)

        self._update_regime_centroid(regime_id, h_t)
        self.latest_regime_id = regime_id
        self.latest_regime_confidence = float(confidence)
        self.smoothed_regime_confidence = float(
            (1.0 - self.confidence_ema_alpha) * self.smoothed_regime_confidence
            + self.confidence_ema_alpha * confidence
        )

    def get_latest_regime_info(self) -> Dict[str, Any]:
        return {
            "regime_id": self.latest_regime_id,
            "confidence": float(self.smoothed_regime_confidence),
            "raw_confidence": float(self.latest_regime_confidence),
            "n_regimes": int(self.n_regimes),
        }

    def _auto_choose_config(self, z_train: np.ndarray) -> AutoConfig:
        z_std = float(np.std(z_train))
        z_range = float(np.max(z_train) - np.min(z_train))

        activation = "tanh" if z_range <= 4.0 else "swish"

        if z_std < 0.10:
            learning_rate = 2e-3
            optimizer_name = "adam"
            l2_strength = 1e-4
            hidden_units = (16, 12)
            bottleneck_dim = 2
        elif z_std < 0.50:
            learning_rate = 1e-3
            optimizer_name = "adam"
            l2_strength = 5e-4
            hidden_units = (24, 16)
            bottleneck_dim = 2
        else:
            learning_rate = 8e-4
            optimizer_name = "adamw"
            l2_strength = 1e-3
            hidden_units = (32, 24)
            bottleneck_dim = 2

        return AutoConfig(
            activation=activation,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            l2_strength=l2_strength,
            hidden_units=hidden_units,
            bottleneck_dim=bottleneck_dim,
        )

    def _make_optimizer(self, cfg: AutoConfig):
        if cfg.optimizer_name == "adamw":
            try:
                return keras.optimizers.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.l2_strength)
            except Exception:
                return keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        return keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    def _build_model(self, cfg: AutoConfig):
        model = UnsupervisedLatentMapper(
            input_dim=self.pca_dim,
            output_dim=self.output_dim,
            config=cfg,
            smoothness_weight=0.10,
            variance_weight=0.08,
            target_variance=0.020,
        )
        model.compile(optimizer=self._make_optimizer(cfg), run_eagerly=False)

        dummy = np.zeros((2, self.pca_dim), dtype=np.float32)
        model(dummy)
        self.model = model
        self.auto_config = cfg

    def warmup_train(self, z_sequence: np.ndarray, epochs: int = 40, batch_size: int = 32, verbose: int = 0):
        z_sequence = np.asarray(z_sequence, dtype=np.float32)
        if z_sequence.ndim != 2 or z_sequence.shape[1] != self.pca_dim:
            raise ValueError(f"z_sequence must have shape (N, {self.pca_dim}).")
        if z_sequence.shape[0] < 8:
            raise ValueError("Need at least 8 latent rows for warmup training.")

        self.z_mean = np.mean(z_sequence, axis=0).astype(np.float32)
        self.z_var = np.var(z_sequence, axis=0).astype(np.float32) + 1e-6
        self.z_count = int(z_sequence.shape[0])
        z_norm = self._normalize_z(z_sequence)

        cfg = self._auto_choose_config(z_norm)
        self._build_model(cfg)

        self.model.fit(z_norm, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose)
        self.stream_buffer.clear()
        for row in z_sequence[-self.stream_buffer.maxlen:]:
            self.stream_buffer.append(np.asarray(row, dtype=np.float32))

        self._initialize_regimes(z_sequence)
        self.save()

    def partial_update(self, epochs: int = 2, batch_size: int = 32, verbose: int = 0):
        if self.model is None:
            return
        if len(self.stream_buffer) < max(16, self.train_window):
            return

        z_window = np.asarray(list(self.stream_buffer)[-self.train_window:], dtype=np.float32)
        if float(np.mean(np.var(z_window, axis=0))) < 1e-4:
            return
        z_window_n = self._normalize_z(z_window)
        self.model.fit(z_window_n, epochs=epochs, batch_size=min(batch_size, len(z_window_n)), shuffle=False, verbose=verbose)

    def interpolate_latents(self, z_prev: np.ndarray, z_curr: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        z_prev = np.asarray(z_prev, dtype=np.float32).reshape(-1)
        z_curr = np.asarray(z_curr, dtype=np.float32).reshape(-1)
        if z_prev.shape[0] != self.pca_dim or z_curr.shape[0] != self.pca_dim:
            raise ValueError(f"Both z_prev and z_curr must be length {self.pca_dim}.")

        n_steps = steps or self.traversal_steps
        alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
        return np.stack([(1.0 - a) * z_prev + a * z_curr for a in alphas], axis=0)

    def latent_to_visual_parameters(self, z: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Run warmup_train(...) or load().")

        z = np.asarray(z, dtype=np.float32).reshape(1, -1)
        if z.shape[1] != self.pca_dim:
            raise ValueError(f"z must have {self.pca_dim} components.")
        z = self._normalize_z(z)

        _, params = self.model(z, training=False)
        params_np = params.numpy()[0]
        return {name: float(value) for name, value in zip(self.param_names, params_np)}

    def traversal_parameters(self, z_prev: np.ndarray, z_curr: np.ndarray, steps: Optional[int] = None) -> List[Dict[str, float]]:
        z_interp = self.interpolate_latents(z_prev, z_curr, steps)
        return [self.latent_to_visual_parameters(z) for z in z_interp]

    def process_stream_step(self, z_t: np.ndarray) -> Dict[str, float]:
        z_t = np.asarray(z_t, dtype=np.float32).reshape(-1)
        if z_t.shape[0] != self.pca_dim:
            raise ValueError(f"z_t must have length {self.pca_dim}.")
        if self.model is None:
            raise RuntimeError("Model is not initialized. Run warmup_train(...) or load().")

        self._update_z_stats(z_t)

        self.stream_buffer.append(z_t)
        self.step_count += 1

        if self.step_count % self.train_every == 0:
            self.partial_update(epochs=1, batch_size=32, verbose=0)

        self._online_regime_step(z_t)

        return self.latent_to_visual_parameters(z_t)

    def save(self):
        if self.model is None or self.auto_config is None:
            return

        self.model.save_weights(str(self.weights_path))
        config_payload = {
            "pca_dim": self.pca_dim,
            "param_names": self.param_names,
            "n_regimes": self.n_regimes,
            "auto_config": {
                "activation": self.auto_config.activation,
                "optimizer_name": self.auto_config.optimizer_name,
                "learning_rate": self.auto_config.learning_rate,
                "l2_strength": self.auto_config.l2_strength,
                "hidden_units": list(self.auto_config.hidden_units),
                "bottleneck_dim": self.auto_config.bottleneck_dim,
            },
            "regime_state": {
                "centroids": self.regime_centroids.tolist() if self.regime_centroids is not None else None,
                "counts": self.regime_counts.tolist() if self.regime_counts is not None else None,
                "latest_regime_id": self.latest_regime_id,
                "latest_regime_confidence": float(self.latest_regime_confidence),
                "smoothed_regime_confidence": float(self.smoothed_regime_confidence),
            },
            "z_norm_state": {
                "mean": self.z_mean.tolist(),
                "var": self.z_var.tolist(),
                "count": int(self.z_count),
            },
        }
        self.config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    def load(self) -> bool:
        if not self.config_path.exists() or not self.weights_path.exists():
            return False

        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.n_regimes = int(payload.get("n_regimes", self.n_regimes))
        cfg_raw = payload["auto_config"]
        cfg = AutoConfig(
            activation=cfg_raw["activation"],
            optimizer_name=cfg_raw["optimizer_name"],
            learning_rate=float(cfg_raw["learning_rate"]),
            l2_strength=float(cfg_raw["l2_strength"]),
            hidden_units=(int(cfg_raw["hidden_units"][0]), int(cfg_raw["hidden_units"][1])),
            bottleneck_dim=int(cfg_raw["bottleneck_dim"]),
        )

        self._build_model(cfg)
        self.model.load_weights(str(self.weights_path))

        regime_state = payload.get("regime_state", {})
        centroids = regime_state.get("centroids") if isinstance(regime_state, dict) else None
        counts = regime_state.get("counts") if isinstance(regime_state, dict) else None

        if centroids is not None:
            arr = np.asarray(centroids, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == self.n_regimes:
                self.regime_centroids = arr
        if counts is not None:
            arr = np.asarray(counts, dtype=np.float32).reshape(-1)
            if arr.shape[0] == self.n_regimes:
                self.regime_counts = arr

        self.latest_regime_id = regime_state.get("latest_regime_id") if isinstance(regime_state, dict) else None
        self.latest_regime_confidence = float(regime_state.get("latest_regime_confidence", 0.0)) if isinstance(regime_state, dict) else 0.0
        self.smoothed_regime_confidence = float(regime_state.get("smoothed_regime_confidence", self.latest_regime_confidence)) if isinstance(regime_state, dict) else self.latest_regime_confidence

        z_norm_state = payload.get("z_norm_state", {})
        if isinstance(z_norm_state, dict):
            mean_arr = np.asarray(z_norm_state.get("mean", self.z_mean.tolist()), dtype=np.float32).reshape(-1)
            var_arr = np.asarray(z_norm_state.get("var", self.z_var.tolist()), dtype=np.float32).reshape(-1)
            if mean_arr.shape[0] == self.pca_dim:
                self.z_mean = mean_arr
            if var_arr.shape[0] == self.pca_dim:
                self.z_var = np.maximum(var_arr, 1e-6)
            self.z_count = int(z_norm_state.get("count", self.z_count))
        return True


_DEFAULT_MAPPER: Optional[StreamingLatentVisualMapper] = None


def set_default_mapper(mapper: StreamingLatentVisualMapper):
    global _DEFAULT_MAPPER
    _DEFAULT_MAPPER = mapper


def latent_to_visual_parameters(z: np.ndarray) -> Dict[str, float]:
    if _DEFAULT_MAPPER is None:
        raise RuntimeError("Default mapper is not set. Call set_default_mapper(mapper) first.")
    return _DEFAULT_MAPPER.latent_to_visual_parameters(z)
