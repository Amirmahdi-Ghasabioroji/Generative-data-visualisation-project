"""
Streaming unsupervised mapper from 3D PCA latent vectors to generative visual parameters.

This module is designed for live Bitcoin PCA streams and learns online without labels.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

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

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.model_dir / "latent_mapper.weights.h5"
        self.config_path = self.model_dir / "latent_mapper_config.json"

        self.model: Optional[UnsupervisedLatentMapper] = None
        self.auto_config: Optional[AutoConfig] = None

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

        cfg = self._auto_choose_config(z_sequence)
        self._build_model(cfg)

        self.model.fit(z_sequence, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose)
        self.save()

        self.stream_buffer.clear()
        for row in z_sequence[-self.stream_buffer.maxlen:]:
            self.stream_buffer.append(np.asarray(row, dtype=np.float32))

    def partial_update(self, epochs: int = 2, batch_size: int = 32, verbose: int = 0):
        if self.model is None:
            return
        if len(self.stream_buffer) < max(16, self.train_window):
            return

        z_window = np.asarray(list(self.stream_buffer)[-self.train_window:], dtype=np.float32)
        self.model.fit(z_window, epochs=epochs, batch_size=min(batch_size, len(z_window)), shuffle=False, verbose=verbose)

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

        self.stream_buffer.append(z_t)
        self.step_count += 1

        if self.step_count % self.train_every == 0:
            self.partial_update(epochs=1, batch_size=32, verbose=0)

        return self.latent_to_visual_parameters(z_t)

    def save(self):
        if self.model is None or self.auto_config is None:
            return

        self.model.save_weights(str(self.weights_path))
        config_payload = {
            "pca_dim": self.pca_dim,
            "param_names": self.param_names,
            "auto_config": {
                "activation": self.auto_config.activation,
                "optimizer_name": self.auto_config.optimizer_name,
                "learning_rate": self.auto_config.learning_rate,
                "l2_strength": self.auto_config.l2_strength,
                "hidden_units": list(self.auto_config.hidden_units),
                "bottleneck_dim": self.auto_config.bottleneck_dim,
            },
        }
        self.config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    def load(self) -> bool:
        if not self.config_path.exists() or not self.weights_path.exists():
            return False

        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
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
        return True


_DEFAULT_MAPPER: Optional[StreamingLatentVisualMapper] = None


def set_default_mapper(mapper: StreamingLatentVisualMapper):
    global _DEFAULT_MAPPER
    _DEFAULT_MAPPER = mapper


def latent_to_visual_parameters(z: np.ndarray) -> Dict[str, float]:
    if _DEFAULT_MAPPER is None:
        raise RuntimeError("Default mapper is not set. Call set_default_mapper(mapper) first.")
    return _DEFAULT_MAPPER.latent_to_visual_parameters(z)
