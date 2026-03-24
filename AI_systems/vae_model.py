"""
Variational Autoencoder (VAE)
Purpose: Compress the BTC market + Bluesky social feature matrix into a
         low-dimensional latent space for clustering, visualisation, and
         latent space traversal.
Stack:   TensorFlow + Keras + numpy
Mode:    Batch (train once on dataset, encode for downstream use)

Input layout — flat concatenated vector built from feature_matrix.py output:
    X = np.hstack([market_features, social_features, cross_features])
    Default shape: (n_samples, 42)
        market_features  → 12 cols  (log_return, RSI, volatility, volume, …)
        social_features  → 27 cols  (engagement, ML sentiment/cluster/topic signals, …)
        cross_features   →  3 cols  (sentiment×return, post_count×rv, …)

Architecture — dual-stream encoder, single decoder:
    Market stream  : 12 → Dense(64,relu)→BN → Dense(32,relu)→BN  →  (32,)
    Social stream  : 12 → Dense(64,relu)→BN → Dense(32,relu)→BN  →  (32,)
    Cross stream   :  3 → Dense(16,relu)→BN                       →  (16,)
    Merge          : Concat(80) → Dense(64,relu) → Dropout(0.1)
    Latent heads   : z_mean(16), z_log_var(16)
    Decoder        : 16 → Dense(64,relu)→BN → Dense(80,relu)→BN → Dense(42,linear)

Why dual-stream:
    Market (RSI, log-returns, volatility) and social (engagement, sentiment,
    post counts) have completely different statistical distributions. Routing
    them through separate Dense streams before merging lets each modality
    build its own representation first. The merged latent space then captures
    cross-modal relationships (e.g. sentiment diverging from price) without
    one modality dominating the gradients.

Dependencies:
    pip install tensorflow numpy
"""
import numpy as np
import tensorflow as tf
import argparse
import json
from pathlib import Path
keras = tf.keras
layers = tf.keras.layers


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING LAYER
# Implements the reparameterisation trick so gradients can flow through
# the stochastic sampling step during backpropagation.
# Formula: z = z_mean + ε · exp(0.5 · z_log_var),  ε ~ N(0, I)
# ═══════════════════════════════════════════════════════════════════════════════
class Sampling(layers.Layer):
    """Reparameterisation trick: z = z_mean + ε · exp(0.5 · z_log_var)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# VAE
# ═══════════════════════════════════════════════════════════════════════════════

class VAE(keras.Model):
    """
    Dual-stream Variational Autoencoder for BTC market + Bluesky social data.

    The encoder splits the flat input into three modality streams (market,
    social, cross), processes each independently, then merges them into a
    shared low-dimensional latent space. The decoder reconstructs the full
    flat vector from the latent sample.

    Parameters
    ----------
    market_dim   : number of market feature columns        (default 12)
    social_dim   : number of social feature columns        (default 27)
    cross_dim    : number of cross-modal feature columns   (default  3)
    latent_dim   : latent space size                       (default 16)
    dropout_rate : dropout on the encoder merge layer      (default 0.1)
    """

    def __init__(
        self,
        market_dim:   int   = 12,
        social_dim:   int   = 27,
        cross_dim:    int   = 3,
        latent_dim:   int   = 16,
        dropout_rate: float = 0.1,
        kl_anneal_steps: int = 2000,
        beta_start_ratio: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.market_dim   = market_dim
        self.social_dim   = social_dim
        self.cross_dim    = cross_dim
        self.input_dim    = market_dim + social_dim + cross_dim
        self.latent_dim   = latent_dim
        self.dropout_rate = dropout_rate
        self.kl_anneal_steps = int(max(1, kl_anneal_steps))
        self.beta_start_ratio = float(np.clip(beta_start_ratio, 0.0, 1.0))

        # β scales KL loss so its magnitude matches reconstruction loss.
        # Reconstruction sums over input_dim dims, KL over latent_dim dims —
        # without β the KL term is ~input_dim/latent_dim times too small,
        # causing posterior collapse (encoder ignores the prior).
        self.beta = self.input_dim / self.latent_dim
        self.beta_start = self.beta * self.beta_start_ratio
        self.train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.sampling = Sampling()
        self.encoder  = self._build_encoder()
        self.decoder  = self._build_decoder()

        self.total_loss_tracker          = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker             = keras.metrics.Mean(name="kl_loss")
        self.beta_tracker                = keras.metrics.Mean(name="beta")

    # ─────────────────────────────────────────────────────────────────────────
    # ENCODER
    # Three parallel streams handle each modality independently, then merge.
    #
    #  market_input (12) → Dense(64)→BN→Dense(32)→BN ─┐
    #  social_input (12) → Dense(64)→BN→Dense(32)→BN ─┼─→ Concat(80)→Dense(64)→Dropout
    #  cross_input  ( 3) → Dense(16)→BN              ─┘          │
    #                                                    z_mean(16), z_log_var(16)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_encoder(self) -> keras.Model:
        full_input = keras.Input(shape=(self.input_dim,), name="encoder_input")

        # Slice the flat vector into modality windows
        social_start = self.market_dim
        social_end   = self.market_dim + self.social_dim

        market_in = full_input[:, :social_start]
        social_in = full_input[:, social_start:social_end]
        cross_in  = full_input[:, social_end:]

        # ── Market stream: 12 → 64 → 32 ──
        m = layers.Dense(64, activation="relu", name="market_dense_1")(market_in)
        m = layers.BatchNormalization(name="market_bn_1")(m)
        m = layers.Dense(32, activation="relu", name="market_dense_2")(m)
        m = layers.BatchNormalization(name="market_bn_2")(m)

        # ── Social stream: 12 → 64 → 32 ──
        s = layers.Dense(64, activation="relu", name="social_dense_1")(social_in)
        s = layers.BatchNormalization(name="social_bn_1")(s)
        s = layers.Dense(32, activation="relu", name="social_dense_2")(s)
        s = layers.BatchNormalization(name="social_bn_2")(s)

        # ── Cross stream: 3 → 16 (smaller — derived interaction terms) ──
        c = layers.Dense(16, activation="relu", name="cross_dense_1")(cross_in)
        c = layers.BatchNormalization(name="cross_bn_1")(c)

        # ── Merge: Concat(32+32+16=80) → Dense(64) → Dropout ──
        merged = layers.Concatenate(name="modality_merge")([m, s, c])
        merged = layers.Dense(96, activation="relu", name="merge_dense_1")(merged)
        merged = layers.BatchNormalization(name="merge_bn_1")(merged)
        merged = layers.Dense(64, activation="relu", name="merge_dense_2")(merged)
        merged = layers.Dropout(self.dropout_rate, name="merge_dropout")(merged)

        # ── Latent distribution heads ──
        z_mean    = layers.Dense(self.latent_dim, name="z_mean")(merged)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(merged)

        return keras.Model(full_input, [z_mean, z_log_var], name="encoder")

    # ─────────────────────────────────────────────────────────────────────────
    # DECODER
    # Single stream — latent space is already fused across modalities.
    # Symmetric expansion back to full input_dim.
    #
    #  z(16) → Dense(64)→BN → Dense(80)→BN → Dense(input_dim, linear)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_decoder(self) -> keras.Model:
        z_input = keras.Input(shape=(self.latent_dim,), name="decoder_input")

        x = layers.Dense(96, activation="relu", name="dec_dense_1")(z_input)
        x = layers.BatchNormalization(name="dec_bn_1")(x)
        x = layers.Dense(80, activation="relu", name="dec_dense_2")(x)
        x = layers.BatchNormalization(name="dec_bn_2")(x)

        # Linear output — input features are continuous, not bounded to [0,1]
        outputs = layers.Dense(
            self.input_dim, activation="linear", name="decoder_output"
        )(x)

        return keras.Model(z_input, outputs, name="decoder")

    # ─────────────────────────────────────────────────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────────────────────────────────────────────────
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        # Clip log-variance to prevent exp() overflow in Sampling and KL loss
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        return reconstruction

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING STEP
    # ─────────────────────────────────────────────────────────────────────────
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        data = tf.cast(data, tf.float32)

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            # Clip log-variance to prevent exp() overflow
            z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z)

            recon_sq = tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            recon_l1 = tf.reduce_sum(tf.abs(data - reconstruction), axis=1)
            reconstruction_loss = tf.reduce_mean(0.80 * recon_sq + 0.20 * recon_l1)

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
                )
            )

            # Linear KL warmup prevents posterior collapse early in training.
            step_f = tf.cast(self.train_step_counter, tf.float32)
            warmup = tf.clip_by_value(step_f / float(self.kl_anneal_steps), 0.0, 1.0)
            beta_t = self.beta_start + warmup * (self.beta - self.beta_start)
            total_loss = reconstruction_loss + beta_t * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.beta_tracker.update_state(beta_t)
        self.train_step_counter.assign_add(1)

        return {
            "loss":                self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":             self.kl_loss_tracker.result(),
            "beta":               self.beta_tracker.result(),
        }

    # Keras uses this to reset metric trackers at the start of each epoch
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.beta_tracker,
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, X, epochs=100, batch_size=64):
        """
        Train the VAE.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, input_dim)
            Concatenated [market | social | cross] feature matrix.
            Build from Data_Pipeline/feature_matrix.py output:
                X = np.hstack([market_features, social_features, cross_features])
        epochs     : training epochs (default 100)
        batch_size : samples per gradient step (default 64)
        """
        X = np.asarray(X, dtype=np.float32)
        self.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=1e-3,
                amsgrad=True,
                clipnorm=1.0,
            )
        )
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=4,
                min_lr=1e-5,
                verbose=1,
            ),
        ]
        super().fit(X, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=callbacks, verbose=1)
        print("[✓] VAE training complete")

    # ─────────────────────────────────────────────────────────────────────────
    # ENCODE
    # Returns z_mean — more stable than sampled z for downstream clustering
    # and visualisation because it has no stochastic noise.
    # ─────────────────────────────────────────────────────────────────────────
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode X into latent space. Call after training.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, input_dim)

        Returns
        -------
        np.ndarray of shape (n_samples, latent_dim)
        """
        z_mean, _ = self.encoder(tf.cast(X, tf.float32))
        return z_mean.numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD WEIGHTS
    # ─────────────────────────────────────────────────────────────────────────
    def save_weights(self, filepath: str):
        """Save trained weights.  e.g. vae.save_weights('AI_systems/vae_artifacts/vae_weights.weights.h5')"""
        # Custom train_step uses encoder/decoder directly, so build the outer
        # model graph explicitly before saving to satisfy Keras checks.
        dummy = tf.zeros((1, self.input_dim))
        self(dummy)
        super().save_weights(filepath)
        print(f"[✓] VAE weights saved → {filepath}")

    def load_weights(self, filepath: str):
        """Load previously saved weights.  e.g. vae.load_weights('AI_systems/vae_artifacts/vae_weights.weights.h5')"""
        # Dummy forward pass builds the computation graph before weight loading
        dummy = tf.zeros((1, self.input_dim))
        self(dummy)
        super().load_weights(filepath)
        print(f"[✓] VAE weights loaded ← {filepath}")


def load_feature_matrix(data_dir: str = "vae_model/data") -> np.ndarray:
    """
    Load the combined feature matrix exported by feature_matrix.py.

    Prefers full_features.npy and falls back to hstacking the 3 split files.
    """
    base = tf.io.gfile.join(data_dir, "full_features.npy")
    if tf.io.gfile.exists(base):
        return np.load(base).astype(np.float32)

    market = np.load(tf.io.gfile.join(data_dir, "market_features.npy"))
    social = np.load(tf.io.gfile.join(data_dir, "social_features.npy"))
    cross = np.load(tf.io.gfile.join(data_dir, "cross_features.npy"))
    return np.hstack([market, social, cross]).astype(np.float32)


def build_temporal_context(X: np.ndarray, context_window: int = 1) -> np.ndarray:
    """
    Build flattened rolling windows for temporal context modeling.

    Example: context_window=3 transforms (N, D) -> (N-2, 3D)
    using [t-2, t-1, t] concatenation at each row t.
    """
    if context_window <= 1:
        return X.astype(np.float32)

    n, d = X.shape
    if n < context_window:
        return np.empty((0, d * context_window), dtype=np.float32)

    out = np.empty((n - context_window + 1, d * context_window), dtype=np.float32)
    for i in range(context_window - 1, n):
        out[i - context_window + 1] = X[i - context_window + 1: i + 1].reshape(-1)
    return out


def load_feature_matrix_with_context(
    data_dir: str = "vae_model/data",
    context_window: int = 1,
) -> np.ndarray:
    """
    Load feature matrix and optionally apply temporal context windowing.

    Prefers precomputed context artifact when available, otherwise builds on load.
    """
    if context_window > 1:
        precomputed = tf.io.gfile.join(data_dir, f"full_features_ctx_w{context_window}.npy")
        if tf.io.gfile.exists(precomputed):
            return np.load(precomputed).astype(np.float32)

    base = load_feature_matrix(data_dir=data_dir)
    return build_temporal_context(base, context_window=context_window)


def get_context_dims(
    context_window: int,
    market_dim: int = 12,
    social_dim: int = 27,
    cross_dim: int = 3,
) -> tuple[int, int, int]:
    """Return VAE dims adjusted for flattened temporal windows."""
    return (
        market_dim * context_window,
        social_dim * context_window,
        cross_dim * context_window,
    )


def train_from_feature_artifacts(
    data_dir: str,
    context_window: int = 1,
    epochs: int = 30,
    batch_size: int = 128,
    latent_dim: int = 16,
    weights_out: str = "AI_systems/vae_artifacts/vae_weights.weights.h5",
    latent_out: str = "AI_systems/vae_artifacts/latent_vectors.npy",
    summary_out: str = "AI_systems/vae_artifacts/vae_training_summary.json",
) -> dict:
    """Train VAE from exported feature artifacts and save model outputs."""
    X = load_feature_matrix_with_context(data_dir=data_dir, context_window=context_window)
    if X.size == 0:
        raise ValueError("Loaded feature matrix is empty. Check feature exports/context window.")

    if context_window > 1:
        m_dim, s_dim, c_dim = get_context_dims(context_window)
    else:
        m_dim, s_dim, c_dim = 12, 27, 3

    vae = VAE(
        market_dim=m_dim,
        social_dim=s_dim,
        cross_dim=c_dim,
        latent_dim=latent_dim,
    )

    print(f"[i] Training VAE on X shape={X.shape}, context_window={context_window}")
    vae.fit(X, epochs=epochs, batch_size=batch_size)

    latent = vae.encode(X)

    Path(weights_out).parent.mkdir(parents=True, exist_ok=True)
    Path(latent_out).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_out).parent.mkdir(parents=True, exist_ok=True)

    vae.save_weights(weights_out)
    np.save(latent_out, latent.astype(np.float32))

    summary = {
        "data_dir": data_dir,
        "context_window": int(context_window),
        "input_shape": [int(X.shape[0]), int(X.shape[1])],
        "latent_shape": [int(latent.shape[0]), int(latent.shape[1])],
        "dims": {
            "market_dim": int(m_dim),
            "social_dim": int(s_dim),
            "cross_dim": int(c_dim),
            "input_dim": int(m_dim + s_dim + c_dim),
            "latent_dim": int(latent_dim),
        },
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "artifacts": {
            "weights": weights_out,
            "latent_vectors": latent_out,
        },
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[✓] Latent vectors saved → {latent_out}")
    print(f"[✓] Training summary saved → {summary_out}")
    return summary


def _run_smoke_test():
    """Small synthetic run for quick dev checks."""
    print("=" * 60)
    print("  VAE smoke test  —  market + social + cross  (42-dim)")
    print("=" * 60)

    n = 500
    market = np.random.normal(size=(n, 12)).astype(np.float32)
    social = np.random.normal(size=(n, 27)).astype(np.float32)
    cross  = np.random.normal(size=(n,  3)).astype(np.float32)
    X = np.hstack([market, social, cross])

    vae = VAE(market_dim=12, social_dim=27, cross_dim=3, latent_dim=16)
    vae.fit(X, epochs=2, batch_size=64)
    latent = vae.encode(X)
    print(f"[✓] Input shape  : {X.shape}")
    print(f"[✓] Latent shape : {latent.shape}")
    print("[✓] Done")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Train VAE on exported feature artifacts.")
    parser.add_argument("--data-dir", default="vae_model/data", help="Directory containing feature .npy files")
    parser.add_argument("--context-window", type=int, default=1, help="Temporal context window size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimension")
    parser.add_argument("--weights-out", default="AI_systems/vae_artifacts/vae_weights.weights.h5", help="Output VAE weights path")
    parser.add_argument("--latent-out", default="AI_systems/vae_artifacts/latent_vectors.npy", help="Output latent vectors path")
    parser.add_argument("--summary-out", default="AI_systems/vae_artifacts/vae_training_summary.json", help="Output training summary path")
    parser.add_argument("--smoke", action="store_true", help="Run synthetic smoke test instead of real training")
    args = parser.parse_args()

    if args.smoke:
        _run_smoke_test()
    else:
        train_from_feature_artifacts(
            data_dir=args.data_dir,
            context_window=args.context_window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            weights_out=args.weights_out,
            latent_out=args.latent_out,
            summary_out=args.summary_out,
        )
