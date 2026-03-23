"""
MLP Mapping Network
Purpose: Map VAE latent vectors z (default 16 dims) to visual parameter
         vectors theta (default 5 dims) for the generative visual engine.
Stack: TensorFlow + Keras + numpy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _to_unit(arr: np.ndarray, low_q: float = 5.0, high_q: float = 95.0) -> np.ndarray:
    """Robustly map a 1D array to [0, 1] using quantiles."""
    arr = np.asarray(arr, dtype=np.float32)
    lo = np.percentile(arr, low_q)
    hi = np.percentile(arr, high_q)
    if hi - lo <= 1e-8:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def build_meaningful_theta_targets(
    market_features: np.ndarray,
    social_features: np.ndarray,
    cross_features: np.ndarray,
) -> np.ndarray:
    """
    Build interpretable theta labels from engineered features.

    Theta semantics:
    - theta[0] turbulence    : volatility + sentiment/price disagreement
    - theta[1] color         : fear<->greed sentiment axis
    - theta[2] distortion    : uncertainty / regime ambiguity
    - theta[3] fragmentation : topic/cluster dispersion + sparse social signal
    - theta[4] speed         : return momentum + volume intensity
    """
    m = np.asarray(market_features, dtype=np.float32)
    s = np.asarray(social_features, dtype=np.float32)
    c = np.asarray(cross_features, dtype=np.float32)

    if m.ndim != 2 or s.ndim != 2 or c.ndim != 2:
        raise ValueError("Expected 2D arrays for market/social/cross features.")
    if not (len(m) == len(s) == len(c)):
        raise ValueError("Feature arrays must have matching sample counts.")
    if m.shape[1] < 12 or s.shape[1] < 27 or c.shape[1] < 3:
        raise ValueError("Unexpected feature dimensions. Expected market>=12, social>=27, cross>=3.")

    # Market indices (from Data_Pipeline.feature_matrix MARKET_COLS)
    log_return = m[:, 0]
    rv_12 = m[:, 3]
    volume_z_48 = m[:, 5]

    # Social indices (from SOCIAL_COLS)
    fear_greed_mean = s[:, 12]
    fear_greed_std = s[:, 13]
    sentiment_neutral_share = s[:, 15]
    sentiment_label_net = s[:, 17]
    cluster_unique_ratio = s[:, 18]
    cluster_entropy = s[:, 19]
    topic_price_action = s[:, 20]
    topic_volatility = s[:, 21]
    social_empty_flag = s[:, 26]

    # Cross indices
    sentiment_net_x_return = c[:, 0]

    abs_return_u = _to_unit(np.abs(log_return))
    rv12_u = _to_unit(rv_12)
    volume_u = _to_unit(volume_z_48)
    price_action_u = _to_unit(topic_price_action)
    topic_vol_u = _to_unit(topic_volatility)
    disagreement_u = _to_unit(np.abs(sentiment_net_x_return))
    uncertainty_u = _to_unit(0.6 * fear_greed_std + 0.4 * cluster_entropy)
    social_sparse_u = np.clip(social_empty_flag, 0.0, 1.0)

    # Signed sentiment axis: fear(0) -> neutral(0.5) -> greed(1)
    signed_sent = 0.55 * _to_unit(sentiment_label_net) + 0.45 * _to_unit(fear_greed_mean)

    theta_turbulence = 0.55 * rv12_u + 0.30 * disagreement_u + 0.15 * topic_vol_u
    theta_color = signed_sent
    theta_distortion = 0.50 * uncertainty_u + 0.25 * disagreement_u + 0.25 * topic_vol_u
    theta_fragmentation = (
        0.45 * _to_unit(cluster_unique_ratio)
        + 0.30 * _to_unit(cluster_entropy)
        + 0.15 * _to_unit(1.0 - sentiment_neutral_share)
        + 0.10 * social_sparse_u
    )
    theta_speed = 0.45 * abs_return_u + 0.35 * volume_u + 0.20 * price_action_u

    theta = np.vstack([
        np.clip(theta_turbulence, 0.0, 1.0),
        np.clip(theta_color, 0.0, 1.0),
        np.clip(theta_distortion, 0.0, 1.0),
        np.clip(theta_fragmentation, 0.0, 1.0),
        np.clip(theta_speed, 0.0, 1.0),
    ]).T.astype(np.float32)

    return theta


class MappingNetwork(keras.Model):
    """Small MLP that maps latent vectors z -> theta visual controls."""

    def __init__(self, latent_dim: int = 16, theta_dim: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.theta_dim = theta_dim

        self.dense1 = layers.Dense(64, activation="relu", name="hidden_1")
        self.dense2 = layers.Dense(32, activation="relu", name="hidden_2")
        self.output_layer = layers.Dense(theta_dim, activation="sigmoid", name="output")

        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        z, theta_target = data
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        theta_target = tf.convert_to_tensor(theta_target, dtype=tf.float32)

        with tf.GradientTape() as tape:
            theta_pred = self(z, training=True)
            loss = tf.reduce_mean(tf.keras.losses.mse(theta_target, theta_pred))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def fit(self, Z, theta_targets, epochs=50, batch_size=32, learning_rate=1e-3):
        Z = tf.convert_to_tensor(Z, dtype=tf.float32)
        theta_targets = tf.convert_to_tensor(theta_targets, dtype=tf.float32)
        super().compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        super().fit(
            x=Z,
            y=theta_targets,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
        )
        print(f"[✓] Mapping network trained for {epochs} epochs")

    def fit_from_features(
        self,
        Z: np.ndarray,
        market_features: np.ndarray,
        social_features: np.ndarray,
        cross_features: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> np.ndarray:
        """
        Build meaningful theta targets from features and train in one call.

        Returns the generated theta targets for inspection/reuse.
        """
        theta_targets = build_meaningful_theta_targets(
            market_features=market_features,
            social_features=social_features,
            cross_features=cross_features,
        )
        self.fit(
            Z=Z,
            theta_targets=theta_targets,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        return theta_targets

    def map_latent(self, Z: np.ndarray) -> np.ndarray:
        """Predict visual controls theta from latent vectors."""
        z = tf.convert_to_tensor(Z, dtype=tf.float32)
        return self(z, training=False).numpy()

    def save_weights(self, filepath: str):
        super().save_weights(filepath)
        print(f"[✓] Mapping network weights saved → {filepath}")

    def load_weights(self, filepath: str):
        dummy = tf.zeros((1, self.latent_dim))
        self(dummy)
        super().load_weights(filepath)
        print(f"[✓] Mapping network weights loaded ← {filepath}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    n = 400
    z = np.random.normal(size=(n, 16)).astype(np.float32)
    market = np.random.normal(size=(n, 12)).astype(np.float32)
    social = np.random.normal(size=(n, 27)).astype(np.float32)
    cross = np.random.normal(size=(n, 3)).astype(np.float32)

    theta_targets = build_meaningful_theta_targets(
        market_features=market,
        social_features=social,
        cross_features=cross,
    )

    model = MappingNetwork(latent_dim=16, theta_dim=5)
    model.fit(z, theta_targets, epochs=2, batch_size=64)
    theta_pred = model.map_latent(z)

    print(f"[✓] z shape      : {z.shape}")
    print(f"[✓] theta shape  : {theta_pred.shape}")
    print("[✓] Mapping network smoke test done")
