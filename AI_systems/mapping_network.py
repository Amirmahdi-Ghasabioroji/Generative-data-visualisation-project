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
    theta_targets = np.clip(np.random.uniform(size=(n, 5)).astype(np.float32), 0.0, 1.0)

    model = MappingNetwork(latent_dim=16, theta_dim=5)
    model.fit(z, theta_targets, epochs=2, batch_size=64)
    theta_pred = model.map_latent(z)

    print(f"[✓] z shape      : {z.shape}")
    print(f"[✓] theta shape  : {theta_pred.shape}")
    print("[✓] Mapping network smoke test done")
