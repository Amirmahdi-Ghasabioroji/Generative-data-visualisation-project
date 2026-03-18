"""
MLP Mapping Network
Purpose: Maps VAE latent vectors z (32 dims) to visual parameter vectors θ (5 dims).
         Each value in θ controls one aspect of the visualisation:
         θ[0] — particle turbulence
         θ[1] — colour palette
         θ[2] — noise distortion
         θ[3] — geometry fragmentation
         θ[4] — speed of motion
Stack: TensorFlow + Keras + numpy
Mode: Trained once on latent vectors from VAE, then used to drive visuals

Dependencies:
    pip install tensorflow numpy
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MappingNetwork(keras.Model):
    """
    Small MLP that maps VAE latent vectors z to visual parameter vectors θ.

    Input  : z of shape (n_samples, latent_dim)  e.g. (300, 32)
    Output : θ of shape (n_samples, theta_dim)   e.g. (300, 5)
    """

    def __init__(self, latent_dim: int = 32, theta_dim: int = 5, **kwargs):
        """
        Initialises the MappingNetwork.

        Parameters
        ----------
        latent_dim : size of the VAE latent vector z (default 32)
        theta_dim  : number of visual parameters to output (default 5)
        """
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.theta_dim  = theta_dim

        # ── Hidden layers ─────────────────────────────────────────────────────
        # Two Dense layers that learn the mapping from z to θ
        self.dense1 = layers.Dense(64, activation="relu", name="hidden_1")
        self.dense2 = layers.Dense(32, activation="relu", name="hidden_2")

        # ── Output layer ──────────────────────────────────────────────────────
        # Sigmoid activation keeps all θ values between 0 and 1
        # so every visual parameter is on the same scale
        self.output_layer = layers.Dense(theta_dim, activation="sigmoid", name="output")

        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        # ─────────────────────────────────────────────────────────────────────

    def call(self, inputs):
        """
        Forward pass — takes z in, outputs θ.

        Parameters
        ----------
        inputs : tf.Tensor of shape (n_samples, latent_dim)

        Returns
        -------
        tf.Tensor of shape (n_samples, theta_dim)
        """
        # Pass through hidden layer 1 (32 → 64)
        x = self.dense1(inputs)

        # Pass through hidden layer 2 (64 → 32)
        x = self.dense2(x)

        # Pass through output layer (32 → 5)
        return self.output_layer(x)    
    
    def save_weights(self, filepath: str):
        """
        Save trained weights to filepath.
        Example: mapping_net.save_weights('AI_systems/mapping_network_weights')
        """
        super().save_weights(filepath)
        print(f"[✓] Mapping network weights saved → {filepath}")

    def load_weights(self, filepath: str):
        """
        Load previously saved weights from filepath.
        Example: mapping_net.load_weights('AI_systems/mapping_network_weights')
        """
        # Dummy pass to build the graph before loading weights
        dummy = tf.zeros((1, self.latent_dim))
        self(dummy)
        super().load_weights(filepath)
        print(f"[✓] Mapping network weights loaded ← {filepath}")


             
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
         
         
         def fit(self, Z, theta_targets, epochs=50, batch_size=32):
             Z = tf.convert_to_tensor(Z, dtype=tf.float32)
             theta_targets = tf.convert_to_tensor(theta_targets, dtype=tf.float32)
         
             # Compile model with Adam optimizer (loss is computed in train_step)
             super().compile(optimizer=tf.keras.optimizers.Adam())
         
             # Use Keras fit to handle batching and epochs
             super().fit(
                 x=Z,
                 y=theta_targets,
                 epochs=epochs,
                 batch_size=batch_size
             )
             print(f"[✓] Mapping network trained for {epochs} epochs")
             # ═══════════════════════════════════════════════════════════════════════════
