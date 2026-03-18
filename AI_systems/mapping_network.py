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

        # ── HRIDITA: LOSS TRACKERS ────────────────────────────────────────────
        # Add a keras.metrics.Mean tracker here called self.loss_tracker
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

        # ═══════════════════════════════════════════════════════════════════════════
    # ── HRIDITA: METRICS PROPERTY ────────────────────────────────────────────
    # Add a @property called metrics that returns [self.loss_tracker]
    # Keras uses this to reset the tracker at the start of each epoch.
    # ═══════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # ── HRIDITA: TRAINING STEP ───────────────────────────────────────────────
    # Add a custom train_step(self, data) method here.
    # Inside it should:
    #   1. Unpack data into (z, theta_target)
    #   2. Use tf.GradientTape to track gradients
    #   3. Run forward pass → self(z) → theta_pred
    #   4. Compute loss between theta_pred and theta_target (MSE)
    #   5. Apply gradients via self.optimizer
    #   6. Update self.loss_tracker
    #   7. Return {"loss": self.loss_tracker.result()}
    # ═══════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # ── HRIDITA: FIT / TRAINING LOOP ─────────────────────────────────────────
    # Add a fit(self, Z, theta_targets, epochs=50, batch_size=32) method here.
    # It should:
    #   1. Cast Z and theta_targets to tf.float32
    #   2. Compile the model with Adam optimiser and MSE loss
    #   3. Call super().fit() to run the training loop
    #   4. Print a confirmation when done
    # ═══════════════════════════════════════════════════════════════════════════