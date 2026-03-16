"""
Variational Autoencoder (VAE)
Purpose: Compress high-dimensional feature matrices into a low-dimensional
latent space for clustering, visualisation, and latent space traversal.
Stack: TensorFlow + Keras + numpy
Mode: Batch (train once on static dataset, encode for downstream use)

Dependencies:
    pip install tensorflow numpy
"""
import numpy as np
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers


# ═══════════════════════════════════════════════════════════════════════════════
# It takes [z_mean, z_log_var] as input and returns a sampled latent vector z.
# Formula: z = z_mean + eps * exp(0.5 * z_log_var)  where eps ~ N(0, I)
# This needs to be a keras Layer subclass with a call() method.
class Sampling(layers.Layer):
    """
    Sampling layer using the reparameterisation trick.
    z = z_mean + eps * exp(0.5 * z_log_var)
    where eps ~ N(0, I)
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# VAE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class VAE(keras.Model):
    """
    Variational Autoencoder for compressing high-dimensional feature matrices
    into a low-dimensional latent space (latent_dim=32).
    """

    def __init__(self, input_dim: int, latent_dim: int = 32, **kwargs):
        """
        Initialises the VAE.

        Parameters
        ----------
        input_dim  : size of the input feature vector (e.g. 399)
        latent_dim : size of the compressed latent space (default 32)
        beta       : weight on KL loss — set to input_dim/latent_dim so
                     reconstruction and KL contribute equally (prevents
                     posterior collapse)
        """
        super().__init__(**kwargs)
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        # Beta balances reconstruction vs KL — default keeps both terms roughly
        # equal in magnitude: recon sums over input_dim, KL over latent_dim.
        self.beta = input_dim / latent_dim

        # Build encoder and decoder models
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.sampling = Sampling()

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    # ═══════════════════════════════════════════════════════════════════════════
    # ENCODER
    # Compresses input (399 dims) down to two 32-dim vectors:
    #   z_mean    — the mean of the latent distribution
    #   z_log_var — the log variance of the latent distribution
    # Architecture: 399 → Dense(256, relu) → Dense(128, relu) → z_mean, z_log_var
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_encoder(self) -> keras.Model:
        inputs    = keras.Input(shape=(self.input_dim,), name="encoder_input")

        # First hidden layer — compresses 399 → 256
        x         = layers.Dense(256, activation="relu")(inputs)

        # Second hidden layer — compresses 256 → 128
        x         = layers.Dense(128, activation="relu")(x)

        # Two output heads — both output 32-dim vectors
        z_mean    = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        return keras.Model(inputs, [z_mean, z_log_var], name="encoder")

    # ═══════════════════════════════════════════════════════════════════════════
    # DECODER
    # Mirror image of the encoder — takes a 32-dim latent vector
    # and expands it back up to 399 dims (the original input size)
    # Architecture: 32 → Dense(128, relu) → Dense(256, relu) → 399
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_decoder(self) -> keras.Model:
        inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")

        # First hidden layer — expands 32 → 128
        x      = layers.Dense(128, activation="relu")(inputs)

        # Second hidden layer — expands 128 → 256
        x      = layers.Dense(256, activation="relu")(x)

        # Output layer — expands back to original input size (399)
        # Linear activation because input features are continuous (not 0-1)
        outputs = layers.Dense(self.input_dim, activation="linear", name="decoder_output")(x)

        return keras.Model(inputs, outputs, name="decoder")

    # ═══════════════════════════════════════════════════════════════════════════
    # Replace this placeholder with the full forward pass:
    #   1. Pass inputs through self.encoder → get z_mean, z_log_var
    #   2. Pass [z_mean, z_log_var] through Sampling layer → get z
    #   3. Pass z through self.decoder → get reconstruction
    #   4. Return reconstruction
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        # Clip log-variance to prevent exp() overflow in Sampling and KL loss
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        return reconstruction
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Add a custom train_step(self, data) method here.
    # Inside it should:
    #   1. Use tf.GradientTape to track gradients
    #   2. Run encoder → reparameterise → decoder
    #   3. Compute reconstruction loss (MSE between input and reconstruction)
    #   4. Compute KL divergence loss
    #   5. Sum them into total_loss
    #   6. Apply gradients via self.optimizer
    #   7. Update the three loss trackers
    #   8. Return a dict of the three loss values
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            # Clip log-variance to prevent exp() overflow
            z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
            z = self.sampling([z_mean, z_log_var])
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction), axis=1
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
                )
            )

            # Beta weighting keeps reconstruction and KL losses in the same
            # magnitude range — prevents the KL term from being ignored
            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    # ═══════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # Add a @property called metrics that returns a list of the three trackers.
    # Keras uses this to reset them at the start of each epoch.
     
    @property
    def metrics(self):
        return [
          self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
                ]

    # ═══════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    
    # Add a fit(self, X, epochs=50, batch_size=32) method here.
    # It should:
    #   1. Cast X to tf.float32
    #   2. Compile the model with Adam optimiser
    #   3. Call super().fit() to run the training loop
    #   4. Print a confirmation when done

    def fit(self, X, epochs=50, batch_size=32):
        X = tf.cast(X, tf.float32)

        self.compile(optimizer=keras.optimizers.Adam())

        super().fit(
            X,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
        )

        print("[✓] VAE training complete")
    



    # ═══════════════════════════════════════════════════════════════════════════
    # ENCODE
    # Exposes the latent mean vectors for use in Static_Bluesky.py.
    # Returns z_mean (not sampled z) — more stable for clustering/visualisation.
    # Shape: (n_samples, latent_dim) e.g. (300, 32)
    # ═══════════════════════════════════════════════════════════════════════════
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encodes input X into latent space.
        Call this after training to get vectors for clustering + visualisation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, input_dim)

        Returns
        -------
        np.ndarray of shape (n_samples, latent_dim)
        """
        X_tensor   = tf.cast(X, tf.float32)
        z_mean, _  = self.encoder(X_tensor)
        return z_mean.numpy()

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE WEIGHTS
    # Saves the trained model weights to disk so training doesn't need
    # to restart from scratch every run.
    # ═══════════════════════════════════════════════════════════════════════════
    def save_weights(self, filepath: str):
        """
        Save trained weights to filepath.
        Example: vae.save_weights('AI_systems/vae_weights')
        """
        super().save_weights(filepath)
        print(f"[✓] VAE weights saved → {filepath}")

    # ═══════════════════════════════════════════════════════════════════════════
    # LOAD WEIGHTS
    # Loads previously saved weights back into the model.
    # Runs a dummy forward pass first because Keras needs to build the
    # computation graph before it can load weights into it.
    # ═══════════════════════════════════════════════════════════════════════════
    def load_weights(self, filepath: str):
        """
        Load previously saved weights from filepath.
        Example: vae.load_weights('AI_systems/vae_weights')
        """
        # Dummy pass to build the graph before loading weights
        dummy = tf.zeros((1, self.input_dim))
        self(dummy)
        super().load_weights(filepath)
        print(f"[✓] VAE weights loaded ← {filepath}")



