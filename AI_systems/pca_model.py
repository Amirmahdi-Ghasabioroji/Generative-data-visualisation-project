"""
PCA Module — Dimensionality Reduction
Purpose: Reduce dataset dimensionality while keeping most variance.
Stack: numpy only
Covers: Fit, Transform, Fit+Transform, Inverse Transform
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# ─────────────────────────────────────────────────────────────
# STAGE 1: PCA CLASS DEFINITION
# What it does: Encapsulates all PCA logic in one class.
# Why it matters: Keeps code clean and reusable — you can
# create multiple PCA instances with different n_components.
# ─────────────────────────────────────────────────────────────

class PCA:
    def __init__(self, n_components: int):
        # Number of principal components to keep
        self.n_components = n_components
        
        # Placeholder for mean vector, principal components, and variance info
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self._fig = None
        self._ax = None
        self._scatter = None
        self._latest_point = None
        self._cbar = None
        self._axis_limits = None

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: FIT METHOD
    # What it does: Learns the principal components from the data.
    # Why it matters: Calculates directions of maximum variance
    # which we will use to reduce dimensionality.
    # ─────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray):
        # Step 1: Compute mean of each feature
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_  # Center data around zero

        # Step 2: Compute covariance matrix of centered data
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Eigen decomposition of covariance matrix
        # Eigenvectors = directions of maximum variance
        # Eigenvalues = amount of variance along each direction
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort eigenvectors by decreasing eigenvalue
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Step 5: Keep only the top k components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]

        # Step 6: Compute explained variance ratio
        # Shows how much of the total variance is captured
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: TRANSFORM METHOD
    # What it does: Projects new data onto learned principal components.
    # Why it matters: Reduces dimensionality while preserving variance.
    # ─────────────────────────────────────────────────────────────
    def transform(self, X: np.ndarray):
        # Center the data first using learned mean
        X_centered = X - self.mean_

        # Project onto principal components
        return np.dot(X_centered, self.components_)

    # ─────────────────────────────────────────────────────────────
    # STAGE 4: FIT_TRANSFORM METHOD
    # What it does: Convenience method to fit and immediately transform data.
    # Why it matters: Simplifies workflow in one step.
    # ─────────────────────────────────────────────────────────────
    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

    # ─────────────────────────────────────────────────────────────
    # STAGE 5: INVERSE_TRANSFORM METHOD
    # What it does: Maps reduced data back to original feature space.
    # Why it matters: Useful to approximate original data from PCA representation.
    # ─────────────────────────────────────────────────────────────
    def inverse_transform(self, X_reduced: np.ndarray):
        return np.dot(X_reduced, self.components_.T) + self.mean_

    # ─────────────────────────────────────────────────────────────
    # STAGE 6: 3D PLOTTING FOR PIPELINE USE
    # What it does: Renders/updates a non-blocking 3D scatter plot
    # from PCA-reduced data.
    # Why it matters: Lets real-time pipelines call this method each
    # update without freezing the stream loop.
    # ─────────────────────────────────────────────────────────────
    def plot_3d_scatter(
        self,
        X_reduced: np.ndarray,
        title: str = "PCA: 3D Projection",
        color: str = "skyblue",
        point_size: int = 20,
    ):
        if X_reduced.ndim != 2 or X_reduced.shape[1] < 3:
            raise ValueError("X_reduced must have shape (N, 3+) for 3D plotting")

        if self._fig is None or self._ax is None:
            plt.style.use("dark_background")
            plt.ion()
            self._fig = plt.figure(figsize=(8, 6))
            self._ax = self._fig.add_subplot(111, projection='3d')
            self._ax.set_xlabel("PC1")
            self._ax.set_ylabel("PC2")
            self._ax.set_zlabel("PC3")
            self._ax.grid(alpha=0.25)

        x_vals = X_reduced[:, 0]
        y_vals = X_reduced[:, 1]
        z_vals = X_reduced[:, 2]
        color_idx = np.linspace(0, 1, X_reduced.shape[0])
        rgba_colors = cm.rainbow(color_idx)
        rgba_colors[:, 3] = np.linspace(0.20, 0.95, X_reduced.shape[0])

        x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
        z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))

        if self._axis_limits is None:
            self._axis_limits = {
                "x": [x_min, x_max],
                "y": [y_min, y_max],
                "z": [z_min, z_max],
            }
        else:
            self._axis_limits["x"][0] = min(self._axis_limits["x"][0], x_min)
            self._axis_limits["x"][1] = max(self._axis_limits["x"][1], x_max)
            self._axis_limits["y"][0] = min(self._axis_limits["y"][0], y_min)
            self._axis_limits["y"][1] = max(self._axis_limits["y"][1], y_max)
            self._axis_limits["z"][0] = min(self._axis_limits["z"][0], z_min)
            self._axis_limits["z"][1] = max(self._axis_limits["z"][1], z_max)

        def padded_limits(low: float, high: float) -> tuple[float, float]:
            if low == high:
                return low - 0.5, high + 0.5
            pad = (high - low) * 0.1
            return low - pad, high + pad

        x_low, x_high = padded_limits(*self._axis_limits["x"])
        y_low, y_high = padded_limits(*self._axis_limits["y"])
        z_low, z_high = padded_limits(*self._axis_limits["z"])
        self._ax.set_xlim(x_low, x_high)
        self._ax.set_ylim(y_low, y_high)
        self._ax.set_zlim(z_low, z_high)

        if self._scatter is None:
            self._scatter = self._ax.scatter(
                x_vals,
                y_vals,
                z_vals,
                c=rgba_colors,
                edgecolor='none',
                s=point_size,
            )
            self._cbar = self._fig.colorbar(
                cm.ScalarMappable(cmap="rainbow"),
                ax=self._ax,
                pad=0.1,
                fraction=0.03,
            )
            self._cbar.set_label("Time (old → new)", rotation=270, labelpad=14)
        else:
            self._scatter._offsets3d = (x_vals, y_vals, z_vals)
            self._scatter.set_sizes(np.full(x_vals.shape[0], point_size))
            self._scatter.set_facecolors(rgba_colors)

        if self._latest_point is None:
            self._latest_point = self._ax.scatter(
                [x_vals[-1]],
                [y_vals[-1]],
                [z_vals[-1]],
                c='white',
                edgecolor='black',
                s=point_size * 2,
                marker='o',
            )
        else:
            self._latest_point._offsets3d = ([x_vals[-1]], [y_vals[-1]], [z_vals[-1]])
            self._latest_point.set_sizes(np.array([point_size * 2]))

        self._ax.set_title(title)

        self._fig.canvas.draw_idle()
        plt.pause(0.001)
        return self._fig, self._ax

    def fit_transform_plot_3d(self, X: np.ndarray, title: str = "PCA: 3D Projection"):
        X_reduced = self.fit_transform(X)
        self.plot_3d_scatter(X_reduced, title=title)
        return X_reduced


# ─────────────────────────────────────────────────────────────
# STAGE 7: TEST BLOCK
# What it does: Simple example to check that PCA works.
# Why it matters: Quick sanity check before integrating into larger project.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    

    # Generate random dataset (100 samples, 10 features)
    X = np.random.rand(100, 10)

    # Create PCA instance to reduce to 3 components
    pca = PCA(n_components=3)

    # Fit, reduce and plot in one call
    X_reduced = pca.fit_transform_plot_3d(X, title="PCA: 3D Projection of Random Data")
    plt.show()

    # Print explained variance ratio for reference
    print("Explained variance ratio:", pca.explained_variance_ratio_)

