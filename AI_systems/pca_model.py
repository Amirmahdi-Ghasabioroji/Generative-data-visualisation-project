"""
PCA Module — Dimensionality Reduction
Purpose: Reduce dataset dimensionality while keeping most variance.
Stack: numpy only
Covers: Fit, Transform, Fit+Transform, Inverse Transform
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button
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
        self._ax_3d = None
        self._ax_2d = None
        self._ax_scree = None
        self._ax_error = None
        self._scatter = None
        self._latest_point = None
        self._cbar = None
        self._view_cbar = None
        self._view_cbar_mode = None
        self._view_cbar_ax = None
        self._axis_limits = None
        self._error_history = []
        self._view_mode = "3d"
        self._toggle_button_ax = None
        self._toggle_button = None
        self._latest_X_reduced = None
        self._latest_title = "PCA: 3D Projection"
        self._latest_point_size = 20

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
        point_size: int = 20,
    ):
        if X_reduced.ndim != 2 or X_reduced.shape[1] < 3:
            raise ValueError("X_reduced must have shape (N, 3+) for 3D plotting")

        if self._fig is None or self._ax is None:
            plt.style.use("dark_background")
            plt.ion()
            self._fig = plt.figure(figsize=(6, 4))
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
    # STAGE 8: RECONSTRUCTION ERROR UPDATE
    # What it does: Computes MSE between original X and its
    # PCA reconstruction, appends to history, and redraws the line.
    # ─────────────────────────────────────────────────────────────
    def _update_reconstruction_error(self, X_original: np.ndarray):
        X_reduced = self.transform(X_original)
        X_reconstructed = self.inverse_transform(X_reduced)
        mse = float(np.mean((X_original - X_reconstructed) ** 2))
        self._error_history.append(mse)

        ax = self._ax_error
        ax.cla()
        ax.set_facecolor("#111111")
        ax.set_title("Reconstruction Error over Time", fontsize=11, color="white", pad=8)
        ax.set_xlabel("Update Step", fontsize=10, color="white", labelpad=6)
        ax.set_ylabel("MSE", fontsize=10, color="white", labelpad=6)

        steps = np.arange(1, len(self._error_history) + 1)
        ax.plot(steps, self._error_history, color="#00e5ff", linewidth=1.5)
        ax.fill_between(steps, self._error_history, alpha=0.15, color="#00e5ff")
        ax.scatter(steps[-1], self._error_history[-1], color="white", s=40, zorder=5)
        ax.text(
            steps[-1], self._error_history[-1],
            f"  {mse:.5f}", color="white", fontsize=9, va="center"
        )

        # Force integer x-axis ticks — prevents float step labels like 0.96, 1.02
        ax.set_xticks(steps)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # ─────────────────────────────────────────────────────────────
    # STAGE 9: 3D SCATTER UPDATE
    # What it does: Draws/refreshes the 3D projection scatter.
    # ─────────────────────────────────────────────────────────────
    def _update_3d_scatter(self, X_reduced, title, point_size, labels=None, sil_score=None, entropy=None):
        x_vals, y_vals, z_vals = X_reduced[:,0], X_reduced[:,1], X_reduced[:,2]

        # Colour by cluster if labels provided, otherwise fall back to original rainbow
        if labels is not None:
            rgba_colors = cm.tab10(labels / labels.max())
            rgba_colors[:,3] = np.linspace(0.20, 0.95, X_reduced.shape[0])
        else:
            color_idx = np.linspace(0, 1, X_reduced.shape[0])
            rgba_colors = cm.rainbow(color_idx)
            rgba_colors[:,3] = np.linspace(0.20, 0.95, X_reduced.shape[0])

        x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
        z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))

        if self._axis_limits is None:
            self._axis_limits = {"x": [x_min, x_max], "y": [y_min, y_max], "z": [z_min, z_max]}
        else:
            self._axis_limits["x"][0] = min(self._axis_limits["x"][0], x_min)
            self._axis_limits["x"][1] = max(self._axis_limits["x"][1], x_max)
            self._axis_limits["y"][0] = min(self._axis_limits["y"][0], y_min)
            self._axis_limits["y"][1] = max(self._axis_limits["y"][1], y_max)
            self._axis_limits["z"][0] = min(self._axis_limits["z"][0], z_min)
            self._axis_limits["z"][1] = max(self._axis_limits["z"][1], z_max)

        def padded(lo, hi):
            if lo == hi:
                return lo - 0.5, hi + 0.5
            pad = (hi - lo) * 0.1
            return lo - pad, hi + pad

        self._ax_3d.set_xlim(*padded(*self._axis_limits["x"]))
        self._ax_3d.set_ylim(*padded(*self._axis_limits["y"]))
        self._ax_3d.set_zlim(*padded(*self._axis_limits["z"]))

        # Ensure axis labels and ticks are clearly white
        for label in [self._ax_3d.xaxis.label, self._ax_3d.yaxis.label, self._ax_3d.zaxis.label]:
            label.set_color("white")
            label.set_fontsize(10)
        self._ax_3d.tick_params(colors="white", labelsize=8)

        if self._scatter is None:
            self._scatter = self._ax_3d.scatter(
                x_vals, y_vals, z_vals,
                c=rgba_colors, edgecolor="none", s=point_size,
            )
        else:
            self._scatter._offsets3d = (x_vals, y_vals, z_vals)
            self._scatter.set_sizes(np.full(x_vals.shape[0], point_size))
            self._scatter.set_facecolors(rgba_colors)

        # Colorbar — extra padding so it never overlaps the 3D axes
        if self._view_cbar_mode != "3d":
            if self._view_cbar is not None:
                try:
                    self._view_cbar.remove()
                except Exception:
                    pass
            if self._view_cbar_ax is not None:
                self._view_cbar_ax.cla()
            self._view_cbar = self._fig.colorbar(
                cm.ScalarMappable(cmap="rainbow"),
                cax=self._view_cbar_ax,
            )
            self._view_cbar.set_label("Time (old → new)", rotation=270, labelpad=16, color="white", fontsize=9)
            self._view_cbar.ax.tick_params(colors="white", labelsize=8)
            self._view_cbar_mode = "3d"

        if self._latest_point is None:
            self._latest_point = self._ax_3d.scatter(
                [x_vals[-1]], [y_vals[-1]], [z_vals[-1]],
                c="white", edgecolor="black", s=point_size * 2.5, marker="o", zorder=10,
            )
        else:
            self._latest_point._offsets3d = ([x_vals[-1]], [y_vals[-1]], [z_vals[-1]])
            self._latest_point.set_sizes(np.array([point_size * 2.5]))
            

        self._ax_3d.set_title(title, pad=12, fontsize=12, color="white")

        if sil_score is not None and entropy is not None:
            self._ax_3d.text2D(
                0.02, 0.05,
                f"Silhouette: {sil_score:.3f}   Entropy: {entropy:.3f}",
                transform=self._ax_3d.transAxes,
                fontsize=8,
                color="white",
                alpha=0.8,
            )

    def _update_2d_hexbin(self, X_reduced: np.ndarray, title: str):
        x_vals = X_reduced[:, 0]
        y_vals = X_reduced[:, 1]

        ax = self._ax_2d
        ax.cla()
        ax.set_facecolor("#111111")
        ax.grid(alpha=0.25)
        ax.set_xlabel("PC1", fontsize=10, color="white")
        ax.set_ylabel("PC2", fontsize=10, color="white")
        ax.set_title(f"{title} — 2D Density", pad=10, fontsize=11, color="white")
        ax.tick_params(colors="white", labelsize=8)

        hb = ax.hexbin(x_vals, y_vals, gridsize=35, mincnt=1, cmap="viridis")
        ax.scatter([x_vals[-1]], [y_vals[-1]], c="white", edgecolor="black", s=45, zorder=10)

        if self._view_cbar_mode != "2d":
            if self._view_cbar is not None:
                try:
                    self._view_cbar.remove()
                except Exception:
                    pass
            if self._view_cbar_ax is not None:
                self._view_cbar_ax.cla()
            self._view_cbar = self._fig.colorbar(hb, cax=self._view_cbar_ax)
            self._view_cbar.set_label("Points per bin", color="white", fontsize=9)
            self._view_cbar.ax.tick_params(colors="white", labelsize=8)
            self._view_cbar_mode = "2d"
        else:
            self._view_cbar.update_normal(hb)

    def _update_scree(self):
        ax = self._ax_scree
        ax.cla()
        ax.set_facecolor("#111111")
        ax.set_title("Explained Variance", fontsize=11, color="white", pad=8)
        ax.set_xlabel("Principal Component", fontsize=10, color="white", labelpad=6)
        ax.set_ylabel("Variance Ratio", fontsize=10, color="white", labelpad=6)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        if self.explained_variance_ratio_ is None:
            ax.text(0.5, 0.5, "Fit PCA first", ha="center", va="center", color="white")
            ax.set_xticks([])
            return

        ratios = self.explained_variance_ratio_
        indices = np.arange(1, len(ratios) + 1)

        bars = ax.bar(indices, ratios, color="#4ea1ff", alpha=0.85, width=0.5)
        ax.plot(indices, np.cumsum(ratios), color="#ffd166", linewidth=1.5, marker="o", markersize=4)

        # Bar value labels — clear gap above each bar
        for bar, ratio in zip(bars, ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{ratio*100:.1f}%",
                ha="center", va="bottom", fontsize=8, color="white"
            )

        ax.set_xticks(indices)
        ax.set_xticklabels([f"PC{i}" for i in indices], fontsize=9, color="white")
        ax.set_ylim(0, min(1.05, ratios.max() * 1.45))

    def _init_figure(self):
        plt.style.use("dark_background")
        plt.ion()

        self._fig = plt.figure(figsize=(12, 7))
        self._fig.patch.set_facecolor("#0d0d0d")

        gs = self._fig.add_gridspec(
            2, 2,
            width_ratios=[1.5, 1.0],
            height_ratios=[1.0, 1.0],
            left=0.05, right=0.95,
            top=0.88, bottom=0.08,
            hspace=0.52,
            wspace=0.38,
        )

        self._ax_3d = self._fig.add_subplot(gs[:, 0], projection="3d")
        self._ax_3d.set_xlabel("PC1", fontsize=10)
        self._ax_3d.set_ylabel("PC2", fontsize=10)
        self._ax_3d.set_zlabel("PC3", fontsize=10)
        self._ax_3d.grid(alpha=0.25)

        self._ax_2d = self._fig.add_subplot(gs[:, 0])
        self._ax_2d.set_visible(False)

        self._ax_scree = self._fig.add_subplot(gs[0, 1])
        self._ax_error = self._fig.add_subplot(gs[1, 1])

        # Fixed colorbar axis to prevent subplot area shrinking on mode toggle
        self._view_cbar_ax = self._fig.add_axes([0.92, 0.17, 0.015, 0.62])
        self._view_cbar_ax.set_facecolor("#0d0d0d")

        # ── Toggle button — bright blue, bold white text, easy to read ──
        self._toggle_button_ax = self._fig.add_axes([0.43, 0.915, 0.14, 0.058])
        self._toggle_button = Button(
            self._toggle_button_ax,
            "Switch to 2D",
            color="#2979ff",
            hovercolor="#5c9eff",
        )
        self._toggle_button.label.set_color("white")
        self._toggle_button.label.set_fontsize(11)
        self._toggle_button.label.set_fontweight("bold")
        self._toggle_button.on_clicked(self._toggle_projection)

    def _apply_view_mode(self):
        if self._view_mode == "3d":
            self._ax_3d.set_visible(True)
            self._ax_2d.set_visible(False)
            if self._toggle_button is not None:
                self._toggle_button.label.set_text("Switch to 2D")
        else:
            self._ax_3d.set_visible(False)
            self._ax_2d.set_visible(True)
            if self._toggle_button is not None:
                self._toggle_button.label.set_text("Switch to 3D")

    def _toggle_projection(self, _event):
        self._view_mode = "2d" if self._view_mode == "3d" else "3d"
        self._apply_view_mode()
        if self._latest_X_reduced is not None:
            if self._view_mode == "3d":
                self._update_3d_scatter(self._latest_X_reduced, self._latest_title, self._latest_point_size)
            else:
                self._update_2d_hexbin(self._latest_X_reduced, self._latest_title)
        if self._fig is not None:
            self._fig.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────
    # STAGE 10: MAIN UNIFIED PLOT METHOD
    # What it does: Initialises the shared figure (once) and
    # refreshes all three panels in one call.
    # ─────────────────────────────────────────────────────────────
# plot_unified — add labels/metrics params
    def plot_unified(
        self,
        X_original: np.ndarray,
        X_reduced: np.ndarray,
        title: str = "PCA: 3D Projection",
        point_size: int = 20,
        labels=None,
        sil_score=None,
        entropy=None,
    ):
        if X_reduced.ndim != 2 or X_reduced.shape[1] < 3:
            raise ValueError("X_reduced must have shape (N, 3+) for 3D plotting.")
        if self._fig is None or self._ax_3d is None:
            self._init_figure()
        self._latest_X_reduced = X_reduced
        self._latest_title = title
        self._latest_point_size = point_size
        if self._view_mode == "3d":
            self._update_3d_scatter(X_reduced, title, point_size,
                                    labels=labels, sil_score=sil_score, entropy=entropy)
        else:
            self._update_2d_hexbin(X_reduced, title)
        self._update_scree()
        self._update_reconstruction_error(X_original)
        self._apply_view_mode()
        self._fig.canvas.draw_idle()
        plt.pause(0.001)
        plt.show(block=True)
        return self._fig

    # ─────────────────────────────────────────────────────────────
    # STAGE 11: CONVENIENCE: FIT → TRANSFORM → PLOT IN ONE CALL
    # ─────────────────────────────────────────────────────────────
   # fit_transform_plot — add labels/metrics params
    def fit_transform_plot(
        self,
        X: np.ndarray,
        title: str = "PCA: 3D Projection",
        point_size: int = 20,
        labels=None,
        sil_score=None,
        entropy=None,
    ):
        X_reduced = self.fit_transform(X)
        self.plot_unified(X, X_reduced, title=title, point_size=point_size,
                        labels=labels, sil_score=sil_score, entropy=entropy)
        return X_reduced


# ────────────────────────────────────────────────────────────
# STAGE 7: TEST BLOCK
# What it does: Simple example to check that PCA works.
# Why it matters: Quick sanity check before integrating into larger project.
# ────────────────────────────────────────────────────────────
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


