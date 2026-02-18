"""
PCA Module — Dimensionality Reduction
Purpose: Reduce dataset dimensionality while keeping most variance.
Stack: numpy only
Covers: Fit, Transform, Fit+Transform, Inverse Transform
"""

import numpy as np
import matplotlib.pyplot as plt
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
# STAGE 6: TEST BLOCK
# What it does: Simple example to check that PCA works.
# Why it matters: Quick sanity check before integrating into larger project.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    

    # Generate random dataset (100 samples, 10 features)
    X = np.random.rand(100, 10)

    # Create PCA instance to reduce to 3 components
    pca = PCA(n_components=3)

    # Fit PCA and reduce data
    X_reduced = pca.fit_transform(X)

    # 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X_reduced[:, 0], 
        X_reduced[:, 1], 
        X_reduced[:, 2], 
        c='skyblue', edgecolor='k', s=50
    )
    ax.set_title("PCA: 3D Projection of Random Data")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()

    # Print explained variance ratio for reference
    print("Explained variance ratio:", pca.explained_variance_ratio_)

