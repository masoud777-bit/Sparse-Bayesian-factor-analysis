"""Synthetic data with known ground truth (loadings, scores, noise)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter

from .model import grid_edges


def _sparse_loadings(rng: np.random.Generator, p: int, k: int, active_per_factor: int) -> np.ndarray:
    """p x k loading matrix; each factor loads on `active_per_factor` random features."""
    W = np.zeros((p, k))
    for f in range(k):
        idx = rng.choice(p, size=active_per_factor, replace=False)
        W[idx, f] = rng.choice([-1.0, 1.0], size=active_per_factor) * rng.uniform(0.7, 1.5, active_per_factor)
    return W


def make_sparse_factor_data(n: int = 500, p: int = 60, k: int = 4, active_per_factor: int = 15,
                            noise_range=(0.3, 1.0), seed: int = 0,
                            noise: str = "gaussian") -> Dict[str, np.ndarray]:
    """iid factor-analysis data:  X = Z W' + noise,  Z ~ N(0, I).

    ``noise="gaussian"``: noise_j ~ N(0, psi_j) (the model is correctly specified).
    ``noise="t3"``      : Student-t noise with 3 d.o.f., rescaled to variance psi_j
                          (heavy tails: the Gaussian model is *mis*-specified).
    """
    rng = np.random.default_rng(seed)
    W = _sparse_loadings(rng, p, k, active_per_factor)
    Z = rng.standard_normal((n, k))
    psi = rng.uniform(*noise_range, size=p)
    if noise == "gaussian":
        eps = rng.standard_normal((n, p))
    elif noise == "t3":
        eps = rng.standard_t(3, size=(n, p)) / np.sqrt(3.0)   # Var(t_3) = 3
    else:
        raise ValueError(f"unknown noise type: {noise}")
    X = Z @ W.T + eps * np.sqrt(psi)
    return dict(X=X, W=W, Z=Z, psi=psi)


def make_spatial_factor_data(n_rows: int = 30, n_cols: int = 30, k: int = 3, p: int = 100,
                             active_per_factor: int = 25, noise_range=(1.5, 3.0),
                             length_scale: float = 3.0, seed: int = 0) -> Dict[str, np.ndarray]:
    """Spatially smooth factor maps on a grid (synthetic 2D spatial factor maps).

    Each factor is a Gaussian-filtered white-noise field (unit variance); the observed
    'expression' of each feature is a sparse linear combination plus heteroscedastic noise.
    """
    rng = np.random.default_rng(seed)
    n = n_rows * n_cols
    Z = np.zeros((n, k))
    for f in range(k):
        field = gaussian_filter(rng.standard_normal((n_rows, n_cols)), sigma=length_scale, mode="reflect")
        field = (field - field.mean()) / field.std()
        Z[:, f] = field.ravel()
    W = _sparse_loadings(rng, p, k, active_per_factor)
    psi = rng.uniform(*noise_range, size=p)
    X = Z @ W.T + rng.standard_normal((n, p)) * np.sqrt(psi)
    return dict(X=X, W=W, Z=Z, psi=psi, edges=grid_edges(n_rows, n_cols))


def random_holdout(shape, frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """Boolean array marking a random fraction of entries as held out (MCAR)."""
    rng = np.random.default_rng(seed)
    return rng.random(shape) < frac
