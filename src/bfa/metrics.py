"""Evaluation utilities.

* Rotation-invariant recovery metrics (factor models are only identified up to rotation).
* Predictive-distribution diagnostics: PIT, central-interval coverage, log predictive
  density, closed-form Gaussian CRPS (a proper scoring rule, Gneiting & Raftery 2007).
* A *correct* PIT recalibrator (Kuleshov et al. 2018 style): the recalibrated predictive
  CDF is  F_cal(y) = R(F(y)),  R = empirical CDF of the PIT values on a calibration set.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy.stats import norm

# ------------------------------------------------------------------ recovery


def principal_angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the column spaces of A and B (rotation-invariant)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def cov_rel_error(W_hat: np.ndarray, W_true: np.ndarray) -> float:
    """Relative Frobenius error of the low-rank covariance W W' (invariant to rotations)."""
    C_hat, C_true = W_hat @ W_hat.T, W_true @ W_true.T
    return float(np.linalg.norm(C_hat - C_true) / np.linalg.norm(C_true))


def procrustes_align(W_hat: np.ndarray, W_ref: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes: rotate W_hat towards W_ref (uses the truth -> evaluation only)."""
    U, _, Vt = np.linalg.svd(W_hat.T @ W_ref, full_matrices=False)
    return W_hat @ (U @ Vt)


def varimax(W: np.ndarray, gamma: float = 1.0, n_iter: int = 200, tol: float = 1e-8
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Varimax rotation (no ground truth needed): returns rotated loadings and the rotation."""
    p, k = W.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(n_iter):
        L = W @ R
        u, s, vt = np.linalg.svd(W.T @ (L ** 3 - (gamma / p) * L @ np.diag((L ** 2).sum(0))))
        R = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return W @ R, R


# ----------------------------------------------------- predictive distributions


def gaussian_pit(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return norm.cdf((y - mean) / np.sqrt(var))


def mean_log_pred_density(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    return float(np.mean(norm.logpdf(y, loc=mean, scale=np.sqrt(var))))


def gaussian_crps(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    """Closed-form CRPS of N(mean, var) at y, averaged over entries."""
    sd = np.sqrt(var)
    z = (y - mean) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def coverage(y: np.ndarray, mean: np.ndarray, var: np.ndarray, level: float) -> float:
    """Empirical coverage of the central Gaussian interval with nominal probability `level`."""
    half = norm.ppf(0.5 + level / 2.0) * np.sqrt(var)
    return float(np.mean(np.abs(y - mean) <= half))


def coverage_curve(y, mean, var, levels: Iterable[float]) -> np.ndarray:
    return np.array([coverage(y, mean, var, lv) for lv in levels])


class PITRecalibrator:
    """Recalibrate Gaussian predictive distributions using PIT values from a calibration set.

    ``fit`` stores the *sorted* PIT values (their empirical CDF R).  A recalibrated central
    interval with nominal level 1-a is  [Q(R^{-1}(a/2)), Q(R^{-1}(1-a/2))]  where Q is the
    original Gaussian quantile function and R^{-1} the empirical quantile function of the PITs.
    """

    def fit(self, pit_cal: np.ndarray) -> "PITRecalibrator":
        self.pit_sorted_ = np.sort(np.asarray(pit_cal, dtype=float).ravel())
        return self

    def transform_pit(self, pit: np.ndarray) -> np.ndarray:
        """R(pit): approximately Uniform(0,1) if the recalibration worked."""
        n = len(self.pit_sorted_)
        return (np.searchsorted(self.pit_sorted_, pit, side="right")) / (n + 1.0)

    def interval(self, mean: np.ndarray, var: np.ndarray, level: float):
        a = 1.0 - level
        lo_p, hi_p = np.quantile(self.pit_sorted_, [a / 2.0, 1.0 - a / 2.0])
        lo_p, hi_p = np.clip([lo_p, hi_p], 1e-6, 1 - 1e-6)
        sd = np.sqrt(var)
        return mean + norm.ppf(lo_p) * sd, mean + norm.ppf(hi_p) * sd

    def coverage(self, y, mean, var, level: float) -> float:
        lo, hi = self.interval(mean, var, level)
        return float(np.mean((y >= lo) & (y <= hi)))
