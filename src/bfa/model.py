"""Deterministic Variational Inference for Sparse Bayesian Factor Analysis.

Formulation:
    x_i | z_i, W, psi ~ N(W z_i, diag(psi))
    W[:, k]           ~ N(0, (1 / alpha_k) * I)     (ARD precision per factor)
    z[:, k]           ~ N(0, Q^{-1})                Q = I + rho * L

Inference:
    Fully factorized mean-field approximation q(W)q(Z). Gaussian likelihood
    yields an exact closed-form ELBO.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch

LOG2PI = math.log(2.0 * math.pi)


def _softplus_inv(x: float) -> float:
    return math.log(math.expm1(x))


def grid_edges(n_rows: int, n_cols: int) -> np.ndarray:
    """Undirected 4-neighbourhood edges of an n_rows x n_cols grid (row-major node ids)."""
    idx = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    horiz = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    vert = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    return np.concatenate([horiz, vert], axis=0)


@dataclass
class FitResult:
    elbo_trace: List[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


class BayesianFactorAnalysis:
    """Sparse Bayesian factor analysis (ARD prior) with optional graph-smooth factor scores.

    Parameters
    ----------
    n_factors : maximal number of factors K_max (ARD shrinks the unused ones to zero).
    spatial   : use the graph prior Q = I + rho * L on the factor scores (needs ``edges``).
    ard       : learn one prior precision per factor (True) or keep alpha fixed at 1 (False).
    lr, n_iter, tol : Adam step size, maximal iterations, relative ELBO tolerance.
    seed      : seed for the (small) random initialisation.
    """

    def __init__(self, n_factors: int = 10, spatial: bool = False, ard: bool = True,
                 lr: float = 0.03, n_iter: int = 3000, tol: float = 1e-7, seed: int = 0,
                 dtype: torch.dtype = torch.float64):
        self.K = int(n_factors)
        self.spatial = bool(spatial)
        self.ard = bool(ard)
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.dtype = dtype
        self.result_: Optional[FitResult] = None

    # ------------------------------------------------------------------ fitting
    def fit(self, X: np.ndarray, mask: Optional[np.ndarray] = None,
            edges: Optional[np.ndarray] = None, verbose: bool = False) -> "BayesianFactorAnalysis":
        """Fit the model. Missing entries are NaN in ``X`` and/or ``mask == 0``."""
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        obs = ~np.isnan(X)
        if mask is not None:
            obs &= np.asarray(mask).astype(bool)
        self.n_, self.p_ = n, p

        # standardise columns using observed entries only
        Xo = np.where(obs, X, np.nan)
        self.mean_ = np.nanmean(Xo, axis=0)
        self.sd_ = np.nanstd(Xo, axis=0)
        self.sd_[self.sd_ < 1e-12] = 1.0
        Xs = np.where(obs, (X - self.mean_) / self.sd_, 0.0)

        Xt = torch.tensor(Xs, dtype=self.dtype)
        Mt = torch.tensor(obs.astype(float), dtype=self.dtype)
        n_obs = Mt.sum()
        K = self.K

        # ---- initialisation from an SVD of the (zero-filled) standardised data
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        r = min(K, len(S))
        W0 = np.zeros((p, K))
        m0 = np.zeros((n, K))
        W0[:, :r] = Vt[:r].T * S[:r] / math.sqrt(n)
        m0[:, :r] = U[:, :r] * math.sqrt(n)
        rng = np.random.default_rng(self.seed)
        W0 += 0.01 * rng.standard_normal(W0.shape)
        m0 += 0.01 * rng.standard_normal(m0.shape)
        resid = Xs - m0 @ W0.T
        psi0 = np.clip((resid ** 2 * obs).sum(0) / np.maximum(obs.sum(0), 1), 0.05, None)

        self.mu_W = torch.tensor(W0, dtype=self.dtype, requires_grad=True)
        self.rho_W = torch.full((p, K), _softplus_inv(0.1), dtype=self.dtype, requires_grad=True)
        self.m = torch.tensor(m0, dtype=self.dtype, requires_grad=True)
        self.rho_s = torch.full((n, K), _softplus_inv(0.1), dtype=self.dtype, requires_grad=True)
        self.log_psi = torch.tensor(np.log(psi0), dtype=self.dtype, requires_grad=True)
        self.alpha = torch.ones(K, dtype=self.dtype)
        params = [self.mu_W, self.rho_W, self.m, self.rho_s, self.log_psi]

        if self.spatial:
            if edges is None:
                raise ValueError("spatial=True requires `edges` (E x 2 integer array).")
            self._setup_graph(edges, n)
            self.log_rho = torch.tensor(math.log(1.0), dtype=self.dtype, requires_grad=True)
            params.append(self.log_rho)

        opt = torch.optim.Adam(params, lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_iter,
                                                           eta_min=self.lr * 0.02)
        trace: List[float] = []
        converged = False
        for it in range(self.n_iter):
            if self.ard:
                self._update_alpha()
            opt.zero_grad()
            elbo = self._elbo(Xt, Mt)
            (-elbo / n_obs).backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            trace.append(float(elbo.detach()))
            if verbose and it % 200 == 0:
                print(f"iter {it:5d}  ELBO/obs = {trace[-1] / float(n_obs):.5f}")
            if it > 200 and abs(trace[-1] - trace[-51]) < self.tol * abs(trace[-1]) * 50:
                converged = True
                break
        self.result_ = FitResult(trace, len(trace), converged)
        self._Xt, self._Mt = Xt, Mt
        return self

    # ------------------------------------------------------- ELBO ingredients
    def _sig2_W(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.rho_W) ** 2

    def _s(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.rho_s) ** 2

    def _setup_graph(self, edges: np.ndarray, n: int) -> None:
        edges = np.asarray(edges, dtype=np.int64)
        self._edges = torch.tensor(edges)
        deg = np.zeros(n)
        np.add.at(deg, edges[:, 0], 1.0)
        np.add.at(deg, edges[:, 1], 1.0)
        L = np.diag(deg)
        L[edges[:, 0], edges[:, 1]] -= 1.0
        L[edges[:, 1], edges[:, 0]] -= 1.0
        self._deg = torch.tensor(deg, dtype=self.dtype)
        self._lap_eig = torch.tensor(np.clip(np.linalg.eigvalsh(L), 0.0, None), dtype=self.dtype)

    @property
    def rho_(self) -> float:
        """Learned smoothness parameter of the graph prior (0 for the iid prior)."""
        return float(torch.exp(self.log_rho.detach())) if self.spatial else 0.0

    @torch.no_grad()
    def _update_alpha(self) -> None:
        """Closed-form empirical-Bayes update of the ARD precisions."""
        denom = (self.mu_W ** 2 + self._sig2_W()).sum(0)
        self.alpha = (self.p_ / denom).clamp(1e-3, 1e8)

    def expected_sq_error(self, X: torch.Tensor) -> torch.Tensor:
        """E_q[(x_ij - w_j' z_i)^2] for all (i, j); exact under the mean-field q."""
        mu, sig2, m, s = self.mu_W, self._sig2_W(), self.m, self._s()
        mean_pred = m @ mu.T
        var_term = (m ** 2 + s) @ sig2.T + s @ (mu ** 2).T
        return (X - mean_pred) ** 2 + var_term

    def kl_z(self) -> torch.Tensor:
        """KL( q(Z) || p(Z) ), summed over factors."""
        m, s = self.m, self._s()
        n = self.n_
        if not self.spatial:
            return 0.5 * (s + m ** 2 - 1.0 - torch.log(s)).sum()
        rho = torch.exp(self.log_rho)
        i, j = self._edges[:, 0], self._edges[:, 1]
        tr_QS = ((1.0 + rho * self._deg)[:, None] * s).sum(0)                 # (K,)
        quad = (m ** 2).sum(0) + rho * ((m[i] - m[j]) ** 2).sum(0)            # (K,)
        logdet_Q = torch.log1p(rho * self._lap_eig).sum()                     # scalar
        return 0.5 * (tr_QS + quad - n - logdet_Q - torch.log(s).sum(0)).sum()

    def kl_W(self) -> torch.Tensor:
        """KL( q(W) || p(W | alpha) )."""
        sig2, mu, a = self._sig2_W(), self.mu_W, self.alpha[None, :]
        return (-0.5 * torch.log(a * sig2) + 0.5 * a * (sig2 + mu ** 2) - 0.5).sum()

    def _elbo(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        psi = torch.exp(self.log_psi)[None, :]
        ell = -0.5 * (M * (LOG2PI + torch.log(psi) + self.expected_sq_error(X) / psi)).sum()
        return ell - self.kl_z() - self.kl_W()

    def elbo(self) -> float:
        return float(self._elbo(self._Xt, self._Mt).detach())

    # ------------------------------------------------------------- read-outs
    @torch.no_grad()
    def loadings(self, original_units: bool = True) -> np.ndarray:
        """Posterior-mean loadings (p x K).

        ``original_units=True`` rescales rows by the feature standard deviations so that the
        result is comparable with the loadings of the data-generating model; ``False`` returns
        the loadings of the internally standardised data.
        """
        W = self.mu_W.detach().numpy().copy()
        return W * self.sd_[:, None] if original_units else W

    @torch.no_grad()
    def scores(self) -> np.ndarray:
        """Posterior-mean factor scores (n x K)."""
        return self.m.detach().numpy().copy()

    @torch.no_grad()
    def factor_variance_share(self) -> np.ndarray:
        """Share of the total standardised variance carried by each factor's loadings."""
        return ((self.mu_W ** 2).sum(0) / self.p_).numpy()

    def active_factors(self, threshold: float = 0.01) -> np.ndarray:
        """Indices of factors whose variance share exceeds ``threshold`` (ARD-selected set)."""
        return np.where(self.factor_variance_share() > threshold)[0]

    @torch.no_grad()
    def noise_variance(self) -> np.ndarray:
        """Per-feature noise variance in original units."""
        return torch.exp(self.log_psi).numpy() * self.sd_ ** 2

    @torch.no_grad()
    def posterior_predictive(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gaussian approximation to the predictive mean and variance of every entry.

        Variance = noise + the variational uncertainty of w_j' z_i (both W and Z);
        returned in original units.  Used for held-out (masked) entries.
        """
        mu, sig2, m, s = self.mu_W, self._sig2_W(), self.m, self._s()
        mean = m @ mu.T
        # Variance of the bilinear product w_jk * z_ik
        var = (m ** 2 + s) @ sig2.T + s @ (mu ** 2).T + torch.exp(self.log_psi)[None, :]
        return mean.numpy() * self.sd_ + self.mean_, var.numpy() * self.sd_ ** 2
