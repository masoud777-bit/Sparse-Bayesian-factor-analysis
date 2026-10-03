import numpy as np
import pytest
import torch

from bfa import BayesianFactorAnalysis, data, grid_edges, metrics


def _small_fit(spatial=False, n_iter=60, seed=0):
    if spatial:
        d = data.make_spatial_factor_data(3, 4, k=2, p=8, active_per_factor=4, seed=seed)
        m = BayesianFactorAnalysis(n_factors=3, spatial=True, n_iter=n_iter, seed=seed)
        m.fit(d["X"], edges=d["edges"])
    else:
        d = data.make_sparse_factor_data(n=30, p=8, k=2, active_per_factor=4, seed=seed)
        m = BayesianFactorAnalysis(n_factors=3, n_iter=n_iter, seed=seed).fit(d["X"])
    return m, d


def test_expected_squared_error_matches_monte_carlo():
    """The closed-form E_q[(x - w'z)^2] must agree with a brute-force Monte-Carlo estimate."""
    m, _ = _small_fit()
    rng = np.random.default_rng(0)
    S = 40000
    mu, sig = m.mu_W.detach().numpy(), np.sqrt(m._sig2_W().detach().numpy())
    zm, zs = m.m.detach().numpy(), np.sqrt(m._s().detach().numpy())
    Xt = m._Xt.numpy()
    acc = np.zeros_like(Xt)
    for _ in range(S // 1000):
        W = mu[None] + sig[None] * rng.standard_normal((1000,) + mu.shape)      # (S, p, K)
        Z = zm[None] + zs[None] * rng.standard_normal((1000,) + zm.shape)       # (S, n, K)
        pred = np.einsum("snk,spk->snp", Z, W)
        acc += ((Xt[None] - pred) ** 2).sum(0)
    mc = acc / S
    exact = m.expected_sq_error(m._Xt).detach().numpy()
    np.testing.assert_allclose(exact, mc, rtol=0.05, atol=0.03)


def test_spatial_kl_matches_dense_gaussian_kl():
    """Spatial KL(q(Z)||p(Z)) must equal the textbook dense-matrix Gaussian KL."""
    m, d = _small_fit(spatial=True)
    n = m.n_
    edges = d["edges"]
    L = np.zeros((n, n))
    for a, b in edges:
        L[a, a] += 1; L[b, b] += 1; L[a, b] -= 1; L[b, a] -= 1
    Q = np.eye(n) + m.rho_ * L
    mean, S = m.m.detach().numpy(), m._s().detach().numpy()
    prior_cov = np.linalg.inv(Q)                     # textbook route: explicit prior covariance

    def gaussian_kl(m1, S1, S0):
        """KL( N(m1, S1) || N(0, S0) ) for dense covariances."""
        d = len(m1)
        S0inv = np.linalg.inv(S0)
        return 0.5 * (np.trace(S0inv @ S1) + m1 @ S0inv @ m1 - d
                      + np.linalg.slogdet(S0)[1] - np.linalg.slogdet(S1)[1])

    total = sum(gaussian_kl(mean[:, k], np.diag(S[:, k]), prior_cov) for k in range(mean.shape[1]))
    assert float(m.kl_z().detach()) == pytest.approx(total, rel=1e-8)


def test_spatial_prior_reduces_to_iid_when_rho_is_zero():
    m, _ = _small_fit(spatial=True)
    kl_spatial_rho0 = None
    with torch.no_grad():
        m.log_rho.fill_(-40.0)                       # rho ~ 4e-18
        kl_spatial_rho0 = float(m.kl_z())
    m.spatial = False
    kl_iid = float(m.kl_z().detach())
    assert kl_spatial_rho0 == pytest.approx(kl_iid, rel=1e-9)


def test_elbo_increases_during_fitting():
    m, _ = _small_fit(n_iter=150)
    tr = m.result_.elbo_trace
    assert tr[-1] > tr[0] + 1.0
    assert np.mean(tr[-10:]) > np.mean(tr[:10])


def test_ard_recovers_number_of_factors_and_subspace():
    d = data.make_sparse_factor_data(n=400, p=40, k=3, active_per_factor=12, seed=3)
    m = BayesianFactorAnalysis(n_factors=8, n_iter=2500, seed=3).fit(d["X"])
    act = m.active_factors()
    assert len(act) == 3
    ang = metrics.principal_angles_deg(m.loadings()[:, act], d["W"])
    assert ang.max() < 10.0


def test_missing_values_are_ignored_in_fit_and_predicted_reasonably():
    d = data.make_sparse_factor_data(n=400, p=40, k=3, active_per_factor=12, seed=4)
    hold = data.random_holdout(d["X"].shape, 0.1, seed=1)
    Xtr = d["X"].copy(); Xtr[hold] = np.nan
    m = BayesianFactorAnalysis(n_factors=6, n_iter=2500, seed=4).fit(Xtr)
    mean, var = m.posterior_predictive()
    rmse = np.sqrt(np.mean((d["X"][hold] - mean[hold]) ** 2))
    col_mean_rmse = np.sqrt(np.mean((d["X"][hold] - np.nanmean(Xtr, 0)[np.where(hold)[1]]) ** 2))
    assert rmse < 0.8 * col_mean_rmse
    assert 0.8 < metrics.coverage(d["X"][hold], mean[hold], var[hold], 0.9) < 0.97


def test_spatial_model_requires_edges():
    d = data.make_sparse_factor_data(n=20, p=5, k=1, active_per_factor=3)
    with pytest.raises(ValueError):
        BayesianFactorAnalysis(n_factors=2, spatial=True, n_iter=5).fit(d["X"])


def test_grid_edges_count():
    assert len(grid_edges(3, 4)) == 3 * 3 + 2 * 4      # horizontal + vertical
