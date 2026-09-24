import numpy as np
from scipy.stats import kstest

from bfa import metrics


def test_principal_angles_are_rotation_invariant():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 4))
    R, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    assert metrics.principal_angles_deg(A, A @ R).max() < 1e-5
    B = rng.standard_normal((30, 4))
    assert metrics.principal_angles_deg(A, B).max() > 30.0


def test_covariance_error_is_zero_for_rotated_copy():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((20, 3))
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    assert metrics.cov_rel_error(W @ R, W) < 1e-10


def test_procrustes_undoes_a_rotation():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((20, 3))
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    np.testing.assert_allclose(metrics.procrustes_align(W @ R, W), W, atol=1e-8)


def test_varimax_increases_sparsity_of_a_rotated_sparse_matrix():
    rng = np.random.default_rng(3)
    W = np.zeros((40, 3))
    for k in range(3):
        W[k * 10:(k + 1) * 10, k] = rng.uniform(0.8, 1.5, 10)
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    Wm = W @ R
    Wv, _ = metrics.varimax(Wm)
    assert (Wv ** 4).sum() > (Wm ** 4).sum()


def test_pit_of_correct_gaussian_predictive_is_uniform_and_covered():
    rng = np.random.default_rng(4)
    mean, var = rng.normal(size=5000), rng.uniform(0.5, 2.0, 5000)
    y = mean + rng.standard_normal(5000) * np.sqrt(var)
    assert kstest(metrics.gaussian_pit(y, mean, var), "uniform").pvalue > 0.01
    assert abs(metrics.coverage(y, mean, var, 0.9) - 0.9) < 0.02


def test_gaussian_crps_matches_monte_carlo():
    rng = np.random.default_rng(5)
    y, mean, var = np.array([0.7]), np.array([0.2]), np.array([1.5])
    s1 = mean + np.sqrt(var) * rng.standard_normal(200000)
    s2 = mean + np.sqrt(var) * rng.standard_normal(200000)
    mc = np.mean(np.abs(s1 - y)) - 0.5 * np.mean(np.abs(s1 - s2))
    assert abs(metrics.gaussian_crps(y, mean, var) - mc) < 5e-3


def test_pit_recalibration_repairs_heavy_tailed_miscalibration():
    """Gaussian predictive for t_3 data over-covers central intervals; recalibration fixes it."""
    rng = np.random.default_rng(6)
    n = 40000
    y = rng.standard_t(3, n) / np.sqrt(3.0)        # unit variance heavy tails
    mean, var = np.zeros(n), np.ones(n)
    cal, test = slice(0, n // 2), slice(n // 2, n)
    rec = metrics.PITRecalibrator().fit(metrics.gaussian_pit(y[cal], mean[cal], var[cal]))
    raw50 = metrics.coverage(y[test], mean[test], var[test], 0.5)
    fix50 = rec.coverage(y[test], mean[test], var[test], 0.5)
    assert abs(raw50 - 0.5) > 0.08 and abs(fix50 - 0.5) < 0.02
    fixed_pit = rec.transform_pit(metrics.gaussian_pit(y[test], mean[test], var[test]))
    assert kstest(fixed_pit, "uniform").pvalue > 0.01
