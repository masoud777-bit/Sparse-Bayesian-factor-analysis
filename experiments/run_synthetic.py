"""Experiment 1 -- recovery, automatic factor selection and calibrated uncertainty (iid prior).

For every seed we simulate  X = Z W' + noise  (n=500, p=60, 4 sparse factors, heteroscedastic
noise) and

  (A) fit the model to the complete data with K_max = 10 factors and compare with PCA and
      sklearn's maximum-likelihood FactorAnalysis (both told the *true* K = 4);
  (B) hide 10 % of the entries, fit on the rest, and evaluate the predictive distribution of
      the hidden entries: RMSE, log predictive density, CRPS, interval coverage, and the same
      after a PIT recalibration fitted on half of the hidden entries and tested on the other half.

Run:  python experiments/run_synthetic.py --seeds 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis

from bfa import BayesianFactorAnalysis, data, metrics

LEVELS = np.linspace(0.05, 0.99, 20)
REPORT_LEVELS = (0.5, 0.8, 0.9, 0.95)


def iterative_svd_impute(X: np.ndarray, hold: np.ndarray, rank: int, n_iter: int = 200) -> np.ndarray:
    """Hard-impute baseline (point estimates only): alternate rank-`rank` SVD and refilling."""
    Xf = X.copy()
    col_mean = np.nanmean(np.where(hold, np.nan, X), axis=0)
    Xf[hold] = np.take(col_mean, np.where(hold)[1])
    for _ in range(n_iter):
        mu = Xf.mean(0)
        U, S, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
        low = (U[:, :rank] * S[:rank]) @ Vt[:rank] + mu
        Xf[hold] = low[hold]
    return Xf


def run_seed(seed: int, n_iter: int, noise: str) -> dict:
    d = data.make_sparse_factor_data(n=500, p=60, k=4, seed=seed, noise=noise)
    X, W_true = d["X"], d["W"]
    out = {"seed": seed}

    # ---- (A) parameter recovery on complete data
    t0 = time.time()
    m = BayesianFactorAnalysis(n_factors=10, n_iter=n_iter, seed=seed).fit(X)
    act = m.active_factors()
    W_hat = m.loadings()[:, act]
    out["A_time_s"] = time.time() - t0
    out["A_n_active"] = int(len(act))
    out["A_converged"] = bool(m.result_.converged)
    k_ang = min(W_hat.shape[1], 4)
    out["A_bfa_max_angle"] = float(metrics.principal_angles_deg(W_hat, W_true)[:k_ang].max())
    out["A_bfa_cov_err"] = metrics.cov_rel_error(W_hat, W_true)
    pca = PCA(n_components=4).fit(X)
    W_pca = pca.components_.T * np.sqrt(pca.explained_variance_)
    out["A_pca_max_angle"] = float(metrics.principal_angles_deg(W_pca, W_true).max())
    out["A_pca_cov_err"] = metrics.cov_rel_error(W_pca, W_true)
    fa = FactorAnalysis(n_components=4, random_state=seed, max_iter=1000).fit(X)
    W_fa = fa.components_.T
    out["A_fa_max_angle"] = float(metrics.principal_angles_deg(W_fa, W_true).max())
    out["A_fa_cov_err"] = metrics.cov_rel_error(W_fa, W_true)
    if seed == 0:
        out["_elbo_trace"] = m.result_.elbo_trace
        out["_var_share"] = m.factor_variance_share().tolist()
        Wv, _ = metrics.varimax(W_hat)
        # rotation resolved by Procrustes towards the truth -> for the figure only, never for metrics
        out["_W_est_aligned"] = (metrics.procrustes_align(Wv, W_true) if Wv.shape[1] == 4 else Wv).tolist()
        out["_W_true"] = W_true.tolist()

    # ---- (B) predictive uncertainty for held-out entries
    hold = data.random_holdout(X.shape, 0.10, seed=1000 + seed)
    Xtrain = X.copy()
    Xtrain[hold] = np.nan
    mb = BayesianFactorAnalysis(n_factors=10, n_iter=n_iter, seed=seed).fit(Xtrain)
    mean, var = mb.posterior_predictive()
    ii, jj = np.where(hold)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ii))
    cal, test = perm[: len(ii) // 2], perm[len(ii) // 2:]
    y = X[ii, jj]
    mu_h, var_h = mean[ii, jj], var[ii, jj]

    def block(sel, prefix):
        yy, mm, vv = y[sel], mu_h[sel], var_h[sel]
        out[f"{prefix}_rmse"] = float(np.sqrt(np.mean((yy - mm) ** 2)))
        out[f"{prefix}_lpd"] = metrics.mean_log_pred_density(yy, mm, vv)
        out[f"{prefix}_crps"] = metrics.gaussian_crps(yy, mm, vv)
        for lv in REPORT_LEVELS:
            out[f"{prefix}_cov{int(lv*100)}"] = metrics.coverage(yy, mm, vv, lv)

    block(test, "B_raw")
    rec = metrics.PITRecalibrator().fit(metrics.gaussian_pit(y[cal], mu_h[cal], var_h[cal]))
    for lv in REPORT_LEVELS:
        out[f"B_recal_cov{int(lv*100)}"] = rec.coverage(y[test], mu_h[test], var_h[test], lv)
    out["B_bfa_active"] = int(len(mb.active_factors()))

    Xmean = np.take(np.nanmean(Xtrain, axis=0), jj)
    out["B_rmse_column_mean"] = float(np.sqrt(np.mean((y[test] - Xmean[test]) ** 2)))
    Ximp = iterative_svd_impute(X, hold, rank=4)
    out["B_rmse_hard_impute_K4"] = float(np.sqrt(np.mean((y[test] - Ximp[ii, jj][test]) ** 2)))
    out["B_noise_floor_rmse"] = float(np.sqrt(np.mean(d["psi"][jj[test]])))  # irreducible error (true z known)

    out["_cov_curve_raw"] = metrics.coverage_curve(y[test], mu_h[test], var_h[test], LEVELS).tolist()
    out["_cov_curve_recal"] = [rec.coverage(y[test], mu_h[test], var_h[test], lv) for lv in LEVELS]
    return out


def summarise(rows):
    keys = [k for k in rows[0] if not k.startswith("_") and k != "seed"]
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0} for k in keys}


def make_figures(rows_by, outdir):
    r0 = rows_by["gaussian"][0]
    # ARD / variance share
    fig, ax = plt.subplots(figsize=(5, 3.2))
    share = np.array(r0["_var_share"])
    ax.bar(range(1, len(share) + 1), np.maximum(share, 1e-6), color=["#2a6f97" if s > 0.01 else "#bbbbbb" for s in share])
    ax.set_yscale("log")
    ax.axhline(0.01, ls="--", c="k", lw=0.8)
    ax.set_xlabel("factor (K_max = 10, true K = 4)")
    ax.set_ylabel("share of standardised variance")
    ax.set_title("ARD switches off the unneeded factors")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_ard_variance_share.png"), dpi=160); plt.close(fig)

    # calibration curves, one panel per noise scenario
    fig, axes = plt.subplots(1, len(rows_by), figsize=(4.2 * len(rows_by), 4.2), squeeze=False)
    titles = {"gaussian": "correctly specified (Gaussian noise)", "t3": "misspecified (Student-t, 3 d.o.f. noise)"}
    for ax, (scen, rws) in zip(axes[0], rows_by.items()):
        raw = np.array([r["_cov_curve_raw"] for r in rws]); rec = np.array([r["_cov_curve_recal"] for r in rws])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
        for arr, lab, c in ((raw, "variational predictive", "#c1121f"), (rec, "after PIT recalibration", "#2a6f97")):
            mu = arr.mean(0)
            sd = arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1])
            ax.plot(LEVELS, mu, c=c, label=lab); ax.fill_between(LEVELS, mu - sd, mu + sd, color=c, alpha=0.2)
        ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage (held-out entries)")
        ax.set_title(titles.get(scen, scen), fontsize=9); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_calibration.png"), dpi=160); plt.close(fig)

    # loadings (seed 0) -- rotation resolved by varimax + Procrustes to truth (evaluation only)
    Wt, We = np.array(r0["_W_true"]), np.array(r0["_W_est_aligned"])
    if We.shape == Wt.shape:
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        v = max(abs(Wt).max(), abs(We).max())
        for ax, M, ttl in zip(axes, (Wt, We), ("true loadings", "estimated (varimax, Procrustes-aligned)")):
            ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v); ax.set_title(ttl, fontsize=8)
            ax.set_xlabel("factor"); ax.set_ylabel("feature")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_loadings.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    tr = np.array(r0["_elbo_trace"]) / (500 * 60)
    ax.plot(tr); ax.set_xlabel("iteration"); ax.set_ylabel("ELBO per entry")
    ax.set_ylim(tr[len(tr) // 20], tr.max() + 0.005)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig4_elbo.png"), dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    rows_by, summary = {}, {}
    for scen in ("gaussian", "t3"):
        rows = []
        for sd in range(a.seeds):
            r = run_seed(sd, a.n_iter, scen)
            rows.append(r)
            print(f"[{scen}] seed {sd}: active={r['A_n_active']}  angle(BFA/FA/PCA)={r['A_bfa_max_angle']:.1f}/"
                  f"{r['A_fa_max_angle']:.1f}/{r['A_pca_max_angle']:.1f}  cov95 raw={r['B_raw_cov95']:.3f} "
                  f"recal={r['B_recal_cov95']:.3f}", flush=True)
        rows_by[scen] = rows
        summary[scen] = summarise(rows)
    with open(os.path.join(a.out, "synthetic_summary.json"), "w") as f:
        json.dump({"n_seeds": a.seeds, "summary": summary}, f, indent=2)
    make_figures(rows_by, a.out)
    for scen in summary:
        print(scen, json.dumps({k: round(v["mean"], 3) for k, v in summary[scen].items()}))


if __name__ == "__main__":
    main()
