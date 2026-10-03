"""Experiment 2 -- does a graph-smoothness prior on the factor scores help? (toy 'spatial' data)

A 30 x 30 grid of spots carries 3 smooth latent factor maps; 100 features are noisy sparse
linear read-outs of them (low signal-to-noise ratio).  We compare

    iid     : z_i ~ N(0, I)                        (spatial=False)
    spatial : z[:, k] ~ N(0, (I + rho L)^{-1})     (spatial=True, rho learned by the ELBO)

on (i) recovery of the score space (rotation-invariant: principal angles / mean squared canonical
correlation with the true factor maps) and (ii) predictive quality for 10 % held-out entries.

Run:  python experiments/run_spatial.py --seeds 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from bfa import BayesianFactorAnalysis, data, metrics

ROWS, COLS, K_TRUE = 30, 30, 3


def score_recovery(scores: np.ndarray, Z_true: np.ndarray) -> dict:
    """Rotation-invariant comparison of estimated scores with the true factor maps."""
    ang = metrics.principal_angles_deg(scores - scores.mean(0), Z_true - Z_true.mean(0))
    k = min(scores.shape[1], Z_true.shape[1])
    return {"max_angle": float(ang[:k].max()), "mean_sq_canon_corr": float(np.mean(np.cos(np.radians(ang[:k])) ** 2))}


def fit_and_eval(X, hold, edges, spatial, K, seed, n_iter):
    Xtr = X.copy()
    Xtr[hold] = np.nan
    m = BayesianFactorAnalysis(n_factors=K, spatial=spatial, n_iter=n_iter, seed=seed)
    m.fit(Xtr, edges=edges if spatial else None)
    mean, var = m.posterior_predictive()
    y, mu, v = X[hold], mean[hold], var[hold]
    act = m.active_factors()
    return m, act, {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "lpd": metrics.mean_log_pred_density(y, mu, v),
        "crps": metrics.gaussian_crps(y, mu, v),
        "cov90": metrics.coverage(y, mu, v, 0.9),
        "n_active": int(len(act)),
        "rho": m.rho_,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=2500)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    rows = []
    fig_payload = None
    for seed in range(a.seeds):
        d = data.make_spatial_factor_data(ROWS, COLS, K_TRUE, p=100, seed=seed)
        X, Z, edges = d["X"], d["Z"], d["edges"]
        hold = data.random_holdout(X.shape, 0.10, seed=500 + seed)
        rec = {"seed": seed}
        for name, spatial in (("iid", False), ("spatial", True)):
            m, act, ev = fit_and_eval(X, hold, edges, spatial, K=6, seed=seed, n_iter=a.n_iter)
            rec.update({f"{name}_{k}": v for k, v in ev.items()})
            rec.update({f"{name}_{k}": v for k, v in score_recovery(m.scores()[:, act], Z).items()})
            if seed == 0:
                fig_payload = fig_payload or {"Z": Z}
                fig_payload[name] = (m.scores()[:, act], m.rho_)
        pca = PCA(n_components=K_TRUE).fit(X)   # baseline sees the complete data (an advantage)
        rec.update({f"pca_{k}": v for k, v in score_recovery(pca.transform(X), Z).items()})
        rows.append(rec)
        print(f"seed {seed}: canon-corr iid={rec['iid_mean_sq_canon_corr']:.3f} spatial={rec['spatial_mean_sq_canon_corr']:.3f} "
              f"(PCA {rec['pca_mean_sq_canon_corr']:.3f}) | RMSE iid={rec['iid_rmse']:.3f} spatial={rec['spatial_rmse']:.3f} "
              f"| rho={rec['spatial_rho']:.2f}", flush=True)

    keys = [k for k in rows[0] if k != "seed"]
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])),
                   "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0} for k in keys}
    with open(os.path.join(a.out, "spatial_summary.json"), "w") as f:
        json.dump({"n_seeds": a.seeds, "summary": summary}, f, indent=2)

    # figure: true maps vs. recovered maps (each true factor regressed on the estimated scores -> rotation-free)
    Z = fig_payload["Z"]
    fig, axes = plt.subplots(3, K_TRUE, figsize=(2.3 * K_TRUE, 6.6))
    for r, (label, key) in enumerate((("true", None), ("iid prior", "iid"), ("spatial prior", "spatial"))):
        for f in range(K_TRUE):
            if key is None:
                img = Z[:, f]
            else:
                S = fig_payload[key][0]
                coef, *_ = np.linalg.lstsq(np.c_[S, np.ones(len(S))], Z[:, f], rcond=None)
                img = np.c_[S, np.ones(len(S))] @ coef
            axes[r, f].imshow(img.reshape(ROWS, COLS), cmap="viridis"); axes[r, f].set_xticks([]); axes[r, f].set_yticks([])
            if f == 0:
                axes[r, f].set_ylabel(label)
            if r == 0:
                axes[r, f].set_title(f"factor {f + 1}", fontsize=9)
    fig.suptitle("Recovered factor maps (seed 0; best linear match to each true map)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig5_spatial_maps.png"), dpi=160); plt.close(fig)

    for k in ("iid_mean_sq_canon_corr", "spatial_mean_sq_canon_corr", "pca_mean_sq_canon_corr", "iid_rmse", "spatial_rmse",
              "iid_lpd", "spatial_lpd", "iid_cov90", "spatial_cov90", "spatial_rho"):
        print(f"{k:32s} {summary[k]['mean']:.3f} +- {summary[k]['sd']:.3f}")


if __name__ == "__main__":
    main()
