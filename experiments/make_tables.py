"""Turn results/*.json into Markdown tables (results/RESULTS.md)."""
import json
import os

R = os.path.join(os.path.dirname(__file__), "..", "results")


def f(d, k, nd=3):
    return f"{d[k]['mean']:.{nd}f} ± {d[k]['sd']:.{nd}f}"


def main():
    syn = json.load(open(os.path.join(R, "synthetic_summary.json")))
    spa = json.load(open(os.path.join(R, "spatial_summary.json")))
    out = []
    ns = syn["n_seeds"]
    out.append(f"All numbers are mean ± sd over {ns} independent simulated data sets (seeds 0..{ns - 1}).\n")
    for scen, title in (("gaussian", "Correctly specified model (Gaussian noise)"),
                        ("t3", "Misspecified model (Student-t noise, 3 d.o.f.)")):
        d = syn["summary"][scen]
        out.append(f"### Experiment 1 - {title}\n")
        out.append("**(A) Parameter recovery, complete data (n=500, p=60, true K=4; K_max=10 for the Bayesian model).**\n")
        out.append("| method | factors used | max principal angle to true loadings (deg) | rel. error of W W' |")
        out.append("|---|---|---|---|")
        out.append(f"| Bayesian FA (ARD), K chosen automatically | {f(d, 'A_n_active', 1)} | {f(d, 'A_bfa_max_angle', 2)} | {f(d, 'A_bfa_cov_err')} |")
        out.append(f"| sklearn FactorAnalysis, K=4 given | 4 | {f(d, 'A_fa_max_angle', 2)} | {f(d, 'A_fa_cov_err')} |")
        out.append(f"| PCA, K=4 given | 4 | {f(d, 'A_pca_max_angle', 2)} | {f(d, 'A_pca_cov_err')} |\n")
        out.append("**(B) Held-out entries (10 % hidden). Coverage columns: empirical coverage of the central interval with the given nominal level.**\n")
        out.append("| predictive distribution | RMSE | CRPS | cov 50 % | cov 80 % | cov 90 % | cov 95 % |")
        out.append("|---|---|---|---|---|---|---|")
        out.append(f"| variational posterior predictive | {f(d, 'B_raw_rmse')} | {f(d, 'B_raw_crps')} | {f(d, 'B_raw_cov50')} | {f(d, 'B_raw_cov80')} | {f(d, 'B_raw_cov90')} | {f(d, 'B_raw_cov95')} |")
        out.append(f"| + PIT recalibration (fit on half of the held-out entries) | same | - | {f(d, 'B_recal_cov50')} | {f(d, 'B_recal_cov80')} | {f(d, 'B_recal_cov90')} | {f(d, 'B_recal_cov95')} |\n")
        out.append(f"Point-prediction reference (RMSE): column mean {f(d, 'B_rmse_column_mean')}, rank-4 iterative-SVD imputation {f(d, 'B_rmse_hard_impute_K4')}, "
                   f"irreducible noise floor {f(d, 'B_noise_floor_rmse')}.\n")
    d = spa["summary"]
    out.append(f"### Experiment 2 - graph-smooth factor scores (30x30 grid, 3 factors, p=100, low SNR; {spa['n_seeds']} seeds)\n")
    out.append("| model | mean squared canonical correlation with true factor maps | held-out RMSE | held-out log predictive density | cov 90 % |")
    out.append("|---|---|---|---|---|")
    out.append(f"| iid prior | {f(d, 'iid_mean_sq_canon_corr')} | {f(d, 'iid_rmse')} | {f(d, 'iid_lpd')} | {f(d, 'iid_cov90')} |")
    out.append(f"| spatial (graph) prior, rho learned | {f(d, 'spatial_mean_sq_canon_corr')} | {f(d, 'spatial_rmse')} | {f(d, 'spatial_lpd')} | {f(d, 'spatial_cov90')} |")
    out.append(f"| PCA (K=3, sees complete data) | {f(d, 'pca_mean_sq_canon_corr')} | - | - | - |\n")
    out.append(f"Learned smoothness rho: {f(d, 'spatial_rho', 1)}.\n")
    with open(os.path.join(R, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
