All numbers are mean ± sd over 8 independent simulated data sets (seeds 0..7).

### Experiment 1 - Correctly specified model (Gaussian noise)

**(A) Parameter recovery, complete data (n=500, p=60, true K=4; K_max=10 for the Bayesian model).**

| method | factors used | max principal angle to true loadings (deg) | rel. error of W W' |
|---|---|---|---|
| Bayesian FA (ARD), K chosen automatically | 4.0 ± 0.0 | 4.68 ± 0.26 | 0.143 ± 0.026 |
| sklearn FactorAnalysis, K=4 given | 4 | 4.70 ± 0.24 | 0.141 ± 0.025 |
| PCA, K=4 given | 4 | 4.83 ± 0.21 | 0.146 ± 0.024 |

**(B) Held-out entries (10 % hidden). Coverage columns: empirical coverage of the central interval with the given nominal level.**

| predictive distribution | RMSE | CRPS | cov 50 % | cov 80 % | cov 90 % | cov 95 % |
|---|---|---|---|---|---|---|
| variational posterior predictive | 0.840 ± 0.019 | 0.468 ± 0.010 | 0.497 ± 0.007 | 0.798 ± 0.007 | 0.896 ± 0.007 | 0.949 ± 0.002 |
| + PIT recalibration (fit on half of the held-out entries) | same | - | 0.491 ± 0.013 | 0.796 ± 0.016 | 0.895 ± 0.015 | 0.946 ± 0.009 |

Point-prediction reference (RMSE): column mean 1.406 ± 0.031, rank-4 iterative-SVD imputation 0.841 ± 0.019, irreducible noise floor 0.806 ± 0.015.

### Experiment 1 - Misspecified model (Student-t noise, 3 d.o.f.)

**(A) Parameter recovery, complete data (n=500, p=60, true K=4; K_max=10 for the Bayesian model).**

| method | factors used | max principal angle to true loadings (deg) | rel. error of W W' |
|---|---|---|---|
| Bayesian FA (ARD), K chosen automatically | 4.0 ± 0.0 | 4.50 ± 0.55 | 0.136 ± 0.022 |
| sklearn FactorAnalysis, K=4 given | 4 | 4.53 ± 0.55 | 0.135 ± 0.021 |
| PCA, K=4 given | 4 | 4.71 ± 0.49 | 0.143 ± 0.018 |

**(B) Held-out entries (10 % hidden). Coverage columns: empirical coverage of the central interval with the given nominal level.**

| predictive distribution | RMSE | CRPS | cov 50 % | cov 80 % | cov 90 % | cov 95 % |
|---|---|---|---|---|---|---|
| variational posterior predictive | 0.862 ± 0.099 | 0.415 ± 0.010 | 0.635 ± 0.008 | 0.868 ± 0.006 | 0.924 ± 0.005 | 0.952 ± 0.004 |
| + PIT recalibration (fit on half of the held-out entries) | same | - | 0.503 ± 0.017 | 0.805 ± 0.013 | 0.904 ± 0.004 | 0.953 ± 0.004 |

Point-prediction reference (RMSE): column mean 1.417 ± 0.082, rank-4 iterative-SVD imputation 0.867 ± 0.097, irreducible noise floor 0.806 ± 0.015.

### Experiment 2 - graph-smooth factor scores (30x30 grid, 3 factors, p=100, low SNR; 8 seeds)

| model | mean squared canonical correlation with true factor maps | held-out RMSE | held-out log predictive density | cov 90 % |
|---|---|---|---|---|
| iid prior | 0.916 ± 0.013 | 1.518 ± 0.016 | -1.827 ± 0.009 | 0.901 ± 0.003 |
| spatial (graph) prior, rho learned | 0.969 ± 0.004 | 1.505 ± 0.017 | -1.819 ± 0.010 | 0.899 ± 0.002 |
| PCA (K=3, sees complete data) | 0.922 ± 0.012 | - | - | - |

Learned smoothness rho: 92.5 ± 8.9.
