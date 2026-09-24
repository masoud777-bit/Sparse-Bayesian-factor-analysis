# Sparse Bayesian Factor Analysis via Exact Variational Inference

A PyTorch implementation of sparse Bayesian factor analysis (BFA) designed to investigate latent subspace recovery, automatic dimensionality selection via ARD, and spatial regularization on arbitrary graph structures.

The core motivation is methodological: evaluating how deterministic mean-field variational inference behaves under well-specified versus heavy-tailed observation noise, and verifying whether graph-Laplacian priors reliably enforce spatial coherence in latent representations.

## Key Properties

- **Closed-form ELBO:** With a Gaussian likelihood and Gaussian factor priors, the expected log-likelihood under the mean-field family $q(W)q(Z)$ is tractable in closed form. This eliminates Monte Carlo gradient noise during optimization.
- **Factor pruning via ARD:** Factor-specific precision hyper-parameters $\alpha_k$ are updated via empirical Bayes, shrinking superfluous factors to zero.
- **Spatial regularization:** Latent factor scores can be endowed with a graph-precision prior $Q = I + \rho L$ (where $L$ is the graph Laplacian), encouraging spatially smooth factor maps across irregular grids or spatial coordinates.
- **Identifiability-aware metrics:** Factor recovery is evaluated with rotation-invariant criteria: Grassmannian subspace distance, principal angles, and relative error on $W W^\top$.
- **Predictive calibration:** Assessment on held-out entries includes empirical coverage curves, CRPS, and post-hoc PIT recalibration.

## Model Formulation

Given centered observations $x_i \in \mathbb{R}^p$ for $i = 1, \dots, n$ and a latent capacity $K_{\max}$:

$$x_i \mid z_i, W, \psi \sim \mathcal{N}(W z_i, \operatorname{diag}(\psi))$$
$$W_{jk} \mid \alpha_k \sim \mathcal{N}\left(0, \alpha_k^{-1}\right)$$
$$z_k \sim \mathcal{N}\left(0, Q^{-1}\right), \quad Q = I + \rho L$$

Setting $\rho = 0$ recovers the standard i.i.d. latent prior $z_i \sim \mathcal{N}(0, I)$.

### Variational Objective

Under the fully factorized variational family:

$$q(W, Z) = \prod_{j=1}^p \prod_{k=1}^K \mathcal{N}(\mu_{jk}^W, \sigma_{jk}^{W2}) \prod_{i=1}^n \prod_{k=1}^K \mathcal{N}(m_{ik}^z, s_{ik}^z)$$

the expected reconstruction error simplifies to:

$$\mathbb{E}_q\left[(x_{ij} - w_j^\top z_i)^2\right] = (x_{ij} - \mu_j^{W\top} m_i^z)^2 + \sum_{k=1}^K \left[ \sigma_{jk}^{W2}(m_{ik}^{z2} + s_{ik}^z) + \mu_{jk}^{W2} s_{ik}^z \right]$$

The objective is optimized with Adam using analytical gradients for the variational parameters. The ARD precisions are updated in closed form via empirical Bayes:

$$\alpha_k = \frac{p}{\sum_{j=1}^p (\mu_{jk}^{W2} + \sigma_{jk}^{W2})}$$

### Identifiability

A factor model is identified only up to an orthogonal rotation of the latent space, plus sign and permutation. Individual loadings are therefore not interpretable on their own. To handle this:

- Recovery is measured with rotation-invariant quantities: principal angles between the true and estimated subspaces, and the relative error of the low-rank covariance $W W^\top$.
- Varimax gives a rotation that requires no ground truth. The loadings heat-map (`results/fig3_loadings.png`) is additionally aligned to the truth with orthogonal Procrustes, but for display only.

## Empirical Validation

All experiments run on synthetic benchmarks with known ground truth.

1. **Dimensionality recovery:** On $n=500, p=60$ with 4 true sparse factors, ARD correctly prunes capacity from $K_{\max}=10$ to 4 active components across all runs.
2. **Noise misspecification:** Under Student-$t$ noise ($\nu=3$), Gaussian variational predictive intervals over-cover central regions. Empirical PIT recalibration corrects this.
3. **Spatial factor smoothing:** On a 2D lattice (a stand-in for spatial coordinates), the graph prior improves factor correlation with the ground truth compared to i.i.d. factor priors.

### Results

Reproduce with the commands below. All numbers are mean $\pm$ sd over 8 simulated data sets. Full tables: [`results/RESULTS.md`](results/RESULTS.md).

**Experiment 1 — iid prior ($n=500, p=60$, 4 true factors, $K_{\max}=10$):**

- ARD selects **exactly 4 factors in all 16 runs** (both noise scenarios) without being told $K$ (`results/fig1_ard_variance_share.png`).
- Loading-subspace recovery is on par with PCA and maximum-likelihood factor analysis that are *given* the true $K$ (max principal angle ~4.5–4.8 degrees for all three). No claim of superiority is made.
- Uncertainty for 10% held-out entries: when the model is correctly specified, the variational predictive intervals are already well calibrated (90% interval covers $0.896 \pm 0.007$). With **heavy-tailed (Student-t, 3 d.o.f.) noise**, the Gaussian predictive over-covers central intervals (nominal 50% $\to 0.635 \pm 0.008$). A **PIT recalibration** fitted on half of the held-out entries repairs this on the other half ($0.503 \pm 0.017$). See `results/fig2_calibration.png`.

**Experiment 2 — graph-smooth factors ($30 \times 30$ grid, 3 factors, $p=100$, low SNR):**

- The graph prior with a learned smoothness $\rho$ recovers the latent maps better than the iid prior (mean squared canonical correlation $0.969 \pm 0.004$ vs $0.916 \pm 0.013$; PCA $0.922 \pm 0.012$), with a small gain in held-out RMSE ($1.505$ vs $1.518$) and log predictive density. See `results/fig5_spatial_maps.png`.

## Reproduce

```bash
conda create -n bfa python=3.12 -y
conda activate bfa
pip install -r requirements.txt

pytest -q
python experiments/run_synthetic.py --seeds 8
python experiments/run_spatial.py   --seeds 8
python experiments/make_tables.py
```

Tested with Python 3.11/3.12 and PyTorch 2.x (CPU).

## Repository Layout

```text
├── src/bfa/
│   ├── model.py        # Core BFA module & graph setup
│   ├── metrics.py      # Rotation-invariant metrics, PIT, CRPS
│   └── data.py         # Synthetic generators (i.i.d. & spatial)
├── experiments/        # Benchmarking scripts
├── tests/              # Analytical vs. numerical sanity checks
└── results/            # Summary tables and diagnostic plots
```

## Limitations

- **Synthetic data only.** Nothing here is evidence about real single-cell or spatial-transcriptomics data. Those are counts with overdispersion and batch effects; a Gaussian likelihood would need to be replaced by a negative-binomial or Poisson one, and then the ELBO is no longer closed-form in the likelihood term.
- **Mean-field VI** typically understates posterior variances. Here the predictive intervals for held-out entries are dominated by the noise term and are well calibrated, but the marginal posteriors of the loadings and scores have not been checked for calibration (e.g. by simulation-based calibration).
- $\alpha_k$, $\psi_j$, and $\rho$ are point estimates (type-II maximum likelihood), not fully Bayesian.
- $q(Z)$ uses free local parameters ($n \times K$), and the spatial prior is fitted full-batch with a dense eigendecomposition of the Laplacian ($O(n^3)$, fine up to a few thousand spots). A mini-batch / amortised version and a sparse solver would be needed at scale.
- The PCA baseline sees the complete data (an advantage); iterative-SVD imputation gives point predictions only.
- The variance-share threshold (1%) used to count active factors is a convention, not a test.

## Possible Extensions

- Sparsity-inducing local shrinkage (horseshoe / spike-and-slab) instead of factor-wise ARD.
- Non-Gaussian likelihoods (Poisson / Negative Binomial) for count-based biological observations.
- Multi-view data integration with shared and view-specific factor representations.
- Simulation-based calibration (SBC) of the variational posterior.
- Amortised variational inference encoders for scaling to massive sample sizes.

## References

- Tipping, M. E., & Bishop, C. M. (1999). Probabilistic principal component analysis. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 61(3), 611–622. https://doi.org/10.1111/1467-9868.00196
- Bishop, C. M. (1999). Variational principal components. In *Proceedings of the Ninth International Conference on Artificial Neural Networks (ICANN’99)*, Vol. 1, pp. 509–514. IEE.

- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877. https://doi.org/10.1080/01621459.2017.1285773

- Bhattacharya, A., & Dunson, D. B. (2011). Sparse Bayesian infinite factor models. *Biometrika*, 98(2), 291–306. https://doi.org/10.1093/biomet/asr013

- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. https://doi.org/10.1198/016214506000001437

- Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. In *Proceedings of the 35th International Conference on Machine Learning (ICML)*, *Proceedings of Machine Learning Research*, 80, 2796–2804. PMLR.

## Author

Developed by **Maghsoud Fadakar**  
Independent implementation exploring variational inference and spatial priors for probabilistic latent factor models.

## License

MIT License - see [LICENSE](LICENSE) for details.