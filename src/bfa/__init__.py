"""bfa: sparse Bayesian factor analysis with variational inference (PyTorch)."""
from .model import BayesianFactorAnalysis, grid_edges
from . import data, metrics

__all__ = ["BayesianFactorAnalysis", "grid_edges", "data", "metrics"]
__version__ = "0.1.0"
