"""Symbolic moment propagation.

Walks back from a graph node, collecting upstream priors and constructing a
linear-in-priors decomposition of the node's distribution. Anything that
breaks linearity (sigmoid, exp, product of two priors, …) raises
`NotImplementedError` rather than producing a wrong answer.

Public API:
    Moments                       — linear decomposition: (mean, coeffs)
    propagate_moments(var)        — recursive walker; returns Moments
    prior_predictive_moments(rv)  — per-likelihood (mean, variance) of an output

Op handlers live in `ops.py` (Add/Sub/Mul/Dot/DimShuffle/...). Per-likelihood
handlers live in `likelihoods.py` (Normal, Bernoulli, Poisson).
"""

from pymc_models.calibration.moments import likelihoods as _likelihoods  # noqa: F401

# Importing `ops` and `likelihoods` triggers handler registration on the singledispatchers.
from pymc_models.calibration.moments import ops as _ops  # noqa: F401
from pymc_models.calibration.moments.core import (
    Moments,
    propagate_moments,
    register_moments,
    register_scalar_moments,
)
from pymc_models.calibration.moments.likelihoods import (
    prior_predictive_moments,
    register_likelihood,
)

__all__ = [
    "Moments",
    "prior_predictive_moments",
    "propagate_moments",
    "register_likelihood",
    "register_moments",
    "register_scalar_moments",
]
