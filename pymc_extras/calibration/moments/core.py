"""Symbolic moment propagation through a deterministic graph of priors.

A `Moments` value records two things, summed at the end to give Var[X]:

  - `coeffs[θ_i]`: linear coefficient on prior θ_i, contributing
                    `c_iᵀ · Cov[θ_i] · c_i` to Var[X]. Per-prior because
                    shared linear paths (`c₁·θ + c₂·θ`) need coefficient
                    summation, and pytensor doesn't normalize linear forms.
  - `extra_var`:    additional variance from paths that aren't linear-in-priors
                    (e.g. after `exp(η)` produces LogNormal moments). Already
                    a numeric variance — no per-prior decomposition.

Both are summed in `total_variance()`. There's no hierarchy; "extra" just
means "variance the coefficient machinery couldn't represent."

Op handlers are registered with `@register_moments(Op)` (or
`register_scalar_moments` for `Elemwise` scalar ops). Linear ops (Add,
Mul-by-constant, Dot, …) preserve the coefficient decomposition. Non-linear
ops (Exp, Sigmoid, …) collapse to `coeffs={}` and populate `extra_var`.

Coefficient layout convention: for a prior `θ` of `prior.ndim` axes, the
coefficient `coeffs[θ]` is shape `value_shape + prior.shape` — value axes
leading, prior axes trailing. Every handler assumes and preserves this.
`extra_var` (when present) has the value's shape, no prior axes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import singledispatch

import pytensor.tensor as pt

from pymc.distributions.distribution import SymbolicRandomVariable
from pymc_models.calibration.moments.rvs import _rv_moments
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.basic import NormalRV
from pytensor.tensor.random.op import RandomVariable

# Both base `RandomVariable` (pytensor.tensor.random.op) and PyMC's
# `SymbolicRandomVariable` (a graph-rewritten OpFromGraph that wraps an inner
# RV graph — used for HalfCauchy, Kumaraswamy, WeibullBeta, …) are leaves
# from the propagator's point of view.
_RV_OP_TYPES = (RandomVariable, SymbolicRandomVariable)

__all__ = [
    "Moments",
    "propagate_moments",
    "register_moments",
    "register_scalar_moments",
]


@dataclass(frozen=True)
class Moments:
    """Decomposition of a graph node's mean and variance.

    Attributes
    ----------
    mean : Variable
        E[X].
    coeffs : Mapping[Variable, Variable]
        ``θ_rv → ∂X/∂θ`` for each upstream prior contributing to X linearly.
        For scalar priors the coefficient is a scalar; for multivariate priors
        it is shape-compatible with the prior so that ``c · Cov[θ] · cᵀ`` is
        well-defined.
    extra_var : Variable | None
        Additional variance contribution from non-linear paths (e.g. the
        LogNormal variance after ``exp(η)``). Already a numeric variance,
        not a coefficient. ``None`` is shorthand for zero.
    """

    mean: Variable
    coeffs: Mapping[Variable, Variable]
    extra_var: Variable | None = None

    def total_variance(self) -> Variable:
        """Σ_i c_i · Cov[θ_i] · c_iᵀ  +  extra_var.
        Scalar priors collapse the quadratic form to ``c² · σ²``."""
        contribs: list[Variable] = []
        for prior, c in self.coeffs.items():
            _, prior_cov = _rv_moments(prior)
            contribs.append(_quadratic_form(c, prior_cov, prior))
        if self.extra_var is not None:
            contribs.append(self.extra_var)
        if not contribs:
            return pt.zeros_like(self.mean)
        total = contribs[0]
        for c in contribs[1:]:
            total = total + c
        return total


def _quadratic_form(coeff: Variable, prior_cov: Variable, prior: Variable) -> Variable:
    """Compute the per-output-element variance contribution of one prior.

    Three regimes:
      - scalar prior:        prior_cov is scalar σ², coeff is scalar c
                             → c² · σ²
      - vector prior, diagonal cov stored as a vector of variances:
                             prior_cov is shape (k,), coeff is (..., k)
                             → Σ_j coeff[..., j]² · prior_cov[j]
      - vector prior, full cov:
                             prior_cov is (k, k), coeff is (..., k)
                             → diag(coeff · prior_cov · coeffᵀ)
    """
    if prior.ndim == 0:
        return coeff**2 * prior_cov
    if prior_cov.ndim == prior.ndim:
        # Diagonal cov stored as a vector of per-slot variances.
        return (coeff**2 * prior_cov).sum(axis=-1)
    # Full covariance matrix.
    return (coeff @ prior_cov * coeff).sum(axis=-1)


@singledispatch
def _moments_op(op, node: Apply, child_moments: list[Moments]) -> Moments:
    raise NotImplementedError(
        f"propagate_moments: no handler registered for Op {type(op).__name__}. "
        f"Add one in pymc_models.calibration.moments.ops, or rewrite the model "
        f"to avoid this op in the prior subgraph."
    )


@singledispatch
def _moments_scalar_op(scalar_op, node: Apply, child_moments: list[Moments]) -> Moments:
    raise NotImplementedError(
        f"propagate_moments: no handler registered for Elemwise(scalar_op="
        f"{type(scalar_op).__name__}). Add one in pymc_models.calibration.moments.ops, "
        f"or rewrite the model to avoid this op in the prior subgraph."
    )


@_moments_op.register(Elemwise)
def _elemwise_double_dispatch(op: Elemwise, node: Apply, child_moments: list[Moments]) -> Moments:
    return _moments_scalar_op(op.scalar_op, node, child_moments)


def register_moments(op_type):
    """Register a handler for a non-Elemwise Op."""

    def decorator(fn):
        _moments_op.register(op_type)(fn)
        return fn

    return decorator


def register_scalar_moments(scalar_op_type):
    """Register a handler for an Elemwise scalar op (Add, Mul, etc.)."""

    def decorator(fn):
        _moments_scalar_op.register(scalar_op_type)(fn)
        return fn

    return decorator


def propagate_moments(var: Variable, _cache: dict | None = None) -> Moments:
    """Walk back through the deterministic graph from `var`, returning a
    linear-in-priors `Moments` decomposition.

    Memoizes by `id(var)` within a single call: the same node visited via
    multiple paths produces a single shared `Moments` object. The cache
    parameter is exposed only so callers can reuse propagation across multiple
    output nodes within a single graph build (e.g. computing both E[y] and
    Var[y] for several outputs of the same model). It is *not* safe to reuse
    a cache across calls that may raise — partial state survives.
    """
    if _cache is None:
        _cache = {}
    cache_key = id(var)
    if cache_key in _cache:
        return _cache[cache_key]
    result = _propagate_uncached(var, _cache)
    _cache[cache_key] = result
    return result


def _propagate_uncached(var: Variable, cache: dict) -> Moments:
    if isinstance(var, Constant) or var.owner is None:
        # Constants and disowned variables (shared, placeholder, input) are
        # deterministic from the propagator's perspective.
        return Moments(mean=var, coeffs={})

    if isinstance(var.owner.op, _RV_OP_TYPES):
        return _rv_to_moments(var)

    child_moments = [propagate_moments(inp, cache) for inp in var.owner.inputs]
    return _moments_op(var.owner.op, var.owner, child_moments)


def _rv_to_moments(rv: Variable) -> Moments:
    param_moments = [propagate_moments(p) for p in rv.owner.inputs[2:]]
    has_random_hyper = any(pm.coeffs or pm.extra_var is not None for pm in param_moments)

    if has_random_hyper:
        # Some distributions admit closed-form marginal moments via the law
        # of iterated expectation when their parameters are themselves
        # priors. Normal is the canonical case (Eight Schools, hierarchical
        # regression). Other distributions (Beta, Gamma with both random,
        # …) generally don't, so we refuse.
        if isinstance(rv.owner.op, NormalRV):
            return _hierarchical_normal_moments(rv, param_moments)
        raise NotImplementedError(
            f"propagate_moments: prior {rv.name or '<unnamed>'} of type "
            f"{type(rv.owner.op).__name__} has hyperparameter(s) that depend "
            f"on other priors. Iterated-expectation moments are only "
            f"implemented for NormalRV; other families would need their own "
            f"closed-form derivation."
        )

    mean, _ = _rv_moments(rv)
    dtype = mean.type.dtype
    if rv.ndim == 0:
        coeff = pt.cast(pt.as_tensor_variable(1.0), dtype)
    elif rv.ndim == 1:
        coeff = pt.eye(rv.shape[0], dtype=dtype)
    else:
        raise NotImplementedError(
            f"propagate_moments: prior {rv.name or '<unnamed>'} has ndim "
            f"{rv.ndim}; only scalar and vector priors are supported."
        )
    return Moments(mean=mean, coeffs={rv: coeff})


def _hierarchical_normal_moments(rv: Variable, param_moments) -> Moments:
    """Marginal moments of `X ~ Normal(μ, σ)` when μ and/or σ are themselves
    priors. Via the law of iterated expectation:

        E[X]   = E[μ]
        Var[X] = Var[μ]  +  E[σ²]
               = Var[μ]  +  Var[σ]  +  E[σ]²

    Decomposed as `X = μ + σ · ε` (reparameterization, ε ⊥ μ, σ): the linear
    `μ`-part inherits μ_param's `coeffs` and `extra_var`; the `σ·ε` part is
    a fresh Normal of variance E[σ²], stored as additional `extra_var`.
    Two `X_i, X_j` sharing the same μ correctly correlate through the
    inherited `coeffs[μ]`, while their `σε` legs are independent.
    """
    mu_m, sigma_m = param_moments
    e_sigma_sq = sigma_m.total_variance() + sigma_m.mean**2

    new_extra = e_sigma_sq if mu_m.extra_var is None else mu_m.extra_var + e_sigma_sq

    # Broadcast to the RV's actual shape (PyMC may have used `size=` to
    # broadcast scalar params to a vector RV).
    new_mean = mu_m.mean
    if rv.ndim > 0:
        zeros = pt.zeros_like(rv)
        new_mean = new_mean + zeros
        new_extra = new_extra + zeros
        new_coeffs = {p: c + zeros for p, c in mu_m.coeffs.items()}
    else:
        new_coeffs = dict(mu_m.coeffs)

    return Moments(mean=new_mean, coeffs=new_coeffs, extra_var=new_extra)
