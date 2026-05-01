"""Per-likelihood prior-predictive moment dispatchers.

Given an output RV `y ~ <Likelihood>(params(θ))` whose parameters are a
deterministic function of upstream priors θ, returns symbolic `(E[y], Var[y])`
via the law of total variance:

    E[y]   = E_θ[ E[y|θ] ]
    Var[y] = Var_θ[ E[y|θ] ]  +  E_θ[ Var[y|θ] ]

Both pieces come from `propagate_moments` on the parameter graphs. Link
functions (`exp`, `sigmoid`) get their own handlers in `ops.py` that produce
the appropriate closed-form / approximate moments, so likelihood handlers
don't need to special-case them.
"""

from collections.abc import Callable

from pymc_models.calibration.moments.core import propagate_moments
from pytensor.graph.basic import Variable
from pytensor.tensor.random.basic import (
    BernoulliRV,
    BinomialRV,
    CategoricalRV,
    NormalRV,
    PoissonRV,
    StudentTRV,
)

_OutputHandler = Callable[[Variable], tuple[Variable, Variable]]
_LIKELIHOOD_HANDLERS: dict[type, _OutputHandler] = {}


def register_likelihood(rv_class):
    def decorator(fn):
        _LIKELIHOOD_HANDLERS[rv_class] = fn
        return fn

    return decorator


def prior_predictive_moments(rv: Variable) -> tuple[Variable, Variable]:
    """Symbolic (mean, variance) of the prior predictive of an output RV."""
    op_class = type(rv.owner.op)
    if op_class not in _LIKELIHOOD_HANDLERS:
        raise NotImplementedError(
            f"prior_predictive_moments: no handler for likelihood "
            f"{op_class.__name__}. Register one with @register_likelihood."
        )
    return _LIKELIHOOD_HANDLERS[op_class](rv)


@register_likelihood(NormalRV)
def _normal(rv):
    """y ~ Normal(μ, σ).
    E[y]   = E[μ]
    Var[y] = Var[μ] + E[σ²]
    """
    _, _, mu_param, sigma_param = rv.owner.inputs
    mu_moments = propagate_moments(mu_param)
    sigma_moments = propagate_moments(sigma_param)
    e_sigma_sq = sigma_moments.total_variance() + sigma_moments.mean**2
    return mu_moments.mean, mu_moments.total_variance() + e_sigma_sq


@register_likelihood(BernoulliRV)
def _bernoulli(rv):
    """y ~ Bernoulli(p). Var[y] = E[p](1 − E[p]) — Var[p] terms cancel since y² = y.

    A `sigmoid` link is handled transparently by the Sigmoid op handler in
    `ops.py`, which produces the appropriate (mean, extra_var) for `p`.
    """
    _, _, p_param = rv.owner.inputs
    p_moments = propagate_moments(p_param)
    e_p = p_moments.mean
    return e_p, e_p * (1 - e_p)


@register_likelihood(BinomialRV)
def _binomial(rv):
    """y ~ Binomial(n, p).
    E[y]   = n · E[p]
    Var[y] = n(n−1) · Var[p]  +  n · E[p](1 − E[p])
    """
    _, _, n_param, p_param = rv.owner.inputs
    p_moments = propagate_moments(p_param)
    e_p = p_moments.mean
    var_p = p_moments.total_variance()
    return n_param * e_p, n_param * (n_param - 1) * var_p + n_param * e_p * (1 - e_p)


@register_likelihood(CategoricalRV)
def _categorical(rv):
    """y ~ Categorical(p) over K classes. Treated as one-hot:
        E[y_k]   = E[p_k]
        Var[y_k] = E[p_k](1 − E[p_k])
    Returns per-class (mean, variance) vectors. The integer-label "mean" of
    a Categorical isn't a meaningful elicitation target and isn't reported.
    """
    _, _, p_param = rv.owner.inputs
    p_moments = propagate_moments(p_param)
    return p_moments.mean, p_moments.mean * (1 - p_moments.mean)


@register_likelihood(PoissonRV)
def _poisson(rv):
    """y ~ Poisson(λ). E[y] = E[λ], Var[y] = Var[λ] + E[λ].

    Log link `λ = exp(η)` is handled transparently by the Exp op handler.
    """
    _, _, lam_param = rv.owner.inputs
    lam_moments = propagate_moments(lam_param)
    return lam_moments.mean, lam_moments.total_variance() + lam_moments.mean


@register_likelihood(StudentTRV)
def _studentt(rv):
    """y ~ StudentT(ν, μ, σ).
    E[y]   = E[μ]                                 (defined for ν > 1)
    Var[y] = Var[μ] + E[σ²] · E[ν / (ν − 2)]      (defined for ν > 2)
    """
    _, _, nu_param, mu_param, sigma_param = rv.owner.inputs
    mu_moments = propagate_moments(mu_param)
    sigma_moments = propagate_moments(sigma_param)
    e_sigma_sq = sigma_moments.total_variance() + sigma_moments.mean**2
    nu_moments = propagate_moments(nu_param)
    nu_factor = nu_moments.mean / (nu_moments.mean - 2)
    return mu_moments.mean, mu_moments.total_variance() + e_sigma_sq * nu_factor
