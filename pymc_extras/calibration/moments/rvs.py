"""Closed-form moments for individual RandomVariable nodes.

`_rv_moments(rv)` returns `(mean_expr, covariance_expr)` for a leaf RV node.
For univariate RVs the covariance is the scalar variance σ². For multivariate
RVs (MvNormal) it is the full covariance matrix.

The lambdas adapt PyMC's user-facing parameterization (which lands as
positional inputs on the underlying RV) to whatever the corresponding
`pytensor_distributions` submodule expects, and cast every input to floatX so
the (mean, var) pair has a consistent dtype regardless of how PyMC stored
individual constants.
"""

import pytensor
import pytensor.tensor as pt
import pytensor_distributions.beta as _pytd_beta
import pytensor_distributions.exponential as _pytd_exp
import pytensor_distributions.gamma as _pytd_gamma
import pytensor_distributions.halfcauchy as _pytd_halfcauchy
import pytensor_distributions.halfnormal as _pytd_halfnormal
import pytensor_distributions.inversegamma as _pytd_invgamma
import pytensor_distributions.kumaraswamy as _pytd_kumaraswamy
import pytensor_distributions.lognormal as _pytd_lognormal
import pytensor_distributions.normal as _pytd_normal
import pytensor_distributions.uniform as _pytd_uniform

from pymc.distributions.continuous import HalfCauchyRV as PymcHalfCauchyRV
from pymc.distributions.continuous import KumaraswamyRV as PymcKumaraswamyRV
from pytensor.graph.basic import Variable
from pytensor.tensor.random.basic import (
    BetaRV,
    ExponentialRV,
    GammaRV,
    HalfNormalRV,
    InvGammaRV,
    LogNormalRV,
    MvNormalRV,
    NormalRV,
    UniformRV,
)

# Per-RV closed-form (mean, var). `var` is scalar variance for univariate RVs.
# Heavy-tailed distributions whose moments don't exist (HalfCauchy) return
# `pt.inf`; downstream computations propagate that honestly so the user sees
# an infinite variance rather than a wrong number.
_UNIVARIATE = {
    NormalRV: lambda p: (_pytd_normal.mean(*p), _pytd_normal.var(*p)),
    HalfNormalRV: lambda p: (_pytd_halfnormal.mean(p[1]), _pytd_halfnormal.var(p[1])),
    LogNormalRV: lambda p: (_pytd_lognormal.mean(*p), _pytd_lognormal.var(*p)),
    GammaRV: lambda p: (_pytd_gamma.mean(p[0], 1 / p[1]), _pytd_gamma.var(p[0], 1 / p[1])),
    InvGammaRV: lambda p: (_pytd_invgamma.mean(*p), _pytd_invgamma.var(*p)),
    BetaRV: lambda p: (_pytd_beta.mean(*p), _pytd_beta.var(*p)),
    ExponentialRV: lambda p: (_pytd_exp.mean(1 / p[0]), _pytd_exp.var(1 / p[0])),
    UniformRV: lambda p: (_pytd_uniform.mean(*p), _pytd_uniform.var(*p)),
    # PyMC SymbolicRandomVariables: distinct from the bare pytensor classes.
    PymcHalfCauchyRV: lambda p: (_pytd_halfcauchy.mean(p[0]), _pytd_halfcauchy.var(p[0])),
    PymcKumaraswamyRV: lambda p: (_pytd_kumaraswamy.mean(*p), _pytd_kumaraswamy.var(*p)),
}


def _rv_moments(rv: Variable) -> tuple[Variable, Variable]:
    """(mean, covariance) of an RV. Covariance is the scalar variance for
    univariate RVs and the full Σ matrix for multivariate ones.

    For univariate RVs, broadcasts mean and var to the RV's actual output
    shape — PyMC routinely stores scalar params and adds size= to broadcast
    (e.g. `pm.Normal("β", mu=0, sigma=10, dims="coefs")` with `coefs` of
    size 2 stores `(0, 10)` and produces a shape-(2,) RV). Without this
    broadcast, downstream Subtensor on the mean would index out of bounds.
    """
    op_class = type(rv.owner.op)
    floatx = pytensor.config.floatX
    params = tuple(pt.cast(p, floatx) for p in rv.owner.inputs[2:])

    if op_class in _UNIVARIATE:
        mean, var = _UNIVARIATE[op_class](params)
        if rv.ndim > 0:
            # Use additive broadcasting (`+ zeros_like(rv)`) rather than
            # `broadcast_to`/`SpecifyShape` — those introduce shape assertions
            # that can clash with downstream `SpecifyShape` ops on the same
            # subgraph. Plain Add broadcasts via Elemwise, no constraint.
            zeros = pt.zeros_like(rv)
            mean = mean + zeros
            var = var + zeros
        return mean, var

    if op_class is MvNormalRV:
        mu, cov = params
        return mu, cov

    raise NotImplementedError(
        f"No closed-form moments registered for RandomVariable {op_class.__name__}."
    )
