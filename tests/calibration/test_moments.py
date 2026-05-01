"""Tests for symbolic moment propagation."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pymc_models.calibration.moments import (
    prior_predictive_moments,
    propagate_moments,
)


def _eval(expr):
    """Compile and evaluate a constant graph; return scalar float or array."""
    out = pytensor.function([], pt.as_tensor_variable(expr))()
    if np.ndim(out) == 0:
        return float(out)
    return np.asarray(out)


# ---------- propagate_moments core cases ----------


def test_constant_has_zero_variance():
    c = pt.constant(2.5)
    m = propagate_moments(c)
    np.testing.assert_allclose(_eval(m.mean), 2.5, atol=0)
    np.testing.assert_allclose(_eval(m.total_variance()), 0.0, atol=0)


def test_single_normal_prior():
    with pm.Model():
        beta = pm.Normal("beta", mu=0.5, sigma=2.0)
    m = propagate_moments(beta)
    np.testing.assert_allclose(_eval(m.mean), 0.5, rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 4.0, rtol=1e-6)


def test_constant_times_prior():
    with pm.Model():
        beta = pm.Normal("beta", mu=0.5, sigma=2.0)
    expr = beta * 3.0
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), 1.5, rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 9.0 * 4.0, rtol=1e-6)


def test_sum_of_independent_priors():
    with pm.Model():
        a = pm.Normal("a", mu=0.5, sigma=0.4)
        b = pm.Normal("b", mu=0.0, sigma=1.0)
    expr = a + 2.0 * b
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), 0.5, rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 0.4**2 + 4.0 * 1.0**2, rtol=1e-6)


def test_shared_prior_combines_coefficients():
    """θ + θ should give Var = (1+1)² σ_θ² = 4 σ_θ², not 2 σ_θ² (two independent θ)."""
    with pm.Model():
        theta = pm.Normal("theta", mu=0.0, sigma=1.0)
    expr = theta + theta
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), 0.0, atol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 4.0, rtol=1e-6)


def test_shared_prior_in_composite_path():
    """α = θ + 1, β = θ - 1. μ = α + 2·β = 3θ - 1. Var[μ] = 9 σ_θ²."""
    with pm.Model():
        theta = pm.Normal("theta", mu=0.0, sigma=1.0)
    alpha = theta + 1.0
    beta = theta - 1.0
    expr = alpha + 2.0 * beta
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), -1.0, rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 9.0, rtol=1e-6)


def test_subtraction():
    with pm.Model():
        a = pm.Normal("a", mu=2.0, sigma=0.5)
        b = pm.Normal("b", mu=1.0, sigma=0.7)
    expr = a - b
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), 1.0, rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 0.5**2 + 0.7**2, rtol=1e-6)


def test_negation():
    with pm.Model():
        a = pm.Normal("a", mu=2.0, sigma=1.5)
    expr = -a
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), -2.0, rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 2.25, rtol=1e-6)


def test_division_by_constant():
    with pm.Model():
        a = pm.Normal("a", mu=2.0, sigma=1.0)
    expr = a / 4.0
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), 0.5, rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 1.0 / 16, rtol=1e-6)


def test_halfnormal_prior():
    sigma_prior = 2.0
    with pm.Model():
        beta = pm.HalfNormal("beta", sigma=sigma_prior)
    m = propagate_moments(beta * 1.5)
    expected_mean = 1.5 * sigma_prior * np.sqrt(2 / np.pi)
    expected_var = 1.5**2 * sigma_prior**2 * (1 - 2 / np.pi)
    np.testing.assert_allclose(_eval(m.mean), expected_mean, rtol=1e-5)
    np.testing.assert_allclose(_eval(m.total_variance()), expected_var, rtol=1e-5)


# ---------- propagate_moments error cases ----------


def test_mul_of_two_priors_errors():
    with pm.Model():
        a = pm.Normal("a", mu=1.0, sigma=1.0)
        b = pm.Normal("b", mu=1.0, sigma=1.0)
    with pytest.raises(NotImplementedError, match="Mul of two prior-dependent"):
        propagate_moments(a * b)


def test_division_by_prior_errors():
    with pm.Model():
        a = pm.Normal("a", mu=1.0, sigma=1.0)
        b = pm.Normal("b", mu=2.0, sigma=0.5)
    with pytest.raises(NotImplementedError, match="Division by"):
        propagate_moments(a / b)


def test_unsupported_op_errors():
    """An op without a registered handler errors loudly."""
    with pm.Model():
        a = pm.Normal("a", mu=1.0, sigma=0.5)
    with pytest.raises(NotImplementedError, match="no handler"):
        # cosh has no handler; non-linear ops without closed-form moments
        # should refuse rather than guess.
        propagate_moments(pt.cosh(a))


def test_unsupported_prior_errors():
    with pm.Model():
        beta = pm.Wald("beta", mu=1.0, lam=2.0)
    with pytest.raises(NotImplementedError):
        propagate_moments(beta)


# ---------- Dot product / vector priors ----------


def test_expand_dims_on_scalar_prior():
    """ExpandDims should apply to mean and the (scalar) coefficient via the
    same op — coefficient broadcasts to the new shape."""
    with pm.Model():
        beta = pm.Normal("beta", mu=0.5, sigma=2.0)
    expr = pt.expand_dims(beta, axis=0)
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), [0.5], rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), [4.0], rtol=1e-6)


def test_squeeze_on_scalar_prior():
    with pm.Model():
        beta = pm.Normal("beta", mu=0.5, sigma=2.0)
    expr = pt.squeeze(pt.expand_dims(beta, axis=0), axis=0)
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), 0.5, rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 4.0, rtol=1e-6)


def test_expand_dims_on_vector_prior():
    """For a vector prior, the coefficient has a trailing prior axis that
    must be preserved when the value's leading axes get reshuffled."""
    with pm.Model():
        beta = pm.Normal("beta", mu=np.zeros(2), sigma=np.array([1.0, 0.5]), shape=(2,))
    expr = pt.expand_dims(beta, axis=0)  # (2,) -> (1, 2)
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), np.zeros((1, 2)), atol=1e-6)
    # Var per element: sigma_beta[i]**2 broadcast through the shape op
    np.testing.assert_allclose(_eval(m.total_variance()), np.array([[1.0, 0.25]]), rtol=1e-6)


def test_set_subtensor_scalar_prior():
    """`pt.set_subtensor(zeros(3)[0], θ)` puts θ at index 0."""
    with pm.Model():
        theta = pm.Normal("theta", mu=0.5, sigma=2.0)
    x = pt.zeros(3)
    expr = pt.set_subtensor(x[0], theta)
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), [0.5, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), [4.0, 0.0, 0.0], rtol=1e-6)


def test_inc_subtensor_scalar_prior():
    """`pt.inc_subtensor(x[0], θ)` adds θ at index 0; remaining elements unchanged."""
    with pm.Model():
        theta = pm.Normal("theta", mu=0.5, sigma=2.0)
    x = pt.constant(np.array([1.0, 2.0, 3.0]))
    expr = pt.inc_subtensor(x[0], theta)
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), [1.5, 2.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), [4.0, 0.0, 0.0], rtol=1e-6)


def test_inc_subtensor_combines_with_existing_prior():
    """`x = β·x_data; result = inc_subtensor(x[0], β)`. At index 0 the
    coefficient is `x_data[0] + 1`; elsewhere it's `x_data[i]`."""
    with pm.Model():
        beta = pm.Normal("beta", mu=0.0, sigma=1.5)
    x_data = pt.constant(np.array([2.0, 1.0, 0.5]))
    base = beta * x_data
    expr = pt.inc_subtensor(base[0], beta)
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), [0.0, 0.0, 0.0], atol=1e-6)
    expected = np.array([3.0**2, 1.0**2, 0.5**2]) * 1.5**2
    np.testing.assert_allclose(_eval(m.total_variance()), expected, rtol=1e-6)


def test_advanced_set_subtensor_with_array_indices():
    """Array-based indexing with non-constant indices: assign θ values at
    arbitrary positions. Pinning the op type guards against pytensor folding
    constant-index advanced subtensors into a different op."""
    from pytensor.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1

    with pm.Model():
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        b = pm.Normal("b", mu=0.0, sigma=2.0)
    x = pt.zeros(4)
    idxs = pt.lvector("idxs")
    values = pt.stack([a, b])
    expr = pt.set_subtensor(x[idxs], values)
    assert isinstance(expr.owner.op, AdvancedIncSubtensor | AdvancedIncSubtensor1)
    m = propagate_moments(expr)
    f_mean = pytensor.function([idxs], m.mean)
    f_var = pytensor.function([idxs], m.total_variance())
    np.testing.assert_allclose(f_mean(np.array([1, 3])), [0.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(f_var(np.array([1, 3])), [0.0, 1.0, 0.0, 4.0], rtol=1e-6)


def test_set_subtensor_vector_prior():
    """Vector prior placed into a slice of a longer vector."""
    with pm.Model():
        beta = pm.Normal("beta", mu=np.zeros(2), sigma=np.array([1.0, 0.5]), shape=(2,))
    x = pt.zeros(4)
    expr = pt.set_subtensor(x[1:3], beta)
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), [0.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), [0.0, 1.0, 0.25, 0.0], rtol=1e-6)


def test_dimshuffle_transpose_on_vector_prior():
    """DimShuffle that permutes value axes should also permute the coefficient's
    value axes while leaving the trailing prior axis alone."""
    with pm.Model():
        beta = pm.Normal("beta", mu=np.zeros(2), sigma=np.array([1.0, 0.5]), shape=(2,))
    # Make a 2D value by ExpandDims then transpose: (2,) -> (1, 2) -> (2, 1)
    expanded = pt.expand_dims(beta, axis=0)
    transposed = expanded.dimshuffle(1, 0)
    m = propagate_moments(transposed)
    np.testing.assert_allclose(_eval(m.mean), np.zeros((2, 1)), atol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), np.array([[1.0], [0.25]]), rtol=1e-6)


@pytest.mark.xfail(
    reason="`X @ β` for vector β lowers to Blockwise(Dot) with non-zero "
    "batch_ndim, which `_blockwise` currently rejects. Fix lives in "
    "ops.py:_blockwise — the core Dot handler already does the right thing "
    "in the no-batch case.",
    strict=True,
)
def test_dot_with_constant_matrix():
    """y = X @ β; X constant, β vector prior. Expected Var[y] = (X² · σ²)."""
    X_val = np.array([[1.0, 2.0], [0.5, -1.0]])
    with pm.Model():
        beta = pm.Normal("beta", mu=np.zeros(2), sigma=np.array([1.0, 0.5]), shape=(2,))
    expr = pt.constant(X_val) @ beta
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), np.zeros(2), atol=1e-6)
    expected = (X_val**2) @ np.array([1.0, 0.25])
    np.testing.assert_allclose(_eval(m.total_variance()), expected, rtol=1e-5)


# ---------- prior_predictive_moments dispatchers ----------


def test_prior_predictive_normal_glm():
    x_val = 2.0
    sigma_y = 0.5
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0.3, sigma=1.5)
        pm.Normal("y", mu=beta * x_val, sigma=sigma_y, observed=np.zeros(1))
    e_y, var_y = prior_predictive_moments(model.named_vars["y"])
    np.testing.assert_allclose(_eval(e_y), 0.3 * x_val, rtol=1e-6)
    expected_var = x_val**2 * 1.5**2 + sigma_y**2
    np.testing.assert_allclose(_eval(var_y), expected_var, rtol=1e-5)


def test_prior_predictive_bernoulli_with_beta_prior():
    """y ~ Bernoulli(p), p ~ Beta(α, β). E[y] = α/(α+β); Var[y] = E[y](1-E[y])."""
    a, b = 2.0, 5.0
    with pm.Model() as model:
        p = pm.Beta("p", alpha=a, beta=b)
        pm.Bernoulli("y", p=p, observed=np.array([0]))
    e_y, var_y = prior_predictive_moments(model.named_vars["y"])
    expected_e = a / (a + b)
    expected_v = expected_e * (1 - expected_e)
    np.testing.assert_allclose(_eval(e_y), expected_e, rtol=1e-5)
    np.testing.assert_allclose(_eval(var_y), expected_v, rtol=1e-5)


def test_prior_predictive_unsupported_likelihood_errors():
    with pm.Model() as model:
        pm.Wald("y", mu=1.0, lam=2.0, observed=np.ones(1))
    with pytest.raises(NotImplementedError, match="no handler for likelihood"):
        prior_predictive_moments(model.named_vars["y"])


# ---------- Memoization ----------


def test_memoization_same_object_for_repeated_node():
    """Cached results are object-identical (same `Moments`) AND share their
    underlying mean/coeff graphs — the second call must not rebuild the graph."""
    with pm.Model():
        theta = pm.Normal("theta", mu=0.0, sigma=1.0)
    cache: dict = {}
    a = propagate_moments(theta, cache)
    b = propagate_moments(theta, cache)
    assert a is b
    assert a.mean is b.mean
    assert a.coeffs is b.coeffs


# ---------- Handlers that previously had no test coverage ----------


def test_pow_constant_only_succeeds():
    """`Pow(const, const)` should constant-fold to a Moments with no priors."""
    expr = pt.constant(2.0) ** pt.constant(3.0)
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), 8.0, rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 0.0, atol=0)


def test_reshape_on_scalar_prior():
    with pm.Model():
        beta = pm.Normal("beta", mu=0.5, sigma=2.0)
    # broadcast scalar to (3,), then reshape to (1, 3).
    base = beta * pt.ones(3)
    expr = base.reshape((1, 3))
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), np.full((1, 3), 0.5), rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), np.full((1, 3), 4.0), rtol=1e-6)


def test_reshape_with_vector_prior_errors():
    with pm.Model():
        beta = pm.Normal("beta", mu=np.zeros(2), sigma=np.array([1.0, 0.5]), shape=(2,))
    expr = beta.reshape((1, 2))
    with pytest.raises(NotImplementedError, match="Reshape"):
        propagate_moments(expr)


def test_specify_shape_on_scalar_prior():
    with pm.Model():
        beta = pm.Normal("beta", mu=0.5, sigma=2.0)
    base = beta * pt.ones(3)
    expr = pt.specify_shape(base, (3,))
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), np.full(3, 0.5), rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), np.full(3, 4.0), rtol=1e-6)


def test_subtensor_on_vector_prior():
    """Indexing a vector prior at a scalar position picks out one slot."""
    with pm.Model():
        beta = pm.Normal("beta", mu=np.zeros(3), sigma=np.array([1.0, 0.5, 2.0]), shape=(3,))
    expr = beta[1]  # second component
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), 0.0, atol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), 0.25, rtol=1e-6)


def test_dot_constant_inputs_succeeds():
    """`pt.dot(M, v)` on two constants — exercises the constant branch of `_dot`."""
    M = pt.constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    v = pt.constant(np.array([1.0, 1.0]))
    expr = pt.dot(M, v)
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), [3.0, 7.0], rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), [0.0, 0.0], atol=0)


def test_make_vector_with_mismatched_priors():
    """`pt.stack([θ_a, θ_b, c])` where c has no prior dependence — c's column
    in the per-prior coefficients should be zero."""
    with pm.Model():
        a = pm.Normal("a", mu=1.0, sigma=0.5)
        b = pm.Normal("b", mu=2.0, sigma=1.5)
    expr = pt.stack([a, b, pt.constant(7.0)])
    m = propagate_moments(expr)
    np.testing.assert_allclose(_eval(m.mean), [1.0, 2.0, 7.0], rtol=1e-6)
    np.testing.assert_allclose(_eval(m.total_variance()), [0.25, 2.25, 0.0], rtol=1e-6)


# ---------- Likelihood handlers ----------


def test_prior_predictive_normal_with_prior_on_sigma():
    """The Normal likelihood handler must compute E[σ²] = Var[σ] + E[σ]²
    when σ itself is a prior, not punt to a Pow-non-linear error."""
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=2.0)
        pm.Normal("y", mu=beta, sigma=sigma, observed=np.zeros(1))
    e_y, var_y = prior_predictive_moments(model.named_vars["y"])
    # E[y] = E[β] = 0
    np.testing.assert_allclose(_eval(e_y), 0.0, atol=1e-6)
    # Var[y] = Var[beta] + E[sigma**2]; sigma ~ HalfNormal(2) => E[sigma**2] = 4
    np.testing.assert_allclose(_eval(var_y), 1.0 + 4.0, rtol=1e-5)


def test_prior_predictive_poisson():
    """y ~ Poisson(λ), λ ~ Gamma(α, β=rate).
    E[y] = α/β; Var[y] = α/β² + α/β."""
    a, b = 3.0, 2.0
    with pm.Model() as model:
        lam = pm.Gamma("lam", alpha=a, beta=b)
        pm.Poisson("y", mu=lam, observed=np.zeros(1, dtype=int))
    e_y, var_y = prior_predictive_moments(model.named_vars["y"])
    expected_mean = a / b
    expected_var = a / b**2 + a / b
    np.testing.assert_allclose(_eval(e_y), expected_mean, rtol=1e-5)
    np.testing.assert_allclose(_eval(var_y), expected_var, rtol=1e-5)


def test_exp_of_normal_prior_is_lognormal():
    """`exp(η)` for η ~ Normal: exact LogNormal moments."""
    mu = 0.5
    sigma = 1.5
    with pm.Model():
        eta = pm.Normal("eta", mu=mu, sigma=sigma)
    expr = pt.exp(eta)
    m = propagate_moments(expr)
    expected_mean = np.exp(mu + sigma**2 / 2)
    expected_var = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    np.testing.assert_allclose(_eval(m.mean), expected_mean, rtol=1e-5)
    np.testing.assert_allclose(_eval(m.total_variance()), expected_var, rtol=1e-5)


def test_exp_collapses_per_prior_decomposition():
    """After `exp`, `coeffs` is empty and the variance lives in `extra_var`.
    This means a downstream linear path that *also* touches the same prior
    won't double-count or miss the covariance — the framework just adds the
    extra_var to whatever linear contributions remain."""
    with pm.Model():
        eta = pm.Normal("eta", mu=0.0, sigma=1.0)
    m = propagate_moments(pt.exp(eta))
    assert m.coeffs == {}
    assert m.extra_var is not None


def test_sigmoid_of_normal_prior_uses_mackay():
    """E[sigmoid(η)] ≈ sigmoid(μ / sqrt(1 + π·σ²/8)) (MacKay).
    Var[sigmoid(η)] ≈ sigmoid'(μ)² · σ² (delta method)."""
    mu = 1.5
    sigma = 2.0
    with pm.Model():
        eta = pm.Normal("eta", mu=mu, sigma=sigma)
    expr = pt.sigmoid(eta)
    m = propagate_moments(expr)
    expected_mean = 1.0 / (1.0 + np.exp(-mu / np.sqrt(1 + np.pi * sigma**2 / 8)))
    np.testing.assert_allclose(_eval(m.mean), expected_mean, rtol=1e-5)
    sig_mu = 1.0 / (1.0 + np.exp(-mu))
    expected_var = (sig_mu * (1 - sig_mu)) ** 2 * sigma**2
    np.testing.assert_allclose(_eval(m.total_variance()), expected_var, rtol=1e-5)


def test_prior_predictive_poisson_log_link():
    """y ~ Poisson(exp(α + β·x)). Closed form via LogNormal moments on η."""
    with pm.Model() as model:
        a = pm.Normal("a", mu=0.0, sigma=0.5)
        b = pm.Normal("b", mu=0.0, sigma=0.3)
        eta = a + b * 1.0
        pm.Poisson("y", mu=pt.exp(eta), observed=np.zeros(1, dtype=int))
    e_y, var_y = prior_predictive_moments(model.named_vars["y"])
    # E[η] = 0, Var[η] = 0.25 + 0.09 = 0.34
    var_eta = 0.25 + 0.09
    e_lam = np.exp(0 + var_eta / 2)
    var_lam = (np.exp(var_eta) - 1) * np.exp(0 + var_eta)
    np.testing.assert_allclose(_eval(e_y), e_lam, rtol=1e-5)
    np.testing.assert_allclose(_eval(var_y), var_lam + e_lam, rtol=1e-5)


def test_prior_predictive_bernoulli_logit_link():
    """y ~ Bernoulli(sigmoid(α + β·x)). MacKay's approximation under the hood."""
    with pm.Model() as model:
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        b = pm.Normal("b", mu=0.0, sigma=0.5)
        eta = a + b * 2.0
        pm.Bernoulli("y", p=pt.sigmoid(eta), observed=np.zeros(1, dtype=int))
    e_y, var_y = prior_predictive_moments(model.named_vars["y"])
    # E[η] = 0, Var[η] = 1 + 4·0.25 = 2
    var_eta = 1.0 + 4 * 0.25
    expected_e = 1.0 / (1.0 + np.exp(0))  # sigmoid of 0 = 0.5 (μ_η was 0)
    np.testing.assert_allclose(_eval(e_y), expected_e, rtol=1e-5)
    np.testing.assert_allclose(_eval(var_y), expected_e * (1 - expected_e), rtol=1e-5)


def test_eight_schools_hierarchical_normal():
    """Canonical hierarchical Normal model:
        μ ~ Normal(0, σ_μ),  τ ~ HalfNormal(σ_τ)
        θ_i ~ Normal(μ, τ),  obs_i ~ Normal(θ_i, σ_i)

    Marginal moments via law of iterated expectation:
        E[obs_i]   = E[μ] = 0
        Var[obs_i] = Var[μ] + E[τ²] + σ_i²
    """
    J = 8
    sigma_obs = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    sigma_mu = 5.0
    sigma_tau = 5.0

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=sigma_mu)
        tau = pm.HalfNormal("tau", sigma=sigma_tau)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)
        pm.Normal("obs", mu=theta, sigma=sigma_obs, observed=np.zeros(J))

    e_y, var_y = prior_predictive_moments(model.named_vars["obs"])
    expected_var = sigma_mu**2 + sigma_tau**2 + sigma_obs**2
    np.testing.assert_allclose(_eval(e_y), np.zeros(J), atol=1e-6)
    np.testing.assert_allclose(_eval(var_y), expected_var, rtol=1e-5)


def test_eight_schools_with_halfcauchy_tau_is_infinite():
    """Same model but with HalfCauchy(τ): E[τ²] = ∞, so Var[obs] = ∞."""
    J = 8
    sigma_obs = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)
        pm.Normal("obs", mu=theta, sigma=sigma_obs, observed=np.zeros(J))

    e_y, var_y = prior_predictive_moments(model.named_vars["obs"])
    np.testing.assert_allclose(_eval(e_y), np.zeros(J), atol=1e-6)
    var_vals = _eval(var_y)
    assert np.all(np.isinf(var_vals)), f"expected all-inf, got {var_vals}"


def test_hierarchical_beta_still_refused():
    """Beta with random hyperparameters has no closed-form moments
    (E[α/(α+β)] depends on the joint hyperprior); we should error cleanly."""
    with pm.Model() as model:
        ab = pm.HalfNormal("ab", sigma=10, shape=2)
        pm.Beta("theta", alpha=ab[0], beta=ab[1], shape=3)
    with pytest.raises(NotImplementedError, match="hyperparameter"):
        propagate_moments(model.named_vars["theta"])


def test_prior_predictive_normal_constant_sigma_unchanged():
    """Sanity: the σ-handling refactor mustn't regress the constant-σ case."""
    x_val = 2.0
    sigma_y = 0.5
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0.3, sigma=1.5)
        pm.Normal("y", mu=beta * x_val, sigma=sigma_y, observed=np.zeros(1))
    e_y, var_y = prior_predictive_moments(model.named_vars["y"])
    np.testing.assert_allclose(_eval(e_y), 0.3 * x_val, rtol=1e-6)
    np.testing.assert_allclose(_eval(var_y), x_val**2 * 1.5**2 + sigma_y**2, rtol=1e-5)
