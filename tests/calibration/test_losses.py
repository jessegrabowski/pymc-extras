"""Tests for closed-form KL helpers used as calibration regularizers.

Each KL is checked for:
  - Zero when both distributions are identical.
  - Non-negative on a perturbed distinct pair.
  - Numerical agreement with an MC estimate based on the corresponding
    pytensor-distributions logpdf for a reasonable parameter pair.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytensor_distributions.beta as pytd_beta
import pytensor_distributions.uniform as pytd_uniform
import pytest

from pymc_models.calibration.losses import (
    kl_beta,
    kl_exponential,
    kl_gamma,
    kl_halfnormal,
    kl_lognormal,
    kl_normal,
    moment_distance,
)

RNG = np.random.default_rng(0)


def _eval(expr) -> float:
    return float(pytensor.function([], pt.as_tensor_variable(expr))())


def _mc_kl(family, params_p, params_q, n=200_000, seed=0):
    """MC estimate of KL(p || q) using `family.rvs` and `family.logpdf`."""
    rng = np.random.default_rng(seed)
    samples = family.rvs(*params_p, size=n, random_state=rng)
    log_p = family.logpdf(samples, *params_p)
    log_q = family.logpdf(samples, *params_q)
    return float(np.mean(log_p - log_q))


def test_kl_normal_zero_when_identical():
    assert _eval(kl_normal(1.0, 2.0, 1.0, 2.0)) == pytest.approx(0.0, abs=1e-12)


def test_kl_normal_matches_scipy_mc():
    import scipy.stats as st

    val = _eval(kl_normal(0.5, 1.0, 0.0, 1.5))
    mc = _mc_kl(st.norm, (0.5, 1.0), (0.0, 1.5))
    assert val == pytest.approx(mc, rel=0.02)


def test_kl_halfnormal_zero_when_identical():
    assert _eval(kl_halfnormal(1.5, 1.5)) == pytest.approx(0.0, abs=1e-12)


def test_kl_halfnormal_matches_mc():
    import scipy.stats as st

    val = _eval(kl_halfnormal(1.0, 1.5))
    mc = _mc_kl(st.halfnorm, (0.0, 1.0), (0.0, 1.5))
    assert val == pytest.approx(mc, rel=0.02)


def test_kl_lognormal_zero_when_identical():
    assert _eval(kl_lognormal(0.0, 1.0, 0.0, 1.0)) == pytest.approx(0.0, abs=1e-12)


def test_kl_lognormal_matches_mc():
    import scipy.stats as st

    val = _eval(kl_lognormal(0.5, 0.6, 0.2, 0.8))
    # scipy's lognorm is parameterised as (s, scale=exp(mu))
    mc = _mc_kl(st.lognorm, (0.6, 0.0, np.exp(0.5)), (0.8, 0.0, np.exp(0.2)))
    assert val == pytest.approx(mc, rel=0.02)


def test_kl_gamma_zero_when_identical():
    assert _eval(kl_gamma(2.0, 1.5, 2.0, 1.5)) == pytest.approx(0.0, abs=1e-12)


def test_kl_gamma_matches_mc():
    import scipy.stats as st

    # PyMC `beta` is rate; scipy.gamma uses scale=1/rate.
    val = _eval(kl_gamma(2.0, 1.5, 3.0, 0.7))
    mc = _mc_kl(
        st.gamma,
        (2.0, 0.0, 1.0 / 1.5),
        (3.0, 0.0, 1.0 / 0.7),
    )
    assert val == pytest.approx(mc, rel=0.02)


def test_kl_beta_zero_when_identical():
    assert _eval(kl_beta(2.0, 5.0, 2.0, 5.0)) == pytest.approx(0.0, abs=1e-12)


def test_kl_beta_matches_mc():
    import scipy.stats as st

    val = _eval(kl_beta(2.0, 5.0, 3.0, 4.0))
    mc = _mc_kl(st.beta, (2.0, 5.0), (3.0, 4.0))
    assert val == pytest.approx(mc, rel=0.02)


def test_kl_exponential_zero_when_identical():
    assert _eval(kl_exponential(1.5, 1.5)) == pytest.approx(0.0, abs=1e-12)


def test_kl_exponential_matches_mc():
    import scipy.stats as st

    val = _eval(kl_exponential(1.5, 0.7))
    mc = _mc_kl(st.expon, (0.0, 1.0 / 1.5), (0.0, 1.0 / 0.7))
    assert val == pytest.approx(mc, rel=0.02)


def test_all_kls_non_negative():
    assert _eval(kl_normal(0.5, 1.0, 0.0, 1.5)) > 0
    assert _eval(kl_halfnormal(1.0, 1.5)) > 0
    assert _eval(kl_lognormal(0.5, 0.6, 0.2, 0.8)) > 0
    assert _eval(kl_gamma(2.0, 1.5, 3.0, 0.7)) > 0
    assert _eval(kl_beta(2.0, 5.0, 3.0, 4.0)) > 0
    assert _eval(kl_exponential(1.5, 0.7)) > 0


def test_moment_distance_zero_when_identical():
    assert _eval(moment_distance(pytd_beta, (2.0, 5.0), (2.0, 5.0))) == pytest.approx(
        0.0, abs=1e-12
    )


def test_moment_distance_positive_when_different():
    val = _eval(moment_distance(pytd_beta, (2.0, 5.0), (3.0, 4.0)))
    assert val > 0


def test_moment_distance_uniform_handles_support_shift():
    """Uniform's bounds are support-defining; KL would be infinite when the new
    upper bound exceeds the old. moment_distance stays finite."""
    val = _eval(moment_distance(pytd_uniform, (0.0, 1.0), (0.0, 2.0)))
    assert np.isfinite(val) and val > 0


def test_moment_distance_is_symbolic_in_inputs():
    """The function should accept TensorVariables and produce a graph (so it
    can be differentiated against lifted shareds in a calibration loss)."""
    a = pt.scalar("a")
    b = pt.scalar("b")
    expr = moment_distance(pytd_beta, (a, b), (3.0, 4.0))
    f = pytensor.function([a, b], expr)
    np.testing.assert_allclose(f(3.0, 4.0), 0.0, atol=1e-12)
    assert f(2.0, 5.0) > 0
