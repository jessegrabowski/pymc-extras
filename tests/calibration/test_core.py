"""End-to-end calibration on a toy linear-regression model.

Setup:
    β ~ Normal(μ_β, σ_β)
    σ_y is fixed
    y = β · x + N(0, σ_y)

Closed-form prior-predictive variance:
    Var[y] = x² · σ_β² + σ_y²

Calibration objective:
    minimise   (Var[y] - target_var)²  +  kl_weight · KL(N(0, 1) || N(μ_β, σ_β))

The first term pulls the prior toward implying the requested output variance;
the KL pulls the new prior toward the user's original prior. The "smallest
adjustment that gets a reasonable variance."
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc_models.calibration import calibrate_priors, prior_predictive_moments


def _gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    """KL(N(mu_p, sigma_p) || N(mu_q, sigma_q)). Closed form."""
    return pt.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2) - 0.5


def test_calibrate_variance_with_kl_regularizer():
    x_val = 2.0
    sigma_y = 0.5
    target_var = 4.0
    kl_weight = 0.1

    with pm.Model() as model:
        pm.Normal("beta", mu=0.0, sigma=1.0)

    def loss(params):
        mu_beta = params["beta.mu"]
        sigma_beta = params["beta.sigma"]
        var_y = x_val**2 * sigma_beta**2 + sigma_y**2
        var_match = (var_y - target_var) ** 2
        kl = _gaussian_kl(0.0, 1.0, mu_beta, sigma_beta)
        return var_match + kl_weight * kl

    result = calibrate_priors(
        model,
        loss=loss,
        target_priors=["beta"],
        n_steps=2000,
        learning_rate=0.02,
        tol=1e-9,
    )

    final_sigma = float(result.values["beta.sigma"])
    final_mu = float(result.values["beta.mu"])
    final_var_y = x_val**2 * final_sigma**2 + sigma_y**2

    np.testing.assert_allclose(final_var_y, target_var, rtol=0.05)
    assert final_sigma > 0.0
    np.testing.assert_allclose(final_mu, 0.0, atol=0.1)
    assert result.history[-1] < result.history[0]


def test_calibrate_returns_initial_value_when_already_optimal():
    """If the original prior already yields the target variance, calibration
    should leave it nearly unchanged thanks to the KL pull."""
    x_val = 1.0
    sigma_y = 0.5
    target_var = 1.0**2 * 1.0**2 + sigma_y**2  # exactly Var[y] under Normal(0, 1)

    with pm.Model() as model:
        pm.Normal("beta", mu=0.0, sigma=1.0)

    def loss(params):
        mu_beta = params["beta.mu"]
        sigma_beta = params["beta.sigma"]
        var_y = x_val**2 * sigma_beta**2 + sigma_y**2
        var_match = (var_y - target_var) ** 2
        kl = _gaussian_kl(0.0, 1.0, mu_beta, sigma_beta)
        return var_match + 1.0 * kl

    result = calibrate_priors(
        model,
        loss=loss,
        target_priors=["beta"],
        n_steps=500,
        learning_rate=0.02,
    )

    np.testing.assert_allclose(float(result.values["beta.mu"]), 0.0, atol=0.05)
    np.testing.assert_allclose(float(result.values["beta.sigma"]), 1.0, atol=0.05)


def test_calibrate_variance_with_auto_moments():
    """Same calibration objective, but `Var[y]` is derived automatically from
    the model graph via `gaussian_glm_moments`. The user no longer has to write
    out the variance decomposition by hand."""
    x_val = 2.0
    sigma_y = 0.5
    target_var = 4.0
    kl_weight = 0.1

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=beta * x_val, sigma=sigma_y, observed=np.zeros(1))

    _, var_y = prior_predictive_moments(model.named_vars["y"])
    var_y = var_y.squeeze()  # observed y has shape (1,); collapse to scalar.

    def loss(params):
        mu_beta = params["beta.mu"]
        sigma_beta = params["beta.sigma"]
        var_match = (var_y - target_var) ** 2
        kl = _gaussian_kl(0.0, 1.0, mu_beta, sigma_beta)
        return var_match + kl_weight * kl

    result = calibrate_priors(
        model,
        loss=loss,
        target_priors=["beta"],
        n_steps=2000,
        learning_rate=0.02,
        tol=1e-9,
    )

    final_sigma = float(result.values["beta.sigma"])
    final_var_y = x_val**2 * final_sigma**2 + sigma_y**2
    np.testing.assert_allclose(final_var_y, target_var, rtol=0.05)
    assert final_sigma > 0


def test_calibrate_priors_no_targets_raises():
    with pm.Model() as model:
        pm.Normal("beta", mu=0.0, sigma=1.0)

    def loss(params):
        return pt.constant(0.0)

    import pytest

    with pytest.raises(ValueError, match="No hyperparameters"):
        calibrate_priors(model, loss=loss, target_priors=[], n_steps=10)


def test_calibrate_priors_non_scalar_loss_raises():
    with pm.Model() as model:
        pm.Normal("beta", mu=0.0, sigma=1.0)

    def loss(params):
        return pt.stack([params["beta.mu"], params["beta.sigma"]])

    import pytest

    with pytest.raises(ValueError, match="scalar"):
        calibrate_priors(model, loss=loss, target_priors=["beta"], n_steps=10)
