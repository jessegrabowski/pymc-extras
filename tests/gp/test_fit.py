import jax.numpy as jnp
import numpy as np
import optax
import pymc as pm
import pytest

from pymc_extras.gp import SVGP, WhitenedSVGP, fit_jax


@pytest.mark.parametrize("cls", [SVGP, WhitenedSVGP])
def test_fit_jax_reduces_loss_on_regression(cls):
    rng = np.random.default_rng(0)
    n_data = 200
    n_inducing = 10
    input_dim = 1
    X = rng.uniform(-3, 3, size=(n_data, input_dim))
    y_true = np.sin(X).sum(axis=1, keepdims=True)
    y = y_true + 0.1 * rng.normal(size=(n_data, 1))

    z_init = np.linspace(-3, 3, n_inducing).reshape(-1, input_dim)

    with pm.Model() as model:
        svgp = cls(
            mean_func=pm.gp.mean.Zero(),
            cov_func=pm.gp.cov.ExpQuad(input_dim=input_dim, ls=1.0),
            sigma=0.1,
            z_init=z_init,
        )

    optimizer = optax.adam(learning_rate=1e-2)
    idata = fit_jax(
        svgp,
        X,
        y,
        optimizer=optimizer,
        n_steps=200,
        batch_size=32,
        model=model,
        seed=0,
        progress=False,
    )

    history = np.asarray(idata.fit.loss_history)
    assert history.shape == (200,)
    assert np.isfinite(history).all()
    assert history[-10:].mean() < history[:10].mean() - 1.0

    assert "inducing_points" in idata.fit.data_vars
    assert "variational_mean" in idata.fit.data_vars
    assert "variational_cholesky" in idata.fit.data_vars


def test_fit_jax_respects_init_params():
    """Passing init_params should override model.initial_point()."""
    rng = np.random.default_rng(1)
    n_data = 50
    n_inducing = 5
    input_dim = 1
    X = rng.uniform(-2, 2, size=(n_data, input_dim))
    y = np.sin(X).reshape(-1, 1)
    z_init = np.linspace(-2, 2, n_inducing).reshape(-1, input_dim)

    with pm.Model() as model:
        svgp = SVGP(
            mean_func=pm.gp.mean.Zero(),
            cov_func=pm.gp.cov.ExpQuad(input_dim=input_dim, ls=1.0),
            sigma=0.1,
            z_init=z_init,
        )

    var_names = [v.name for v in model.continuous_value_vars]
    init_pt = model.initial_point()
    custom_init = tuple(jnp.asarray(init_pt[name]) + 0.1 for name in var_names)

    optimizer = optax.sgd(learning_rate=1e-12)
    idata = fit_jax(
        svgp,
        X,
        y,
        optimizer=optimizer,
        n_steps=1,
        batch_size=10,
        model=model,
        init_params=custom_init,
        seed=0,
        progress=False,
    )

    # After 1 step at lr=1e-12, params should be ~unchanged from custom_init.
    # SVGP internals are in idata.fit; user params would be in idata.posterior.
    pred_point = svgp.prediction_point(idata)
    for name, init in zip(var_names, custom_init):
        np.testing.assert_allclose(
            np.asarray(pred_point[name]).squeeze(), np.asarray(init).squeeze(), atol=1e-5
        )


def test_fit_jax_returns_idata_with_hyperparams():
    """fit_jax should return constrained hyperparams (sigma, etc.) in posterior."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(200, 1))
    y = np.sin(X) + 0.1 * rng.normal(size=(200, 1))

    with pm.Model() as model:
        sigma = pm.Exponential("sigma", scale=1.0)
        svgp = WhitenedSVGP(
            mean_func=pm.gp.mean.Zero(),
            cov_func=pm.gp.cov.ExpQuad(1, ls=1.0),
            sigma=sigma,
            z_init=np.linspace(-3, 3, 8).reshape(-1, 1),
        )

    idata = fit_jax(
        svgp,
        X,
        y,
        optimizer=optax.adam(1e-2),
        n_steps=200,
        batch_size=32,
        model=model,
        progress=False,
    )

    assert "sigma" in idata.posterior.data_vars
    assert "sigma_log__" in idata.unconstrained_posterior.data_vars
    sigma_val = float(idata.posterior["sigma"].squeeze())
    assert sigma_val > 0  # constrained space is positive
    assert sigma_val != 1.0  # should have moved from the init


def test_fit_mlx_reduces_loss_on_regression():
    """Smoke test: fit_mlx requires floatX='float32' (MLX GPU is float32-only)."""
    pytest.importorskip("mlx")
    pytest.importorskip("mlx.optimizers")
    import mlx.optimizers as mlx_opt
    import pytensor

    from pymc_extras.gp import fit_mlx

    rng = np.random.default_rng(0)
    n_data = 200
    n_inducing = 8
    input_dim = 1
    X = rng.uniform(-3, 3, size=(n_data, input_dim)).astype("float32")
    y = (np.sin(X) + 0.1 * rng.normal(size=(n_data, 1))).astype("float32")
    z_init = np.linspace(-3, 3, n_inducing, dtype="float32").reshape(-1, input_dim)

    prev_floatX = pytensor.config.floatX
    pytensor.config.floatX = "float32"
    try:
        with pm.Model() as model:
            svgp = WhitenedSVGP(
                mean_func=pm.gp.mean.Zero(),
                cov_func=pm.gp.cov.ExpQuad(input_dim, ls=1.0),
                sigma=0.1,
                z_init=z_init,
            )

        opt = mlx_opt.Adam(learning_rate=1e-3)
        idata = fit_mlx(
            svgp,
            X,
            y,
            optimizer=opt,
            n_steps=200,
            batch_size=32,
            model=model,
            progress=False,
        )
    finally:
        pytensor.config.floatX = prev_floatX

    history = np.asarray(idata.fit.loss_history)
    assert history.shape == (200,)
    assert np.isfinite(history).all()
    assert history[-10:].mean() < history[:10].mean() - 1.0
    assert "inducing_points" in idata.fit.data_vars
    assert "variational_mean" in idata.fit.data_vars
