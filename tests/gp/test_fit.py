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
            input_dim=input_dim,
            n_data=n_data,
            mean_func=pm.gp.mean.Zero(),
            cov_func=pm.gp.cov.ExpQuad(input_dim=input_dim, ls=1.0),
            sigma=0.1,
            z_init=z_init,
        )

    optimizer = optax.adam(learning_rate=1e-2)
    final_params, history = fit_jax(
        svgp,
        X,
        y,
        optimizer=optimizer,
        n_steps=200,
        batch_size=32,
        model=model,
        seed=0,
    )

    history = np.asarray(history)
    assert history.shape == (200,)
    assert np.isfinite(history).all()
    # First 10 steps avg vs last 10 steps avg — loss should drop substantively.
    assert history[-10:].mean() < history[:10].mean() - 1.0

    # Returned params dict has expected keys and finite values.
    assert set(final_params).issuperset({"z", "variational_mean", "vrc"})
    for name, val in final_params.items():
        val = np.asarray(val)
        assert np.isfinite(val).all(), f"non-finite values in {name}"


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
            input_dim=input_dim,
            n_data=n_data,
            mean_func=pm.gp.mean.Zero(),
            cov_func=pm.gp.cov.ExpQuad(input_dim=input_dim, ls=1.0),
            sigma=0.1,
            z_init=z_init,
        )

    var_names = [v.name for v in model.continuous_value_vars]
    # Perturb the default initial point to a recognizable custom value. Don't zero
    # out vrc — that's a singular Cholesky factor and the loss is -inf there.
    init_pt = model.initial_point()
    custom_init = tuple(jnp.asarray(init_pt[name]) + 0.1 for name in var_names)

    # n_steps=0 isn't a thing with scan (length must be > 0); use 1 step with very
    # small lr so the result is essentially the init.
    optimizer = optax.sgd(learning_rate=1e-12)
    final_params, _ = fit_jax(
        svgp,
        X,
        y,
        optimizer=optimizer,
        n_steps=1,
        batch_size=10,
        model=model,
        init_params=custom_init,
        seed=0,
    )

    for name, init in zip(var_names, custom_init):
        np.testing.assert_allclose(np.asarray(final_params[name]), np.asarray(init), atol=1e-6)


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
                input_dim=input_dim,
                n_data=n_data,
                mean_func=pm.gp.mean.Zero(),
                cov_func=pm.gp.cov.ExpQuad(input_dim, ls=1.0),
                sigma=0.1,
                z_init=z_init,
            )

        opt = mlx_opt.Adam(learning_rate=1e-3)
        final, hist = fit_mlx(svgp, X, y, optimizer=opt, n_steps=200, batch_size=32, model=model)
    finally:
        pytensor.config.floatX = prev_floatX

    assert hist.shape == (200,)
    assert np.isfinite(hist).all()
    assert hist[-10:].mean() < hist[:10].mean() - 1.0
    assert set(final).issuperset({"z", "variational_mean", "vrc"})
