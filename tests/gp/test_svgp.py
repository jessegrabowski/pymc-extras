import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.linalg

from pymc_extras.gp.svgp import SVGP, WhitenedSVGP

JITTER = 1e-6


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(42)
    n_inducing = 5
    n_test = 7
    input_dim = 2
    ls = 1.5

    z = rng.normal(size=(n_inducing, input_dim)).astype("float64")
    t = rng.normal(size=(n_test, input_dim)).astype("float64")
    y = rng.normal(size=(n_test, 1)).astype("float64")

    mu_q = rng.normal(size=n_inducing).astype("float64")
    raw = rng.normal(size=(n_inducing, n_inducing))
    L_q = np.tril(raw).astype("float64")
    L_q[np.diag_indices(n_inducing)] = np.abs(np.diag(L_q)) + 0.5

    Kzz = _rbf(z, z, ls) + JITTER * np.eye(n_inducing)
    Lz = scipy.linalg.cholesky(Kzz, lower=True)
    Kzt = _rbf(z, t, ls)
    Ktt = _rbf(t, t, ls)
    Ktt_diag = np.ones(n_test)

    return dict(
        n_inducing=n_inducing,
        n_test=n_test,
        input_dim=input_dim,
        ls=ls,
        z=z,
        t=t,
        y=y,
        mu_q=mu_q,
        L_q=L_q,
        Kzz=Kzz,
        Lz=Lz,
        Kzt=Kzt,
        Ktt=Ktt,
        Ktt_diag=Ktt_diag,
    )


def _rbf(a, b, ls):
    sqd = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * sqd / ls**2)


def _build_svgp(problem, cls=SVGP, sigma=0.1, n_data=100):
    kernel = pm.gp.cov.ExpQuad(input_dim=problem["input_dim"], ls=problem["ls"])
    mean_fn = pm.gp.mean.Zero()
    model = pm.Model()
    with model:
        svgp = cls(
            input_dim=problem["input_dim"],
            n_data=n_data,
            mean_func=mean_fn,
            cov_func=kernel,
            sigma=sigma,
            z_init=problem["z"],
        )
    return svgp, model


def _point(problem):
    n = problem["n_inducing"]
    return {
        "z": problem["z"],
        "variational_mean": problem["mu_q"],
        "vrc": problem["L_q"][np.tril_indices(n)],
    }


def _eval(tensor, model, point, extra_inputs=None):
    [tensor_v] = model.replace_rvs_by_values([tensor])
    inputs = list(pm.inputvars([tensor_v]))
    if extra_inputs:
        inputs = inputs + list(extra_inputs)
    f = pytensor.function(inputs, tensor_v, on_unused_input="ignore")
    kwargs = {v.name: point[v.name] for v in inputs if v.name in point}
    return f(**kwargs)


def test_kl_divergence_matches_closed_form(problem):
    Kzz, Lz = problem["Kzz"], problem["Lz"]
    L_q, mu_q = problem["L_q"], problem["mu_q"]
    S_q = L_q @ L_q.T
    diff = -mu_q
    trace = np.trace(np.linalg.solve(Kzz, S_q))
    mahal = diff @ np.linalg.solve(Kzz, diff)
    logdet_q = 2 * np.sum(np.log(np.diag(L_q)))
    logdet_p = 2 * np.sum(np.log(np.diag(Lz)))
    ref = 0.5 * (trace + mahal + logdet_p - logdet_q - problem["n_inducing"])

    svgp, model = _build_svgp(problem)
    with model:
        kl = svgp.kl_divergence()
    got = _eval(kl, model, _point(problem))

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_predict_mean_and_cov_match_closed_form(problem):
    Kzz, Lz = problem["Kzz"], problem["Lz"]
    Kzt, Ktt = problem["Kzt"], problem["Ktt"]
    L_q, mu_q = problem["L_q"], problem["mu_q"]

    Kzz_inv_Kzt = np.linalg.solve(Kzz, Kzt)
    Lz_inv_Kzt = scipy.linalg.solve_triangular(Lz, Kzt, lower=True)
    LqT_K = L_q.T @ Kzz_inv_Kzt

    ref_mean = Kzz_inv_Kzt.T @ mu_q
    ref_cov = Ktt - Lz_inv_Kzt.T @ Lz_inv_Kzt + LqT_K.T @ LqT_K

    svgp, model = _build_svgp(problem)
    t_const = pt.as_tensor_variable(problem["t"])
    with model:
        mean, cov = svgp.predict(t_const)
    got_mean = _eval(mean, model, _point(problem))
    got_cov = _eval(cov, model, _point(problem))

    np.testing.assert_allclose(got_mean, ref_mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got_cov, ref_cov, rtol=1e-10, atol=1e-12)


def test_predict_diag_matches_full_predict_diagonal(problem):
    svgp, model = _build_svgp(problem)
    t_const = pt.as_tensor_variable(problem["t"])
    with model:
        mean_full, cov_full = svgp.predict(t_const)
        mean_diag, var_diag = svgp.predict_diag(t_const)

    point = _point(problem)
    mf = _eval(mean_full, model, point)
    cf = _eval(cov_full, model, point)
    md = _eval(mean_diag, model, point)
    vd = _eval(var_diag, model, point)

    np.testing.assert_allclose(md, mf, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vd, np.diag(cf), rtol=1e-10, atol=1e-10)


def test_variational_expectation_matches_closed_form(problem):
    # Reference: -0.5 * (log(2π) + log(σ²) + ((y − m)² + v) / σ²)  per-point
    sigma = 0.3
    svgp, model = _build_svgp(problem, sigma=sigma)

    t_const = pt.as_tensor_variable(problem["t"])
    y_const = pt.as_tensor_variable(problem["y"])

    with model:
        mean, var = svgp.predict_diag(t_const)
        ve = svgp.variational_expectation(t_const, y_const)

    point = _point(problem)
    m = _eval(mean, model, point)
    v = _eval(var, model, point)
    got = _eval(ve, model, point)

    y_flat = problem["y"].squeeze(-1)
    sq_err = (y_flat - m) ** 2
    ref = -0.5 * (np.log(2 * np.pi) + np.log(sigma**2) + (sq_err + v) / sigma**2)

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_elbo_equals_scaled_varexp_minus_kl(problem):
    sigma = 0.3
    n_data = 500
    svgp, model = _build_svgp(problem, sigma=sigma, n_data=n_data)

    t_const = pt.as_tensor_variable(problem["t"])
    y_const = pt.as_tensor_variable(problem["y"])

    with model:
        ve = svgp.variational_expectation(t_const, y_const)
        kl = svgp.kl_divergence()
        elbo = svgp.elbo(t_const, y_const)

    point = _point(problem)
    ve_val = _eval(ve, model, point)
    kl_val = _eval(kl, model, point)
    elbo_val = _eval(elbo, model, point)

    ref = (n_data / problem["n_test"]) * ve_val.sum() - kl_val
    np.testing.assert_allclose(elbo_val, ref, rtol=1e-10, atol=1e-12)


# -- Whitened parameterization --------------------------------------------------
#
# In the whitened parameterization, ``q(v) = N(μ̃, L̃ L̃ᵀ)`` over
# ``v = L_z⁻¹ (u − μ_z)``. Setting μ̃ = L_z⁻¹ (μ_q − μ_z) and L̃ = L_z⁻¹ L_q
# gives an equivalent ``q(u)`` to the unwhitened SVGP with parameters (μ_q, L_q),
# and the predictive distributions agree exactly.


def _whitened_equivalents(problem):
    """μ̃, L̃ that make WhitenedSVGP match the SVGP fixture's q(u)."""
    Lz = problem["Lz"]
    mu_q = problem["mu_q"]
    L_q = problem["L_q"]
    mu_tilde = scipy.linalg.solve_triangular(Lz, mu_q, lower=True)
    L_tilde = scipy.linalg.solve_triangular(Lz, L_q, lower=True)
    return mu_tilde, L_tilde


def _whitened_point(problem):
    n = problem["n_inducing"]
    mu_tilde, L_tilde = _whitened_equivalents(problem)
    return {
        "z": problem["z"],
        "variational_mean": mu_tilde,
        "vrc": L_tilde[np.tril_indices(n)],
    }


def test_whitened_kl_matches_closed_form(problem):
    mu_tilde, L_tilde = _whitened_equivalents(problem)
    m = problem["n_inducing"]
    ref = 0.5 * (
        np.sum(L_tilde**2) + np.sum(mu_tilde**2) - 2 * np.sum(np.log(np.diag(L_tilde))) - m
    )

    svgp, model = _build_svgp(problem, cls=WhitenedSVGP)
    with model:
        kl = svgp.kl_divergence()
    got = _eval(kl, model, _whitened_point(problem))

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_whitened_predict_matches_unwhitened_with_equivalent_params(problem):
    """Whitened (μ̃, L̃) should give the same predictive distribution as
    unwhitened (μ_q, L_q) when (μ̃, L̃) are the whitened transforms of (μ_q, L_q)."""
    t_const = pt.as_tensor_variable(problem["t"])

    # Unwhitened reference
    svgp_u, model_u = _build_svgp(problem, cls=SVGP)
    with model_u:
        mean_u, cov_u = svgp_u.predict(t_const)
    mu_u = _eval(mean_u, model_u, _point(problem))
    cov_u = _eval(cov_u, model_u, _point(problem))

    # Whitened with equivalent params
    svgp_w, model_w = _build_svgp(problem, cls=WhitenedSVGP)
    with model_w:
        mean_w, cov_w = svgp_w.predict(t_const)
    mu_w = _eval(mean_w, model_w, _whitened_point(problem))
    cov_w = _eval(cov_w, model_w, _whitened_point(problem))

    np.testing.assert_allclose(mu_w, mu_u, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cov_w, cov_u, rtol=1e-10, atol=1e-10)


def test_whitened_kl_matches_unwhitened_with_equivalent_params(problem):
    """Same equivalence: KL should agree."""
    svgp_u, model_u = _build_svgp(problem, cls=SVGP)
    with model_u:
        kl_u_t = svgp_u.kl_divergence()
    kl_u = _eval(kl_u_t, model_u, _point(problem))

    svgp_w, model_w = _build_svgp(problem, cls=WhitenedSVGP)
    with model_w:
        kl_w_t = svgp_w.kl_divergence()
    kl_w = _eval(kl_w_t, model_w, _whitened_point(problem))

    np.testing.assert_allclose(kl_w, kl_u, rtol=1e-10, atol=1e-10)


def test_whitened_predict_diag_matches_full_predict_diagonal(problem):
    svgp, model = _build_svgp(problem, cls=WhitenedSVGP)
    t_const = pt.as_tensor_variable(problem["t"])
    with model:
        mean_full, cov_full = svgp.predict(t_const)
        mean_diag, var_diag = svgp.predict_diag(t_const)

    point = _whitened_point(problem)
    mf = _eval(mean_full, model, point)
    cf = _eval(cov_full, model, point)
    md = _eval(mean_diag, model, point)
    vd = _eval(var_diag, model, point)

    np.testing.assert_allclose(md, mf, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vd, np.diag(cf), rtol=1e-10, atol=1e-10)
