import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pymc_models.calibration.lifter import lift_hyperparameters
from pymc_models.calibration.targets import resolve_targets
from pytensor.graph.replace import graph_replace
from pytensor.tensor.sharedvar import SharedVariable


@pytest.fixture
def simple_model():
    with pm.Model() as model:
        pm.Normal("beta", mu=0.5, sigma=2.0)
        pm.HalfNormal("sigma", sigma=1.5)
    return model


def test_resolve_variable_form(simple_model):
    beta = simple_model.named_vars["beta"]
    targets = resolve_targets(simple_model, [beta])
    assert {t.param_name for t in targets} == {"mu", "sigma"}
    assert all(t.rv_name == "beta" for t in targets)


def test_resolve_string_form(simple_model):
    targets = resolve_targets(simple_model, ["beta"])
    assert {t.param_name for t in targets} == {"mu", "sigma"}


def test_resolve_dotted_form(simple_model):
    targets = resolve_targets(simple_model, ["beta.sigma"])
    assert len(targets) == 1
    assert targets[0].rv_name == "beta"
    assert targets[0].param_name == "sigma"
    assert targets[0].input_index == 3


def test_resolve_mixed(simple_model):
    targets = resolve_targets(simple_model, ["beta.mu", "sigma"])
    keys = sorted((t.rv_name, t.param_name) for t in targets)
    assert keys == [("beta", "mu"), ("sigma", "sigma")]


def test_resolve_unknown_rv(simple_model):
    with pytest.raises(KeyError, match="zorg"):
        resolve_targets(simple_model, ["zorg"])


def test_resolve_unknown_param(simple_model):
    with pytest.raises(KeyError, match="tau"):
        resolve_targets(simple_model, ["beta.tau"])


def test_resolve_unsupported_distribution():
    with pm.Model() as model:
        pm.Wald("w", mu=1.0, lam=2.0)
    with pytest.raises(NotImplementedError, match="WaldRV"):
        resolve_targets(model, ["w"])


def test_resolve_unnamed_variable_raises():
    with pm.Model():
        rv = pm.Normal.dist(mu=0.0, sigma=1.0)
    rv.name = None
    with pytest.raises(ValueError, match="must have a name"):
        resolve_targets(pm.Model(), [rv])


def test_lift_creates_shared(simple_model):
    lifted = lift_hyperparameters(simple_model, ["beta.sigma"])
    assert len(lifted) == 1
    h = lifted[0]
    assert isinstance(h.shared_var, SharedVariable)
    assert h.shared_var.name == "beta.sigma"
    assert h.name == "beta.sigma"
    np.testing.assert_array_equal(h.shared_var.get_value(), np.float32(2.0))
    np.testing.assert_array_equal(h.initial_value, np.float32(2.0))


def test_lift_initial_values_match_originals(simple_model):
    lifted = lift_hyperparameters(simple_model, ["beta", "sigma"])
    by_name = {h.name: h for h in lifted}
    np.testing.assert_array_equal(by_name["beta.mu"].initial_value, np.float32(0.5))
    np.testing.assert_array_equal(by_name["beta.sigma"].initial_value, np.float32(2.0))
    np.testing.assert_array_equal(by_name["sigma.sigma"].initial_value, np.float32(1.5))


def test_lift_substitution_preserves_logp(simple_model):
    """Replacing constants with shareds at their initial values should not change logp."""
    lifted = lift_hyperparameters(simple_model, ["beta", "sigma"])

    original_logp = simple_model.logp()
    replacements = {h.constant_node: h.shared_var for h in lifted}
    new_logp = graph_replace(original_logp, replacements)

    inputs = simple_model.value_vars
    f_orig = pytensor.function(inputs, original_logp)
    f_new = pytensor.function(inputs, new_logp)

    rng = np.random.default_rng(0)
    test_args = [rng.standard_normal(size=v.type.shape).astype(v.type.dtype) for v in inputs]
    np.testing.assert_allclose(f_orig(*test_args), f_new(*test_args), rtol=1e-5)


def test_lift_substitution_tracks_shared_changes(simple_model):
    """Updating the shared variable should change downstream graph evaluation."""
    lifted = lift_hyperparameters(simple_model, ["beta.sigma"])
    h = lifted[0]

    original_logp = simple_model.logp()
    new_logp = graph_replace(original_logp, {h.constant_node: h.shared_var})

    inputs = simple_model.value_vars
    f = pytensor.function(inputs, new_logp)

    rng = np.random.default_rng(0)
    test_args = [rng.standard_normal(size=v.type.shape).astype(v.type.dtype) for v in inputs]
    baseline = float(f(*test_args))

    h.shared_var.set_value(np.float32(0.5))
    perturbed = float(f(*test_args))

    assert not np.isclose(baseline, perturbed)


def test_lift_no_constant_in_subgraph_raises():
    """When the parameter is fully determined by RVs (no constant to lift), error."""
    with pm.Model() as model:
        log_sigma = pm.Normal("log_sigma", mu=0.0, sigma=1.0)
        pm.Normal("beta", mu=0.0, sigma=pt.exp(log_sigma))
    with pytest.raises(ValueError, match="no Constant found"):
        lift_hyperparameters(model, ["beta.sigma"])


def test_lift_blocks_at_random_variable():
    """Hierarchical prior: `mu` of `beta` is itself an RV. The walker stops at the
    upstream RV, finds no constants, and errors instead of reaching into the
    upstream prior's parameters."""
    with pm.Model() as model:
        pm.Normal("mu_hyper", mu=0.0, sigma=1.0)
        pm.Normal("beta", mu=model.named_vars["mu_hyper"], sigma=1.0)
    with pytest.raises(ValueError, match="no Constant found"):
        lift_hyperparameters(model, ["beta.mu"])


def test_lift_ambiguous_multiple_constants_raises():
    """Two distinct constants in the same parameter subgraph: lifter can't choose."""
    with pm.Model() as model:
        # sigma = 2.0 * 0.5 — two separate Constants feeding the same slot
        sigma = pt.constant(2.0) * pt.constant(0.5)
        pm.Normal("beta", mu=0.0, sigma=sigma)
    with pytest.raises(ValueError, match="ambiguous"):
        lift_hyperparameters(model, ["beta.sigma"])


def test_lift_walks_through_reciprocal_for_gamma_rate():
    """`pm.Gamma(beta=rate)` puts a Reciprocal between the RV and the user's value."""
    with pm.Model() as model:
        pm.Gamma("g", alpha=2.0, beta=0.5)
    lifted = lift_hyperparameters(model, ["g.beta"])
    assert len(lifted) == 1
    np.testing.assert_array_equal(lifted[0].shared_var.get_value(), np.float32(0.5))


def test_lift_tau_parameterization_is_ambiguous():
    """PyMC's `tau→sigma` translation is `tau ** -0.5 * sign(tau)`, which puts a
    literal `-0.5` into the subgraph alongside the user's value. That's two
    distinct constants → ambiguous. Use the canonical `sigma=` parameterization
    if you want to lift this hyperparameter."""
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, tau=2.0)
    with pytest.raises(ValueError, match="ambiguous"):
        lift_hyperparameters(model, ["x.sigma"])


def test_lift_gamma_rate_preserves_semantics():
    """The transformation chain (here, Reciprocal) stays in the graph after
    substitution, so the lifted shared keeps its rate semantics."""
    with pm.Model() as model:
        pm.Gamma("g", alpha=2.0, beta=0.5)
    rv = model.named_vars["g"]
    lifted = lift_hyperparameters(model, ["g.beta"])
    h = lifted[0]

    new_scale = graph_replace(rv.owner.inputs[3], {h.constant_node: h.shared_var})
    f = pytensor.function([], new_scale)
    np.testing.assert_allclose(f(), 1.0 / 0.5, rtol=1e-6)
    h.shared_var.set_value(np.float32(2.0))
    np.testing.assert_allclose(f(), 1.0 / 2.0, rtol=1e-6)


def test_lift_exponential_lam():
    with pm.Model() as model:
        pm.Exponential("e", lam=1.5)
    lifted = lift_hyperparameters(model, ["e.lam"])
    assert len(lifted) == 1
    np.testing.assert_array_equal(lifted[0].shared_var.get_value(), np.float32(1.5))


def test_lift_exponential_scale():
    with pm.Model() as model:
        pm.Exponential("e", scale=2.0)
    lifted = lift_hyperparameters(model, ["e.scale"])
    assert len(lifted) == 1
    np.testing.assert_array_equal(lifted[0].shared_var.get_value(), np.float32(2.0))
