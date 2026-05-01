"""Orchestrator for prior calibration.

`calibrate_priors(model, loss, target_priors, ...)` lifts the requested
hyperparameters into tunable shared variables, lets the user build a symbolic
loss graph in terms of those shareds (typically using `pytensor-distributions`
for closed-form moments / quantiles), then runs Adam against the loss until
either `n_steps` is reached or the change is below tolerance.

This is the "no forward sampling" path: the loss only sees the lifted
hyperparameters, so it must be expressible analytically (closed-form moments,
quantiles, KL divergences, etc.). Sample-based losses come later.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc_models.calibration.lifter import LiftedHyperparameter, lift_hyperparameters
from pymc_models.calibration.targets import TargetSpec
from pytensor.graph.replace import graph_replace

LossFn = Callable[[dict[str, pt.TensorVariable]], pt.TensorVariable]


@dataclass
class CalibrationResult:
    values: dict[str, np.ndarray]
    history: np.ndarray  # loss per step
    converged: bool
    lifted: list[LiftedHyperparameter]


def calibrate_priors(
    model,
    loss: LossFn,
    target_priors: Iterable[TargetSpec],
    *,
    n_steps: int = 500,
    learning_rate: float = 0.05,
    tol: float = 1e-6,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    seed: int | None = None,
) -> CalibrationResult:
    """Tune the requested prior hyperparameters by minimising the user's loss."""
    lifted = lift_hyperparameters(model, target_priors)
    if not lifted:
        raise ValueError("No hyperparameters were lifted; nothing to calibrate.")

    params = {h.name: h.shared_var for h in lifted}
    loss_expr = loss(params)
    if loss_expr.ndim != 0:
        raise ValueError(f"Loss must be a scalar TensorVariable; got ndim={loss_expr.ndim}.")

    # Substitute lifted constants → shareds in the user's loss expression.
    # Lets the user reference moments / probabilities computed from the model
    # graph directly (e.g. via `gaussian_glm_moments(model.named_vars["y"])`)
    # — those expressions reference the original constants, which we swap.
    constant_to_shared = {h.constant_node: h.shared_var for h in lifted}
    loss_expr = graph_replace(loss_expr, constant_to_shared, strict=False)

    shared_vars = [h.shared_var for h in lifted]
    grads = pt.grad(loss_expr, shared_vars)
    loss_and_grad = pytensor.function([], [loss_expr, *grads])

    np.random.default_rng(seed)
    m = [np.zeros_like(h.initial_value) for h in lifted]
    v = [np.zeros_like(h.initial_value) for h in lifted]

    history = np.empty(n_steps, dtype=np.float64)
    converged = False
    prev_loss = np.inf

    for t in range(1, n_steps + 1):
        out = loss_and_grad()
        loss_val = float(out[0])
        grad_vals = [np.asarray(g) for g in out[1:]]
        history[t - 1] = loss_val

        for i, (h, g) in enumerate(zip(lifted, grad_vals, strict=True)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            m_hat = m[i] / (1 - beta1**t)
            v_hat = v[i] / (1 - beta2**t)
            update = learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            new_value = h.shared_var.get_value() - update.astype(h.initial_value.dtype)
            h.shared_var.set_value(new_value)

        if abs(prev_loss - loss_val) < tol:
            history = history[:t]
            converged = True
            break
        prev_loss = loss_val

    values = {h.name: h.shared_var.get_value() for h in lifted}
    return CalibrationResult(values=values, history=history, converged=converged, lifted=lifted)
