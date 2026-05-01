"""Lift constant prior hyperparameters into tunable shared variables.

The lifter does not mutate the model. It collects `(constant_node, shared_var)`
records that callers can apply to a graph via `pytensor.graph.replace.graph_replace`
when they build the forward / loss graph downstream.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pytensor

from pymc_models.calibration.targets import TargetParam, TargetSpec, resolve_targets
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.sharedvar import SharedVariable


@dataclass(frozen=True)
class LiftedHyperparameter:
    rv: Variable
    rv_name: str
    param_name: str
    constant_node: Constant
    shared_var: SharedVariable
    initial_value: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.rv_name}.{self.param_name}"


def lift_hyperparameters(model, target_priors: Iterable[TargetSpec]) -> list[LiftedHyperparameter]:
    """Resolve targets and create tunable shared variables for each.

    Reads the current numeric value at each target slot, allocates a shared
    variable initialized to that value, and records the (original constant,
    new shared) pair so the caller can substitute downstream.
    """
    targets = resolve_targets(model, target_priors)
    lifted: list[LiftedHyperparameter] = []

    for tp in targets:
        input_var = tp.rv.owner.inputs[tp.input_index]
        const_node = _find_lifted_constant(input_var, tp)

        value = np.asarray(const_node.data).copy()
        shared = pytensor.shared(value.copy(), name=f"{tp.rv_name}.{tp.param_name}")
        lifted.append(
            LiftedHyperparameter(
                rv=tp.rv,
                rv_name=tp.rv_name,
                param_name=tp.param_name,
                constant_node=const_node,
                shared_var=shared,
                initial_value=value,
            )
        )

    return lifted


def _find_lifted_constant(input_var: Variable, tp: TargetParam) -> Constant:
    """Walk back from `input_var` and return the unique user-supplied Constant.

    Uses `pytensor.graph.traversal.ancestors` with all upstream `RandomVariable`
    outputs as blockers, so a hierarchical prior is treated as opaque — we never
    reach into another distribution's parameters. Whatever deterministic ops sit
    between the constant and the RV input slot stay in the graph after
    substitution, preserving PyMC's parameterization translations (e.g.
    rate→scale via `Reciprocal`).
    """
    rv_blockers = [
        var
        for var in ancestors([input_var])
        if var.owner is not None and isinstance(var.owner.op, RandomVariable)
    ]
    constants = {
        id(var): var
        for var in ancestors([input_var], blockers=rv_blockers)
        if isinstance(var, Constant)
    }

    if not constants:
        raise ValueError(
            f"{tp.rv_name}.{tp.param_name}: no Constant found in the subgraph "
            f"feeding input slot {tp.input_index}. The hyperparameter is fully "
            f"determined by random variables — there is nothing to lift."
        )
    if len(constants) > 1:
        raise ValueError(
            f"{tp.rv_name}.{tp.param_name}: ambiguous lift — found "
            f"{len(constants)} distinct Constants in the subgraph feeding input "
            f"slot {tp.input_index}. Rewrite the model so the parameter is "
            f"derived from a single user-supplied value."
        )
    return next(iter(constants.values()))
