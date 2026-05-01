"""Whitelist resolution for prior calibration.

Maps user-supplied target specifications onto concrete `(rv, param_name, input_index)`
records. Three accepted target forms:

    [beta]          # a Variable in scope; lift every supported parameter
    ["beta"]        # an RV name; lift every supported parameter
    ["beta.sigma"]  # an RV name with a specific parameter; lift only that one

The registry below pins each `RandomVariable` subclass to the user-facing parameter
names PyMC uses, with the corresponding index into `node.inputs` (which is laid out
as `(rng, size, *params)`). Distributions whose user-facing parameterization differs
from the underlying pytensor RV's positional inputs (e.g. `pm.Gamma(alpha, beta)`
where `beta` is rate but the RV stores `1/beta`) are intentionally omitted; they
need a graph-walk to find the original constant and that's a follow-up.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import pymc as pm

from pymc.distributions.continuous import HalfCauchyRV, KumaraswamyRV
from pytensor.graph.basic import Variable
from pytensor.tensor.random.basic import (
    BetaRV,
    CauchyRV,
    ExponentialRV,
    GammaRV,
    HalfNormalRV,
    InvGammaRV,
    LaplaceRV,
    LogNormalRV,
    NormalRV,
    ParetoRV,
    StudentTRV,
    TriangularRV,
    UniformRV,
)

# Per-RV: user-facing param name -> input index in node.inputs.
# node.inputs are laid out as (rng, size, *params), so positional params start at 2.
# Multiple names can map to the same slot when PyMC accepts dual parameterizations
# (e.g. Exponential with `lam` or `scale`). The lifter walks back from this slot
# through the deterministic subgraph until it hits a Constant, so any rate↔scale
# or sigma↔tau translation PyMC inserts is handled transparently.
RV_PARAMS: dict[type, dict[str, int]] = {
    NormalRV: {"mu": 2, "sigma": 3},
    HalfNormalRV: {"sigma": 3},
    LogNormalRV: {"mu": 2, "sigma": 3},
    UniformRV: {"lower": 2, "upper": 3},
    StudentTRV: {"nu": 2, "mu": 3, "sigma": 4},
    BetaRV: {"alpha": 2, "beta": 3},
    CauchyRV: {"alpha": 2, "beta": 3},
    HalfCauchyRV: {"beta": 2},
    LaplaceRV: {"mu": 2, "b": 3},
    KumaraswamyRV: {"a": 2, "b": 3},
    ParetoRV: {"alpha": 2, "m": 3},
    TriangularRV: {"lower": 2, "c": 3, "upper": 4},
    GammaRV: {"alpha": 2, "beta": 3},
    InvGammaRV: {"alpha": 2, "beta": 3},
    ExponentialRV: {"lam": 2, "scale": 2},
}


@dataclass(frozen=True)
class TargetParam:
    rv: Variable
    rv_name: str
    param_name: str
    input_index: int


TargetSpec = Variable | str


def resolve_targets(model: pm.Model, target_priors: Iterable[TargetSpec]) -> list[TargetParam]:
    """Resolve a whitelist into concrete `TargetParam` records."""
    out: list[TargetParam] = []

    for spec in target_priors:
        rv, rv_name, requested_param = _parse_spec(model, spec)
        rv_class = type(rv.owner.op)

        if rv_class not in RV_PARAMS:
            raise NotImplementedError(
                f"Distribution {rv_class.__name__!r} (for {rv_name!r}) is not in the "
                f"calibration registry. Supported: "
                f"{sorted(c.__name__ for c in RV_PARAMS)}."
            )

        available = RV_PARAMS[rv_class]
        if requested_param is None:
            param_names = list(available.keys())
        else:
            if requested_param not in available:
                raise KeyError(
                    f"{rv_name}.{requested_param!r} is not a tunable parameter. "
                    f"Available for {rv_class.__name__}: {sorted(available)}."
                )
            param_names = [requested_param]

        for name in param_names:
            out.append(
                TargetParam(
                    rv=rv,
                    rv_name=rv_name,
                    param_name=name,
                    input_index=available[name],
                )
            )

    return out


def _parse_spec(model: pm.Model, spec: TargetSpec) -> tuple[Variable, str, str | None]:
    if isinstance(spec, str):
        if "." in spec:
            rv_name, param_name = spec.split(".", 1)
        else:
            rv_name = spec
            param_name = None
        if rv_name not in model.named_vars:
            raise KeyError(f"{rv_name!r} not found in model.named_vars")
        rv = model.named_vars[rv_name]
        return rv, rv_name, param_name

    if isinstance(spec, Variable):
        if spec.name is None:
            raise ValueError("Variable target must have a name set")
        if spec.name not in model.named_vars:
            raise KeyError(f"{spec.name!r} not found in model.named_vars")
        return spec, spec.name, None

    raise TypeError(f"Target spec must be a Variable or string, got {type(spec).__name__}")
