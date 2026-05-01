"""Per-Op handlers for moment propagation."""

import numpy as np
import pytensor.scalar.basic as ps
import pytensor.scalar.math as ps_math
import pytensor.tensor as pt

from pymc_models.calibration.moments.core import (
    Moments,
    register_moments,
    register_scalar_moments,
)
from pytensor.graph.basic import Apply
from pytensor.tensor.basic import (
    Alloc,
    MakeVector,
    ScalarFromTensor,
    TensorFromScalar,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Dot
from pytensor.tensor.shape import Reshape, SpecifyShape
from pytensor.tensor.special import LogSoftmax, Softmax, softmax
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)


def _is_constant_moments(m: Moments) -> bool:
    """A Moments value with no prior dependence — neither linear coeffs nor
    extra-var contributions."""
    return not m.coeffs and m.extra_var is None


def _broadcast_against_prior(value_factor, prior_ndim: int):
    """Pad a value-shaped tensor with trailing singleton axes so it broadcasts
    cleanly against a coefficient tensor whose trailing `prior_ndim` axes are
    the prior axes. No-op for scalar priors."""
    if prior_ndim == 0:
        return value_factor
    return value_factor[(...,) + (None,) * prior_ndim]


def _sum_extra(extras: list) -> "object | None":
    """Sum a list of extra_var values (None == zero), returning None if all
    inputs are None."""
    real = [e for e in extras if e is not None]
    if not real:
        return None
    total = real[0]
    for e in real[1:]:
        total = total + e
    return total


def _merge_coeffs(coeff_dicts: list[dict]) -> dict:
    """Merge per-prior coefficient dicts, summing on overlap (used when the
    *same* prior legitimately reaches multiple operands of the same node, as
    when `Σ x_i * θ` reduces over a shared `θ`)."""
    merged: dict = {}
    for d in coeff_dicts:
        for prior, c in d.items():
            if prior in merged:
                merged[prior] = merged[prior] + c
            else:
                merged[prior] = c
    return merged


# ----- Elemwise scalar ops -----


@register_scalar_moments(ps.Add)
def _add(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    """Linear combination — coefficients sum on shared priors, extra_vars sum
    (independent under our propagator's assumption)."""
    return Moments(
        mean=sum(m.mean for m in children),
        coeffs=_merge_coeffs([dict(m.coeffs) for m in children]),
        extra_var=_sum_extra([m.extra_var for m in children]),
    )


@register_scalar_moments(ps.Sub)
def _sub(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    a, b = children
    new_coeffs = dict(a.coeffs)
    for prior, c in b.coeffs.items():
        if prior in new_coeffs:
            new_coeffs[prior] = new_coeffs[prior] - c
        else:
            new_coeffs[prior] = -c
    # Var[A - B] = Var[A] + Var[B] under independence, same as Add.
    new_extra = _sum_extra([a.extra_var, b.extra_var])
    return Moments(mean=a.mean - b.mean, coeffs=new_coeffs, extra_var=new_extra)


@register_scalar_moments(ps.Neg)
def _neg(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    (a,) = children
    # Var[-X] = Var[X]; extra_var unchanged.
    return Moments(
        mean=-a.mean,
        coeffs={p: -c for p, c in a.coeffs.items()},
        extra_var=a.extra_var,
    )


@register_scalar_moments(ps.Mul)
def _mul(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    """Mul is linear only when at most one operand depends on priors."""
    non_const_idxs = [i for i, m in enumerate(children) if not _is_constant_moments(m)]
    if len(non_const_idxs) > 1:
        raise NotImplementedError(
            "Mul of two prior-dependent expressions is non-linear; "
            "propagate_moments cannot handle it without dropping the linear "
            "decomposition."
        )
    if not non_const_idxs:
        const_product = children[0].mean
        for m in children[1:]:
            const_product = const_product * m.mean
        return Moments(mean=const_product, coeffs={})
    (i,) = non_const_idxs
    a = children[i]
    const_factor = None
    for j, m in enumerate(children):
        if j == i:
            continue
        const_factor = m.mean if const_factor is None else const_factor * m.mean
    new_extra = const_factor**2 * a.extra_var if a.extra_var is not None else None
    return Moments(
        mean=const_factor * a.mean,
        coeffs={p: _broadcast_against_prior(const_factor, p.ndim) * c for p, c in a.coeffs.items()},
        extra_var=new_extra,
    )


@register_scalar_moments(ps.TrueDiv)
def _truediv(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    """Division is linear only when the denominator is constant in priors."""
    a, b = children
    if not _is_constant_moments(b):
        raise NotImplementedError(
            "Division by a prior-dependent expression is non-linear; "
            "propagate_moments cannot handle it."
        )
    new_extra = a.extra_var / b.mean**2 if a.extra_var is not None else None
    return Moments(
        mean=a.mean / b.mean,
        coeffs={p: c / _broadcast_against_prior(b.mean, p.ndim) for p, c in a.coeffs.items()},
        extra_var=new_extra,
    )


@register_scalar_moments(ps.Identity)
def _identity(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    """`pm.Deterministic(...)` wraps its expression in `Elemwise(Identity)`;
    moments pass through unchanged."""
    (a,) = children
    return a


@register_scalar_moments(ps.Cast)
def _cast(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    """Cast applies to mean, each coefficient, and the extra_var — dtype
    change only, distributional structure preserved."""
    (a,) = children
    target_dtype = scalar_op.o_type.dtype
    new_extra = pt.cast(a.extra_var, target_dtype) if a.extra_var is not None else None
    return Moments(
        mean=pt.cast(a.mean, target_dtype),
        coeffs={p: pt.cast(c, target_dtype) for p, c in a.coeffs.items()},
        extra_var=new_extra,
    )


@register_scalar_moments(ps.Exp)
def _exp(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    """y = exp(η). Treats η as approximately Normal so y ~ LogNormal:

        E[y]   = exp(μ + σ²/2)
        Var[y] = (exp(σ²) − 1) · exp(2μ + σ²)

    Exact when η is itself Normal (i.e. a linear combination of Normal
    priors); a moment-matching approximation otherwise. Collapses the
    per-prior decomposition: `coeffs={}` and the variance lives in
    `extra_var` because there's no honest linear coefficient on each prior
    after `exp`.
    """
    (a,) = children
    mu = a.mean
    sigma_sq = a.total_variance()
    new_mean = pt.exp(mu + sigma_sq / 2)
    new_extra = (pt.exp(sigma_sq) - 1) * pt.exp(2 * mu + sigma_sq)
    return Moments(mean=new_mean, coeffs={}, extra_var=new_extra)


@register_moments(Softmax)
def _softmax(op: Softmax, node: Apply, children: list[Moments]) -> Moments:
    """y = softmax(η, axis=op.axis). Treats η as approximately Normal with
    independent components — applies a MacKay-style element-wise compression
    before the softmax, generalizing the binary `_sigmoid` handler:

        E[y_k] ≈ softmax_k(  μ_k / sqrt(1 + π·σ_k²/8)  )
        Var[y_k] ≈ E[y_k] · (1 − E[y_k])               (Bernoulli bound)

    For binary softmax([η, 0]) this reduces exactly to the binary sigmoid
    formulas. For the general multinomial case it's a heuristic — the exact
    multivariate softmax-Normal expectation has no closed form. Independence
    between η components is implicit in the per-element correction, which is
    accurate when off-diagonal covariance is small.
    """
    (a,) = children
    mu = a.mean
    sigma_sq = a.total_variance()
    mu_corrected = mu / pt.sqrt(1 + np.pi * sigma_sq / 8)
    new_mean = softmax(mu_corrected, axis=op.axis)
    new_extra = new_mean * (1 - new_mean)
    return Moments(mean=new_mean, coeffs={}, extra_var=new_extra)


@register_moments(LogSoftmax)
def _log_softmax(op: LogSoftmax, node: Apply, children: list[Moments]) -> Moments:
    """y = log_softmax(η). Mean uses the same MacKay-corrected log of the
    Softmax expectation; variance is the delta-method first-order
    (gradient of log_softmax is `1 − softmax`)."""
    (a,) = children
    mu = a.mean
    sigma_sq = a.total_variance()
    mu_corrected = mu / pt.sqrt(1 + np.pi * sigma_sq / 8)
    sm = softmax(mu_corrected, axis=op.axis)
    new_mean = pt.log(sm)
    # d log_softmax_k / d eta_k = 1 - softmax_k; off-diag terms treated as
    # independent under the same simplifying assumption.
    new_extra = (1 - sm) ** 2 * sigma_sq
    return Moments(mean=new_mean, coeffs={}, extra_var=new_extra)


@register_scalar_moments(ps_math.Sigmoid)
def _sigmoid(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    """y = sigmoid(η). Treats η as approximately Normal:

        E[y]   ≈ sigmoid(μ / sqrt(1 + π·σ²/8))                   (MacKay)
        Var[y] ≈ min(  sigmoid'(μ)² · σ²,                        (delta)
                       E[y] · (1 − E[y])  )                       (Bernoulli bound)

    Delta method is correct for small σ_η; the Bernoulli bound caps the
    variance for large σ_η (where sigmoid(η) approaches a Bernoulli).
    Taking the min is conservative on both sides. This is the variance of
    `sigmoid(η)` itself; the Bernoulli *output* variance under
    `y ~ Bernoulli(sigmoid(η))` is `E[y](1−E[y])` (which the Bernoulli
    likelihood handler computes directly via `y² = y`).
    """
    (a,) = children
    mu = a.mean
    sigma_sq = a.total_variance()
    new_mean = pt.sigmoid(mu / pt.sqrt(1 + np.pi * sigma_sq / 8))
    sig_mu = pt.sigmoid(mu)
    delta_var = (sig_mu * (1 - sig_mu)) ** 2 * sigma_sq
    bernoulli_bound = new_mean * (1 - new_mean)
    new_extra = pt.minimum(delta_var, bernoulli_bound)
    return Moments(mean=new_mean, coeffs={}, extra_var=new_extra)


@register_scalar_moments(ps.Reciprocal)
def _reciprocal(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    """`1/x` — supported only when x is constant in priors. PyMC's Gamma /
    Exponential rate-parameterizations store `Reciprocal(constant)` at the
    underlying RV's input, so this comes up routinely in hyperparameter walks
    even when the user's model isn't doing anything fancy."""
    (a,) = children
    if not _is_constant_moments(a):
        raise NotImplementedError(
            "Reciprocal of a prior-dependent expression is non-linear; "
            "propagate_moments cannot handle it."
        )
    return Moments(mean=1.0 / a.mean, coeffs={})


@register_scalar_moments(ps.Pow)
def _pow(scalar_op, node: Apply, children: list[Moments]) -> Moments:
    """Pow is supported only when both base and exponent are constant in priors
    (so the result is itself a constant). Anything else is non-linear."""
    a, b = children
    if not _is_constant_moments(a) or not _is_constant_moments(b):
        raise NotImplementedError(
            "Pow with a prior-dependent base or exponent is non-linear; "
            "propagate_moments cannot handle it."
        )
    return Moments(mean=a.mean**b.mean, coeffs={})


# ----- Tensor-level ops -----


@register_moments(Dot)
def _dot(op: Dot, node: Apply, children: list[Moments]) -> Moments:
    """`Dot(A, x)` is linear when at most one operand is prior-dependent.

    `extra_var` propagates through Dot the same way variance does for a
    linear transform: if the prior side has extra variance per element σ²_e,
    then `Var[(A @ x)_i] += sum_j A[i,j]² · σ²_e[j]`, i.e. `(A**2) @ σ²_e`.
    """
    a, b = children
    a_const = _is_constant_moments(a)
    b_const = _is_constant_moments(b)
    if not a_const and not b_const:
        raise NotImplementedError(
            "Dot of two prior-dependent operands is non-linear; "
            "propagate_moments cannot handle it."
        )
    if a_const and b_const:
        return Moments(mean=a.mean @ b.mean, coeffs={})
    if a_const:
        new_extra = (a.mean**2) @ b.extra_var if b.extra_var is not None else None
        return Moments(
            mean=a.mean @ b.mean,
            coeffs={p: a.mean @ c for p, c in b.coeffs.items()},
            extra_var=new_extra,
        )
    new_extra = a.extra_var @ (b.mean**2) if a.extra_var is not None else None
    return Moments(
        mean=a.mean @ b.mean,
        coeffs={p: c @ b.mean for p, c in a.coeffs.items()},
        extra_var=new_extra,
    )


@register_moments(DimShuffle)
def _dimshuffle(op: DimShuffle, node: Apply, children: list[Moments]) -> Moments:
    """Pass moments through a DimShuffle (which subsumes ExpandDims and Squeeze).

    Mean and extra_var have the same shape as the value, so the original
    DimShuffle applies directly. For coefficients on vector priors, the
    coefficient carries `prior.ndim` trailing axes that must remain untouched;
    we extend `input_ndim` and `new_order` to leave those axes alone.
    """
    a = children[0]
    new_mean = op(a.mean)
    new_coeffs = {p: _dimshuffle_on_coeff(op, c, p.ndim) for p, c in a.coeffs.items()}
    new_extra = op(a.extra_var) if a.extra_var is not None else None
    return Moments(mean=new_mean, coeffs=new_coeffs, extra_var=new_extra)


def _dimshuffle_on_coeff(op: DimShuffle, c: Apply, prior_ndim: int):
    if prior_ndim == 0:
        return op(c)
    n = op.input_ndim
    extended_input_ndim = n + prior_ndim
    extended_order = list(op.new_order) + [n + i for i in range(prior_ndim)]
    new_op = DimShuffle(input_ndim=extended_input_ndim, new_order=extended_order)
    return new_op(c)


def _shape_only_handler_factory(op_name):
    """Factory for shape-only ops (Reshape, SpecifyShape, Alloc) that take a
    value plus shape inputs. Mean, coefficients, and extra_var all broadcast
    via the same op. Vector priors are refused for now (would require
    extending the shape input with `prior.shape`)."""

    def handler(op, node: Apply, children: list[Moments]) -> Moments:
        a = children[0]
        _require_scalar_priors(a, op_name)
        new_mean = op.make_node(a.mean, *node.inputs[1:]).outputs[0]
        new_coeffs = {p: op.make_node(c, *node.inputs[1:]).outputs[0] for p, c in a.coeffs.items()}
        new_extra = (
            op.make_node(a.extra_var, *node.inputs[1:]).outputs[0]
            if a.extra_var is not None
            else None
        )
        return Moments(mean=new_mean, coeffs=new_coeffs, extra_var=new_extra)

    handler.__name__ = f"_{op_name}_handler"
    return handler


register_moments(Reshape)(_shape_only_handler_factory("Reshape"))
register_moments(SpecifyShape)(_shape_only_handler_factory("SpecifyShape"))
register_moments(Alloc)(_shape_only_handler_factory("Alloc"))


@register_moments(TensorFromScalar)
def _tensor_from_scalar(op, node: Apply, children: list[Moments]) -> Moments:
    """Wraps a 0-d scalar into a 0-d tensor — distributional structure
    unchanged. Mean, coefficients, and extra_var pass through the same op."""
    (a,) = children
    new_extra = op(a.extra_var) if a.extra_var is not None else None
    return Moments(
        mean=op(a.mean),
        coeffs={p: op(c) for p, c in a.coeffs.items()},
        extra_var=new_extra,
    )


@register_moments(ScalarFromTensor)
def _scalar_from_tensor(op, node: Apply, children: list[Moments]) -> Moments:
    """Inverse of `TensorFromScalar`; same passthrough story."""
    (a,) = children
    new_extra = op(a.extra_var) if a.extra_var is not None else None
    return Moments(
        mean=op(a.mean),
        coeffs={p: op(c) for p, c in a.coeffs.items()},
        extra_var=new_extra,
    )


@register_moments(Subtensor)
def _subtensor(op: Subtensor, node: Apply, children: list[Moments]) -> Moments:
    """Index the value axes of mean, coefficients, and extra_var. The op's
    `idx_list` is indexed by the leading value axes; trailing prior axes
    pass through unchanged because Subtensor leaves untouched axes alone."""
    a = children[0]
    idx_inputs = node.inputs[1:]
    new_extra = op(a.extra_var, *idx_inputs) if a.extra_var is not None else None
    return Moments(
        mean=op(a.mean, *idx_inputs),
        coeffs={p: op(c, *idx_inputs) for p, c in a.coeffs.items()},
        extra_var=new_extra,
    )


def _advanced_subtensor_handler(op, node: Apply, children: list[Moments]) -> Moments:
    """Read-side fancy indexing (`x[idx]` where `idx` is a tensor). Index the
    leading value axes of mean, each coefficient, and extra_var with the same
    indices; trailing prior axes are unaffected. Vector-prior support is
    refused because advanced indexing can intermix value and prior axes."""
    a = children[0]
    _require_scalar_priors(a, type(op).__name__)
    idx_inputs = node.inputs[1:]
    new_extra = op(a.extra_var, *idx_inputs) if a.extra_var is not None else None
    return Moments(
        mean=op(a.mean, *idx_inputs),
        coeffs={p: op(c, *idx_inputs) for p, c in a.coeffs.items()},
        extra_var=new_extra,
    )


register_moments(AdvancedSubtensor)(_advanced_subtensor_handler)
register_moments(AdvancedSubtensor1)(_advanced_subtensor_handler)


def _require_scalar_priors(moments: Moments, op_name: str) -> None:
    for prior in moments.coeffs:
        if prior.ndim > 0:
            raise NotImplementedError(
                f"{op_name} on an expression with vector-prior contributions "
                f"is not supported (prior {prior.name or '<unnamed>'} has "
                f"ndim {prior.ndim})."
            )


def _zeros_coeff(value_var, prior):
    """Build a zeros tensor of shape `value_var.shape + prior.shape` (the
    coefficient layout: leading value axes, trailing prior axes)."""
    shape_parts = list(value_var.shape)
    if prior.ndim > 0:
        shape_parts = shape_parts + list(prior.shape)
    return pt.zeros(shape_parts)


def _union_priors(*coeff_dicts):
    """Ordered union of priors across multiple coeff dicts, preserving first-seen order."""
    seen: set[int] = set()
    out: list = []
    for d in coeff_dicts:
        for p in d:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def _zeros_value(value_var):
    """Build a zeros tensor of `value_var`'s shape (for extra_var placeholders
    on operands that don't have a non-linear contribution)."""
    return pt.zeros(list(value_var.shape))


def _inc_set_subtensor_handler(op, node: Apply, children: list[Moments]) -> Moments:
    """Shared handler for `(Advanced)IncSubtensor` ops.

    `set_subtensor(x[idx], y)` and `inc_subtensor(x[idx], y)` both produce an
    `IncSubtensor` (or `AdvancedIncSubtensor`) op; they're distinguished by
    the op's `set_instead_of_inc` flag. Either way we propagate moments by:
      mean   ← op(mean_x, mean_y, *idx)
      coeff  ← op(coeff_x, coeff_y, *idx)
    For each prior θ: if θ is missing from x's or y's coeffs, fill with zeros
    of the right `value_shape + prior_shape`. The op carries the indices in
    its state, so re-applying it to the coefficients reuses the same indexing.
    """
    a, b, *_ = children  # x, y, *idx_moments (idx moments unused; op carries idx state)
    _, _, *idx_inputs = node.inputs
    new_mean = op(a.mean, b.mean, *idx_inputs)

    new_coeffs: dict = {}
    for prior in _union_priors(a.coeffs, b.coeffs):
        cx = a.coeffs.get(prior)
        if cx is None:
            cx = _zeros_coeff(a.mean, prior)
        cy = b.coeffs.get(prior)
        if cy is None:
            cy = _zeros_coeff(b.mean, prior)
        new_coeffs[prior] = op(cx, cy, *idx_inputs)

    # extra_var: combine the same way as coeffs — apply the op with zero
    # placeholders for whichever side is missing.
    if a.extra_var is None and b.extra_var is None:
        new_extra = None
    else:
        ex = a.extra_var if a.extra_var is not None else _zeros_value(a.mean)
        ey = b.extra_var if b.extra_var is not None else _zeros_value(b.mean)
        new_extra = op(ex, ey, *idx_inputs)

    return Moments(mean=new_mean, coeffs=new_coeffs, extra_var=new_extra)


register_moments(IncSubtensor)(_inc_set_subtensor_handler)
register_moments(AdvancedIncSubtensor)(_inc_set_subtensor_handler)
register_moments(AdvancedIncSubtensor1)(_inc_set_subtensor_handler)


@register_moments(MakeVector)
def _make_vector(op: MakeVector, node: Apply, children: list[Moments]) -> Moments:
    """`MakeVector(*scalars)` stacks N scalar inputs into a length-N vector
    (what `pt.stack([a, b, ...])` lowers to for scalar inputs)."""
    all_priors = _union_priors(*[c.coeffs for c in children])
    for prior in all_priors:
        if prior.ndim > 0:
            raise NotImplementedError(
                f"MakeVector with vector-prior contributions is not supported "
                f"(prior {prior.name or '<unnamed>'} has ndim {prior.ndim})."
            )
    new_mean = op(*[c.mean for c in children])
    new_coeffs: dict = {}
    for prior in all_priors:
        per_child_scalars = [
            child.coeffs.get(prior, pt.zeros_like(child.mean)) for child in children
        ]
        new_coeffs[prior] = op(*per_child_scalars)

    if any(c.extra_var is not None for c in children):
        per_child_extra = [
            c.extra_var if c.extra_var is not None else pt.zeros_like(c.mean) for c in children
        ]
        new_extra = op(*per_child_extra)
    else:
        new_extra = None

    return Moments(mean=new_mean, coeffs=new_coeffs, extra_var=new_extra)


@register_moments(Blockwise)
def _blockwise(op: Blockwise, node: Apply, children: list[Moments]) -> Moments:
    """Blockwise wraps a core op (e.g. Dot for `pt.matmul`). For a degenerate
    Blockwise that adds no batch dims, we can delegate to the core op's
    handler. Genuinely batched Blockwise needs Jacobian-aware propagation
    over the batch axes and isn't yet supported."""
    from pymc_models.calibration.moments.core import _moments_op

    if op.batch_ndim(node) > 0:
        raise NotImplementedError(
            f"Blockwise({type(op.core_op).__name__}) with batch ndim "
            f"{op.batch_ndim(node)} is not yet supported. The core op's "
            f"handler would need to propagate Jacobians over the batch axes."
        )
    core_node = op.core_op.make_node(*node.inputs)
    return _moments_op(op.core_op, core_node, children)
