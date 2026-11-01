"""
Compile a scan step with small static shapes to straight-line scalar code in the numba backend.

The numba backend compiles every node of a scan step to its own function, and each one pays for argument
refcounting, an output allocation and a kernel sized at runtime. With small static shapes those costs dwarf the
arithmetic. ``ScalarizedStep`` wraps the whole step, and its numba dispatch keeps one scalar local per tensor
element, so arrays only exist for the step's inputs and outputs.
"""

import math

from hashlib import sha256
from itertools import product

import numpy as np

from pytensor.compile import optdb
from pytensor.compile.builders import FinalizedOpFromGraph, OpFromGraph
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.compile.rewriting import inline_ofg_node
from pytensor.graph import node_rewriter
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter
from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    cache_key_for_constant,
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.elemwise import _reduce_identity, scalar_in_place_fn
from pytensor.link.numba.dispatch.string_codegen import (
    CODE_TOKEN,
    build_source_code,
    create_tuple_string,
)
from pytensor.scalar.basic import ScalarOp, ScalarType, get_scalar_type
from pytensor.scan.op import Scan
from pytensor.tensor.basic import ScalarFromTensor, TensorFromScalar
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.shape import SpecifyShape
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    IncSubtensor,
    Subtensor,
    get_constant_idx,
)
from pytensor.tensor.type import TensorType

MAX_SCALARIZED_ELEMENTS = 64
MAX_SCALARIZED_OPERATIONS = 20_000

RELABELING_OPS = (ScalarFromTensor, TensorFromScalar, DeepCopyOp, ViewOp, SpecifyShape)
SUBTENSOR_OPS = (Subtensor, AdvancedSubtensor)
INC_SUBTENSOR_OPS = (IncSubtensor, AdvancedIncSubtensor)
# Numba arithmetic already returns these dtypes, so only other results need an explicit cast.
UNCAST_DTYPES = {"bool", "int64", "uint64", "float64", "complex128"}


class ScalarizedStep(FinalizedOpFromGraph):
    """A scan step whose numba implementation holds every tensor element in its own scalar local."""

    def __str__(self):
        return "ScalarizedStep"


def shape_of(var):
    return () if isinstance(var.type, ScalarType) else var.type.shape


def inlined(fgraph):
    """Mutable copy of ``fgraph`` with every nested ``OpFromGraph`` inlined, so constants reach their clients."""
    fgraph = fgraph.unfreeze()
    while nodes := [node for node in fgraph.toposort() if isinstance(node.op, OpFromGraph)]:
        for node in nodes:
            fgraph.replace_all(zip(node.outputs, inline_ofg_node(node), strict=True))
    return fgraph


def static_index(node):
    """A ``*Subtensor`` node's index as a NumPy index tuple, or None unless every entry is constant."""
    # get_constant_idx skips the first input, which is the indexed array here and the update for an increment.
    first_input = 1 if isinstance(node.op, INC_SUBTENSOR_OPS) else 0
    entries = get_constant_idx(
        node.op.idx_list,
        node.inputs[first_input:],
        allow_partial=True,
        only_process_constants=True,
    )
    index = []
    for entry in entries:
        if isinstance(entry, slice):
            if any(isinstance(bound, Variable) for bound in (entry.start, entry.stop, entry.step)):
                return None
            index.append(entry)
        elif isinstance(entry, Constant):
            index.append(entry.data)
        elif isinstance(entry, Variable):
            return None
        else:
            index.append(entry)
    return tuple(index)


def element_positions(shape, index):
    """Flat C-order positions in an array of ``shape`` that ``index`` selects, in the shape NumPy gives them."""
    return np.arange(math.prod(shape)).reshape(shape)[index]


def operation_count(node):
    """Scalar operations the translation emits for ``node``, or None when it has no translation."""
    op = node.op
    if isinstance(op, (*RELABELING_OPS, DimShuffle)):
        return 0
    if isinstance(op, Elemwise | ScalarOp):
        return math.prod(shape_of(node.outputs[0]))
    if isinstance(op, Dot):
        if node.inputs[0].type.ndim != 2 or node.inputs[1].type.ndim != 2:
            return None
        (rows, inner), (_, cols) = node.inputs[0].type.shape, node.inputs[1].type.shape
        return rows * inner * cols
    if isinstance(op, CAReduce):
        return math.prod(shape_of(node.inputs[0]))
    if isinstance(op, (*SUBTENSOR_OPS, *INC_SUBTENSOR_OPS)):
        index = static_index(node)
        if index is None:
            return None
        return (
            element_positions(shape_of(node.inputs[0]), index).size
            if isinstance(op, INC_SUBTENSOR_OPS)
            else 0
        )
    return None


def graph_operation_count(fgraph):
    """Scalar operations the translation of ``fgraph`` emits, or None when it cannot be scalarized."""
    fgraph = inlined(fgraph)
    for var in fgraph.variables:
        if isinstance(var.type, ScalarType):
            continue
        if not isinstance(var.type, TensorType) or None in var.type.shape:
            return None
        if math.prod(var.type.shape) > MAX_SCALARIZED_ELEMENTS:
            return None
    total = 0
    for node in fgraph.apply_nodes:
        count = operation_count(node)
        if count is None:
            return None
        total += count
    return total


def cast(expression, dtype):
    """``expression`` narrowed to ``dtype`` where numba arithmetic would promote it."""
    return expression if dtype in UNCAST_DTYPES else f"np.{dtype}({expression})"


class StepScalarizer:
    """Translate a graph into the body of one numba function with a scalar local per tensor element."""

    def __init__(self, **funcify_kwargs):
        self.funcify_kwargs = funcify_kwargs
        self.lines = []
        self.env = {"np": np}
        self.kernels = {}
        self.kernel_cache_keys = []
        self.constants = {}
        self.constant_values = {}
        self.dot_sums = {}
        self.counter = 0

    def fresh(self, prefix="v"):
        self.counter += 1
        return f"{prefix}{self.counter}"

    def constant(self, value, dtype):
        """Name of a global holding ``value`` as a ``dtype`` scalar, which numba types as that dtype."""
        value = np.asarray(value).astype(dtype)[()]
        key = cache_key_for_constant(value)
        if key not in self.constants:
            name = self.fresh("constant")
            self.env[name] = value
            self.constants[key] = name
            self.constant_values[name] = value
        return self.constants[key]

    def scalar_kernel(self, scalar_op, scalar_node):
        key = (scalar_op, *(str(var.type) for var in (*scalar_node.inputs, *scalar_node.outputs)))
        if key not in self.kernels:
            name = self.fresh("kernel")
            self.env[name], cache_key = numba_funcify_and_cache_key(
                scalar_op, node=scalar_node, **self.funcify_kwargs
            )
            self.kernel_cache_keys.append(cache_key)
            self.kernels[key] = name
        return self.kernels[key]

    def call(self, kernel, arguments, n_outputs):
        targets = [self.fresh() for _ in range(n_outputs)]
        self.lines.append(f"{', '.join(targets)} = {kernel}({', '.join(arguments)})")
        return targets

    def dot_element(self, pairs, dtype):
        """
        One element of a matrix product: the sum of ``a * b`` over ``pairs``.

        Only exact simplifications apply: ``1 * x`` is ``x`` and constant products fold. A ``0 * x`` term with a
        runtime ``x`` stays, since it is NaN when ``x`` is. Elements with the same products, such as the transposed
        entries of ``X @ X.T``, share one local.
        """
        terms, constant = [], 0
        for a, b in pairs:
            a_value, b_value = self.constant_values.get(a), self.constant_values.get(b)
            if a_value is not None and b_value is not None:
                constant += a_value * b_value
            elif a_value == 1:
                terms.append(b)
            elif b_value == 1:
                terms.append(a)
            else:
                terms.append(" * ".join(sorted((a, b))))
        if constant != 0:
            terms.append(self.constant(constant, dtype))
        if not terms:
            return self.constant(0, dtype)
        if len(terms) == 1 and " " not in terms[0]:
            return terms[0]
        key = tuple(sorted(terms))
        if key not in self.dot_sums:
            self.dot_sums[key] = self.fresh()
            self.lines.append(f"{self.dot_sums[key]} = {cast(' + '.join(key), dtype)}")
        return self.dot_sums[key]

    def translate_graph(self, fgraph, input_elements):
        """Elements of ``fgraph``'s outputs, given the elements of its inputs."""
        fgraph = inlined(fgraph)
        elements = dict(zip(fgraph.inputs, input_elements, strict=True))

        def get(var):
            if isinstance(var, Constant) and var not in elements:
                data = np.asarray(var.data)
                elements[var] = {
                    index: self.constant(data[index], data.dtype)
                    for index in np.ndindex(data.shape)
                }
            return elements[var]

        for node in fgraph.toposort():
            outputs = self.translate_node(node, [get(var) for var in node.inputs])
            elements.update(zip(node.outputs, outputs, strict=True))
        return [get(var) for var in fgraph.outputs]

    def reduce(self, scalar_op, accumulator, member, kernel):
        """Fold ``member`` into the local ``accumulator`` in place, as pytensor's numba reductions do."""
        try:
            lines = scalar_in_place_fn(scalar_op, "()", accumulator, member)
        except NotImplementedError:
            lines = [f"{accumulator} = {kernel()}({accumulator}, {member})"]
        self.lines += [
            line
            if isinstance(line, CODE_TOKEN)
            else line.replace(f"{accumulator}[()]", accumulator)
            for line in lines
        ]

    def translate_node(self, node, inputs):
        op = node.op

        if isinstance(op, RELABELING_OPS):
            return [inputs[0]]

        if isinstance(op, DimShuffle):
            elements = {}
            for index in np.ndindex(shape_of(node.outputs[0])):
                input_index = [0] * node.inputs[0].type.ndim
                for output_axis, input_axis in enumerate(op.new_order):
                    if input_axis != "x":
                        input_index[input_axis] = index[output_axis]
                elements[index] = inputs[0][tuple(input_index)]
            return [elements]

        if isinstance(op, Elemwise | ScalarOp):
            if isinstance(op, Elemwise):
                kernel = self.scalar_kernel(op.scalar_op, op.make_scalar_node(*node.inputs))
            else:
                kernel = self.scalar_kernel(op, node)
            outputs = [{} for _ in node.outputs]
            for index in np.ndindex(shape_of(node.outputs[0])):
                arguments = [
                    var_elements[
                        tuple(
                            0 if length == 1 else i
                            for i, length in zip(index, shape_of(var), strict=True)
                        )
                    ]
                    for var, var_elements in zip(node.inputs, inputs, strict=True)
                ]
                for output, name in zip(
                    outputs, self.call(kernel, arguments, len(node.outputs)), strict=True
                ):
                    output[index] = name
            return outputs

        if isinstance(op, Dot):
            (rows, inner), (_, cols) = node.inputs[0].type.shape, node.inputs[1].type.shape
            return [
                {
                    (i, j): self.dot_element(
                        [(inputs[0][i, k], inputs[1][k, j]) for k in range(inner)],
                        node.outputs[0].dtype,
                    )
                    for i, j in product(range(rows), range(cols))
                }
            ]

        if isinstance(op, CAReduce):
            axes = range(node.inputs[0].type.ndim) if op.axis is None else op.axis
            input_dtype, output_dtype = node.inputs[0].dtype, node.outputs[0].dtype
            accumulator_dtype = getattr(op, "acc_dtype", None) or output_dtype
            scalar_type = get_scalar_type(accumulator_dtype)

            def kernel():
                return self.scalar_kernel(
                    op.scalar_op, op.scalar_op.make_node(scalar_type(), scalar_type())
                )

            identity = self.constant(
                _reduce_identity(op.scalar_op.identity, accumulator_dtype), accumulator_dtype
            )
            accumulators = {}
            for output_index in np.ndindex(shape_of(node.outputs[0])):
                accumulators[output_index] = self.fresh("accumulator")
                self.lines.append(f"{accumulators[output_index]} = {identity}")
            for input_index in np.ndindex(shape_of(node.inputs[0])):
                output_index = tuple(i for axis, i in enumerate(input_index) if axis not in axes)
                member = inputs[0][input_index]
                if input_dtype != accumulator_dtype:
                    member = f"np.{accumulator_dtype}({member})"
                self.reduce(op.scalar_op, accumulators[output_index], member, kernel)
            elements = {}
            for output_index, accumulator in accumulators.items():
                if np.dtype(accumulator_dtype).kind == "c" and np.dtype(output_dtype).kind != "c":
                    # Numba has no direct cast from complex to real.
                    accumulator = f"{accumulator}.real"
                if accumulator_dtype != output_dtype:
                    accumulator = f"np.{output_dtype}({accumulator})"
                elements[output_index] = accumulator
            return [elements]

        if isinstance(op, SUBTENSOR_OPS):
            x_shape = shape_of(node.inputs[0])
            x_flat = [inputs[0][index] for index in np.ndindex(x_shape)]
            positions = element_positions(x_shape, static_index(node))
            return [{index: x_flat[positions[index]] for index in np.ndindex(positions.shape)}]

        if isinstance(op, INC_SUBTENSOR_OPS):
            x_shape, y_shape = shape_of(node.inputs[0]), shape_of(node.inputs[1])
            x_flat = [inputs[0][index] for index in np.ndindex(x_shape)]
            y_flat = [inputs[1][index] for index in np.ndindex(y_shape)]
            positions = element_positions(x_shape, static_index(node))
            y_positions = np.broadcast_to(np.arange(len(y_flat)).reshape(y_shape), positions.shape)
            # A plain increment accumulates repeated positions, one that ignores duplicates keeps the last write.
            accumulate = not getattr(op, "ignore_duplicates", False)
            result = list(x_flat)
            for index in np.ndindex(positions.shape):
                position, value = positions[index], y_flat[y_positions[index]]
                if op.set_instead_of_inc:
                    result[position] = value
                else:
                    name = self.fresh()
                    base = result[position] if accumulate else x_flat[position]
                    self.lines.append(
                        f"{name} = {cast(f'{base} + {value}', node.outputs[0].dtype)}"
                    )
                    result[position] = name
            return [{index: result[i] for i, index in enumerate(np.ndindex(x_shape))}]

        raise NotImplementedError(f"No scalar translation for {type(op).__name__}")

    def source(self, fgraph, name):
        """Source of a function taking ``fgraph``'s inputs and returning its outputs, as arrays or scalars."""
        arguments = [f"x{j}" for j in range(len(fgraph.inputs))]
        input_elements = []
        for argument, var in zip(arguments, fgraph.inputs, strict=True):
            if isinstance(var.type, ScalarType):
                input_elements.append({(): argument})
                continue
            var_elements = {}
            for index in np.ndindex(shape_of(var)):
                local = self.fresh("load")
                self.lines.append(
                    f"{local} = {argument}[{create_tuple_string(list(map(str, index)))}]"
                )
                var_elements[index] = local
            input_elements.append(var_elements)

        returned = []
        for var, elements in zip(
            fgraph.outputs, self.translate_graph(fgraph, input_elements), strict=True
        ):
            if isinstance(var.type, ScalarType):
                returned.append(elements[()])
                continue
            array = self.fresh("out")
            self.lines.append(f"{array} = np.empty({var.type.shape}, dtype=np.{var.type.dtype})")
            self.lines += [
                f"{array}[{create_tuple_string(list(map(str, index)))}] = {local}"
                for index, local in elements.items()
            ]
            returned.append(array)

        result = returned[0] if len(returned) == 1 else create_tuple_string(returned)
        return build_source_code(
            [
                f"def {name}({', '.join(arguments)}):",
                CODE_TOKEN.INDENT,
                *self.lines,
                f"return {result}",
                CODE_TOKEN.DEDENT,
            ]
        )


@node_rewriter([Scan])
def scan_step_to_scalarized_step(fgraph, node):
    """Wrap the inner graph of a scan whose step has small static shapes in a :class:`ScalarizedStep`."""
    op = node.op
    inner = op.fgraph
    if any(isinstance(inner_node.op, ScalarizedStep) for inner_node in inner.apply_nodes):
        return None
    operations = graph_operation_count(inner)
    if operations is None or operations > MAX_SCALARIZED_OPERATIONS:
        return None

    step = ScalarizedStep(list(inner.inputs), list(inner.outputs), fgraph=inner)
    inner_inputs = [var.type() for var in inner.inputs]
    new_op = Scan(
        inner_inputs,
        step(*inner_inputs, return_list=True),
        op.info,
        mode=op.mode,
        truncate_gradient=op.truncate_gradient,
        name=op.name,
        profile=op.profile,
        allow_gc=op.allow_gc,
        strict=op.strict,
    )
    new_op.destroy_map = op.destroy_map
    new_op.view_map = op.view_map
    outputs = new_op(*node.inputs, return_list=True)
    copy_stack_trace(node.outputs, outputs)
    return outputs


optdb.register(
    scan_step_to_scalarized_step.__name__,
    dfs_rewriter(scan_step_to_scalarized_step),
    "numba",
    # After the scan inner graphs are baked for the backend at position 100.
    position=100.5,
)


@register_funcify_and_cache_key(ScalarizedStep)
def numba_funcify_ScalarizedStep(op, node, ofg_memo=None, **kwargs):
    _ = kwargs.pop("storage_map", None)

    if ofg_memo is not None and op in ofg_memo:
        return ofg_memo[op]

    scalarizer = StepScalarizer(ofg_memo=ofg_memo, **kwargs)
    source = scalarizer.source(op.fgraph, "scalarized_step")
    scalarized_step = compile_numba_function_src(source, "scalarized_step", scalarizer.env)

    if any(key is None for key in scalarizer.kernel_cache_keys):
        cache_key = None
    else:
        cache_version = 2
        cache_key = sha256(
            str(
                (
                    type(op),
                    source,
                    scalarizer.kernel_cache_keys,
                    # Constants live in the function's globals, not its source.
                    sorted(scalarizer.constants.items()),
                    cache_version,
                )
            ).encode()
        ).hexdigest()

    # Every subscript is a literal derived from static shapes, so it is in bounds.
    result = numba_basic.numba_njit(scalarized_step, boundscheck=False), cache_key
    if ofg_memo is not None:
        ofg_memo[op] = result
    return result
