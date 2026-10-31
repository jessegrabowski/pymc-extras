import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.compile.mode import get_mode
from pytensor.scan.op import Scan

from pymc_extras.statespace.filters.scalarized_step import ScalarizedStep

NUMBA_MODE = "NUMBA"
REFERENCE_MODE = get_mode(NUMBA_MODE).excluding("scan_step_to_scalarized_step")


def scalarized_scans(fn):
    return [
        all(isinstance(inner.op, ScalarizedStep) for inner in node.op.fgraph.apply_nodes)
        for node in fn.maker.fgraph.apply_nodes
        if isinstance(node.op, Scan)
    ]


def compile_scan_and_gradient(step, x0, mode):
    trajectory = pytensor.scan(
        lambda x: pt.tanh(step(x)), outputs_info=[x0], n_steps=4, return_updates=False
    )
    cost = trajectory.sum()
    return pytensor.function([x0], [cost, pytensor.grad(cost, x0)], mode=mode)


@pytest.mark.parametrize(
    "step, dtype",
    [
        pytest.param(
            lambda x: x @ x.mT + x[1:, ::2].sum() * x[0, 2], "float64", id="dot and basic read"
        ),
        pytest.param(
            lambda x: x + x[[2, 0, 2]].sum(0) + x[np.array([True, False, True])].sum(),
            "float64",
            id="advanced reads",
        ),
        pytest.param(
            lambda x: pt.inc_subtensor(pt.set_subtensor(x[0, 1:], x[2, :2] * 2.0)[:, 0], x[1]),
            "float64",
            id="basic set and inc",
        ),
        pytest.param(
            lambda x: pt.set_subtensor(x[[0, 2], [1, 1]], x[1, :2]), "float64", id="advanced set"
        ),
        pytest.param(
            lambda x: pt.inc_subtensor(x[[0, 0, 2]], x[1] * 0.1), "float64", id="advanced inc"
        ),
        pytest.param(
            lambda x: pt.inc_subtensor(x[[0, 0, 2]], x[1] * 0.1, ignore_duplicates=True),
            "float64",
            id="advanced inc ignoring duplicates",
        ),
        pytest.param(
            lambda x: x * 0.5 + x.sum(axis=0) - x.max(axis=1, keepdims=True),
            "float32",
            id="float32 axis reductions",
        ),
    ],
)
def test_scalarized_scan_matches_reference(step, dtype):
    x0 = pt.matrix("x0", shape=(3, 3), dtype=dtype)
    fn = compile_scan_and_gradient(step, x0, NUMBA_MODE)
    assert scalarized_scans(fn) == [True, True]

    x_value = (np.arange(9.0).reshape(3, 3) / 10).astype(dtype)
    reference = compile_scan_and_gradient(step, x0, REFERENCE_MODE)
    rtol = 1e-12 if dtype == "float64" else 1e-6
    for actual, expected in zip(fn(x_value), reference(x_value), strict=True):
        assert actual.dtype == expected.dtype
        assert_allclose(actual, expected, rtol=rtol)


def test_zero_times_nan_stays_nan():
    zeros_on_left = pt.constant(np.array([[0.0, 1.0], [0.0, 0.0]]))
    x0 = pt.matrix("x0", shape=(2, 2))
    trajectory = pytensor.scan(
        lambda x: zeros_on_left @ x + 1.0, outputs_info=[x0], n_steps=2, return_updates=False
    )
    fn = pytensor.function([x0], trajectory[-1], mode=NUMBA_MODE)
    assert scalarized_scans(fn) == [True]

    # The second row of the second step is 0 * nan + 1, so it is nan only if the zero product is kept.
    assert_allclose(fn(np.array([[np.nan, 1.0], [2.0, 3.0]])), [[np.nan, 2.0], [np.nan, 1.0]])


@pytest.mark.parametrize(
    "shape, step",
    [
        pytest.param((None, 3), pt.tanh, id="dynamic shape"),
        pytest.param((9, 9), pt.tanh, id="too large"),
        pytest.param(
            (3, 3), lambda x: pt.linalg.cholesky(x @ x.mT + pt.eye(3)), id="unsupported op"
        ),
    ],
)
def test_scan_is_not_scalarized(shape, step):
    x0 = pt.matrix("x0", shape=shape)
    trajectory = pytensor.scan(step, outputs_info=[x0], n_steps=4, return_updates=False)
    fn = pytensor.function([x0], trajectory[-1], mode=NUMBA_MODE)

    assert scalarized_scans(fn) == [False]
