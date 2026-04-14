import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from pymc.pytensorf import rewrite_pregrad
from pymc.sampling.jax import get_jaxified_graph
from pymc.util import get_default_varnames

from pymc_extras.inference.laplace_approx.idata import (
    add_data_to_inference_data,
    map_results_to_inference_data,
)

# Register PyTensor → MLX dispatch entries that are missing upstream. The
# import-and-register is guarded so importing this module does not require mlx.
try:
    from functools import reduce as _reduce

    import mlx.core as _mx

    from pytensor.link.mlx.dispatch import mlx_funcify as _mlx_funcify
    from pytensor.scalar.basic import Add as _Add
    from pytensor.scalar.basic import Clip as _Clip
    from pytensor.scalar.basic import Mul as _Mul
    from pytensor.tensor.math import Dot as _Dot

    def _stream(*xs):
        """Pick the MLX stream: CPU if any input is float64 (GPU is float32-only)."""
        for x in xs:
            if getattr(x, "dtype", None) == _mx.float64:
                return _mx.cpu
        return None  # default device

    @_mlx_funcify.register(_Clip)
    def _mlx_funcify_Clip(op, **kwargs):
        def clip(x, lo, hi):
            return _mx.clip(x, lo, hi, stream=_stream(x, lo, hi))

        return clip

    # Override the generic ScalarOp variadic dispatch for Add/Mul: upstream
    # stacks all args along a new axis and reduces, which fails on mixed-shape
    # inputs (e.g. a 0-d Python scalar mixed with 0-d arrays inside a fused
    # Composite). MLX has no variadic add/multiply, so we fold pairwise.
    @_mlx_funcify.register(_Add)
    def _mlx_funcify_Add(op, **kwargs):
        return lambda *args: _reduce(lambda a, b: _mx.add(a, b, stream=_stream(a, b)), args)

    @_mlx_funcify.register(_Mul)
    def _mlx_funcify_Mul(op, **kwargs):
        return lambda *args: _reduce(lambda a, b: _mx.multiply(a, b, stream=_stream(a, b)), args)

    @_mlx_funcify.register(_Dot)
    def _mlx_funcify_Dot(op, **kwargs):
        return lambda x, y: _mx.matmul(x, y, stream=_stream(x, y))

except ImportError:
    pass


def _coerce_xy(X, y, input_dim, asarray):
    """Coerce X to (n, input_dim) and y to (n, 1)."""
    X = asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[1] != input_dim:
        raise ValueError(f"X must be shape (n, {input_dim}); got {tuple(X.shape)}.")
    y = asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"y must be shape (n,) or (n, 1); got {tuple(y.shape)}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y disagree on n: X is {tuple(X.shape)}, y is {tuple(y.shape)}.")
    return X, y


def _fit_results_to_idata(model, svgp, value_var_point, loss_history, grad_norm_history=None):
    """Package SVGP fit results into an ``arviz.InferenceData``.

    - ``posterior`` group: user-defined model parameters only (sigma, eta, ell, …).
      SVGP internals (inducing points, variational params) are excluded.
    - ``unconstrained_posterior`` group: transformed (value-var-space) values for
      the user-defined params.
    - ``observed_data`` and ``constant_data``: pulled from the model.
    - ``fit`` group: per-step ``loss_history``, ``grad_norm``, plus the SVGP
      internal state (``inducing_points``, ``variational_mean``,
      ``variational_cholesky``) needed to reconstruct predictions.
    """
    point = {k: np.asarray(v) for k, v in value_var_point.items()}
    internal_names = svgp._internal_rv_names

    unobserved_vars = get_default_varnames(model.unobserved_value_vars, include_transformed=True)
    f_unobs = model.compile_fn(unobserved_vars, mode="FAST_COMPILE", point_fn=True)
    values = f_unobs(point)
    optimized_point = {v.name: val for v, val in zip(unobserved_vars, values)}

    # Split: user params go to posterior; SVGP internals go to fit group.
    user_point = {k: v for k, v in optimized_point.items() if k not in internal_names}

    idata = map_results_to_inference_data(user_point, model=model, include_transformed=True)
    idata = add_data_to_inference_data(idata, progressbar=False, model=model)

    fit_data = {
        "loss_history": xr.DataArray(np.asarray(loss_history), dims=["step"]),
    }
    if grad_norm_history is not None:
        fit_data["grad_norm"] = xr.DataArray(np.asarray(grad_norm_history), dims=["step"])
    for name in internal_names:
        if name in optimized_point:
            arr = np.asarray(optimized_point[name])
            dims = [f"{name}_dim_{i}" for i in range(arr.ndim)]
            fit_data[name] = xr.DataArray(arr, dims=dims)
    idata.add_groups(fit=xr.Dataset(fit_data))
    return idata


def _make_rich_progress(desc, *, disable=False):
    """Return a PyMC-style ``ToggleableProgress`` table for fit loops.

    The table uses column headers (``show_header=True``) and a ``SIMPLE_HEAD``
    box style, matching PyMC's ``pm.sample`` / ``better-optimize`` progress bars.
    """
    from collections.abc import Iterable

    from rich.box import SIMPLE_HEAD
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Column, Table
    from rich.text import Text

    class _FieldColumn(TextColumn):
        """Render a numeric task field with a column header."""

        def __init__(self, field, header, fmt=".3f", style="cyan"):
            super().__init__("", table_column=Column(header=header))
            self._field = field
            self._fmt = fmt
            self._style = style

        def render(self, task):
            val = task.fields.get(self._field)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                return Text("—")
            return Text(f"{val:{self._fmt}}", style=self._style)

    class _StepColumn(TextColumn):
        """Show current step (completed count) with a 'Step' header."""

        def __init__(self):
            super().__init__("", table_column=Column(header="Step"))

        def render(self, task):
            return Text(str(int(task.completed)))

    class ToggleableProgress(Progress):
        def __init__(self, *args, **kwargs):
            self.is_enabled = kwargs.pop("disable", None) is not True
            if self.is_enabled:
                super().__init__(*args, **kwargs)

        def __enter__(self):
            if self.is_enabled:
                self.start()
            return self

        def __exit__(self, *args):
            if self.is_enabled:
                super().__exit__(*args)

        def add_task(self, *args, **kwargs):
            if self.is_enabled:
                return super().add_task(*args, **kwargs)
            return None

        def advance(self, task_id, advance=1):
            if self.is_enabled:
                super().advance(task_id, advance)

        def update(self, task_id, **kwargs):
            if self.is_enabled:
                super().update(task_id, **kwargs)

        def make_tasks_table(self, tasks: Iterable) -> Table:
            table_columns = (
                (Column(no_wrap=True) if isinstance(c, str) else c.get_table_column().copy())
                for c in self.columns
            )
            table = Table(
                *table_columns,
                padding=(0, 1),
                expand=self.expand,
                show_header=True,
                show_edge=True,
                box=SIMPLE_HEAD,
            )
            for task in tasks:
                if task.visible:
                    table.add_row(
                        *(
                            c.format(task=task) if isinstance(c, str) else c(task)
                            for c in self.columns
                        )
                    )
            return table

    return ToggleableProgress(
        BarColumn(),
        _StepColumn(),
        _FieldColumn("loss", header="Loss"),
        _FieldColumn("grad_norm", header="|Grad|"),
        TimeRemainingColumn(table_column=Column(header="Remaining")),
        TimeElapsedColumn(table_column=Column(header="Elapsed")),
        disable=disable,
    )


def _scan_progress_bar(n_steps, print_rate=None, desc="fit_jax"):
    """Decorator that adds a Rich progress bar to a ``jax.lax.scan`` body.

    Adapted from BlackJAX's progress bar pattern: a single scan with
    ``jax.experimental.io_callback`` pushing updates out to a host-side Rich
    bar at ``print_rate``-step intervals (and once at the end).

    The decorated body must take a scan input that is either an ``iter_num``
    (int) or a tuple whose first element is ``iter_num``.
    """
    if print_rate is None:
        print_rate = max(1, n_steps // 100)

    state = {"progress": None, "task": None}

    def _update(iter_num, loss_val, grad_norm):
        if state["progress"] is None:
            state["progress"] = _make_rich_progress(desc)
            state["progress"].start()
            state["task"] = state["progress"].add_task(
                desc, total=int(n_steps), loss=float(loss_val), grad_norm=float(grad_norm)
            )
        state["progress"].update(
            state["task"],
            completed=int(iter_num) + 1,
            loss=float(loss_val),
            grad_norm=float(grad_norm),
        )

    def stop():
        if state["progress"] is not None:
            state["progress"].stop()
            state["progress"] = None
            state["task"] = None

    def decorator(body):
        def wrapped(carry, scan_input):
            iter_num = scan_input[0] if isinstance(scan_input, tuple) else scan_input
            new_carry, (loss_val, grad_norm) = body(carry, scan_input)

            jax.lax.cond(
                (iter_num % print_rate == 0) | (iter_num == n_steps - 1),
                lambda: jax.experimental.io_callback(_update, None, iter_num, loss_val, grad_norm),
                lambda: None,
            )
            return new_carry, (loss_val, grad_norm)

        wrapped.stop = stop
        return wrapped

    return decorator


def fit_jax(
    svgp,
    X_data,
    y_data,
    optimizer,
    n_steps,
    *,
    batch_size,
    model=None,
    init_params=None,
    seed=0,
    progress=True,
    print_rate=None,
):
    """Fit an :class:`SVGP` (or :class:`WhitenedSVGP`) by stochastic ELBO maximization.

    The ELBO graph is built from ``svgp.elbo(X_batch, y_batch)``, jaxified, and
    optimized via a single ``jax.lax.scan`` over minibatches sampled with
    replacement. When ``progress=True`` a tqdm bar updates from inside the scan
    via ``jax.experimental.io_callback`` (BlackJAX-style).

    Parameters
    ----------
    svgp : SVGP or WhitenedSVGP
    X_data : array, shape (n_data, input_dim)
    y_data : array, shape (n_data, 1)
    optimizer : optax.GradientTransformation
    n_steps : int
    batch_size : int
    model : pm.Model, optional
        Defaults to the active model context.
    init_params : tuple of arrays, optional
        Initial values for the model's continuous value variables, in the order
        of ``model.continuous_value_vars``. Defaults to ``model.initial_point()``.
    seed : int
        Seed for the JAX PRNG used to sample minibatches.
    progress : bool
        Show a tqdm bar with the latest loss in the postfix.
    print_rate : int, optional
        Steps between bar updates. Defaults to ``max(1, n_steps // 100)``.

    Returns
    -------
    idata : az.InferenceData
        ``posterior`` (constrained), ``unconstrained_posterior`` (value-var
        space), ``observed_data``, ``constant_data``, plus a ``fit`` group with
        the per-step ``loss_history``.
    """
    model = pm.modelcontext(model)

    X_data, y_data = _coerce_xy(X_data, y_data, svgp.input_dim, jnp.asarray)
    n_data = X_data.shape[0]

    X_batch = pt.tensor("X_batch", shape=(batch_size, svgp.input_dim))
    y_batch = pt.tensor("y_batch", shape=(batch_size, 1))

    loss = -svgp.elbo(X_batch, y_batch, n_data)
    [loss_v] = model.replace_rvs_by_values([loss])

    value_vars = model.continuous_value_vars
    var_names = [v.name for v in value_vars]

    f_loss_jax = get_jaxified_graph([X_batch, y_batch, *value_vars], outputs=[loss_v])

    def f_loss(params, X, y):
        return f_loss_jax(X, y, *params)[0]

    if init_params is None:
        point = model.initial_point()
        init_params = tuple(jnp.asarray(point[name]) for name in var_names)
    else:
        init_params = tuple(jnp.asarray(p) for p in init_params)

    opt_state = optimizer.init(init_params)
    keys = jr.split(jr.PRNGKey(seed), n_steps)

    def _grad_norm(grads):
        return jnp.sqrt(sum(jnp.sum(g**2) for g in grads))

    def step(carry, scan_input):
        # scan_input is (iter_num, key) when progress=True, else just key
        key = scan_input[1] if isinstance(scan_input, tuple) else scan_input
        params, opt_state = carry
        idx = jr.choice(key, n_data, (batch_size,), replace=True)
        X_b = X_data[idx]
        y_b = y_data[idx]
        loss_val, grads = jax.value_and_grad(f_loss)(params, X_b, y_b)
        grad_norm = _grad_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (loss_val, grad_norm)

    if progress:
        body = _scan_progress_bar(n_steps, print_rate=print_rate)(step)
        scan_inputs = (jnp.arange(n_steps), keys)
    else:
        body = step
        scan_inputs = keys

    (final_params, _), (loss_history, grad_norm_history) = jax.lax.scan(
        body, (init_params, opt_state), scan_inputs
    )
    if hasattr(body, "stop"):
        body.stop()

    value_var_point = dict(zip(var_names, final_params))
    return _fit_results_to_idata(model, svgp, value_var_point, loss_history, grad_norm_history)


def fit_mlx(
    svgp,
    X_data,
    y_data,
    optimizer,
    n_steps,
    *,
    batch_size,
    model=None,
    init_params=None,
    seed=0,
    progress=True,
    print_rate=None,
):
    """Fit an :class:`SVGP` (or :class:`WhitenedSVGP`) using PyTensor's MLX backend.

    The ELBO and its gradient are constructed symbolically with PyTensor (so
    autodiff is PyTensor's, not MLX's), then compiled via ``pytensor.function(
    ..., mode="MLX")``. The training loop is a plain Python loop using one of
    Apple's :mod:`mlx.optimizers` for the parameter update.

    All linear-algebra ops (Cholesky, triangular solve) currently fall back to
    CPU in PyTensor's MLX backend; matmul/elementwise/reductions run on the
    Apple GPU. For typical inducing-set sizes (a few hundred) the CPU linalg is
    not the bottleneck.

    Parameters
    ----------
    svgp : SVGP or WhitenedSVGP
    X_data : array, shape (n_data, input_dim)
    y_data : array, shape (n_data, 1)
    optimizer : mlx.optimizers.Optimizer
    n_steps : int
    batch_size : int
    model : pm.Model, optional
        Defaults to the active model context.
    init_params : tuple of arrays, optional
        Initial values for the model's continuous value variables, in the order
        of ``model.continuous_value_vars``. Defaults to ``model.initial_point()``.
    seed : int
        Seed for the NumPy PRNG used to sample minibatches.
    progress : bool
        Show a Rich progress bar with the latest loss in the postfix.
    print_rate : int, optional
        Steps between bar updates. Defaults to ``max(1, n_steps // 100)``.

    Returns
    -------
    idata : az.InferenceData
        ``posterior`` (constrained), ``unconstrained_posterior`` (value-var
        space), ``observed_data``, ``constant_data``, plus a ``fit`` group with
        the per-step ``loss_history``.
    """
    try:
        import mlx.core as mx
    except ImportError as e:
        raise ImportError(
            "fit_mlx requires the `mlx` package. Install with `pip install mlx`."
        ) from e

    model = pm.modelcontext(model)

    X_data, y_data = _coerce_xy(X_data, y_data, svgp.input_dim, np.asarray)
    n_data = X_data.shape[0]

    X_batch = pt.tensor("X_batch", shape=(batch_size, svgp.input_dim))
    y_batch = pt.tensor("y_batch", shape=(batch_size, 1))

    loss = -svgp.elbo(X_batch, y_batch, n_data)
    [loss_v] = model.replace_rvs_by_values([loss])
    loss_v = rewrite_pregrad(loss_v)

    value_vars = list(model.continuous_value_vars)
    var_names = [v.name for v in value_vars]
    grads = pt.grad(loss_v, value_vars)

    f_value_and_grad = pytensor.function(
        [X_batch, y_batch, *value_vars],
        [loss_v, *grads],
        mode="MLX",
    )

    if init_params is None:
        point = model.initial_point()
        init_params = [point[name] for name in var_names]

    params = {name: mx.array(np.asarray(p)) for name, p in zip(var_names, init_params)}
    optimizer.init(params)

    rng = np.random.default_rng(seed)
    loss_history = np.empty(n_steps, dtype=np.float64)
    grad_norm_history = np.empty(n_steps, dtype=np.float64)

    if print_rate is None:
        print_rate = max(1, n_steps // 100)

    progress_obj = _make_rich_progress("fit_mlx") if progress else None
    task = None
    if progress_obj is not None:
        progress_obj.start()
        task = progress_obj.add_task(
            "fit_mlx", total=n_steps, loss=float("nan"), grad_norm=float("nan")
        )

    try:
        for step in range(n_steps):
            idx = rng.choice(n_data, batch_size, replace=True)
            X_b = X_data[idx]
            y_b = y_data[idx]
            loss_val, *grad_vals = f_value_and_grad(X_b, y_b, *(params[name] for name in var_names))
            gn = float(np.linalg.norm(np.concatenate([np.asarray(g).ravel() for g in grad_vals])))
            grads_dict = dict(zip(var_names, grad_vals))
            params = optimizer.apply_gradients(grads_dict, params)
            loss_history[step] = float(loss_val)
            grad_norm_history[step] = gn

            if progress_obj is not None and (step % print_rate == 0 or step == n_steps - 1):
                progress_obj.update(task, completed=step + 1, loss=loss_history[step], grad_norm=gn)
    finally:
        if progress_obj is not None:
            progress_obj.stop()

    value_var_point = {name: np.asarray(params[name]) for name in var_names}
    return _fit_results_to_idata(model, svgp, value_var_point, loss_history, grad_norm_history)
