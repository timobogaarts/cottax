'''
scipy's SLSQP answering a `ConstrainedMinimiser`: host callback.
'''

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from .....core import Place
from .....core.frozen import Frozen, choice
from .....core.driven import Start, Steps
from .....relational.contracts import ConstrainedMinimiser
from .....consumers.execution.interface import DriverImplementation
from .....consumers.execution.interface_types import ByRole, Context, Posed, Unknowns
from .canonical_nlp import Canonical, box, canonical


type Unravel = Callable[[jax.Array], tuple[Any, ...]]


class _Flat[V: Place](Frozen):
    '''The posed problem over one flat vector of unknowns, as scipy sees it. Static: the
    compiled programs key on it.'''

    posed   : Posed[V]
    unravel : Unravel
    '''The flat vector back into the unknowns.'''

    def groups(self, context: Context, x: jax.Array) -> Canonical:
        '''The canonical form at `x`, each group stacked into one array of scalars.'''
        return self.evaluated(context, x)[0]

    def evaluated(self, context: Context, x: jax.Array) -> tuple[Canonical, jax.Array]:
        '''The groups at `x`, and the box parallel to `x` as `(2, n)`: lower then upper,
        `-inf` / `inf` where unbounded. One walk of the model for both.'''
        unknowns = self.unravel(x)
        values   = self.posed.conditions(unknowns, context)
        form     = canonical(self.posed.condition, unknowns, values)
        lower, upper = box(self.posed.condition, unknowns, values)
        over = lambda bounds, free: ravel_pytree(tuple(
            jnp.broadcast_to(jnp.asarray(free if b is None else b, jnp.result_type(u)),
                             jnp.shape(u))
            for b, u in zip(bounds, unknowns, strict=True)))[0]
        return (Canonical(*(jnp.stack([jnp.asarray(v) for v in group]) if group
                            else jnp.zeros(0) for group in form)),
                jnp.stack([over(lower, -jnp.inf), over(upper, jnp.inf)]))

    def vector(self, context: Context, x: jax.Array) -> jax.Array:
        '''The groups as one vector: what a Jacobian transform differentiates.'''
        return jnp.concatenate(self.groups(context, x))


@functools.partial(jax.jit, static_argnums=0)
def _values(flat: _Flat, context: Context, x: jax.Array) -> tuple[Canonical, jax.Array]:
    return flat.evaluated(context, x)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _jacobian(transform: Callable, flat: _Flat, context: Context, x: jax.Array) -> jax.Array:
    '''One row per entry of `flat.vector`, in its order.'''
    return transform(functools.partial(flat.vector, context))(x)


class _Memo:
    '''Caching class for SciPy.'''

    def __init__(self, values: Callable, jacobian: Callable, dtype: Any, device: Any) -> None:
        self.computed = (values, jacobian)
        self.dtype, self.device = dtype, device
        self.ends : list[int] | None = None
        self.box  : np.ndarray | None = None
        '''`(lower, upper)` over `x`, from the first point evaluated: the box is fixed.'''
        self._values_by_key   = functools.lru_cache(maxsize=1)(self._values)
        self._jacobian_by_key = functools.lru_cache(maxsize=1)(self._jacobian)

    def values_at(self, x: np.ndarray) -> Canonical:
        return self._values_by_key(np.asarray(x, float).tobytes())

    def jacobian_at(self, x: np.ndarray) -> Canonical:
        return self._jacobian_by_key(np.asarray(x, float).tobytes())

    def _run(self, kind: int, key: bytes) -> Any:
        x = np.frombuffer(key).astype(self.dtype)
        return self.computed[kind](jax.device_put(x, self.device))

    def _values(self, key: bytes) -> Canonical:
        groups, limits = self._run(0, key)
        values = Canonical(*(np.asarray(group, float) for group in groups))
        if self.box is None:
            self.box = np.asarray(limits, float)
        self.ends = list(np.cumsum([group.size for group in values])[:-1])
        return values

    def _jacobian(self, key: bytes) -> Canonical:
        if self.ends is None:
            self._values_by_key(key)
        return Canonical(*np.split(np.asarray(self._run(1, key), float), self.ends))


@choice(DriverImplementation)
class SLSQP(DriverImplementation[ConstrainedMinimiser]):
    '''`scipy.optimize.minimize(method='SLSQP')` on the canonical form: the objective,
    the equalities as `eq`, the inequalities negated as `ineq` (scipy's `>= 0`), and the
    box as scipy's `bounds`, so every iterate stays inside it.'''

    needs    = frozenset({Start})
    provides = frozenset({Steps})

    tol       : float = 1e-8
    max_steps : int   = 200
    strict    : bool  = True
    '''Raise when scipy does not report success; off, the answer is taken as it stands.'''
    jacobian  : Callable = jax.jacfwd
    '''How the Jacobian is taken: `jax.jacfwd`, `jax.jacrev`, or any hashable function of
    a function, such as one pulling rows back one at a time to save memory.'''

    def __call__[V: Place](self, posed: Posed[V], context: Context,
                           data: ByRole) -> tuple[Unknowns, ByRole]:
        start, unravel = ravel_pytree(data[Start])
        host = functools.partial(_minimise, self, _Flat(posed, unravel),
                                 jax.devices()[0])    # asked here: in the callback it is the CPU
        if any(isinstance(leaf, jax.core.Tracer)
               for leaf in jax.tree.leaves((context, start))):
            answer, steps = jax.pure_callback(
                host, (jax.ShapeDtypeStruct(start.shape, start.dtype),
                       jax.ShapeDtypeStruct((), jax.dtypes.canonicalize_dtype(int))),
                context, start, vmap_method='sequential')
        else:
            answer, steps = map(jnp.asarray, host(context, start))
        return tuple(unravel(answer)), ({Steps: (steps,)} if Steps in posed.reports else {})


def _objective(x: np.ndarray, memo: _Memo) -> float:
    (objective,) = memo.values_at(x).objectives
    return float(objective)


def _gradient(x: np.ndarray, memo: _Memo) -> np.ndarray:
    (gradient,) = memo.jacobian_at(x).objectives
    return gradient


def _group(x: np.ndarray, memo: _Memo, group: str, sign: float) -> np.ndarray:
    return sign * getattr(memo.values_at(x), group)


def _group_jacobian(x: np.ndarray, memo: _Memo, group: str, sign: float) -> np.ndarray:
    return sign * getattr(memo.jacobian_at(x), group)


SCIPY = (('eq', 'equalities', 1.0), ('ineq', 'inequalities', -1.0))
'''Each constraint group as scipy's kind, and its sign: `g <= 0` is scipy's `-g >= 0`.'''


def _minimise(slsqp: SLSQP, flat: _Flat, device: Any, context: Context,
              start: Any) -> tuple[np.ndarray, np.ndarray]:
    '''One scipy solve on the host: `(x, nit)`, `x` in the start's dtype, the model
    evaluated on `device`.'''
    dtype, x0 = np.asarray(start).dtype, np.asarray(start, float)
    context = jax.device_put(context, device)
    memo = _Memo(functools.partial(_values, flat, context),
                 functools.partial(_jacobian, slsqp.jacobian, flat, context), dtype, device)
    at_start = memo.values_at(x0)
    constraints = [{'type': kind, 'fun': _group, 'jac': _group_jacobian,
                    'args': (memo, group, sign)}
                   for kind, group, sign in SCIPY if getattr(at_start, group).size]
    bounds = (None if not np.isfinite(memo.box).any() else
              [tuple(None if not np.isfinite(b) else float(b) for b in pair)
               for pair in memo.box.T])

    result = minimize(_objective, x0, args=(memo,), jac=_gradient, method='SLSQP',
                      bounds=bounds, constraints=constraints, tol=slsqp.tol,
                      options={'maxiter': slsqp.max_steps})
    if slsqp.strict and not result.success:
        raise RuntimeError(f'SLSQP did not converge: {result.message}')
    return (np.asarray(result.x).astype(dtype),
            np.asarray(result.nit, jax.dtypes.canonicalize_dtype(int)))
