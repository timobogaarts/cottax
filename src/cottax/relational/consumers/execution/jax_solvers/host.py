'''
A solver that runs on the host, as a `HostMinimiser`: all it writes is `minimise`, handed a
`HostProblem`, plain numpy in the solver's own units. The rest is here: the `Scaling`, one flat
vector of unknowns, the compiled programs, one evaluation per point, `pure_callback` under a
trace, and the reports.
'''

import functools
from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .....core import Place
from .....core.driven import DriverContract, Role, Start
from .....core.frozen import AbstractClassVar
from .....relational import Scaling
from .....relational.contracts import Scales
from .....consumers.execution.interface import DriverImplementation
from .....consumers.execution.interface_types import ByRole, Context, Posed, Unknowns
from .canonical_nlp import Canonical, scaled, scaled_unknowns, unscaled_unknowns


@eqx.filter_jit
def _values(posed: Posed, unravel: Callable, context: Context, scaling: Scaling, y: jax.Array
            ) -> tuple[Canonical, jax.Array]:
    '''The groups at `y`, each one vector, and the box parallel to `y` as `(2, n)`: lower then
    upper, `-inf` / `inf` where unbounded, all read in `scaling`, which `y` is in too. One walk
    of the model for both.'''
    solved   = unravel(y)
    unknowns = unscaled_unknowns(posed.condition, solved, scaling)
    form, (lower, upper) = scaled(posed.condition, unknowns, posed.conditions(unknowns, context),
                                  scaling)
    over = lambda bounds, free: ravel_pytree(tuple(
        jnp.broadcast_to(jnp.asarray(free if b is None else b, jnp.result_type(u)), jnp.shape(u))
        for b, u in zip(bounds, solved, strict=True)))[0]
    return (Canonical(*(ravel_pytree(group)[0] for group in form)),
            jnp.stack([over(lower, -jnp.inf), over(upper, jnp.inf)]))


@eqx.filter_jit
def _jacobian(transform: Callable, posed: Posed, unravel: Callable, context: Context,
              scaling: Scaling, y: jax.Array) -> tuple[Canonical, jax.Array, jax.Array]:
    '''`_values` and one Jacobian row per entry of the groups, in order, off one trace.'''
    def vector(y: jax.Array) -> tuple[jax.Array, tuple[Canonical, jax.Array]]:
        groups, limits = _values(posed, unravel, context, scaling, y)
        return jnp.concatenate(groups), (groups, limits)

    jacobian, (groups, limits) = transform(vector, has_aux=True)(y)
    return groups, limits, jacobian


def _compiled(program: Callable, static: tuple, fixed: Any, leaves: tuple) -> Callable:
    '''`program` compiled now, from the shapes of `leaves`, which are being traced: nothing is
    running yet. A first compile in the callback can hang on a GPU, as XLA's autotuner times
    its candidates with the GPU to itself, and the graph holds it, waiting on this callback.
    The callback gets each leaf as numpy, so the example is of its dtype and not weakly typed.'''
    arrays, *rest = jax.tree.map(lambda leaf: np.zeros(jnp.shape(leaf), jnp.asarray(leaf).dtype),
                                 leaves)
    return functools.partial(program.lower(*static, eqx.combine(arrays, fixed), *rest).compile(),
                             *static)


def _not_asked(program: str, *_: Any) -> None:
    raise RuntimeError(f'`{program}` is not in this solver\'s `asks`, so it was not compiled ahead')


class HostProblem:
    '''
    The problem as a solver on the host reads it: `min objectives` subject to `equalities = 0`,
    `inequalities <= 0` and `lower <= y <= upper`, all numpy, all in the solver's units.

    `values` and `jacobians` each keep their last point, and a Jacobian's values come with it.
    One evaluation at `y0`, of the program the solver will want (`gradient`), gives `lower`,
    `upper` and the sizes. The box is fixed through a solve.
    '''

    def __init__(self, values: Callable, jacobians: Callable, y0: np.ndarray,
                 magnitudes: np.ndarray | None = None, gradient: bool = True) -> None:
        self._programs  = values, jacobians
        self.y0         = np.asarray(y0, float)
        self.magnitudes = magnitudes
        self._value     : tuple = (None,)
        self._jacobian  : tuple = (None,)
        (self.jacobians if gradient else self.values)(self.y0)

    def values(self, y: np.ndarray) -> Canonical:
        '''The groups at `y`, each one vector. No Jacobian is computed for this.'''
        y = np.asarray(y, float)
        if self._jacobian[0] == y.tobytes():
            return self._jacobian[1]
        if self._value[0] != y.tobytes():
            self._value = y.tobytes(), self._read(*self._programs[0](y))
        return self._value[1]

    def jacobians(self, y: np.ndarray) -> Canonical:
        '''Each group's Jacobian at `y`, one row per entry. The values there come with it.'''
        y = np.asarray(y, float)
        if self._jacobian[0] != y.tobytes():
            groups, limits, jacobian = self._programs[1](y)
            groups = self._read(groups, limits)
            ends = np.cumsum([group.size for group in groups])[:-1]
            self._jacobian = y.tobytes(), groups, Canonical(*np.split(np.asarray(jacobian, float), ends))
        return self._jacobian[2]

    def unknowns(self, y: np.ndarray) -> np.ndarray:
        '''`y` in the model's units, flat: for reporting a point. Plain numpy, at the flat
        vector's precision: nothing compiles or evaluates for it.'''
        return np.asarray(np.asarray(y, self.magnitudes.dtype) * self.magnitudes, float)

    def _read(self, groups: Canonical, limits: Any) -> Canonical:
        groups = Canonical(*(np.asarray(group, float) for group in groups))
        self.lower, self.upper = np.asarray(limits, float)
        self.num_equality, self.num_inequality = groups.equalities.size, groups.inequalities.size
        return groups


class HostMinimiser[D: DriverContract](DriverImplementation[D]):
    '''
    A solver on the host answering a minimisation. An implementation says what it reports, by
    role with its dtype, and solves in `minimise`:

        class Mine(HostMinimiser[ConstrainedMinimiser]):
            reported = {Steps: int}
            provides = frozenset(reported)

            def minimise(self, problem):
                ...
                return y, {Steps: steps}

    Only what a contract asks comes back. Under a contract taking `Scales` the solver sees
    everything read in that `Scaling`.

    A value of the context that is no array, a Python scalar among them, is compile-time: a model
    may branch on it, and another value compiles again. Declaring a value does not convert it,
    so give an array for what changes between solves.

    jax cannot differentiate through a callback, so a solver taking a gradient cannot sit above
    one of these: it fails at the trace. A gradient-free one can.
    '''

    needs    = frozenset({Start})
    reported : AbstractClassVar[Mapping[type[Role], Any]]
    '''Each role `minimise` returns a value for, and the dtype it comes back in.'''
    asks = frozenset({'values', 'jacobians'})
    '''The programs `minimise` calls, compiled with the graph under a trace, so a trace that
    computes nothing, such as `Traced`, compiles them too. `jacobians` gives the values at its
    point as well, so a solver wanting both at every point names it alone.'''
    jacobian = staticmethod(jax.jacfwd)
    '''How a Jacobian is taken: any hashable function of a function, taking `has_aux=True` as
    `jax.jacfwd` and `jax.jacrev` do.'''

    @abstractmethod
    def minimise(self, problem: HostProblem) -> tuple[Any, Mapping[type[Role], Any]]:
        '''The answer as a flat `y`, and a value per role of `reported`. In numpy: under a trace
        this runs in a callback, where on a GPU an eager jax operation can hang as a compile does.'''

    def __call__[V: Place](self, posed: Posed[V], context: Context,
                           data: ByRole) -> tuple[Unknowns, ByRole]:
        (scaling,) = data.get(Scales, (Scaling(),))
        start, unravel = ravel_pytree(scaled_unknowns(posed.condition, tuple(data[Start]), scaling))
        magnitudes, _ = ravel_pytree(unscaled_unknowns(posed.condition, unravel(jnp.ones_like(start)),
                                                       scaling))
        arrays, fixed = eqx.partition(context, eqx.is_array)
        traced = any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves((arrays, scaling, start)))

        def program(name: str, fn: Callable, *static: Any) -> Callable:
            if not traced:
                return functools.partial(fn, *static)
            return (_compiled(fn, static, fixed, (arrays, scaling, start)) if name in self.asks
                    else functools.partial(_not_asked, name))

        values    = program('values', _values, posed, unravel)
        jacobians = program('jacobians', _jacobian, self.jacobian, posed, unravel)
        dtypes    = {role: jax.dtypes.canonicalize_dtype(dtype) for role, dtype in self.reported.items()}
        device    = jax.devices()[0]    # asked here: in the callback it is the CPU

        def host(arrays: Any, scaling: Scaling, start: Any, magnitudes: Any) -> tuple[np.ndarray, ...]:
            '''One solve: `y` in the start's dtype, then each of `reported` in its own, the model
            evaluated on `device`.'''
            context, scaling = eqx.combine(jax.device_put(arrays, device), fixed), jax.device_put(scaling, device)
            at = lambda program: lambda y: program(context, scaling,
                                                   jax.device_put(np.asarray(y, start.dtype), device))
            y, reports = self.minimise(HostProblem(at(values), at(jacobians), start, np.asarray(magnitudes),
                                                   'jacobians' in self.asks))
            return (np.asarray(y, start.dtype),
                    *(np.asarray(reports[role], dtype) for role, dtype in dtypes.items()))

        if traced:
            answer, *reports = jax.pure_callback(
                host, (jax.ShapeDtypeStruct(start.shape, start.dtype),
                       *(jax.ShapeDtypeStruct((), dtype) for dtype in dtypes.values())),
                arrays, scaling, start, magnitudes, vmap_method='sequential')
        else:
            answer, *reports = map(jnp.asarray, host(arrays, scaling, start, magnitudes))
        reported = dict(zip(dtypes, reports, strict=True))
        return (unscaled_unknowns(posed.condition, tuple(unravel(answer)), scaling),
                {role: (reported[role],) for role in posed.reports})
