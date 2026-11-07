'''Offers `Diffrax`, what runs an `Integrator` in jax, chosen per contract in `ImplementDrivers`.'''

from typing import Any

import diffrax
import jax
import jax.numpy as jnp

from ...core import Place
from ...core.frozen import choice
from ...consumers.execution.interface import DriverImplementation
from ...consumers.execution.interface_types import ByRole, Context, Posed, Unknowns
from ..contracts import Integrator, Times, Trajectory


@choice(DriverImplementation)
class Diffrax(DriverImplementation[Integrator]):
    '''`diffrax.diffeqsolve` from the condition's initial values over its span, an adaptive
    step from `rtol` and `atol` landing on the condition's kinks (`step_ts`) and jumps
    (`jump_ts`), saving the end, and in the same solve the `Times` where a `Trajectory`
    is asked back. The state is the states as they are, a tuple, the solver's time given at
    the condition's `time` where it names one, and the context its `args`. `adjoint` is how it differentiates: the default reverse mode only,
    `diffrax.ForwardMode()` forward mode only, `diffrax.DirectAdjoint()` both.'''

    needs    = frozenset()
    provides = frozenset({Trajectory})

    solver    : diffrax.AbstractSolver   = diffrax.Tsit5()
    rtol      : float                    = 1e-6
    atol      : float                    = 1e-6
    max_steps : int                      = 4096
    adjoint   : diffrax.AbstractAdjoint  = diffrax.RecursiveCheckpointAdjoint()

    def __call__[V: Place](self, posed: Posed[V], context: Context,
                           data: ByRole) -> tuple[Unknowns, ByRole]:
        ode = posed.condition
        *initial, (t0, t1) = posed.fixed((*ode.initials, ode.span), context)
        (ts,) = data[Times] if Trajectory in posed.reports else (None,)
        kinks, jumps = (_breaks(posed.fixed(at, context)) for at in (ode.kinks, ode.jumps))
        controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)
        if kinks is not None or jumps is not None:
            controller = diffrax.ClipStepSizeController(controller, step_ts=kinks, jump_ts=jumps)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(_rates), self.solver, t0, t1, None, tuple(initial),
            args=(posed, context), saveat=diffrax.SaveAt(ts=ts, t1=True),
            stepsize_controller=controller, adjoint=self.adjoint, max_steps=self.max_steps)
        timed = ode.time is not None
        final = (*jax.tree.map(lambda y: y[-1], solution.ys), *((jnp.asarray(t1),) if timed else ()))
        if Trajectory not in posed.reports:
            return final, {}
        histories = (*jax.tree.map(lambda y: y[:-1], solution.ys), *((ts,) if timed else ()))
        return final, {Trajectory: histories}


def _breaks(read: tuple[Any, ...]) -> jax.Array | None:
    '''The times read at each kink (or jump) place, one sorted array; `None` where none is.'''
    if not read:
        return None
    return jnp.sort(jnp.concatenate([jnp.ravel(jnp.asarray(t)) for t in read]))


def _rates(t: Any, states: Unknowns, args: tuple[Posed, Context]) -> tuple[Any, ...]:
    '''The vector field: the rates, the conditions the map answers first, each its state's.
    The unknowns are the states, in order, then the time where the condition gives one:
    the solver's own `t`, exact where a step lands on a jump.'''
    posed, context = args
    unknowns = (*states, t) if posed.condition.time is not None else states
    return tuple(posed.conditions(unknowns, context)[:len(posed.condition.conditions)])

