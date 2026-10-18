'''Offers `Diffrax`, what runs an `Integrator` in jax, chosen per contract in `ImplementDrivers`.'''

from typing import Any

import diffrax
import jax

from ...core import Place
from ...core.frozen import choice
from ...consumers.execution.interface import DriverImplementation
from ...consumers.execution.interface_types import ByRole, Context, Posed, Unknowns
from ..contracts import Integrator, Times, Trajectory


@choice(DriverImplementation)
class Diffrax(DriverImplementation[Integrator]):
    '''`diffrax.diffeqsolve` from the condition's initial values over its span, an adaptive
    step from `rtol` and `atol`, saving the end, and in the same solve the `Times` where a `Trajectory`
    is asked back. The state is the unknowns as they are, a tuple, and the context its
    `args`. `adjoint` is how it differentiates: the default reverse mode only,
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
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(_rates), self.solver, t0, t1, None, tuple(initial),
            args=(posed, context), saveat=diffrax.SaveAt(ts=ts, t1=True),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            adjoint=self.adjoint, max_steps=self.max_steps)
        final = tuple(jax.tree.map(lambda y: y[-1], solution.ys))
        if Trajectory not in posed.reports:
            return final, {}
        return final, {Trajectory: tuple(jax.tree.map(lambda y: y[:-1], solution.ys))}


def _rates(t: Any, unknowns: Unknowns, args: tuple[Posed, Context]) -> tuple[Any, ...]:
    '''The vector field: the rates, the conditions the map answers first, each its state's
    (the unknowns are the states, in order).'''
    posed, context = args
    return tuple(posed.conditions(unknowns, context)[:len(posed.condition.conditions)])

