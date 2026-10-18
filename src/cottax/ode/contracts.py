'''
Driver contracts: the level `Integrator`, its final `Integration` (the condition's initial-value
problem integrated; `trajectory` adds the `Times` saved at and reports each state's
`Trajectory`), and `CONTRACTS`, the default.
'''

from ..core import Place
from ..core.driven import DriverContract, PerProblem, PerUnknown, Role
from .ode_condition import is_ode


class Times(PerProblem):
    '''The times a trajectory is saved at: one array, within the span and monotone in the
    direction of integration; its length is the trajectory's.'''


class Trajectory(PerUnknown):
    '''Reported: each state at every one of the `Times`, stacked along a leading axis.'''


class Integrator[V: Place](DriverContract[V]):
    '''Answers an ODE by integrating it.'''
    accepts = staticmethod(is_ode)


class Integration[V: Place](Integrator[V]):
    '''The final state its answer; `trajectory` takes the `Times` and reports each state's
    `Trajectory` at them: one sampling of the solution, the integrator's to give.'''
    trajectory : bool

    @property
    def takes(self) -> tuple[type[Role], ...]:
        return (Times,) if self.trajectory else ()
    @property
    def reports(self) -> tuple[type[Role], ...]:
        return (Trajectory,) if self.trajectory else ()


CONTRACTS = (Integration(trajectory=False),)
'''The default contracts: an implementation answers a problem by the first one it implements.'''
