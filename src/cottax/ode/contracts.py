'''
Driver contracts: the level `Integrator`, its final `Integration` (the condition's initial-value
problem integrated; asked for each state's `Trajectory`, it takes the `Times` saved at), and
`CONTRACTS`, the default.
'''

from ..core import Place
from ..core.conditional import DriverContract, PerProblem, PerUnknown
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
    '''The final state its answer; may report each state's `Trajectory`, and then takes the
    `Times` it is saved at: one sampling of the solution, the integrator's to give.'''
    takes, may_report, report_needs = (), (Trajectory,), {Trajectory: (Times,)}


CONTRACTS = (Integration(),)
'''The default contracts: an implementation answers a problem by the first one it implements.'''
