'''
Executable view that can be scheduled.
'''

from functools import cached_property

from ..frozen import Frozen
from .executable import Executable
from .dependencies import Dependencies
from .graph import View
from .nesting import Nesting
from .places import Label, Place

class Run[N: Label](Frozen):
    '''A function node, run.'''
    node : N
class Enclosed[N: Label, V: Place](Frozen):
    '''Enclosed subgraph (higher-order function)'''

    encloser : N
    interior : 'Schedule[N, V]'

type Step[N: Label, V: Place] = Run[N] | Enclosed[N, V]
class Schedule[N: Label, V: Place](View[N, V]):
    '''An executable graph's steps, in the order they run, each `Enclosed` holding its
    interior's. Asks `Executable`.'''

    checks = (Executable.check,)

    @cached_property
    def steps(self) -> tuple[Step[N, V], ...]:
        '''
        Recursive steps.
        '''
        outer = set(Nesting.of(self.graph).outermost)
        found : list[Step[N, V]] = []
        for component in Dependencies.of(self.graph).components:
            if len(component) == 1:
                found.append(Run(component[0]))
            else:
                encloser, = (p for p in component if p in outer)
                interior = Nesting.of(self.graph).interior(encloser)
                found.append(Enclosed(encloser, Schedule.of(interior)))
        return tuple(found)
