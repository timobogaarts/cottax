'''An executable graph read as steps: `Run` a function node, `Loop` an encloser's cycle.'''

from functools import cached_property

from ..frozen import Frozen
from .executable import Executable
from .dependencies import Dependencies
from .graph import Graph, View
from .nesting import Nesting
from .places import Label, Place


class Run[N: Label](Frozen):
    '''A function node, run.'''

    node : N


class Loop[N: Label, V: Place](Frozen):
    '''A cycle: its outermost encloser, and what one iteration of it runs -- a solve, a
    sweep, a DoE, a switch alike.'''

    encloser : N
    interior : Graph[N, V]
    '''The cycle without `encloser`, proved executable. Anything more is asked of it where
    it is needed: nothing proved of the whole is copied to it.'''


type Step[N: Label, V: Place] = Run[N] | Loop[N, V]
'''What a run does with one component of an executable graph, by label.'''


class Schedule[N: Label, V: Place](View[N, V]):
    '''An executable graph's steps, in the order they run. Asks `Executable`.'''

    checks = (Executable.check,)

    @cached_property
    def steps(self) -> tuple[Step[N, V], ...]:
        '''
        One per component, in dependency order. One node is a `Run` -- an encloser is never
        alone on its cycle -- and several are a cycle, a `Loop` of its one outermost
        encloser over the interior, itself proved executable.
        '''
        outer = set(Nesting.of(self.graph).outermost)
        found : list[Step[N, V]] = []
        for component in Dependencies.of(self.graph).components:
            if len(component) == 1:
                found.append(Run(component[0]))
            else:
                encloser, = (p for p in component if p in outer)
                interior = Nesting.of(self.graph).interior(encloser)
                Executable.prove(interior)
                found.append(Loop(encloser, interior))
        return tuple(found)
