'''An executable graph read as steps: `Run` a function node, `Solve` a cycle.'''

from ..frozen import Frozen, checked, remembered
from .executable import Executable
from .dependencies import Dependencies
from .graph import Graph
from .nesting import Nesting
from .places import Label, Place


class Run[N: Label](Frozen):
    '''A function node, run.'''

    node : N


class Solve[N: Label, V: Place](Frozen):
    '''A cycle: its outermost problem, and what one iteration of it runs.'''

    problem  : N
    interior : Graph[N, V]
    '''The cycle without `problem`, proved executable. Anything more is asked of it where it
    is needed: nothing proved of the whole is copied to it.'''


type Step[N: Label, V: Place] = Run[N] | Solve[N, V]
'''What a run does with one component of an executable graph, by label.'''


@remembered
@checked(Executable.check)
def steps[N: Label, V: Place](graph: Graph[N, V]) -> tuple[Step[N, V], ...]:
    '''
    One per component, in dependency order, of a graph proved executable. One node is
    a `Run` -- a problem is never alone on its cycle -- and several are a cycle, a `Solve`
    of its one outermost problem over the interior. Remembered on the graph, as its proofs
    are.
    '''
    outer = set(Nesting.of(graph).outermost)
    found : list[Step[N, V]] = []
    for component in Dependencies.of(graph).components:
        if len(component) == 1:
            found.append(Run(component[0]))
        else:
            problem, = (p for p in component if p in outer)
            interior = Nesting.of(graph).interior(problem)
            Executable.prove(interior)
            found.append(Solve(problem, interior))
    return tuple(found)
