'''
The roles the shipped contracts give their places, and where each driven problem of a graph
takes one. A role is a class, never a value; its `index` says what its places stand for.
'''

from ..maths import Label, Graph, Place, conditions
from .driver import PerProblem, PerUnknown, Role
from .node import is_driven


class Start(PerUnknown):
    '''Where each unknown starts.'''


class Steps(PerProblem):
    '''Reported: how many iterations the solve took.'''


def role_slots[N: Label, V: Place](graph: Graph[N, V], role: type[Role]
                                   ) -> tuple[tuple[V, N, V | None], ...]:
    '''Each place a driven problem takes or reports `role` at, in binding order: the place,
    its problem, and what it stands for (`role.index`: an unknown, `None` for the problem as
    a whole). A place two problems take is there twice.'''
    return tuple((place, name, subject)
                 for name in conditions(graph.definitions) if is_driven(node := graph[name])
                 and (by := node.driver.places_for(role, node.condition)) is not None
                 for subject, place in by.items())
