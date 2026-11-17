'''
The roles the shipped contracts give their places, and where each driven problem of a graph
takes one. A role is a class, never a value; its `arity` is its kind's.
'''

from ...core import Label, Graph, Place, is_driven
from .driver import PerProblem, PerUnknown, Role


class Start(PerUnknown):
    '''Where each unknown starts.'''


class Steps(PerProblem):
    '''Reported: how many iterations the solve took.'''


def role_slots[N: Label, V: Place](graph: Graph[N, V], role: type[Role]
                                   ) -> tuple[tuple[V, N, V | None], ...]:
    '''Each place a driven problem takes or reports `role` at, in binding order: the place,
    its problem, and for a per-unknown role the unknown it pairs with (`None` for a
    per-problem one). A place two problems take is there twice.'''
    return tuple((place, name, unknown)
                 for name in graph.enclosers if is_driven(node := graph[name])
                 and (places := node.driver.places_of(role)) is not None
                 for place, unknown in (zip(places, node.owns_inside) if issubclass(role, PerUnknown)
                                        else ((places[0], None),)))
