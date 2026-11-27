'''
Graph operations making Edits and changing the graph.

In many cases, the Edit is not independent of the Graph and might not even be possible to give without changing the graph
(e.g. composites that conditionally apply Edits). 
'''
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import ClassVar, Self

from ..frozen import Frozen, PerLevel, Violation, first
from ..maths.graph import Graph
from ..maths.places import Label, Place
from ..maths.rules import AnyGraph, Rules
from .edit import Edit

class GraphOp[N: Label, V: Place](Frozen):
    '''Graph Operation taking Graphs and creating Edits conditionally.'''

    needs : ClassVar[type[Rules]] = AnyGraph
    '''Needed for Op validity (e.g. requires a particular View; although this is checked per View, provides intent here)'''

    checks_graph : PerLevel[tuple[Callable[[Self, Graph], Iterable[Violation]], ...]] = ()
    '''`(op, graph) -> violations`: whether the op makes sense on this graph.'''

    @abstractmethod
    def _changed[G: Graph[N, V]](self, graph: G) -> tuple[G, tuple[Edit[N, V], ...]]:
        '''new Graph and Edit from this op, assuming the checks (needs & checks_graph) held'''

    def _with_edits[G: Graph[N, V]](self, graph: G) -> tuple[G, tuple[Edit[N, V], ...]]:        
        try:
            first(self.needs.check(graph)) # check whether the graph satisfies the needs rule
            first(_asked(self, graph))     # check whether this graph and this op can combine
            return self._changed(graph)    # apply op and return new graph + edit (can still raise if edit itself is invalid)
        except Exception as exception:
            exception.add_note(f'refused by {_short(self)}')
            raise

    def apply[G: Graph[N, V]](self, graph: G) -> G:        
        return self._with_edits(graph)[0]

    def edits(self, graph: Graph[N, V]) -> tuple[Edit[N, V], ...]:        
        return self._with_edits(graph)[1]

    def __radd__(self, base: Graph[N, V]):
        '''`graph + op` starts a plan. Here and not on the graph, which knows no plans.'''
        from .plan import Plan
        if not isinstance(base, Graph):
            return NotImplemented
        return Plan(base) + self


class PrimitiveOp[N: Label, V: Place](GraphOp[N, V]):
    '''An op that is one edit: an `Edit` with intent (which can use intent-based checks)'''

    @abstractmethod
    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        '''Edit it makes (assuming its needs/checks_graph hold)'''
        ...

    def _changed[G: Graph[N, V]](self, graph: G) -> tuple[G, tuple[Edit[N, V], ...]]:
        edit = self._edit(graph)
        return edit.applied(graph), (edit,)

class Composite[N: Label, V: Place](GraphOp[N, V]):
    '''An op made of other ops, applied in turn.

    An op decided in stages nests `Composite`s: each is decided when it is applied, off the
    graph the last left, and reads earlier results off that graph or is handed them::

        def _ops(self, graph):                      # the first stage, off `graph`
            return (*first_ops(graph), Second(before=graph))
    '''

    @abstractmethod
    def _ops(self, graph: Graph[N, V]) -> Sequence[GraphOp[N, V]]:
        '''The ops this one is, decided off `graph`; each is checked as it is applied.'''

    def _changed[G: Graph[N, V]](self, graph: G) -> tuple[G, tuple[Edit[N, V], ...]]:
        return in_turn(self._ops(graph), graph)


def in_turn[G: Graph](ops: Iterable[GraphOp], graph: G) -> tuple[G, tuple[Edit, ...]]:
    '''`ops` applied in turn from `graph`: the graph they leave and their edits, in order.'''
    edits: list[Edit] = []
    for op in ops:
        graph, done = op._with_edits(graph)
        edits += done
    return graph, tuple(edits)


def _short(op: GraphOp) -> str:
    '''`op` as a refusal names it: its repr, or its kind where that repr is long.'''
    shown = repr(op)
    return shown if len(shown) <= 80 else f'{type(op).__name__}(...)'


def _asked(op: GraphOp, graph: Graph) -> Iterator[Violation]:
    '''The first level of `op` refusing it in `checks_graph`, root first: all of that level's.'''
    for level in reversed(type(op).__mro__):
        if found := [v for check in level.__dict__.get('checks_graph', ()) for v in check(op, graph)]:
            yield from found
            return
