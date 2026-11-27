'''
An Edit to a graph: what it changes to an existing graph.

Refused where it does not fit the graph it is applied to, so it is safe on any graph: on the
one its op built it for, or replayed onto another.

These are built by GraphOps (which may make a Edit depend on the graph given) and aren't built by users.
They should use the existing GraphOps, which check what they mean before building one.
'''

from collections.abc import Iterator
from dataclasses import replace

from ..frozen import Frozen, OrderedMap, Violation, first
from ..maths.definition import NodeDefinition
from ..maths.graph import Graph, NameTaken, NotBound
from ..maths.places import Label, Place

class Edit[N: Label, V: Place](Frozen):
    '''
    Edit to a graph in the ordering: unnest->(remove, redefine, insert)->nest

    This ordering ensures a minimal set of Edits for an operation to remain a Valid graph (e.g. a removal Edit would be Unnest only, then remove/redefine/insert Edit instead of a single Edit).    
    '''
    # ------- changes
    unnest   : frozenset[N]                       = frozenset()
    
    remove   : frozenset[N]                       = frozenset()
    redefine : OrderedMap[N, NodeDefinition[V]]   = OrderedMap()
    insert   : OrderedMap[N, NodeDefinition[V]]   = OrderedMap()

    nest     : OrderedMap[N, N]                   = OrderedMap()

    def applied[G: Graph[N, V]](self, graph: G) -> G:
        '''Applied to graph; refused where it does not fit it.'''
        first(_misfits(self, graph))
        definitions = OrderedMap({n: self.redefine.get(n, d) for n, d in graph.definitions.items()
                                  if n not in self.remove} | dict(self.insert))
        within = OrderedMap((i, o) for i, o in graph.within.items() if i not in self.unnest)

        return replace(graph, definitions=definitions, within=within | self.nest)


def _misfits(edit: Edit, graph: Graph) -> Iterator[Violation]:
    '''
    Where the Edit has no meaning on the graph (e.g. no silent no-ops or inserting already present nodes).
    Does not necessarily leave a Valid graph (see `Plan.keeps` for a way to enforce that)
    '''
    seen, twice = set(), {}
    for n in (*edit.remove, *edit.redefine, *edit.insert):
        if n in seen:
            twice[n] = None
        seen.add(n)
    if twice:
        yield NamedTwice(tuple(twice))
    for n in (*edit.remove, *edit.redefine):
        if n not in graph.definitions:
            yield NotBound(n)
    for n in edit.insert:
        if n in graph.definitions:
            yield NameTaken(n)
    for n in edit.unnest:
        if n not in graph.within:
            yield NotNested(n)
    for n in edit.nest:
        if n in graph.within and n not in edit.unnest:
            yield AlreadyNested(n, graph.within[n])


# ----------------------------------------------------------------- violation finals

class NotNested(Violation):
    inner : Label
    error    = ValueError
    template = '{inner!r} is not nested inside anything, so there is nothing to undo'

class AlreadyNested(Violation):
    inner : Label
    outer : Label
    error    = ValueError
    template = ('{inner!r} is already answered inside {outer!r} -- a problem has one '
                 'enclosing iteration, so `Unnest` it before nesting it somewhere else')

class NamedTwice(Violation):
    names : tuple[Label, ...]
    error    = ValueError
    template = 'an edit names {names!r} in two of remove / redefine / insert: each says one thing of a name'

