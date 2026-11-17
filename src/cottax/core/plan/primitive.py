'''
The primitive ops, the only ones that build a graph: `Rebind` on the bindings (`Insert`,
`Delete`, `Redefine`), `Nest` and `Unnest` on the nesting.
'''

from abc import abstractmethod
from collections.abc import Collection, Iterator, Mapping
from dataclasses import replace

from ..maths.definition import NodeDefinition
from ..frozen import Frozen, OrderedMap, Violation, first
from ..maths.graph import Graph, NestingLoop
from ..maths.places import Label, Place
from .algebra import GraphOp


# ---------------------------------------------------------------- the level

class Primitive[N: Label, V: Place](GraphOp[N, V]):
    '''
    An op that builds a graph. Every result is proved what `apply` keeps (`WellFormed`, the
    nesting's rules and the bindings'). What a valid
    result would hide (a name replaced, a nesting pair dropped) the op refuses first, in
    `checks_graph`.
    '''


# ================================================================ the primitives

# ---------------------------------------------------------------- the binding primitive

class Rebinding[N: Label, V: Place](Frozen):
    '''Names removed, and names defined: in place where bound, appended where new. The two
    are disjoint.'''

    remove : frozenset[N]
    define : OrderedMap[N, NodeDefinition[V]]


class Rebind[N: Label, V: Place](Primitive[N, V]):
    '''Remove and define names in one step, so only the result has to be valid.'''

    @abstractmethod
    def rebinding(self, graph: Graph[N, V]) -> Rebinding[N, V]:
        '''What this op removes and defines of `graph`.'''

    def _apply[G: Graph[N, V]](self, graph: G) -> G:
        '''Order kept, redefinitions in place, new names appended. A nesting pair through a
        removed node is dropped; an op that must not drop one refuses first.'''
        r = self.rebinding(graph)
        definitions = {n: r.define.get(n, d) for n, d in graph.definitions.items() if n not in r.remove}
        definitions |= {n: d for n, d in r.define.items() if n not in definitions}
        return replace(
            graph,
            definitions=OrderedMap(definitions),
            within=OrderedMap((i, o) for i, o in graph.within.items()
                              if i not in r.remove and o not in r.remove))


# ---------------------------------------------------------------- the intent of a rebinding

def check_names_free[N: Label](op: 'Insert', g: Graph[N, Place]) -> Iterator[Violation]:
    '''The inserted names are not bound yet.'''
    for name in op.nodes:
        if name in g.definitions:
            yield NameTaken(name)


def check_names_bound[N: Label](op: 'Delete', g: Graph[N, Place]) -> Iterator[Violation]:
    '''The removed names are bound.'''
    for name in op.nodes:
        if name not in g.definitions:
            yield NotBound(name)


def stranded[N: Label](g: Graph[N, Place], removed: Collection[N]) -> Iterator[Violation]:
    '''A nesting pair losing exactly one end to `removed`. A nesting is a decision the plan
    recorded, so it is not dropped silently: `Unnest` first. Losing both ends is fine.'''
    if broken := [(i, o) for i, o in g.within.items() if (i in removed) != (o in removed)]:
        yield Stranded(tuple(i for i, _ in broken), tuple(o for _, o in broken))


def check_delete_keeps_nesting_whole[N: Label](op: 'Delete', g: Graph[N, Place]) -> Iterator[Violation]:
    '''No nesting pair loses one end to the deleted nodes.'''
    yield from stranded(g, set(op.nodes))


# ---------------------------------------------------------------- the four

def check_inserts(op: 'Insert') -> Iterator[Violation]:
    '''Something is inserted.'''
    if not op.nodes:
        yield NothingInserted()


class Insert[N: Label, V: Place](Rebind[N, V]):
    '''Add nodes to the graph.'''

    nodes : OrderedMap[N, NodeDefinition[V]]
    checks = (check_inserts,)

    checks_graph = (check_names_free,)

    def rebinding(self, graph: Graph[N, V]) -> Rebinding[N, V]:
        return Rebinding(frozenset(), self.nodes)

    def __repr__(self) -> str:
        return f'insert({", ".join(str(n) for n in self.nodes)})'


def check_deletes(op: 'Delete') -> Iterator[Violation]:
    if not op.nodes:
        yield NothingDeleted()


class Delete[N: Label](Rebind[N, Place]):
    '''Remove nodes. What the remaining nodes read from them becomes a boundary input.'''

    nodes : tuple[N, ...]

    checks = (check_deletes,)

    checks_graph = (check_names_bound, check_delete_keeps_nesting_whole)

    def rebinding(self, graph: Graph[N, Place]) -> Rebinding[N, Place]:
        return Rebinding(frozenset(self.nodes), OrderedMap())

    def __repr__(self) -> str:
        return f'delete({", ".join(str(n) for n in self.nodes)})'


class Redefine[N: Label, V: Place](Rebind[N, V]):
    """Replace definitions in place. The names must be bound, which is checked once
    `redefined` has named them. What a node carries is carried here: an op says what it
    reads and owns, and never mentions its attributes."""

    edits_attributes = False
    """Whether `redefined` itself decides the attributes: `Replace`, `SetAttribute`,
    `RemoveAttribute`."""

    @abstractmethod
    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        """The new definition of each node this op touches."""

    def rebinding(self, graph: Graph[N, V]) -> Rebinding[N, V]:
        redefined = OrderedMap(self.redefined(graph))
        first(NotBound(name) for name in redefined if name not in graph.definitions)
        if not self.edits_attributes:
            redefined = OrderedMap((name, replace(after, attributes=graph[name].attributes))
                                   for name, after in redefined.items())
        return Rebinding(frozenset(), redefined)


# ---------------------------------------------------------------- Nesting


def check_nesting[N: Label](op: 'Nest', g: Graph[N, Place]) -> Iterator[Violation]:
    """
    What only the op knows about its new pair: the inner is not nested yet (the pair would
    replace that nesting), and the outer is not inside the inner (the pair would close a
    loop). The other nesting rules are checked on the result.
    """
    inner, outer = op.inner, op.outer
    if (already := g.within.get(inner)) is not None:
        yield AlreadyNested(inner, already)
        return
    chain = [outer]
    while (at := g.within.get(chain[-1])) is not None and at not in chain:
        chain.append(at)
        if at == inner:
            yield NestingLoop(inner, outer, tuple(chain))
            return


def check_unnesting[N: Label](op: 'Unnest', g: Graph[N, Place]) -> Iterator[Violation]:
    """The inner is nested."""
    if op.inner not in g.within:
        yield NotNested(op.inner)

class Nest[N: Label](Primitive[N, Place]):
    '''Record that one problem is answered inside another's iteration.'''

    inner : N
    outer : N

    checks_graph = (check_nesting,)

    def _apply[G: Graph[N, Place]](self, graph: G) -> G:
        return replace(graph, within=graph.within | {self.inner: self.outer})

    def __repr__(self) -> str:
        return f'nest({self.inner} in {self.outer})'


class Unnest[N: Label](Primitive[N, Place]):
    '''The inverse of `Nest`.'''

    inner : N

    checks_graph = (check_unnesting,)

    def _apply[G: Graph[N, Place]](self, graph: G) -> G:
        return replace(graph, within=OrderedMap((i, o) for i, o in graph.within.items() if i != self.inner))

    def __repr__(self) -> str:
        return f'unnest({self.inner})'


# ----------------------------------------------------------------- violation finals

class NothingInserted(Violation):
    error    = ValueError
    template = 'insert(): nothing to insert'

class NothingDeleted(Violation):
    error    = ValueError
    template = 'delete(): nothing to delete'

class NameTaken(Violation):
    name : Label
    error    = ValueError
    template = 'node name {name!r} is already bound -- an insert adds a binding, it does not replace one'

class NotBound(Violation):
    name : Label
    error    = KeyError
    template = 'not a node of this graph: {name!r} -- an edit names what is bound'

class Stranded(Violation):
    inner : tuple[Label, ...]
    outer : tuple[Label, ...]
    error    = ValueError
    template = ('{inner} in {outer} would lose one end and keep the other -- `Unnest` it first, '
                 'so a plan never quietly voids what an earlier op recorded')

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
