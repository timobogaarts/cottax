'''
The primitive ops, the only ones that build a graph: `Rebind` on the bindings (`Insert`,
`Delete`, `Redefine`, `Combine`), `Nest` and `Unnest` on the nesting.
'''

import operator
from abc import abstractmethod
from collections.abc import Collection, Iterator, Mapping
from dataclasses import replace
from functools import reduce

from ..definition import NodeDefinition
from ...frozen import Frozen, OrderedMap, Violation, first
from ..nesting import NestingGraph, NestingLoop
from ..places import Label, Place
from .algebra import GraphOp


# ---------------------------------------------------------------- the level

class Primitive[N: Label, V: Place](GraphOp[N, V]):
    '''
    An op that builds a graph, through the checked constructors: every rule is checked on
    every result. What a valid result would hide (a name replaced, a nesting pair dropped)
    the op refuses first, in `checks_graph`.
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
    def rebinding(self, graph: NestingGraph[N, V]) -> Rebinding[N, V]:
        '''What this op removes and defines of `graph`.'''

    def _apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        '''Order kept, redefinitions in place, new names appended. A nesting pair through a
        removed node is dropped; an op that must not drop one refuses first.'''
        r = self.rebinding(graph)
        definitions = {n: r.define.get(n, d) for n, d in graph.definitions.items() if n not in r.remove}
        definitions |= {n: d for n, d in r.define.items() if n not in definitions}
        return replace(
            graph,
            graph=replace(graph.graph, definitions=OrderedMap(definitions)),
            within=OrderedMap((i, o) for i, o in graph.within.items()
                              if i not in r.remove and o not in r.remove))


# ---------------------------------------------------------------- the intent of a rebinding

def check_names_free[N: Label](op: 'Insert', g: NestingGraph[N, Place]) -> Iterator[Violation]:
    '''The inserted names are not bound yet.'''
    for name in op.nodes:
        if name in g.definitions:
            yield NameTaken(name)


def check_names_bound[N: Label](op: 'Delete', g: NestingGraph[N, Place]) -> Iterator[Violation]:
    '''The removed names are bound.'''
    for name in op.nodes:
        if name not in g.definitions:
            yield NotBound(name)


def stranded[N: Label](g: NestingGraph[N, Place], removed: Collection[N]) -> Iterator[Violation]:
    '''A nesting pair losing exactly one end to `removed`. A nesting is a decision the plan
    recorded, so it is not dropped silently: `Unnest` first. Losing both ends is fine.'''
    if broken := [(i, o) for i, o in g.within.items() if (i in removed) != (o in removed)]:
        yield Stranded(tuple(i for i, _ in broken), tuple(o for _, o in broken))


def check_delete_keeps_nesting_whole[N: Label](op: 'Delete', g: NestingGraph[N, Place]) -> Iterator[Violation]:
    '''No nesting pair loses one end to the deleted nodes.'''
    yield from stranded(g, set(op.nodes))


def check_fold_keeps_nesting_whole[N: Label](op: 'Combine', g: NestingGraph[N, Place]) -> Iterator[Violation]:
    '''No nesting pair loses one end to the nodes folded away (all but the one at `at`).'''
    yield from stranded(g, set(op.nodes) - {op.at})


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

    def rebinding(self, graph: NestingGraph[N, V]) -> Rebinding[N, V]:
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

    def rebinding(self, graph: NestingGraph[N, Place]) -> Rebinding[N, Place]:
        return Rebinding(frozenset(self.nodes), OrderedMap())

    def __repr__(self) -> str:
        return f'delete({", ".join(str(n) for n in self.nodes)})'


class Redefine[N: Label, V: Place](Rebind[N, V]):
    """Replace definitions in place. The names must be bound, which is checked once
    `redefined` has named them."""

    @abstractmethod
    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        """The new definition of each node this op touches."""

    def rebinding(self, graph: NestingGraph[N, V]) -> Rebinding[N, V]:
        redefined = OrderedMap(self.redefined(graph))
        first(NotBound(name) for name in redefined if name not in graph.definitions)
        return Rebinding(frozenset(), redefined)


def check_nodes(op: 'Combine') -> Iterator[Violation]:
    '''At least two nodes, none twice, of a kind that has a `+`.'''
    if len(op.nodes) < 2:
        yield NothingToCombine()
    if len(set(op.nodes)) != len(op.nodes):
        yield NodeTwice(tuple(op.nodes))
    if getattr(op.kind, '__add__', None) is None:
        yield NotAddable(op.kind.__name__)


def check_folds_one_kind(op: 'Combine', graph: NestingGraph) -> Iterator[Violation]:
    '''Every folded node is of the declared kind, so `definition` can use its `+`.'''
    off = tuple(n for n in op.nodes if not isinstance(graph[n], op.kind))
    if off:
        found = dict.fromkeys(type(graph[n]).__name__ for n in off)
        yield NotOfKind(off, op.kind.__name__, ', '.join(found))


def check_lands_on_its_own(op: 'Combine', graph: NestingGraph) -> Iterator[Violation]:
    '''The fold lands on a node it folds, or on a name nothing holds.'''
    if op.at not in op.nodes and op.at in graph.definitions:
        yield NameTaken(op.at)


class Combine[N: Label, V: Place, D: NodeDefinition](Rebind[N, V]):
    '''
    Fold several nodes into one with their `+`: the others are removed and the fold is
    defined at `at`, in one rebinding. Whether the fold fits the graph depends on the `+`,
    so that is checked on the result.
    '''

    at : N
    '''Where the fold lands: one of `nodes` (redefined in place) or a new name.'''

    nodes : tuple[N, ...]

    kind : type[D]
    '''The class whose `+` folds the nodes; each node must be one.'''

    checks = (check_nodes,)
    checks_graph = (check_folds_one_kind, check_lands_on_its_own, check_fold_keeps_nesting_whole)

    def definition(self, graph: NestingGraph[N, V]) -> D:
        '''The `+` of the folded nodes.'''
        return reduce(operator.add, (graph[n] for n in self.nodes))

    def rebinding(self, graph: NestingGraph[N, V]) -> Rebinding[N, V]:
        return Rebinding(frozenset(self.nodes) - {self.at}, OrderedMap({self.at: self.definition(graph)}))

    def __repr__(self) -> str:
        return f'combine({self.at} <- {", ".join(str(n) for n in self.nodes)})'

# ---------------------------------------------------------------- Nesting


def check_nesting[N: Label](op: 'Nest', g: NestingGraph[N, Place]) -> Iterator[Violation]:
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


def check_unnesting[N: Label](op: 'Unnest', g: NestingGraph[N, Place]) -> Iterator[Violation]:
    """The inner is nested."""
    if op.inner not in g.within:
        yield NotNested(op.inner)

class Nest[N: Label](Primitive[N, Place]):
    '''Record that one problem is answered inside another's iteration.'''

    inner : N
    outer : N

    checks_graph = (check_nesting,)

    def _apply[G: NestingGraph[N, Place]](self, graph: G) -> G:
        return replace(graph, within=graph.within | {self.inner: self.outer})

    def __repr__(self) -> str:
        return f'nest({self.inner} in {self.outer})'


class Unnest[N: Label](Primitive[N, Place]):
    '''The inverse of `Nest`.'''

    inner : N

    checks_graph = (check_unnesting,)

    def _apply[G: NestingGraph[N, Place]](self, graph: G) -> G:
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

class NothingToCombine(Violation):
    error    = ValueError
    template = 'combine(): fewer than two nodes, so nothing to combine'

class NodeTwice(Violation):
    nodes : tuple[Label, ...]
    error    = ValueError
    template = 'combine(): {nodes!r} names a node twice'

class NotAddable(Violation):
    kind : str
    error    = TypeError
    template = 'combine(): a {kind} defines no `+`, so no two of them can be folded'

class NotOfKind(Violation):
    nodes : tuple[Label, ...]
    kind  : str
    found : str
    error    = TypeError
    template = ('combine({kind}): {nodes!r} is {found}, and a fold is well-defined over '
                'one kind only')

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
