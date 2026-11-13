'''
The primitive graph operations, each building its graph through the checked constructors, so
every rule is asked of every result: `Rebind` on the bindings
(its finals `Insert`, `Delete`, `Redefine`, `Combine`), `Nest` and `Unnest` on the nesting.
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
    An op that builds a graph, through the checked constructors: every rule of the result
    is asked in full, after every primitive. What a valid result would not show -- a name
    replaced, a pair dropped -- is the op's to refuse first, in its `checks_graph`.
    '''


# ================================================================ the primitives

# ---------------------------------------------------------------- the binding primitive

class Rebinding[N: Label, V: Place](Frozen):
    '''What a `Rebind` changes of the bindings: names removed, and names defined -- in
    place where bound, appended where new. The two are disjoint.'''

    remove : frozenset[N]
    define : OrderedMap[N, NodeDefinition[V]]


class Rebind[N: Label, V: Place](Primitive[N, V]):
    '''
    The binding primitive: names removed and names defined, in one step, so only the
    result need be valid. A derived op says what (`rebinding`) and its intent
    (`checks_graph`).
    '''

    @abstractmethod
    def rebinding(self, graph: NestingGraph[N, V]) -> Rebinding[N, V]:
        '''What this op removes and defines of `graph`.'''

    def _apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        '''The bindings kept in their order, the defined ones in place; the new ones
        appended; a nesting pair through a removed node dropped (a derived op that may
        not strand one refuses first).'''
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
    '''What it inserts, it binds anew: nothing holds the names.'''
    for name in op.nodes:
        if name in g.definitions:
            yield NameTaken(name)


def check_names_bound[N: Label](op: 'Delete', g: NestingGraph[N, Place]) -> Iterator[Violation]:
    '''What it removes is bound.'''
    for name in op.nodes:
        if name not in g.definitions:
            yield NotBound(name)


def stranded[N: Label](g: NestingGraph[N, Place], removed: Collection[N]) -> Iterator[Violation]:
    '''A node removed goes with its edges; a nesting is not an edge but a decision the
    plan recorded, so a pair losing exactly one end is `Stranded` -- `Unnest` first.
    Both ends going is fine.'''
    if broken := [(i, o) for i, o in g.within.items() if (i in removed) != (o in removed)]:
        yield Stranded(tuple(i for i, _ in broken), tuple(o for _, o in broken))


def check_delete_keeps_nesting_whole[N: Label](op: 'Delete', g: NestingGraph[N, Place]) -> Iterator[Violation]:
    '''No nesting pair loses one end to the nodes removed.'''
    yield from stranded(g, set(op.nodes))


def check_fold_keeps_nesting_whole[N: Label](op: 'Combine', g: NestingGraph[N, Place]) -> Iterator[Violation]:
    '''No nesting pair loses one end to the nodes folded away: all but the one at `at`.'''
    yield from stranded(g, set(op.nodes) - {op.at})


# ---------------------------------------------------------------- the four

def check_inserts(op: 'Insert') -> Iterator[Violation]:
    '''Something is inserted.'''
    if not op.nodes:
        yield NothingInserted()


class Insert[N: Label, V: Place](Rebind[N, V]):
    '''
    Add nodes to the graph.
    '''

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
    '''
    Remove named nodes. What a survivor read from a removed node becomes an unowned
    input, which is a well-formed graph. A nesting is a recorded decision, so an end
    going alone is refused (`Stranded`).
    '''

    nodes : tuple[N, ...]

    checks = (check_deletes,)

    checks_graph = (check_names_bound, check_delete_keeps_nesting_whole)

    def rebinding(self, graph: NestingGraph[N, Place]) -> Rebinding[N, Place]:
        return Rebinding(frozenset(self.nodes), OrderedMap())

    def __repr__(self) -> str:
        return f'delete({", ".join(str(n) for n in self.nodes)})'


class Redefine[N: Label, V: Place](Rebind[N, V]):
    """
    Replace definitions **in place**: a name it redefines is bound, which is asked once
    `redefined` has said which.
    """

    @abstractmethod
    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        """The new definition for each node this touches. Names not here are untouched."""

    def rebinding(self, graph: NestingGraph[N, V]) -> Rebinding[N, V]:
        redefined = OrderedMap(self.redefined(graph))
        first(NotBound(name) for name in redefined if name not in graph.definitions)
        return Rebinding(frozenset(), redefined)


def check_nodes(op: 'Combine') -> Iterator[Violation]:
    '''
    At least two nodes, none twice, and a kind that can be folded at all.    
    '''
    if len(op.nodes) < 2:
        yield NothingToCombine()
    if len(set(op.nodes)) != len(op.nodes):
        yield NodeTwice(tuple(op.nodes))
    if getattr(op.kind, '__add__', None) is None:
        yield NotAddable(op.kind.__name__)


def check_folds_one_kind(op: 'Combine', graph: NestingGraph) -> Iterator[Violation]:
    '''Every node folded is of the declared kind: a fold is well-defined over one only,
    and this is what lets `definition` take the `+` for granted.'''
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
    Several nodes folded into one, by the `+` they define: the folded nodes removed and
    the fold defined at `at`, in one rebinding, so only the result need be valid. Whether
    the result keeps one producer per place and exact reads is the `+`'s business, which a
    fold cannot see: the rules of the result say.
    '''

    at : N
    '''
    Resulting location: if in nodes, that node is redefined instead.
    '''

    nodes : tuple[N, ...]
    '''
    What is combined
    '''

    kind : type[D]
    '''The class whose `+` folds the nodes; each node must be one.'''

    checks = (check_nodes,)
    checks_graph = (check_folds_one_kind, check_lands_on_its_own, check_fold_keeps_nesting_whole)

    def definition(self, graph: NestingGraph[N, V]) -> D:
        '''
        New node definition: the `+` of the nodes it folds. Every one is of the declared
        kind by `check_folds_one_kind`, so the `+` is the one that kind defines.
        '''
        return reduce(operator.add, (graph[n] for n in self.nodes))

    def rebinding(self, graph: NestingGraph[N, V]) -> Rebinding[N, V]:
        '''The fold read off *this* graph, where every node still is; the binding of
        `at` kept in place if it is folded, else appended.'''
        return Rebinding(frozenset(self.nodes) - {self.at}, OrderedMap({self.at: self.definition(graph)}))

    def __repr__(self) -> str:
        return f'combine({self.at} <- {", ".join(str(n) for n in self.nodes)})'

# ---------------------------------------------------------------- Nesting


def check_nesting[N: Label](op: 'Nest', g: NestingGraph[N, Place]) -> Iterator[Violation]:
    """
    What only the op can say of its one new pair: the inner is not already answered
    inside something (the pair would replace that one), and the outer is not answered
    inside the inner (the loop named by the pair that closes it). Every other rule of a
    nesting -- ends bound, both problems, nothing inside itself -- is the graph's, asked
    of the result.
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
    """Unnest check that it was a nested one."""
    if op.inner not in g.within:
        yield NotNested(op.inner)

class Nest[N: Label](Primitive[N, Place]):
    '''
    Record that one problem is answered *inside* another's iteration.
    '''

    inner : N
    outer : N

    checks_graph = (check_nesting,)

    def _apply[G: NestingGraph[N, Place]](self, graph: G) -> G:
        return replace(graph, within=graph.within | {self.inner: self.outer})

    def __repr__(self) -> str:
        return f'nest({self.inner} in {self.outer})'


class Unnest[N: Label](Primitive[N, Place]):
    '''
    The inverse of `Nest`: the problem is answered at its own level again.
    '''

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
