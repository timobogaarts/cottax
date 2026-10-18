'''
The primitive graph operations, each proved through `rebound`: `Rebind` on the bindings
(its finals `Insert`, `Delete`, `Redefine`, `Combine`), `Nest` and `Unnest` on the nesting.
'''

import operator
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import reduce
from typing import Any

from ..definition import NodeDefinition
from ...frozen import AbstractClassVar, Frozen, OrderedMap, PerLevel, Violation, first
from ..graph import (
    AbstractGraph, check_one_producer_per_place, check_reads_are_exact, owned_against,
    owned_overlaps, owners_of, readers_of, reads_exact_against,
)
from ..kinds import is_problem
from ..nesting import (
    NestingGraph, NestingLoop, NotAProblem, SelfNested, Unbound, check_nesting_ends_are_problems,
    check_nesting_ends_bound, check_nesting_ends_on_one_cycle, check_no_nesting_loops,
)
from ..places import Label, Place, overlaps
from .algebra import GraphOp, _owed


# ---------------------------------------------------------------- what stands in for a rule

def unbreakable(before, after, delta) -> Iterator[Violation]:
    '''A rule the op cannot break, by argument: nothing is asked.'''
    return iter(())


def in_full(before, after, delta) -> Iterator[Violation]:
    '''No proof: the rule is asked of the result. Never called -- `rebound` knows it.'''
    raise AssertionError('in_full is a marker, not a proof')


# ---------------------------------------------------------------- the level

class Edited[N: Label, V: Place](Frozen):
    '''What a primitive changes: the delta its checks read, and the new definitions or
    nesting, `None` for one left as it is.'''

    delta       : Any
    definitions : OrderedMap[N, NodeDefinition[V]] | None
    within      : OrderedMap[N, N] | None


class Primitive[N: Label, V: Place](GraphOp[N, V]):
    '''
    An op that builds a graph. Two additional delta-checks:
    
    - Stage 3, `checks_delta`: what it needs of its delta, `(graph, delta) -> Iterator[Violation]`, asked before building. 
    - Stage 4, `covers`: every rule of NestingGraph and how it is covered. 
    '''

    checks_delta : PerLevel[tuple[Callable, ...]] = ()
    '''Each level's own, accumulated root first as `checks` are: a level adds to its
    ancestors' and never replaces them.'''
    covers       : AbstractClassVar[Mapping[Callable, Callable]]
    '''Bound once, by the primitive: a level below cannot rebind it (`Frozen`).'''

    @abstractmethod
    def edited(self, graph: NestingGraph[N, V]) -> Edited[N, V]:
        '''What this op changes of `graph`, which may assume every `checks_graph` case
        holds.'''

    def _apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        e = self.edited(graph)
        return rebound(self, graph, e.delta, definitions=e.definitions, within=e.within)


# ---------------------------------------------------------------- the proof

_fully_checked = ContextVar('fully_checked', default=False)


@contextmanager
def fully_checked():
    '''Every proved rule asked in full as well: a proof passing what its rule refuses is
    a bug in the op, not in the graph. The test suite runs under it.'''
    token = _fully_checked.set(True)
    try:
        yield
    finally:
        _fully_checked.reset(token)


def rebound[G: NestingGraph](op: Primitive, graph: G, delta, *, definitions=None, within=None) -> G:
    '''
    `op`'s delta checks asked of `graph` and `delta`; then `graph` with `definitions` or
    `within` changed, and every rule of the result asked as `op.covers` says: through its
    proof, not at all where `unbreakable`, in full where `in_full`. A rule a final adds,
    which no op can name, is asked in full.
    '''
    for delta_check in type(op).per_level('checks_delta'):
        first(delta_check(graph, delta))
    inner = graph.graph if definitions is None else graph.graph._carried(definitions=definitions)
    child = graph._carried(graph=inner, **({} if within is None else {'within': within}))
    for before, after in ((graph.graph, inner), (graph, child)):
        for rule in type(after).per_level('checks'):
            if (proof := op.covers.get(rule, in_full)) is in_full:
                if (owed := _owed.get()) is not None and not _fully_checked.get():
                    owed.setdefault(rule, after is child)       # asked of the composition's result
                else:
                    first(rule(after) or ())
                continue
            first(proof(before, after, delta))
            if _fully_checked.get() and (missed := next(iter(rule(after) or ()), None)):
                raise AssertionError(f'{type(op).__name__}: {proof.__name__} passed what '
                                     f'{rule.__name__} refuses -- {missed.message}')
    return child


# ================================================================ the primitives

# ---------------------------------------------------------------- the binding primitive

class Rebinding[N: Label, V: Place](Frozen):
    '''What a `Rebind` changes of the bindings: names removed, and names defined -- in
    place where bound, appended where new. The two are disjoint.'''

    remove : frozenset[N]
    define : OrderedMap[N, NodeDefinition[V]]


def _owns_moved[N: Label, V: Place](
    g      : AbstractGraph[N, V],
    define : Mapping[N, NodeDefinition[V]],
) -> bool:
    '''Whether a node defined in place owns other than it did.'''
    return any(n in g.definitions and d.owns != g.definitions[n].owns for n, d in define.items())


def check_rebound_owns_disjoint[N: Label, V: Place](
    before : AbstractGraph[N, V],
    after  : AbstractGraph[N, V],
    r      : Rebinding[N, V],
) -> Iterator[Violation]:
    '''`check_one_producer_per_place` of a rebinding. A removal frees places, and a node
    redefined owning what it owned moves none: nothing to ask. New nodes: their owns
    among themselves, then against the standing owners. Once something is removed while
    something is defined, or an own moved in place, the rule of the result.'''
    if not r.define:
        return
    if r.remove or _owns_moved(before, r.define):
        yield from check_one_producer_per_place(after)
        return
    new_owned = [(n, v) for n, d in r.define.items() if n not in before.definitions for v in d.owns]
    yield from owned_overlaps(new_owned)
    yield from owned_against(new_owned, before.owners, before.owned_below)


def check_rebound_reads_exact[N: Label, V: Place](
    before : AbstractGraph[N, V],
    after  : AbstractGraph[N, V],
    r      : Rebinding[N, V],
) -> Iterator[Violation]:
    '''`check_reads_are_exact` of a rebinding. A removal only turns reads into boundary
    inputs: nothing to ask. Otherwise the defined nodes' reads against every owner after,
    and the new owns against the reads that already stand. Once something is removed
    while something is defined, or an own moved in place, the rule of the result.'''
    if not r.define:
        return
    if r.remove or _owns_moved(before, r.define):
        yield from check_reads_are_exact(after)
        return
    new_owns = owners_of({n: d for n, d in r.define.items() if n not in before.definitions})
    reads    = readers_of(r.define)
    yield from reads_exact_against(reads, before.owners, before.owned_below)
    yield from reads_exact_against(reads, new_owns)
    stood = {r: before.readers[r] for _, r in overlaps(new_owns, before.readers, before.read_below)}
    yield from reads_exact_against(stood, new_owns)          # the reads standing, against the new owns


class Rebind[N: Label, V: Place](Primitive[N, V]):
    '''
    The binding primitive: names removed and names defined, in one step, so only the
    result need be valid. A derived op says what (`rebinding`) and its intent
    (`checks_delta`, of the `Rebinding`); the proof is this level's, and not overridden.
    '''

    covers = {
        check_one_producer_per_place:    check_rebound_owns_disjoint,
        check_reads_are_exact:           check_rebound_reads_exact,
        check_nesting_ends_bound:        in_full,
        check_no_nesting_loops:          in_full,
        check_nesting_ends_are_problems: in_full,
        check_nesting_ends_on_one_cycle: in_full}

    @abstractmethod
    def rebinding(self, graph: NestingGraph[N, V]) -> Rebinding[N, V]:
        '''What this op removes and defines of `graph`.'''

    def edited(self, graph: NestingGraph[N, V]) -> Edited[N, V]:
        '''The bindings kept in their order, the defined ones in place; the new ones
        appended; a nesting pair through a removed node dropped (a derived op that may
        not strand one refuses first).'''
        r = self.rebinding(graph)
        definitions = {n: r.define.get(n, d) for n, d in graph.definitions.items() if n not in r.remove}
        definitions |= {n: d for n, d in r.define.items() if n not in definitions}
        within = (OrderedMap((i, o) for i, o in graph.within.items() if i not in r.remove and o not in r.remove)
                  if r.remove else None)
        return Edited(r, OrderedMap(definitions), within)


# ---------------------------------------------------------------- the intent of a rebinding

def check_names_free[N: Label](g: NestingGraph[N, Place], r: Rebinding) -> Iterator[Violation]:
    '''What it defines, it binds anew: nothing holds the names.'''
    for name in r.define:
        if name in g.definitions:
            yield NameTaken(name)


def check_names_bound[N: Label](g: NestingGraph[N, Place], r: Rebinding) -> Iterator[Violation]:
    '''What it removes or redefines is bound.'''
    for name in (*r.remove, *r.define):
        if name not in g.definitions:
            yield NotBound(name)


def check_delete_keeps_nesting_whole[N: Label](g: NestingGraph[N, Place], r: Rebinding) -> Iterator[Violation]:
    '''A node removed goes with its edges; a nesting is not an edge but a decision the
    plan recorded, so a pair losing exactly one end is `Stranded` -- `Unnest` first.
    Both ends going is fine.'''
    if broken := [(i, o) for i, o in g.within.items() if (i in r.remove) != (o in r.remove)]:
        yield Stranded(tuple(i for i, _ in broken), tuple(o for _, o in broken))


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

    checks_delta = (check_names_free,)

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

    checks_delta = (check_names_bound, check_delete_keeps_nesting_whole)

    def rebinding(self, graph: NestingGraph[N, Place]) -> Rebinding[N, Place]:
        return Rebinding(frozenset(self.nodes), OrderedMap())

    def __repr__(self) -> str:
        return f'delete({", ".join(str(n) for n in self.nodes)})'


class Redefine[N: Label, V: Place](Rebind[N, V]):
    """
    Replace definitions **in place**.
    """

    checks_delta = (check_names_bound,)

    @abstractmethod
    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        """The new definition for each node this touches. Names not here are untouched."""

    def rebinding(self, graph: NestingGraph[N, V]) -> Rebinding[N, V]:
        return Rebinding(frozenset(), OrderedMap(self.redefined(graph)))


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
    fold cannot see: `Rebind`'s proof asks those rules of the result.
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
    checks_graph = (check_folds_one_kind, check_lands_on_its_own)
    checks_delta = (check_delete_keeps_nesting_whole,)

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


def check_nesting[N: Label](g: NestingGraph[N, Place], pair: tuple[N, N]) -> Iterator[Violation]:
    """
    What one new pair needs of the nesting alone: both ends bound; the inner not
    already answered inside something; both problems; a forest still (not a problem
    inside itself, no loop through what already encloses the outer). That the pair --
    and every pair now under `inner` -- shares a cycle at its level is
    `check_nesting_ends_on_one_cycle` of the result, which `rebound` asks next.
    """
    inner, outer = pair
    if inner not in g.definitions or outer not in g.definitions:
        yield Unbound(inner, outer)
        return
    if inner == outer:
        yield SelfNested(inner)
        return
    if (already := g.within.get(inner)) is not None:
        yield AlreadyNested(inner, already)
        return
    if missing := tuple(e for e in (inner, outer) if not is_problem(g.definitions[e])):
        yield NotAProblem(missing)
        return
    chain = g.enclosing(outer)
    if inner in chain:
        yield NestingLoop(inner, outer, (outer, *chain[:chain.index(inner) + 1]))


def check_unnesting[N: Label](g: NestingGraph[N, Place], inner: N) -> Iterator[Violation]:
    """Unnest check that it was a nested one."""
    if inner not in g.within:
        yield NotNested(inner)

class Nest[N: Label](Primitive[N, Place]):
    '''
    Record that one problem is answered *inside* another's iteration.
    '''

    inner : N
    outer : N

    # The pair's own facts from the delta; the bindings do not change; every nesting
    # rule is asked of the result.
    checks_delta = (check_nesting,)
    covers = {
        check_one_producer_per_place:    unbreakable,   # the bindings are untouched
        check_reads_are_exact:           unbreakable,
        check_nesting_ends_bound:        in_full,
        check_no_nesting_loops:          in_full,
        check_nesting_ends_are_problems: in_full,
        check_nesting_ends_on_one_cycle: in_full}

    def edited(self, graph: NestingGraph[N, Place]) -> Edited[N, Place]:
        return Edited((self.inner, self.outer), None, graph.within | {self.inner: self.outer})

    def __repr__(self) -> str:
        return f'nest({self.inner} in {self.outer})'


class Unnest[N: Label](Primitive[N, Place]):
    '''
    The inverse of `Nest`: the problem is answered at its own level again.
    '''

    inner : N

    # The delta check is that there was a pair. Dropping one moves the level of every
    # pair under `inner`, so the nesting rules are asked of the result.
    checks_delta = (check_unnesting,)
    covers = {
        check_one_producer_per_place:    unbreakable,   # the bindings are untouched
        check_reads_are_exact:           unbreakable,
        check_nesting_ends_bound:        in_full,
        check_no_nesting_loops:          in_full,
        check_nesting_ends_are_problems: in_full,
        check_nesting_ends_on_one_cycle: in_full}

    def edited(self, graph: NestingGraph[N, Place]) -> Edited[N, Place]:
        return Edited(self.inner, None, OrderedMap((i, o) for i, o in graph.within.items() if i != self.inner))

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
