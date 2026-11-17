'''
The graph: node definitions bound to names, and the nesting over them (`within`, inner ->
outer). Stored, and read as data, with nothing checked at construction: what is read off the
bindings alone, and the three rules of the nesting, which read names and kinds only and are
`WellFormed`'s.

Kept on a graph besides its fields, outside its equality, hash and pickle: its own
`cached_property`s, and the four functions above it that are `remembered` on it --
`rules.proofs`, `dependencies.dependencies`, `nesting.nesting`, `steps.steps` (pinned in
`tests/test_layering.py`).

What follows from owns meeting reads -- owners, edges, cycles, order -- is not on it: it is
a view (`dependencies`, `nesting`), got by proving the graph well formed (`WellFormed`).
Until then a place may have several producers, so there is no edge to read.
'''

from collections.abc import Iterable, Iterator
from typing import Self
from functools import cached_property

from ..frozen import Frozen, OrderedMap, Violation
from .definition import NodeDefinition
from .kinds import enclosers, is_encloser
from .places import Label, Place


# ---------------------------------------------------------------- the rules: what a nesting obeys

def check_nesting_ends_bound(g: 'Graph') -> Iterator[Violation]:
    '''Check the within edge has both nodes in the graph.

    W ⊆ N × N
    '''
    for inner, outer in g.within.items():
        if inner not in g.definitions or outer not in g.definitions:
            yield Unbound(inner, outer)


def check_nesting_ends_are_enclosers(g: 'Graph') -> Iterator[Violation]:
    '''
    Nesting can only be between enclosers.

    W ⊆ N_P × N_P
    '''
    for inner, outer in g.within.items():
        if missing := tuple(e for e in (inner, outer)
                            if e in g.definitions and not is_encloser(g.definitions[e])):
            yield NotAnEncloser(missing)


def check_no_nesting_loops(g: 'Graph') -> Iterator[Violation]:
    '''
    Nesting can not be looped: this would imply nesting within itself in some form.

    (N_P, W) is a forest

    Only the acyclicity is checked here: `within` is a mapping, so one outer per inner
    ((n,m), (n,m') ∊ W ⟹ m = m') holds by construction, and acyclic with out-degree
    at most one is a forest.
    '''
    seen : set = set()
    for start in g.within:
        if start in seen:
            continue
        chain : list = []
        at = start
        while at is not None and at not in seen:
            if at in chain:
                if g.within[at] == at:
                    yield SelfNested(at)
                else:
                    yield NestingLoop(at, g.within[at], tuple(chain[chain.index(at):]))
                break
            chain.append(at)
            at = g.within.get(at)
        seen.update(chain)


# ---------------------------------------------------------------- the graph

class Graph[N: Label, V: Place](Frozen):
    '''
    Definitions bound to names, and a nesting decision over them: `within`, inner -> outer.
    '''

    definitions : OrderedMap[N, NodeDefinition[V]]
    within      : OrderedMap[N, N] = OrderedMap()

    # ------------------------------------------------------------ the bindings
    @cached_property
    def readers(self) -> dict[V, tuple[N, ...]]:
        '''
        Which V is read by which N (multiple)
        '''        
        out : dict[V, list[N]] = {}
        for n, d in self.definitions.items():
            for v in dict.fromkeys(d.reads):
                out.setdefault(v, []).append(n)
        return {v: tuple(ns) for v, ns in out.items()}

    @cached_property
    def nodes(self) -> tuple[N, ...]:
        '''Every node's name, in binding order.'''
        return tuple(self.definitions)
    
    @cached_property
    def variables(self) -> tuple[V, ...]:
        '''Every variable the graph names: owned by a node, or read from outside.'''
        return tuple(dict.fromkeys(self.owned_variables + self.read_variables))

    @cached_property
    def owned_variables(self) -> tuple[V, ...]:
        return tuple(var for name in self.nodes for var in self[name].owns)

    @cached_property
    def read_variables(self) -> tuple[V, ...]:
        return tuple(dict.fromkeys(var for name in self.nodes for var in self[name].reads))

    # ------------------------------------------------------------ resolution

    def __getitem__(self, name: N) -> NodeDefinition[V]:
        '''Resolve a name. The only step from a name to the definition it names.'''
        try:
            return self.definitions[name]
        except KeyError:
            raise KeyError(f'not a node of this graph: {name!r}') from None

    __iter__ = None
    '''Not iterable: `nodes` is the sequence and `definitions` the mapping.'''

    def __contains__(self, name: N) -> bool:
        '''Whether a node is in the graph'''
        return name in self.definitions

    # ------------------------------------------------------------ the nesting

    @cached_property
    def enclosers(self) -> tuple[N, ...]:
        '''The enclosers, in binding order: the ends a nesting pair may have.'''
        return enclosers(self.definitions)

    @cached_property
    def outermost_enclosers(self) -> tuple[N, ...]:
        '''Enclosers that are not nested in any other'''
        return tuple(s for s in self.enclosers if self.within.get(s) is None)

    def enclosing(self, problem: N) -> tuple[N, ...]:
        '''The enclosers `problem` is nested in transitively: A(q), every encloser with a
        path from q in (N_P, W).'''

        if problem not in self.definitions:
            raise KeyError(f'not a node of this graph: {problem!r}')
        if not is_encloser(self.definitions[problem]):
            raise ValueError(f'{problem!r} is not a problem, so it is nested in nothing and '
                             f'iterates nothing -- enclosing, cycle_of and interior are a problem\'s')
        chain : list[N] = []
        at = self.within.get(problem)
        while at is not None and at != problem and at not in chain:
            chain.append(at)
            at = self.within.get(at)
        return tuple(chain)

    # ------------------------------------------------------------ restriction
    # Every other edit is an op (`plan/`).

    @cached_property
    def _position(self) -> dict[N, int]:
        '''Each node's place in the binding order: what `subgraph` sorts by.'''
        return {n: i for i, n in enumerate(self.definitions)}

    def subgraph(self, keep: Iterable[N]) -> Self:
        '''
        Only the nodes in `keep`, with their definitions, in the parent's binding order --
        so a subgraph of a set is one graph however the set was ordered -- and the nesting
        pairs with both ends kept. O(k log k) in what is kept, not O(V): a subgraph per
        component stays linear over all of them.
        '''
        kept = set(keep)
        if unknown := [n for n in kept if n not in self._position]:
            raise KeyError(f'not node(s) of this graph: {unknown!r}')
        order = sorted(kept, key=self._position.__getitem__)
        return self._carried(definitions=OrderedMap((n, self.definitions[n]) for n in order),
                             within=OrderedMap((i, o) for i, o in self.within.items()
                                               if i in kept and o in kept))


# ----------------------------------------------------------------- violation finals

class Unbound(Violation):
    inner : Label
    outer : Label
    error    = KeyError
    template = 'nesting {inner!r} in {outer!r}: an end is not a node of this graph'


class NotAnEncloser(Violation):
    nodes : tuple[Label, ...]
    error    = ValueError
    template = ('{nodes} determines nothing, so no iteration of it answers anything -- an end '
                 'of a nesting is a problem: determine it, naming the unknowns it fixes, or '
                 '`Unnest` it first so the plan says the nesting went')


class SelfNested(Violation):
    inner : Label
    error    = ValueError
    template = '{inner!r} inside itself -- an iteration does not answer the problem it is'


class NestingLoop(Violation):
    inner : Label
    outer : Label
    chain : tuple[Label, ...]
    error    = ValueError
    template = ('{outer!r} is already answered inside {inner!r} ({chain}), so nesting the one '
                 'in the other closes a loop -- nesting is a forest')
