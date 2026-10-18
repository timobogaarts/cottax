'''
Nesting graph: Graph + nesting forest
'''

from collections.abc import Iterable, Iterator, Mapping
from typing import Self
from functools import cached_property

from ..frozen import Frozen, OrderedMap, Violation, first
from .definition import NodeDefinition
from .graph import AbstractGraph
from .kinds import Function, is_problem, problems
from .places import Label, Place


# ---------------------------------------------------------------- the rules: what a nesting obeys

def check_nesting_ends_bound(g: 'NestingGraph') -> Iterator[Violation]:
    '''Check the within edge has both nodes in the graph.

    W ⊆ N × N
    '''
    for inner, outer in g.within.items():
        if inner not in g.definitions or outer not in g.definitions:
            yield Unbound(inner, outer)


def check_nesting_ends_are_problems(g: 'NestingGraph') -> Iterator[Violation]:
    '''
    Nesting can only be between problems.

    W ⊆ N_P × N_P
    '''
    for inner, outer in g.within.items():
        if missing := tuple(e for e in (inner, outer)
                            if e in g.definitions and not is_problem(g.definitions[e])):
            yield NotAProblem(missing)


def check_no_nesting_loops(g: 'NestingGraph') -> Iterator[Violation]:
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


def check_nesting_ends_on_one_cycle(g: 'NestingGraph') -> Iterator[Violation]:
    '''
    Both ends of a nesting are on one cycle: the inner lies in the outer's cycle. The
    inner's output is required for the outer's inputs and vice-versa -- otherwise the
    inner does not depend on the outer's iterate, or the outer never reads its answer,
    and it could run outside.

    p ~ q  in  N \\ A(q)   ∀ (p, q) ∊ W       i.e.  p ∊ S_q, the cycle of q

    A pair with an end unbound or not a problem is skipped: those are other rules'.
    '''
    for inner, outer in g.within.items():
        if not all(e in g.definitions and is_problem(g.definitions[e]) for e in (inner, outer)):
            continue
        if inner not in g.cycle_of(outer):
            yield Unreachable(inner, outer)


def check_blocks_partition(g: 'NestingGraph', blocks: Iterable[Iterable]) -> Iterator[Violation]:
    '''Each block is at least two nodes, and no node is in two.'''
    position, seen, twice = g.graph._position, set(), {}
    for block in blocks:
        if len(set(block)) < 2:
            yield BlockTooSmall(tuple(block))
        for n in dict.fromkeys(block):
            if n in seen:
                twice[n] = None
            seen.add(n)
    if twice:
        yield NodeInTwoBlocks(tuple(sorted(twice, key=position.__getitem__)))


def check_blocks_own_something(g: 'NestingGraph', blocks: Iterable[Iterable]) -> Iterator[Violation]:
    '''A block contracts into a computation, so it owns something.'''
    for block in blocks:
        if not any(g.definitions[n].owns for n in block):
            yield BoxOwnsNothing(tuple(block))


# ---------------------------------------------------------------- the graph, nested

class NestingGraph[N: Label, V: Place](Frozen):
    '''
    A graph plus nesting decision: `within`, inner -> outer
    '''

    graph  : AbstractGraph[N, V]
    within : OrderedMap[N, N]
    checks = (check_nesting_ends_bound, check_no_nesting_loops,
              check_nesting_ends_are_problems, check_nesting_ends_on_one_cycle)

    # ------------------------------------------------------------ the bindings, its own

    @property
    def definitions(self) -> Mapping[N, NodeDefinition[V]]:
        return self.graph.definitions

    @property
    def nodes(self) -> tuple[N, ...]:
        return self.graph.nodes

    def __getitem__(self, name: N) -> NodeDefinition[V]:
        return self.graph[name]

    __iter__ = None
    '''Not iterable: `nodes` is the sequence and `definitions` the mapping.'''

    def __contains__(self, name: N) -> bool:
        return name in self.graph

    # ------------------------------------------------------------ the nesting

    @cached_property
    def problems(self) -> tuple[N, ...]:
        '''The problems, in binding order: the ends a nesting pair may have.'''
        return problems(self.definitions)

    @cached_property
    def outermost_problems(self) -> tuple[N, ...]:
        '''Problems that are not nested in any other'''
        return tuple(s for s in self.problems if self.within.get(s) is None)

    def enclosing(self, problem: N) -> tuple[N, ...]:
        '''The problems `problem` is nested in transitively: A(q), every problem with a
        path from q in (N_P, W).'''

        if problem not in self.definitions:
            raise KeyError(f'not a node of this graph: {problem!r}')
        if not is_problem(self.definitions[problem]):
            raise ValueError(f'{problem!r} is not a problem, so it is nested in nothing and '
                             f'iterates nothing -- enclosing, cycle_of and interior are a problem\'s')
        chain : list[N] = []
        at = self.within.get(problem)
        while at is not None and at != problem and at not in chain:
            chain.append(at)
            at = self.within.get(at)
        return tuple(chain)

    @cached_property
    def _levels(self) -> dict[frozenset, AbstractGraph]:
        '''The graph with each set of enclosing problems taken out, built once per set: the
        problems nested alike share their level.'''
        return {}

    def cycle_of(self, problem: N) -> tuple[N, ...]:
        '''
        The cycle `problem` is on, and itself: its strongly connected component in the
        graph with the problems enclosing it taken out, in binding order.

        S_p = {k ∊ N \\ A(p) : k ~ p in N \\ A(p)}   (~ in the induced subgraph, not in G)
        '''
        out = frozenset(self.enclosing(problem))
        if (level := self._levels.get(out)) is None:
            level = self._levels[out] = self.graph.subgraph(set(self.nodes) - out)
        below = set(level.descendants([problem]))
        return tuple(n for n in level.ancestors([problem]) if n in below)

    def interior(self, problem: N) -> Self:
        '''
        What one iteration of `problem` runs: its cycle with it taken out.
        '''
        return self.subgraph(n for n in self.cycle_of(problem) if n != problem)

    # ------------------------------------------------------------ restriction

    def subgraph(self, keep: Iterable[N]) -> Self:
        '''
        Keep only those nodes in `keep`, with their definitions and the edges among them,
        dropping pairs with one end not kept.
        '''
        kept  = set(keep)
        child = self._carried(graph=self.graph.subgraph(kept),
                                within=OrderedMap((i, o) for i, o in self.within.items() if i in kept and o in kept))
        first(check_nesting_ends_on_one_cycle(child))
        return child

    def contract(self, blocks: tuple[tuple[N, ...], ...]) -> Self:
        '''
        Each block of at least two nodes contracted into one `Function` at its first
        member's place: reading what the block reads from outside, owning what it owns.
        Every pair with an end in a block is dropped, since a box is no problem. A pair
        outside keeps its cycle, since a contraction only adds paths, so nothing is re-asked.
        '''
        position = self.graph._position
        members = [n for block in blocks for n in block]
        if unknown := [n for n in members if n not in position]:
            raise KeyError(f'not node(s) of this graph: {unknown!r}')
        first(check_blocks_partition(self, blocks))
        orders = [tuple(sorted(block, key=position.__getitem__)) for block in blocks]
        first(check_blocks_own_something(self, orders))

        boxed : dict[N, NodeDefinition[V]] = {}
        for order in orders:
            inside = [self.definitions[m] for m in order]
            owns  = tuple(dict.fromkeys(v for d in inside for v in d.owns))
            owned = set(owns)
            reads = tuple(dict.fromkeys(v for d in inside for v in d.reads if v not in owned))
            boxed[order[0]] = Function(reads, owns)
        gone = set(members) - set(boxed)
        definitions = OrderedMap((n, boxed.get(n, d)) for n, d in self.definitions.items() if n not in gone)
        return self._carried(graph=self.graph._carried(definitions=definitions),
                             within=OrderedMap((i, o) for i, o in self.within.items()
                                               if i not in members and o not in members))


# ---------------------------------------------------------------- a proof about one

class ProvedGraph[N: Label, V: Place](Frozen):
    '''
    Wraps a nesting graph and forwards the methods. Additional checks can be put on here. 
    '''

    graph : NestingGraph[N, V]

    @property
    def definitions(self) -> Mapping[N, NodeDefinition[V]]:      return self.graph.definitions
    @property
    def nodes(self) -> tuple[N, ...]:                            return self.graph.nodes
    @property
    def within(self) -> Mapping[N, N]:                           return self.graph.within
    @property
    def problems(self) -> tuple[N, ...]:                         return self.graph.problems
    @property
    def outermost_problems(self) -> tuple[N, ...]:               return self.graph.outermost_problems
    def __getitem__(self, name: N) -> NodeDefinition[V]:          return self.graph[name]
    def __contains__(self, name: N) -> bool:                      return name in self.graph
    def enclosing(self, problem: N) -> tuple[N, ...]:             return self.graph.enclosing(problem)
    def cycle_of(self, problem: N) -> tuple[N, ...]:              return self.graph.cycle_of(problem)
    def interior(self, problem: N) -> NestingGraph[N, V]:         return self.graph.interior(problem)
    def subgraph(self, keep: Iterable[N]) -> NestingGraph[N, V]:  return self.graph.subgraph(keep)

    __iter__ = None
    '''Not iterable: `nodes` is the sequence and `definitions` the mapping.'''


# ----------------------------------------------------------------- violation finals

class Unbound(Violation):
    inner : Label
    outer : Label
    error    = KeyError
    template = 'nesting {inner!r} in {outer!r}: an end is not a node of this graph'


class NotAProblem(Violation):
    nodes : tuple[Label, ...]
    error    = ValueError
    template = ('{nodes} determines nothing, so no iteration of it answers anything -- an end '
                 'of a nesting is a problem: `Determine` it, naming the unknowns it fixes, or '
                 '`Unnest` it first so the plan says the nesting went')


class Unreachable(Violation):
    inner : Label
    outer : Label
    error    = ValueError
    template = ('{inner!r} and {outer!r} are not on one cycle, so no iteration of the one '
                 're-runs the other and there is nothing for the nesting to mean -- '
                 '`Unnest` it before cutting or rewiring the pair apart')


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


class BlockTooSmall(Violation):
    block : tuple[Label, ...]
    error    = ValueError
    template = 'block {block} is not two nodes -- a box contracts at least two into one'


class NodeInTwoBlocks(Violation):
    nodes : tuple[Label, ...]
    error    = ValueError
    template = '{nodes} each in two blocks -- a node is contracted into one box at most'


class BoxOwnsNothing(Violation):
    block : tuple[Label, ...]
    error    = ValueError
    template = ('block {block} owns nothing, so as one node it would compute nothing and be '
                 'unreachable -- a box is a computation, so contract only what produces')
