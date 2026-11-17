'''
The nesting read with the edges: the cycle a problem is on, what one iteration of it runs,
blocks contracted into boxes. A view of a well-formed graph, as `dependencies` is.
'''

from collections.abc import Iterable, Iterator
from dataclasses import replace

from functools import cached_property

from ..frozen import Frozen, OrderedMap, Violation, first, remembered
from .definition import NodeDefinition
from .dependencies import WellFormed, dependencies
from .graph import Graph
from .kinds import FunctionalNode, owns_inside
from .places import Label, Place
from .rules import prove, proved


# ---------------------------------------------------------------- the rules

def check_nesting_ends_on_one_cycle(g: 'Graph') -> Iterator[Violation]:
    '''
    Both ends of a nesting are on one cycle: the inner lies in the outer's cycle. The
    inner's output is required for the outer's inputs and vice-versa -- otherwise the
    inner does not depend on the outer's iterate, or the outer never reads its answer,
    and it could run outside.

    p ~ q  in  N \\ A(q)   ∀ (p, q) ∊ W       i.e.  p ∊ S_q, the cycle of q

    Not a rule of `WellFormed`: nothing that reads a nesting needs it, so a plan may cut
    a nested pair apart and `Unnest` it after. `ExecutableGraph` asks it.
    '''
    for inner, outer in g.within.items():
        if inner not in nesting(g).cycle_of(outer):
            yield Unreachable(inner, outer)


def check_blocks_partition(g: 'Graph', blocks: Iterable[Iterable]) -> Iterator[Violation]:
    '''Each block is at least two nodes, and no node is in two.'''
    position, seen, twice = g._position, set(), {}
    for block in blocks:
        if len(set(block)) < 2:
            yield BlockTooSmall(tuple(block))
        for n in dict.fromkeys(block):
            if n in seen:
                twice[n] = None
            seen.add(n)
    if twice:
        yield NodeInTwoBlocks(tuple(sorted(twice, key=position.__getitem__)))


def check_blocks_own_something(g: 'Graph', blocks: Iterable[Iterable]) -> Iterator[Violation]:
    '''A block contracts into a computation, so it owns something.'''
    for block in blocks:
        if not any(g.definitions[n].owns for n in block):
            yield BoxOwnsNothing(tuple(block))


# ---------------------------------------------------------------- the view

class Nesting[N: Label, V: Place](Frozen):
    '''
    The cycles of a well-formed graph's problems, read with the nesting. Built from a graph
    proved `WellFormed`: getting the view is the proof.
    '''

    graph : Graph[N, V]

    checks = (proved(WellFormed, at='graph'),)

    @cached_property
    def outermost(self) -> tuple[N, ...]:
        '''The nodes that iterate a cycle (own something inside) and are nested in none, in
        binding order: what the whole graph answers by a solve of its own.'''
        within = self.graph.within
        return tuple(n for n, d in self.graph.definitions.items()
                     if owns_inside(d) and n not in within)

    def enclosing(self, problem: N) -> tuple[N, ...]:
        '''What `problem` is nested in, transitively, innermost first: A(q), every node with
        a path from q in (N_P, W). A forest (`WellFormed`), so the walk ends.'''
        if problem not in self.graph.definitions:
            raise KeyError(f'not a node of this graph: {problem!r}')
        if not owns_inside(self.graph.definitions[problem]):
            raise ValueError(f'{problem!r} owns nothing inside, so it is nested in nothing and '
                             f'iterates nothing -- enclosing, cycle_of and interior are a problem\'s')
        chain : list[N] = []
        at = self.graph.within.get(problem)
        while at is not None:
            chain.append(at)
            at = self.graph.within.get(at)
        return tuple(chain)

    @cached_property
    def _levels(self) -> dict[frozenset, Graph[N, V]]:
        '''The graph with each set of problems taken out, built once per set: the problems
        nested alike share their level, and its dependencies.'''
        return {}

    def cycle_of(self, problem: N) -> tuple[N, ...]:
        '''
        The cycle `problem` is on, and itself: its strongly connected component in the
        graph with what it is nested in taken out, in binding order.

        S_p = {k ∊ N \\ A(p) : k ~ p in N \\ A(p)}   (~ in the induced subgraph, not in G)
        '''
        out = frozenset(self.enclosing(problem))
        if (level := self._levels.get(out)) is None:
            level = self._levels[out] = self.graph.subgraph(set(self.graph.nodes) - out)
        deps  = dependencies(level)
        below = set(deps.descendants([problem]))
        return tuple(n for n in deps.ancestors([problem]) if n in below)

    def interior(self, problem: N) -> Graph[N, V]:
        '''
        What one iteration of `problem` runs: its cycle with it taken out. A value of its
        own: what is proved of it is asked of it.
        '''
        return self.graph.subgraph(n for n in self.cycle_of(problem) if n != problem)

    def contract(self, blocks: tuple[tuple[N, ...], ...]) -> Graph[N, V]:
        '''
        Each block of at least two nodes contracted into one `FunctionalNode` at its first
        member's place: reading what the block reads from outside, owning what it owns.
        Every pair with an end in a block is dropped, since a box is no encloser. Every rule
        is asked of the result: the nesting's by its constructor, then `WellFormed`.
        '''
        position = self.graph._position
        members = [n for block in blocks for n in block]
        if unknown := [n for n in members if n not in position]:
            raise KeyError(f'not node(s) of this graph: {unknown!r}')
        first(check_blocks_partition(self.graph, blocks))
        orders = [tuple(sorted(block, key=position.__getitem__)) for block in blocks]
        first(check_blocks_own_something(self.graph, orders))

        boxed : dict[N, NodeDefinition[V]] = {}
        for order in orders:
            inside = [self.graph.definitions[m] for m in order]
            owns  = tuple(dict.fromkeys(v for d in inside for v in d.owns))
            owned = set(owns)
            reads = tuple(dict.fromkeys(v for d in inside for v in d.reads if v not in owned))
            boxed[order[0]] = FunctionalNode(reads, owns)
        gone = set(members) - set(boxed)
        definitions = OrderedMap((n, boxed.get(n, d)) for n, d in self.graph.definitions.items() if n not in gone)
        result = replace(self.graph, definitions=definitions,
                         within=OrderedMap((i, o) for i, o in self.graph.within.items()
                                           if i not in members and o not in members))
        prove(result, WellFormed)
        return result



@remembered
def nesting[N: Label, V: Place](graph: Graph[N, V]) -> Nesting[N, V]:
    '''The nesting view of `graph`, proving it `WellFormed`; one per graph.'''
    return Nesting(graph)


# ----------------------------------------------------------------- violation finals

class Unreachable(Violation):
    inner : Label
    outer : Label
    error    = ValueError
    template = ('{inner!r} and {outer!r} are not on one cycle, so no iteration of the one '
                 're-runs the other and there is nothing for the nesting to mean -- '
                 '`Unnest` it, or keep the pair on one cycle')


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
