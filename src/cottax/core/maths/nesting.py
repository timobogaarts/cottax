'''
Nesting view with queries related to Valid nesting.
'''

from collections.abc import Mapping
from functools import cached_property

from .dependencies import Dependencies
from .valid import Valid
from .graph import Graph, View
from .kinds import enclosers, owns_inside
from .places import Label, Place


# ---------------------------------------------------------------- the view

class Nesting[N: Label, V: Place](View[N, V]):
    '''
    Nesting view: assumes Valid graph.
    '''
    checks = (Valid.check,)

    @cached_property
    def outermost(self) -> tuple[N, ...]:
        '''Outermost enclosers'''
        within = self.graph.within
        return tuple(n for n in enclosers(self.graph.definitions) if n not in within)

    def enclosing(self, encloser: N) -> tuple[N, ...]:
        '''
        A(q) = {n : q -> n , n ∊ N_p}

        with -> indicating a path in the forest W. (the set of all enclosers that enclose q, its nesting ancestors)
        '''
        if encloser not in self.graph.definitions:
            raise KeyError(f'not a node of this graph: {encloser!r}')
        if not owns_inside(self.graph.definitions[encloser]):
            raise ValueError(f'{encloser!r} owns nothing inside, so it is nested in nothing and '
                             f'iterates nothing -- enclosing, cycle_of and interior are an encloser\'s')
        chain : list[N] = []
        at = self.graph.within.get(encloser)
        while at is not None:
            chain.append(at)
            at = self.graph.within.get(at)
        return tuple(chain)

    @cached_property
    def _cycles(self) -> Mapping[N, tuple[N, ...]]:
        '''
        Each encloser's cycle, and itself: its strongly connected component in the graph with
        what it is nested in taken out, in binding order. 

        S_p = {k ∊ N \\ A(p) : k ~ p in N \\ A(p)}   (~ in the induced subgraph, not in G)
        '''
        levels : dict[frozenset[N], Dependencies[N, V]] = {}
        cycles : dict[N, tuple[N, ...]] = {}
        for e in enclosers(self.graph.definitions):
            out = frozenset(self.enclosing(e))
            if (deps := levels.get(out)) is None:
                deps = levels[out] = Dependencies.of(self.graph.subgraph(set(self.graph.nodes) - out))
            below = set(deps.descendants([e]))
            cycles[e] = tuple(n for n in deps.ancestors([e]) if n in below)
        return cycles

    def cycle_of(self, encloser: N) -> tuple[N, ...]:
        '''The cycle that encloser p has: its strongly connected component in the graph with what it is 
        nested in taken out. 

        S_p = {k ∊ N \\ A(p) : k ~ p in N \\ A(p)}   (~ in the induced subgraph, not in G)
        '''
        self.enclosing(encloser)                # refused as `enclosing` refuses a non-encloser
        return self._cycles[encloser]

    def interior(self, encloser: N) -> Graph[N, V]:
        '''
        See `cycle_of`: without that encloser itself.
        '''
        return self.graph.subgraph(n for n in self.cycle_of(encloser) if n != encloser)
