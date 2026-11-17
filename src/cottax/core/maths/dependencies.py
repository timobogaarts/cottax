'''
What follows from owns meeting reads: who owns each place, the edges, the components and
their order, reachability. A view of the graph, got by proving it well formed (`WellFormed`,
the two binding rules): only then does each place have one producer and each read meet an
owned place exactly, so an edge is a fact of the nodes and cannot disagree with them.
'''

from collections.abc import Iterable, Iterator, Mapping
from functools import cached_property

import networkx as nx

from ..frozen import Frozen, Violation, remembered
from .definition import NodeDefinition
from .graph import (
    Graph, check_nesting_ends_are_enclosers, check_nesting_ends_bound, check_no_nesting_loops,
)
from .places import Label, Place, comparables_in, is_prefix, overlaps
from .rules import Rules, proved


# ---------------------------------------------------------------- the rules: what the bindings obey

def check_one_producer_per_place(g: Graph) -> Iterator[Violation]:
    '''
    The owned places, over every node, form an antichain (do not overlap).

    o || m  ∀  n_i ≠ n_j,  n_i, n_j ∊ N:   o ∊ O_i, m ∊ O_j        
    '''
    yield from owned_overlaps((n, v) for n, d in g.definitions.items() for v in d.owns)


def owned_overlaps[N: Label, V: Place](owned: Iterable[tuple[N, V]]) -> Iterator[Violation]:
    '''Every comparable pair of `(owner, place)`s, the later against each earlier one it
    meets, so each violation names its own two owners.'''
    seen  : dict[V, N] = {}
    below : set[V]     = set()
    for node, place in owned:
        for other in comparables_in(place, seen, below):
            yield owned_comparably(place, node, other, seen[other])
        seen.setdefault(place, node)
        below.update(place.prefixes)


def check_reads_are_exact(g: Graph) -> Iterator[Violation]:
    '''
    No read is strictly inside or around another node's owned place: a read is
    exactly an owned place, or unrelated to every one.

    o || p  or  o = p   ∀ n_i, n_j ∊ N:  o ∊ O_i, p ∊ I_j        
    '''
    readers, owners = g.readers, owners_of(g.definitions)
    for place, other in overlaps(readers, owners):
        if other != place:                                   # equal is the (correct) edge
            yield (ReadInsideOwner if is_prefix(other, place) else OwnerInsideRead)(
                place, other, tuple(readers[place]), owners[other])


WellFormed = Rules('WellFormed', (check_nesting_ends_bound, check_no_nesting_loops,
                                  check_nesting_ends_are_enclosers,
                                  check_one_producer_per_place, check_reads_are_exact))
'''The graph is readable: its nesting is a forest over enclosers bound in it (the three rules
of `graph`), and its bindings obey the two above, so `owners` is a function and every edge
exact. What an op keeps (`apply`), and what a view is built on.'''


# ---------------------------------------------------------------- the view

class Dependencies[N: Label, V: Place](Frozen):
    '''
    The edges of a well-formed graph, and everything read off them. Built from a graph
    proved `WellFormed`: getting the view is the proof, so no query is left unguarded.
    '''

    graph : Graph[N, V]

    checks = (proved(WellFormed, at='graph'),)

    @cached_property
    def owners(self) -> dict[V, N]:
        '''Which V is output (owned) by which N. '''
        return owners_of(self.graph.definitions)

    @cached_property
    def components(self) -> tuple[tuple[N, ...], ...]:
        '''The strongly connected components, in dependency order, members in the order
        bound (TOSCC). Of the components ready at once, the one bound first comes first, so
        a graph bound in dependency order comes back as bound.'''
        if not self.graph.definitions:
            return ()               # an empty graph condenses to nothing
        rank      = {n: i for i, n in enumerate(self.graph.definitions)}
        condensed = nx.condensation(self._nx_dependencies)
        first     = {c: min(rank[m] for m in condensed.nodes[c]['members']) for c in condensed}
        return tuple(
            tuple(sorted(condensed.nodes[c]['members'], key=rank.__getitem__))
            for c in nx.lexicographical_topological_sort(condensed, key=first.__getitem__)
        )

    @cached_property
    def boundary_inputs(self) -> tuple[V, ...]:
        owned = set(self.graph.owned_variables)
        return tuple(var for var in self.graph.read_variables if var not in owned)

    @cached_property
    def boundary_outputs(self) -> tuple[V, ...]:
        '''Variables owned by a node but never read.'''
        read = set(self.graph.read_variables)
        return tuple(var for var in self.graph.owned_variables if var not in read)


    @cached_property
    def _nx_dependencies(self) -> nx.DiGraph:
        '''Dependency graph in networkx format.'''
        return nx_dependencies(self.graph.definitions)

    # ------------------------------------------------------------ order

    @cached_property
    def topological_order(self) -> tuple[N, ...]:
        '''
        The nodes in dependency order, raising if the graph has a cycle.
        '''
        if self.cycles:
            raise ValueError(f'computational graph has a dependency cycle: {self.cycles!r}')
        return self.components_order

    @cached_property
    def components_order(self) -> tuple[N, ...]:
        '''In run order; components are adjacent nodes'''
        return tuple(x for st in self.components for x in st)

    @cached_property
    def cycles(self) -> tuple[tuple[N, ...], ...]:
        '''
        A cycle iff the component is length larger than 1 (no self-loops)
        '''
        return tuple(c for c in self.components if len(c) > 1)

    @cached_property
    def is_acyclic(self) -> bool:        
        return not self.cycles

    # ------------------------------------------------------------ reachability

    def _walk(self, of: Iterable[N], *, up: bool) -> tuple[N, ...]:
        seeds = tuple(dict.fromkeys(of))
        if unknown := [n for n in seeds if n not in self.graph.definitions]:
            raise KeyError(f'not node(s) of this graph: {unknown!r}')
        deps  = self._nx_dependencies.reverse(copy=False) if up else self._nx_dependencies
        found = {n for layer in nx.bfs_layers(deps, seeds) for n in layer}
        return tuple(n for n in self.graph.definitions if n in found)     # in binding order

    def ancestors(self, of: Iterable[N]) -> tuple[N, ...]:
        '''Every node `of` depends on (including itself)'''
        return self._walk(of, up=True)

    def descendants(self, of: Iterable[N]) -> tuple[N, ...]:
        '''Every node depending on `of` (including itself)'''
        return self._walk(of, up=False)

    def reach(self, varying: Iterable[V]) -> tuple[N, ...]:
        '''Every node depending on the *variables* `varying` (except the owner itself)'''
        varying = tuple(dict.fromkeys(varying))
        known   = set(self.graph.variables)
        if unknown := [v for v in varying if v not in known]:
            raise KeyError(f'not variable(s) of this graph: {unknown!r}')
        return self.descendants(n for v in varying for n in self.graph.readers.get(v, ()))

    # ------------------------------------------------------------ cycles

    @cached_property
    def feedback_variables(self) -> tuple[V, ...]:
        '''
        Which variables are read in the same SCC as their producer.
        '''
        at = {n: i for i, c in enumerate(self.components) for n in c}
        return tuple(v for v, n in self.owners.items()
                     if any(at[r] == at[n] for r in self.graph.readers.get(v, ())))

    def closing_readers(self, var: V) -> tuple[N, ...]:
        '''The readers of `var` that close a cycle back onto its producer -- the
        minimal set that has to stop reading the computed value for the cycle to break.'''
        owner = self.owners.get(var)
        if owner is None:
            raise KeyError(f'{var!r} has no producer, so nothing closes a cycle on it')
        return tuple(reader for reader in self.graph.readers.get(var, ())
                     if owner in self.descendants([reader]))


@remembered
def dependencies[N: Label, V: Place](graph: Graph[N, V]) -> Dependencies[N, V]:
    '''The dependency view of `graph`, proving it `WellFormed`; one per graph.'''
    return Dependencies(graph)


# ---------------------------------------------------------------- derived maps

def owners_of[N: Label, V: Place](definitions: Mapping[N, NodeDefinition[V]]) -> dict[V, N]:
    '''Who owns each place. One producer per place is `check_one_producer_per_place`'s rule; assumed here.'''
    return {v: n for n, d in definitions.items() for v in d.owns}


# ---------------------------------------------------------------- the projection

def nx_dependencies[N: Label](definitions: Mapping[N, NodeDefinition]) -> nx.DiGraph:
    '''Networkx dependency graph'''
    owners = owners_of(definitions)
    g = nx.DiGraph()
    g.add_nodes_from(definitions)
    for m, d in definitions.items():
        for v in d.reads:
            if (n := owners.get(v)) is not None:
                g.add_edge(n, m)
    return g


# ----------------------------------------------------------------- violation finals

class OwnedTwice(Violation):
    place   : Place
    node    : Label
    other   : Label
    error    = ValueError
    template = 'output {place!r} is owned by both {other!r} and {node!r}'

class OwnInsideOwner(Violation):
    place   : Place
    witness : Place
    node    : Label
    other   : Label
    error    = ValueError
    template = ('output {place!r} of {node!r} lies inside {witness!r}, owned by {other!r} -- '
                 'two nodes would determine one place')

class OwnAroundOwner(Violation):
    place   : Place
    witness : Place
    node    : Label
    other   : Label
    error    = ValueError
    template = ('output {witness!r} of {other!r} lies inside {place!r}, owned by {node!r} -- '
                 'two nodes would determine one place')

class ReadInsideOwner(Violation):
    place   : Place
    witness : Place
    readers : tuple[Label, ...]
    owner   : Label
    error    = ValueError
    template = ('{readers} read {place!r}, which lies inside {witness!r}, owned by {owner!r} -- '
                 'reads are matched by equality, so this would silently be a boundary input')

class OwnerInsideRead(Violation):
    place   : Place
    witness : Place
    readers : tuple[Label, ...]
    owner   : Label
    error    = ValueError
    template = ('output {witness!r} of {owner!r} lies inside {place!r}, read by {readers} -- '
                 'that read would silently miss this output')

#--------------------------------------------------------------------- Violation pickers
# These always yield a violation, separate from the main function to reduce them to just
# checking whether there is a violation.
def owned_comparably[N: Label, V: Place](place: V, node: N, other: V, owner: N) -> Violation:
    '''The violation for `place`, owned by `node`, comparable to `other`, owned by
    `owner`.'''
    if other == place:
        return OwnedTwice(place, node, owner)
    return (OwnInsideOwner if is_prefix(other, place) else OwnAroundOwner)(place, other, node, owner)
