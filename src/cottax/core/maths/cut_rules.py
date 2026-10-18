'''
Rules on how to cut a graph's couplings.
'''

from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from functools import cached_property

from ..frozen import Frozen, Violation, first
from .graph import AbstractGraph
from .kinds import FunctionalNode
from .nesting import NestingGraph
from .places import Label, Place


type ToCut[N: Label, V: Place] = Mapping[V, tuple[N, ...]]
'''One closure's worth: which variables are cut, for which readers.'''


# ---------------------------------------------------------------- a rule, and where it is asked

class CutRule[N: Label, V: Place](Frozen):
    '''What to cut in the graph handed: the coupling of its function nodes, one
    group per closure.'''

    @abstractmethod
    def to_cuts(self, graph: AbstractGraph[N, V]) -> tuple[ToCut[N, V], ...]:
        '''One group per closure.'''


# ---------------------------------------------------------------- apply rule per functional cycle
type Opened[N: Label, V: Place] = tuple[tuple[N, ...], tuple[ToCut[N, V], ...]]
'''One cycle opened: its members in binding order, and the groups the rule cut it into.'''


def cuts_per_cycle[N: Label, V: Place](graph: NestingGraph[N, V], rule: CutRule[N, V]
                                       ) -> tuple[Opened[N, V], ...]:
    '''
    The rule asked of each functional cycle: the condition nodes dropped, and each
    *cyclic* component of what is left handed to the rule, alone. Per cycle, its members
    and its groups.
    '''
    bodies = graph.graph.subgraph(n for n, d in graph.definitions.items()
                                  if isinstance(d, FunctionalNode))
    return tuple((component, rule.to_cuts(bodies.subgraph(component)))
                 for component in bodies.components if len(component) > 1)


def per_functional_cycle[N: Label, V: Place](graph: NestingGraph[N, V], rule: CutRule[N, V]
                                             ) -> tuple[ToCut[N, V], ...]:
    '''`cuts_per_cycle`'s groups, every cycle's in a row.'''
    return tuple(group for _, groups in cuts_per_cycle(graph, rule) for group in groups)


# ---------------------------------------------------------------- Coupling helpers and CutRule implementation

def coupling_reads[N: Label, V: Place](graph: AbstractGraph[N, V]) -> ToCut[N, V]:
    '''Every read of every variable owned in the graph handed, holding nothing back.'''
    return {var: readers for var in graph.owners if (readers := graph.readers.get(var, ()))}


def gauss_seidel[N: Label, V: Place](graph: AbstractGraph[N, V],
                                     order: Sequence[N]) -> ToCut[N, V]:
    '''`coupling_reads` with every reader that follows its owner in `order` dropped:
    only the back edges are opened.'''
    at = {n: i for i, n in enumerate(order)}
    out: dict[V, tuple[N, ...]] = {}
    for var, readers in coupling_reads(graph).items():
        if back := tuple(r for r in readers if at[r] < at[graph.owners[var]]):
            out[var] = back
    return out


def minimal_feedback_order[N: Label, V: Place](graph: AbstractGraph[N, V],
                                               limit: int) -> tuple[N, ...]:
    '''
    The order of `graph`'s nodes a Gauss-Seidel sweep opens the fewest variables in.
    Exact: a DP over subsets, a node's cost after a set being how many of its
    couplings that set reads. Ties fall to binding order.
    '''
    nodes = graph.nodes
    n = len(nodes)
    first(() if n <= limit else (CycleTooLarge(n, limit),))
    index = {node: i for i, node in enumerate(nodes)}
    # owned[i]: one bitmask per coupling variable node i owns, of its readers. The
    # variable is cut iff any reader precedes i.
    owned: list[list[int]] = [[] for _ in nodes]
    for var, readers in coupling_reads(graph).items():
        mask = 0
        for reader in readers:
            mask |= 1 << index[reader]
        owned[index[graph.owners[var]]].append(mask)
    best: list = [None] * (1 << n)          # (cost, last node) per subset placed as a prefix
    best[0] = (0, -1)
    for subset in range(1 << n):
        if best[subset] is None:
            continue
        cost, _ = best[subset]
        for i in range(n):
            bit = 1 << i
            if subset & bit:
                continue
            added = sum(1 for mask in owned[i] if mask & subset)
            if best[subset | bit] is None or cost + added < best[subset | bit][0]:
                best[subset | bit] = (cost + added, i)
    order, subset = [], (1 << n) - 1
    while subset:
        _, i = best[subset]
        order.append(nodes[i])
        subset &= ~(1 << i)
    return tuple(reversed(order))


def check_keys_distinct(rule: 'ByName') -> Iterator[Violation]:
    '''No node keys two rules: which one applies would be the order written.'''
    keys = [n for n, _ in rule.rules]
    if repeated := tuple(dict.fromkeys(k for k in keys if keys.count(k) > 1)):
        yield KeyRepeats(tuple(str(k) for k in repeated))


class ByName[N: Label, V: Place](CutRule[N, V]):
    '''
    A rule per cycle, keyed by a node on it. Cycles are disjoint, so exactly one key
    lies on the graph handed; none or several is refused, never the first taken.
    '''

    rules : tuple[tuple[N, CutRule[N, V]], ...]

    checks = (check_keys_distinct,)

    @cached_property
    def _index(self) -> dict[N, CutRule[N, V]]:
        return dict(self.rules)

    def to_cuts(self, graph: AbstractGraph[N, V]) -> tuple[ToCut[N, V], ...]:
        keyed = tuple(n for n in graph.nodes if n in self._index)
        first(() if len(keyed) == 1
              else (NoRule(tuple(str(n) for n in graph.nodes)),) if not keyed
              else (SeveralRules(tuple(str(n) for n in keyed)),))
        return self._index[keyed[0]].to_cuts(graph)

    def __repr__(self) -> str:
        return f'by_name({", ".join(f"{n}: {r!r}" for n, r in self.rules)})'


class NoCuts[N: Label, V: Place](CutRule[N, V]):
    '''Nothing opened, said rather than fallen into.'''

    def to_cuts(self, graph: AbstractGraph[N, V]) -> tuple[ToCut[N, V], ...]:
        return ()

    def __repr__(self) -> str:
        return 'no_cuts'


class Jacobi[N: Label, V: Place](CutRule[N, V]):
    '''Every coupling, for every reader, in one closure: each node runs from the
    previous iterate alone.'''

    def to_cuts(self, graph: AbstractGraph[N, V]) -> tuple[ToCut[N, V], ...]:
        group = coupling_reads(graph)
        return (group,) if group else ()

    def __repr__(self) -> str:
        return 'jacobi'


class Sweep[N: Label, V: Place](CutRule[N, V]):
    '''A Gauss-Seidel sweep: the back edges of an order only, in one closure, since a
    forward edge keeps this sweep's value.'''

    @abstractmethod
    def order_of(self, graph: AbstractGraph[N, V]) -> tuple[N, ...]:
        '''The order this graph is swept in, naming its nodes exactly once each.'''

    def to_cuts(self, graph: AbstractGraph[N, V]) -> tuple[ToCut[N, V], ...]:
        group = gauss_seidel(graph, self.order_of(graph))
        return (group,) if group else ()


class GaussSeidel[N: Label, V: Place](Sweep[N, V]):
    '''In binding order.'''

    def order_of(self, graph: AbstractGraph[N, V]) -> tuple[N, ...]:
        return graph.nodes

    def __repr__(self) -> str:
        return 'gauss_seidel'


class GaussSeidelMinimal[N: Label, V: Place](Sweep[N, V]):
    '''In the order `minimal_feedback_order` finds: the fewest variables opened.'''

    limit : int
    '''Above this many nodes on one cycle the exact search (`2^n n`) is refused.'''

    def order_of(self, graph: AbstractGraph[N, V]) -> tuple[N, ...]:
        return minimal_feedback_order(graph, self.limit)

    def __repr__(self) -> str:
        return f'gauss_seidel_minimal(limit={self.limit})'


class GaussSeidelOrdered[N: Label, V: Place](Sweep[N, V]):
    '''In the order written, which names the graph's nodes exactly once each or is
    refused: an order meant for one cycle and got wrong is what leaves it unswept.'''

    order : tuple[N, ...]

    def order_of(self, graph: AbstractGraph[N, V]) -> tuple[N, ...]:
        first(() if len(self.order) == len(graph.nodes) and set(self.order) == set(graph.nodes)
              else (OrderMismatch(tuple(str(n) for n in self.order),
                                  tuple(str(n) for n in graph.nodes)),))
        return self.order

    def __repr__(self) -> str:
        return f'gauss_seidel({", ".join(map(str, self.order))})'


# ---------------------------------------------------------------- violation finals

class OrderMismatch(Violation):
    order : tuple
    nodes : tuple
    error    = ValueError
    template = ('gauss_seidel: the order {order!r} does not name the nodes {nodes!r} exactly '
                'once each -- an order sweeps one cycle, and one per cycle is `ByName`')


class CycleTooLarge(Violation):
    size  : int
    limit : int
    error    = ValueError
    template = ('{size} nodes on one cycle is beyond the exact order search (limit {limit}): '
                'raise `GaussSeidelMinimal(limit=)`, or hand `GaussSeidelOrdered` an order')


class KeyRepeats(Violation):
    keys : tuple
    error    = ValueError
    template = 'by_name: {keys!r} each key more than one rule'


class NoRule(Violation):
    nodes : tuple
    error    = ValueError
    template = ('no rule keyed by a node of {nodes!r}: key one, `NoCuts()` to leave it '
                'alone, or use the one rule alone for every cycle')


class SeveralRules(Violation):
    keyed : tuple
    error    = ValueError
    template = ('several rules keyed on one cycle, by {keyed!r}: cycles are disjoint, so '
                'one key lies on one')
