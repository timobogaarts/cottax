'''
`Combine`: several condition nodes folded into one, by the join their condition's kind
answers (`ModelReading.join`). An architecture's fold, and a study's by hand; not core's,
since what joins two conditions is said by the language that states them.
'''

from collections.abc import Collection, Iterator
from functools import reduce

from ...core import (
    ConditionalNode, Label, Graph, Place, Rebind, Rebinding, Violation, is_driven, kind_of,
)
from ...core.frozen import OrderedMap, answer_for_class
from ...core.maths.graph import NameTaken
from .interface import ModelReading
from .refusals import (
    CarriesAttributes, FoldsDriven, NodeTwice, NotAddable, NotOfKind, NothingToCombine, Stranded,
)


def stranded[N: Label](g: Graph[N, Place], removed: Collection[N]) -> Iterator[Violation]:
    '''A nesting pair losing exactly one end to `removed`. A nesting is a decision the plan
    recorded, so it is not dropped silently: `Unnest` first. Losing both ends is fine.'''
    if broken := [(i, o) for i, o in g.within.items() if (i in removed) != (o in removed)]:
        yield Stranded(tuple(i for i, _ in broken), tuple(o for _, o in broken))


def join_of(kind: type):
    '''The join `kind`'s `ModelReading` answers, `None` for a kind folded into nothing or
    answered by none.'''
    reading = answer_for_class(ModelReading, kind, None)
    return None if reading is None else reading.join


def check_nodes(op: 'Combine') -> Iterator[Violation]:
    '''At least two nodes, none twice, of a kind that joins.'''
    if len(op.nodes) < 2:
        yield NothingToCombine()
    if len(set(op.nodes)) != len(op.nodes):
        yield NodeTwice(tuple(op.nodes))
    if join_of(op.kind) is None:
        yield NotAddable(op.kind.__name__)


def check_folds_one_kind(op: 'Combine', graph: Graph) -> Iterator[Violation]:
    '''Every folded node is a condition node whose condition is of the declared kind, so the
    kind's join folds them.'''
    off = tuple(n for n in op.nodes
                if not isinstance(graph[n], ConditionalNode) or not isinstance(kind_of(graph[n]), op.kind))
    if off:
        found = dict.fromkeys(type(kind_of(graph[n])).__name__ for n in off)
        yield NotOfKind(off, op.kind.__name__, ', '.join(found))


def check_folds_no_attributes(op: 'Combine', graph: Graph) -> Iterator[Violation]:
    '''No folded node carries an attribute: what two nodes carry has no join.'''
    for name in op.nodes:
        if name in graph.definitions and (carried := graph[name].attributes):
            yield CarriesAttributes(name, tuple(k.__name__ for k in carried))


def check_folds_undriven(op: 'Combine', graph: Graph) -> Iterator[Violation]:
    '''No folded node carries a driver: which algorithm answers a join is not read off the
    two, so a fold is of conditions, driven after.'''
    if driven := tuple(n for n in op.nodes if n in graph.definitions and is_driven(graph[n])):
        yield FoldsDriven(driven)


def check_lands_on_its_own(op: 'Combine', graph: Graph) -> Iterator[Violation]:
    '''The fold lands on a node it folds, or on a name nothing holds.'''
    if op.at not in op.nodes and op.at in graph.definitions:
        yield NameTaken(op.at)


def check_fold_keeps_nesting_whole[N: Label](op: 'Combine', g: Graph[N, Place]
                                            ) -> Iterator[Violation]:
    '''No nesting pair loses one end to the nodes folded away (all but the one at `at`).'''
    yield from stranded(g, set(op.nodes) - {op.at})


class Combine[N: Label, V: Place](Rebind[N, V]):
    '''
    Fold several condition nodes into one by their kind's join: the others are removed and
    the fold is defined at `at`, in one rebinding. Whether the fold fits the graph depends on
    the join, so that is checked on the result.
    '''

    at : N
    '''Where the fold lands: one of `nodes` (redefined in place) or a new name.'''

    nodes : tuple[N, ...]

    kind : type
    '''The condition class whose join folds the nodes; each node's condition must be one.
    The op's intent, so a field: a replay against a changed model refuses rather than
    folding what it finds.'''

    checks = (check_nodes,)
    checks_graph = (check_folds_one_kind, check_folds_no_attributes, check_folds_undriven,
                    check_lands_on_its_own, check_fold_keeps_nesting_whole)

    def definition(self, graph: Graph[N, V]) -> ConditionalNode[V]:
        '''The node holding the join of the folded conditions, in the order given.'''
        join = join_of(self.kind)
        return ConditionalNode(reduce(join, (graph[n].condition for n in self.nodes)))

    def rebinding(self, graph: Graph[N, V]) -> Rebinding[N, V]:
        return Rebinding(frozenset(self.nodes) - {self.at}, OrderedMap({self.at: self.definition(graph)}))

    def __repr__(self) -> str:
        return f'combine({self.at} <- {", ".join(str(n) for n in self.nodes)})'
