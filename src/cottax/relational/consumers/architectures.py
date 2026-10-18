'''Registers how an architecture places a relational condition (`ModelReading`), with why one
is no implicit model; offers its closing, `FixedPointClosing`, each group of cuts closed by a
`FixedPointCut`.'''

from ...core import Cut, FunctionalNode, Label, Place, PlaceToPlace, ToCut, Violation
from ...core.frozen import answers, choice
from ...consumers.architectures.interface import Closing, ModelReading
from ...consumers.architectures.interface_types import At
from ...core.driven import condition_of
from ..classification import is_equalities
from ..relational_condition import RelationalCondition
from ..rewrites import FixedPointCut


@answers(RelationalCondition)
class RelationalModel(ModelReading):
    '''An implicit model where it states equalities alone, each condition written by a
    function node and every unknown read by one of them; its body those functions, a closed
    cut beside them changing nothing, the copy converging to what it copies; folded by its
    `+`.'''

    def not_implicit(self, graph, problem):
        node, owners = graph[problem], graph.graph.owners
        if not is_equalities(condition_of(node)):
            return (NotEqualities(str(problem)),)
        if off := tuple(c for c in node.conditions
                        if c not in owners or not isinstance(graph[owners[c]], FunctionalNode)):
            return (ConditionOffBody(str(problem), tuple(map(str, off))),)
        read = {v for b in self.body(graph, problem) for v in graph[b].reads}
        if unread := tuple(u for u in node.unknowns if u not in read):
            return (UnknownUnread(str(problem), tuple(map(str, unread))),)
        return ()

    def body(self, graph, problem):
        owners = graph.graph.owners
        return tuple(dict.fromkeys(owners[c] for c in graph[problem].conditions if c in owners))

    def combines_as(self, condition):
        return RelationalCondition


@choice(Closing)
class FixedPointClosing[N: Label, V: Place](Closing[N, V]):
    '''One group closed as one problem: a copy per variable, named by `copies`, and
    the closure bound where `at` says.'''

    copies : PlaceToPlace[V]
    at     : At[N, V]

    def __call__(self, cycle: tuple[N, ...], group: ToCut[N, V]) -> FixedPointCut[N, V]:
        return FixedPointCut(tuple(Cut(var, readers, self.copies) for var, readers in group.items()),
                             self.at(cycle, group))

    def bound_at(self, cycle: tuple[N, ...], group: ToCut[N, V]) -> N:
        return self.at(cycle, group)

    def __repr__(self) -> str:
        return f'closing({self.copies!r}, {self.at!r})'


class NotEqualities(Violation):
    problem : str
    error    = ValueError
    template = ('{problem} is not an implicit model: it states an objective or an inequality, '
                'so it is a design and not a model -- name it as the top, or leave it out')


class ConditionOffBody(Violation):
    problem    : str
    conditions : tuple
    error    = ValueError
    template = ('{problem} is not an implicit model: {conditions!r} not owned by a function '
                'node -- a loop closed through another problem is one solve written as '
                'several; `Combine` them first')


class UnknownUnread(Violation):
    problem  : str
    unknowns : tuple
    error    = ValueError
    template = ('{problem} is not an implicit model: its body does not read {unknowns!r} '
                '-- a residual computed through a chain is not a discipline; make the '
                'chain its body')
