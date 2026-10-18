'''
The edits putting a driver on and taking it off.
'''

from collections.abc import Callable, Iterator, Mapping

from ...core import (
    ComposedOp, ConditionalNode, Label, NestingGraph, NodeDefinition, Place, Redefine, Violation,
    violations,
)
from .driver import DriverContract
from .nodes import Driven, DrivenTwice, is_driven


def check_drivable(op: 'Assign', graph: NestingGraph) -> Iterator[Violation]:
    '''An algorithm goes on a condition node without one, and the node it makes is well
    formed -- asked here, so a refusal is data before the graph is touched. The kind is
    asked for `Implement`'s reason: the wrapper would otherwise wrap the wrong level. A
    second driver is asked first, since its reports would otherwise read as owned twice.'''
    node = graph[op.problem]
    if not isinstance(node, ConditionalNode):
        yield NotAConditionalNode(op.problem, type(node).__name__)
    elif is_driven(node):
        yield DrivenTwice(node, type(op.contract).__name__)
    else:
        yield from violations(Driven, node, op.contract, op.data, op.reports)


class Assign[N: Label, V: Place](Redefine[N, V]):
    '''Attach a driver to a condition node, and say where each of its roles sits: the
    algorithm choice and its places, recorded in the plan and carried by the node. Where a
    place sits is a decision, so the op records it: a surface supplies the obvious ones.'''

    problem : N
    contract  : DriverContract[V]
    data    : tuple[tuple[V, ...], ...]
    reports : tuple[tuple[V, ...], ...]

    checks_graph = (check_drivable,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.problem: Driven(graph[self.problem], self.contract, self.data, self.reports)}

    def __repr__(self) -> str:
        return f'assign({self.problem} <- {type(self.contract).__name__})'


def check_assignable(op: 'AssignByRule', graph: NestingGraph) -> Iterator[Violation]:
    '''A condition node to read the unknowns of, before the places are made from them.'''
    node = graph[op.problem]
    if not isinstance(node, ConditionalNode):
        yield NotAConditionalNode(op.problem, type(node).__name__)


class AssignByRule[N: Label, V: Place](ComposedOp[N, V]):
    '''`Assign` at the places `places(problem, unknowns, contract)` gives, the problem's
    unknowns read when it is applied: the rule is recorded, so a changed model has its places
    derived again.'''

    problem  : N
    contract : DriverContract[V]
    places   : Callable[..., tuple[tuple[tuple[V, ...], ...], tuple[tuple[V, ...], ...]]]

    checks_graph = (check_assignable,)

    def composed(self, graph: NestingGraph[N, V]) -> tuple[Assign[N, V]]:
        unknowns = graph[self.problem].unknowns
        return (Assign(self.problem, self.contract, *self.places(self.problem, unknowns,
                                                                 self.contract)),)

    def __repr__(self) -> str:
        return f'assign({self.problem} <- {type(self.contract).__name__}, at its default places)'


def check_undrives_a_driven_node(op: 'Undrive', graph: NestingGraph) -> Iterator[Violation]:
    '''There is an algorithm to drop.'''
    node = graph[op.problem]
    if not is_driven(node):
        yield NotDriven(op.problem, type(node).__name__)


class Undrive[N: Label, V: Place](Redefine[N, V]):
    '''The inverse of `Assign`: the condition node alone again, and the driver's places
    gone with it -- `undriven`, asked of the node, so a wrapper refuses for itself.'''

    problem : N

    checks_graph = (check_undrives_a_driven_node,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.problem: graph[self.problem].undriven()}

    def __repr__(self) -> str:
        return f'undrive({self.problem})'


# ----------------------------------------------------------------- violation finals

class NotAConditionalNode(Violation):
    problem : object
    kind    : str
    error    = TypeError
    template = ('{problem!r} is a {kind}, not a condition node, so there is nothing for a driver '
                'to answer')


class NotDriven(Violation):
    problem : object
    kind    : str
    error    = TypeError
    template = ('{problem!r} is a {kind} and carries no driver, so there is nothing to '
                'undrive')
