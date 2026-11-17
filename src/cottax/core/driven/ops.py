'''
The edits putting a driver on and taking it off.
'''

import dataclasses
from collections.abc import Callable, Iterator, Mapping

from ...core import (
    ComposedOp, Condition, ConditionalNode, Label, Graph, NodeDefinition, Place, Redefine, Violation,
    is_driven, violations,
)
from .driver import Driver, DriverContract


def driven(node: ConditionalNode | Condition, contract: DriverContract, data: tuple = (),
           reports: tuple = ()) -> ConditionalNode:
    '''`node` answered by `contract`, its roles at `data` and `reports`: how a node is built
    driven, outside a plan. A condition is taken as the node holding it.'''
    if isinstance(node, Condition):
        return ConditionalNode(node, driver=Driver(contract, data, reports))
    return dataclasses.replace(node, driver=Driver(contract, data, reports))


def check_drivable(op: 'Assign', graph: Graph) -> Iterator[Violation]:
    '''An algorithm goes on a condition node without one, and the node it makes is well
    formed -- asked here, so a refusal is data before the graph is touched.'''
    node = graph[op.problem]
    if not isinstance(node, ConditionalNode):
        yield NotAConditionalNode(op.problem, type(node).__name__)
    else:
        yield from driving_violations(node, op.contract, op.data, op.reports)


def driving_violations(node: ConditionalNode, contract: DriverContract, data: tuple = (),
                       reports: tuple = ()) -> Iterator[Violation]:
    '''Every rule `driven(node, contract, data, reports)` breaks, as data. A second driver
    is refused first, since its reports would otherwise read as owned twice. Then the node's
    place rules, then whether the driver fits it (`proof.check_driver_fits`).'''
    if is_driven(node):
        yield DrivenTwice(node, type(contract).__name__)
        return
    driver = Driver(contract, data, reports)
    yield from violations(ConditionalNode, node.condition, driver=driver, attributes=node.attributes)
    yield from driver.violations_on(node.condition)


class Assign[N: Label, V: Place](Redefine[N, V]):
    '''Attach a driver to a condition node, and say where each of its roles sits: the
    algorithm choice and its places, recorded in the plan and carried by the node. Where a
    place sits is a decision, so the op records it: a surface supplies the obvious ones.'''

    problem : N
    contract  : DriverContract[V]
    data    : tuple[tuple[V, ...], ...]
    reports : tuple[tuple[V, ...], ...]

    checks_graph = (check_drivable,)

    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.problem: driven(graph[self.problem], self.contract, self.data, self.reports)}

    def __repr__(self) -> str:
        return f'assign({self.problem} <- {type(self.contract).__name__})'


def check_assignable(op: 'AssignByRule', graph: Graph) -> Iterator[Violation]:
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

    def composed(self, graph: Graph[N, V]) -> tuple[Assign[N, V]]:
        unknowns = graph[self.problem].owns_inside
        return (Assign(self.problem, self.contract, *self.places(self.problem, unknowns,
                                                                 self.contract)),)

    def __repr__(self) -> str:
        return f'assign({self.problem} <- {type(self.contract).__name__}, at its default places)'


def check_undrives_a_driven_node(op: 'Undrive', graph: Graph) -> Iterator[Violation]:
    '''There is an algorithm to drop. What the node carries is not asked here: whether it
    fits a node without a driver is its own layer's rule (an implementation's is
    `realisation`'s, asked where that is proved).'''
    node = graph[op.problem]
    if not is_driven(node):
        yield NotDriven(op.problem, type(node).__name__)


class Undrive[N: Label, V: Place](Redefine[N, V]):
    '''The inverse of `Assign`: the condition node alone again, and the driver's places
    gone with it. What the node carries stays on it: what runs the driver no longer fits, so
    `Unimplement` it first, or a realised graph refuses it (`ImplementationWithoutDriver`).'''

    problem : N

    checks_graph = (check_undrives_a_driven_node,)

    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.problem: dataclasses.replace(graph[self.problem], driver=None)}

    def __repr__(self) -> str:
        return f'undrive({self.problem})'


# ----------------------------------------------------------------- violation finals

class DrivenTwice(Violation):
    problem : object
    driver  : str
    error    = TypeError
    template = ('{problem!r} already carries an algorithm, so a {driver} cannot answer it '
                'too -- `Undrive` it first')


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
