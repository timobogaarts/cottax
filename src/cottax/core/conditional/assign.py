'''
The edits putting a driver on and taking it off.
'''

import dataclasses
from collections.abc import Callable, Collection, Iterator, Mapping

from ..frozen import OrderedMap, Violation, violations
from ..maths import Label, Graph, NodeDefinition, Place
from ..plan import ComposedOp, Redefine
from .condition import Condition
from .driver import Driver, DriverContract, NotOffered, Role
from .node import ConditionalNode, is_driven

type ByRole[V] = Mapping[type[Role], tuple[V, ...]]
'''Per role, its places: a driver's data, or its reports.'''


def driven(node: ConditionalNode | Condition, contract: DriverContract,
           data: ByRole = OrderedMap(), reports: ByRole = OrderedMap()) -> ConditionalNode:
    '''`node` answered by `contract`, its roles at `data` and `reports`: how a node is built
    driven, outside a plan. A condition is taken as the node holding it. Refused as `Assign`
    refuses it: a second driver, a driver its contract does not allow, one not fitting.'''
    node = node if isinstance(node, ConditionalNode) else ConditionalNode(node)
    if is_driven(node):
        raise DrivenTwice(node, type(contract).__name__).raised
    return dataclasses.replace(node, driver=Driver(contract, OrderedMap(data), OrderedMap(reports)))


def driving_violations(node: ConditionalNode, contract: DriverContract,
                       data: ByRole = OrderedMap(), reports: ByRole = OrderedMap()
                       ) -> Iterator[Violation]:
    '''Every rule `driven(node, contract, data, reports)` breaks, as data: a second driver
    first, since its reports would otherwise read as owned twice; then the driver's own
    rules; then the node's, its driver fitting the condition among them.'''
    if is_driven(node):
        yield DrivenTwice(node, type(contract).__name__)
        return
    data, reports = OrderedMap(data), OrderedMap(reports)
    if found := list(violations(Driver, contract, data, reports)):
        yield from found
        return
    yield from violations(ConditionalNode, node.condition, driver=Driver(contract, data, reports),
                          attributes=node.attributes)


def check_a_condition_node(op: 'Assign | AssignByRule', graph: Graph) -> Iterator[Violation]:
    '''A condition node to drive -- and, for `AssignByRule`, to read before its places are
    made from it.'''
    if not isinstance(node := graph[op.problem], ConditionalNode):
        yield NotAConditionalNode(op.problem, type(node).__name__)


def check_drivable(op: 'Assign', graph: Graph) -> Iterator[Violation]:
    '''An algorithm goes on a condition node without one, and the node it makes is well
    formed -- asked here, so a refusal is data before the graph is touched.'''
    if isinstance(node := graph[op.problem], ConditionalNode):
        yield from driving_violations(node, op.contract, op.data, op.reports)


class Assign[N: Label, V: Place](Redefine[N, V]):
    '''Attach a driver to a condition node, and say where each of its roles sits: the
    algorithm choice and its places, recorded in the plan and carried by the node. Where a
    place sits is a decision, so the op records it: a surface supplies the obvious ones.'''

    problem  : N
    contract : DriverContract[V]
    data     : OrderedMap[type[Role], tuple[V, ...]] = OrderedMap()
    reports  : OrderedMap[type[Role], tuple[V, ...]] = OrderedMap()

    checks_graph = (check_a_condition_node, check_drivable)

    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.problem: driven(graph[self.problem], self.contract, self.data, self.reports)}

    def __repr__(self) -> str:
        return f'assign({self.problem} <- {type(self.contract).__name__})'


def check_asks_offered(op: 'AssignByRule') -> Iterator[Violation]:
    '''Each role asked is one the contract may report or may be given.'''
    offered = (*op.contract.may_report, *op.contract.may_be_given)
    if extra := tuple(r.__name__ for r in op.asks if r not in offered):
        yield NotOffered(repr(op), 'asked', extra)


class AssignByRule[N: Label, V: Place](ComposedOp[N, V]):
    '''`Assign` at the places `places(problem, condition, contract, role)` gives each role:
    the reports `asks` names, the optional data it names, and the data the contract requires
    for those reports. The condition is read when it is applied: the rule is recorded, so a
    changed model has its places derived again.'''

    problem  : N
    contract : DriverContract[V]
    places   : Callable[..., tuple[V, ...]]
    asks     : Collection[type[Role]] = ()
    '''The optional roles wired: reports asked for, and optional data given.'''

    checks = (check_asks_offered,)
    checks_graph = (check_a_condition_node,)

    def composed(self, graph: Graph[N, V]) -> tuple[Assign[N, V]]:
        contract, condition = self.contract, graph[self.problem].condition
        reports = tuple(r for r in self.asks if r in contract.may_report)
        data = dict.fromkeys((*contract.required(reports), *(r for r in self.asks if r not in reports)))
        at = lambda roles: OrderedMap((r, self.places(self.problem, condition, contract, r))
                                      for r in roles)
        return (Assign(self.problem, contract, at(data), at(reports)),)

    def __repr__(self) -> str:
        return f'assign({self.problem} <- {type(self.contract).__name__}, at its default places)'


def check_undrives_a_driven_node(op: 'Undrive', graph: Graph) -> Iterator[Violation]:
    '''There is an algorithm to drop. What the node carries is not asked here: whether it
    fits a node without a driver is its own layer's rule (an implementation's is
    `realisation`'s, asked where that is proved).'''
    if not is_driven(node := graph[op.problem]):
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
