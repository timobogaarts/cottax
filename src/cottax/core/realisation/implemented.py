'''
Implemented nodes: both a function and a driver. 

Implementation is an arbitrary object: it can be a function, a description of a repo or something else entirely.
'''

from collections.abc import Iterator, Mapping, Sequence

from ...core import (
    ComposedOp, ConditionalNode, FunctionalNode, GraphOp, Label, NestingGraph, NodeDefinition, Place,
    Redefine, Renaming, Violation, violations,
)
from ...core.frozen import AbstractVar, OrderedMap, name_of
from ..driven import DrivenNode, DriverContract, entry_for


def check_implemented_once(implemented: 'Implemented') -> Iterator[Violation]:
    '''One implementation on a node: a second would replace the first in silence.'''
    if isinstance(implemented.node, Implemented):
        yield ImplementedTwice(implemented.node)


class Implemented[V: Place](NodeDefinition[V]):
    '''
    A node with what runs it: a callable, or a description of one (location etc.).
    '''

    node           : AbstractVar[NodeDefinition[V]]
    '''The node run: a final's level narrows it.'''
    implementation : AbstractVar[object]
    '''What runs it.'''

    checks = (check_implemented_once,)

    @property
    def reads(self) -> tuple[V, ...]:                         return self.node.reads
    @property
    def owns(self) -> tuple[V, ...]:                          return self.node.owns
    
    def relabel(self, renaming: Renaming[V]) -> 'Implemented[V]':
        return type(self)(self.node.relabel(renaming), self.implementation)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.node!r} by {name_of(self.implementation)})'


def check_a_function(implemented: 'ImplementedFunction') -> Iterator[Violation]:
    '''A body goes on a function: on anything else it would silently turn it into one.'''
    node = implemented.node
    if not isinstance(node, FunctionalNode):
        yield NotAFunction(node, type(node).__name__)


class ImplementedFunction[V: Place](Implemented[V], FunctionalNode[V]):
    '''A function with its body.'''

    node           : FunctionalNode[V]
    implementation : object

    checks   = (check_a_function,)
    __repr__ = Implemented.__repr__


def check_a_contract(implemented: 'ImplementedDriven') -> Iterator[Violation]:
    '''What runs a driver goes on a driven node: a bare condition node has no contract for it
    to answer.'''
    node = implemented.node
    if not isinstance(node, DrivenNode):
        yield NoContract(node, type(node).__name__)


class ImplementedDriven[V: Place](Implemented[V], DrivenNode[V]):
    '''A driven node with what runs its driver.'''

    node           : DrivenNode[V]
    implementation : object

    checks   = (check_a_contract,)
    __repr__ = Implemented.__repr__

    @property
    def unknowns(self) -> tuple[V, ...]:               return self.node.unknowns
    @property
    def conditions(self) -> tuple[V, ...]:             return self.node.conditions
    @property
    def condition(self) -> ConditionalNode[V]:         return self.node.condition
    @property
    def contract(self) -> DriverContract[V]:                     return self.node.contract
    @property
    def data(self) -> tuple[tuple[V, ...], ...]:       return self.node.data
    @property
    def reports(self) -> tuple[tuple[V, ...], ...]:    return self.node.reports

    def determined(self, unknowns: tuple[V, ...]) -> 'ImplementedDriven[V]':
        '''Refused: take off the outer layer first.'''
        raise DeterminedWhileImplemented(self.unknowns, tuple(unknowns)).raised

    def undriven(self) -> ConditionalNode[V]:
        '''Refused: the implementation runs this driver, so it comes off first.'''
        raise UndrivenWhileImplemented(type(self.contract).__name__).raised


def is_implemented(d: object) -> bool:
    return isinstance(d, Implemented)


# ----------------------------------------------------------------- which final, per level

def implemented_as(node: NodeDefinition) -> type[Implemented] | None:
    '''The final implementing `node`, by its level; `None` for a level nothing implements.
    A wrapper joins its level, so a second implementation reaches `ImplementedTwice`.'''
    if isinstance(node, FunctionalNode):
        return ImplementedFunction
    if isinstance(node, DrivenNode):
        return ImplementedDriven
    return None


# ----------------------------------------------------------------- the edits

def check_implementable(op: 'Implement', graph: NestingGraph) -> Iterator[Violation]:
    '''Each node it makes is well formed -- asked here, so a refusal is data before the
    graph is touched.'''
    for name, implementation in op.implementations.items():
        node = graph[name]
        if (final := implemented_as(node)) is None:
            yield NotImplementable(name, type(node).__name__)
        else:
            yield from violations(final, node, implementation)


def check_implements_something(op: 'Implement') -> Iterator[Violation]:
    if not op.implementations:
        yield NothingImplemented()


class Implement[N: Label, V: Place](Redefine[N, V]):
    '''Supply what runs some nodes -- a function's body, a driven node's driver
    implementation -- in one op, since choosing a backend is one decision. Their places
    are the nodes' own and are handed straight on.'''

    implementations : OrderedMap[N, object]

    checks       = (check_implements_something,)
    checks_graph = (check_implementable,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {name: implemented_as(graph[name])(graph[name], implementation)
                for name, implementation in self.implementations.items()}

    def __repr__(self) -> str:
        return 'implement(' + ', '.join(f'{name} <- {name_of(i)}'
                                        for name, i in self.implementations.items()) + ')'


def check_unimplements_implemented(op: 'Unimplement', graph: NestingGraph) -> Iterator[Violation]:
    '''There is an implementation to drop, on each.'''
    for name in op.nodes:
        if not isinstance(node := graph[name], Implemented):
            yield NotImplementedYet(name, type(node).__name__)


def check_unimplements_something(op: 'Unimplement') -> Iterator[Violation]:
    '''Some node, none twice.'''
    if not op.nodes:
        yield NothingImplemented()
    if len(set(op.nodes)) != len(op.nodes):
        yield NodeNamedTwice(tuple(dict.fromkeys(n for n in op.nodes if op.nodes.count(n) > 1)))


class Unimplement[N: Label, V: Place](Redefine[N, V]):
    '''The inverse of `Implement`: the implementations dropped, the nodes kept.'''

    nodes : tuple[N, ...]

    checks       = (check_unimplements_something,)
    checks_graph = (check_unimplements_implemented,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {name: graph[name].node for name in self.nodes}

    def __repr__(self) -> str:
        return f'unimplement({", ".join(map(str, self.nodes))})'


_NONE = object()
'''No entry: an implementation may be any object, `None` too.'''


class ImplementDrivers[N: Label, V: Place](ComposedOp[N, V]):
    '''What runs every driven node not yet implemented, chosen by its driver's type: the
    entry of the most specific contract along its ancestry, whatever order the map is
    written in. A rule, so on a changed model it chooses again; a node no entry covers is
    left for the proof to refuse, and a graph with none to implement is refused by
    `Implement`.'''

    implementations : OrderedMap[type[DriverContract], object]

    def composed(self, graph: NestingGraph[N, V]) -> Sequence[GraphOp[N, V]]:
        chosen = ((name, entry_for(self.implementations, node.contract, _NONE))
                  for name in graph.nodes
                  if isinstance(node := graph[name], DrivenNode) and not is_implemented(node))
        return (Implement(OrderedMap((name, i) for name, i in chosen if i is not _NONE)),)

    def __repr__(self) -> str:
        return 'implement_drivers(' + ', '.join(f'{kind.__name__} <- {name_of(i)}'
                                                for kind, i in self.implementations.items()) + ')'


# ----------------------------------------------------------------- violation finals

class ImplementedTwice(Violation):
    node : object
    error    = TypeError
    template = ('{node!r} already has an implementation -- replacing one with another '
                'silently is not a rewrite; `Unimplement` it first')

class NotAFunction(Violation):
    node : object
    kind : str
    error    = TypeError
    template = ('{node!r} is a {kind}, which is answered rather than run -- only a function '
                'takes a body')

class NotImplementable(Violation):
    node : object
    kind : str
    error    = TypeError
    template = ('nothing implements a {kind}, which is what {node!r} is -- only a function '
                'or a driven node is implemented')

class NothingImplemented(Violation):
    error    = ValueError
    template = 'names no node, so there is nothing to implement or take off'

class NodeNamedTwice(Violation):
    nodes : tuple
    error    = ValueError
    template = 'unimplement(): {nodes!r} named twice'

class NotImplementedYet(Violation):
    node : object
    kind : str
    error    = TypeError
    template = ('{node!r} is a {kind} and carries no implementation, so there is nothing to '
                'take off')

class NoContract(Violation):
    node : object
    kind : str
    error    = TypeError
    template = ('{node!r} is a {kind} with no driver, so there is no contract for an '
                'implementation to answer -- `Assign` one first')

class DeterminedWhileImplemented(Violation):
    unknowns : tuple
    asked    : tuple
    error    = TypeError
    template = ('this node is implemented over {unknowns!r}, so it cannot be made to determine '
                '{asked!r} -- `Unimplement` it first')

class UndrivenWhileImplemented(Violation):
    driver : str
    error    = TypeError
    template = ('this {driver} is implemented, and the implementation runs it, so the driver '
                'cannot be taken off under it -- `Unimplement` it first')
