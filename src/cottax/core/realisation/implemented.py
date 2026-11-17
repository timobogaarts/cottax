'''
What runs a node: `Implementation`, an attribute the node carries -- a function its body, a
driven node what runs its driver -- and the edits setting it and taking it off.

Implementation is an arbitrary object: it can be a function, a description of a repo or something else entirely.
'''

import dataclasses
from collections.abc import Iterator, Sequence

from ...core import (
    Attribute, ComposedOp, FunctionalNode, GraphOp, Label, Graph, NodeDefinition, Place,
    RemoveAttribute, SetAttribute, Violation,
)
from ...core.frozen import OrderedMap, name_of
from ...core import is_driven
from ...core.driven import DriverContract, entry_for


class Implementation:
    '''
    The key of what runs a node: a callable, or a description of one (location etc.).
    The node's kind says how it is used: a function is called with what it reads, a driven
    node's driver is run by it.
    '''


def check_implementation_fits[N: Label, V: Place](name: N, node: NodeDefinition[V]
                                                 ) -> Iterator[Violation]:
    '''An implementation is on a function, or on a condition node with a driver: something
    runs a function, and what runs a driver; a bare condition node takes a driver first.'''
    if is_implemented(node) and not (isinstance(node, FunctionalNode) or is_driven(node)):
        yield ImplementationWithoutDriver(name, type(node).__name__)


def check_implementable(op: 'Implement', graph: Graph) -> Iterator[Violation]:
    '''Each node named can carry an implementation: asked of the node it would be, before
    the graph is touched.'''
    for name, value in op.values.items():
        yield from check_implementation_fits(name, implemented(graph[name], value))


def is_implemented(node: NodeDefinition) -> bool:
    return Implementation in node.attributes


def implementation_of(node: NodeDefinition) -> object:
    '''What runs `node`: asked of an implemented one.'''
    return node.attributes[Implementation]


def implemented[D: NodeDefinition](node: D, implementation: object) -> D:
    '''`node` with what runs it: how a node is built implemented, outside a plan.'''
    return dataclasses.replace(node, attributes=OrderedMap({**node.attributes, Implementation: implementation}))


# ----------------------------------------------------------------- the edits

class Implement[N: Label, V: Place](SetAttribute[N, V]):
    '''Supply what runs some nodes -- a function's body, a driven node's driver
    implementation -- in one op, since choosing a backend is one decision.'''

    kind = Implementation

    checks_graph = (check_implementable,)


class Unimplement[N: Label, V: Place](RemoveAttribute[N, V]):
    '''The inverse of `Implement`: the implementations dropped, the nodes kept.'''

    kind = Implementation


_NONE = object()
'''No entry: an implementation may be any object, `None` too.'''


class ImplementDrivers[N: Label, V: Place](ComposedOp[N, V]):
    '''What runs every driven node not yet implemented, chosen by its driver's type: the
    entry of the most specific contract along its ancestry, whatever order the map is
    written in. A rule, so on a changed model it chooses again; a node no entry covers is
    left for the proof to refuse, and a graph with none to implement is refused by
    `Implement`.'''

    implementations : OrderedMap[type[DriverContract], object]

    def composed(self, graph: Graph[N, V]) -> Sequence[GraphOp[N, V]]:
        chosen = ((name, entry_for(self.implementations, node.driver.contract, _NONE))
                  for name in graph.nodes
                  if is_driven(node := graph[name]) and not is_implemented(node))
        return (Implement(OrderedMap((name, i) for name, i in chosen if i is not _NONE)),)

    def __repr__(self) -> str:
        return 'implement_drivers(' + ', '.join(f'{kind.__name__} <- {name_of(i)}'
                                                for kind, i in self.implementations.items()) + ')'


# ----------------------------------------------------------------- violation finals

class ImplementationWithoutDriver(Violation):
    node : object
    kind : str
    error    = TypeError
    template = ('{node!r}: Implementation does not fit a {kind} with no driver -- nothing says '
                'what it would run. `Assign` a driver first, or take the Implementation off')
