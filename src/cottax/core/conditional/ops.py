'''
The condition node's own op: `SetOwnsInside`, what a condition node owns inside -- a
choice for a condition that stores it. `relational.Determine` / `Undetermine` say it in
their words.
'''

import dataclasses
from collections.abc import Iterator, Mapping

from ..frozen import Violation
from ..maths import Label, Graph, NodeDefinition, Place
from ..plan import Redefine
from .node import ConditionalNode, is_driven, kind_of


# ---------------------------------------------------------------- what is owned inside

def check_owned_once(op: 'SetOwnsInside') -> Iterator[Violation]:
    """No place is owned inside twice."""
    if twice := tuple(dict.fromkeys(p for p in op.owns_inside if op.owns_inside.count(p) > 1)):
        yield OwnedInsideTwice(twice)


def stores_owns_inside(node: object) -> bool:
    """`node` is a condition node whose condition stores `owns_inside` as a field: what
    `SetOwnsInside` can set."""
    return isinstance(node, ConditionalNode) and 'owns_inside' in {
        f.name for f in dataclasses.fields(node.condition)}


def check_owns_inside_is_stored(op: 'SetOwnsInside', graph: Graph) -> Iterator[Violation]:
    """The node's condition stores `owns_inside` and the node carries no driver: a condition
    that derives it -- from its states -- has no choice to set, and a driver's slots are laid
    out per unknown."""
    if not stores_owns_inside(node := graph[op.node]):
        yield OwnsInsideNotAChoice(op.node, type(kind_of(node)).__name__)
    elif is_driven(node):
        yield OwnsInsideDriven(op.node)


class SetOwnsInside[N: Label, V: Place](Redefine[N, V]):
    """A condition node owns `owns_inside` inside, and nothing else of it changes: with
    none it is a requirement, with some an encloser. For a condition that stores
    `owns_inside` as a field."""

    node        : N
    owns_inside : tuple[V, ...]

    checks = (check_owned_once,)
    checks_graph = (check_owns_inside_is_stored,)

    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        node = graph[self.node]
        return {self.node: dataclasses.replace(
            node, condition=dataclasses.replace(node.condition, owns_inside=self.owns_inside))}

    def __repr__(self) -> str:
        return f'set_owns_inside({self.node} <- {", ".join(str(p) for p in self.owns_inside)})'


# ----------------------------------------------------------------- violation finals

class OwnedInsideTwice(Violation):
    places : tuple[Place, ...]
    error    = ValueError
    template = '{places!r} is owned inside twice'

class OwnsInsideNotAChoice(Violation):
    node : Label
    kind : str
    error    = TypeError
    template = '{node!r} is a {kind}, and what it owns inside is not a choice'

class OwnsInsideDriven(Violation):
    node : Label
    error    = TypeError
    template = ('{node!r} carries a driver, whose places are laid out per unknown -- '
                '`Undrive` it first')
