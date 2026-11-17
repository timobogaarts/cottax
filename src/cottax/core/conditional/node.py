'''
The condition node: a `Condition`, the value it states, and what is attached to it
(`Attachment`) -- its driver. Above the maths, which knows neither: every rule reads a node
through `AbstractConditionalNode`'s four names, and the driver's places are in them.
'''

import dataclasses
from abc import abstractmethod
from dataclasses import field

from ..frozen import AbstractVar, Frozen
from ..maths.definition import Renaming, check_owns_are_separate, check_reads_outside_owns
from ..maths.kinds import AbstractConditionalNode, check_reads_on_one_side, check_states_something
from ..maths.places import Place


# ---------------------------------------------------------------- the condition, a value

class Condition[V: Place](Frozen):
    """
    What a condition node states, as a value over places: the four names and a way to rename
    them. Not a node: a node holds one (`ConditionalNode`).
    Checked on its own at construction, so a node is built from one already well formed.
    """

    owns_inside   : AbstractVar[tuple[V, ...]]
    """What it gives to its subgraph (cycle)."""
    reads_inside  : AbstractVar[tuple[V, ...]]
    """What it reads back from its subgraph (cycle)."""
    reads_outside : AbstractVar[tuple[V, ...]]
    """What it reads that its cycle does not produce."""
    owns_outside  : AbstractVar[tuple[V, ...]]
    """What it owns that its cycle does not read."""

    checks = (check_owns_are_separate, check_reads_outside_owns,
              check_states_something, check_reads_on_one_side)
    '''A node's place rules, then a condition node's: a condition is well formed in itself,
    whatever is attached to it later.'''

    @property
    def owns(self) -> tuple[V, ...]:   return self.owns_inside + self.owns_outside
    @property
    def reads(self) -> tuple[V, ...]:  return self.reads_inside + self.reads_outside

    @abstractmethod
    def relabel(self, renaming: Renaming[V]) -> 'Condition[V]':
        """The same condition with its places renamed."""


class Attachment[V: Place](Frozen):
    """
    What a node carries that has places: a detachable bundle of them, added to what the node
    reads (`reads`) and owns (`owns`) -- on a condition node, both outside its cycle. Renamed
    with the node. Known here by this protocol only: what a concrete one is (a driver: an
    algorithm, its roles), and whether it fits the condition it is attached to, live above.
    An `Attribute` has no places; an attachment does.
    """

    reads : AbstractVar[tuple[V, ...]]
    """The places it adds to what the node reads."""
    owns  : AbstractVar[tuple[V, ...]]
    """The places it adds to what the node owns."""

    @abstractmethod
    def relabel(self, renaming: Renaming[V]) -> 'Attachment[V]':
        """The same attachment with its places renamed."""


class ConditionalNode[V: Place](AbstractConditionalNode[V]):
    """
    The condition node: a condition, and what answers it, attached. Final: a kind of
    condition is a `Condition`, not a node class. Its rules are a node's place rules over
    the four names; whether the driver fits the condition is the driven layer's.
    """

    condition : Condition[V]
    """What it states."""
    driver    : 'Attachment[V] | None' = field(default=None, kw_only=True)
    """What answers it, attached: what it reads is read outside, what it owns owned outside.
    A field, since its places are edges; the maths knows it as an `Attachment` only."""

    @property
    def owns_inside(self) -> tuple[V, ...]:   return self.condition.owns_inside
    @property
    def reads_inside(self) -> tuple[V, ...]:  return self.condition.reads_inside
    @property
    def reads_outside(self) -> tuple[V, ...]:
        """The condition's, then what its driver takes."""
        return self.condition.reads_outside + (() if self.driver is None else self.driver.reads)
    @property
    def owns_outside(self) -> tuple[V, ...]:
        """The condition's, then what its driver reports."""
        return self.condition.owns_outside + (() if self.driver is None else self.driver.owns)

    def relabel(self, renaming: Renaming[V]) -> 'ConditionalNode[V]':
        """The condition and the driver renamed alike; the attributes kept."""
        return dataclasses.replace(
            self, condition=self.condition.relabel(renaming),
            driver=None if self.driver is None else self.driver.relabel(renaming))

    def __repr__(self) -> str:
        driver = '' if self.driver is None else f' by {self.driver!r}'
        return f'{self.condition!r}{driver}'


def kind_of(node: object) -> object:
    """What a node is answered by: a condition node's condition, any other node itself."""
    return node.condition if isinstance(node, ConditionalNode) else node


def is_driven(node: object) -> bool:
    """A condition node carrying a driver."""
    return isinstance(node, ConditionalNode) and node.driver is not None
