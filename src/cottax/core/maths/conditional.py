'''
The condition node: a `Condition`, the value it states, and what is attached to it
(`Attachment`) -- its driver. Above `kinds`, which knows neither: every rule reads a node
through `AbstractConditionalNode`'s four names, and the driver's places are in them.
'''

import dataclasses
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import field

from ..frozen import AbstractVar, Frozen, Violation, first
from .definition import Renaming, attributes_text, check_owns_are_separate, check_reads_outside_owns
from .kinds import AbstractConditionalNode, check_reads_on_one_side, check_states_something
from .places import Place


# ---------------------------------------------------------------- the condition, a value

class Condition[V: Place](Frozen):
    """
    What a condition node states, as a value over places: the four names, a way to rename
    them, and for a kind that joins, its join. Not a node: a node holds one
    (`conditional.ConditionalNode`).
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

    join = None
    """`join(self, other) -> Condition`: the kind's join of two conditions; `None` for a
    kind that does not join."""

    @property
    def owns(self) -> tuple[V, ...]:   return self.owns_inside + self.owns_outside
    @property
    def reads(self) -> tuple[V, ...]:  return self.reads_inside + self.reads_outside

    @abstractmethod
    def relabel(self, renaming: Renaming[V]) -> 'Condition[V]':
        """The same condition with its places renamed."""

    def __add__(self, other: 'Condition[V]') -> 'Condition[V]':
        """The kind's `join`; none for a kind without one, or with a non-condition."""
        if self.join is None or not isinstance(other, Condition):
            return NotImplemented
        return type(self).join(self, other)


class Attachment[V: Place](Frozen):
    """
    What a node carries that has places: a detachable bundle of them, added to what the node
    reads (`reads`) and owns (`owns`) -- on a condition node, both outside its cycle. Renamed
    with the node, joined when two nodes are, and asked whether it fits the condition it is
    attached to. The maths knows it by this protocol only: what a concrete one is (a driver:
    an algorithm, its roles) lives above. An `Attribute` has no places; an attachment does.
    """

    reads : AbstractVar[tuple[V, ...]]
    """The places it adds to what the node reads."""
    owns  : AbstractVar[tuple[V, ...]]
    """The places it adds to what the node owns."""

    @abstractmethod
    def relabel(self, renaming: Renaming[V]) -> 'Attachment[V]':
        """The same attachment with its places renamed."""

    @abstractmethod
    def violations_on(self, condition: Condition[V]) -> Iterator[Violation]:
        """Every way it does not fit `condition`: the rules it brings."""

    @abstractmethod
    def join_violations(self, other: 'Attachment[V]', condition: Condition[V],
                        other_condition: Condition[V]) -> Iterator[Violation]:
        """Every reason it does not join `other`, attached to `condition` and
        `other_condition`."""

    @abstractmethod
    def joined(self, other: 'Attachment[V]', condition: Condition[V],
               other_condition: Condition[V], joined: Condition[V]) -> 'Attachment[V]':
        """The attachment of `joined`, the join of `condition` and `other_condition`; may
        assume `join_violations` found nothing."""


def check_driver_fits(node: 'ConditionalNode') -> Iterator[Violation]:
    """What is attached fits the condition: its own rules. The condition passed its own
    when it was built."""
    if node.driver is not None:
        yield from node.driver.violations_on(node.condition)


class ConditionalNode[V: Place](AbstractConditionalNode[V]):
    """
    The condition node: a condition, and what answers it, attached. Final: a kind of
    condition is a `Condition`, not a node class.
    """

    condition : Condition[V]
    """What it states."""
    driver    : 'Attachment[V] | None' = field(default=None, kw_only=True)
    """What answers it, attached: what it reads is read outside, what it owns owned outside.
    A field, since its places are edges; the maths knows it as an `Attachment` only."""

    checks = (check_driver_fits,)

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

    def __add__(self, other: 'ConditionalNode[V]') -> 'ConditionalNode[V]':
        """The join: the conditions by their kind's `join`, then the drivers by their own
        `joined`. A driven node does not join a bare one, since what answers the join is
        not read off the two (`driver_join_violations`). What either carries is not joined."""
        if not isinstance(other, ConditionalNode) or self.condition.join is None:
            return NotImplemented
        joined = self.condition.__add__(other.condition)
        if joined is NotImplemented:
            return joined
        first(driver_join_violations(self, other))
        return ConditionalNode(joined, driver=None if self.driver is None else self.driver.joined(
            other.driver, self.condition, other.condition, joined))

    def __repr__(self) -> str:
        driver = '' if self.driver is None else f' by {self.driver!r}'
        return f'{self.condition!r}{driver}{attributes_text(self)}'


def driver_join_violations(node: ConditionalNode, other: ConditionalNode) -> Iterator[Violation]:
    """Every reason the drivers of two condition nodes do not join: one driven and one
    not, or what the driver says of the other."""
    if (node.driver is None) != (other.driver is None):
        yield DrivenOnOneSide(node, other)
    elif node.driver is not None:
        yield from node.driver.join_violations(other.driver, node.condition, other.condition)


def kind_of(node: object) -> object:
    """What a node is answered and folded by: a condition node's condition, any other node
    itself."""
    return node.condition if isinstance(node, ConditionalNode) else node


def addable(kind: type) -> bool:
    """Whether nodes of `kind` have a `+`: a condition kind through its `join`."""
    if issubclass(kind, Condition):
        return kind.join is not None
    return getattr(kind, '__add__', None) is not None


def is_driven(node: object) -> bool:
    """A condition node carrying a driver."""
    return isinstance(node, ConditionalNode) and node.driver is not None


# ----------------------------------------------------------------- violation finals

class DrivenOnOneSide(Violation):
    node  : object
    other : object
    error    = ValueError
    template = ('{node!r} and {other!r}: one is driven and one is not, so what answers their '
                'join is not read off the two -- drive both alike, or `Undrive` the one')
