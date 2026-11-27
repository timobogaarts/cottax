'''
The condition node: a `Condition`, the value it states, and the `Driver` answering it. Above
the maths, which knows neither: every rule reads a node through `AbstractConditionalNode`'s
four names, and the driver's places are in them.
'''

import dataclasses
from collections.abc import Iterator
from dataclasses import field

from ..frozen import Violation
from ..maths.definition import Renaming
from ..maths.kinds import AbstractConditionalNode
from ..maths.places import Place
from .condition import Condition
from .driver import Driver


def check_driver_fits(node: 'ConditionalNode') -> Iterator[Violation]:
    '''The driver fits the condition it is on: its contract accepts it, and each role sits at
    as many places as the role's `index` has entries on it. A node whose driver does not fit
    is never a meaningful state, so it is never built. Asked after the place rules.'''
    if node.driver is not None:
        yield from node.driver.violations_on(node.condition)


class ConditionalNode[V: Place](AbstractConditionalNode[V]):
    """
    The condition node: a condition, and the driver answering it. Final: a kind of
    condition is a `Condition`, not a node class. Its rules: a node's place rules over the
    four names, then that the driver fits the condition (`check_driver_fits`).
    """

    condition : Condition[V]
    """What it states."""
    driver    : 'Driver[V] | None' = field(default=None, kw_only=True)
    """What answers it: what it takes is read outside, what it reports owned outside. A
    field, since its places are edges; kept apart from the condition, since the driver is
    judged against the condition alone and taken off again (`Undrive`)."""

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

    def __repr__(self) -> str:
        driver = '' if self.driver is None else f' by {self.driver!r}'
        return f'{self.condition!r}{driver}'


def kind_of(node: object) -> object:
    """What a node is answered by: a condition node's condition, any other node itself."""
    return node.condition if isinstance(node, ConditionalNode) else node


def is_driven(node: object) -> bool:
    """A condition node carrying a driver."""
    return isinstance(node, ConditionalNode) and node.driver is not None
