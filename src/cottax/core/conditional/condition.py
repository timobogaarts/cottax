'''
The condition, a value: what a condition node states, over places.
Mirrors an AbstractConditionalNode but isn't a node: a ConditionalNode holds
both the condition and the driver (which changes edges). To avoid a situation of a 
ConditionalNode without a driver and one with being different types, this is separate.

Additionally, you could define operations on just the condition which may not make sense on the node itself
(like a +).

If a driver is not present, it reduces anyway to exactly the condition.
'''
from abc import abstractmethod
from ..frozen import AbstractVar, Frozen
from ..maths.definition import Renaming, check_owns_are_separate, check_reads_outside_owns
from ..maths.kinds import check_reads_on_one_side, check_states_something
from ..maths.places import Place

class Condition[V: Place](Frozen):
    """
    Condition: mirrors a AbstractConditional node.
    """
    owns_inside   : AbstractVar[tuple[V, ...]]
    """What it gives to its subgraph (cycle):"""
    reads_inside  : AbstractVar[tuple[V, ...]]
    """What it reads back from its subgraph (cycle)."""
    reads_outside : AbstractVar[tuple[V, ...]]
    """What it reads that its cycle does not produce."""
    owns_outside  : AbstractVar[tuple[V, ...]]
    """What it owns that its cycle does not read."""

    checks = (check_owns_are_separate, check_reads_outside_owns,
              check_states_something, check_reads_on_one_side)
    '''A node's place rules, then a condition node's: a condition is well formed in itself,
    whatever drives it later.'''
    @property
    def owns(self) -> tuple[V, ...]:   return self.owns_inside + self.owns_outside
    @property
    def reads(self) -> tuple[V, ...]:  return self.reads_inside + self.reads_outside
    @abstractmethod
    def relabel(self, renaming: Renaming[V]) -> 'Condition[V]':
        """The same condition with its places renamed."""
