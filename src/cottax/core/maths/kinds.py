'''Node kinds: functional and conditional.'''

import dataclasses
from abc import abstractmethod
from collections.abc import Iterator, Mapping

from ..frozen import AbstractVar, Violation
from .definition import NodeDefinition, Renaming, relabelled
from .places import Label, Place

# ---------------------------------------------------------------- the two branches

def check_owns_something(d: 'NodeDefinition') -> Iterator[Violation]:
    '''
    Check a node owns something: a computation's rule, which a condition node does not add.

    O_n ≠ ∅  ∀ n ∊ N_F
    '''
    if not d.owns:
        yield OwnsNothing()


class FunctionalNode[V: Place](NodeDefinition[V]):
    """A computation: reads in, owns out. Only checked it produces something (beyond the base NodeDefinition checks)"""

    checks = (check_owns_something,)


class Function[V: Place](FunctionalNode[V]):
    """A computation: what it reads and what it owns."""

    reads : tuple[V, ...]
    owns  : tuple[V, ...]

    def relabel(self, renaming: Renaming[V]) -> 'Function[V]':
        return dataclasses.replace(self,
                                   reads = relabelled(self.reads, renaming),
                                   owns  = relabelled(self.owns,  renaming))

    def __repr__(self) -> str:
        return (f'Function({", ".join(str(v) for v in self.reads)} -> '
                f'{", ".join(str(v) for v in self.owns)})')


def check_states_something(node: 'ConditionalNode') -> Iterator[Violation]:
    '''A condition node owns or reads something inside: one with neither has no edges to
    a cycle, and is in no graph's structure.

    O^in_n ≠ ∅  or  I^in_n ≠ ∅  ∀ n ∊ N_C
    '''
    if not node.owns_inside and not node.reads_inside:
        yield StatesNothing(type(node).__name__)

def check_reads_on_one_side(node: 'ConditionalNode') -> Iterator[Violation]:
    '''A read is inside or outside, not both: the one must be produced on the node's
    cycle and the other must not.

    I^in_n ∩ I^out_n = ∅  ∀ n ∊ N_C
    '''
    if both := tuple(dict.fromkeys(v for v in node.reads_inside if v in node.reads_outside)):
        yield ReadOnBothSides(both)


class ConditionalNode[V: Place](NodeDefinition[V]):
    """
    Conditional Node.

    Technically a higher-order function: it uses a subgraph. 
    """

    owns_inside   : AbstractVar[tuple[V, ...]]
    """What it gives to its subgraph (cycle)."""
    reads_inside  : AbstractVar[tuple[V, ...]]
    """What it reads back from its subgraph (cycle)"""
    reads_outside : AbstractVar[tuple[V, ...]]
    """What it reads that its cycle does not produce"""
    owns_outside  : AbstractVar[tuple[V, ...]]
    """What it owns that its cycle does not read"""

    checks = (check_states_something, check_reads_on_one_side)

    @property
    def owns(self) -> tuple[V, ...]:   return self.owns_inside + self.owns_outside
    @property
    def reads(self) -> tuple[V, ...]:  return self.reads_inside + self.reads_outside

    @abstractmethod
    def determined(self, unknowns: tuple[V, ...]) -> 'ConditionalNode[V]':
        """The same condition node, owning these inside. What `Determine` and `Undetermine`
        edit; a final that cannot have them changed refuses here."""

# ---------------------------------------------------------------- the predicates
def is_requirement(d: object) -> bool:
    """A requirement: a condition node that reads inside and owns nothing there: n ∊ N_C \\ N_P."""
    return isinstance(d, ConditionalNode) and not bool(d.owns_inside)

def is_problem(d: object) -> bool:
    """A problem: a condition node that owns something inside: n ∊ N_P = {n ∊ N_C : O^in_n ≠ ∅}."""
    return isinstance(d, ConditionalNode) and bool(d.owns_inside)


def problems[N: Label](definitions: Mapping[N, NodeDefinition]) -> tuple[N, ...]:
    """The problems among `definitions`, in the order given."""
    return tuple(n for n, d in definitions.items() if is_problem(d))

def requirements[N: Label](definitions: Mapping[N, NodeDefinition]) -> tuple[N, ...]:
    """The requirements, in the order given."""
    return tuple(n for n, d in definitions.items() if is_requirement(d))


# ----------------------------------------------------------------- violation finals

class StatesNothing(Violation):
    node : str
    error    = ValueError
    template = '{node}: an empty condition node -- it owns and reads nothing inside, so it is in no structure'

class ReadOnBothSides(Violation):
    places : tuple
    error    = ValueError
    template = ('reads {places!r} both inside and outside: a read inside is produced on the '
                "node's cycle and a read outside is not, so one place cannot be both")

class OwnsNothing(Violation):
    error    = ValueError
    template = 'a functional node must own at least one variable'