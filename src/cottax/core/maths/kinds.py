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
    '''A condition node determines something or holds against something: one with
    neither has no edges, and is in no graph's structure.

    U_n ≠ ∅  or  C_n ≠ ∅  ∀ n ∊ N_C
    '''
    if not node.unknowns and not node.conditions:
        yield StatesNothing(type(node).__name__)


class ConditionalNode[V: Place](NodeDefinition[V]):
    """
    A condition node: what it determines and what it holds against. With or without the
    algorithm answering it.

    No shape-matching of the unknowns against the conditions: the shape of the arrays is
    unknown at this time, and one 'vector' unknown against 200 conditions is legitimate.
    It is a driver's question anyway -- a least-squares driver does not want it forbidden.
    """

    unknowns   : AbstractVar[tuple[V, ...]]
    """What this node determines; `()` for a requirement."""
    conditions : AbstractVar[tuple[V, ...]]
    """What it holds against: its reads, less a driver's data."""

    checks = (check_states_something,)

    @abstractmethod
    def determined(self, unknowns: tuple[V, ...]) -> 'ConditionalNode[V]':
        """The same condition node, determining these. What `Determine` and `Undetermine`
        edit; a final that cannot have them changed refuses here."""

# ---------------------------------------------------------------- the predicates
def is_requirement(d: object) -> bool:
    """A requirement: a condition node that holds against something and determines nothing: n ∊ N_C \\ N_P."""
    return isinstance(d, ConditionalNode) and not bool(d.unknowns)

def is_problem(d: object) -> bool:
    """A problem: a condition node that determines something: n ∊ N_P = {n ∊ N_C : U_n ≠ ∅}."""
    return isinstance(d, ConditionalNode) and bool(d.unknowns)


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
    template = '{node}: an empty condition node -- no unknowns and no conditions, so it is in no structure'

class OwnsNothing(Violation):
    error    = ValueError
    template = 'a functional node must own at least one variable'