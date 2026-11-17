'''Node kinds: functional and conditional.'''

import dataclasses
from collections.abc import Iterator, Mapping

from ..frozen import AbstractVar, Violation
from .definition import NodeDefinition, Renaming, attributes_text, relabelled
from .places import Label, Place, overlaps

# ---------------------------------------------------------------- the two branches

def check_owns_something(d: 'NodeDefinition') -> Iterator[Violation]:
    '''
    Check a node owns something: a computation's rule, which a condition node does not add.

    O_n ≠ ∅  ∀ n ∊ N_F
    '''
    if not d.owns:
        yield OwnsNothing()


class FunctionalNode[V: Place](NodeDefinition[V]):
    """A computation: what it reads and what it owns. Only checked it produces something
    (beyond the base NodeDefinition checks). """

    reads : tuple[V, ...]
    owns  : tuple[V, ...]

    checks = (check_owns_something,)

    def relabel(self, renaming: Renaming[V]) -> 'FunctionalNode[V]':
        return dataclasses.replace(self,
                                   reads = relabelled(self.reads, renaming),
                                   owns  = relabelled(self.owns,  renaming))

    def __repr__(self) -> str:
        return (f'FunctionalNode({", ".join(str(v) for v in self.reads)} -> '
                f'{", ".join(str(v) for v in self.owns)}){attributes_text(self)}')


def check_states_something(c: 'AbstractConditionalNode') -> Iterator[Violation]:
    '''A condition owns or reads something inside: one with neither has no edges to a
    cycle, and is in no graph's structure. It reads only the four names, so whatever has them
    may ask it (`conditional.Condition` does).

    O^in_n ≠ ∅  or  I^in_n ≠ ∅  ∀ n ∊ N_C
    '''
    if not c.owns_inside and not c.reads_inside:
        yield StatesNothing(type(c).__name__)

def check_reads_on_one_side(c: 'AbstractConditionalNode') -> Iterator[Violation]:
    '''A read is inside or outside, not both: the one must be produced on the node's
    cycle and the other must not (incomparable)

    i || j  ∀ i ∊ I^in_n,  j ∊ I^out_n,  ∀ n ∊ N_C
    '''
    if pairs := tuple(dict.fromkeys(overlaps(c.reads_inside, c.reads_outside))):
        inside, outside = zip(*pairs)
        yield ReadOnBothSides(inside, outside)


# ---------------------------------------------------------------- the condition node

class AbstractConditionalNode[V: Place](NodeDefinition[V]):
    """
    Conditional Node.

    Technically a higher-order function: it uses a subgraph.
    """

    owns_inside  : AbstractVar[tuple[V, ...]]
    owns_outside : AbstractVar[tuple[V, ...]]

    reads_inside : AbstractVar[tuple[V, ...]]
    reads_outside: AbstractVar[tuple[V, ...]]

    checks = (check_states_something, check_reads_on_one_side)

    @property
    def owns(self) -> tuple[V, ...]:   return self.owns_inside + self.owns_outside
    @property
    def reads(self) -> tuple[V, ...]:  return self.reads_inside + self.reads_outside


# ---------------------------------------------------------------- the predicates
def is_encloser(d: object) -> bool:
    """An encloser: a condition node that owns something inside, and so has a cycle around
    it -- a solve, an integration. n ∊ N_P = {n ∊ N_C : O^in_n ≠ ∅}."""
    return isinstance(d, AbstractConditionalNode) and bool(d.owns_inside)


def enclosers[N: Label](definitions: Mapping[N, NodeDefinition]) -> tuple[N, ...]:
    """The enclosers among `definitions`, in the order given."""
    return tuple(n for n, d in definitions.items() if is_encloser(d))


# ----------------------------------------------------------------- violation finals

class StatesNothing(Violation):
    node : str
    error    = ValueError
    template = '{node}: an empty condition node -- it owns and reads nothing inside, so it is in no structure'

class ReadOnBothSides(Violation):
    inside  : tuple
    outside : tuple
    error    = ValueError
    template = ('reads {inside!r} inside and {outside!r} outside, pairwise the same place or '
                "one inside the other: a read inside is produced on the node's cycle and a "
                'read outside is not, and a place is produced with what lies inside it, so '
                'the two sides cannot share any of it')

class OwnsNothing(Violation):
    error    = ValueError
    template = 'a functional node must own at least one variable'