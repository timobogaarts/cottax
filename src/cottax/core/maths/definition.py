"""
Node definition and its checks
"""

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping

from ..frozen import AbstractVar, Frozen, Violation
from .places import Place, self_overlaps, is_prefix, overlaps




def check_owns_are_separate(d: 'NodeDefinition') -> Iterator[Violation]:
    '''
    Check that no two of this node's owns overlap: none of them are comparable.

    o || m ∀ o,m  ∊ O_n, o ≠ m
    '''
    for place, other in self_overlaps(d.owns):
        yield (OwnsTwice if other == place else OwnsNested)(place, other)


def check_reads_outside_owns(d: 'NodeDefinition') -> Iterator[Violation]:
    '''
    Check whether the reads and owns are not comparable.

    o || i ∀ o ∊ O_n,  i ∊ I_n
    '''
    for place, own in overlaps(d.reads, d.owns):
        yield (ReadsOwn(place) if own == place else
               ReadInsideOwn(place, own) if is_prefix(own, place) else
               OwnInsideRead(place, own))




type Renaming[V: Place] = Mapping[V, V]
'''Old place -> new, per node. Keyed by the place, so `fn(x, x)` renames both reads.'''


# ---------------------------------------------------------------- the definition: one that passed them
class NodeDefinition[V: Place](Frozen):
    """
    Node definition: has owns and reads and a way to relabel them.
    Checked at construction by the two rules above; each level adds its own.
    """

    owns   : AbstractVar[tuple[V, ...]]
    reads  : AbstractVar[tuple[V, ...]]
    checks = (check_owns_are_separate, check_reads_outside_owns)

    @abstractmethod
    def relabel(self, renaming: Renaming[V]) -> 'NodeDefinition[V]':
        """The same definition with its places renamed."""



def relabelled[V: Place](places: Iterable[V], renaming: Renaming[V]) -> tuple[V, ...]:
    '''A field of places renamed. What a `relabel` implementation is made of.'''
    return tuple(renaming.get(v, v) for v in places) # either the new name or the old name (if not in the renaming)

# ---------------------------------------------------------------- violation finals.

class OwnsTwice(Violation):
    place : Place
    other : Place
    error    = ValueError
    template = 'duplicate output name {place!r}'

class OwnsNested(Violation):
    place : Place
    other : Place
    error    = ValueError
    template = ('owns {place!r} and {other!r}, one inside the other -- two outputs written to '
                 'one storage')

class ReadsOwn(Violation):
    place : Place
    error    = ValueError
    template = ('reads {place!r}, which it also owns. Reading the name a node produces back '
                 'makes it mean the value assumed and the value produced at once -- `Cut` it '
                 'and read the copy')

class ReadInsideOwn(Violation):
    place : Place
    own   : Place
    error    = ValueError
    template = ('reads {place!r}, which lies inside {own!r}, which it owns -- reads are matched '
                 'by equality, so this read would silently miss its own output')

class OwnInsideRead(Violation):
    place : Place
    own   : Place
    error    = ValueError
    template = ('owns {own!r}, which lies inside {place!r}, which it reads -- that read would '
                 'silently be a boundary input')

