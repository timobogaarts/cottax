"""
A place: a word over segments, and the prefix order on words.
"""

from collections.abc import Callable, Collection, Hashable, Iterable, Iterator, Set
from typing import Protocol, Self

type Label = Hashable
'''
Where a node is bound: a key, and nothing more.
'''

class Place(Protocol):
    """
    A word over segments, as a path into a nested tree, and its strict prefixes. 
    """

    segments : tuple[Hashable, ...]
    '''The word, in order from the root: finite, each segment compared by equality and hashed.'''

    prefixes : tuple[Self, ...]
    '''`{q : q ⊏ self}`, root first: the root, `.x`, `.x.y` of `.x.y.z`. (implementations can cache this since it's often used)'''


type PlaceToPlace[P: Place] = Callable[[P], P]
'''
A function on places: what an op that derives a name is handed.
'''


# ---------------------------------------------------------------- prefixing and ordering.
# This uses set notation: ⊏ is strictly contained in, ⊑ is contained in or equal to
# For e.g. pytree segments
#     a ⊏ b means a is a prefix of b (e.g. a : State.x, b : State.x.y.z), 
#     a ⊑ b means a is a prefix of b or equal to b (e.g. a : State.x.y.z, b : State.x.y.z)
#     
# This makes it easier to connect it to e.g. Lean later.

def is_prefix[P: Place](a: P, b: P) -> bool:
    """`a ⊑ b`, equality included."""
    return b.segments[:len(a.segments)] == a.segments

def comparable[P: Place](a: P, b: P) -> bool:
    """`a ⊑ b or b ⊑ a`."""
    return is_prefix(a, b) or is_prefix(b, a)


#--------------------------------------------------------- the lookup
def comparables_in[P: Place](place: P, among: Collection[P],
                             below: 'Set[P] | None' = None) -> Iterator[P]:
    """
    Every place in `among` comparable to `place`: itself, its prefixes root first, then
    its extensions in the order of `among`. `below` is `{q : q ⊏ x, x ∊ among}` when a
    caller keeps it, so a place with no extension costs no scan of `among`.
    """
    if place in among:
        yield place
    yield from (q for q in place.prefixes if q in among)
    if place in (strictly_below(among) if below is None else below):
        yield from (x for x in among if x != place and is_prefix(place, x))


def strictly_below[P: Place](among: Iterable[P]) -> set[P]:
    """`{q : q ⊏ x, x ∊ among}`: every place with an extension in `among`."""
    return {q for x in among for q in x.prefixes}

# ---------------------------------------------------------------- the lookup, over a sequence: what the rules spell

def overlaps[P: Place](places: Iterable[P], others: Collection[P],
                       below: 'Set[P] | None' = None) -> Iterator[tuple[P, P]]:
    """Overlaps between the two places. `below` is `strictly_below(others)` when a caller
    keeps it (a graph caches its owners' and readers')."""
    below = strictly_below(others) if below is None else below
    for place in places:
        for other in comparables_in(place, others, below):
            yield place, other


def self_overlaps[P: Place](places: Iterable[P]) -> Iterator[tuple[P, P]]:
    """Every comparable pair among `places`, each once, as `(later, earlier)` in the
    order given -- a duplicate is a pair, a place with itself is not. Empty exactly when
    the places form an antichain. `overlaps` of each place with those already seen."""
    seen  : set[P] = set()
    below : set[P] = set()
    for place in places:
        for other in comparables_in(place, seen, below):
            yield place, other
        seen.add(place)
        below.update(place.prefixes)
