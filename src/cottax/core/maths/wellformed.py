'''
What a readable graph obeys: `WellFormed`, five rules. Three on the nesting -- its ends are
bound, own something inside, and form a forest -- and two on the bindings -- one producer
per place, and every read exactly an owned place or unrelated to all of them. Written
against the stored graph and `owners_of`, never against the views they guard.
'''

from collections.abc import Iterable, Iterator, Mapping

from ..frozen import Violation
from .definition import NodeDefinition
from .graph import Graph
from .kinds import owns_inside
from .places import Label, Place, comparables_in, is_prefix, overlaps
from .rules import Rules


# ---------------------------------------------------------------- the nesting

def check_nesting_ends_bound(g: Graph) -> Iterator[Violation]:
    '''Check the within edge has both nodes in the graph.

    W ⊆ N × N
    '''
    for inner, outer in g.within.items():
        if inner not in g.definitions or outer not in g.definitions:
            yield Unbound(inner, outer)


def check_nesting_ends_own_inside(g: Graph) -> Iterator[Violation]:
    '''
    Both ends of a nesting own something inside: only a condition node that hands unknowns
    to its cycle iterates anything for another to be nested in.

    W ⊆ N_P × N_P
    '''
    for inner, outer in g.within.items():
        if missing := tuple(e for e in (inner, outer)
                            if e in g.definitions and not owns_inside(g.definitions[e])):
            yield EndOwnsNothingInside(missing)


def check_no_nesting_loops(g: Graph) -> Iterator[Violation]:
    '''
    Nesting can not be looped: this would imply nesting within itself in some form.

    (N_P, W) is a forest

    Only the acyclicity is checked here: `within` is a mapping, so one outer per inner
    ((n,m), (n,m') ∊ W ⟹ m = m') holds by construction, and acyclic with out-degree
    at most one is a forest.
    '''
    seen : set = set()
    for start in g.within:
        if start in seen:
            continue
        chain : list = []
        at = start
        while at is not None and at not in seen:
            if at in chain:
                if g.within[at] == at:
                    yield SelfNested(at)
                else:
                    yield NestingLoop(at, g.within[at], tuple(chain[chain.index(at):]))
                break
            chain.append(at)
            at = g.within.get(at)
        seen.update(chain)


# ---------------------------------------------------------------- the bindings

def check_one_producer_per_place(g: Graph) -> Iterator[Violation]:
    '''
    The owned places, over every node, form an antichain (do not overlap).

    o || m  ∀  n_i ≠ n_j,  n_i, n_j ∊ N:   o ∊ O_i, m ∊ O_j        
    '''
    yield from owned_overlaps((n, v) for n, d in g.definitions.items() for v in d.owns)


def owned_overlaps[N: Label, V: Place](owned: Iterable[tuple[N, V]]) -> Iterator[Violation]:
    '''Every comparable pair of `(owner, place)`s, the later against each earlier one it
    meets, so each violation names its own two owners.'''
    seen  : dict[V, N] = {}
    below : set[V]     = set()
    for node, place in owned:
        for other in comparables_in(place, seen, below):
            yield owned_comparably(place, node, other, seen[other])
        seen.setdefault(place, node)
        below.update(place.prefixes)


def check_reads_are_exact(g: Graph) -> Iterator[Violation]:
    '''
    No read is strictly inside or around another node's owned place: a read is
    exactly an owned place, or unrelated to every one.

    o || p  or  o = p   ∀ n_i, n_j ∊ N:  o ∊ O_i, p ∊ I_j        
    '''
    readers, owners = g.readers, owners_of(g.definitions)
    for place, other in overlaps(readers, owners):
        if other != place:                                   # equal is the (correct) edge
            yield (ReadInsideOwner if is_prefix(other, place) else OwnerInsideRead)(
                place, other, tuple(readers[place]), owners[other])


def owners_of[N: Label, V: Place](definitions: Mapping[N, NodeDefinition[V]]) -> dict[V, N]:
    '''Who owns each place. One producer per place is `check_one_producer_per_place`'s rule; assumed here.'''
    return {v: n for n, d in definitions.items() for v in d.owns}


class WellFormed(Rules):
    '''The graph is well-formed (valid)'''
    rules = (check_nesting_ends_bound, check_no_nesting_loops, check_nesting_ends_own_inside,
             check_one_producer_per_place, check_reads_are_exact)


# ----------------------------------------------------------------- violation finals

class Unbound(Violation):
    inner : Label
    outer : Label
    error    = KeyError
    template = 'nesting {inner!r} in {outer!r}: an end is not a node of this graph'


class EndOwnsNothingInside(Violation):
    nodes : tuple[Label, ...]
    error    = ValueError
    template = ('{nodes} determines nothing, so no iteration of it answers anything -- an end '
                 'of a nesting is a problem: determine it, naming the unknowns it fixes, or '
                 '`Unnest` it first so the plan says the nesting went')


class SelfNested(Violation):
    inner : Label
    error    = ValueError
    template = '{inner!r} inside itself -- an iteration does not answer the problem it is'


class NestingLoop(Violation):
    inner : Label
    outer : Label
    chain : tuple[Label, ...]
    error    = ValueError
    template = ('{outer!r} is already answered inside {inner!r} ({chain}), so nesting the one '
                 'in the other closes a loop -- nesting is a forest')


class OwnedTwice(Violation):
    place   : Place
    node    : Label
    other   : Label
    error    = ValueError
    template = 'output {place!r} is owned by both {other!r} and {node!r}'

class OwnInsideOwner(Violation):
    place   : Place
    witness : Place
    node    : Label
    other   : Label
    error    = ValueError
    template = ('output {place!r} of {node!r} lies inside {witness!r}, owned by {other!r} -- '
                 'two nodes would determine one place')

class OwnAroundOwner(Violation):
    place   : Place
    witness : Place
    node    : Label
    other   : Label
    error    = ValueError
    template = ('output {witness!r} of {other!r} lies inside {place!r}, owned by {node!r} -- '
                 'two nodes would determine one place')

class ReadInsideOwner(Violation):
    place   : Place
    witness : Place
    readers : tuple[Label, ...]
    owner   : Label
    error    = ValueError
    template = ('{readers} read {place!r}, which lies inside {witness!r}, owned by {owner!r} -- '
                 'reads are matched by equality, so this would silently be a boundary input')

class OwnerInsideRead(Violation):
    place   : Place
    witness : Place
    readers : tuple[Label, ...]
    owner   : Label
    error    = ValueError
    template = ('output {witness!r} of {owner!r} lies inside {place!r}, read by {readers} -- '
                 'that read would silently miss this output')

#--------------------------------------------------------------------- Violation pickers
# These always yield a violation, separate from the main function to reduce them to just
# checking whether there is a violation.
def owned_comparably[N: Label, V: Place](place: V, node: N, other: V, owner: N) -> Violation:
    '''The violation for `place`, owned by `node`, comparable to `other`, owned by
    `owner`.'''
    if other == place:
        return OwnedTwice(place, node, owner)
    return (OwnInsideOwner if is_prefix(other, place) else OwnAroundOwner)(place, other, node, owner)
