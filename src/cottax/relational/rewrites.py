'''The rewrites on a relational condition: `Determine` and `Undetermine`, which name and drop
its unknowns, and `FixedPointCut`, cuts closed by a problem on their copies.'''

from collections.abc import Iterator, Sequence

from ..core import (
    ComposedOp, Cut, GraphOp, Insert, Label, NestingGraph, Place, SetOwnsInside, Violation,
)
from ..core.frozen import OrderedMap
from ..core.maths.plan.ops import stores_owns_inside
from .relational_condition import Eq, Relation, RelationalCondition


# ---------------------------------------------------------------- the unknowns

def check_unknowns(op: 'Determine') -> Iterator[Violation]:
    """Something is determined."""
    if not op.owns_inside:
        yield NoUnknowns()


def check_determines_nothing_yet(op: 'Determine', graph: NestingGraph) -> Iterator[Violation]:
    """The node determines nothing yet: unknowns are set, never added to."""
    node = graph[op.node]
    if stores_owns_inside(node) and node.owns_inside:       # else `SetOwnsInside` refuses it
        yield AlreadyDetermined(op.node, node.owns_inside)


def check_determines_something(op: 'Undetermine', graph: NestingGraph) -> Iterator[Violation]:
    """There is something to drop."""
    node = graph[op.node]
    if stores_owns_inside(node) and not node.owns_inside:   # else `SetOwnsInside` refuses it
        yield AlreadyUndetermined(op.node)


class Determine[N: Label, V: Place](SetOwnsInside[N, V]):
    """Name the unknowns a requirement determines: `SetOwnsInside` in the words of a
    relation, on a node that determines nothing yet."""

    checks = (check_unknowns,)
    checks_graph = (check_determines_nothing_yet,)

    def __repr__(self) -> str:
        return f'determine({self.node} <- {", ".join(str(u) for u in self.owns_inside)})'


class Undetermine[N: Label, V: Place](SetOwnsInside[N, V]):
    """The inverse of `Determine`: the unknowns dropped, the condition node kept."""

    owns_inside : tuple[V, ...] = ()

    checks_graph = (check_determines_something,)

    def __repr__(self) -> str:
        return f'undetermine({self.node})'


# ---------------------------------------------------------------- cut, closed

def check_cuts(op: 'FixedPointCut') -> Iterator[Violation]:
    '''Something is cut, and no variable twice: two copies of one place would assume two
    values for it.'''
    if not op.cuts:
        yield NothingCut()
        return
    if dupes := tuple(dict.fromkeys(v for v in op.vars if op.vars.count(v) > 1)):
        yield CutTwice(dupes)


class FixedPointCut[N: Label, V: Place](ComposedOp[N, V]):
    '''
    Several cuts and the problem that determines every copy they mint: `^copy.x = x` per
    cut, so the condition *is* the next iterate and the loop is re-closed through it.
    '''

    cuts : tuple[Cut[N, V], ...]
    at   : N
    '''Where the problem closing the cuts is bound -- a binding of its own.'''

    checks = (check_cuts,)

    @property
    def vars(self) -> tuple[V, ...]:
        '''What was cut, in the order the cuts were given.'''
        return tuple(c.var for c in self.cuts)

    @property
    def copies(self) -> tuple[V, ...]:
        '''The copies the readers end up reading, one per cut.'''
        return tuple(c.copy for c in self.cuts)

    @property
    def definition(self) -> RelationalCondition[V]:
        '''One relation per cut, paired by the order the cuts were given.'''
        return RelationalCondition(
            relations  = tuple(Relation(m, v, Eq) for m, v in zip(self.copies, self.vars)),
            objectives = (),
            owns_inside = self.copies, reads_outside=(),
        )

    def composed(self, graph: NestingGraph[N, V]) -> Sequence[GraphOp[N, V]]:
        return (*self.cuts, Insert(OrderedMap({self.at: self.definition})))

    def __repr__(self) -> str:
        cut = ', '.join(f'{v}->{m}' for v, m in zip(self.vars, self.copies))
        return f'fixed_point_cut({cut} @ {self.at})'


# ----------------------------------------------------------------- violation finals

class NothingCut(Violation):
    error    = ValueError
    template = 'fixed_point_cut(): nothing cut'

class CutTwice(Violation):
    places : tuple
    error    = ValueError
    template = ('{places!r} is cut twice in one closure -- two copies of one place would '
                'assume two values for it')


class NoUnknowns(Violation):
    error    = ValueError
    template = 'determine(): no unknowns -- a condition node that determines nothing is what it already is'


class AlreadyDetermined(Violation):
    requirement : Label
    unknowns    : tuple[Place, ...]
    error    = TypeError
    template = ('{requirement!r} already determines {unknowns!r} -- `Undetermine` it first, so '
                'the earlier choice is dropped deliberately rather than silently replaced')


class AlreadyUndetermined(Violation):
    problem : Label
    error    = TypeError
    template = '{problem!r} already determines nothing, so there is nothing to take off'
