'''The rewrite that writes a relation: `FixedPointCut`, cuts closed by a problem on their copies.'''

from collections.abc import Iterator, Sequence

from ..core import (
    ComposedOp, Cut, GraphOp, Insert, Label, NestingGraph, Place, Violation,
)
from ..core.frozen import OrderedMap
from .relational_condition import Eq, Relation, RelationalCondition


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
            unknowns   = self.copies, parameters=(),
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
