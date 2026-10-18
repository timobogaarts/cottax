'''A condition node's relations sorted into a `Programme`: objective, equalities,
inequalities and bounds, by place alone.'''

from collections.abc import Iterator

from ..core import Place, Violation, first
from ..core.frozen import Frozen, OrderedMap
from .relational_condition import Eq, Le, Objective, Relation, RelationalCondition


class Programme[V: Place](Frozen):
    '''A condition as `min objectives` subject to `equalities`, `inequalities` and a box:
    places only.'''

    objectives   : tuple[Objective[V], ...]
    '''Each a place and its sense, as written.'''
    equalities   : tuple[Relation[V], ...]
    '''`lhs = rhs`, a side of `None` zero.'''
    inequalities : tuple[Relation[V], ...]
    '''`lhs <= rhs`, the bounds taken out.'''
    lower        : OrderedMap[V, tuple[V | None, ...]]
    '''Per unknown, in `unknowns` order, the sides bounding it from below as met; `None`
    is zero.'''
    upper        : OrderedMap[V, tuple[V | None, ...]]
    '''Per unknown, the sides bounding it from above, likewise.'''


def check_programmable(condition: RelationalCondition) -> Iterator[Violation]:
    '''Every relation an `Eq` or a `Le`: a programme has no other group.'''
    for r in condition.relations:
        if r.op is not Eq and r.op is not Le:
            yield NotProgrammable(r)


def programme[V: Place](condition: RelationalCondition[V]) -> Programme[V]:
    '''Each relation in its group, in `condition.relations` order; a bound to its unknown's
    side of the box.'''
    first(check_programmable(condition))
    eqs, les = [], []
    lower : dict[V, list[V | None]] = {u: [] for u in condition.unknowns}
    upper : dict[V, list[V | None]] = {u: [] for u in condition.unknowns}
    for r in condition.relations:
        match _bounding(condition, r):
            case ('lower', u, side): lower[u].append(side)
            case ('upper', u, side): upper[u].append(side)
            case None:               (eqs if r.op is Eq else les).append(r)
    return Programme(condition.objectives, tuple(eqs), tuple(les),
                     OrderedMap((u, tuple(s)) for u, s in lower.items()),
                     OrderedMap((u, tuple(s)) for u, s in upper.items()))


def _bounding[V: Place](condition: RelationalCondition[V],
                        r: Relation[V]) -> tuple[str, V, V | None] | None:
    '''`('lower', u, side)` for `side <= u`, `('upper', u, side)` for `u <= side`, where the
    side is a parameter or zero; `None` for any other relation.'''
    if r.op is not Le:
        return None
    fixed = lambda side: side is None or side in condition.parameters
    if r.rhs in condition.unknowns and fixed(r.lhs):
        return 'lower', r.rhs, r.lhs
    if r.lhs in condition.unknowns and fixed(r.rhs):
        return 'upper', r.lhs, r.rhs
    return None


# ----------------------------------------------------------------- violation finals

class NotProgrammable(Violation):
    relation : object
    error    = TypeError
    template = ('{relation!r}: a nonlinear programme reads `Eq` and `Le` alone, so another '
                '`Rel` has no group')
