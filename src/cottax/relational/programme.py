'''A condition node's relations sorted into a `Programme`: objective, equalities,
inequalities and bounds, by place alone; and a `Scaling`, a driver's scales for them.'''

from collections.abc import Iterator
from typing import Any, NamedTuple

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


class Scaling[V: Place](NamedTuple):
    '''
    A problem's scales, in one value a driver reads. Each divides what it scales; only a
    scale's magnitude counts, and what has none stays as it is. An unknown's is the unit its
    solver works in, so its bounds follow it: a bound has no scale of its own.

    A constraint is keyed by its `Relation`, exactly, or by a place: every constraint whose
    first side neither zero nor a parameter is that place, `h` for `h <= h_max`, the copy for
    `^copy.y = y`. A `Relation` key wins over a place key.
    '''

    unknowns    : OrderedMap[V, Any] = OrderedMap()
    objective   : Any = None
    constraints : OrderedMap[Relation[V] | V, Any] = OrderedMap()


def check_scaling(condition: RelationalCondition, scaling: Scaling) -> Iterator[Violation]:
    '''Every key of `scaling` something of `condition`: a stale or misspelled one would
    otherwise leave what it meant unscaled, in silence.'''
    if not isinstance(scaling, Scaling):
        yield NotAScaling(type(scaling).__name__)
        return
    form = programme(condition)
    rows = (*form.equalities, *form.inequalities)
    meant_by = lambda key: (lambda r: r == key) if isinstance(key, Relation) \
        else (lambda r: _subject(condition, r) == key)
    for u in scaling.unknowns:
        if u not in condition.unknowns:
            yield ScaleOnNoUnknown(u, condition.unknowns)
    for key in scaling.constraints:
        if not any(map(meant_by(key), rows)):
            yield (ScaleOnABound(key) if any(map(meant_by(key), condition.relations))
                   else ScaleOnNoConstraint(key))


def scales_by_group[V: Place](condition: RelationalCondition[V], scaling: Scaling[V]
                              ) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    '''Each scale of `scaling` where it applies, `None` where there is none: per unknown in
    `condition.unknowns` order, per equality, and per inequality in `programme`'s order. Keys
    that say nothing of `condition` are refused.'''
    first(check_scaling(condition, scaling))
    form = programme(condition)
    given = scaling.constraints
    row = lambda r: given[r] if r in given else given.get(_subject(condition, r))
    return (tuple(scaling.unknowns.get(u) for u in condition.unknowns),
            tuple(map(row, form.equalities)), tuple(map(row, form.inequalities)))


def _subject[V: Place](condition: RelationalCondition[V], r: Relation[V]) -> V | None:
    '''The place a relation is about: its first side neither zero nor a parameter.'''
    return next((s for s in (r.lhs, r.rhs)
                 if s is not None and s not in condition.parameters), None)


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


class NotAScaling(Violation):
    got      : str
    error    = TypeError
    template = ('a problem\'s scales are one `Scaling`, not a {got}: write `flat.scaling(...)`')


class ScaleOnNoUnknown(Violation):
    place    : object
    unknowns : tuple
    error    = ValueError
    template = ('a scale at {place!r}: this problem determines {unknowns!r}, so it scales '
                'nothing here')


class ScaleOnNoConstraint(Violation):
    key      : object
    error    = ValueError
    template = ('a scale for {key!r}: no constraint of this problem is that, or has it as its '
                'first side neither zero nor a parameter, so it scales nothing here. A '
                'constraint about it on another side is keyed by its `Relation`')


class ScaleOnABound(Violation):
    key      : object
    error    = ValueError
    template = ('a scale for {key!r}: that is a bound, which follows the scale of the unknown '
                'it bounds')
