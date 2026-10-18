'''
The relational declaring part of a `Surface`: `RelationalDeclaring`, the implicit models,
the constraints and the condition nodes; and the readings of its arguments (`within_of` and `against_of` for the constraints, `bounds_of`, `design_of`,
`targets_of`), written over the declaring part's `term` and `per_place`.
'''

from collections.abc import Iterable, Mapping
from typing import Any

from ....core import NodeDefinition, Place, PlaceToPlace
from ....core.frozen import OrderedMap, choice
from ... import Eq, Max, Min, Objective, Rel, Relation, RelationalCondition
from ....consumers.declaration.interface import Declaring
from ....consumers.declaration.interface_types import (
    Declaration, Keyword, PlaceWithValue, Side, declares,
)
from ....consumers.surface.interface import Part
from ....consumers.surface.interface_types import Named
from ...mints import Lower, Target, Upper
from .declarations import (
    Bounds, Constraint, EqualityConstraint, FixedPoint, InequalityConstraint, NotAPair, Problem,
    Residual,
)


type Pair[P]    = tuple[Side[P] | None, Side[P] | None]
# A lower and upper side, each a name, a place, a value (possibly tagged), or None for no bound

type Within[P]  = Pair[P] | Mapping[Named[P], Pair[P]] | None
# Lower and upper bounds, either one pair for the one place there is, or a mapping of each place to its pair, or None for no bounds.

type Against[P] = Side[P] | Mapping[Named[P], Side[P]] | None
# An equality's target, either one side for the one output there is, or a mapping of each output to its side, or None for no target.

type SubjectTo  = Constraint | RelationalCondition | Iterable[Constraint | RelationalCondition]
# What a problem is subject to: one constraint wrapper or bare requirement, or a collection.

type Design[P]  = tuple[Named[P], ...] | list[Named[P]] | Mapping[Named[P], Pair[P]]
# A problem's unknowns, either a sequence of names, or a mapping of each unknown to its bounds


def within_of[V](surface: Declaring, fn: Any, outputs: tuple[V, ...], within: Within
                 ) -> Bounds[V]:
    '''A constraint's `within=`: each output's `(lo, hi)`, at most zero where none is given.'''
    pairs = dict.fromkeys(outputs, (None, 0)) if within is None else within
    return bounds_of(surface, outputs, pairs)


def against_of[V](surface: Declaring, fn: Any, outputs: tuple[V, ...], against: Against
                  ) -> OrderedMap[V, PlaceWithValue[V] | None]:
    '''An equality constraint's `against=`: each output's target, zero where none is given.'''
    return targets_of(surface, outputs, against)


@choice(Part)
class RelationalDeclaring(Declaring):
    '''
    The relational declarations: implicit models and constraints as functions, and the
    relational conditions over them.
    '''

    # ------------------------------------------------------------ wrappers

    residual = declares(
        Residual,
        '''A function returning a residual per unknown, and the problem where each vanishes:
        `residual(fn, *unknowns)`, or `@residual(*unknowns)`.''')

    fixed_point = declares(
        FixedPoint,
        '''A function returning each unknown's next value, and the problem holding each equal
        to it: `fixed_point(fn, *unknowns)`, or `@fixed_point(*unknowns)`.''')

    inequality_constraint = declares(
        InequalityConstraint,
        '''A function computing its outputs, each within its `(lo, hi)` -- one pair for the
        one output there is, or a mapping of pairs -- and with none given at most zero:
        `inequality_constraint(fn, *outputs, within=(None, 'g_max'))`, or the decorator.''',
        bounds=Keyword('within', within_of))

    equality_constraint = declares(
        EqualityConstraint,
        '''A function computing its outputs, each equal to its target -- one for the one
        output there is, or a mapping -- and with none given zero:
        `equality_constraint(fn, *outputs, against='h_target')`, or the decorator.''',
        targets=Keyword('against', against_of))

    # ------------------------------------------------------------ condition nodes

    def requirement(self, condition: Place, op: type[Rel] = Eq) -> RelationalCondition:
        '''`c op 0`, determining nothing: held by a problem subject to it, or where an
        architecture puts it.'''
        return RelationalCondition((Relation(condition, None, op),), (), (), ())

    def satisfy(self, *, design: Design, subject_to: SubjectTo = ()) -> Problem:
        '''Any point of `design` (names, or a mapping to `(lo, hi)`) meeting the constraints
        it is subject to (one, or a collection), nothing minimised: a root find where those
        are equalities.'''
        return _problem(self, (), design, subject_to)

    def optimise(self, *, design: Design, minimise: Place | None = None,
                 maximise: Place | None = None, subject_to: SubjectTo = ()) -> Problem:
        '''Minimise or maximise one objective over `design` (names, or a mapping to
        `(lo, hi)`), subject to constraints -- one, or a collection: constraint wrappers in
        the model, or bare requirements, held by the problem.'''
        if (minimise is None) == (maximise is None):
            raise TypeError('an optimisation drives one objective: give `minimise=` or '
                            '`maximise=`')
        objective, sense = (minimise, Min) if maximise is None else (maximise, Max)
        return _problem(self, (Objective(objective, sense),), design, subject_to)




def _problem(surface: Declaring, objectives: tuple[Objective, ...], design: Design,
             subject_to: SubjectTo) -> Problem:
    unknowns, bounds = design_of(surface, design)
    one = isinstance(subject_to, (Declaration, NodeDefinition))
    return Problem(unknowns, bounds, objectives, (subject_to,) if one else tuple(subject_to))


# ---------------------------------------------------------------- the relational readings

def bounds_of[V](surface: Declaring, among: tuple[V, ...], within: Within) -> Bounds[V]:
    '''The `(lo, hi)` written on `among`, one pair for the one place there is or a mapping of
    pairs: each side a term, minted at `^lower` or `^upper` where a value, or `None` for no
    bound. Anything but a pair is refused.'''
    pairs = surface.per_place(among, within) if within is not None else {}
    for x, pair in pairs.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise NotAPair(x, repr(pair)).raised
    def side(i: int, value_at: PlaceToPlace[V]) -> OrderedMap[V, PlaceWithValue[V] | None]:
        return OrderedMap((x, surface.term(x, pair[i], value_at))
                          for x, pair in pairs.items() if pair[i] is not None)
    return Bounds(side(0, Lower()), side(1, Upper()))


def design_of[V](surface: Declaring, design: Design) -> tuple[tuple[V, ...], Bounds[V]]:
    '''A problem's unknowns, in order, and their bounds: names, or a mapping of each to its
    `(lo, hi)`. What stands for places (its `Written` answer) is all of them, a key one
    alone.'''
    if isinstance(design, str):
        raise TypeError(f'design={design!r}: a design is names, or a mapping to `(lo, hi)` -- '
                        f'write one unknown as `({design!r},)`')
    if isinstance(design, Mapping):
        unknowns = tuple(map(surface.place, design))
    else:
        unknowns = surface.places(design)
    return unknowns, bounds_of(surface, unknowns, design if isinstance(design, Mapping) else None)


def targets_of[V](surface: Declaring, outputs: tuple[V, ...], against: Against
                  ) -> OrderedMap[V, PlaceWithValue[V] | None]:
    '''An equality's target per output, one for the one output there is or a mapping, and
    with none given zero; each a term, minted at `^target` where a value.'''
    written = surface.per_place(outputs, dict.fromkeys(outputs, 0) if against is None
                                else against)
    return OrderedMap((x, surface.term(x, w, Target())) for x, w in written.items())
