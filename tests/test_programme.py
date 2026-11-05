'''
`relational.programme`: a condition sorted into objectives, equalities, inequalities and
a box, as written and without a value -- in a toy place of plain strings. A `Rel` it has
no group for is refused, and so by `canonical` and `box`, which compute over it.
'''

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from cottax.core.frozen import OrderedMap
from cottax.relational.consumers.execution.jax_solvers.canonical_nlp import box, canonical, scaled
from cottax.relational.mints import Scale
from cottax.relational import (
    Eq, Le, Max, Min, Objective, Programme, Rel, Relation, RelationalCondition, programme,
)
from cottax.relational.programme import AttachedBound, NotProgrammable, check_programmable


@dataclass(frozen=True)
class Word:
    '''A place as a tuple of strings: the whole of `Place`.'''

    segments : tuple[str, ...]

    def __str__(self) -> str:
        return '.'.join(self.segments)


x, y, f, g, h, k, lo, lo2, hi, c = (Word((s,)) for s in
                                    ('x', 'y', 'f', 'g', 'h', 'k', 'lo', 'lo2', 'hi', 'c'))


def test_each_relation_in_its_group_as_written():
    '''Both senses, an equality with and without a zero side, an inequality, a lower and an
    upper bound, two lower bounds on one unknown in the order met, a `Le` against a
    condition (an inequality, not a bound) and an unknown with no bound.'''
    rels = (Relation(lo, x, Le),            # lower, x
            Relation(g, None, Eq),          # equality against zero
            Relation(x, c, Le),             # against a condition: an inequality
            Relation(None, y, Le),          # lower, y, zero
            Relation(h, k, Le),             # inequality
            Relation(y, f, Eq),             # equality, two sides
            Relation(x, hi, Le),            # upper, x
            Relation(lo2, x, Le))           # lower, x, second
    node = RelationalCondition(rels, (Objective(f, Min), Objective(g, Max)), (x, y),
                               (lo, lo2, hi))
    assert programme(node) == Programme(
        (Objective(f, Min), Objective(g, Max)),
        (rels[1], rels[5]),
        (rels[2], rels[4]),
        OrderedMap({x: (lo, lo2), y: (None,)}),
        OrderedMap({x: (hi,), y: ()}))


def test_a_le_between_two_unknowns_is_an_inequality():
    '''A bound is against a parameter or zero, never against another unknown.'''
    rel  = Relation(x, y, Le)
    node = RelationalCondition((rel,), (Objective(f, Min),), (x, y), ())
    assert programme(node).inequalities == (rel,)
    assert programme(node).lower == OrderedMap({x: (), y: ()})


def test_a_requirement_has_no_box():
    '''No unknowns: every `Le` is an inequality, and the box keys nothing.'''
    rels = (Relation(g, None, Le), Relation(None, h, Le), Relation(f, k, Eq))
    form = programme(RelationalCondition(rels, (), (), ()))
    assert (form.objectives, form.equalities, form.inequalities) == ((), (rels[2],),
                                                                      rels[:2])
    assert form.lower == form.upper == OrderedMap()


class Ne(Rel):
    '''A relation no programme has a group for.'''
    symbol = '!='


def test_an_unknown_rel_is_refused():
    '''One kind, whichever reading meets it first: `programme`, and `canonical` and `box`
    through it.'''
    rel  = Relation(x, g, Ne)
    node = RelationalCondition((Relation(g, None, Eq), rel), (), (x,), ())
    assert list(check_programmable(node)) == [NotProgrammable(rel)]
    for read in (programme,
                 lambda n: canonical(n, (1.0,), (2.0,)),
                 lambda n: box(n, (1.0,), (2.0,))):
        with pytest.raises(TypeError, match='reads `Eq` and `Le` alone'):
            read(node)


def test_scaled_reads_each_residual_in_what_it_attaches_and_the_box_in_y():
    '''`x - y = 6` read in `|k| = 2` (a scale is a magnitude, its sign dropped), `g <= 0`
    attaching nothing as it is; the box `1 <= x` over a column of -2, its sides swapped.'''
    scale = OrderedMap({Scale: k})
    node = RelationalCondition((Relation(x, y, Eq, scale), Relation(g, None, Le), Relation(lo, x, Le)),
                               (), (x,), (k, lo))
    form, (lower, upper) = scaled(node, (8.0,), (2.0, 5.0, -2.0, 1.0), (-2.0,))
    assert (form.equalities, form.inequalities) == ((3.0,), (5.0,))
    assert (float(lower[0]), float(upper[0])) == (-jnp.inf, -0.5)


def test_scaled_reads_an_objective_in_what_it_attaches_keeping_its_sense():
    '''`max f` is `min -f`, then over `|k| = 2`: a negative scale does not flip the sense.'''
    objective = Objective(f, Max, OrderedMap({Scale: k}))
    node = RelationalCondition((), (objective,), (x,), (k,))
    assert scaled(node, (1.0,), (6.0, -2.0), (1.0,))[0].objectives == (-3.0,)


def test_a_bound_attaches_nothing():
    rel  = Relation(lo, x, Le, OrderedMap({Scale: k}))
    node = RelationalCondition((rel,), (Objective(f, Min),), (x,), (lo, k))
    assert list(check_programmable(node)) == [AttachedBound(rel)]
