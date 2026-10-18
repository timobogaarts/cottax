'''
`relational/contracts.py`: the levels accept by shape, the finals' flags toggle their roles, and
an implementation of a level covers every final under it -- in a toy place of plain
strings, so the contracts are shown generic in the place.
'''

from dataclasses import dataclass

import pytest

from cottax.core import violations
from cottax.relational import Min, Objective
from cottax.core.driven import Driven, DriverContract, PerProblem, PerUnknown
from cottax.core.driven.nodes import NotAccepted
from cottax.core.driven import Start, Steps
from cottax.relational.contracts import (
    ConstrainedMinimisation, ConstrainedMinimiser, FixedPointIteration,
    FixedPointIterator, Minimisation, Minimiser, RootFinder, RootFinding,
)
from cottax.consumers.execution.interface import DriverImplementation
from cottax.consumers.execution.run import check_implements_its_driver
from cottax.core.realisation import ImplementedDriven
from cottax.relational import Eq, Le, Relation, RelationalCondition, is_fixed_point


@dataclass(frozen=True)
class Word:
    '''A place as a tuple of strings: the whole of `Place`.'''

    segments : tuple[str, ...]

    def __str__(self) -> str:
        return '.'.join(self.segments)


def W(s: str) -> Word:
    return Word(tuple(s.split('.')))


U, V, G, R, F, C = map(W, ('u', 'v', 'g', 'r', 'f', 'c'))

ROOT_FIND   = RelationalCondition((Relation(R, None, Eq),), (), (U,), ())
FIXED_POINT = RelationalCondition((Relation(U, G, Eq), Relation(V, G, Eq)), (), (U, V), ())
MINIMISE    = RelationalCondition((), (Objective(F, Min),), (U,), ())
CONSTRAINED = RelationalCondition((Relation(C, None, Le),), (Objective(F, Min),), (U, V), ())

SHAPES = {'root find': ROOT_FIND, 'fixed point': FIXED_POINT,
          'minimise': MINIMISE, 'constrained': CONSTRAINED}

ACCEPTED = {
    RootFinder:           {'root find', 'fixed point'},
    FixedPointIterator:   {'fixed point'},
    Minimiser:            {'minimise'},
    ConstrainedMinimiser: {'minimise', 'constrained'},
}

EVERY = [RootFinding(steps=t) for t in (False, True)] + \
        [FixedPointIteration(steps=t) for t in (False, True)] + \
        [Minimisation(steps=t) for t in (False, True)] + \
        [ConstrainedMinimisation(steps=t) for t in (False, True)]


# ---------------------------------------------------------------- the roles

def test_the_shipped_roles_take_as_many_places_as_their_kind():
    assert [r.arity(FIXED_POINT) for r in (Start, Steps)] == [2, 1]
    assert issubclass(Start, PerUnknown) and issubclass(Steps, PerProblem)


# ---------------------------------------------------------------- the levels

@pytest.mark.parametrize('level, accepted', ACCEPTED.items(),
                         ids=[level.__name__ for level in ACCEPTED])
def test_a_level_accepts_its_shapes_and_refuses_the_rest(level, accepted):
    assert {name for name, node in SHAPES.items() if level.accepts(node)} == accepted


def test_an_iterator_refuses_a_fixed_point_without_one_update_per_unknown():
    '''`u = g`, `u = c` is a fixed point, but an iteration has two updates for `u` and
    none for `v`: a root finder answers it, an iterator does not.'''
    twice = RelationalCondition((Relation(U, G, Eq), Relation(U, C, Eq)), (), (U, V), ())
    once  = RelationalCondition((Relation(U, G, Eq),), (), (U, V), ())
    for node in (twice, once):
        assert is_fixed_point(node) and RootFinder.accepts(node)
        assert not FixedPointIterator.accepts(node)
        assert [type(v) for v in violations(Driven, node, FixedPointIteration(steps=False),
                                             *slots(FixedPointIteration(steps=False), node))
                ] == [NotAccepted]


def test_a_level_is_abstract_and_a_final_is_under_exactly_one_level():
    for level in ACCEPTED:
        with pytest.raises(TypeError, match='abstract'):
            level()
    for final in EVERY:
        assert [level for level in ACCEPTED if isinstance(final, level)] == \
               [next(l for l in ACCEPTED if l.accepts is type(final).accepts)]
        assert isinstance(final, DriverContract)


def test_every_flag_is_required():
    for final in (RootFinding, FixedPointIteration, Minimisation, ConstrainedMinimisation):
        with pytest.raises(TypeError):
            final()


# ---------------------------------------------------------------- the finals

def test_the_flags_toggle_the_optional_roles():
    assert [(d.takes, d.reports) for d in EVERY] == [
        ((Start,), ()),       ((Start,), (Steps,)),
        ((Start,), ()),       ((Start,), (Steps,)),
        ((Start,), ()),       ((Start,), (Steps,)),
        ((Start,), ()),       ((Start,), (Steps,)),
    ]


def test_equal_contracts_hash_alike_and_different_ones_differ():
    '''A contract is a jit key: by value, and by its final.'''
    counted = ConstrainedMinimisation(steps=True)
    assert counted == ConstrainedMinimisation(steps=True)
    assert hash(counted) == hash(ConstrainedMinimisation(steps=True))
    assert len(set(EVERY)) == len(EVERY)
    assert FixedPointIteration(steps=True) != Minimisation(steps=True)
    assert RootFinding[Word](steps=False) == RootFinding(steps=False)


# ---------------------------------------------------------------- driven, in a toy place

def slots(driver: DriverContract, node: RelationalCondition) -> tuple:
    '''Each role at its namespace, one place per unknown or one per problem.'''
    def at(role):
        many = node.unknowns if issubclass(role, PerUnknown) else (W('p'),)
        return tuple(W(f'^{role.__name__.lower()}.{u}') for u in many)
    return tuple(map(at, driver.takes)), tuple(map(at, driver.reports))


# ---------------------------------------------------------------- what implements a contract

class Gap[D: RootFinder](DriverImplementation[D]):
    '''A generic level of implementation: implements nothing until a final names it.'''
    needs    = frozenset({Start})
    provides = frozenset({Steps})

    def __call__(self, posed, context, data):
        raise NotImplementedError


class AnyRootFinder(Gap[RootFinder]):
    '''Every final under the level.'''


class OnlyRootFinding(Gap[RootFinding]):
    '''The final alone.'''


class Minimises(DriverImplementation[ConstrainedMinimisation]):
    '''Needs a start, and reports nothing.'''
    needs    = frozenset({Start})
    provides = frozenset()

    def __call__(self, posed, context, data):
        raise NotImplementedError


def refusals(driver: DriverContract, implementation, condition=ROOT_FIND) -> list:
    node = Driven(condition, driver, *slots(driver, condition))
    return [type(v) for v in check_implements_its_driver(W('p'),
                                                         ImplementedDriven(node, implementation))]


def test_an_implementation_implements_a_level_or_a_final_through_a_generic_level():
    assert AnyRootFinder.implements is RootFinder
    assert OnlyRootFinding.implements is RootFinding
    assert 'implements' not in Gap.__dict__
    with pytest.raises(TypeError, match='abstract'):
        Gap()


def test_an_implementation_of_a_level_answers_every_final_under_it():
    plain = RootFinding(steps=True)
    assert refusals(plain, AnyRootFinder()) == []
    assert refusals(plain, OnlyRootFinding()) == []
    assert refusals(ConstrainedMinimisation(steps=False), Minimises(), CONSTRAINED) == []
