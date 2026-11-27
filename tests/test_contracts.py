'''
`relational/contracts.py`: the levels accept by shape, the finals declare the roles they take
and may report, and an implementation of a level covers every final under it -- in a toy place of plain
strings, so the contracts are shown generic in the place.
'''

from toy_places import Segmented
from cottax.core import Place
from dataclasses import dataclass

import pytest

from cottax.core import ConditionalNode, violations
from cottax.core.frozen import OrderedMap
from cottax.relational import Min, Objective
from cottax.core.conditional import driven, driving_violations
from cottax.core.conditional import DriverContract, PerCondition, PerUnknown
from cottax.core.conditional.driver import NotAccepted
from cottax.core.conditional import Start, Steps
from cottax.relational.contracts import (
    ConstrainedMinimisation, ConstrainedMinimiser, FixedPointIteration,
    FixedPointIterator, Minimisation, Minimiser, RootFinder, RootFinding,
)
from cottax.consumers.execution.interface import DriverImplementation
from cottax.consumers.execution.run import check_implements_its_driver
from cottax.core.realisation import implemented
from cottax.relational import Eq, Le, Relation, RelationalCondition, is_fixed_point


@dataclass(frozen=True)
class Word(Segmented):
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

EVERY = [RootFinding(), FixedPointIteration(), Minimisation(), ConstrainedMinimisation()]


# ---------------------------------------------------------------- the roles

def test_the_shipped_roles_take_as_many_places_as_their_kind():
    '''A role's places stand for what its `index` says: a start for each unknown, the steps
    for the problem as a whole.'''
    assert [r.index(FIXED_POINT) for r in (Start, Steps)] == [(U, V), (None,)]
    assert [r.arity(FIXED_POINT) for r in (Start, Steps)] == [2, 1]
    assert issubclass(Start, PerUnknown) and issubclass(Steps, PerCondition)


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
        assert [type(v) for v in driving_violations(ConditionalNode(node), FixedPointIteration(),
                                                    *slots(FixedPointIteration(), node))
                ] == [NotAccepted]


def test_a_level_is_abstract_and_a_final_is_under_exactly_one_level():
    for level in ACCEPTED:
        with pytest.raises(TypeError, match='abstract'):
            level()
    for final in EVERY:
        assert [level for level in ACCEPTED if isinstance(final, level)] == \
               [next(l for l in ACCEPTED if l.accepts is type(final).accepts)]
        assert isinstance(final, DriverContract)


# ---------------------------------------------------------------- the finals

def test_the_finals_declare_what_they_take_and_may_report():
    '''Every final starts from a `Start` and may report `Steps`; a constrained minimisation
    may be given `Scales` too. Nothing is a flag: a report is asked for by placing it.'''
    from cottax.relational.contracts import Scales
    assert [(tuple(d.takes), tuple(d.may_take), tuple(d.may_report)) for d in EVERY] == [
        ((Start,), (), (Steps,)), ((Start,), (), (Steps,)), ((Start,), (), (Steps,)),
        ((Start,), (Scales,), (Steps,)),
    ]
    assert [d.roles_for((Steps,)) for d in EVERY] == [((Start,), (Steps,))] * 4


def test_equal_contracts_hash_alike_and_different_ones_differ():
    '''A contract is a jit key: by value, and by its final.'''
    assert ConstrainedMinimisation() == ConstrainedMinimisation()
    assert hash(ConstrainedMinimisation()) == hash(ConstrainedMinimisation())
    assert len(set(EVERY)) == len(EVERY)
    assert FixedPointIteration() != Minimisation()
    assert RootFinding[Word]() == RootFinding()


# ---------------------------------------------------------------- driven, in a toy place

def slots(contract: DriverContract, node: RelationalCondition, asks: tuple = ()) -> tuple:
    '''The data the contract requires and the roles `asks` names, each at its namespace: a
    place per entry of its `index`, named after the unknown it stands for, or `p` for the
    problem as a whole.'''
    def at(role):
        return tuple(W(f'^{role.__name__.lower()}.{"p" if s is None else s}')
                     for s in role.index(node))
    data, reports = contract.roles_for(asks)
    return OrderedMap((r, at(r)) for r in data), OrderedMap((r, at(r)) for r in reports)


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


def refusals(driver: DriverContract, implementation, condition=ROOT_FIND, asks=()) -> list:
    node = driven(condition, driver, *slots(driver, condition, asks))
    return [type(v) for v in check_implements_its_driver.node(W('p'),
                                                         implemented(node, implementation))]


def test_an_implementation_implements_a_level_or_a_final_through_a_generic_level():
    assert AnyRootFinder.implements is RootFinder
    assert OnlyRootFinding.implements is RootFinding
    assert 'implements' not in Gap.__dict__
    with pytest.raises(TypeError, match='abstract'):
        Gap()


def test_an_implementation_of_a_level_answers_every_final_under_it():
    assert refusals(RootFinding(), AnyRootFinder(), asks=(Steps,)) == []
    assert refusals(RootFinding(), OnlyRootFinding(), asks=(Steps,)) == []
    assert refusals(ConstrainedMinimisation(), Minimises(), CONSTRAINED) == []
