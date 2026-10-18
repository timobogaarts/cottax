'''
`Running`'s run on a caller's structures (`run(..., read_from=)`, `valued`, `Results.into`), in plain
Python over `test_execution`'s `Word` place and model -- no jax, no pytree. The state is an
`OrderedMap`, held by `Table`, the structure registered for `Word`; the surface is a toy one,
its locator reading a string as a `Word`.
'''

from dataclasses import dataclass

import pytest

from cottax.core.frozen import OrderedMap, answers, violations
from cottax.consumers.surface.interface import Locator
from cottax.consumers.execution import CallableGraph, Inputs, Running, read_from, slots
from cottax.consumers.execution.interface import DataStructure
from cottax.consumers.execution.refusals import ValuedTwice
from cottax.core import AbstractGraph, Function, NestingGraph, answer_for_places
from cottax.core.frozen.answer import Unanswered
from cottax.core.maths.dispatch import NothingToAsk
from cottax.core.driven import Driven
from cottax.core.realisation import ImplementedDriven
from cottax.relational import Eq, Relation, RelationalCondition
from test_execution import (
    CLOSE, F, Newton, Start, U, W, Word, X, call, definitions, exact_u, flat, nested, newton,
)


@answers(Word)
class Table(DataStructure[Word]):
    '''A structure for these tests: a mapping, holding the places it has a key for.'''

    def holds(self, data, place) -> bool:
        return place in data

    def read(self, data, places) -> OrderedMap:
        places = tuple(places)
        if missing := tuple(p for p in places if p not in data):
            raise KeyError(f'no place {missing!r}')
        return OrderedMap((p, data[p]) for p in places)

    def write(self, data, values) -> OrderedMap:
        return data | values


class Words(Locator):
    '''A string is the `Word` it spells.'''

    places = Word
    nodes  = Word

    def place(self, s): return W(s)
    def node(self, s):  return W(s)


@dataclass(frozen=True)
class Hat:
    '''A cut's copy of a word: `^hat.<word>`.'''

    def __call__(self, word): return W(f'^hat.{word}')


class OnWords(Running):
    copies = Hat()


s = OnWords(Words())


START, BUDGET, STEPS = W('^start.u'), W('^budget.DiscA.close'), W('^steps.DiscA.close')
BUDGETED = {Inputs: {BUDGET: 50}}


def held(state) -> dict:
    '''Each input and each start read where `state` holds it.'''
    return {Inputs: state, Start: state}


def test_a_state_values_each_input_it_holds_and_each_start_at_its_unknown():
    valued = s.valued(flat(), BUDGETED, held(OrderedMap({X: 1.0, U: 2.0})))
    assert dict(valued.values) == {X: 1.0, BUDGET: 50, START: 2.0}


def test_only_what_is_read_from_a_structure_is_read():
    state = OrderedMap({X: 1.0, U: 2.0})
    assert dict(s.valued(flat(), BUDGETED | {Start: {U: 3.0}}, {Inputs: state}).values) \
        == {X: 1.0, BUDGET: 50, START: 3.0}


def test_where_a_datum_is_read_is_the_at_handed_in():
    state = OrderedMap({X: 1.0, U: 2.0, W('guess'): 5.0})
    graph = flat().graph
    found = read_from(graph, slots(graph, at=lambda unknown: W('guess')), {Start: state})
    assert found == ((START, 5.0, "read_from[Start] at Word(segments=('guess',)), "
                                  "for Word(segments=('u',))"),)


def test_a_datum_the_graph_computes_is_not_read():
    warm = flat().graph.definitions[W('Mesh')]
    fed  = flat(OrderedMap({W('Warm'): type(warm)(type(warm.node)((X,), (START,)),
                                                  lambda x: x)}))
    valued = s.valued(fed, BUDGETED, held(OrderedMap({X: 1.0, U: 2.0})))
    assert dict(valued.values) == {X: 1.0, BUDGET: 50}


def test_a_start_shared_by_two_unknowns_is_valued_twice():
    '''Two problems starting from one place, at two unknowns: which value it takes is not
    the state's to say.'''
    v, r_v = W('v'), W('r_v')
    other  = ImplementedDriven(
        Driven(RelationalCondition((Relation(r_v, None, Eq),), (), (v,), ()), Newton(),
               ((START,), (W('^budget.Other'),)), ((W('^steps.Other'),),)), newton)
    graph  = CallableGraph(NestingGraph(AbstractGraph(
        definitions() | {W('DiscC'): call([v, X], [r_v], lambda v, x: v - x),
                         W('Other'): other}), OrderedMap()))
    with pytest.raises(ValueError) as refused:
        s.valued(graph, read_from=held(OrderedMap({X: 1.0, U: 2.0, v: 3.0})))
    assert refused.value.violation == ValuedTwice(START, ('read_from[Start] at ' + repr(U),
                                                          'read_from[Start] at ' + repr(v)))


def test_a_run_reads_by_name_and_writes_back_where_the_state_holds():
    state = OrderedMap({X: 1.0, U: 1.0})
    out   = s.run(flat(), BUDGETED, held(state))
    back  = out.into(state)
    assert back[U] == pytest.approx(exact_u(1.0), rel=1e-10)
    assert back == OrderedMap({X: 1.0, U: out[U]})             # no slot for f: not written
    assert out['f'] == pytest.approx(2.0 * exact_u(1.0) + 1.0, rel=1e-10)
    assert STEPS in out and START not in back


def test_a_value_held_at_an_owned_place_is_never_read_and_is_overwritten():
    # Only the start reads `u`, so a junk value at the computed `f` changes nothing.
    state = OrderedMap({X: 1.0, U: 1.0, F: None})
    clean = s.run(flat(), BUDGETED, held(state)).into(state)
    junk  = state | {F: -99.0}
    junk  = s.run(flat(), BUDGETED, held(junk)).into(junk)
    assert junk[F] == clean[F] != -99.0


def test_an_input_given_by_name_and_read_from_a_structure_is_given_twice():
    with pytest.raises(ValueError) as refused:
        s.valued(flat(), {Inputs: {BUDGET: 50, 'x': 2.0}}, held(OrderedMap({X: 1.0, U: 1.0})))
    assert refused.value.violation == ValuedTwice(
        X, ("values[Inputs]['x']", 'read_from[Inputs] at ' + repr(X)))


def test_an_input_neither_held_nor_given_is_refused_by_name():
    with pytest.raises(KeyError, match='budget'):
        s.run(flat(), read_from=held(OrderedMap({X: 1.0, U: 1.0})))


def test_a_nested_solve_runs_on_a_state():
    state = OrderedMap({W('t'): 6.0, X: 1.0, U: 1.0, F: None})
    out   = s.run(nested(), {Inputs: {BUDGET: 50, W('^budget.target'): 50}}, held(state))
    back  = out.into(state)
    assert back[F] == pytest.approx(6.0, abs=1e-8)
    assert CLOSE not in back


def test_a_run_is_valued_run_and_written_back():
    '''The steps taken apart give what `run` gives; an input is never written.'''
    from cottax.consumers.execution import run, write_back
    state  = OrderedMap({X: 1.0, U: 1.0, F: -99.0})
    values = run(s.valued(flat(), BUDGETED, held(state)))
    back   = write_back(flat().graph, state, values)
    assert back == s.run(flat(), BUDGETED, held(state)).into(state)
    assert back[X] == 1.0 and back[F] == values[F] != -99.0


# ---------------------------------------------------------------- where a problem takes a role

def test_role_slots_pair_each_place_with_its_problem_and_unknown():
    from cottax.core.driven import role_slots
    from test_execution import Budget, History
    graph = nested().graph
    assert role_slots(graph, Start) == ((START, CLOSE, U), (W('^start.x'), W('target'), X))
    assert role_slots(graph, Budget) == ((BUDGET, CLOSE, None),
                                         (W('^budget.target'), W('target'), None))
    assert role_slots(graph, History) == ()                     # a role no driver takes


# ---------------------------------------------------------------- the structure registered

def test_the_structure_is_the_one_registered_for_the_places_class():
    assert type(answer_for_places(DataStructure, flat().graph)) is Table


def test_a_place_class_indexed_twice_is_refused():
    '''A second structure for a class would be chosen by import order.'''
    class Again(DataStructure):
        def holds(self, data, place): ...
        def read(self, data, places): ...
        def write(self, data, values): ...

    with pytest.raises(TypeError, match=r'Word has the DataStructure answer .*Table already'):
        answers(Word)(Again)


class Other:
    '''A place: `Unindexed` has no structure.'''

    def __init__(self, s): self.s = s
    def __hash__(self):    return hash(self.s)
    def __eq__(self, o):   return type(o) is type(self) and o.s == self.s
    def __str__(self):     return self.s

    @property
    def segments(self):    return (self.s,)


class Unindexed(Other):
    pass


def over(cls) -> NestingGraph:
    '''A graph whose places are of `cls`.'''
    return NestingGraph(AbstractGraph(OrderedMap({W('f'): Function((cls('x'),), (cls('y'),))})),
                        OrderedMap())


def test_a_place_class_nobody_indexes_is_refused_by_name():
    with pytest.raises(TypeError, match='test_execution_state.Unindexed has no '
                                        'DataStructure answer: import the module'):
        answer_for_places(DataStructure, over(Unindexed))
    assert [type(v) for v in violations(answer_for_places, DataStructure, over(Unindexed))] \
        == [Unanswered]


def test_a_graph_with_no_place_is_refused():
    empty = NestingGraph(AbstractGraph(OrderedMap()), OrderedMap())
    assert [type(v) for v in violations(answer_for_places, DataStructure, empty)] == [NothingToAsk]
