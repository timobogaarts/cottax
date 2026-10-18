'''
`consumers/execution/`: a realised graph proved callable and run in plain Python, in a toy
place of plain strings -- no jax, no path, no pytree -- so both are shown generic in the
place.

The model is `test_realisation_run`'s, one implicit discipline exact to
`u = (k + sqrt(k^2 + 4k)) / 2` with `k = 1 + x`, and nested under an outer problem
finding the `x` that makes `f - t` vanish: one iteration of it a whole inner solve.
'''

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from cottax.core import (
    AbstractGraph, Function, GraphWithValues, NestingGraph, Solve, violations,
)
from cottax.core.maths.valued import InputMissing, NotAPlace, OwnedGiven
from cottax.core.frozen import OrderedMap
from cottax.core.driven import Driven, DrivenGraph, DriverContract, PerProblem, PerUnknown, Role, is_driven
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.interface_types import ConditionMap, Context, Posed
from cottax.consumers.execution.refusals import (
    DataMissing, ImplementsAnother, NotADriverImplementation, NotCallable, ProvedBelow,
    ReportUnavailable, ReturnedUnlike,
)
from cottax.consumers.execution.interface import DriverImplementation
from cottax.consumers.execution.run import InteriorConditionMap
from cottax.core.realisation import (
    ImplementDrivers, ImplementedDriven, ImplementedFunction, RealisedGraph, Unimplement,
)
from cottax.relational import Eq, Relation, RelationalCondition, is_root_find


@dataclass(frozen=True)
class Word:
    '''A place as a tuple of strings: the whole of `Place`.'''

    segments : tuple[str, ...]

    def __str__(self) -> str:
        return '.'.join(self.segments)


def W(s: str) -> Word:
    return Word(tuple(s.split('.')))


X, K, U, Y, R_U, F, T, G = map(W, ('x', 'k', 'u', 'y', 'r_u', 'f', 't', 'g'))
CLOSE, TARGET = W('DiscA.close'), W('target')


# ---------------------------------------------------------------- the contract and its implementation

class Start(PerUnknown):
    '''Where each unknown starts.'''

class Budget(PerProblem):
    '''How many steps the solve may take.'''

class Steps(PerProblem):
    '''How many it took.'''

class History(PerProblem):
    '''Every gap met, in order.'''


class Newton(DriverContract):
    '''The contract: what it accepts, and its places. The history is reported on request,
    so `reports` is a field.'''
    accepts = staticmethod(is_root_find)
    takes   = (Start, Budget)
    reports : tuple[type[Role], ...] = (Steps,)


class Bisection(DriverContract):
    '''Another contract, answered by nothing here.'''
    accepts = staticmethod(is_root_find)
    takes   = (Start, Budget)
    reports = (Steps,)


class ScalarNewton(DriverImplementation[Newton]):
    '''Scalar, finite differences, on the one condition held against zero -- the value
    at it is the gap, so this implementation reads no relation.'''

    needs    = frozenset({Start, Budget})
    provides = frozenset({Steps, History})

    def __call__(self, posed, context, data):
        (x,), (budget,) = data[Start], data[Budget]
        gaps = []
        def gap(x):
            gaps.append(r := posed.conditions((x,), context)[0])
            return r
        for step in range(budget):
            r = gap(x)
            if abs(r) < 1e-13:
                break
            h = 1e-7 * (abs(x) + 1.0)
            x -= r * h / (gap(x + h) - r)
        found = {Steps: (step,), History: (tuple(gaps),)}
        return (x,), {role: found[role] for role in posed.reports}


newton = ScalarNewton()


class Spy(DriverImplementation[Newton]):
    '''`newton`, keeping what it was handed.'''

    needs    = ScalarNewton.needs
    provides = ScalarNewton.provides
    seen : list = field(default_factory=list, compare=False)

    def __call__(self, posed, context, data):
        self.seen.append((posed, context, dict(data)))
        return newton(posed, context, data)


class Returns(DriverImplementation[Newton]):
    '''An implementation answering whatever `fn` makes of the conditions, the context and
    the data: needs nothing, and reports steps alone.'''

    needs    = frozenset()
    provides = frozenset({Steps})
    fn : Callable

    def __call__(self, posed, context, data):
        return self.fn(posed.conditions, context, data)


# ---------------------------------------------------------------- the model

def call(reads, owns, fn) -> ImplementedFunction:
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), fn)


def places(problem: Word, unknown: Word, driver: DriverContract) -> tuple:
    '''Where the driver's data and reports sit: `Assign`'s last two fields, a report at
    its role's namespace.'''
    return (((W(f'^start.{unknown}'),), (W(f'^budget.{problem}'),)),
            tuple((W(f'^{role.__name__.lower()}.{problem}'),) for role in driver.reports))


def driven(problem: Word, condition, implementation=newton, driver=Newton()) -> ImplementedDriven:
    (unknown,) = condition.unknowns
    return ImplementedDriven(Driven(condition, driver, *places(problem, unknown, driver)),
                             implementation)


def definitions(close=newton) -> OrderedMap:
    return OrderedMap({
        W('Mesh')     : call([X],       [K],   lambda x: 1.0 + x),
        W('DiscA')    : call([U, K, Y], [R_U], lambda u, k, y: u * u - k * y),
        CLOSE         : driven(CLOSE, RelationalCondition((Relation(R_U, None, Eq),), (), (U,), ()),
                               close),
        W('DiscB')    : call([U],       [Y],   lambda u: 1.0 + u),
        W('Objective'): call([U, Y],    [F],   lambda u, y: u + y),
    })


def flat(changes=OrderedMap()) -> CallableGraph:
    return CallableGraph(NestingGraph(AbstractGraph(definitions() | changes), OrderedMap()))


def nested() -> CallableGraph:
    outer = driven(TARGET, RelationalCondition((Relation(G, None, Eq),), (), (X,), ()))
    aim   = call([F, T], [G], lambda f, t: f - t)
    return CallableGraph(NestingGraph(AbstractGraph(definitions() | {W('Aim'): aim,
                                                                     TARGET: outer}),
                                      OrderedMap({CLOSE: TARGET})))


def exact_u(x: float) -> float:
    k = 1.0 + x
    return (k + math.sqrt(k * k + 4.0 * k)) / 2.0


INNER = {W('^start.u'): 1.0, W('^budget.DiscA.close'): 50}


def ran(graph, **values):
    return run(GraphWithValues(graph, OrderedMap({X: 1.0, **INNER} | {W(k): v for k, v in values.items()})))


# ---------------------------------------------------------------- the run

def test_a_run_computes_every_place_and_the_report_lands_at_its_place():
    out = ran(flat())
    assert isinstance(out, OrderedMap)
    assert math.isclose(out[U], exact_u(1.0), rel_tol=1e-12)
    assert out[F] == out[U] + out[Y] and out[X] == 1.0
    assert 0 < out[W('^steps.DiscA.close')] < 50


def test_the_data_arrives_one_tuple_per_role_and_the_conditions_as_values():
    spy = Spy()
    graph = CallableGraph(NestingGraph(AbstractGraph(definitions(close=spy)), OrderedMap()))
    run(GraphWithValues(graph, OrderedMap({X: 1.0, W('^start.u'): 2.0, W('^budget.DiscA.close'): 7})))
    posed, context, data = spy.seen[0]
    conditions = posed.conditions
    assert posed.condition is graph[CLOSE].condition and posed.reports == Newton().reports
    assert not is_driven(posed.condition)   # what the values mean, nothing of the graph
    assert data == {Start: (2.0,), Budget: (7,)}
    assert isinstance(conditions, InteriorConditionMap) and isinstance(conditions, ConditionMap)
    step = next(s for s in graph.steps if isinstance(s, Solve))
    assert InteriorConditionMap.posing(graph[CLOSE], step.interior) == conditions
    assert conditions.unknowns == (U,) and conditions.context_places == (K,) and context == (2.0,)
    assert conditions.conditions == posed.condition.conditions
    assert conditions((2.0,), context) == (2.0 * 2.0 - 2.0 * 3.0,)
    with pytest.raises(ValueError):
        conditions((2.0, 3.0), context)
    with pytest.raises(ValueError):
        conditions((2.0,), ())


def test_the_condition_map_holds_no_value_and_one_serves_every_context():
    '''Places and a graph: it hashes, compares by value, and the context is an argument.
    A graph hashes when its mappings do, so this one is built of `OrderedMap`s.'''
    spy = Spy()
    graph = CallableGraph(NestingGraph(AbstractGraph(definitions(close=spy)),
                                       OrderedMap()))
    for x in (1.0, 2.0):
        run(GraphWithValues(graph, OrderedMap({X: x, W('^start.u'): 1.0, W('^budget.DiscA.close'): 50})))
    (first, one, _), (second, two, _) = spy.seen
    assert first == second and hash(first) == hash(second) and one != two
    assert first.conditions((2.0,), two) == second.conditions((2.0,), two)


def test_one_iteration_of_the_outer_problem_is_a_whole_solve_of_the_inner():
    out = run(GraphWithValues(nested(), OrderedMap({T: 5.0, W('^start.x'): 0.0,
                                                    W('^budget.target'): 50, **INNER})))
    assert math.isclose(out[F], 5.0, rel_tol=1e-9)
    assert math.isclose(out[X], 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(out[U], exact_u(out[X]), rel_tol=1e-9)


def test_the_inputs_are_exactly_what_nothing_owns():
    graph = flat()
    given = OrderedMap({X: 1.0, U: 3.0, W('x2'): 0.0, W('^start.u'): 1.0})
    assert [type(v) for v in violations(GraphWithValues, graph, given)] == [
        InputMissing, OwnedGiven, NotAPlace]
    with pytest.raises(KeyError, match='no value given'):
        GraphWithValues(graph, OrderedMap({X: 1.0}))


def test_a_run_takes_a_graph_proved_callable():
    '''A graph proved only to a lower level is refused by name, before anything runs.'''
    valued = GraphWithValues(RealisedGraph(flat().graph), OrderedMap({X: 1.0, **INNER}))
    assert [type(v) for v in violations(run, valued)] == [ProvedBelow]
    with pytest.raises(TypeError, match='a run takes a `CallableGraph`'):
        run(valued)


def test_what_comes_back_is_counted():
    '''A function's miscount is refused naming the node; a driver's raises where it binds.
    Nothing binds short.'''
    with pytest.raises(ValueError) as refused:
        ran(flat({W('DiscB'): call([U], [Y, W('z')], lambda u: 1.0 + u)}))
    assert refused.value.args[0] == ReturnedUnlike(W('DiscB'), 2, (Y, W('z')), 'one float').message
    for close, error in ((lambda c, x, d: ((1.0, 2.0), {Steps: (1,)}), ValueError),
                         (lambda c, x, d: ((1.0,), {}), KeyError),
                         (lambda c, x, d: (1.0, 3), TypeError)):
        graph = CallableGraph(NestingGraph(AbstractGraph(definitions(close=Returns(close))),
                                           OrderedMap()))
        with pytest.raises(error):
            ran(graph)


# ---------------------------------------------------------------- the proof

def test_a_nested_solve_holds_the_same_proof_at_every_depth():
    graph = nested().graph
    for level in (DrivenGraph, RealisedGraph, CallableGraph):
        outer, = [s for s in level(graph).steps if isinstance(s, Solve)]
        inner, = [s for s in outer.interior.steps if isinstance(s, Solve)]
        assert (outer.problem, inner.problem) == (TARGET, CLOSE)
        assert type(outer.interior) is type(inner.interior) is level


# ---------------------------------------------------------------- what the proof refuses

@dataclass(frozen=True)
class RepoRef:
    '''A description of what runs a node: drawn, rewritten, proved, and not called here.'''
    url : str


def kinds(graph: NestingGraph) -> list[type]:
    return [type(x) for x in violations(CallableGraph, graph)]


def test_a_description_is_refused_a_call():
    changed = definitions() | {W('Mesh'): call([X], [K], RepoRef('git://mesh'))}
    assert kinds(NestingGraph(AbstractGraph(changed), OrderedMap())) == [NotCallable]


class Tolerance(PerProblem):
    '''A role `Newton` does not take.'''


class Tolerant(DriverImplementation[Newton]):
    '''Needs a tolerance, which no `Newton` hands it.'''

    needs    = frozenset({Start, Tolerance})
    provides = frozenset({Steps})

    def __call__(self, posed, context, data):
        raise AssertionError('refused before it runs')


def test_a_driven_node_runs_by_an_implementation_of_its_own_driver():
    '''A bare callable says nothing of what it implements, an implementation of another
    contract answers something else, and one of this contract may lack a role this driver
    has: each refused before anything runs. Data it does not need is fine (`Returns`).'''
    condition = definitions()[CLOSE].condition
    for node, refused in ((driven(CLOSE, condition, lambda d, c, x: None), NotADriverImplementation),
                          (driven(CLOSE, condition, newton, Bisection()), ImplementsAnother),
                          (driven(CLOSE, condition, Tolerant()), DataMissing),
                          (driven(CLOSE, condition, Returns(print), Newton(reports=(Steps, History))),
                           ReportUnavailable)):
        changed = definitions() | {CLOSE: node}
        assert kinds(NestingGraph(AbstractGraph(changed), OrderedMap())) == [refused]


# ---------------------------------------------------------------- choosing and changing drivers

def test_implementations_are_chosen_by_contract_and_chosen_again_on_what_the_graph_holds():
    bare = Unimplement((CLOSE,)).apply(flat().graph)
    op   = ImplementDrivers(OrderedMap({Bisection: Returns(print), Newton: newton}))
    (implement,) = op.composed(bare)
    assert implement.implementations == OrderedMap({CLOSE: newton})
    assert math.isclose(ran(CallableGraph(op.apply(bare)))[U], exact_u(1.0), rel_tol=1e-12)
    with pytest.raises(ValueError, match='nothing to implement'):
        op.apply(flat().graph)


def test_the_most_specific_contract_is_chosen_whatever_order_the_map_is_written_in():
    bare = Unimplement((CLOSE,)).apply(flat().graph)
    for written in ({DriverContract: Returns(print), Newton: newton}, {Newton: newton, DriverContract: Returns(print)}):
        (implement,) = ImplementDrivers(OrderedMap(written)).composed(bare)
        assert implement.implementations == OrderedMap({CLOSE: newton})



def test_a_report_the_driver_asks_for_lands_and_one_it_does_not_is_not_made():
    condition = definitions()[CLOSE].condition
    history   = Newton(reports=(Steps, History))
    out = ran(flat({CLOSE: driven(CLOSE, condition, newton, history)}))
    assert len(out[W('^history.DiscA.close')]) > out[W('^steps.DiscA.close')] > 0
    assert W('^history.DiscA.close') not in ran(flat())


class Explicit(ConditionMap):
    '''A condition map in closed form: an implementation tried without a graph.'''

    fn : Callable[[tuple[Any, ...], Context], tuple[Any, ...]]

    def __call__(self, unknowns: tuple[Any, ...], context: Context) -> tuple[Any, ...]:
        return self.fn(unknowns, context)


def test_an_implementation_runs_against_a_condition_map_in_closed_form():
    '''`u^2 - k` with `k` in the context: no graph, no proof, no run.'''
    condition = definitions()[CLOSE].condition
    square    = Explicit(lambda u, context: (u[0] * u[0] - context[0],))
    (u,), reports = newton(Posed(condition, (Steps, History), square, ()), (4.0,),
                           {Start: (1.0,), Budget: (50,)})
    assert math.isclose(u, 2.0, rel_tol=1e-12)
    assert reports[History][0][0] == 1.0 - 4.0 and len(reports[History][0]) > reports[Steps][0]

# ---------------------------------------------------------------- the contract, off the type parameter

def test_the_contract_is_never_written_and_is_a_driver_class():
    with pytest.raises(TypeError, match='read off the type parameter'):
        class Written(DriverImplementation[Newton]):
            implements = Bisection
    with pytest.raises(TypeError, match='not a driver class'):
        class Forward(DriverImplementation['Newton']):
            pass
    with pytest.raises(TypeError, match='bound once'):
        class Below(ScalarNewton):
            needs = frozenset()
