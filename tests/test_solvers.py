'''
`relational/consumers/execution/jax_solvers/`: the canonical form and the next values over given values, and each
optimistix implementation answering its contract, called directly and run through
`execution`. Float64, as a solve to a tight tolerance needs.
'''

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import DictKey

from cottax.core import Function, GraphWithValues
from cottax.relational import Min, Objective
from cottax.core.frozen import OrderedMap
from cottax.core.driven import Driven
from cottax.core.driven import Start, Steps
from cottax.relational.contracts import FixedPointIteration, Minimisation, RootFinding
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.interface_types import ConditionMap, Posed
import purposes
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedDriven, ImplementedFunction
from cottax.relational import Eq, Le, Relation, RelationalCondition
from cottax.relational.consumers.execution.jax_solvers.canonical_nlp import Canonical, canonical
from cottax.relational.consumers.execution.jax_solvers import BFGS, Newton, Picard
from cottax.relational.consumers.execution.jax_solvers.canonical_nlp import next_values
from graphs import graph_of


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def V(name: str) -> VarPath:
    return VarPath((DictKey(name),))


def N(name: str) -> NodePath:
    return NodePath((DictKey(name),))


STARTS = purposes.Start()

TIGHT = dict(rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------- toy condition maps

class SquareGap(ConditionMap):
    '''`u * u - z` at the one condition, `z` the context.'''

    def __call__(self, unknowns, context):
        (u,), (z,) = unknowns, context
        return (u * u - z,)


class TwoOfOne(ConditionMap):
    '''Two scalar conditions of one unknown: `u0 * u0 - 4` and `u1 - 1` of a vector,
    or both of one scalar.'''

    def __call__(self, unknowns, context):
        u = jnp.atleast_1d(unknowns[0])
        return (u[0] * u[0] - 4.0, u[-1] - 1.0)


# ---------------------------------------------------------------- the forms

MIXED = RelationalCondition((Relation(V('u'), V('a'), Eq),
                             Relation(None, V('c'), Le),
                             Relation(V('b'), None, Eq)), (Objective(V('f'), Min),), (V('u'),), ())


def test_the_canonical_form_reads_each_side_from_the_unknowns_the_values_or_zero():
    '''`u = a`, `0 <= c`, `b = 0` with objective `f`: equalities and inequalities apart,
    each in written order, an inequality as `<= 0`.'''
    assert MIXED.conditions == (V('f'), V('a'), V('c'), V('b'))
    got = canonical(MIXED, (10.0,), (7.0, 1.0, 3.0, 2.0))
    assert got == Canonical(objectives=(7.0,), equalities=(9.0, 2.0), inequalities=(-3.0,))


def test_a_residual_is_formed_leaf_wise_over_pytree_sides():
    condition = RelationalCondition((Relation(V('a'), V('b'), Eq),
                                     Relation(None, V('b'), Eq)), (), (V('u'),), ())
    d, neg = canonical(condition, (0.0,), ({'x': 5.0, 'y': (1.0,)},
                                            {'x': 2.0, 'y': (4.0,)})).equalities
    assert d == {'x': 3.0, 'y': (-3.0,)}
    assert neg == {'x': -2.0, 'y': (-4.0,)}


def test_next_values_are_the_right_sides_in_the_order_of_the_unknowns():
    condition = RelationalCondition((Relation(V('v'), V('gv'), Eq),
                                     Relation(V('u'), V('gu'), Eq)), (), (V('u'), V('v')), ())
    assert next_values(condition, (0.0, 0.0), (1.0, 2.0)) == (2.0, 1.0)


# ---------------------------------------------------------------- called directly

def driven_node(condition, driver, reports=()):
    '''A start per unknown.'''
    return Driven(condition, driver, (tuple(STARTS(u) for u in condition.unknowns),),
                  tuple((p,) for p in reports))


def posed(node, conditions) -> Posed:
    '''What an implementation is posed for `node`, tried through `conditions`.'''
    return Posed(node.condition, node.contract.reports, conditions, ())


SQUARE_ROOT = RelationalCondition((Relation(V('r'), None, Eq),), (), (V('u'),), ())


@pytest.mark.slow
def test_a_newton_reports_its_steps_only_when_asked():
    quiet  = driven_node(SQUARE_ROOT, RootFinding(steps=False))
    counts = driven_node(SQUARE_ROOT, RootFinding(steps=True), [V('n')])
    start  = {Start: (jnp.asarray(1.0),)}

    (u,), reports = Newton(**TIGHT)(posed(quiet, SquareGap()), (jnp.asarray(9.0),), start)
    assert float(u) == pytest.approx(3.0, rel=1e-10) and reports == {}

    (u,), reports = Newton(**TIGHT)(posed(counts, SquareGap()), (jnp.asarray(9.0),), start)
    ((steps,),) = reports.values()
    assert set(reports) == {Steps} and int(steps) > 1


CALLS = []


class CountedSquareGap(ConditionMap):
    '''`SquareGap`, counting its calls: under optimistix's own jit, a call is a trace.'''

    def __call__(self, unknowns, context):
        CALLS.append(1)
        (u,), (z,) = unknowns, context
        return (u * u - z,)


def test_an_eager_newton_compiles_once_per_node_not_once_per_call():
    '''What optimistix is handed is a module-level function, and the node and the map ride
    in `args` as static keys, so a second solve of the same node is a cache hit.'''
    node, start = driven_node(SQUARE_ROOT, RootFinding(steps=False)), 1.0
    CALLS.clear()
    for z in (4.0, 9.0, 16.0):
        (u,), _ = Newton(**TIGHT)(posed(node, CountedSquareGap()), (jnp.asarray(z),),
                                  {Start: (jnp.asarray(start),)})
        assert float(u) == pytest.approx(math.sqrt(z), rel=1e-10)
        if z == 4.0:
            traced = len(CALLS)
    assert len(CALLS) == traced


@pytest.mark.slow
def test_a_newton_counts_squareness_in_leaves_not_relations():
    node = driven_node(RelationalCondition((Relation(V('r1'), None, Eq),
                                            Relation(V('r2'), None, Eq)), (), (V('u'),), ()),
                       RootFinding(steps=False))
    (u,), _ = Newton(**TIGHT)(posed(node, TwoOfOne()), (), {Start: (jnp.asarray([1.0, 0.0]),)})
    assert u.tolist() == pytest.approx([2.0, 1.0], rel=1e-10)

    with pytest.raises(ValueError, match='gives 2 for 1'):
        Newton(**TIGHT)(posed(node, TwoOfOne()), (), {Start: (jnp.asarray(1.0),)})


# ---------------------------------------------------------------- run through a graph

def call(reads, owns, fn) -> ImplementedFunction:
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), fn)


def driven(condition, driver, implementation, reports=()) -> ImplementedDriven:
    return ImplementedDriven(driven_node(condition, driver, reports), implementation)


def cosine(driver, implementation, reports=()) -> CallableGraph:
    '''`u = a * cos(u)`, a contraction for `a < 1`.'''
    return CallableGraph(graph_of(OrderedMap({
        N('g')  : call([V('u'), V('a')], [V('gu')], lambda u, a: a * jnp.cos(u)),
        N('fix'): driven(RelationalCondition((Relation(V('u'), V('gu'), Eq),), (), (V('u'),), ()),
                         driver, implementation, reports),
    })))


# `(u - a)^2 + 1`, least at `u = a`.
BOWL = CallableGraph(graph_of(OrderedMap({
    N('f')  : call([V('u'), V('a')], [V('f')], lambda u, a: (u - a) ** 2 + 1.0),
    N('opt'): driven(RelationalCondition((), (Objective(V('f'), Min),), (V('u'),), ()),
                     Minimisation(steps=True), BFGS(**TIGHT), [V('n')]),
})))


def jitted(graph, wanted):
    traces = []

    @eqx.filter_jit
    def go(graph, values):
        traces.append(1)
        out = run(GraphWithValues(graph, OrderedMap(values.items())))
        return tuple(out[w] for w in wanted)

    return (lambda values: go(graph, OrderedMap({k: jnp.asarray(v)
                                              for k, v in values.items()}))), traces


@pytest.mark.parametrize('driver, implementation', [
    (RootFinding(steps=True), Newton(**TIGHT)),
    (FixedPointIteration(steps=True), Picard(**TIGHT)),
], ids=['newton', 'picard'])
def test_a_fixed_point_runs_on_newton_and_picard_tracing_once(driver, implementation):
    '''A fixed point is equalities too, so a `RootFinding` answers it through its gaps;
    a `FixedPointIteration` steps to its right side.'''
    go, traces = jitted(cosine(driver, implementation, [V('n')]), (V('u'), V('n')))
    for a in (0.5, 0.25):
        u, n = go({V('a'): a, STARTS(V('u')): 0.5})
        assert float(u) == pytest.approx(a * math.cos(float(u)), abs=1e-10)
        assert int(n) > 1
    assert len(traces) == 1


@pytest.mark.slow
def test_a_bfgs_minimises_the_objective_tracing_once():
    go, traces = jitted(BOWL, (V('u'), V('f'), V('n')))
    for a in (2.0, -1.0):
        u, f, n = go({V('a'): a, STARTS(V('u')): 0.0})
        assert float(u) == pytest.approx(a, abs=1e-8)
        assert float(f) == pytest.approx(1.0, abs=1e-12)
        assert int(n) >= 1
    assert len(traces) == 1


def test_the_proof_refuses_a_solver_on_a_contract_it_does_not_implement():
    with pytest.raises(TypeError, match='implements a RootFinder'):
        CallableGraph(graph_of(OrderedMap({
            N('f')  : call([V('u'), V('a')], [V('f')], lambda u, a: (u - a) ** 2),
            N('opt'): driven(RelationalCondition((), (Objective(V('f'), Min),), (V('u'),), ()),
                             Minimisation(steps=False), Newton()),
        })))
    with pytest.raises(TypeError, match='implements a FixedPointIterator'):
        cosine(RootFinding(steps=False), Picard())
