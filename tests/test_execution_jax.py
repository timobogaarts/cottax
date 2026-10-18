'''
`consumers/execution/` under jax: `solvers.Newton` answering a `RootFinding`, and the run
jitted.

The map is `fn(y, args)` as it stands: the unknowns are optimistix's `y`, the context
its `args`. A fixed point on a root finder is `test_solvers.py`'s.

Run in float64 (`jax.enable_x64`, scoped to each test here), as a solve to a tight
tolerance needs. What is pinned: the graph, its nodes and a condition map are static (hashable, compared
by value), the inputs and the context are traced, so one trace serves every value.
'''

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import DictKey

from cottax.core import Function, GraphWithValues, Solve
from cottax.core.frozen import OrderedMap
from cottax.core.driven import Driven
from cottax.relational.contracts import RootFinding
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.run import InteriorConditionMap
import purposes
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedDriven, ImplementedFunction
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.relational.consumers.execution.jax_solvers import Newton
from graphs import graph_of


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def V(name: str) -> VarPath:
    return VarPath((DictKey(name),))


def N(name: str) -> NodePath:
    return NodePath((DictKey(name),))


# ---------------------------------------------------------------- the contract and its implementation

newton = Newton(rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------- the models

def call(reads, owns, fn) -> ImplementedFunction:
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), fn)


def driven(relations, unknowns) -> ImplementedDriven:
    condition = RelationalCondition(tuple(relations), (), tuple(unknowns), ())
    return ImplementedDriven(Driven(condition, RootFinding(steps=False),
                                    tuple((STARTS(u),) for u in unknowns), ()), newton)


STARTS = purposes.Start()


def start(u: str) -> VarPath:
    return STARTS(V(u))


# `u * u == z`, behind a short chain: a root find against zero.
SQUARE = CallableGraph(graph_of(OrderedMap({
    N('mid'): call([V('u')], [V('w')], lambda u: u * 1.0),
    N('res'): call([V('w'), V('z')], [V('r')], lambda w, z: w * w - z),
    N('h')  : driven([Relation(V('r'), None, Eq)], [V('u')]),
})))

def exact_u(x):
    k = 1.0 + x
    return (k + math.sqrt(k * k + 4.0 * k)) / 2.0


# `test_execution`'s model: `u * u = (1 + x) * (1 + u)` nested under the `x` making
# `f = u + y` hit `t`, one outer iteration a whole inner solve.
NESTED = CallableGraph(graph_of(OrderedMap({
    N('Mesh')     : call([V('x')], [V('k')], lambda x: 1.0 + x),
    N('DiscA')    : call([V('u'), V('k'), V('y')], [V('r_u')], lambda u, k, y: u * u - k * y),
    N('close')    : driven([Relation(V('r_u'), None, Eq)], [V('u')]),
    N('DiscB')    : call([V('u')], [V('y')], lambda u: 1.0 + u),
    N('Objective'): call([V('u'), V('y')], [V('f')], lambda u, y: u + y),
    N('Aim')      : call([V('f'), V('t')], [V('g')], lambda f, t: f - t),
    N('target')   : driven([Relation(V('g'), None, Eq)], [V('x')]),
}), OrderedMap({N('close'): N('target')})))


def jitted(graph, wanted):
    '''A run under `filter_jit`, counting its traces. The graph is a static leaf; the
    inputs cross as an `OrderedMap` and become the run's `GraphWithValues` inside the trace.'''
    traces = []

    @eqx.filter_jit
    def go(graph, values):
        traces.append(1)
        return run(GraphWithValues(graph, OrderedMap(values.items())))[wanted]

    return (lambda values: go(graph, OrderedMap({k: jnp.asarray(v)
                                              for k, v in values.items()}))), traces


# ---------------------------------------------------------------- the run

def test_a_root_find_runs_eagerly():
    out = run(GraphWithValues(SQUARE, OrderedMap({V('z'): jnp.asarray(9.0),
                                                  start('u'): jnp.asarray(1.0)})))
    assert float(out[V('u')]) == pytest.approx(3.0, rel=1e-10)
    assert float(out[V('r')]) == pytest.approx(0.0, abs=1e-10)


def test_a_whole_run_traces_once_and_compiles():
    go, traces = jitted(SQUARE, V('u'))
    got = [float(go({V('z'): z, start('u'): 1.0})) for z in (4.0, 9.0, 16.0)]
    assert got == pytest.approx([2.0, 3.0, 4.0], rel=1e-10)
    assert len(traces) == 1


@pytest.mark.slow
def test_a_nested_solve_differentiates_through_the_inner_one_and_traces_once():
    '''The outer Newton's Jacobian runs through the inner `root_find` by optimistix's
    implicit adjoint. The inner start is past `u = k / 2`, where its residual is flat.'''
    go, traces = jitted(NESTED, V('x'))
    for t in (5.0, 7.0):
        x = float(go({V('t'): t, start('x'): 0.5, start('u'): 3.0}))
        u = exact_u(x)
        assert u + (1.0 + u) == pytest.approx(t, rel=1e-10)
    assert len(traces) == 1


# ---------------------------------------------------------------- the condition map

def condition_map(graph, problem):
    '''The map `run` hands the driver of `problem`, outermost in `graph`.'''
    (step,) = [s for s in graph.steps if isinstance(s, Solve) and s.problem == problem]
    return InteriorConditionMap.posing(graph[problem], step.interior)


def test_the_condition_map_is_differentiable_in_the_unknowns():
    cm = condition_map(SQUARE, N('h'))
    assert cm.context_places == (V('z'),)
    ((d,),) = jax.jacfwd(cm, argnums=0)((jnp.asarray(3.0),), (jnp.asarray(9.0),))
    assert float(d) == pytest.approx(6.0)          # d(u^2 - z)/du at u = 3


def test_a_condition_map_is_a_static_key_and_its_context_is_traced():
    '''Two maps built apart compare and hash alike, so a jit keyed on one serves the
    other, and every context value, with one trace.'''
    traces = []

    @eqx.filter_jit
    def at(cm, unknowns, context):
        traces.append(1)
        return cm(unknowns, context)

    first, second = condition_map(SQUARE, N('h')), condition_map(SQUARE, N('h'))
    assert first is not second and first == second and hash(first) == hash(second)
    for cm, z in ((first, 4.0), (second, 9.0)):
        (r,) = at(cm, (jnp.asarray(2.0),), (jnp.asarray(z),))
        assert float(r) == pytest.approx(4.0 - z)
    assert len(traces) == 1
