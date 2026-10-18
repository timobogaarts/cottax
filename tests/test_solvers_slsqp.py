'''
`jax_solvers.SLSQP` answering a `ConstrainedMinimisation` through `execution`: the
constraints and their signs, bounds, what it reports, the Jacobian transform, the memo,
and running under a trace and without one. Float64, as `test_solvers.py`.
'''

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import DictKey

import equinox as eqx
from cottax.core import Function, GraphWithValues
from cottax.relational import Min, Objective
from cottax.core.frozen import OrderedMap
from cottax.core.driven import Driven
from cottax.core.driven import Start, Steps
from cottax.relational.contracts import ConstrainedMinimisation
from cottax.consumers.execution import CallableGraph, run
import purposes
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedDriven, ImplementedFunction
from cottax.relational import Eq, Le, Relation, RelationalCondition
from cottax.relational.consumers.execution.jax_solvers.canonical_nlp import Canonical, box, canonical
from cottax.relational.consumers.execution.jax_solvers import SLSQP, scipy_slsqp
from cottax.relational.consumers.execution.jax_solvers.scipy_slsqp import _Memo
from graphs import graph_of


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def V(name: str) -> VarPath:
    return VarPath((DictKey(name),))


def N(name: str) -> NodePath:
    return NodePath((DictKey(name),))


STARTS, LOWER, UPPER = purposes.Start(), purposes.Lower(), purposes.Upper()


def call(reads, owns, fn) -> ImplementedFunction:
    return ImplementedFunction(Function(tuple(V(r) for r in reads),
                                        tuple(V(o) for o in owns)), fn)


def counted(calls, name, fn):
    def wrapped(*a):
        calls[name] = calls.get(name, 0) + 1
        return fn(*a)
    return wrapped


def optimise(objective, design, relations, implementation, bounded=False, steps=False
             ) -> ImplementedDriven:
    '''One objective over the design, a start per unknown, and when asked a step count at
    `.n` and bounds `^lower.x <= x <= ^upper.x`: relations over parameters.'''
    unknowns   = tuple(V(d) for d in design)
    parameters = tuple(b(u) for u in unknowns for b in (LOWER, UPPER)) if bounded else ()
    bounds     = tuple(r for u in unknowns for r in (Relation(LOWER(u), u, Le),
                                                     Relation(u, UPPER(u), Le))) if bounded else ()
    condition  = RelationalCondition(tuple(relations) + bounds, (Objective(V(objective), Min),), unknowns,
                                     parameters)
    return ImplementedDriven(
        Driven(condition, ConstrainedMinimisation(steps=steps),
               (tuple(STARTS(u) for u in unknowns),), ((V('n'),),) if steps else ()),
        implementation)


def solve(nodes, values):
    graph = CallableGraph(graph_of(OrderedMap(nodes)))
    return run(GraphWithValues(graph, OrderedMap({k: jnp.asarray(v) if not isinstance(v, tuple)
                                                  else v for k, v in values.items()})))


# ---------------------------------------------------------------- the problem it answers

def test_an_equality_and_an_inequality_are_both_respected():
    '''`min x^2 + y^2` on `x + y = 2` (both sides computed) and `x >= 1.5`: the
    inequality binds, (1.5, 0.5).'''
    out = solve({
        N('m')  : call(['x', 'y'], ['f', 'h', 'two', 'g'],
                       lambda x, y: (x ** 2 + y ** 2, x + y, 2.0 + 0.0 * x, 1.5 - x)),
        N('opt'): optimise('f', ['x', 'y'], (Relation(V('h'), V('two'), Eq),
                                             Relation(V('g'), None, Le)), SLSQP()),
    }, {STARTS(V('x')): 3.0, STARTS(V('y')): -1.0})
    assert float(out[V('x')]) == pytest.approx(1.5, abs=1e-6)
    assert float(out[V('y')]) == pytest.approx(0.5, abs=1e-6)


def test_the_inequality_sign_is_cottaxs_not_scipys():
    '''`(x - 5)^2` under `x - 1 <= 0` is `x = 1`; the wrong negation converges to 5.'''
    out = solve({
        N('m')  : call(['x'], ['f', 'g'], lambda x: ((x - 5.0) ** 2, x - 1.0)),
        N('opt'): optimise('f', ['x'], (Relation(V('g'), None, Le),), SLSQP()),
    }, {STARTS(V('x')): 0.0})
    assert float(out[V('x')]) == pytest.approx(1.0, abs=1e-6)


def test_a_bound_is_broadcast_over_an_unknown_that_holds_an_array():
    out = solve({
        N('m')  : call(['x'], ['f'], lambda x: jnp.sum((x - 3.0) ** 2)),
        N('opt'): optimise('f', ['x'], (), SLSQP(), bounded=True),
    }, {STARTS(V('x')): jnp.zeros(4), LOWER(V('x')): -1.0, UPPER(V('x')): 0.5})
    assert out[V('x')].shape == (4,)
    assert out[V('x')].tolist() == pytest.approx([0.5] * 4, abs=1e-6)


BOWL = {N('m'): call(['x'], ['f'], lambda x: (x - 3.0) ** 2)}


def test_steps_are_reported_only_when_the_driver_asks_for_them():
    quiet = solve(BOWL | {N('opt'): optimise('f', ['x'], (), SLSQP())},
                  {STARTS(V('x')): 0.0})
    loud  = solve(BOWL | {N('opt'): optimise('f', ['x'], (), SLSQP(), steps=True)},
                  {STARTS(V('x')): 0.0})
    assert V('n') not in quiet
    assert int(loud[V('n')]) >= 1


def test_strict_refuses_a_solve_that_did_not_converge_and_lax_takes_it():
    steep = {N('m'): call(['x'], ['f'], lambda x: jnp.sum((x - 3.0) ** 4 + x ** 2))}
    with pytest.raises(RuntimeError, match='did not converge'):
        solve(steep | {N('opt'): optimise('f', ['x'], (), SLSQP(max_steps=1))},
              {STARTS(V('x')): jnp.full(3, 50.0)})
    out = solve(steep | {N('opt'): optimise('f', ['x'], (), SLSQP(max_steps=1,
                                                                  strict=False))},
                {STARTS(V('x')): jnp.full(3, 50.0)})
    assert out[V('x')].shape == (3,)


# ---------------------------------------------------------------- one call per point

def counting(calls, implementation):
    '''`(x - 3)^2` under `1 - x <= 0` and `x - 2 <= 0`, each condition its own node, so
    asking them one at a time would show as unequal counts.'''
    return {
        N('opt'): optimise('f', ['x'], (Relation(V('g1'), None, Le),
                                        Relation(V('g2'), None, Le)), implementation),
        N('m')  : call(['x'], ['f'],  counted(calls, 'm',  lambda x: (x - 3.0) ** 2)),
        N('c1') : call(['x'], ['g1'], counted(calls, 'c1', lambda x: 1.0 - x)),
        N('c2') : call(['x'], ['g2'], counted(calls, 'c2', lambda x: x - 2.0)),
    }


def test_one_model_call_per_program_not_one_per_condition_or_point():
    '''One trace for the values, one for the Jacobian, one walk after convergence -- and
    the same three from a start that takes a different number of steps.'''
    calls: dict = {}
    out = solve(counting(calls, SLSQP()), {STARTS(V('x')): 0.0})
    assert float(out[V('x')]) == pytest.approx(2.0, abs=1e-6)
    assert calls == {'m': 3, 'c1': 3, 'c2': 3}

    for x0 in (-5.0, 1.7):
        again: dict = {}
        solve(counting(again, SLSQP()), {STARTS(V('x')): x0})
        assert again == calls


def by_rows(fn):
    '''A consumer's own Jacobian: rows pulled back one at a time off one forward pass.'''
    def jacobian(x):
        primal, pull_back = jax.vjp(fn, x)
        return jnp.stack([pull_back(row)[0] for row in jnp.eye(primal.shape[0])])
    return jacobian


def test_the_jacobian_is_whichever_transform_the_implementation_holds():
    out = solve(counting({}, SLSQP(jacobian=by_rows)), {STARTS(V('x')): 0.0})
    assert float(out[V('x')]) == pytest.approx(2.0, abs=1e-6)


def test_a_limit_held_fixed_is_read_as_a_parameter():
    '''`(x - 5)^2` under `x <= g_max`, the limit a place the run is given: `x = g_max`.'''
    condition = RelationalCondition((Relation(V('g'), V('g_max'), Le),), (Objective(V('f'), Min),), (V('x'),),
                                    (V('g_max'),))
    out = solve({
        N('m')  : call(['x'], ['f', 'g'], lambda x: ((x - 5.0) ** 2, x)),
        N('opt'): ImplementedDriven(Driven(condition, ConstrainedMinimisation(steps=False),
                                           ((STARTS(V('x')),),), ()), SLSQP()),
    }, {STARTS(V('x')): 0.0, V('g_max'): 2.0})
    assert float(out[V('x')]) == pytest.approx(2.0, abs=1e-6)


def test_a_bound_is_the_boxs_and_every_other_relation_the_forms():
    '''`lo <= x`, `x <= 0`, `0 <= y`, and a tighter `lo2 <= x`: the box holds the tightest
    of each side; the limit `g <= g_max` and the equality stay in the form.'''
    x, y = V('x'), V('y')
    condition = RelationalCondition(
        (Relation(V('lo'), x, Le), Relation(x, None, Le), Relation(None, y, Le),
         Relation(V('lo2'), x, Le), Relation(V('g'), V('g_max'), Le), Relation(V('h'), None, Eq)),
        (Objective(V('f'), Min),), (x, y), (V('lo'), V('lo2'), V('g_max')))
    values = dict(zip(condition.reads, (7.0, 1.0, 3.0, -4.0, -1.0, 2.0)))  # f g h lo lo2 g_max
    at = tuple(values[p] for p in condition.reads)
    assert condition.reads == tuple(map(V, ('f', 'g', 'h', 'lo', 'lo2', 'g_max')))
    form = canonical(condition, (0.5, 0.5), at)
    assert form == Canonical((7.0,), (3.0,), (1.0 - 2.0,))
    lower, upper = box(condition, (0.5, 0.5), at)
    assert float(lower[0]) == -1.0 and lower[1] == 0.0
    assert upper == (0.0, None)


def test_the_memo_computes_each_kind_once_per_point_and_keeps_the_last_of_each():
    '''A value is the form's groups; a Jacobian comes as one matrix and is split into
    them by the values' sizes. Each kind keeps its own last point.'''
    calls: list = []
    memo = _Memo(lambda x: calls.append('v') or (Canonical(x, x[:0], 2 * x), jnp.zeros((2, 1))),
                 lambda x: calls.append('j') or jnp.stack([x, 2 * x]),
                 jnp.float64, jax.devices()[0])
    one, two = np.array([1.0]), np.array([2.0])
    memo.values_at(one), memo.values_at(one)
    jacobian = memo.jacobian_at(one)
    assert calls == ['v', 'j']
    assert [group.tolist() for group in jacobian] == [[[1.0]], [], [[2.0]]]
    memo.values_at(two), memo.jacobian_at(one)          # the Jacobian at one is still kept
    assert calls == ['v', 'j', 'v']
    memo.values_at(one), memo.jacobian_at(two)          # values moved on to two: both anew
    assert calls == ['v', 'j', 'v', 'v', 'j']


def test_a_line_search_asks_values_it_never_differentiates(monkeypatch):
    '''`e^x - 2x` from 5: scipy's line search visits points it asks a value alone at.'''
    counts = {'values': 0, 'jacobian': 0}

    def counting_program(kind, program):
        def counted(*a):
            counts[kind] += 1
            return program(*a)
        return counted

    for kind in counts:
        monkeypatch.setattr(scipy_slsqp, f'_{kind}',
                            counting_program(kind, getattr(scipy_slsqp, f'_{kind}')))
    out = solve({N('m')  : call(['x'], ['f'], lambda x: jnp.exp(x) - 2.0 * x),
                 N('opt'): optimise('f', ['x'], (), SLSQP())}, {STARTS(V('x')): 5.0})
    assert float(out[V('x')]) == pytest.approx(np.log(2.0), abs=1e-5)
    assert counts['values'] > counts['jacobian'] >= 1


# ---------------------------------------------------------------- traced and not

@pytest.mark.slow
def test_under_a_jit_it_runs_through_pure_callback_tracing_once():
    traces: list = []
    graph = CallableGraph(graph_of(OrderedMap({
        N('m')  : call(['x', 'a'], ['f', 'g'], lambda x, a: ((x - a) ** 2, x - 1.0)),
        N('opt'): optimise('f', ['x'], (Relation(V('g'), None, Le),), SLSQP(),
                           bounded=True, steps=True),
    })))

    @eqx.filter_jit
    def go(graph, values):
        traces.append(1)
        out = run(GraphWithValues(graph, values))
        return out[V('x')], out[V('n')]

    for a, want in ((5.0, 1.0), (-2.0, -0.5)):
        x, n = go(graph, OrderedMap({V('a'): jnp.asarray(a), STARTS(V('x')): jnp.asarray(0.0),
                                     LOWER(V('x')): jnp.asarray(-0.5),
                                     UPPER(V('x')): jnp.asarray(9.0)}))
        assert float(x) == pytest.approx(want, abs=1e-6)
        assert int(n) >= 1
    assert len(traces) == 1


def test_with_nothing_traced_the_host_is_called_directly(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError('pure_callback entered with nothing traced')

    monkeypatch.setattr(jax, 'pure_callback', refuse)
    out = solve(BOWL | {N('opt'): optimise('f', ['x'], (), SLSQP())}, {STARTS(V('x')): 0.0})
    assert float(out[V('x')]) == pytest.approx(3.0, abs=1e-6)


def test_an_eager_solve_compiles_once_per_node_not_once_per_solve():
    calls: dict = {}
    nodes = counting(calls, SLSQP())
    for x0 in (0.0, -5.0, 1.7):
        solve(nodes, {STARTS(V('x')): x0})
    assert calls['m'] == 2 + 3     # two traces, then one walk per solve


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='needs a GPU')
def test_under_a_jit_the_model_is_evaluated_on_the_gpu_and_not_the_callbacks_cpu(monkeypatch):
    '''Inside `pure_callback` a numpy point would be computed on the CPU; the memo puts it
    on the default device first.'''
    platforms: set = set()
    values = scipy_slsqp._values

    def recording(*a):
        out = values(*a)
        platforms.update(device.platform for group in out for device in group.devices())
        return out

    monkeypatch.setattr(scipy_slsqp, '_values', recording)
    graph = CallableGraph(graph_of(OrderedMap({
        N('m')  : call(['x', 'a'], ['f'], lambda x, a: jnp.sum((x - a) ** 2)),
        N('opt'): optimise('f', ['x'], (), SLSQP()),
    })))
    x = eqx.filter_jit(lambda g, v: run(GraphWithValues(g, v))[V('x')])(
        graph, OrderedMap({V('a'): jnp.full(4, 3.0), STARTS(V('x')): jnp.zeros(4)}))
    assert x.tolist() == pytest.approx([3.0] * 4, abs=1e-6)
    assert platforms == {'gpu'}


def test_the_answer_comes_back_in_the_callers_dtype():
    out = solve({N('m'): call(['x'], ['f'], lambda x: jnp.sum((x - 3.0) ** 2)), N('opt'): optimise('f', ['x'], (), SLSQP())},
                {STARTS(V('x')): jnp.zeros(2, jnp.float32)})
    assert out[V('x')].dtype == jnp.float32
