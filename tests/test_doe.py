'''
The design of experiments, a third language beside `relational` and `ode`: its condition node,
proved by core with nothing edited for it, the contract answering it, its run in jax,
and its declaration through the shipped surface.
'''

import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import GetAttrKey

import cottax.assemblies.jax as cx
from cottax.core import ExecutableGraph, FunctionalNode, GraphWithValues, violations
from cottax.core.frozen import OrderedMap
from cottax.core.driven import Assign, Driven, DrivenGraph, Start
from cottax.core.driven.nodes import NotAccepted
from cottax.core.realisation import implemented
from cottax.core.refusal_kinds import Constant, OwnReadInsideCycle, ReadInsideCycle
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.jax_jit import JitGraph
from cottax.doe import DoeCondition, Recorded, Varied
from cottax.doe.contracts import Sampling
from cottax.doe.consumers.execution import JaxDoe, PlansDiffer
from cottax.pytree.path import NodePath, VarPath
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.relational.contracts import RootFinding
from graphs import graph_of


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def V(key: str) -> VarPath:
    return VarPath((GetAttrKey(key),))

def N(key: str) -> NodePath:
    return NodePath((GetAttrKey(key),))


DOE = DoeCondition((Varied(V('x'), V('P')),), (Recorded(V('h'), V('H')),))
'''`x` set to each point of `P`, `h` recorded into `H`.'''


def model(**nodes):
    '''The DOE over `h = F(x)`, `nodes` binding more or replacing one by name.'''
    return graph_of({N('doe'): DOE, N('F'): FunctionalNode((V('x'),), (V('h'),)),
                     **{N(k): d for k, d in nodes.items()}})


def test_a_doe_owns_its_factors_inside_and_its_records_outside():
    assert (DOE.owns_inside, DOE.reads_inside, DOE.reads_outside, DOE.owns_outside) == (
        (V('x'),), (V('h'),), (V('P'),), (V('H'),))
    g = ExecutableGraph(model())
    assert g.enclosers == (N('doe'),) and g.cycle_of(N('doe')) == (N('doe'), N('F'))


@pytest.mark.parametrize('nodes, kind', [
    ({'doe': DoeCondition(DOE.factors, (*DOE.responses, Recorded(V('c'), V('C')))),
      'c': FunctionalNode((), (V('c'),))}, Constant),             # a response no factor moves
    ({'p': FunctionalNode((V('x'),), (V('P'),))}, ReadInsideCycle),       # a plan on the cycle
    ({'F': FunctionalNode((V('x'), V('H')), (V('h'),))}, OwnReadInsideCycle),  # a record on it
], ids=['constant response', 'plan on the cycle', 'record on the cycle'])
def test_core_refuses_a_doe_by_its_own_rules(nodes, kind):
    assert [type(v) for v in violations(ExecutableGraph, model(**nodes))] == [kind]


def test_sampling_answers_a_doe_and_nothing_else():
    DrivenGraph(Assign(N('doe'), Sampling(), (), ())(model()))
    root_find = RelationalCondition((Relation(V('h'), None, Eq),), (), (V('x'),), ())
    assert [type(v) for v in violations(Driven, root_find, Sampling(), (), ())] == [NotAccepted]


def test_a_relabelling_moves_every_place_of_a_doe():
    assert DOE.relabel({V(k): V(k + '2') for k in 'xPhH'}) == DoeCondition(
        (Varied(V('x2'), V('P2')),), (Recorded(V('h2'), V('H2')),))


# ================================================================ the run

PLAN = jnp.arange(1., 6.)
'''Five points.'''


def call(fn, reads, owns):
    '''A function node, run by `fn`.'''
    return implemented(FunctionalNode(tuple(map(V, reads)), tuple(map(V, owns))), fn)

def square(x):
    return x**2

def sampled(doe=DOE, batch_size=None):
    '''`doe` driven by `Sampling`, run by `JaxDoe`.'''
    return implemented(Driven(doe, Sampling(), (), ()), JaxDoe(batch_size))

def swept(nodes, within=(), **values):
    '''Every place after a run of `nodes`, `values` at its inputs by key.'''
    graph = CallableGraph(graph_of(nodes, within))
    return run(GraphWithValues(graph, OrderedMap({V(k): v for k, v in values.items()})))


@pytest.mark.parametrize('batch_size', [None, 2, 5])
def test_a_doe_records_each_response_at_every_point_of_its_plan(batch_size):
    '''One point at a time, two at a time and the one left, or all at once: alike.'''
    out = swept({N('doe'): sampled(batch_size=batch_size), N('F'): call(square, ['x'], ['h'])},
                P=PLAN)
    assert jnp.array_equal(out[V('H')], PLAN**2)


def test_a_problem_nested_in_a_doe_is_solved_at_every_point():
    '''`u^2 = x` by Newton inside the DOE: its answer recorded per point.'''
    pytest.importorskip('optimistix')
    from cottax.relational.consumers.execution.jax_solvers import Newton
    root = RelationalCondition((Relation(V('r'), None, Eq),), (), (V('u'),), ())
    out = swept({N('doe'): sampled(DoeCondition(DOE.factors, (Recorded(V('u'), V('U')),))),
                 N('q'):   implemented(Driven(root, RootFinding(steps=False), ((V('u0'),),), ()),
                                       Newton(rtol=1e-12, atol=1e-12)),
                 N('r'):   call(lambda x, u: u**2 - x, ['x', 'u'], ['r'])},
                within=((N('q'), N('doe')),), P=PLAN, u0=1.)
    assert jnp.allclose(out[V('U')], jnp.sqrt(PLAN))


def test_a_jitted_run_differentiates_the_records_by_the_plan():
    eqx = pytest.importorskip('equinox')
    graph = JitGraph(graph_of({N('doe'): sampled(), N('F'): call(square, ['x'], ['h'])}))
    total = eqx.filter_jit(
        lambda plan: run(GraphWithValues(graph, OrderedMap({V('P'): plan})))[V('H')].sum())
    assert jnp.allclose(jax.grad(total)(PLAN), 2 * PLAN)


def test_plans_unlike_in_length_are_refused():
    doe = DoeCondition((*DOE.factors, Varied(V('y'), V('Q'))), DOE.responses)
    with pytest.raises(ValueError) as refused:
        swept({N('doe'): sampled(doe), N('F'): call(lambda x, y: x * y, ['x', 'y'], ['h'])},
              P=PLAN, Q=PLAN[:3])
    assert isinstance(refused.value.violation, PlansDiffer)


@pytest.mark.xfail(strict=True, reason="pending a check of the condition kind: a model off a "
                                       "DOE's cycle reading its interior reads what one point left")
def test_a_model_off_the_cycle_reading_a_doe_interior_is_refused():
    assert list(violations(ExecutableGraph, model(G=FunctionalNode((V('h'),), (V('g'),)))))


# ================================================================ the surface

s = cx.flat

@s.function('y_1')
def D1(x_1, z_1, z_2, y_2): return z_1**2 + z_2 + x_1 - 0.2 * y_2

@s.function('y_2')
def D2(y_1, z_1, z_2): return jnp.sqrt(y_1) + z_1 + z_2

@s.function('f')
def obj(x_1, z_2, y_1, y_2): return x_1**2 + z_2 + y_1 + jnp.exp(-y_2)


def test_sample_declares_a_doe_its_plans_given_or_named_its_records_minted_or_named():
    p = s.place
    assert s.sample(over=('x',), record=('h',)).condition == DoeCondition(
        (Varied(p('x'), p(cx.points('x'))),), (Recorded(p('h'), p(cx.record('h'))),))
    named = s.sample(over={'x': PLAN, 'y': 'ys'}, record={'h': 'hs'})
    assert named.condition == DoeCondition(
        (Varied(p('x'), p(cx.points('x'))), Varied(p('y'), p('ys'))), (Recorded(p('h'), p('hs')),))
    assert tuple(named.given) == (p(cx.points('x')),)
    assert jnp.array_equal(named.given[p(cx.points('x'))], PLAN)


def test_sellars_analysis_under_a_doe_records_what_a_run_per_point_gives():
    pytest.importorskip('optimistix')
    analysis, newton = s.fixed_point_cut('y_2', 'D1'), cx.Newton(rtol=1e-10, atol=1e-10)
    under_doe = (s.declare(D1, D2, obj, {'sweep': s.sample(over={'z_1': PLAN}, record=('f',))})
                 + analysis + s.nest(cx.closure('y_2'), 'sweep') + s.drive(cx.JaxDoe(), newton))
    alone = s.declare(D1, D2, obj) + analysis + s.drive(newton)
    inputs, start = {'z_2': 2., 'x_1': 1.}, {Start: {cx.copy('y_2'): 1.}}
    each = [s.run(alone, {cx.Inputs: {**inputs, 'z_1': z}, **start})['f'] for z in PLAN]
    out = s.run(under_doe, {cx.Inputs: inputs, **start})
    assert jnp.allclose(out[cx.record('f')], jnp.array(each))
