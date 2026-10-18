'''
The ODE condition, a second language beside `relational`, through every consumer with
nothing edited for it but its own answers. Declared through a surface joining three
vocabularies, as a study with its own join writes it; driven from both features'
`CONTRACTS`; run, differentiated and jitted with `diffrax`; nested under a relational
optimiser; written to JSON and read back; drawn; and generic in the place, run on plain
floats over a toy `Word` place.
'''

import math
import re

import jax
import jax.numpy as jnp
import pytest

import cottax.assemblies.jax as cx

from cottax.core import (
    AbstractGraph, Combine, Determine, ExecutableGraph, Function, GaussSeidel, GraphWithValues,
    NestingGraph, Rename, Undetermine,
)
from cottax.core.maths.definition import OwnsTwice, ReadsOwn
from cottax.core.maths.executable import SeveralOutermost
from cottax.core.maths.plan.ops import AlreadyDetermined
from cottax.core.frozen import OrderedMap, choice, violations
from cottax.core.maths.plan import Nest, Plan, refusals
from cottax.core.driven import Assign, DrivenGraph, Start
from cottax.core.realisation import Implement, ImplementedFunction
from cottax.consumers.declaration import declare
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.interface import DriverImplementation
from cottax.consumers.json import read_json, write_json
from cottax.consumers.visualization import render_html, render_matrix_svg, render_xdsm_svg, to_dot
from cottax.consumers.visualization.xdsm import xdsm_struct
from cottax.pytree.consumers.surface import node_of as node
from cottax.pytree.path import VarPath
from cottax.relational import Eq, FixedPointCut, Relation, RelationalCondition
from cottax.consumers.architectures import IDF, MDA, MDF, SAND
from cottax.consumers.architectures.refusals import NotCombinable
from cottax.ode import Dt, OdeCondition, is_ode
from cottax.ode.contracts import Integration, Integrator
from cottax.ode.ode_condition import DeterminesItsStates, SpansDiffer
from cottax.ode.consumers.visualization import INTEGRATE
import cottax.ode.consumers  # noqa: F401  registers its answers
from test_core import Word


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def diffrax_run(**settings):
    '''`Diffrax` at a tight tolerance: the tests needing it skip where it is not installed.'''
    pytest.importorskip('diffrax')
    from cottax.ode.consumers.execution import Diffrax
    return Diffrax(**{'rtol': 1e-10, 'atol': 1e-10, **settings})


# ================================================================ the shipped surface

s = cx.flat
'''The shipped surface over a plain dict, relational's and the ODE's parts among its own.'''


drive = s.drive


def from_state(state) -> dict:
    '''Each input and each start read where `state` holds it.'''
    return {cx.Inputs: state, Start: state}
CLOSING = cx.flat.closing
def solve_at(at):
    '''Where the ODE beside `at` binds: `^solve.at`.'''
    return cx.absolute.node(cx.solve(at))


def close(var: str, reader: str) -> FixedPointCut:
    '''`reader` reads `^copy.var`, closed by a problem at `^solve.var`.'''
    return s.fixed_point_cut(var, reader)


def architecture(final, top='opt'):
    return final(GaussSeidel(), CLOSING, node(top))


def span_of(x: str) -> VarPath:
    '''Where the span of the ODE whose first state is `x` sits by default: `^span.x`.'''
    return s.place(cx.span(x))


def ivp(span, **initial) -> dict:
    '''The initial values and the span at their default places, as inputs by name.'''
    first = next(iter(initial))
    return {s.place(cx.initial(x)): v for x, v in initial.items()} | {span_of(first): span}


@s.rate('x')
def decay(x, k):
    return -k * x


@s.rate('x', 'v')
def oscillator(x, v, w):
    return v, -w * w * x


def driven(*models, implementation) -> CallableGraph:
    return CallableGraph((Plan(declare(*models).graph) + drive(implementation)).graph)


def final(graph, values, at='x'):
    return run(GraphWithValues(graph, OrderedMap(values)))[s.place(at)]


# ================================================================ the node

def test_the_rate_declares_a_body_and_beside_it_the_ode():
    g = declare(decay).graph
    assert g.nodes == (node('decay'), solve_at('decay'))
    ode = g[solve_at('decay')]
    assert ode == OdeCondition((Dt(s.place('x'), s.place(cx.rate('x')), s.place(cx.initial('x'))),),
                               span_of('x'))
    assert ode.conditions == (s.place(cx.rate('x')),)
    assert ode.reads == (*ode.conditions, *ode.parameters) == (
        s.place(cx.rate('x')), s.place(cx.initial('x')), span_of('x'))
    assert ode.owns == ode.unknowns == (s.place('x'),)
    assert is_ode(ode) and ExecutableGraph(g).graph == g


def test_a_state_twice_or_a_state_that_is_a_rate_is_refused_by_core():
    '''It owns its states and reads its rates, so core's rules of a node say both.'''
    x, y, r, x0, t = Word('x'), Word('y'), Word('r'), Word('x0'), Word('t')
    assert [type(v) for v in violations(OdeCondition, (Dt(x, r, x0), Dt(x, y, x0)), t)] == [OwnsTwice]
    assert [type(v) for v in violations(OdeCondition, (Dt(x, r, x0), Dt(r, x, x0)), t)] == [
        ReadsOwn, ReadsOwn]
    assert [type(v) for v in violations(OdeCondition, (Dt(x, r, x),), t)] == [ReadsOwn]


def test_an_ode_determines_its_states_and_nothing_else():
    '''Handed its states it is itself; `Undetermine` and a subset are refused by the node,
    and `Determine` before it, since an ODE is always a problem.'''
    x, v, r, q = Word('x'), Word('v'), Word('r'), Word('q')
    ode = OdeCondition((Dt(x, r, Word('x0')), Dt(v, q, Word('v0'))), Word('t'))
    assert ode.determined((x, v)) is ode
    for unknowns in ((), (x,), (v, x)):
        with pytest.raises(ValueError, match='an ODE determines its states and nothing else'):
            ode.determined(unknowns)
    g = NestingGraph(AbstractGraph(OrderedMap({
        Word('body'): Function((x, v), (r, q)), Word('ode'): ode})), OrderedMap())
    refused = DeterminesItsStates((x, v), ())
    with pytest.raises(ValueError, match=refused.message.replace('(', r'\(').replace(')', r'\)')):
        Undetermine(Word('ode')).apply(g)
    assert [type(v) for v in refusals(Determine(Word('ode'), (x,)), g)] == [AlreadyDetermined]


# ================================================================ generic in the place

@choice(DriverImplementation)
class Euler(DriverImplementation[Integrator]):
    '''Forward Euler in plain Python: what runs an integrator needs no jax.'''
    needs    = frozenset()
    provides = frozenset()
    steps    : int

    def __call__(self, posed, context, data):
        ode = posed.condition
        *x, (t0, t1) = posed.fixed((*ode.initials, ode.span), context)
        h = (t1 - t0) / self.steps
        for _ in range(self.steps):
            x = tuple(xi + h * ri for xi, ri in zip(x, posed.conditions(x, context)))
        return x, {}



def test_over_a_toy_place_an_ode_is_proved_rewritten_driven_and_run_on_floats():
    '''Asked of a place only what core asks: `Word` is segments and a `str`.'''
    w = Word
    g = NestingGraph(AbstractGraph(OrderedMap({
        w('body'): ImplementedFunction(Function((w('x'), w('k')), (w('r'),)), lambda x, k: -k * x),
        w('ode') : OdeCondition((Dt(w('x'), w('r'), w('x0')),), w('span')),
    })), OrderedMap())
    assert ExecutableGraph(g).graph == g
    moved = Rename(w('ode'), ((w('r'), w('q')),)).apply(
        Rename(w('body'), ((w('r'), w('q')),)).apply(g))
    assert moved[w('ode')].rates == (w('q'),)
    solved = (Plan(g) + Assign(w('ode'), Integration(trajectory=False), (), ())
              + Implement(OrderedMap({w('ode'): Euler(10_000)}))).graph
    assert DrivenGraph(solved).graph == solved
    out = run(GraphWithValues(CallableGraph(solved),
                              OrderedMap({w('k'): 1.0, w('x0'): 1.0, w('span'): (0.0, 1.0)})))
    assert out[w('x')] == pytest.approx(math.exp(-1.0), rel=1e-4)


# ================================================================ run with diffrax

def test_decay_runs_on_a_dict_its_initial_value_given_and_its_final_one_written_back():
    '''A state's name means its final value: its slot is written, never read as the
    initial one.'''
    graph = driven(decay, implementation=diffrax_run())
    state = {'x': None, 'k': 0.5}
    out = s.run(graph, {cx.Inputs: ivp((0.0, 2.0), x=1.0)}, read_from={cx.Inputs: state})
    assert out.into(state)['x'] == pytest.approx(math.exp(-1.0), rel=1e-8)
    assert out[s.place(cx.rate('x'))] == pytest.approx(-0.5 * math.exp(-1.0), rel=1e-8)


def test_two_states_a_harmonic_oscillator_the_state_a_tuple():
    graph = driven(oscillator, implementation=diffrax_run())
    out = run(GraphWithValues(graph, OrderedMap({s.place('w'): 2.0, **ivp((0.0, 1.5), x=1.0, v=0.0)})))
    assert out[s.place('x')] == pytest.approx(math.cos(3.0), abs=1e-8)
    assert out[s.place('v')] == pytest.approx(-2.0 * math.sin(3.0), abs=1e-8)


@s.rate('z')
def spin(z, w):
    '''A state that is a small pytree of arrays: a phase per oscillator, and its amplitude.'''
    return {'phase': w, 'amplitude': -0.1 * z['amplitude']}


def test_a_state_that_is_an_array_or_a_small_pytree():
    graph = driven(decay, implementation=diffrax_run())
    k = jnp.array([0.5, 1.0, 2.0])
    x = final(graph, {s.place('k'): k, **ivp((0.0, 1.0), x=jnp.ones(3))})
    assert jnp.allclose(x, jnp.exp(-k), rtol=1e-8)
    graph = driven(spin, implementation=diffrax_run())
    z = final(graph, {s.place('w'): jnp.array([1.0, 3.0]),
                      **ivp((0.0, 2.0), z={'phase': jnp.zeros(2), 'amplitude': 1.0})}, 'z')
    assert jnp.allclose(z['phase'], jnp.array([2.0, 6.0]))
    assert z['amplitude'] == pytest.approx(math.exp(-0.2), rel=1e-8)


# ---------------------------------------------------------------- derivatives

def decayed(graph):
    return lambda k: final(graph, {s.place('k'): k, **ivp((0.0, 2.0), x=1.0)})


def test_the_final_state_differentiates_in_reverse_by_the_default_adjoint():
    '''`RecursiveCheckpointAdjoint`, diffrax's default: reverse mode only.'''
    dx = jax.grad(decayed(driven(decay, implementation=diffrax_run())))(0.5)
    assert dx == pytest.approx(-2.0 * math.exp(-1.0), rel=1e-8)


def test_the_final_state_differentiates_forward_by_forward_mode():
    '''`diffrax.ForwardMode()`: forward mode only, the default refusing a `jvp`.'''
    diffrax = pytest.importorskip('diffrax')
    forward = decayed(driven(decay, implementation=diffrax_run(adjoint=diffrax.ForwardMode())))
    assert jax.jacfwd(forward)(0.5) == pytest.approx(-2.0 * math.exp(-1.0), rel=1e-8)
    with pytest.raises(TypeError, match='custom_vjp'):
        jax.jacfwd(decayed(driven(decay, implementation=diffrax_run())))(0.5)


@pytest.mark.slow
def test_a_jitted_run_traces_once_and_a_new_k_compiles_nothing():
    eqx = pytest.importorskip('equinox')
    graph, traces = driven(decay, implementation=diffrax_run()), []

    @eqx.filter_jit
    def go(graph, values):
        traces.append(1)
        return run(GraphWithValues(graph, OrderedMap(values.items())))[s.place('x')]

    for k in (0.5, 1.0, 2.0):
        x = go(graph, OrderedMap({s.place('k'): jnp.asarray(k), **ivp(
            (jnp.asarray(0.0), jnp.asarray(2.0)), x=jnp.asarray(1.0))}))
        assert x == pytest.approx(math.exp(-2.0 * k), rel=1e-8)
    assert len(traces) == 1


# ---------------------------------------------------------------- a cycle with a function

@s.function('g')
def squared(x):
    return x * x


@s.rate('x')
def logistic(x, g, k):
    '''`x' = k (x - x^2)`, its square computed by a function on the ODE's cycle.'''
    return k * (x - g)


@s.function('y')
def doubled(x):
    return 2.0 * x


def test_an_ode_whose_rate_reads_a_function_of_its_state_and_feeds_another():
    graph = driven(squared, logistic, doubled, implementation=diffrax_run())
    (solve,) = [step for step in graph.steps if hasattr(step, 'interior')]
    assert set(solve.interior.nodes) == {node('squared'), node('logistic')}
    out = run(GraphWithValues(graph, OrderedMap({s.place('k'): 1.0, **ivp((0.0, 3.0), x=0.1)})))
    x = 1.0 / (1.0 + 9.0 * math.exp(-3.0))
    assert out[s.place('x')] == pytest.approx(x, rel=1e-8)
    assert out[s.place('y')] == pytest.approx(2.0 * x, rel=1e-8)


# ================================================================ two ODEs joined

@s.rate('x', span='span')
def position(v):
    return v


@s.rate('v', span='span')
def velocity(x, w):
    return -w * w * x


def test_two_odes_on_one_cycle_are_refused_until_combined_and_then_integrate_as_one():
    '''The oscillator written as two rates over one span: two problems on one cycle, then
    one by `+`.'''
    g = declare(position, velocity).graph
    at, other = solve_at('position'), solve_at('velocity')
    assert [type(v) for v in violations(ExecutableGraph, g)] == [SeveralOutermost]
    plan = Plan(g) + Combine(at, (at, other), OdeCondition)
    assert plan.graph[at] == OdeCondition((
        Dt(s.place('x'), s.place(cx.rate('x')), s.place(cx.initial('x'))),
        Dt(s.place('v'), s.place(cx.rate('v')), s.place(cx.initial('v')))), s.place('span'))
    graph = CallableGraph((plan + drive(diffrax_run())).graph)
    values = {s.place('w'): 2.0, s.place(cx.initial('x')): 1.0, s.place(cx.initial('v')): 0.0}
    out = run(GraphWithValues(graph, OrderedMap({**values, s.place('span'): (0.0, 1.5)})))
    one = run(GraphWithValues(driven(oscillator, implementation=diffrax_run()),
                              OrderedMap({**values, span_of('x'): (0.0, 1.5)})))
    assert out[s.place('x')] == pytest.approx(math.cos(3.0), abs=1e-8)
    assert out[s.place('v')] == pytest.approx(-2.0 * math.sin(3.0), abs=1e-8)
    assert (out[s.place('x')], out[s.place('v')]) == pytest.approx(
        (one[s.place('x')], one[s.place('v')]), abs=1e-12)


def test_an_ode_joins_an_ode_and_nothing_else():
    x, v, r, q, t = Word('x'), Word('v'), Word('r'), Word('q'), Word('t')
    a, b = OdeCondition((Dt(x, r, Word('x0')),), t), OdeCondition((Dt(v, q, Word('v0')),), t)
    assert (a + b).relations == a.relations + b.relations and (b + a).states == (v, x)
    assert (a + b).span == t
    assert [type(e) for e in violations(OdeCondition, a.relations * 2, t)] == [OwnsTwice]
    with pytest.raises(ValueError, match=re.escape(SpansDiffer(t, Word('s')).message)):
        a + OdeCondition(b.relations, Word('s'))
    with pytest.raises(ValueError, match='duplicate output'):    # a state determined twice
        a + a
    root = RelationalCondition((Relation(r, None, Eq),), (), (v,), ())
    for left, right in ((a, root), (root, a)):
        with pytest.raises(TypeError):
            left + right


# ================================================================ an algebraic constraint nested

@s.rate('x')
def relaxing(z):
    return -z


@s.residual('z')
def algebraic(z, x, c):
    return z - c * x


def index_one(c):
    '''`x' = -z`, `0 = z - c x`: the residual's problem nested in the ODE's, so every rate
    is read with `z` solved first.'''
    from cottax.relational.consumers.execution.jax_solvers import Newton
    ode, inner = solve_at('relaxing'), solve_at('algebraic')
    plan = Plan(declare(relaxing, algebraic).graph) + Nest(inner, ode)
    graph = CallableGraph((plan + drive(Newton(rtol=1e-12, atol=1e-12), diffrax_run())).graph)
    return run(GraphWithValues(graph, OrderedMap({
        s.place('c'): c, **ivp((0.0, 0.8), x=1.5),
        s.place(cx.start('z')): 0.0})))


def test_an_algebraic_constraint_nested_in_an_ode_is_solved_at_every_rate():
    g = declare(relaxing, algebraic).graph
    assert [type(v) for v in violations(ExecutableGraph, g)] == [SeveralOutermost]
    out = index_one(2.0)
    assert out[s.place('x')] == pytest.approx(1.5 * math.exp(-1.6), rel=1e-8)
    assert out[s.place('z')] == pytest.approx(2.0 * 1.5 * math.exp(-1.6), rel=1e-8)


def test_the_final_state_of_the_index_one_system_differentiates_through_both_solves():
    dx = jax.grad(lambda c: index_one(c)[s.place('x')])(2.0)
    assert dx == pytest.approx(-0.8 * 1.5 * math.exp(-1.6), rel=1e-6)


# ================================================================ under an optimiser

@s.function('f')
def miss(x, target):
    return (x - target) ** 2


def fitted():
    return declare(decay, miss, {'opt': s.optimise(design=('k',), minimise='f')})


@pytest.mark.slow
def test_an_ode_nested_under_an_optimiser_fits_its_rate():
    from cottax.relational.consumers.execution.jax_solvers import BFGS
    plan = (Plan(fitted().graph) + Nest(solve_at('decay'), node('opt'))
            + drive(BFGS(rtol=1e-10, atol=1e-10), diffrax_run()))
    tree = s.run(CallableGraph(plan.graph), {cx.Inputs: ivp((0.0, 1.0), x=1.0)},
                 read_from=from_state({'k': 0.1, 'target': math.exp(-0.7)}))
    assert tree['k'] == pytest.approx(0.7, rel=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize('final', [MDF, IDF])
def test_mdf_and_idf_nest_the_ode_as_any_implicit_model_and_fit_its_rate(final):
    '''Each nests what the model declares, so the ODE is placed as the hand did it.'''
    from cottax.relational.consumers.execution.jax_solvers import BFGS
    plan = Plan(fitted().graph) + architecture(final)
    assert dict(plan.graph.within) == {solve_at('decay'): node('opt')}
    ExecutableGraph(plan.graph)
    plan += drive(BFGS(rtol=1e-10, atol=1e-10), diffrax_run())
    tree = s.run(CallableGraph(plan.graph), {cx.Inputs: ivp((0.0, 1.0), x=1.0)},
                 read_from=from_state({'k': 0.1, 'target': math.exp(-0.7)}))
    assert tree['k'] == pytest.approx(0.7, rel=1e-6)


def test_sand_refuses_the_ode_it_would_combine_into_the_top():
    '''An ODE combines into nothing an architecture folds, so SAND refuses it before cutting.'''
    assert [type(v) for v in refusals(architecture(SAND), fitted().graph)] == [NotCombinable]
    with pytest.raises(ValueError, match=r"\^solve\['decay'\] combines as nothing, "
                                         r"and sand\(.*\) combines it into \['opt'\]"):
        Plan(fitted().graph) + architecture(SAND)


def fitted_index_one():
    return declare(relaxing, algebraic, miss, {'opt': s.optimise(design=('c',), minimise='f')})


def fit_index_one(plan) -> float:
    '''`c` fitted so that `x(0.8) = 1.5 e^(-1.6)`, from `c = 0.5`: `2`.'''
    from cottax.relational.consumers.execution.jax_solvers import BFGS, Newton
    plan += drive(BFGS(rtol=1e-10, atol=1e-10), Newton(rtol=1e-12, atol=1e-12), diffrax_run())
    return s.run(CallableGraph(plan.graph), {cx.Inputs: ivp((0.0, 0.8), x=1.5)},
                 read_from=from_state({'z': 0.0, 'c': 0.5, 'target': 1.5 * math.exp(-1.6)}))['c']


@pytest.mark.slow
@pytest.mark.parametrize('final', [MDF, IDF])
def test_mdf_and_idf_nest_the_algebraic_constraint_in_the_ode_and_the_ode_in_the_top(final):
    '''The residual lies wholly inside the ODE's body, so it is nested there, and the ODE is
    placed as any implicit model: the fit is the hand-nested one's.'''
    ode, inner = solve_at('relaxing'), solve_at('algebraic')
    plan = Plan(fitted_index_one().graph) + architecture(final)
    assert dict(plan.graph.within) == {inner: ode, ode: node('opt')}
    ExecutableGraph(plan.graph)
    by_hand = Plan(fitted_index_one().graph) + Nest(inner, ode) + Nest(ode, node('opt'))
    c = fit_index_one(plan)
    assert c == pytest.approx(2.0, rel=1e-6)
    assert c == pytest.approx(fit_index_one(by_hand), rel=1e-10)


def test_sand_refuses_the_index_one_ode_as_any():
    assert [type(v) for v in refusals(architecture(SAND), fitted_index_one().graph)] == [NotCombinable]


def test_an_mda_nests_the_algebraic_constraint_in_the_ode():
    from cottax.relational.consumers.execution.jax_solvers import Newton
    plan = Plan(declare(relaxing, algebraic).graph) + MDA(GaussSeidel(), CLOSING)
    assert dict(plan.graph.within) == {solve_at('algebraic'): solve_at('relaxing')}
    graph = CallableGraph((plan + drive(Newton(rtol=1e-12, atol=1e-12), diffrax_run())).graph)
    out = run(GraphWithValues(graph, OrderedMap({
        s.place('c'): 2.0, **ivp((0.0, 0.8), x=1.5),
        s.place(cx.start('z')): 0.0})))
    assert out[s.place('x')] == pytest.approx(index_one(2.0)[s.place('x')], rel=1e-12)


def logistic_at(t: float) -> float:
    '''`x' = x - x^2` from `0.1`, at `t`.'''
    return 1.0 / (1.0 + 9.0 * math.exp(-t))


def test_an_mda_cuts_nothing_inside_the_odes_loop_and_integrates_the_logistic():
    '''A closed cut through the rate loop would hold `squared` at the final state, so the
    ODE's whole cycle is one box: the analysis cuts nothing on it, and the run is the
    logistic.'''
    g = declare(squared, logistic, doubled).graph
    plan = Plan(g) + MDA(GaussSeidel(), CLOSING)
    assert plan.graph == g
    graph = CallableGraph((plan + drive(diffrax_run())).graph)
    x = final(graph, {s.place('k'): 1.0, **ivp((0.0, 0.5), x=0.1)})
    assert x == pytest.approx(logistic_at(0.5), rel=1e-8)


@pytest.mark.slow
def test_mdf_cuts_nothing_inside_the_odes_loop_and_fits_the_logistic_rate():
    '''Under an optimiser the ODE's cycle is a box too: nested in the top, uncut, so the
    rate fitted to the logistic's end value is the logistic's.'''
    from cottax.relational.consumers.execution.jax_solvers import BFGS
    g = declare(squared, logistic, doubled, miss,
                {'opt': s.optimise(design=('k',), minimise='f')}).graph
    plan = Plan(g) + architecture(MDF)
    assert dict(plan.graph.within) == {solve_at('logistic'): node('opt')}
    assert plan.graph.problems == g.problems
    plan += drive(BFGS(rtol=1e-10, atol=1e-10), diffrax_run())
    tree = s.run(CallableGraph(plan.graph), {cx.Inputs: ivp((0.0, 1.0), x=0.1)},
                 read_from=from_state({'k': 0.5, 'target': logistic_at(1.0)}))
    assert tree['k'] == pytest.approx(1.0, rel=1e-5)


def test_a_closure_written_by_hand_through_the_loop_is_a_model_of_its_own():
    '''Shooting, written on purpose: cut by hand, `squared` reads the final state, so the
    closure solves `x = c + (x0 - c) e^(kT)` with `c = x^2`, not the logistic. Built, and
    never refused.'''
    from cottax.relational.consumers.execution.jax_solvers import Picard
    closure = close('x', 'squared')
    plan = (Plan(declare(squared, logistic, doubled).graph) + closure
            + Nest(solve_at('logistic'), closure.at))
    graph = CallableGraph((plan + drive(Picard(rtol=1e-12, atol=1e-12), diffrax_run())).graph)
    x = final(graph, {s.place('k'): 1.0, **ivp((0.0, 0.5), x=0.1),
                      s.place(cx.start(cx.copy('x'))): 0.1})
    assert x == pytest.approx(x * x + (0.1 - x * x) * math.exp(0.5), rel=1e-8)
    assert x != pytest.approx(logistic_at(0.5), rel=1e-3)


# ================================================================ the other consumers

def assigned():
    g = declare(decay).graph
    at = solve_at('decay')
    contract = Integration(trajectory=False)
    return (Plan(g) + Assign(at, contract, *s.role_places(at, (s.place('x'),), contract))).graph


def test_an_ode_driven_by_integration_is_written_to_json_and_read_back():
    g = assigned()
    data = write_json(g)
    node = data['nodes'][1]['definition']
    assert node['kind'] == 'driven' and node['condition']['kind'] == 'ode'
    assert read_json(data) == g


def test_an_ode_implemented_by_diffrax_is_written_to_json_and_read_back():
    g = driven(decay, implementation=diffrax_run()).graph
    assert read_json(write_json(g)) == g


def test_an_ode_is_drawn_in_its_own_category():
    g = assigned()
    struct = xdsm_struct(g, blocks=True)
    assert struct['problem_types'] == {'^solve.decay': 'integrate'}
    assert struct['slots'] == {'^solve.decay': [("ẋ =", ['^rate.x']), ("x(t₀) =", ['^initial.x']),
                                                  ("t ∈", ['^span.x'])]}
    assert INTEGRATE.legend in render_xdsm_svg(g, blocks=True)
    assert INTEGRATE.colour[1] in render_matrix_svg(g) and INTEGRATE.colour[1] in to_dot(g)
    assert INTEGRATE.legend in render_html(g)


def test_an_ode_runs_through_the_shipped_surface_alone():
    '''Declared, driven and run with `cx` and nothing else: the initial state and the span
    read where the rate names them, inputs like any, and drawn in the ODE's notation.'''
    model = cx.flat.declare(cx.flat.rate(decay.fn, 'x', initial='x0', span='span')) \
        + cx.flat.drive(diffrax_run())
    tree  = cx.flat.run(model, {cx.Inputs: {'k': 2.0, 'x0': 1.5, 'span': (0.0, 0.8)}})
    assert tree['x'] == pytest.approx(1.5 * math.exp(-1.6), rel=1e-8)
    assert 'rate(' in cx.render_xdsm_svg(cx.flat.declare(decay))
