'''
An ODE's trajectory, a report of its contract: `Integration(trajectory=True)` takes the
`Times` saved at and reports each state's `Trajectory`. Asked for by the declaration
(`rate(..., record=, times=)`, the ODE declared driven, so a model reading a history is
coupled to it from the start) or by the contract handed to `Drive`. Declared through a
surface joining the vocabularies, as `tests/test_ode.py` is; run with `-s` for the numbers.
'''

import math
import re

import jax
import jax.numpy as jnp
import pytest

import cottax.assemblies.jax as cx
import purposes

from cottax.core import Combine, ExecutableGraph, GaussSeidel, GraphWithValues
from cottax.core.frozen import OrderedMap, answer, choice, violations
from cottax.core.maths.plan import Plan, refusals
from cottax.core.driven import Assign, Driven, DrivenGraph, Start
from cottax.core.realisation import Implement
from cottax.consumers.architectures import IDF, MDF, SAND, AtFirstMember
from cottax.consumers.architectures.refusals import NotCombinable
from cottax.consumers.declaration import declare
from cottax.consumers.execution import CallableGraph, Drive, run
from cottax.consumers.execution.interface import DriverImplementation
from cottax.consumers.execution.refusals import NoneImplements, ReportUnavailable
from cottax.consumers.json import read_json, write_json
from cottax.consumers.visualization import render_html, render_matrix_svg, render_xdsm_svg, to_dot
from cottax.consumers.visualization.xdsm import xdsm_struct
from cottax.pytree.consumers.surface import node_of as node
from cottax.pytree.path import NodePath, VarPath
from cottax.relational.consumers.architectures import FixedPointClosing
from cottax.relational.contracts import CONTRACTS as RELATIONAL_CONTRACTS, RootFinder
from cottax.ode import Dt, OdeCondition
from cottax.ode.contracts import (
    CONTRACTS as ODE_CONTRACTS, Integration, Integrator, Times, Trajectory,
)
from cottax.ode.consumers.declaration import NotAState, Rate, RecordUnpaired
from cottax.ode.consumers.visualization import INTEGRATE
import cottax.ode.consumers  # noqa: F401  registers its answers
import cottax.pytree.consumers  # noqa: F401  a tree's structure


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


s = cx.flat
'''The shipped surface over a plain dict, relational's and the ODE's parts among its own.'''

RECORDING = Integration(trajectory=True)


def diffrax_run(**settings):
    '''`Diffrax` at a tight tolerance: the tests needing it skip where it is not installed.'''
    pytest.importorskip('diffrax')
    from cottax.ode.consumers.execution import Diffrax
    return Diffrax(**{'rtol': 1e-10, 'atol': 1e-10, **settings})


def drive(*implementations, contracts=ODE_CONTRACTS) -> Drive:
    return Drive(implementations, (*RELATIONAL_CONTRACTS, *contracts), s.role_places)


def problem(at: str) -> NodePath:
    name = node(at)
    return s.node(cx.solve(name))


def per_problem(purpose, at: str) -> VarPath:
    '''Where a per-problem role of the ODE beside `at` sits by default: `^times.^solve.at`.'''
    return s.place(purpose(VarPath(problem(at).segments)))


def driven(*models, implementation, **kw) -> CallableGraph:
    return CallableGraph((Plan(declare(*models).graph) + drive(implementation, **kw)).graph)


def values(**by_name) -> OrderedMap:
    return OrderedMap((s.place(k), v) for k, v in by_name.items())


# ================================================================ the models

@s.rate('x', record={'x': 'xs'}, times='times', initial='x0', span='span')
def decay(x, k):
    return -k * x


@s.rate('x')
def bare(x, k):
    return -k * x


@s.function('area')
def area(xs, times):
    '''The time integral of `x`, by the trapezoid rule over the saved times.'''
    return jnp.trapezoid(xs, times)


@s.rate('x', 'v', record={'x': 'xs'}, times='times', initial={'x': 'x0', 'v': 'v0'},
        span='span')
def oscillator(x, v, w):
    return v, -w * w * x


@s.rate('z', record='zs', times='times', initial='z0', span='span')
def spin(z, w):
    return {'phase': w, 'amplitude': -0.1 * z['amplitude']}


# ================================================================ the declaration

def test_a_history_declares_the_ode_driven_its_initial_values_and_span_on_the_condition():
    g = declare(oscillator).graph
    ode = g[problem('oscillator')]
    assert ode.contract == RECORDING and ode.condition == declare(s.rate(
        oscillator.fn, 'x', 'v', initial={'x': 'x0', 'v': 'v0'}, span='span')).graph[problem('oscillator')]
    assert ode.condition.parameters == (s.place('x0'), s.place('v0'), s.place('span'))
    assert (ode.contract.takes, ode.contract.reports) == ((Times,), (Trajectory,))
    assert ode.data == ((s.place('times'),),)
    assert ode.reports == ((s.place('xs'), s.place(cx.trajectory('v'))),)
    assert declare(s.rate(bare.fn, 'x', span='span')).graph[problem('bare')] == OdeCondition(
        (Dt(s.place('x'), s.place(cx.rate('x')), s.place(cx.initial('x'))),), s.place('span'))
    assert declare(bare).graph[problem('bare')] == OdeCondition(
        (Dt(s.place('x'), s.place(cx.rate('x')), s.place(cx.initial('x'))),), s.place(cx.span('x')))


def test_a_value_written_for_a_place_of_the_driver_is_given_zero_too():
    rate = s.rate(bare.fn, 'x', record='xs', times=jnp.linspace(0.0, 1.0, 3), initial=0.0,
                  span=jnp.array([0.0, 1.0]))
    d = declare(rate)
    assert list(d.given) == [s.place(cx.times('x')), s.place(cx.span('x')),
                             s.place(cx.initial('x'))]
    assert d.given[s.place(cx.initial('x'))] == 0.0
    ode = d.graph[problem('bare')]
    assert ode.data == ((s.place(cx.times('x')),),)
    assert ode.condition.parameters == (s.place(cx.initial('x')), s.place(cx.span('x')))


def test_a_record_without_times_or_a_place_not_a_state_is_refused_by_its_kind():
    good = s.rate(bare.fn, 'x', record='xs', times='times')
    fields = (good.fn, good.function_reads, good.outputs)
    def kinds(**kw):
        written = dict(record=good.record, times=good.times, initial=OrderedMap(), span=None) | kw
        return [type(v) for v in violations(Rate, *fields, **written)]
    assert kinds(times=None) == [RecordUnpaired]
    assert kinds(record=OrderedMap()) == [RecordUnpaired]
    assert kinds(record=OrderedMap({s.place('y'): s.place('ys')})) == [NotAState]
    with pytest.raises(ValueError, match='`record=` is given without `times=`'):
        s.rate(bare.fn, 'x', record='xs')
    with pytest.raises(ValueError, match=r"`initial=` names \(VarPath\(\['y'\]\),\)"):
        s.rate(bare.fn, 'x', initial={'y': 'y0'})


# ================================================================ 1. decay

def test_1_decay_the_history_against_the_exponential_the_final_state_bitwise_unchanged():
    ts = jnp.linspace(0.0, 2.0, 5)
    out = run(GraphWithValues(driven(decay, implementation=diffrax_run()), values(
        k=0.5, x0=1.0, span=(0.0, 2.0), times=ts)))
    without = run(GraphWithValues(driven(bare, implementation=diffrax_run()), OrderedMap({
        s.place('k'): 0.5, s.place(cx.initial('x')): 1.0, s.place(cx.span('x')): (0.0, 2.0)})))
    xs, err = out[s.place('xs')], float(jnp.max(jnp.abs(out[s.place('xs')] - jnp.exp(-0.5 * ts))))
    print(f'\n[1] xs={xs.tolist()}\n    max |xs - exp(-kt)| = {err:.2e}; final with '
          f'{float(out[s.place("x")])!r}, without {float(without[s.place("x")])!r}')
    assert xs.shape == (5,) and err < 1e-9
    assert out[s.place('x')] == without[s.place('x')]          # bitwise: one solve, same steps


# ================================================================ 2. shapes

def test_2_two_states_an_array_state_and_a_pytree_state():
    ts = jnp.linspace(0.0, 1.5, 7)
    out = run(GraphWithValues(driven(oscillator, implementation=diffrax_run()), values(
        w=2.0, x0=1.0, v0=0.0, span=(0.0, 1.5), times=ts)))
    xs, vs = out[s.place('xs')], out[s.place(cx.trajectory('v'))]
    assert jnp.allclose(xs, jnp.cos(2 * ts), atol=1e-8)
    assert jnp.allclose(vs, -2 * jnp.sin(2 * ts), atol=1e-8)
    k = jnp.array([0.5, 1.0, 2.0])
    xa = run(GraphWithValues(driven(decay, implementation=diffrax_run()), values(
        k=k, x0=jnp.ones(3), span=(0.0, 1.0), times=ts[:4] / 1.5)))[s.place('xs')]
    assert jnp.allclose(xa, jnp.exp(-(ts[:4] / 1.5)[:, None] * k), rtol=1e-8)
    zs = run(GraphWithValues(driven(spin, implementation=diffrax_run()), values(
        w=jnp.array([1.0, 3.0]), z0={'phase': jnp.zeros(2), 'amplitude': 1.0},
        span=(0.0, 2.0), times=jnp.linspace(0.0, 2.0, 3))))[s.place('zs')]
    shapes = jax.tree.map(jnp.shape, zs)
    print(f'\n[2] x {xs.shape}, v {vs.shape}; array state {xa.shape}; pytree state {shapes}')
    assert (xs.shape, vs.shape, xa.shape) == ((7,), (7,), (4, 3))
    assert shapes == {'phase': (3, 2), 'amplitude': (3,)}


# ================================================================ 3. a plain run on the caller's tree

def test_3_a_plain_run_writes_the_history_into_the_callers_tree():
    graph = driven(decay, area, implementation=diffrax_run())
    ts = jnp.linspace(0.0, 2.0, 5)
    state = {'k': 0.5, 'x0': 1.0, 'span': (0.0, 2.0), 'times': ts,
             'x': None, 'xs': None, 'area': None}
    out  = s.run(graph, read_from={cx.Inputs: state})
    tree = out.into(state)
    print(f'\n[3] tree["xs"]={jnp.asarray(tree["xs"]).tolist()}; x={float(tree["x"])!r}; '
          f'area={float(tree["area"])!r}')
    assert jnp.allclose(tree['xs'], jnp.exp(-0.5 * ts), rtol=1e-9)
    assert tree['x'] == pytest.approx(math.exp(-1.0), rel=1e-9)
    assert set(out) - {s.place(k) for k in tree} == {s.place(cx.rate('x'))}


def test_3_times_written_as_an_array_are_given():
    ts = jnp.linspace(0.0, 2.0, 5)
    d = declare(s.rate(bare.fn, 'x', record={'x': 'xs'}, times=ts, initial='x0', span='span'))
    tree = s.run(d + drive(diffrax_run()),
                 read_from={cx.Inputs: {'k': 0.5, 'x0': 1.0, 'span': (0.0, 2.0)}})
    assert list(d.given) == [s.place(cx.times('x'))]
    assert jnp.allclose(tree['xs'], jnp.exp(-0.5 * ts), rtol=1e-9)


# ================================================================ 4. a function reading the history

@s.rate('y', record={'y': 'ys'}, times='times')
def remembering(y, ys):
    '''A rate reading its own history: on the ODE's cycle.'''
    return -jnp.mean(ys) * y


def test_4_the_history_couples_its_reader_to_the_ode_from_the_declaration():
    g = declare(decay, area).graph
    print(f'\n[4] area reads from {[str(n) for n in g.graph.ancestors((node("area"),))]}')
    assert g.graph.ancestors((node('area'),)) == (node('decay'), problem('decay'), node('area'))
    assert g.graph.owners[s.place('xs')] == problem('decay')
    assert ExecutableGraph(g).graph == g and DrivenGraph(g).graph == g
    kinds = [type(v).__name__ for v in violations(ExecutableGraph, declare(remembering).graph)]
    print(f'    a rate reading its own history: {kinds}')
    assert kinds == ['OwnReadInsideCycle']


# ================================================================ 5. the usual order

@s.function('f')
def misfit(xs, data):
    return jnp.sum((xs - data) ** 2)


def fitted():
    return declare(decay, misfit, {'opt': s.optimise(design=('k',), minimise='f')})


CLOSING = FixedPointClosing(purposes.Copy(), AtFirstMember(purposes.Closure()))


@pytest.mark.slow
@pytest.mark.parametrize('final', [MDF, IDF])
def test_5_an_architecture_first_then_drive_fits_the_rate_to_the_history(final):
    from cottax.relational.consumers.execution.jax_solvers import BFGS
    plan = Plan(fitted().graph) + final(GaussSeidel(), CLOSING, node('opt'))
    assert dict(plan.graph.within) == {problem('decay'): node('opt')}
    ExecutableGraph(plan.graph)
    plan += drive(BFGS(rtol=1e-10, atol=1e-10), diffrax_run())
    ts = jnp.linspace(0.0, 2.0, 9)
    state = {'k': 0.1, 'x0': 1.0, 'span': (0.0, 2.0), 'times': ts, 'data': jnp.exp(-0.7 * ts)}
    tree = s.run(CallableGraph(plan.graph), read_from={cx.Inputs: state, Start: state})
    print(f'\n[5] {final.__name__}: k={float(tree["k"])!r} (0.7), f={float(tree["f"]):.2e}')
    assert tree['k'] == pytest.approx(0.7, rel=1e-6)


def test_5_sand_refuses_a_recording_ode_as_any():
    sand = SAND(GaussSeidel(), CLOSING, node('opt'))
    assert [type(v) for v in refusals(sand, fitted().graph)] == [NotCombinable]


# ================================================================ 6. the mechanism, no sugar

@s.function('area', reads={'xs': s.place(cx.trajectory('x')), 'ts': per_problem(cx.times, 'bare')})
def area_of_bare(xs, ts):
    return jnp.trapezoid(xs, ts)


def test_6_the_flagged_contract_handed_to_drive_or_assigned_by_hand():
    ts = jnp.linspace(0.0, 2.0, 201)
    inputs = OrderedMap({s.place('k'): 0.5, s.place(cx.initial('x')): 1.0,
                         s.place(cx.span('x')): (0.0, 2.0), per_problem(cx.times, 'bare'): ts})
    by_drive = driven(bare, area_of_bare, implementation=diffrax_run(), contracts=(RECORDING,))
    at, g = problem('bare'), declare(bare, area_of_bare).graph
    by_hand = CallableGraph((Plan(g) + Assign(at, RECORDING, *s.role_places(
        at, g[at].unknowns, RECORDING)) + Implement(OrderedMap({at: diffrax_run()}))).graph)
    assert by_drive.graph == by_hand.graph
    out = run(GraphWithValues(by_drive, inputs))
    exact = (1 - math.exp(-1.0)) / 0.5
    print(f'\n[6] area={float(out[s.place("area")])!r}, exact {exact!r}')
    assert out[s.place('area')] == pytest.approx(exact, rel=1e-4)


# ================================================================ 7. derivatives and jit

def area_in(graph, ts):
    return lambda k: run(GraphWithValues(graph, values(k=k, x0=1.0, span=(0.0, 2.0),
                                                       times=ts)))[s.place('area')]


def test_7_grad_in_reverse_and_jacfwd_forward():
    diffrax = pytest.importorskip('diffrax')
    ts = jnp.linspace(0.0, 2.0, 201)
    rev = jax.grad(area_in(driven(decay, area, implementation=diffrax_run()), ts))(0.5)
    fwd = jax.jacfwd(area_in(driven(decay, area, implementation=diffrax_run(
        adjoint=diffrax.ForwardMode())), ts))(0.5)
    exact = (2.0 * math.exp(-1.0) * 0.5 - (1 - math.exp(-1.0))) / 0.25   # d/dk (1 - e^-2k) / k
    print(f'\n[7] grad {float(rev)!r}, jacfwd {float(fwd)!r}, d/dk of the integral {exact!r}')
    assert rev == pytest.approx(exact, rel=1e-4) and fwd == pytest.approx(rev, rel=1e-8)


@pytest.mark.slow
def test_7_jitted_once_across_new_rates_and_new_times_of_one_length():
    eqx = pytest.importorskip('equinox')
    graph, traces = driven(decay, area, implementation=diffrax_run()), []

    @eqx.filter_jit
    def go(graph, values):
        traces.append(1)
        return run(GraphWithValues(graph, values))[s.place('xs')]

    def at(k, ts):
        return values(k=jnp.asarray(k), x0=jnp.asarray(1.0),
                      span=(jnp.asarray(0.0), jnp.asarray(2.0)), times=ts)
    for k, ts in ((0.5, jnp.linspace(0.0, 2.0, 11)), (1.0, jnp.linspace(0.0, 2.0, 11)),
                  (2.0, jnp.linspace(0.0, 1.0, 11) ** 2 * 2.0)):
        assert jnp.allclose(go(graph, at(k, ts)), jnp.exp(-k * ts), rtol=1e-8)
    once = len(traces)
    go(graph, at(1.0, jnp.linspace(0.0, 2.0, 12)))
    print(f'\n[7] traces: {once} over three calls of eleven times, {len(traces)} after twelve')
    assert (once, len(traces)) == (1, 2)


# ================================================================ 8. refusals

@choice(DriverImplementation)
class EndOnly(DriverImplementation[Integrator]):
    '''Forward Euler to the end alone: it reports no history.'''
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


@choice(DriverImplementation)
class Bisection(DriverImplementation[RootFinder]):
    '''Implements a root finder, and so no integrator.'''
    needs    = frozenset()
    provides = frozenset()

    def __call__(self, posed, context, data):
        raise NotImplementedError


def test_8_an_implementation_without_the_report_is_refused_by_the_proof():
    g = (Plan(declare(decay).graph) + Drive((EndOnly(10),), (), s.role_places)).graph
    assert [type(v) for v in violations(CallableGraph, g)] == [ReportUnavailable]
    assert CallableGraph((Plan(declare(bare).graph)
                          + Drive((EndOnly(10),), ODE_CONTRACTS, s.role_places)).graph)


def test_8_drive_on_a_contracted_problem_nothing_given_implements_is_refused():
    refused = NoneImplements(problem('decay'), 'Integration')
    with pytest.raises(ValueError, match=re.escape(refused.message)):
        Plan(declare(decay).graph) + Drive((Bisection(),), RELATIONAL_CONTRACTS, s.role_places)


def test_8_drive_leaves_a_problem_already_implemented_alone():
    once = (Plan(declare(decay).graph) + Drive((EndOnly(10),), (), s.role_places)).graph
    assert (Plan(once) + Drive((EndOnly(20),), (), s.role_places)).graph == once


# ================================================================ 9. two recording ODEs joined

@s.rate('x', initial='x0', span='span', record='xs', times='times')
def position(v):
    return v


@s.rate('v', initial='v0', span='span', record='vs', times='times')
def velocity(x, w):
    return -w * w * x


def test_9_two_odes_recording_alike_join_as_one_driven_problem():
    '''Each declared driven by the same contract over one span and one set of times: their
    join is one integration recording both.'''
    at = problem('position')
    g = (Plan(declare(position, velocity).graph)
         + Combine(at, (at, problem('velocity')), Driven)).graph
    assert g[at].data == ((s.place('times'),),)
    assert g[at].reports == ((s.place('xs'), s.place('vs')),)
    ts = jnp.linspace(0.0, 1.5, 7)
    out = run(GraphWithValues(CallableGraph((Plan(g) + drive(diffrax_run())).graph), values(
        w=2.0, x0=1.0, v0=0.0, span=(0.0, 1.5), times=ts)))
    assert jnp.allclose(out[s.place('xs')], jnp.cos(2 * ts), atol=1e-8)
    assert jnp.allclose(out[s.place('vs')], -2 * jnp.sin(2 * ts), atol=1e-8)


# ================================================================ 10. results by role

def test_10_a_report_is_read_by_role_at_its_state_and_written_keyed_by_the_states():
    '''`out[Trajectory, 'x']` is the history at whatever place it sits; `out.into(history,
    Trajectory)` writes each state's at the state's own place, so a history structure is
    keyed as the state is. A per-problem role has no state's place.'''
    from cottax.consumers.execution.refusals import NoRoleAt, NoSlotFor
    graph = driven(oscillator, implementation=diffrax_run())
    ts = jnp.linspace(0.0, 1.5, 7)
    out = s.run(graph, {cx.Inputs: {'w': 2.0, 'x0': 1.0, 'v0': 0.0, 'span': (0.0, 1.5),
                                    'times': ts}})
    assert out[Trajectory, 'x'] is out['xs']
    assert out[Trajectory, 'v'] is out[s.place(cx.trajectory('v'))]
    assert out[Times, 'v'] is out['times']
    history = out.into({'x': None, 'v': None, 'w': None}, Trajectory)
    assert history['x'] is out['xs'] and history['v'] is out[s.place(cx.trajectory('v'))]
    assert history['w'] is None
    with pytest.raises(KeyError) as refused:
        out[Trajectory, 'w']
    assert refused.value.violation == NoRoleAt('Trajectory', s.place('w'))
    with pytest.raises(TypeError) as refused:
        out.into({}, Times)
    assert refused.value.violation == NoSlotFor('Times')


# ================================================================ 11. JSON and drawings

def test_11_the_declared_driven_ode_is_written_to_json_and_read_back():
    g = declare(decay, area).graph
    data = write_json(g)
    written = data['nodes'][1]['definition']
    print(f'\n[9] contract written {written["contract"]}')
    assert written['kind'] == 'driven' and written['contract']['fields'] == {'trajectory': True}
    assert read_json(data) == g


def test_11_a_recording_ode_is_drawn():
    g = declare(decay, misfit).graph
    struct = xdsm_struct(g, blocks=True)
    assert struct['problem_types'] == {'^solve.decay': 'integrate'}
    assert "xs" in struct['writes']['^solve.decay']
    assert INTEGRATE.legend in render_xdsm_svg(g, blocks=True)
    assert INTEGRATE.colour[1] in render_matrix_svg(g) and INTEGRATE.colour[1] in to_dot(g)
    assert INTEGRATE.legend in render_html(g)
