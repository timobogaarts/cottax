'''
`HostMinimiser`, what a solver on the host is written against: a few lines of adapter, whether a
Jacobian is computed being each call's, what comes back; and `ScipyMinimize`, the methods of
`minimize`, which refuses what a method would drop. Float64, as `test_solvers.py`.
'''

import logging
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from jax.tree_util import DictKey
from scipy.optimize import OptimizeWarning

import cottax.assemblies.jax as cx
import purposes
from cottax.assemblies.jax import flat
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.interface_types import Posed
from cottax.core import Function, GraphWithValues
from cottax.core.driven import Driven, PerProblem, Role, Start, Steps, role_slots
from cottax.core.frozen import OrderedMap
from cottax.core.realisation import ImplementedDriven, ImplementedFunction
from cottax.pytree.path import NodePath, VarPath
from cottax.relational import Eq, Le, Min, Objective, Relation, RelationalCondition, programme
from cottax.relational.consumers.execution.jax_solvers import (
    SLSQP, HostMinimiser, ScipyMinimize, host,
)
from cottax.relational.consumers.execution.jax_solvers.scipy_minimize import METHODS
from cottax.relational.contracts import ConstrainedMinimiser, ConstrainedMinimisation
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
    return ImplementedFunction(Function(tuple(V(r) for r in reads), tuple(V(o) for o in owns)), fn)


def optimise(objective, design, relations, implementation, bounded=False, steps=False,
             contract=None, reports=None) -> ImplementedDriven:
    unknowns   = tuple(V(d) for d in design)
    parameters = tuple(b(u) for u in unknowns for b in (LOWER, UPPER)) if bounded else ()
    bounds     = tuple(r for u in unknowns for r in (Relation(LOWER(u), u, Le),
                                                     Relation(u, UPPER(u), Le))) if bounded else ()
    condition  = RelationalCondition(tuple(relations) + bounds, (Objective(V(objective), Min),),
                                     unknowns, parameters)
    return ImplementedDriven(
        Driven(condition, contract or ConstrainedMinimisation(steps=steps),
               (tuple(STARTS(u) for u in unknowns),),
               reports if reports is not None else (((V('n'),),) if steps else ())),
        implementation)


def graph(nodes):
    return CallableGraph(graph_of(OrderedMap(nodes)))


def solve(nodes, values):
    return run(GraphWithValues(graph(nodes), OrderedMap(
        {k: jnp.asarray(v) for k, v in values.items()})))


def traced_run(g, values):
    return run(GraphWithValues(g, values))


def counted(calls, fn):
    def wrapped(*a):
        calls.append(1)
        return fn(*a)
    return wrapped


def bowl(calls=None):
    '''`(x - 3)^2`, its model counted: one count a trace of it, or the walk after a solve.'''
    fn = lambda x: (x - 3.0) ** 2
    return {N('m'): call(['x'], ['f'], fn if calls is None else counted(calls, fn))}


CAPPED = (Relation(V('g'), None, Le),)


def capped(calls=None):
    '''`(x - 3)^2` under `x - 1 <= 0`: 1.'''
    fn = lambda x: ((x - 3.0) ** 2, x - 1.0)
    return {N('m'): call(['x'], ['f', 'g'], fn if calls is None else counted(calls, fn))}


# ---------------------------------------------------------------- an adapter is a few lines

class GradientDescent(HostMinimiser[ConstrainedMinimiser]):
    '''A whole solver: what `HostMinimiser` leaves to write.'''

    reported = {Steps: int}
    provides = frozenset(reported)
    rate     : float = 0.25

    def minimise(self, problem):
        y = problem.y0
        for step in range(200):
            (gradient,) = problem.jacobians(y).objectives
            y = np.clip(y - self.rate * gradient, problem.lower, problem.upper)
        return y, {Steps: step + 1}


def test_a_solver_is_a_few_lines_and_answers_as_any_driver():
    out = solve(bowl() | {N('opt'): optimise('f', ['x'], (), GradientDescent(), steps=True)},
                {STARTS(V('x')): 0.0})
    assert float(out[V('x')]) == pytest.approx(3.0, abs=1e-6)
    assert int(out[V('n')]) == 200


def test_it_is_read_in_the_scaling_and_answers_in_the_models_units():
    '''In units of 2, the same bowl is `(2y - 3)^2`: a rate of 1/8 steps straight to it, which
    in `x` a rate of 1/8 does not.'''
    @flat.function('f')
    def model(x):
        return (x - 3.0) ** 2

    opt = flat.optimise(minimise='f', design=('x',))
    ran = cx.absolute.declare(model, {'OPT': opt}) \
        + flat.drive(GradientDescent(rate=0.125), contracts=cx.SCALED)
    out = flat.run(ran, {cx.Start: {'x': 0.0}, cx.Scales: {'x': flat.scaling(unknowns={'x': 2.0})}})
    assert float(out['x']) == pytest.approx(3.0, abs=1e-12)


def test_under_a_jit_it_goes_through_the_callback_and_comes_back_the_same():
    nodes = bowl() | {N('opt'): optimise('f', ['x'], (), GradientDescent(), steps=True)}
    x, n = eqx.filter_jit(lambda g, v: (lambda o: (o[V('x')], o[V('n')]))(traced_run(g, v)))(
        graph(nodes), OrderedMap({STARTS(V('x')): jnp.asarray(0.0)}))
    assert float(x) == pytest.approx(3.0, abs=1e-6)
    assert int(n) == 200


# ---------------------------------------------------------------- what comes back

class Verdict(PerProblem):
    '''A report of the test's own.'''


class Judged(ConstrainedMinimiser):
    '''Asks `Start`, and reports `Verdict` and `Steps`.'''

    @property
    def takes(self) -> tuple[type[Role], ...]:   return (Start,)
    @property
    def reports(self) -> tuple[type[Role], ...]: return (Verdict, Steps)


class Judging(HostMinimiser[Judged]):
    reported = {Steps: int, Verdict: bool}
    provides = frozenset(reported)

    def minimise(self, problem):
        return problem.y0, {Steps: 7, Verdict: True}


def test_a_provided_role_it_does_not_report_is_not_asked_for_in_vain():
    '''`run` trusts `provides`, so one claiming more than `reported` fails on its report.'''
    class Claims(HostMinimiser[Judged]):
        reported = {Steps: int}
        provides = frozenset({Steps, Verdict})

        def minimise(self, problem):
            return problem.y0, {Steps: 0}

    nodes = bowl() | {N('opt'): optimise('f', ['x'], (), Claims(), contract=Judged(),
                                         reports=((V('ok'),), (V('n'),)))}
    with pytest.raises(KeyError):
        solve(nodes, {STARTS(V('x')): 0.0})


def test_each_role_asked_comes_back_in_its_dtype_eager_and_traced():
    nodes = bowl() | {N('opt'): optimise('f', ['x'], (), Judging(), contract=Judged(),
                                         reports=((V('ok'),), (V('n'),)))}
    pick = lambda o: (o[V('ok')], o[V('n')])
    for go in (lambda g, v: pick(traced_run(g, v)),
               eqx.filter_jit(lambda g, v: pick(traced_run(g, v)))):
        ok, n = go(graph(nodes), OrderedMap({STARTS(V('x')): jnp.asarray(0.0)}))
        assert ok.dtype == jnp.bool_ and bool(ok) is True
        assert jnp.issubdtype(n.dtype, jnp.integer) and int(n) == 7


def test_a_role_declared_and_not_asked_does_not_come_back():
    condition = RelationalCondition((), (Objective(V('f'), Min),), (V('x'),), ())
    posed = lambda reports: Posed(condition, reports, lambda unknowns, context: (unknowns[0] ** 2,), ())
    data = {Start: (jnp.asarray(0.0),)}
    assert Judging()(posed(()), (), data)[1] == {}
    assert set(Judging()(posed((Verdict,)), (), data)[1]) == {Verdict}


# ---------------------------------------------------------------- a Jacobian is each call's

class Asks(HostMinimiser[ConstrainedMinimiser]):
    '''Stays at the start, having asked what `ask` asks of the problem.'''

    reported = {Steps: int}
    provides = frozenset(reported)
    ask      : object = None
    gradient : bool   = True

    @property
    def asks(self):
        return frozenset({'values', 'jacobians'} if self.gradient else {'values'})

    def minimise(self, problem):
        self.ask(problem)
        return problem.y0, {Steps: 0}


def programs(monkeypatch):
    '''How many times each program runs, counted where the problem is handed them.'''
    runs, init = {'values': 0, 'jacobian': 0}, host.HostProblem.__init__

    def counting(kind, program):
        def run_it(y):
            runs[kind] += 1
            return program(y)
        return run_it

    monkeypatch.setattr(host.HostProblem, '__init__', lambda self, values, jacobians, *rest:
                        init(self, counting('values', values), counting('jacobian', jacobians), *rest))
    return runs


@pytest.mark.parametrize('gradient, want', [(True, {'values': 0, 'jacobian': 1}),
                                             (False, {'values': 1, 'jacobian': 0})])
def test_setting_up_is_one_run_of_the_program_the_solver_wants_and_the_values_are_had(monkeypatch, gradient, want):
    '''The box, the sizes and the values at the start, off one trace and one run: a solver that
    wants both at every point never runs a values program, and one taking no gradient never a
    Jacobian.'''
    runs, calls, seen = programs(monkeypatch), [], {}

    def ask(p):
        form = p.values(p.y0)
        seen.update(lower=p.lower.tolist(), upper=p.upper.tolist(),
                    sizes=(p.num_equality, p.num_inequality),
                    at=(float(form.objectives[0]), float(form.inequalities[0])))

    solve(capped(calls) | {N('opt'): optimise('f', ['x'], CAPPED, Asks(ask=ask, gradient=gradient),
                                              bounded=True)},
          {STARTS(V('x')): 0.0, LOWER(V('x')): -1.0, UPPER(V('x')): 0.5})
    assert runs == want
    assert len(calls) == 2      # one trace, and the walk after
    assert seen == {'lower': [-1.0], 'upper': [0.5], 'sizes': (0, 1), 'at': (9.0, -1.0)}


def test_under_a_jit_nothing_is_traced_or_compiled_in_the_callback(caplog):
    '''On a GPU a compile there can hang. Both programs were compiled with the graph.'''
    calls, seen = [], {}

    def ask(p):
        before = len(calls)
        with caplog.at_level(logging.DEBUG), jax.log_compiles():
            caplog.clear()
            p.values(p.y0 + 1.0), p.jacobians(p.y0 + 2.0)
            seen['compiled'] = [r.message for r in caplog.records if 'ompil' in r.message]
        seen['traces'] = len(calls) - before

    nodes = capped(calls) | {N('opt'): optimise('f', ['x'], CAPPED, Asks(ask=ask))}
    jax.block_until_ready(eqx.filter(eqx.filter_jit(traced_run)(
        graph(nodes), OrderedMap({STARTS(V('x')): jnp.asarray(0.0)})), eqx.is_array))
    assert seen == {'compiled': [], 'traces': 0}


@pytest.mark.parametrize('asks, asked, name', [(frozenset({'values'}), lambda p: p.jacobians(p.y0), 'jacobians'),
                                               (frozenset({'jacobians'}), lambda p: p.values(p.y0 + 1.0), 'values')])
def test_under_a_jit_a_program_not_in_asks_is_refused_and_not_compiled_late(asks, asked, name):
    class Declares(HostMinimiser[ConstrainedMinimiser]):
        reported = {Steps: int}
        provides = frozenset(reported)
        def minimise(self, problem):
            asked(problem)
            return problem.y0, {Steps: 0}

    Declares.asks = asks
    with pytest.raises(Exception, match=f'`{name}` is not in this solver'):
        jax.block_until_ready(eqx.filter(eqx.filter_jit(traced_run)(
            graph(capped() | {N('opt'): optimise('f', ['x'], CAPPED, Declares())}),
            OrderedMap({STARTS(V('x')): jnp.asarray(0.0)})), eqx.is_array))


def test_with_nothing_traced_a_program_not_in_asks_still_runs():
    '''Nothing is running, so a compile cannot hang: `asks` only says what to compile ahead.'''
    seen = []

    class Undeclared(HostMinimiser[ConstrainedMinimiser]):
        reported = {Steps: int}
        provides = frozenset(reported)
        asks = frozenset({'values'})
        def minimise(self, problem):
            seen.append(problem.jacobians(problem.y0).objectives.tolist())
            return problem.y0, {Steps: 0}

    solve(capped() | {N('opt'): optimise('f', ['x'], CAPPED, Undeclared())}, {STARTS(V('x')): 0.0})
    assert seen == [[[-6.0]]]


def test_a_gradient_free_method_compiles_no_jacobian(monkeypatch):
    '''Nor is it handed one, which scipy would warn of.'''
    runs = programs(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        out = solve(capped() | {N('opt'): optimise('f', ['x'], CAPPED, ScipyMinimize(method='COBYQA'))},
                    {STARTS(V('x')): 0.0})
    assert float(out[V('x')]) == pytest.approx(1.0, abs=1e-5)
    assert runs['jacobian'] == 0 and runs['values'] > 1


def test_the_jacobians_are_each_groups_in_its_own_rows():
    '''`min x^2 + y^2`, `x + y = 2`, `x - y <= 0`, the group sizes 1, 1, 1 over two unknowns.'''
    seen = {}

    def ask(p):
        form = p.jacobians(np.array([1.0, 3.0]))
        seen.update(objectives=form.objectives.tolist(), equalities=form.equalities.tolist(),
                    inequalities=form.inequalities.tolist())

    solve({N('m'): call(['x', 'y'], ['f', 'h', 'g'],
                        lambda x, y: (x ** 2 + y ** 2, x + y - 2.0, x - y)),
           N('opt'): optimise('f', ['x', 'y'], (Relation(V('h'), None, Eq), Relation(V('g'), None, Le)),
                              Asks(ask=ask))},
          {STARTS(V('x')): 0.0, STARTS(V('y')): 0.0})
    assert seen == {'objectives': [[2.0, 6.0]], 'equalities': [[1.0, 1.0]],
                    'inequalities': [[1.0, -1.0]]}


# ---------------------------------------------------------------- a point in the model's units

def asked_unknowns(start, scaling, points):
    '''`problem.unknowns` at each of `points`, by an implementation handed `scaling`, and what
    `unscaled_unknowns` says of them.'''
    design = tuple(V(n) for n in start)
    condition = RelationalCondition((), (Objective(V('f'), Min),), design, ())
    seen = []

    def ask(p):
        seen.extend(p.unknowns(np.asarray(y)) for y in points)

    posed = Posed(condition, (), lambda unknowns, context: (jnp.asarray(0.0),), ())
    Asks(ask=ask)(posed, (), {Start: tuple(start.values()), cx.Scales: (scaling,)})
    flat_start, unravel = ravel_pytree(cx.scaled_unknowns(condition, tuple(start.values()), scaling))
    want = [np.asarray(ravel_pytree(cx.unscaled_unknowns(
                condition, unravel(jnp.asarray(y, flat_start.dtype)), scaling))[0], float)
            for y in points]
    return seen, want, design


POINTS = [[0.1, -7.3, 1e-300, 3.14159, 2.0 / 3.0], [1e300, -0.0, 0.0, 1.0, np.pi]]
FIVE = {'a': np.zeros(2), 'b': np.zeros(3)}


@pytest.mark.parametrize('column', [lambda a, b: {},
                                    lambda a, b: {a: jnp.asarray(1.0 / 3.0)},
                                    lambda a, b: {a: jnp.asarray(1e6), b: jnp.asarray(np.pi)},
                                    lambda a, b: {a: jnp.asarray(-7.0), b: jnp.asarray(-1e-3)},
                                    lambda a, b: {a: jnp.asarray([3.0, -1e6]),
                                                  b: jnp.asarray([1.1, 2.2, -3.3])}],
                         ids=['none', 'one', 'both', 'negative', 'arrays'])
def test_unknowns_is_to_the_bit_what_unscaled_unknowns_says(column):
    scaling = cx.Scaling(OrderedMap(column(V('a'), V('b'))))
    seen, want, _ = asked_unknowns(FIVE, scaling, POINTS)
    assert [s.tobytes() for s in seen] == [w.tobytes() for w in want]


def test_unknowns_under_a_jit_is_in_the_models_units_and_retraces_for_no_scale():
    seen, traces = [], []

    def ask(p):
        seen.append(p.unknowns(np.array([3.0])).item())

    def model(x):
        traces.append(1)
        return (x - 3.0) ** 2

    condition = RelationalCondition((), (Objective(V('f'), Min),), (V('x'),), ())
    implementation = ImplementedDriven(Driven(condition, ConstrainedMinimisation(steps=False, scaled=True),
                                              ((STARTS(V('x')),), (V('s'),)), ()), Asks(ask=ask))
    go = eqx.filter_jit(traced_run)
    nodes = {N('m'): call(['x'], ['f'], model), N('opt'): implementation}
    after = []
    for scale in (2.0, -5.0):
        scaling = cx.Scaling(OrderedMap({V('x'): jnp.asarray(scale)}))
        jax.block_until_ready(eqx.filter(go(graph(nodes), OrderedMap(
            {STARTS(V('x')): jnp.asarray(0.0), V('s'): scaling})), eqx.is_array))
        after.append(len(traces))
    assert seen == [6.0, 15.0]
    assert after[1] == after[0]


# ---------------------------------------------------------------- a Python scalar is compile-time

def scalar_nodes(calls):
    return {N('m'): call(['x', 'a'], ['f'], counted(calls, lambda x, a: (x - a) ** 2)),
            N('opt'): optimise('f', ['x'], (), Asks(ask=lambda p: p.values(p.y0)))}


@pytest.mark.parametrize('given, values, traces', [(float, (2.0, 2.0, 3.0), [1, 0, 1]),
                                                   (jnp.asarray, (2.0, 3.0, 4.0), [1, 0, 0])],
                         ids=['scalar', 'array'])
def test_a_python_scalar_compiles_per_value_and_an_array_once(given, values, traces):
    '''Each solve walks the model once after; a trace is any count beyond that.'''
    calls: list = []
    nodes = scalar_nodes(calls)

    def traces_for(a):
        before = len(calls)
        traced_run(graph(nodes), OrderedMap({STARTS(V('x')): jnp.asarray(0.0), V('a'): given(a)}))
        return len(calls) - before - 1

    assert [traces_for(a) for a in values] == traces


def test_a_weak_float32_input_under_x64_compiles_ahead_to_what_the_callback_gets():
    '''Made with x64 off and run with it on: the callback gets each as a float32, so that is
    what compiling ahead takes, though its tracer's weak type would promote to float64.'''
    with jax.enable_x64(False):
        values = OrderedMap({STARTS(V('x')): jnp.asarray(0.0), V('a'): jnp.asarray(3.0)})
    nodes = {N('m'): call(['x', 'a'], ['f'], lambda x, a: (x - a) ** 2),
             N('opt'): optimise('f', ['x'], (), SLSQP())}
    x = eqx.filter_jit(lambda g, v: traced_run(g, v)[V('x')])(graph(nodes), values)
    assert float(x) == pytest.approx(3.0, abs=1e-5)


def test_a_python_scalar_reaches_the_model_as_it_is():
    '''It can be branched on, which a traced value could not be, and declaring it did not change
    it: `valued` converts nothing.'''
    @flat.function('f')
    def model(x, a):
        return (x - (3.0 if a > 1.0 else 7.0)) ** 2

    opt = flat.optimise(minimise='f', design=('x',))
    ran = cx.absolute.declare(model, {'OPT': opt}) + flat.drive(SLSQP())
    assert type(flat.valued(ran, {cx.Start: {'x': 0.0}, cx.Inputs: {'a': 2.0}}).values[flat.place('a')]) is float
    for a, want in ((2.0, 3.0), (0.5, 7.0)):
        out = flat.run(ran, {cx.Start: {'x': jnp.asarray(0.0)}, cx.Inputs: {'a': a}})
        assert float(out['x']) == pytest.approx(want, abs=1e-6)


# ---------------------------------------------------------------- what the old SLSQP refused

@pytest.mark.parametrize('name, implementation', [('SLSQP', SLSQP()), ('layer', GradientDescent())])
def test_a_bound_over_a_vector_unknown(name, implementation):
    out = solve({N('m'): call(['x'], ['f'], lambda x: jnp.sum((x - 3.0) ** 2)),
                 N('opt'): optimise('f', ['x'], (), implementation, bounded=True)},
                {STARTS(V('x')): jnp.zeros(3), LOWER(V('x')): -1.0, UPPER(V('x')): 0.5})
    assert out[V('x')].tolist() == pytest.approx([0.5] * 3, abs=1e-6)


def test_an_array_valued_constraint_on_a_vector_unknown():
    '''It was a `TypeError` while stacking the groups; each entry is a row of its own.'''
    out = solve({N('m'): call(['x'], ['f', 'g'], lambda x: (jnp.sum((x - 3.0) ** 2), x - 1.0)),
                 N('opt'): optimise('f', ['x'], CAPPED, SLSQP())},
                {STARTS(V('x')): jnp.zeros(3)})
    assert out[V('x')].tolist() == pytest.approx([1.0] * 3, abs=1e-6)


def test_a_vector_objective_is_refused_and_not_read_for_its_first_entry():
    with pytest.raises(ValueError):
        solve({N('m'): call(['x'], ['f'], lambda x: (x - 3.0) ** 2),
               N('opt'): optimise('f', ['x'], (), SLSQP())},
              {STARTS(V('x')): jnp.zeros(3)})


def moded(mode):
    '''A model branching on a value that is no array, as one under a `filter_jit` may.'''
    return graph({N('m'): call(['x', 'mode'], ['f'],
                               lambda x, mode: (x - (3.0 if mode == 'three' else 7.0)) ** 2),
                  N('opt'): optimise('f', ['x'], (), SLSQP())})


@pytest.mark.parametrize('go', [traced_run, eqx.filter_jit(traced_run)], ids=['eager', 'jit'])
@pytest.mark.parametrize('mode, want', [('three', 3.0), ('other', 7.0)])
def test_a_value_that_is_no_array_reaches_the_model_as_it_is(go, mode, want):
    '''It was a `TypeError` going into the program or the callback.'''
    out = go(moded(mode), OrderedMap({STARTS(V('x')): jnp.asarray(0.0), V('mode'): mode}))
    assert float(out[V('x')]) == pytest.approx(want, abs=1e-6)


# ---------------------------------------------------------------- the methods of minimize

@pytest.mark.parametrize('method', ['SLSQP', 'trust-constr', 'COBYQA', 'COBYLA'])
@pytest.mark.parametrize('sense', [Le, Eq], ids=['inequality', 'equality'])
def test_a_method_holding_constraints_holds_them(method, sense):
    out = solve(capped() | {N('opt'): optimise('f', ['x'], (Relation(V('g'), None, sense),),
                                               ScipyMinimize(method=method))},
                {STARTS(V('x')): 0.0})
    assert float(out[V('x')]) == pytest.approx(1.0, abs=1e-3)     # trust-constr's barrier


@pytest.mark.parametrize('method', ['SLSQP', 'trust-constr', 'COBYQA', 'COBYLA', 'L-BFGS-B', 'TNC',
                                    'Nelder-Mead', 'Powell'])
def test_a_method_holding_bounds_holds_them(method):
    out = solve(bowl() | {N('opt'): optimise('f', ['x'], (), ScipyMinimize(method=method),
                                             bounded=True)},
                {STARTS(V('x')): 0.0, LOWER(V('x')): -1.0, UPPER(V('x')): 0.5})
    assert float(out[V('x')]) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize('method', ['SLSQP', 'trust-constr', 'COBYQA', 'COBYLA', 'L-BFGS-B', 'TNC',
                                    'Nelder-Mead', 'Powell', 'BFGS', 'CG'])
def test_every_method_answers_a_problem_it_can(method):
    out = solve(bowl() | {N('opt'): optimise('f', ['x'], (), ScipyMinimize(method=method),
                                             steps=True)}, {STARTS(V('x')): 0.0})
    assert float(out[V('x')]) == pytest.approx(3.0, abs=1e-3)
    assert int(out[V('n')]) >= 1


@pytest.mark.parametrize('method', ['L-BFGS-B', 'TNC', 'Nelder-Mead', 'Powell', 'BFGS', 'CG'])
def test_a_method_that_would_drop_a_constraint_is_refused(method):
    with pytest.raises(ValueError, match=f'{method} has no way to hold inequalities'):
        solve(capped() | {N('opt'): optimise('f', ['x'], CAPPED, ScipyMinimize(method=method))},
              {STARTS(V('x')): 0.0})
    with pytest.raises(ValueError, match=f'{method} has no way to hold equalities'):
        solve({N('m'): call(['x'], ['f', 'h'], lambda x: ((x - 3.0) ** 2, x - 1.0)),
               N('opt'): optimise('f', ['x'], (Relation(V('h'), None, Eq),),
                                  ScipyMinimize(method=method))},
              {STARTS(V('x')): 0.0})


@pytest.mark.parametrize('method', ['BFGS', 'CG'])
def test_a_method_that_would_drop_a_bound_is_refused(method):
    with pytest.raises(ValueError, match=f'{method} has no way to hold bounds'):
        solve(bowl() | {N('opt'): optimise('f', ['x'], (), ScipyMinimize(method=method),
                                           bounded=True)},
              {STARTS(V('x')): 0.0, LOWER(V('x')): -1.0, UPPER(V('x')): 0.5})


@pytest.mark.parametrize('method', ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-exact',
                                    'trust-krylov', 'slsqp', 'nonsense'])
def test_a_method_not_on_offer_is_refused(method):
    with pytest.raises(ValueError, match='not a `minimize` method on offer'):
        solve(bowl() | {N('opt'): optimise('f', ['x'], (), ScipyMinimize(method=method))},
              {STARTS(V('x')): 0.0})


def test_a_refusal_is_had_under_a_jit_too():
    nodes = capped() | {N('opt'): optimise('f', ['x'], CAPPED, ScipyMinimize(method='L-BFGS-B'))}
    with pytest.raises(ValueError, match='no way to hold inequalities'):
        eqx.filter_jit(traced_run)(graph(nodes), OrderedMap({STARTS(V('x')): jnp.asarray(0.0)}))


@pytest.mark.parametrize('method', sorted(METHODS))
def test_max_steps_reaches_each_method_and_strict_names_it(method):
    '''One step is too few for any of them on this, each by its own option: `TNC` has no `maxiter`.'''
    steep = {N('m'): call(['x'], ['f'], lambda x: jnp.sum((x - 3.0) ** 4 + x ** 2))}
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)     # scipy's for an option unknown to it
        with pytest.raises(RuntimeError, match=f'{method} did not converge'):
            solve(steep | {N('opt'): optimise('f', ['x'], (), ScipyMinimize(method=method, max_steps=1))},
                  {STARTS(V('x')): jnp.full(3, 50.0)})


def test_slsqp_is_the_method_of_that_name():
    assert SLSQP().method == 'SLSQP' == ScipyMinimize().method
    assert SLSQP(1e-6).jacobian is jax.jacfwd
    assert SLSQP(1e-6, 5).tol == 1e-6 and SLSQP(1e-6, 5).max_steps == 5


# ---------------------------------------------------------------- a real IDF, scaled or not

def idf(driver, M):
    '''An IDF whose couplings are `M`, in the same units of it whatever `M` is.'''
    @flat.function('y1')
    def d1(z, y2):
        return M * (z + 0.2 * y2 / M)

    @flat.function('y2')
    def d2(z, y1):
        return M * (jnp.sqrt(jnp.abs(y1 / M)) + 0.1 * z)

    @flat.function('f')
    def o(z, y1, y2):
        return (z - 2.0) ** 2 + 0.1 * y1 / M + 0.1 * y2 / M

    @flat.inequality_constraint('g')
    def c(z):
        return z - 1.0

    opt = flat.optimise(minimise='f', design=('z',), subject_to=c)
    return (cx.absolute.declare(d1, d2, o, c, {'OPT': opt}) + flat.idf('OPT')
            + flat.drive(driver, cx.Picard(), cx.Newton(), contracts=cx.SCALED))


def idf_scaling(ran, M):
    '''An unknown and its coupling each in units of `M`.'''
    (_, problem, _), = role_slots(ran.graph, cx.Scales)
    condition = ran.graph[problem].condition
    column = {u: jnp.asarray(M if 'y2' in str(u) else 1.0) for u in condition.unknowns}
    rows = OrderedMap((r, column[s]) for r in programme(condition).equalities
                      for s in (r.lhs, r.rhs) if s in condition.unknowns)
    return cx.Scaling(OrderedMap(column), None, rows)


def idf_answer(driver, M, scaled):
    ran = idf(driver, M)
    start = {'z': jnp.asarray(0.0), flat.copy('y2'): jnp.asarray(M)}
    scaling = idf_scaling(ran, M) if scaled else cx.Scaling()
    values = flat.valued(ran, {cx.Start: start, cx.Scales: {'z': scaling}}).values
    out = run(GraphWithValues(cx.JitGraph(ran.graph), values))
    return float(out[flat.place('z')]), float(out[flat.place('y2')]) / M


@pytest.mark.parametrize('driver', [SLSQP(), ScipyMinimize(method='SLSQP'),
                                    ScipyMinimize(method='COBYQA')], ids=['SLSQP', 'named', 'COBYQA'])
def test_a_million_read_in_its_scale_is_the_problem_in_units_of_one(driver):
    '''Unscaled at one and scaled at a million, one answer. Unscaled at a million is left out:
    SLSQP gives a wrong one there, which is what scaling is for.'''
    plain = idf_answer(driver, 1.0, scaled=False)
    assert plain == pytest.approx((1.0, 1.214889), abs=1e-5)
    assert idf_answer(driver, 1e6, scaled=True) == pytest.approx(plain, abs=1e-5)
