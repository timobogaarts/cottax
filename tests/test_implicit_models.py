'''
A wrapped implicit model is one function and two nodes in a cycle, driven by hand; a design
stated over the models that compute its conditions; and `Drive` over a problem of another
language.
'''

import math

import pytest

import cottax.assemblies.jax as cx
from cottax.core import problems, GraphWithValues
from cottax.core.frozen import OrderedMap
from cottax.core.maths.plan import Plan
from cottax.core.driven import DriverContract
from cottax.relational.contracts import CONTRACTS, RootFinding
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.interface import DriverImplementation
from cottax.assemblies.jax import Drive, N, flat
from cottax.relational import Le, is_root_find
from cottax.consumers.visualization.sequencing import Solve, entries
from cottax.relational.consumers.execution.jax_solvers import Newton
from test_answers import Raw                     # a condition node of no language cottax ships

v, n = flat.place, cx.absolute.node


def disc1(z, y1, y2):
    '''`y1 = z + 0.2 * y2`, written as what must vanish.'''
    return z + 0.2 * y2 - y1


def disc2(z, y1):
    return math.sqrt(abs(y1)) + z


def obj(y1, y2):
    return y1 + y2


@pytest.fixture
def sellar():
    return cx.absolute.declare(flat.residual(disc1, 'y1'), flat.function(disc2, 'y2'),
                       flat.function(obj, 'f')).graph


# ---------------------------------------------------------------- the two nodes

def test_an_implicit_model_is_a_body_and_a_problem(sellar):
    assert sellar.nodes == (n('disc1'), cx.absolute.node(cx.solve(N['disc1'])), n('disc2'), n('obj'))
    body, problem = sellar[n('disc1')], sellar[cx.absolute.node(cx.solve(N['disc1']))]
    assert body.reads == (v('z'), v('y1'), v('y2')) and body.owns == (cx.absolute.place(cx.cond(v('y1'))),)
    assert problem.owns == (v('y1'),) and problem.reads == body.owns
    assert is_root_find(problem)


def test_the_pair_is_a_coupled_block_with_a_problem(sellar):
    '''Implicitness is structural: nothing declares it, the cycle is it.'''
    assert not sellar.graph.is_acyclic
    assert problems(sellar.definitions) == (cx.absolute.node(cx.solve(N['disc1'])),)
    solve, _ = entries(sellar)
    assert isinstance(solve, Solve) and solve.problem == cx.absolute.node(cx.solve(N['disc1']))


def test_one_residual_per_unknown():
    def both(a, b):
        return a, b

    g = cx.absolute.declare(flat.residual(both, 'a', 'b')).graph
    assert g[n('both')].owns == (cx.absolute.place(cx.cond(v('a'))), cx.absolute.place(cx.cond(v('b'))))


# ---------------------------------------------------------------- a driver for the tests

class SelfStarting(DriverContract):
    '''A root find taking no data: it starts from zeros.'''
    accepts = staticmethod(is_root_find)
    takes   = ()
    reports = ()

class Relax(DriverImplementation[SelfStarting]):
    '''`u <- u + r`, each residual against zero.'''
    needs    = frozenset()
    provides = frozenset()

    def __call__(self, posed, context, data, tol=1e-12):
        x = (0.0,) * len(posed.condition.unknowns)
        for _ in range(500):
            r = posed.conditions(x, context)        # one residual per relation
            if max(map(abs, r)) < tol:
                break
            x = tuple(xi + ri for xi, ri in zip(x, r))
        return x, {}


def test_the_pair_runs(sellar):
    '''`r = z + 0.2*y2 - y1` has slope -1 in `y1`, so `u <- u + r` converges.'''
    problem = cx.absolute.node(cx.solve(N['disc1']))
    driven  = (Plan(sellar) + cx.absolute.assign(problem, SelfStarting())
               + cx.absolute.implement({problem: Relax()})).graph
    out = run(GraphWithValues(CallableGraph(driven), OrderedMap({v('z'): 1.0})))
    assert out[v('y1')] == pytest.approx(1.0 + 0.2 * out[v('y2')])
    assert out[v('y2')] == pytest.approx(math.sqrt(abs(out[v('y1')])) + 1.0)
    assert out[v('f')]  == pytest.approx(out[v('y1')] + out[v('y2')])


def test_a_closure_is_assigned_where_it_binds(sellar):
    '''What a caller has in hand is the op, not the name it minted out of nothing.'''
    plan = Plan(sellar) + cx.absolute.fixed_point_cut(v('y2'), [n('disc1')])
    assert plan.ops[-1].at == cx.absolute.node(cx.closure(N['y2']))
    driven = plan + cx.absolute.assign(plan.ops[-1].at, RootFinding(steps=False))
    assert driven.graph[plan.ops[-1].at].contract == RootFinding(steps=False)


# ---------------------------------------------------------------- a design over models

def cost(R0, B0):
    return (R0 - 3.0) ** 2 + (B0 - 2.0) ** 2


def margin(R0, B0):
    return 6.0 - R0 - B0                            # g <= 0


def test_the_optimise_pairs_with_the_models_that_compute_its_conditions():
    '''The problem owns the design variables, so the models reading them close a cycle.'''
    graph = cx.absolute.declare({'cost':   flat.function(cost, 'cost'),
                         'margin': flat.function(margin, 'margin'),
                         'design': flat.optimise(minimise='cost', design=('R0', 'B0')).condition
                                   + flat.requirement('margin', Le)}).graph
    assert graph.graph.components == ((n('cost'), n('margin'), n('design')),)
    assert graph.graph.boundary_inputs == ()


# ---------------------------------------------------------------- a problem of another language

class Holding(DriverContract):
    accepts = staticmethod(lambda node: isinstance(node, Raw))
    takes   = ()
    reports = ()


class Hold(DriverImplementation[Holding]):
    needs    = frozenset()
    provides = frozenset()
    def __call__(self, posed, context, data): return (0.0,) * len(posed.condition.unknowns), {}


@pytest.mark.parametrize('first_foreign', [True, False])
def test_drive_answers_each_problem_by_the_implementation_accepting_it(first_foreign):
    '''A shape predicate answers False for a foreign condition node, so the implementations
    may come in either order.'''
    model = {'f':    flat.residual(lambda z, a: a - z, 'a'),
             'g':    flat.function(lambda b: b, 'r'),
             'held': Raw((v('r'),), (v('b'),))}
    declared = cx.absolute.declare(model)
    implementations = (Hold(), Newton()) if first_foreign else (Newton(), Hold())
    drive = Drive(implementations, (*CONTRACTS, Holding()), flat.role_places)
    graph = (declared + drive).graph
    assert isinstance(graph[n('held')].contract, Holding)
    assert isinstance(graph[cx.absolute.node(cx.solve(n('f')))].contract, RootFinding)
