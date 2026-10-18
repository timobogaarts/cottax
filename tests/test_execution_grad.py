'''
`execution.run` under jax's transforms: `grad`, `jacfwd` and `jacrev` through a
solve, by optimistix's implicit adjoint, eagerly and under `filter_jit`.

The models are `test_execution_jax`'s. A derivative is taken w.r.t. the whole
`OrderedMap` of inputs where every input's derivative is asked (it comes back as an
`OrderedMap`), and through a scalar wrapper otherwise.
'''

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import DictKey

from cottax.core import Function, GraphWithValues
from cottax.core.frozen import OrderedMap
from cottax.core.driven import Driven
from cottax.relational.contracts import RootFinding
from cottax.consumers.execution import CallableGraph, run
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


newton = Newton(rtol=1e-10, atol=1e-10)


def call(reads, owns, fn) -> ImplementedFunction:
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), fn)


def driven(relations, unknowns) -> ImplementedDriven:
    condition = RelationalCondition(tuple(relations), (), tuple(unknowns), ())
    return ImplementedDriven(Driven(condition, RootFinding(steps=False),
                                    tuple((STARTS(u),) for u in unknowns), ()), newton)


STARTS = purposes.Start()


def start(u: str) -> VarPath:
    return STARTS(V(u))


SQUARE = CallableGraph(graph_of(OrderedMap({
    N('mid'): call([V('u')], [V('w')], lambda u: u * 1.0),
    N('res'): call([V('w'), V('z')], [V('r')], lambda w, z: w * w - z),
    N('h')  : driven([Relation(V('r'), None, Eq)], [V('u')]),
})))

COSINE = CallableGraph(graph_of(OrderedMap({
    N('g')  : call([V('u'), V('a')], [V('gu')], lambda u, a: a * jnp.cos(u)),
    N('fix'): driven([Relation(V('u'), V('gu'), Eq)], [V('u')]),
})))

NESTED = CallableGraph(graph_of(OrderedMap({
    N('Mesh')     : call([V('x')], [V('k')], lambda x: 1.0 + x),
    N('DiscA')    : call([V('u'), V('k'), V('y')], [V('r_u')], lambda u, k, y: u * u - k * y),
    N('close')    : driven([Relation(V('r_u'), None, Eq)], [V('u')]),
    N('DiscB')    : call([V('u')], [V('y')], lambda u: 1.0 + u),
    N('Objective'): call([V('u'), V('y')], [V('f')], lambda u, y: u + y),
    N('Aim')      : call([V('f'), V('t')], [V('g')], lambda f, t: f - t),
    N('target')   : driven([Relation(V('g'), None, Eq)], [V('x')]),
}), OrderedMap({N('close'): N('target')})))


def of(graph, wanted):
    '''`values -> run(...)[wanted]`, over an `OrderedMap` of inputs.'''
    return lambda values: run(GraphWithValues(graph, values))[wanted]


def inputs(**by_name) -> OrderedMap:
    return OrderedMap({(start(k[6:]) if k.startswith('start_') else V(k)): jnp.asarray(v)
                       for k, v in by_name.items()})


def central(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2.0 * h)


# ---------------------------------------------------------------- the transforms

@pytest.mark.slow
def test_grad_through_a_root_find_is_the_implicit_derivative():
    '''`u = sqrt(z)`; the solution does not depend on its start.'''
    d = jax.grad(of(SQUARE, V('u')))(inputs(z=9.0, start_u=1.0))
    assert isinstance(d, OrderedMap)
    assert float(d[V('z')]) == pytest.approx(1.0 / 6.0, rel=1e-8)
    assert float(d[start('u')]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize('jac', [jax.jacfwd, jax.jacrev])
def test_forward_and_reverse_through_a_fixed_point(jac):
    '''`u = a cos u` gives `du/da = cos u / (1 + a sin u)`.'''
    a = 0.7
    u_of_a = lambda a: of(COSINE, V('u'))(inputs(a=a, start_u=0.5))
    u = float(u_of_a(jnp.asarray(a)))
    d = float(jac(u_of_a)(jnp.asarray(a)))
    assert d == pytest.approx(math.cos(u) / (1.0 + a * math.sin(u)), rel=1e-8)
    assert d == pytest.approx(float(central(u_of_a, jnp.asarray(a))), rel=1e-6)


@pytest.mark.slow
def test_grad_through_a_solve_nested_in_a_solve():
    '''`dx/dt` through the outer Newton, whose Jacobian runs through the inner one.'''
    x_of_t = lambda t: of(NESTED, V('x'))(inputs(t=t, start_x=0.5, start_u=3.0))
    t = jnp.asarray(6.0)
    d = float(jax.grad(x_of_t)(t))
    assert d == pytest.approx(float(central(x_of_t, t)), rel=1e-6)
    u = (6.0 - 1.0) / 2.0                          # x = u^2 / (1 + u) - 1, u = (t - 1) / 2
    assert d == pytest.approx((u * u + 2.0 * u) / (1.0 + u) ** 2 / 2.0, rel=1e-8)


@pytest.mark.parametrize('graph, wanted, values', [
    pytest.param(SQUARE, V('u'), dict(z=9.0, start_u=1.0), id='square'),
    pytest.param(COSINE, V('u'), dict(a=0.7, start_u=0.5), id='cosine', marks=pytest.mark.slow),
    pytest.param(NESTED, V('x'), dict(t=6.0, start_x=0.5, start_u=3.0), id='nested',
                 marks=pytest.mark.slow),
])
def test_a_jitted_gradient_is_the_eager_one(graph, wanted, values):
    f = of(graph, wanted)
    eager = jax.grad(f)(inputs(**values))
    jitted = eqx.filter_jit(jax.grad(f))(inputs(**values))
    assert list(jitted.keys()) == list(eager.keys())
    for k in eager.keys():
        assert float(jitted[k]) == pytest.approx(float(eager[k]), rel=1e-8, abs=1e-12)
