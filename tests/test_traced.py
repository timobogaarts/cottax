'''
`Traced`: the run traced once by jax -- the graph's jaxpr, every place's shape -- and the run
refusing a miscount, and naming the node a failure came from.
'''

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cottax.assemblies.jax as cx
from cottax.core import GraphWithValues
from cottax.core.frozen import OrderedMap, violations
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.jax_jit import Traced
from cottax.consumers.execution.refusals import ReturnedUnlike, TraceFailed

flat = cx.flat
v = flat.place


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def graph(*models):
    return CallableGraph(cx.absolute.declare(*models).graph)


def two(a):
    return a, 2 * a, 3 * a


def test_a_run_refuses_a_miscount_naming_the_node():
    g = graph(flat.function(two, 'b', 'c'))
    with pytest.raises(ValueError) as refused:
        run(GraphWithValues(g, OrderedMap({v('a'): 1.0})))
    assert refused.value.args[0] == ReturnedUnlike(
        cx.absolute.node('two'), 2, (v('b'), v('c')), '3 values').message


def test_one_output_takes_the_whole_return_value():
    g = graph(flat.function(two, 'b'))
    shaped = Traced(GraphWithValues(g, OrderedMap({v('a'): 1.0}))).shapes[v('b')]
    assert [(s.shape, s.dtype) for s in shaped] == [((), jnp.float64)] * 3


def test_every_place_is_shaped_from_shapes_alone():
    @flat.function('b', 'c')
    def split(a, n):
        return a[:n], jnp.sum(a)

    g = graph(split)
    t = Traced(GraphWithValues(g, OrderedMap({v('a'): jax.ShapeDtypeStruct((5,), jnp.float32),
                                              v('n'): 2})))           # a setting stays a value
    assert t.shapes[v('b')] == jax.ShapeDtypeStruct((2,), jnp.float32)
    assert t.shapes[v('c')] == jax.ShapeDtypeStruct((), jnp.float32)
    assert (t.shapes[v('n')].shape, t.shapes[v('n')].dtype) == ((), jnp.int64)


def test_a_solve_is_traced_whole_and_its_interior_shaped():
    @flat.residual('u')
    def square(u, z):
        return u * u - z

    model = cx.absolute.declare(square) + cx.flat.drive(cx.Newton())
    g = CallableGraph(model.graph)
    start = flat.place(cx.start('u'))
    t = Traced(GraphWithValues(g, OrderedMap({v('z'): jnp.zeros(3), start: jnp.ones(3)})))
    assert t.shapes[v('u')] == jax.ShapeDtypeStruct((3,), jnp.float64)
    assert t.shapes[flat.place(cx.cond('u'))] == jax.ShapeDtypeStruct((3,), jnp.float64)


def numpyish(a):
    return float(np.asarray(a).sum())


def test_the_first_failure_is_refused_by_node_and_stays_data():
    shapes = OrderedMap({v('a'): jax.ShapeDtypeStruct((3,), jnp.float64)})
    for model, kind, node in ((flat.function(two, 'b', 'c'), ReturnedUnlike, 'two'),
                              (flat.function(numpyish, 'd'), TraceFailed, 'numpyish')):
        [found] = violations(Traced, GraphWithValues(graph(model), shapes))
        assert (type(found), found.node) == (kind, cx.absolute.node(node))


def test_shapes_that_do_not_match_are_refused_at_the_node():
    @flat.function('c')
    def add(a, b):
        return a + b

    [found] = violations(Traced, GraphWithValues(graph(add), OrderedMap({
        v('a'): jax.ShapeDtypeStruct((3,), jnp.float64),
        v('b'): jax.ShapeDtypeStruct((4,), jnp.float64)})))
    assert type(found) is TraceFailed and found.node == cx.absolute.node('add')


def test_the_whole_graph_is_one_jaxpr():
    g = graph(flat.function(two, 'b'))
    t = Traced(GraphWithValues(g, OrderedMap({v('a'): jax.ShapeDtypeStruct((3,), jnp.float64)})))
    assert [x.aval.shape for x in t.jaxpr.jaxpr.invars] == [(3,)]
    assert len(t.jaxpr.out_avals) == len(jax.tree.leaves(t.shapes))


def test_a_failure_in_a_plain_run_names_its_node():
    try:
        run(GraphWithValues(graph(flat.function(numpyish, 'b')),
                            OrderedMap({v('a'): 'not a number'})))
    except Exception as e:
        assert "raised at node NodePath(['numpyish'])" in e.__notes__
    else:
        raise AssertionError('the run should fail')


def test_a_violation_raised_carries_itself():
    refused = ReturnedUnlike(cx.absolute.node('two'), 2, (v('b'), v('c')), '3 values')
    assert refused.raised.violation is refused
