'''
The graph is static, and re-assembling it gives the same graph.

Two claims, and the second rests on the first. A definition is a jit cache key, so
wrapping the same functions twice must give an *equal* body -- which a plain bound
method never does, since it is minted fresh on every access and compares its receiver by
pointer. And a graph holding no unhashable leaf is hashable outright, so it can be handed
to `jit` as a static argument: hashed once and never flattened.

The refusals themselves (an array, a bound method, a `functools.partial`) are
`test_jax_jit.py`'s.
'''

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from jax.tree_util import DictKey as D

from cottax.core import Function, GraphWithValues
from cottax.core.frozen import OrderedMap
from cottax.consumers.execution import run as run_graph
from cottax.assemblies.jax import flat


def to_graph(*models):
    return flat.declare(*models).graph
from cottax.consumers.execution.jax_jit import JitGraph
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedFunction
from graphs import graph_of


def var(name: str) -> VarPath:
    return VarPath((D(name),))


def node(name: str) -> NodePath:
    return NodePath((D(name),))


def gain(z):
    return 2.0 * z


def offset(y):
    return y + 1.0


def build():
    return to_graph(flat.function(gain, 'y'), flat.function(offset, 'f'))


# ---------------------------------------------------------------- re-assembly

def test_re_assembling_a_graph_gives_an_equal_graph():
    '''The whole point: two runs of the same declarations are one jit cache key.'''
    assert build() == build()
    assert hash(build()) == hash(build())


def test_the_body_is_the_function_and_not_its_wrapper():
    '''The wrapper is a declaration: what runs is the function it wraps.'''
    assert build()[node('gain')].implementation is gain


def test_a_re_assembled_graph_traces_once():
    '''The consequence that is paid for: a graph with no array leaf filters to static, so
    it is the cache key rather than a structure to flatten.'''
    traces = []

    @eqx.filter_jit
    def run(graph, z):
        traces.append(1)
        return run_graph(GraphWithValues(JitGraph(graph), OrderedMap({var('z'): z})))

    for i in range(3):
        out = run(build(), jnp.asarray(float(i)))
    assert len(traces) == 1
    assert out[var('f')] == pytest.approx(5.0)


# ---------------------------------------------------------------- a value in a body

def test_a_callable_value_is_a_body_when_it_compares_by_value():
    '''A size, a count, a switch: it cannot be a variable, since a variable travels in the
    env and a traced integer sizes no array. A frozen callable holding it compares by
    value, so re-assembly is still one cache key.'''
    class Sized(eqx.Module):
        n : int = eqx.field(static=True)
        def __call__(self, z):  return z * self.n

    graph = lambda: to_graph({'sized': flat.function(Sized(3), 'y')})
    assert graph() == graph()
    JitGraph(graph())


# ---------------------------------------------------------------- the ratchet

def _proved(implementation):
    '''The ratchet is a jit fact, so it is asked where the jit cache key is built: a
    node carrying anything at all is a legitimate `ImplementedFunction`, and `JitGraph`
    is what refuses what jit cannot re-assemble.'''
    return JitGraph(graph_of(OrderedMap({
        node('one'): ImplementedFunction(Function((var('z'),), (var('q'),)),
                                 implementation)})))


def test_a_jtu_partial_is_refused():
    '''`jtu.Partial` is a pytree, so its arrays are visible -- but it defines no `__eq__`,
    so it compares by identity and re-assembly is a fresh cache key regardless.'''
    with pytest.raises(TypeError, match='identity'):
        _proved(jtu.Partial(lambda a, z: z, 1.0))


def test_the_ratchet_is_asked_before_the_array_ban():
    '''A plain bound method is one opaque leaf, so `check_hashable` cannot see the arrays
    behind it. Refusing it first is what closes that hole rather than leaving it.'''
    class Hides:
        def __init__(self):  self.v = jnp.ones(3)
        def body(self, z):   return self.v * z

    hidden = Hides().body
    assert not [l for l in jtu.tree_leaves(hidden) if isinstance(l, jnp.ndarray)]
    with pytest.raises(TypeError, match='identity'):
        _proved(hidden)
