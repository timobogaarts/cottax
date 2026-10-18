'''
The container keyed by a `Path`.

`OrderedMap` is the container, and the only one: a plain `dict` cannot stand in, since jax
flattens one by sorting its keys and a `Path` is deliberately unordered.
'''

import jax
import pytest
from jax.tree_util import GetAttrKey

from cottax.core.frozen import OrderedMap
from cottax.pytree.path import NodePath, VarPath
from cottax.core import Function
from cottax.core.realisation import ImplementedFunction
from graphs import graph_of


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def call(reads, owns):
    return ImplementedFunction(Function(tuple(V(r) for r in reads), tuple(V(o) for o in owns)),
                       lambda *a: None)


def test_the_order_written_is_the_order_kept():
    m = OrderedMap({N('c'): 3, N('a'): 1})
    assert isinstance(m, OrderedMap)
    assert list(m) == [N('c'), N('a')] and m[N('a')] == 1


def test_a_graph_is_the_same_graph():
    g = graph_of(OrderedMap({N('b'): call(['x'], ['y']), N('a'): call([], ['x'])}))
    assert isinstance(g.definitions, OrderedMap)
    assert g.nodes == (N('b'), N('a'))                    # binding order, not sorted
    assert g.graph.topological_order == (N('a'), N('b'))
    assert hash(g) == hash(graph_of(OrderedMap(dict(g.definitions))))


# ---------------------------------------------------------------- OrderedMap's own rules

def test_the_container_refuses_a_duplicate_key_itself():
    '''Not only `OrderedMap`. A repeated key is the one thing `OrderedMap` cannot represent
    honestly -- `__iter__` would yield the name twice while `__getitem__` returned the
    last value -- so the type defends it, and `NestingGraph`, which only checks the type, cannot
    be handed a dishonest one.'''
    with pytest.raises(ValueError, match='duplicate key'):
        OrderedMap([(N('a'), 1), (N('a'), 2)])
    with pytest.raises(ValueError, match='duplicate key'):
        graph_of(OrderedMap([(N('a'), call([], ['x'])), (N('a'), call([], ['y']))]))


def test_a_round_trip_takes_the_trusted_door():
    '''The refusal is on `__init__`, so a round trip does not re-pay it.'''
    m = OrderedMap({N('b'): 2, N('a'): 1})
    leaves, treedef = jax.tree_util.tree_flatten(m)
    assert jax.tree_util.tree_unflatten(treedef, leaves) == m


def test_the_views_read_the_tuples_and_not_the_index():
    '''`Mapping`'s mixins reach every value through `__getitem__`, which builds `_index`
    and pays a hash lookup per entry. Both tuples are already here.'''
    m = OrderedMap({N('b'): 2, N('a'): 1})
    assert m.keys()   == (N('b'), N('a'))
    assert m.values() == (2, 1)
    assert m.items()  == ((N('b'), 2), (N('a'), 1))
    assert '_index' not in m.__dict__          # nothing looked a key up to answer those
    assert m[N('a')] == 1 and '_index' in m.__dict__
