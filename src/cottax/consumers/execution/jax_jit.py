'''
`JitGraph`: a callable graph that can be a jit cache key. `Traced`: the run traced once by
jax, nothing computed: the graph's jaxpr and every place's shape.
'''

import dataclasses
import functools
import types
from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
from jax.tree_util import tree_leaves

from ...core import GraphWithValues, Label, NodeDefinition, Place, Violation
from ...core.frozen import Frozen, OrderedMap
from ...core.realisation import is_implemented
from .run import CallableGraph, blamed, check_callable, run
from .refusals import BodyByIdentity, PartialBody, TraceFailed, Unhashable


def check_hashable[N: Label, V: Place](name: N, node: NodeDefinition[V]
                                       ) -> Iterator[Violation]:
    '''
    The node hashes. An array in the graph does not, and is refused: opaque to every
    query, it is an input the graph does not name. The leaves that fail are named.
    '''
    try:
        hash(node)
    except TypeError:
        bad = list(_unhashable(node))
        yield Unhashable(name, type(node).__name__, len(bad), ', '.join(sorted(set(bad))))


def _unhashable(x: object) -> Iterator[str]:
    '''The kind of every leaf under `x` that does not hash: a dataclass (a node, its
    driver, its implementation) walked by its fields, anything else by its pytree leaves.'''
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            yield from _unhashable(getattr(x, f.name))
        return
    for leaf in tree_leaves(x):
        if leaf is not x and dataclasses.is_dataclass(leaf) and not isinstance(leaf, type):
            yield from _unhashable(leaf)
            continue
        try:
            hash(leaf)
        except TypeError:
            yield type(leaf).__name__


def check_reassembles[N: Label, V: Place](name: N, node: NodeDefinition[V]
                                          ) -> Iterator[Violation]:
    '''
    The implementation compares by value, so the same graph built twice is one cache
    key. A bound method compares its receiver by identity, and a `functools.partial`
    (`jax.tree_util.Partial` is one) defines no `__eq__`.
    '''
    if not is_implemented(node):
        return
    implementation = node.implementation
    if isinstance(implementation, types.MethodType):
        yield BodyByIdentity(name, implementation)
    if isinstance(implementation, functools.partial):
        yield PartialBody(name, implementation)


class JitGraph[N: Label, V: Place](CallableGraph[N, V]):
    '''A callable graph whose every node is a jit cache key.'''

    checks_node = (check_hashable, check_reassembles)


# ============================================================================ traced

def _shaped(leaf: Any) -> Any:
    '''An array leaf as its shape and dtype; any other leaf as it is.'''
    if isinstance(leaf, (jax.Array, np.ndarray, np.generic)):
        return jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype)
    return leaf


def check_traced_callable(traced: 'Traced') -> Iterator[Violation]:
    yield from check_callable(traced.valued)


def check_traces(traced: 'Traced') -> Iterator[Violation]:
    '''The run traces: the first refusal it raised, or the node whose body failed.'''
    try:
        traced._traced
    except Exception as e:                                      # noqa: BLE001  any trace failure
        refused = getattr(e, 'violation', None)
        yield refused if refused is not None else TraceFailed(blamed(e), f'{type(e).__name__}: {e}')


class Traced[N: Label, V: Place](Frozen):
    '''
    The run traced once by jax, nothing computed: the whole graph as one jaxpr, and every
    place's shape. Refused: the first refusal the run raises (a body's miscount,
    `ReturnedUnlike`), or a body failing at these shapes (`TraceFailed`), by node. A value
    handed in may be an array or a `jax.ShapeDtypeStruct`; any other value reaches the bodies
    as it is, so a setting stays a setting.

        traced = Traced(GraphWithValues(graph, values))
        traced.shapes, traced.jaxpr
    '''

    valued : GraphWithValues[CallableGraph[N, V], V]

    checks = (check_traced_callable, check_traces)

    @functools.cached_property
    def _traced(self) -> tuple[Any, OrderedMap[V, Any]]:
        values = OrderedMap.trusted(tuple(self.valued.values),
                                    (jax.tree.map(_shaped, v) for v in self.valued.values.values()))
        leaves, tree = jax.tree.flatten(values)
        at = [i for i, leaf in enumerate(leaves) if isinstance(leaf, jax.ShapeDtypeStruct)]

        def traced(*shapes: Any) -> OrderedMap[V, Any]:
            filled = list(leaves)
            for i, shape in zip(at, shapes):
                filled[i] = shape
            return run(GraphWithValues(self.valued.graph, jax.tree.unflatten(tree, filled)))

        return jax.make_jaxpr(traced, return_shape=True)(*(leaves[i] for i in at))

    @property
    def jaxpr(self) -> Any:
        '''The whole graph as one closed jaxpr, its inputs the shapes handed in.'''
        return self._traced[0]

    @property
    def shapes(self) -> OrderedMap[V, Any]:
        '''Every place's shape and dtype (a pytree of them where its value is one): the inputs,
        then what the graph computes.'''
        return self._traced[1]
