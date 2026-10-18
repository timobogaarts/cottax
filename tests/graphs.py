'''The graph constructors the tests share.'''

from typing import Any

from jax.tree_util import tree_leaves_with_path

from cottax.core.frozen import OrderedMap
from cottax.core.maths.graph import AbstractGraph
from cottax.core.maths.nesting import NestingGraph
from cottax.core import NodeDefinition
from cottax.pytree.path import NodePath


def graph_of(definitions, within=()) -> NestingGraph:
    '''A nesting graph over its bindings and its pairs, each a mapping or pairs.'''
    return NestingGraph(AbstractGraph(OrderedMap(definitions)), OrderedMap(within))


def graph_from_pytree(pytree: Any) -> NestingGraph:
    '''A graph of the node definitions in a pytree, each named by its key path, in
    flatten order (a dict's keys sorted).'''
    def is_node(x: Any) -> bool:
        return isinstance(x, NodeDefinition)

    return graph_of((NodePath(keys), leaf)
                    for keys, leaf in tree_leaves_with_path(pytree, is_leaf=is_node)
                    if is_node(leaf))
