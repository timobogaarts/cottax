'''The graph constructors the tests share.'''

from typing import Any

from jax.tree_util import GetAttrKey, tree_leaves_with_path

from cottax.core.frozen import OrderedMap
from cottax.core.maths.graph import AbstractGraph
from cottax.core.maths.nesting import NestingGraph
from cottax.core import Function, NodeDefinition
from cottax.core.realisation import ImplementedFunction
from cottax.pytree.path import NodePath, VarPath
from cottax.relational import Eq, Min, Objective, Relation, RelationalCondition


def graph_of(definitions, within=()) -> NestingGraph:
    '''A nesting graph over its bindings and its pairs, each a mapping or pairs.'''
    return NestingGraph(AbstractGraph(OrderedMap(definitions)), OrderedMap(within))


def three_problems(*nesting: tuple[str, str]) -> NestingGraph:
    '''
    Three problems on one cycle, shaped as ASO: `p` minimises `f` over `z`, the fixed point
    `q` holds its copy `yh` of `y1`, and `r` minimises `f` over `x`. `a` reads `yh` and `b`
    is `r`'s discipline; both read `z`. `nesting` is `(inner, outer)` pairs by name. The
    drawings are pinned on four of its nestings: none; `r` in `q` alone; `q` and `r` in
    `p`, with nothing between them; `r` in `q` in `p`, which is executable.
    '''
    def V(k): return VarPath((GetAttrKey(k),))
    def N(k): return NodePath((GetAttrKey(k),))
    def call(reads, owns):
        return ImplementedFunction(Function(tuple(map(V, reads)), tuple(map(V, owns))),
                                   lambda *a: 0.0)
    return graph_of({
        N('p')  : RelationalCondition((), (Objective(V('f'), Min),), (V('z'),), ()),
        N('q')  : RelationalCondition((Relation(V('yh'), V('y1'), Eq),), (), (V('yh'),), ()),
        N('r')  : RelationalCondition((), (Objective(V('f'), Min),), (V('x'),), ()),
        N('a')  : call(['z', 'yh'], ['y2']),
        N('b')  : call(['z', 'x', 'y2'], ['y1']),
        N('obj'): call(['x', 'y1', 'y2'], ['f']),
    }, tuple((N(inner), N(outer)) for inner, outer in nesting))


NOTHING_NESTED = ()
'''`three_problems`' nestings: none.'''
UNORDERED_IN_P = (('q', 'p'), ('r', 'p'))
'''`q` and `r` in `p`, nothing between them.'''
NESTED = (('r', 'q'), ('q', 'p'))
'''`r` in `q` in `p`: executable.'''


def graph_from_pytree(pytree: Any) -> NestingGraph:
    '''A graph of the node definitions in a pytree, each named by its key path, in
    flatten order (a dict's keys sorted).'''
    def is_node(x: Any) -> bool:
        return isinstance(x, NodeDefinition)

    return graph_of((NodePath(keys), leaf)
                    for keys, leaf in tree_leaves_with_path(pytree, is_leaf=is_node)
                    if is_node(leaf))
