'''
The ops on the bindings and the nesting: `Insert`, `Delete`, `Nest`, `Unnest`.
'''

from collections.abc import Collection

from ..frozen import OrderedMap
from ..maths.definition import NodeDefinition
from ..maths.graph import Graph
from ..maths.places import Label, Place
from .op import PrimitiveOp
from .edit import Edit

def nested_among[N: Label](graph: Graph[N, Place], names: Collection[N]) -> frozenset[N]:
    '''The inners of the pairs with both ends in `names`: what removing them all unnests.'''
    return frozenset(i for i, o in graph.within.items() if i in names and o in names)
# ---------------------------------------------------------------- the bindings

class Insert[N: Label, V: Place](PrimitiveOp[N, V]):
    '''Add nodes to the graph (Edit refuses already present name)'''
    nodes : OrderedMap[N, NodeDefinition[V]]
    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        return Edit(insert=self.nodes)
    def __repr__(self) -> str:  
        return f'insert({", ".join(str(n) for n in self.nodes)})'
    
class Delete[N: Label](PrimitiveOp[N, Place]):
    '''Remove nodes. A nesting pair losing one end is left, both nodes of the pair deleted removes the nesting as well'''

    nodes : tuple[N, ...]
    def _edit(self, graph: Graph[N, Place]) -> Edit[N, Place]:
        '''A pair losing both ends goes with them.'''
        gone = frozenset(self.nodes)
        return Edit(unnest=nested_among(graph, gone), remove=gone)
    def __repr__(self) -> str:
        return f'delete({", ".join(str(n) for n in self.nodes)})'
# ---------------------------------------------------------------- the nesting

class Nest[N: Label](PrimitiveOp[N, Place]):
    '''Record that one problem is answered inside another's iteration.'''
    inner : N
    outer : N
    def _edit(self, graph: Graph[N, Place]) -> Edit[N, Place]:
        return Edit(nest=OrderedMap({self.inner: self.outer}))
    def __repr__(self) -> str:
        return f'nest({self.inner} in {self.outer})'

class Unnest[N: Label](PrimitiveOp[N, Place]):
    '''The inverse of `Nest`.'''
    inner : N
    def _edit(self, graph: Graph[N, Place]) -> Edit[N, Place]:
        return Edit(unnest=frozenset({self.inner}))
    def __repr__(self) -> str:
        return f'unnest({self.inner})'

