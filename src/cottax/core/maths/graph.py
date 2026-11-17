'''
The graph: node definitions bound to names, and the nesting over them (`within`, inner ->
outer). Stored, and read as data. Nothing checked at definition except what is implied through its types.
Its methods do not assume more than this either.
'''

from collections.abc import Iterable
from typing import Self
from functools import cached_property

from ..frozen import Frozen, OrderedMap
from .definition import NodeDefinition
from .places import Label, Place

# ---------------------------------------------------------------- the graph
class Graph[N: Label, V: Place](Frozen):
    '''
    Definitions bound to names, and a nesting decision over them: `within`, inner -> outer.
    '''

    definitions : OrderedMap[N, NodeDefinition[V]]
    within      : OrderedMap[N, N] = OrderedMap()

    # ------------------------------------------------------------ the bindings
    @cached_property
    def readers(self) -> dict[V, tuple[N, ...]]:
        '''
        Which V is read by which N (multiple)
        '''        
        out : dict[V, list[N]] = {}
        for n, d in self.definitions.items():
            for v in dict.fromkeys(d.reads):
                out.setdefault(v, []).append(n)
        return {v: tuple(ns) for v, ns in out.items()}

    @cached_property
    def nodes(self) -> tuple[N, ...]:
        '''Every node's name, in binding order.'''
        return tuple(self.definitions)
    
    @cached_property
    def variables(self) -> tuple[V, ...]:
        '''Every variable the graph names: owned by a node, or read from outside.'''
        return tuple(dict.fromkeys(self.owned_variables + self.read_variables))

    @cached_property
    def owned_variables(self) -> tuple[V, ...]:
        return tuple(var for name in self.nodes for var in self[name].owns)

    @cached_property
    def read_variables(self) -> tuple[V, ...]:
        return tuple(dict.fromkeys(var for name in self.nodes for var in self[name].reads))

    # ------------------------------------------------------------ resolution

    def __getitem__(self, name: N) -> NodeDefinition[V]:        
        try:
            return self.definitions[name]
        except KeyError:
            raise KeyError(f'not a node of this graph: {name!r}') from None

    __iter__ = None
    '''Not iterable: `nodes` is the sequence and `definitions` the mapping.'''

    def __contains__(self, name: N) -> bool:
        '''Whether a node is in the graph'''
        return name in self.definitions

    # ------------------------------------------------------------ the nesting

    # ------------------------------------------------------------ restriction
    # Every other edit is an op (`plan/`).

    @cached_property
    def _position(self) -> dict[N, int]:
        '''Each node's place in the binding order: what `subgraph` sorts by.'''
        return {n: i for i, n in enumerate(self.definitions)}

    def subgraph(self, keep: Iterable[N]) -> Self:
        '''
        Only the nodes in `keep`, with their definitions, in the parent's binding order and the nesting
        pairs with both ends kept.
        '''
        kept = set(keep)
        if unknown := [n for n in kept if n not in self._position]:
            raise KeyError(f'not node(s) of this graph: {unknown!r}')
        order = sorted(kept, key=self._position.__getitem__)
        return self._carried(definitions=OrderedMap((n, self.definitions[n]) for n in order),
                             within=OrderedMap((i, o) for i, o in self.within.items()
                                               if i in kept and o in kept))
