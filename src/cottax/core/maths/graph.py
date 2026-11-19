'''
The graph: node definitions bound to names, and the nesting over them (`within`, inner ->
outer). Stored, and read as data. Nothing checked at definition except what is implied through its types.
Its methods do not assume more than this either. 

For additional facts which need guarantees, use a `View`: facts read off one graph alone and cached on it.
'''

from collections.abc import Iterable
from typing import Self
from functools import cached_property

from ..frozen import Frozen, OrderedMap, remembered
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

    @cached_property
    def boundary_inputs(self) -> tuple[V, ...]:
        '''Variables read but owned by no node: what the graph is handed.'''
        owned = set(self.owned_variables)
        return tuple(var for var in self.read_variables if var not in owned)

    @cached_property
    def boundary_outputs(self) -> tuple[V, ...]:
        '''Variables owned by a node but never read.'''
        read = set(self.read_variables)
        return tuple(var for var in self.owned_variables if var not in read)

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


# ---------------------------------------------------------------- views

class View[N: Label, V: Place](Frozen):
    '''
    Pure functions of the graph that depend on a particular validity check
    the base graph does not know about. The views (and their own cached properties) are cached on the graph.

        class Dependencies(View):
            checks = (ValidDependencies.check,)

        Dependencies.of(graph).owners
    
    '''

    graph : Graph[N, V]

    @classmethod
    def of(cls, graph: Graph[N, V]) -> Self:
        '''The view of `graph`: built the first time, then the same one.'''
        return _view(graph, cls)


@remembered
def _view(graph: Graph, cls: type[View]) -> View:
    '''One view of each kind per graph: the only place a view is constructed.'''
    return cls(graph)
