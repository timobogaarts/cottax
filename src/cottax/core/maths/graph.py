'''
The graph: node definitions bound to names, and the nesting over them (`within`, inner ->
outer). Stored, and read as data. Nothing checked at definition except what is implied through its types.
Its methods do not assume more than this either. 

For additional facts which need guarantees, use a `View`: facts read off one graph alone and cached on it.
'''

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Self
from dataclasses import replace
from functools import cached_property

from ..frozen import Frozen, OrderedMap, Violation, checked, first, remembered
from .definition import NodeDefinition
from .places import Label, Place

# --------------------------------------------------------------- contract & subgraph checks.
type Box[N, V] = Callable[[tuple[N, ...], tuple[V, ...], tuple[V, ...]], NodeDefinition[V]]
'''`(members, reads, owns) -> node`: the node standing for a block, in `Graph.contracted`.'''


def check_kept_bound[N: Label](graph: 'Graph', keep: Iterable[N]) -> Iterator[Violation]:
    '''Every node kept is bound, in the order given.'''
    for n in dict.fromkeys(keep):
        if n not in graph:
            yield NotBound(n)


def check_blocks_partition[N: Label](graph: 'Graph', blocks: Mapping[N, Sequence[N]], box: Box) -> Iterator[Violation]:
    '''Each block is at least two nodes, and no node is in two.'''
    seen, twice = set(), {}
    for block in blocks.values():
        if len(set(block)) < 2:
            yield BlockTooSmall(tuple(block))
        for n in dict.fromkeys(block):
            if n in seen:
                twice[n] = None
            seen.add(n)
    if twice:
        yield NodeInTwoBlocks(tuple(twice))


def check_blocks_bound[N: Label](graph: 'Graph', blocks: Mapping[N, Sequence[N]], box: Box) -> Iterator[Violation]:
    '''Every node of a block is bound.'''
    for n in dict.fromkeys(n for block in blocks.values() for n in block):
        if n not in graph:
            yield NotBound(n)


def check_box_names_free[N: Label](graph: 'Graph', blocks: Mapping[N, Sequence[N]], box: Box) -> Iterator[Violation]:
    '''A box's name is new, or one of its own block's: never another node's.'''
    for name, block in blocks.items():
        if name in graph and name not in block:
            yield NameTaken(name)


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
    def owners(self) -> dict[V, tuple[N, ...]]:
        '''
        Which V is owned by which N (multiple, on a graph with two producers). One owner per
        place is `ValidDependencies`'; `Dependencies.of(graph).owner` the one.
        '''
        out : dict[V, list[N]] = {}
        for n, d in self.definitions.items():
            for v in d.owns:
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
            first((NotBound(name),))

    __iter__ = None
    '''Not iterable: `nodes` is the sequence and `definitions` the mapping.'''

    def __contains__(self, name: N) -> bool:
        '''Whether a node is in the graph'''
        return name in self.definitions

    # ------------------------------------------------------------ projections    

    @cached_property
    def _position(self) -> dict[N, int]:
        '''Each node's place in the binding order: what `subgraph` sorts by.'''
        return {n: i for i, n in enumerate(self.definitions)}

    @checked(check_kept_bound)
    def subgraph(self, keep: Iterable[N]) -> Self:
        '''
        Only the nodes in `keep`, with their definitions, in the parent's binding order and the nesting
        pairs with both ends kept.
        '''
        kept = set(keep)
        order = sorted(kept, key=self._position.__getitem__)
        return replace(self, definitions=OrderedMap((n, self.definitions[n]) for n in order),
                       within=OrderedMap((i, o) for i, o in self.within.items()
                                         if i in kept and o in kept))

    @checked(check_blocks_partition, check_blocks_bound, check_box_names_free)
    def boxed(self, blocks: Mapping[N, Sequence[N]], box: Box[N, V]
              ) -> OrderedMap[N, tuple[tuple[N, ...], NodeDefinition[V]]]:
        '''Per box name: its members in binding order, and the node `box` builds for them.
        What `contracted` and `Contract` share.'''
        out = {}
        for name, block in blocks.items():
            order = tuple(sorted(dict.fromkeys(block), key=self._position.__getitem__))
            owns  = tuple(dict.fromkeys(v for m in order for v in self[m].owns))
            owned = set(owns)
            reads = tuple(dict.fromkeys(v for m in order for v in self[m].reads if v not in owned))
            out[name] = (order, box(order, reads, owns))
        return OrderedMap(out)

    def contracted(self, blocks: Mapping[N, Sequence[N]], box: Box[N, V]) -> Self:
        '''
        Coarsening of the graph: contracting blocks into a box.

        Args:
            blocks: box name -> its members (at least two; no node in two blocks). A name is
                new or one of its own members'.
            box: `(members, reads, owns) -> node`, the node standing for a block.

        Returns:
            The graph with each box reading what its block reads from outside and owning
            what it owns: at its name's place where it is named after a member, appended
            where its name is new (as `Contract` places it). Nesting pairs touching a block
            are dropped.

        Refuses:
            BlockTooSmall, NodeInTwoBlocks, NotBound, NameTaken (`boxed`'s, @checked)        
        '''
        boxes   = self.boxed(blocks, box)
        members = {m for order, _ in boxes.values() for m in order}
        kept    = [(n, boxes[n][1] if n in boxes else d) for n, d in self.definitions.items()
                   if n in boxes or n not in members]
        return replace(self,
                       definitions=OrderedMap(kept + [(n, b) for n, (_, b) in boxes.items() if n not in self]),
                       within=OrderedMap((i, o) for i, o in self.within.items()
                                         if i not in members and o not in members))


# ---------------------------------------------------------------- views


class View[N: Label, V: Place](Frozen):
    '''
    Pure functions of the graph that depend on a particular validity check
    the base graph does not know about. The views (and their own cached properties) are cached on the graph.

        class Dependencies(View):
            checks = (ValidDependencies.check,)

        Dependencies.of(graph).owner
    
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


# ----------------------------------------------------------------- violation finals

class NotBound(Violation):
    '''A name looked up, removed or redefined that the graph does not bind.'''
    name : Label
    error    = KeyError
    template = 'not a node of this graph: {name!r}'


class NameTaken(Violation):
    '''A new binding at a name the graph already binds.'''
    name : Label
    error    = ValueError
    template = 'node name {name!r} is already bound -- a new binding does not replace one'


class BlockTooSmall(Violation):
    block : tuple[Label, ...]
    error    = ValueError
    template = 'block {block} is not two nodes -- a box contracts at least two into one'


class NodeInTwoBlocks(Violation):
    nodes : tuple[Label, ...]
    error    = ValueError
    template = '{nodes} each in two blocks -- a node is contracted into one box at most'
