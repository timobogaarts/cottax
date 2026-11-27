'''
Primitive Ops: bindings, nesting, replacing, renaming, cutting, and (node) attributes.
'''

import dataclasses
from collections.abc import Collection, Iterator, Sequence

from ..maths.definition import Attribute, NodeDefinition
from ..frozen import AbstractClassVar, OrderedMap, Violation, first, name_of
from ..maths.dependencies import Dependencies
from ..maths.graph import Graph, NotBound
from ..maths.places import Label, Place, PlaceToPlace
from ..maths.kinds import FunctionalNode
from ..maths.valid import Valid
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
    '''Remove nodes. A nesting pair losing one end is left, a pair with both nodes deleted removes the nesting.'''
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

# ---------------------------------------------------------------- a definition replaced
class Replace[N: Label, V: Place](PrimitiveOp[N, V]):
    node       : N
    definition : NodeDefinition[V]
    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        return Edit(redefine=OrderedMap({self.node: self.definition}))
    def __repr__(self) -> str:
        return f'replace({self.node})'

# ---------------------------------------------------------------- renaming

def check_renames_places_the_node_has(op: 'Rename', graph: Graph) -> Iterator[Violation]:
    '''A node only renames places it reads or owns.'''
    node = graph[op.node]
    has  = set(node.reads) | set(node.owns)
    if missing := tuple(old for old, _ in op.onto if old not in has):
        yield NotRenamed(op.node, missing)


class Rename[N: Label, V: Place](PrimitiveOp[N, V]):
    """One node's places renamed by a mapping."""
    node : N
    onto : tuple[tuple[V, V], ...]
    """Old place -> the place to use instead. Pairs, since an op is a record and hashes."""
    checks_graph = (check_renames_places_the_node_has,)

    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        return Edit(redefine=OrderedMap({self.node: graph[self.node].relabel(dict(self.onto))}))
    def __repr__(self) -> str:
        return (f'rename({self.node}: '
                + ', '.join(f'{o}->{n}' for o, n in self.onto) + ')')
# ---------------------------------------------------------------- Cutting
def check_readers(op: 'Cut') -> Iterator[Violation]:
    '''A cut has readers; with none, nothing would read the copy.'''
    if not op.readers:
        yield NoReaders(op.var, op.copy)

def check_cuts_a_produced_place(op: 'Cut', graph: Graph) -> Iterator[Violation]:
    '''`var` is produced (a boundary input has nothing to close the cut against), and its
    copy is not in use (a second cut would add readers to the first copy).'''
    owners = Dependencies.of(graph).owner
    if op.var not in owners:
        yield NotProduced(op.var)
    elif op.copy in owners or op.copy in graph.readers:
        yield AlreadyCut(op.var, op.copy)

def check_readers_read_var(op: 'Cut', graph: Graph) -> Iterator[Violation]:
    '''Every reader reads `var`. Renaming on the producer would move the output instead.'''
    if off := tuple(r for r in op.readers if op.var not in graph[r].reads):
        yield DoesNotRead(op.var, off)

class Cut[N: Label, V: Place](PrimitiveOp[N, V]):
    '''Make `readers` read a copy of `var`. The producer and every other reader are
    untouched.'''

    var     : V
    readers : tuple[N, ...]
    copy_of : PlaceToPlace[V]
    needs        = Valid            # its checks read who produces `var`
    checks       = (check_readers,)
    checks_graph = (check_cuts_a_produced_place, check_readers_read_var)

    @property
    def copy(self) -> V:
        '''The boundary variable `readers` end up reading.'''
        return self.copy_of(self.var)
    
    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        return Edit(redefine=OrderedMap(
            {r: graph[r].relabel({self.var: self.copy}) for r in self.readers}))

    def __repr__(self) -> str:
        return f'cut({self.var}->{self.copy})'

# ---------------------------------------------------------------- Attributes

def check_sets_something(op: 'SetAttribute') -> Iterator[Violation]:
    if not op.values:
        yield NoAttributeNamed()

def check_settable(op: 'SetAttribute', graph: Graph) -> Iterator[Violation]:
    """No node carries one under that key already."""
    for node_name in op.values:
        if op.kind in graph[node_name].attributes:
            yield AttributeAlreadySet(node_name, op.kind.__name__)


class SetAttribute[N: Label, V: Place](PrimitiveOp[N, V]):
    """Set attribute kind for a set of nodes to some values."""

    kind   : AbstractClassVar[Attribute]
    values : OrderedMap[N, object]
    checks = (check_sets_something,)
    checks_graph = (check_settable,)

    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        return Edit(redefine=OrderedMap(
            (name, dataclasses.replace(node := graph[name], attributes=node.attributes | {self.kind: value}))
            for name, value in self.values.items()))
    
    def __repr__(self) -> str:
        return (f'{type(self).__name__.lower()}('
                + ', '.join(f'{n} <- {name_of(v)}' for n, v in self.values.items()) + ')')


def check_removes_something(op: 'RemoveAttribute') -> Iterator[Violation]:
    if not op.nodes:
        yield NoAttributeNamed()


def check_removable(op: 'RemoveAttribute', graph: Graph) -> Iterator[Violation]:
    """Each node carries one to take off."""
    for name in op.nodes:
        if op.kind not in graph[name].attributes:
            yield AttributeNotSet(name, op.kind.__name__)

class RemoveAttribute[N: Label, V: Place](PrimitiveOp[N, V]):
    """The inverse of `SetAttribute`."""

    kind  : AbstractClassVar[Attribute]
    nodes : tuple[N, ...]

    checks = (check_removes_something,)
    checks_graph = (check_removable,)

    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        return Edit(redefine=OrderedMap(
            (name, dataclasses.replace(node := graph[name], attributes=OrderedMap(
                (k, a) for k, a in node.attributes.items() if k is not self.kind)))
            for name in self.nodes))

    def __repr__(self) -> str:
        return f'{type(self).__name__.lower()}({", ".join(map(str, self.nodes))})'

# ---------------------------------------------------------------- contraction

def blocks_partition[N: Label](blocks: Sequence[Sequence[N]]) -> Iterator[Violation]:
    '''Each block is at least two nodes, and no node is in two.'''
    seen, twice = set(), {}
    for block in blocks:
        if len(set(block)) < 2:
            yield BlockTooSmall(tuple(block))
        for n in dict.fromkeys(block):
            if n in seen:
                twice[n] = None
            seen.add(n)
    if twice:
        yield NodeInTwoBlocks(tuple(twice))


def blocks_bound[N: Label](graph: Graph[N, Place], blocks: Sequence[Sequence[N]]) -> Iterator[Violation]:
    '''Every node of a block is bound.'''
    for n in dict.fromkeys(n for block in blocks for n in block):
        if n not in graph:
            yield NotBound(n)


def blocks_own_something[N: Label](graph: Graph[N, Place], blocks: Sequence[Sequence[N]]) -> Iterator[Violation]:
    '''A block contracts into a computation, so it owns something.'''
    for block in blocks:
        if not any(graph[n].owns for n in block):
            yield BoxOwnsNothing(tuple(block))


def contraction[N: Label, V: Place](graph: Graph[N, V], blocks: Sequence[Sequence[N]]) -> Edit[N, V]:
    '''
    The edit contracting each block of at least two nodes into one `FunctionalNode` at its
    first member's place in binding order: reading what the block reads from outside, owning
    what it owns, carrying nothing; the other members gone. A pair with an end in a block is
    unnested, since a box encloses nothing. Refuses (raises) blocks that overlap, are too
    small, name unbound nodes, or own nothing.
    '''
    first(blocks_partition(blocks))
    first(blocks_bound(graph, blocks))
    first(blocks_own_something(graph, blocks))
    position = {n: i for i, n in enumerate(graph.nodes)}
    orders   = [tuple(sorted(dict.fromkeys(b), key=position.__getitem__)) for b in blocks]
    members  = {n for order in orders for n in order}
    boxes    = {}
    for order in orders:
        owns  = tuple(dict.fromkeys(v for m in order for v in graph[m].owns))
        owned = set(owns)
        reads = tuple(dict.fromkeys(v for m in order for v in graph[m].reads if v not in owned))
        boxes[order[0]] = FunctionalNode(reads, owns)
    return Edit(unnest=frozenset(i for i, o in graph.within.items() if i in members or o in members),
                remove=frozenset(members - set(boxes)),
                redefine=OrderedMap(boxes))


def contracted[G: Graph](graph: G, blocks: Sequence[Sequence[Label]]) -> G:
    '''`graph` with each block contracted into one box (`contraction`), proved `Valid`. A
    coarsening: a path through a box may appear where none was.'''
    result = contraction(graph, blocks).applied(graph)
    Valid.prove(result)
    return result


# ----------------------------------------------------------------- violation finals

class NoAttributeNamed(Violation):
    error    = ValueError
    template = 'names no node, so there is no attribute to set or take off'

class AttributeAlreadySet(Violation):
    node      : Label
    attribute : str
    error    = TypeError
    template = ('{node!r} already carries a {attribute} -- replacing one with another in silence '
                'is not an edit; take it off first')

class AttributeNotSet(Violation):
    node      : Label
    attribute : str
    error    = TypeError
    template = '{node!r} carries no {attribute}, so there is nothing to take off'

class NoReaders(Violation):
    var  : Place
    copy : Place
    error    = ValueError
    template = 'cut({var}): no readers, so {copy!r} would be a name nothing reads'

class NotRenamed(Violation):
    node   : Label
    places : tuple[Place, ...]
    error    = KeyError
    template = '{node!r} neither reads nor owns {places!r}'

class NotProduced(Violation):
    var : Place
    error    = KeyError
    template = '{var!r} is not computed here, so there is nothing to cut'

class DoesNotRead(Violation):
    var   : Place
    nodes : tuple[Label, ...]
    error    = ValueError
    template = ('cut({var}): {nodes} do not read {var!r}, so there is no read to move onto the '
                'copy -- a cut\'s readers read the variable, and its producer stays as it is')

class AlreadyCut(Violation):
    var  : Place
    copy : Place
    error    = ValueError
    template = ('{var!r} is already cut: its copy {copy} exists. A variable is cut at '
                'most once -- two cuts would assume two values for one place. Cut with '
                'more readers at once to share a copy')


class BlockTooSmall(Violation):
    block : tuple[Label, ...]
    error    = ValueError
    template = 'block {block} is not two nodes -- a box contracts at least two into one'


class NodeInTwoBlocks(Violation):
    nodes : tuple[Label, ...]
    error    = ValueError
    template = '{nodes} each in two blocks -- a node is contracted into one box at most'


class BoxOwnsNothing(Violation):
    block : tuple[Label, ...]
    error    = ValueError
    template = ('block {block} owns nothing, so as one node it would compute nothing and be '
                 'unreachable -- a box is a computation, so contract only what produces')
