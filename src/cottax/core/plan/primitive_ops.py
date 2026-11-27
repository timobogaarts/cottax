'''
Primitive Ops: bindings, nesting, replacing, renaming, cutting, (node) attributes, and Contraction.
'''

import dataclasses
from collections.abc import Collection, Iterator

from ..maths.definition import Attribute, NodeDefinition
from ..frozen import AbstractClassVar, OrderedMap, Violation, name_of
from ..maths.dependencies import Dependencies
from ..maths.graph import Box, Graph, NotBound
from ..maths.places import Label, Place, PlaceToPlace
from ..maths.valid import Valid
from .op import PrimitiveOp
from .edit import Edit


def nested_among[N: Label](graph: Graph[N, Place], names: Collection[N]) -> frozenset[N]:
    '''The inners of the pairs with both ends in `names`: what removing them all unnests.'''
    return frozenset(i for i, o in graph.within.items() if i in names and o in names)


def stranded[N: Label](graph: Graph[N, Place], removed: Collection[N]) -> Iterator[Violation]:
    '''A nesting pair losing exactly one end to `removed`. A nesting is a decision the plan
    recorded, so it is not dropped silently: `Unnest` first. Losing both ends is fine.'''
    if broken := [(i, o) for i, o in graph.within.items() if (i in removed) != (o in removed)]:
        yield Stranded(tuple(i for i, _ in broken), tuple(o for _, o in broken))
# ---------------------------------------------------------------- the bindings

class Insert[N: Label, V: Place](PrimitiveOp[N, V]):
    '''Add nodes to the graph (Edit refuses already present name)'''
    nodes : OrderedMap[N, NodeDefinition[V]]
    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        return Edit(insert=self.nodes)
    def __repr__(self) -> str:  
        return f'insert({", ".join(str(n) for n in self.nodes)})'
    
def check_strands_no_pair(op: 'Delete', graph: Graph) -> Iterator[Violation]:
    '''No nesting pair loses one end and keeps the other.'''
    yield from stranded(graph, set(op.nodes))


class Delete[N: Label](PrimitiveOp[N, Place]):
    '''Remove nodes. A pair with both nodes deleted removes the nesting; one losing one end
    is refused (`Stranded`): `Unnest` it first.'''
    nodes : tuple[N, ...]
    checks_graph = (check_strands_no_pair,)
    def _edit(self, graph: Graph[N, Place]) -> Edit[N, Place]:
        '''A pair losing both ends goes with them.'''
        gone = frozenset(self.nodes)
        return Edit(unnest=nested_among(graph, gone), remove=gone)
    def __repr__(self) -> str:
        return f'delete({", ".join(str(n) for n in self.nodes)})'
# ---------------------------------------------------------------- the nesting

def check_ends_bound(op: 'Nest', graph: Graph) -> Iterator[Violation]:
    '''Both ends of the pair are nodes of the graph.'''
    for n in dict.fromkeys((op.inner, op.outer)):
        if n not in graph:
            yield NotBound(n)


class Nest[N: Label](PrimitiveOp[N, Place]):
    '''Record that one problem is answered inside another's iteration. A loop the pair
    would close is `Valid`'s to refuse, where it is kept.'''
    inner : N
    outer : N
    checks_graph = (check_ends_bound,)
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

# ---------------------------------------------------------------- contraction

class Contract[N: Label, V: Place](PrimitiveOp[N, V]):
    '''
    Blocks contracted into boxes, as a step of a plan: `Graph.contracted` as an edit.
    '''
    blocks : OrderedMap[N, tuple[N, ...]]
    box    : Box[N, V]

    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        boxes   = graph.boxed(self.blocks, self.box)
        members = {m for order, _ in boxes.values() for m in order}
        return Edit(unnest=frozenset(i for i, o in graph.within.items() if i in members or o in members),
                    remove=frozenset(members - set(boxes)),
                    redefine=OrderedMap((n, b) for n, (_, b) in boxes.items() if n in graph),
                    insert=OrderedMap((n, b) for n, (_, b) in boxes.items() if n not in graph))

    def __repr__(self) -> str:
        return 'contract(' + '; '.join(f'{n} <- {", ".join(map(str, ms))}'
                                       for n, ms in self.blocks.items()) + ')'


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
    if op.node not in graph:
        yield NotBound(op.node)
        return
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
    for r in op.readers:
        if r not in graph:
            yield NotBound(r)
    if off := tuple(r for r in op.readers if r in graph and op.var not in graph[r].reads):
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
        if node_name not in graph:
            yield NotBound(node_name)
        elif op.kind in graph[node_name].attributes:
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
        if name not in graph:
            yield NotBound(name)
        elif op.kind not in graph[name].attributes:
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


class Stranded(Violation):
    inner : tuple[Label, ...]
    outer : tuple[Label, ...]
    error    = ValueError
    template = ('{inner} in {outer} would lose one end and keep the other -- `Unnest` it first, '
                 'so a plan never quietly voids what an earlier op recorded')
