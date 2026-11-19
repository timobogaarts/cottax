'''
Ops on `Redefine`: what a node reads and owns renamed (`Rewire`, `Rename`, `Cut`), a
definition replaced (`Replace`), what a condition node owns inside set (`SetOwnsInside`), and
what a node carries set and removed (`SetAttribute`, `RemoveAttribute`); and blocks
contracted into boxes (`Contract`, composed of `Unnest`, `Delete` and `Replace`).
'''

from abc import abstractmethod
import dataclasses
from collections.abc import Iterator, Mapping, Sequence

from ..maths.definition import (
    Attribute, NodeDefinition, Renaming,
)
from ..frozen import AbstractClassVar, OrderedMap, Violation, name_of
from ..maths.dependencies import Dependencies
from ..maths.graph import Graph
from ..maths.places import Label, Place, PlaceToPlace
from ..maths.kinds import FunctionalNode
from .algebra import ComposedOp, GraphOp
from .primitive import Delete, NotBound, Redefine, Unnest


type Relabel[N: Label, V: Place] = Mapping[N, Renaming[V]]
'''Per node, old place -> new. Keyed by place, so `fn(x, x)` renames both.'''


def check_renames_places_the_node_has(op: 'Rewire', graph: Graph) -> Iterator[Violation]:
    '''A node only renames places it reads or owns.'''
    for name, m in op.renaming(graph).items():
        node = graph[name]
        has  = set(node.reads) | set(node.owns)
        if missing := tuple(v for v in m if v not in has):
            yield NotRenamed(name, missing)


class Rewire[N: Label, V: Place](Redefine[N, V]):
    """Rename places a node reads and owns; nothing else of it changes."""

    checks_graph = (check_renames_places_the_node_has,)

    @abstractmethod
    def renaming(self, graph: Graph[N, V]) -> Relabel[N, V]:
        '''Per node, old place -> new. One map serves reads and owns: a node never has a
        place in both.'''

    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {name: graph[name].relabel(m)
                for name, m in self.renaming(graph).items()}

class Replace[N: Label, V: Place](Redefine[N, V]):
    """One binding given a definition the caller already holds, attributes and all: who
    hands over a new definition says what it carries."""

    node       : N
    definition : NodeDefinition[V]

    edits_attributes = True

    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.node: self.definition}

    def __repr__(self) -> str:
        return f'replace({self.node})'


class Rename[N: Label, V: Place](Rewire[N, V]):
    """One node's places renamed by a mapping the caller already holds."""

    node : N
    onto : tuple[tuple[V, V], ...]
    """Old place -> the place to use instead. Pairs, since an op is a record and hashes."""

    def renaming(self, graph: Graph[N, V]) -> Relabel[N, V]:
        return {self.node: dict(self.onto)}

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
    if off := tuple(r for r in op.readers if r in graph and op.var not in graph[r].reads):
        yield DoesNotRead(op.var, off)


class Cut[N: Label, V: Place](Rewire[N, V]):
    '''Make `readers` read a copy of `var`. The producer and every other reader are
    untouched.'''

    var     : V
    readers : tuple[N, ...]
    copy_of : PlaceToPlace[V]
    '''Derives the copy's place from `var`.'''

    checks = (check_readers,)
    checks_graph = (check_cuts_a_produced_place, check_readers_read_var)

    @property
    def copy(self) -> V:
        '''The boundary variable `readers` end up reading.'''
        return self.copy_of(self.var)

    def renaming(self, graph: Graph[N, V]) -> Relabel[N, V]:
        return {r: {self.var: self.copy} for r in self.readers}

    def __repr__(self) -> str:
        return f'cut({self.var}->{self.copy})'


# ---------------------------------------------------------------- what a node carries

def check_sets_something(op: 'SetAttribute') -> Iterator[Violation]:
    if not op.values:
        yield NoAttributeNamed()


def check_settable(op: 'SetAttribute', graph: Graph) -> Iterator[Violation]:
    """No node carries one under that key already. Whether it can carry what it is given is
    the attribute's own layer's rule, which that layer's op adds (`Implement`)."""
    for name in op.values:
        if op.kind in graph[name].attributes:
            yield AttributeAlreadySet(name, op.kind.__name__)


class SetAttribute[N: Label, V: Place](Redefine[N, V]):
    """Each named node carries the value given for it under `kind`; nothing else of it
    changes. An attribute's own op is this with its `kind` named: `Implement`."""

    kind   : AbstractClassVar[Attribute]
    values : OrderedMap[N, object]

    edits_attributes = True

    checks = (check_sets_something,)
    checks_graph = (check_settable,)

    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {name: dataclasses.replace(node := graph[name], attributes=OrderedMap({**node.attributes, self.kind: value}))
                for name, value in self.values.items()}

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


class RemoveAttribute[N: Label, V: Place](Redefine[N, V]):
    """The inverse of `SetAttribute`: what each node carries under `kind` taken off."""

    kind  : AbstractClassVar[Attribute]
    nodes : tuple[N, ...]

    edits_attributes = True

    checks = (check_removes_something,)
    checks_graph = (check_removable,)

    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {name: dataclasses.replace(node := graph[name], attributes=OrderedMap(
                    (k, a) for k, a in node.attributes.items() if k is not self.kind))
                for name in self.nodes}

    def __repr__(self) -> str:
        return f'{type(self).__name__.lower()}({", ".join(map(str, self.nodes))})'

# ---------------------------------------------------------------- contraction

def check_blocks_partition(op: 'Contract') -> Iterator[Violation]:
    '''Each block is at least two nodes, and no node is in two.'''
    seen, twice = set(), {}
    for block in op.blocks:
        if len(set(block)) < 2:
            yield BlockTooSmall(tuple(block))
        for n in dict.fromkeys(block):
            if n in seen:
                twice[n] = None
            seen.add(n)
    if twice:
        yield NodeInTwoBlocks(tuple(twice))


def check_blocks_bound(op: 'Contract', graph: Graph) -> Iterator[Violation]:
    '''Every node of a block is bound.'''
    for n in dict.fromkeys(n for block in op.blocks for n in block):
        if n not in graph:
            yield NotBound(n)


def check_blocks_own_something(op: 'Contract', graph: Graph) -> Iterator[Violation]:
    '''A block contracts into a computation, so it owns something.'''
    for block in op.blocks:
        if all(n in graph for n in block) and not any(graph[n].owns for n in block):
            yield BoxOwnsNothing(tuple(block))


class Contract[N: Label, V: Place](ComposedOp[N, V]):
    '''
    Each block of at least two nodes contracted into one `FunctionalNode` at its first
    member's place in binding order: reading what the block reads from outside, owning
    what it owns; the other members gone. A pair with an end in a block is unnested, since
    a box encloses nothing. A coarsening: a path through a box may appear where none was.
    '''

    blocks : tuple[tuple[N, ...], ...]

    checks       = (check_blocks_partition,)
    checks_graph = (check_blocks_bound, check_blocks_own_something)

    def composed(self, graph: Graph[N, V]) -> Sequence[GraphOp[N, V]]:
        position = graph._position
        orders   = [tuple(sorted(dict.fromkeys(b), key=position.__getitem__)) for b in self.blocks]
        members  = {n for order in orders for n in order}
        boxes    = {}
        for order in orders:
            owns  = tuple(dict.fromkeys(v for m in order for v in graph[m].owns))
            owned = set(owns)
            reads = tuple(dict.fromkeys(v for m in order for v in graph[m].reads if v not in owned))
            boxes[order[0]] = FunctionalNode(reads, owns)
        ops : list[GraphOp[N, V]] = [Unnest(i) for i, o in graph.within.items()
                                     if i in members or o in members]
        if gone := tuple(n for n in graph.nodes if n in members and n not in boxes):
            ops.append(Delete(gone))
        ops += [Replace(n, box) for n, box in boxes.items()]
        return ops

    def __repr__(self) -> str:
        return 'contract(' + ', '.join('{' + ', '.join(map(str, b)) + '}' for b in self.blocks) + ')'


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
