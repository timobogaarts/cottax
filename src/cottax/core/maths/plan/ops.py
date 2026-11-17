'''
Ops on `Redefine`: what a node reads and owns renamed (`Rewire`, `Rename`, `Cut`), a
definition replaced (`Replace`), what a condition node owns inside set (`SetOwnsInside`), and
what a node carries set and removed (`SetAttribute`, `RemoveAttribute`).
'''

from abc import abstractmethod
import dataclasses
from collections.abc import Iterator, Mapping

from ..definition import (
    Attribute, NodeDefinition, Renaming, misfits, with_attributes, without_attribute,
)
from ...frozen import AbstractClassVar, OrderedMap, Violation, name_of
from ..nesting import NestingGraph
from ..conditional import ConditionalNode, is_driven, kind_of
from ..places import Label, Place, PlaceToPlace
from .primitive import Redefine


type Relabel[N: Label, V: Place] = Mapping[N, Renaming[V]]
'''Per node, old place -> new. Keyed by place, so `fn(x, x)` renames both.'''


def check_renames_places_the_node_has(op: 'Rewire', graph: NestingGraph) -> Iterator[Violation]:
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
    def renaming(self, graph: NestingGraph[N, V]) -> Relabel[N, V]:
        '''Per node, old place -> new. One map serves reads and owns: a node never has a
        place in both.'''

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {name: graph[name].relabel(m)
                for name, m in self.renaming(graph).items()}

class Replace[N: Label, V: Place](Redefine[N, V]):
    """One binding given a definition the caller already holds, attributes and all: who
    hands over a new definition says what it carries."""

    node       : N
    definition : NodeDefinition[V]

    edits_attributes = True

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.node: self.definition}

    def __repr__(self) -> str:
        return f'replace({self.node})'


class Rename[N: Label, V: Place](Rewire[N, V]):
    """One node's places renamed by a mapping the caller already holds."""

    node : N
    onto : tuple[tuple[V, V], ...]
    """Old place -> the place to use instead. Pairs, since an op is a record and hashes."""

    def renaming(self, graph: NestingGraph[N, V]) -> Relabel[N, V]:
        return {self.node: dict(self.onto)}

    def __repr__(self) -> str:
        return (f'rename({self.node}: '
                + ', '.join(f'{o}->{n}' for o, n in self.onto) + ')')


# ---------------------------------------------------------------- Cutting

def check_readers(op: 'Cut') -> Iterator[Violation]:
    '''A cut has readers; with none, nothing would read the copy.'''
    if not op.readers:
        yield NoReaders(op.var, op.copy)


def check_cuts_a_produced_place(op: 'Cut', graph: NestingGraph) -> Iterator[Violation]:
    '''`var` is produced (a boundary input has nothing to close the cut against), and its
    copy is not in use (a second cut would add readers to the first copy).'''
    owners = graph.graph.owners
    if op.var not in owners:
        yield NotProduced(op.var)
    elif op.copy in owners or op.copy in graph.graph.readers:
        yield AlreadyCut(op.var, op.copy)


def check_readers_read_var(op: 'Cut', graph: NestingGraph) -> Iterator[Violation]:
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

    def renaming(self, graph: NestingGraph[N, V]) -> Relabel[N, V]:
        return {r: {self.var: self.copy} for r in self.readers}

    def __repr__(self) -> str:
        return f'cut({self.var}->{self.copy})'


# ---------------------------------------------------------------- what is owned inside

def check_owned_once(op: 'SetOwnsInside') -> Iterator[Violation]:
    """No place is owned inside twice."""
    if twice := tuple(dict.fromkeys(p for p in op.owns_inside if op.owns_inside.count(p) > 1)):
        yield OwnedInsideTwice(twice)


def stores_owns_inside(node: object) -> bool:
    """`node` is a condition node whose condition stores `owns_inside` as a field: what
    `SetOwnsInside` can set."""
    return isinstance(node, ConditionalNode) and 'owns_inside' in {
        f.name for f in dataclasses.fields(node.condition)}


def check_owns_inside_is_stored(op: 'SetOwnsInside', graph: NestingGraph) -> Iterator[Violation]:
    """The node's condition stores `owns_inside` and the node carries no driver: a condition
    that derives it -- from its states -- has no choice to set, and a driver's slots are laid
    out per unknown."""
    if not stores_owns_inside(node := graph[op.node]):
        yield OwnsInsideNotAChoice(op.node, type(kind_of(node)).__name__)
    elif is_driven(node):
        yield OwnsInsideDriven(op.node)


class SetOwnsInside[N: Label, V: Place](Redefine[N, V]):
    """A condition node owns `owns_inside` inside, and nothing else of it changes: with
    none it is a requirement, with some an encloser. For a condition that stores
    `owns_inside` as a field."""

    node        : N
    owns_inside : tuple[V, ...]

    checks = (check_owned_once,)
    checks_graph = (check_owns_inside_is_stored,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        node = graph[self.node]
        return {self.node: dataclasses.replace(
            node, condition=dataclasses.replace(node.condition, owns_inside=self.owns_inside))}

    def __repr__(self) -> str:
        return f'set_owns_inside({self.node} <- {", ".join(str(p) for p in self.owns_inside)})'


# ---------------------------------------------------------------- what a node carries

def check_sets_something(op: 'SetAttribute') -> Iterator[Violation]:
    if not op.values:
        yield NoAttributeNamed()


def check_settable(op: 'SetAttribute', graph: NestingGraph) -> Iterator[Violation]:
    """No node carries one under that key already, and each can carry what it is given."""
    for name, value in op.values.items():
        if op.kind in (node := graph[name]).attributes:
            yield AttributeAlreadySet(name, op.kind.__name__)
        yield from misfits(node, {op.kind: value})


class SetAttribute[N: Label, V: Place](Redefine[N, V]):
    """Each named node carries the value given for it under `kind`; nothing else of it
    changes. An attribute's own op is this with its `kind` named: `Implement`."""

    kind   : AbstractClassVar[type[Attribute]]
    values : OrderedMap[N, object]

    edits_attributes = True

    checks = (check_sets_something,)
    checks_graph = (check_settable,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {name: with_attributes(node := graph[name], OrderedMap({**node.attributes, self.kind: value}))
                for name, value in self.values.items()}

    def __repr__(self) -> str:
        return (f'{type(self).__name__.lower()}('
                + ', '.join(f'{n} <- {name_of(v)}' for n, v in self.values.items()) + ')')


def check_removes_something(op: 'RemoveAttribute') -> Iterator[Violation]:
    if not op.nodes:
        yield NoAttributeNamed()


def check_removable(op: 'RemoveAttribute', graph: NestingGraph) -> Iterator[Violation]:
    """Each node carries one to take off."""
    for name in op.nodes:
        if op.kind not in graph[name].attributes:
            yield AttributeNotSet(name, op.kind.__name__)


class RemoveAttribute[N: Label, V: Place](Redefine[N, V]):
    """The inverse of `SetAttribute`: what each node carries under `kind` taken off."""

    kind  : AbstractClassVar[type[Attribute]]
    nodes : tuple[N, ...]

    edits_attributes = True

    checks = (check_removes_something,)
    checks_graph = (check_removable,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {name: without_attribute(graph[name], self.kind) for name in self.nodes}

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

class OwnedInsideTwice(Violation):
    places : tuple[Place, ...]
    error    = ValueError
    template = '{places!r} is owned inside twice'

class OwnsInsideNotAChoice(Violation):
    node : Label
    kind : str
    error    = TypeError
    template = '{node!r} is a {kind}, and what it owns inside is not a choice'

class OwnsInsideDriven(Violation):
    node : Label
    error    = TypeError
    template = ('{node!r} carries a driver, whose places are laid out per unknown -- '
                '`Undrive` it first')

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
