'''
Ops on `Redefine`: what a node reads and owns renamed (`Rewire`, `Rename`, `Cut`), a
definition replaced (`Replace`), what a condition node owns inside set (`SetOwnsInside`), and
what a node carries set and removed (`SetAttribute`, `RemoveAttribute`).
'''

from abc import abstractmethod
import dataclasses
from collections.abc import Iterator, Mapping

from ..maths.definition import (
    Attribute, NodeDefinition, Renaming,
)
from ..frozen import AbstractClassVar, OrderedMap, Violation, name_of
from ..maths.dependencies import Dependencies
from ..maths.graph import Graph
from ..maths.places import Label, Place, PlaceToPlace
from .primitive import Redefine


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
    owners = Dependencies.of(graph).owners
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
