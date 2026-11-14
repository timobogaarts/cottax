'''
Ops on `Redefine`: what a node reads and owns renamed (`Rewire`, `Rename`, `Cut`), a
definition replaced (`Replace`), and what a condition node owns inside set (`SetOwnsInside`).
'''

from abc import abstractmethod
import dataclasses
from collections.abc import Iterator, Mapping

from ..definition import NodeDefinition, Renaming
from ...frozen import Violation
from ..nesting import NestingGraph
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
    """One binding given a definition the caller already holds."""

    node       : N
    definition : NodeDefinition[V]

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
    """`node` stores `owns_inside` as a field: what `SetOwnsInside` can set."""
    return dataclasses.is_dataclass(node) and 'owns_inside' in {
        f.name for f in dataclasses.fields(node)}


def check_owns_inside_is_stored(op: 'SetOwnsInside', graph: NestingGraph) -> Iterator[Violation]:
    """The node stores `owns_inside`: one that derives it -- from its states, from the
    node it wraps -- has no choice to set."""
    if not stores_owns_inside(node := graph[op.node]):
        yield OwnsInsideNotAChoice(op.node, type(node).__name__)


class SetOwnsInside[N: Label, V: Place](Redefine[N, V]):
    """A condition node owns `owns_inside` inside, and nothing else of it changes: with
    none it is a requirement, with some an encloser. For a kind that stores `owns_inside`
    as a field."""

    node        : N
    owns_inside : tuple[V, ...]

    checks = (check_owned_once,)
    checks_graph = (check_owns_inside_is_stored,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.node: dataclasses.replace(graph[self.node], owns_inside=self.owns_inside)}

    def __repr__(self) -> str:
        return f'set_owns_inside({self.node} <- {", ".join(str(p) for p in self.owns_inside)})'


# ----------------------------------------------------------------- violation finals

class OwnedInsideTwice(Violation):
    places : tuple[Place, ...]
    error    = ValueError
    template = '{places!r} is owned inside twice'

class OwnsInsideNotAChoice(Violation):
    node : Label
    kind : str
    error    = TypeError
    template = ('{node!r} is a {kind}, and what it owns inside is not a choice -- if it '
                'carries a driver, `Undrive` it first')

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
