'''
Ops on `Redefine`: what a node reads and owns renamed (`Rewire`, `Rename`, `Cut`), a
definition replaced (`Replace`), and what it determines (`Determine`, `Undetermine`).
'''

from abc import abstractmethod
from collections.abc import Iterator, Mapping

from ..definition import NodeDefinition, Renaming
from ...frozen import Violation
from ..nesting import NestingGraph
from ..places import Label, Place, PlaceToPlace
from .primitive import Redefine


type Relabel[N: Label, V: Place] = Mapping[N, Renaming[V]]
'''Per node, old place -> new. Keyed by place, so `fn(x, x)` renames both.'''


def check_renames_places_the_node_has(op: 'Rewire', graph: NestingGraph) -> Iterator[Violation]:
    '''
    Cannot rename a place the node does not read or own.
    '''
    for name, m in op.renaming(graph).items():
        node = graph[name]
        has  = set(node.reads) | set(node.owns)
        if missing := tuple(v for v in m if v not in has):
            yield NotRenamed(name, missing)


class Rewire[N: Label, V: Place](Redefine[N, V]):
    """
    Rename the places a definition reads and owns only.
    """

    checks_graph = (check_renames_places_the_node_has,)

    @abstractmethod
    def renaming(self, graph: NestingGraph[N, V]) -> Relabel[N, V]:
        '''Per node, old place -> new. One map: a node reads a place or owns it, never
        both, so nothing has to say which field a renaming lands in.'''
        ...

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
    '''A cut has readers: with none, the copy is a name nothing reads.'''
    if not op.readers:
        yield NoReaders(op.var, op.copy)


def check_cuts_a_produced_place(op: 'Cut', graph: NestingGraph) -> Iterator[Violation]:
    '''A boundary input renamed leaves nothing to close the cut against, and a second cut
    of one variable would put more readers on the first copy.'''
    owners = graph.graph.owners
    if op.var not in owners:
        yield NotProduced(op.var)
    elif op.copy in owners or op.copy in graph.graph.readers:
        yield AlreadyCut(op.var, op.copy)


def check_readers_read_var(op: 'Cut', graph: NestingGraph) -> Iterator[Violation]:
    '''Every reader reads `var`. The producer owns it, so renaming there would move the
    output and leave `var` a boundary input.'''
    if off := tuple(r for r in op.readers if r in graph and op.var not in graph[r].reads):
        yield DoesNotRead(op.var, off)


class Cut[N: Label, V: Place](Rewire[N, V]):
    '''
    Rewire the *reads* of `readers` onto a copy of `var`. The producer and `var` are
    untouched, so a reader outside `readers` still gets the computed value.
    '''

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


# ---------------------------------------------------------------- the unknowns

def check_unknowns(op: 'Determine') -> Iterator[Violation]:
    '''Something is determined, and nothing twice.'''
    if not op.unknowns:
        yield NoUnknowns()
    if len(set(op.unknowns)) != len(op.unknowns):
        yield UnknownTwice(tuple(op.unknowns))


def check_determines_nothing_yet(op: 'Determine', graph: NestingGraph) -> Iterator[Violation]:
    '''What a condition node determines is replaced, never added to.'''
    if unknowns := tuple(graph[op.requirement].unknowns):
        yield AlreadyDetermined(op.requirement, unknowns)


def check_determines_something(op: 'Undetermine', graph: NestingGraph) -> Iterator[Violation]:
    '''The inverse: there is something to drop.'''
    if not graph[op.problem].unknowns:
        yield AlreadyUndetermined(op.problem)


class Determine[N: Label, V: Place](Redefine[N, V]):
    '''A requirement, with the unknowns it determines named: `determined`, asked of the
    node, so what makes an answer right stays the condition node's own business.'''

    requirement : N
    unknowns    : tuple[V, ...]

    checks = (check_unknowns,)
    checks_graph = (check_determines_nothing_yet,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.requirement: graph[self.requirement].determined(self.unknowns)}

    def __repr__(self) -> str:
        return f'determine({self.requirement} <- {", ".join(str(u) for u in self.unknowns)})'


class Undetermine[N: Label, V: Place](Redefine[N, V]):
    '''The inverse of `Determine`: the unknowns dropped, the condition node kept.'''

    problem : N

    checks_graph = (check_determines_something,)

    def redefined(self, graph: NestingGraph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        return {self.problem: graph[self.problem].determined(())}

    def __repr__(self) -> str:
        return f'undetermine({self.problem})'


# ----------------------------------------------------------------- violation finals

class NoReaders(Violation):
    var  : Place
    copy : Place
    error    = ValueError
    template = 'cut({var}): no readers, so {copy!r} would be a name nothing reads'

class NoUnknowns(Violation):
    error    = ValueError
    template = 'determine(): no unknowns -- a condition node that determines nothing is what it already is'

class UnknownTwice(Violation):
    unknowns : tuple[Place, ...]
    error    = ValueError
    template = 'determine(): {unknowns!r} names an unknown twice'

class AlreadyDetermined(Violation):
    requirement : Label
    unknowns    : tuple[Place, ...]
    error    = TypeError
    template = ('{requirement!r} already determines {unknowns!r} -- `Undetermine` it first, so '
                'the earlier choice is dropped deliberately rather than silently replaced')

class AlreadyUndetermined(Violation):
    problem : Label
    error    = TypeError
    template = '{problem!r} already determines nothing, so there is nothing to take off'

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
