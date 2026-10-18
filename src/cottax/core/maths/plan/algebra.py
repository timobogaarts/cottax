'''
An edit as a record, and the fold of a sequence of them: `GraphOp` (graph -> graph),
`ComposedOp` (an op that is other ops), `Plan` (a base graph plus the ops applied to
it). Over any `NestingGraph`; the primitives with their proofs are `primitive.py`.
'''

from abc import abstractmethod
from collections.abc import Generator, Iterator, Sequence
from contextvars import ContextVar
from functools import cached_property

from typing import Callable

from ...frozen import Frozen, PerLevel, Violation, first
from ..nesting import NestingGraph
from ..places import Label, Place


def refusals[N: Label, V: Place](op: 'GraphOp[N, V]',
                                 graph: NestingGraph[N, V]) -> 'Iterator[Violation]':
    '''
    Why `op` does not apply to `graph`, as data -- every `checks_graph` case.

    Most derived level first: an op's own precondition is asked before the one it
    inherits, since an inherited check reads what the derived op computes (`Rewire`'s
    reads `Cut`'s renaming).
    '''
    for check in type(op).per_level('checks_graph', derived_first=True):
        yield from check(op, graph) or ()


_owed: ContextVar[dict | None] = ContextVar('owed', default=None)
'''Inside a composition: each rule its steps would ask in full, owed by the composition's
result and asked of it once (`proved_once`); `None` outside one.'''


def proved_once[G: NestingGraph](build: Callable[[], G]) -> G:
    '''`build()`, the rules its steps would ask in full asked once of the graph it returns, if
    this is the outermost composition; inside one, the outermost asks them. A step's own
    preconditions and proofs are asked as ever.'''
    if _owed.get() is not None:
        return build()
    token = _owed.set({})
    try:
        graph, owed = build(), _owed.get()
    finally:
        _owed.reset(token)
    for rule, nesting in owed.items():
        first(rule(graph if nesting else graph.graph) or ())
    return graph


class GraphOp[N: Label, V: Place](Frozen):
    '''
    Graph operation: changing the computational graph.

    Four stages:

    1. op alone: is this a valid op?  [checks of frozen]
    2. op and graph: can this op apply to this graph? [checks_graph]
    3. op, graph, and delta: does this delta make sense to apply [ checks_delta]
    4. new graph: does this graph need less proving [covers]

    3 and 4 are a `Primitive`'s (`primitive.py`), the ops that build a graph. Any other op
    composes them and has only 1 and 2: it proves nothing of its own.
    '''

    checks_graph : PerLevel[tuple[Callable, ...]] = ()
    '''What the op needs of the graph before it can compute anything: `(op, graph) ->
    Iterator[Violation]`, collected by `refusals` and raised by `apply`.'''

    def apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        '''The rewrite: refuses, or returns the graph with this op applied.'''
        first(refusals(self, graph))
        return self._apply(graph)

    @abstractmethod
    def _apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        '''The edit itself, which may assume every `checks_graph` case holds.'''

    def __call__[G: NestingGraph[N, V]](self, graph: G) -> G:
        return self.apply(graph)

    def __radd__(self, base: NestingGraph[N, V]) -> 'Plan[N, V]':
        '''`graph + op`, so a plan need not be spelled to start one. Here rather than
        on the graph, because the layering runs graph -> plan and not back.'''
        if not isinstance(base, NestingGraph):
            return NotImplemented
        return Plan(base) + self


class ComposedOp[N: Label, V: Place](GraphOp[N, V]):
    '''
    An op that *is* other ops, in order, all decided off the graph it meets. It proves
    nothing of its own: every graph it passes on was built by the op that built it, with its
    proofs; what a step would ask in full is asked once, of the result (`proved_once`).
    '''

    @abstractmethod
    def composed(self, graph: NestingGraph[N, V]) -> Sequence[GraphOp[N, V]]:
        '''The ops this one is. Takes a graph, so an expansion may depend on what it
        meets.'''

    def _apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        '''Each op applied to the graph *that* op meets, so its preconditions are
        asked of the right one.'''
        def build() -> G:
            g = graph
            for op in self.composed(g):
                g = op.apply(g)
            return g
        return proved_once(build)

class Stage[N: Label, V: Place](Frozen):
    '''One stage of a `StagedOp`: the ops decided, and the graph they leave.'''

    ops   : tuple[GraphOp[N, V], ...]
    graph : NestingGraph[N, V]


class StagedOp[N: Label, V: Place](GraphOp[N, V]):
    '''
    A composition decided stage by stage: each stage's ops are read off the graph the
    one before left. No proofs
    '''

    @abstractmethod
    def stages(self, graph: NestingGraph[N, V]
               ) -> Generator[Sequence[GraphOp[N, V]], NestingGraph[N, V], None]:
        '''Yield a stage's ops; receive the graph they leave.'''

    def staged(self, graph: NestingGraph[N, V]) -> tuple[Stage[N, V], ...]:
        '''Every stage, each op applied to the graph *that* op meets: the derivation,
        kept.'''
        out, decide = [], self.stages(graph)
        def build() -> NestingGraph[N, V]:
            g = graph
            try:
                ops = next(decide)
                while True:
                    ops = tuple(ops)
                    for op in ops:
                        g = op.apply(g)
                    out.append(Stage(ops, g))
                    ops = decide.send(g)
            except StopIteration:
                return g
        proved_once(build)
        return tuple(out)

    def _apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        stages = self.staged(graph)
        return stages[-1].graph if stages else graph


# ---------------------------------------------------------------- plan

class Plan[N: Label, V: Place](Frozen):
    '''
    A base graph plus the operations applied to it. A `Frozen` with no rules of its
    own: what it buys is the cached hash and a pickle that carries `base` and `ops`
    and no prefix graph, which `graph` rebuilds on demand.
    '''

    base : NestingGraph[N, V]
    ops  : tuple[GraphOp[N, V], ...] = ()

    @cached_property
    def trunk(self) -> 'Plan[N, V] | None':
        '''This plan minus its last op; `None` for the base. Cached, so every prefix
        of a plan is one object and is built once.'''
        return Plan(self.base, self.ops[:-1]) if self.ops else None

    @cached_property
    def graph(self) -> NestingGraph[N, V]:
        '''`base` with every op applied.'''
        return self.base if self.trunk is None else self.ops[-1](self.trunk.graph)

    def __add__(self, op: GraphOp[N, V]) -> 'Plan[N, V]':
        '''Append `op`.'''
        if not isinstance(op, GraphOp):
            raise TypeError(f'cannot add {type(op).__name__} to a Plan')
        graph = op(self.graph)
        plan  = Plan(self.base, self.ops + (op,))
        # prime the trunk by the current graph.        
        plan.__dict__.update(trunk=self, graph=graph)
        return plan

    def __repr__(self) -> str:
        return f'Plan({len(self.base.nodes)} nodes | {" ".join(map(repr, self.ops)) or "-"})'


