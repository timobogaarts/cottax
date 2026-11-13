'''
Graph edits as records: `GraphOp` (graph -> graph), `ComposedOp` and `StagedOp` (ops made
of ops), `Plan` (a base graph plus the ops applied to it). The primitives are `primitive.py`.
'''

from abc import abstractmethod
from collections.abc import Generator, Iterator, Sequence
from functools import cached_property

from typing import Callable

from ...frozen import Frozen, PerLevel, Violation, first
from ..nesting import NestingGraph
from ..places import Label, Place


def refusals[N: Label, V: Place](op: 'GraphOp[N, V]',
                                 graph: NestingGraph[N, V]) -> 'Iterator[Violation]':
    '''Why `op` does not apply to `graph`, as data. Most derived level first: an inherited
    check may read what the derived op computes (`Rewire`'s reads `Cut`'s renaming).'''
    for check in type(op).per_level('checks_graph', derived_first=True):
        yield from check(op, graph) or ()


class GraphOp[N: Label, V: Place](Frozen):
    '''
    A change to the graph, as a record. Checked twice:

    1. the op alone: is it a valid op? [`checks`]
    2. the op and a graph: does it apply to this graph? [`checks_graph`]
    '''

    checks_graph : PerLevel[tuple[Callable, ...]] = ()
    '''`(op, graph) -> Iterator[Violation]`: collected by `refusals`, raised by `apply`.'''

    def apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        '''Refuse, or return the graph with this op applied.'''
        first(refusals(self, graph))
        return self._apply(graph)

    @abstractmethod
    def _apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        '''The edit itself; may assume `checks_graph` passed.'''

    def __call__[G: NestingGraph[N, V]](self, graph: G) -> G:
        return self.apply(graph)

    def __radd__(self, base: NestingGraph[N, V]) -> 'Plan[N, V]':
        '''`graph + op` starts a plan. Here and not on the graph, which knows no plans.'''
        if not isinstance(base, NestingGraph):
            return NotImplemented
        return Plan(base) + self


class ComposedOp[N: Label, V: Place](GraphOp[N, V]):
    '''An op made of other ops, applied in order. It builds no graph itself.'''

    @abstractmethod
    def composed(self, graph: NestingGraph[N, V]) -> Sequence[GraphOp[N, V]]:
        '''The ops this one is, which may depend on `graph`.'''

    def _apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        g = graph
        for op in self.composed(g):
            g = op.apply(g)
        return g

class Stage[N: Label, V: Place](Frozen):
    '''One stage of a `StagedOp`: the ops decided, and the graph they leave.'''

    ops   : tuple[GraphOp[N, V], ...]
    graph : NestingGraph[N, V]


class StagedOp[N: Label, V: Place](GraphOp[N, V]):
    '''A composition decided in stages: each stage's ops depend on the graph the stage
    before left.'''

    @abstractmethod
    def stages(self, graph: NestingGraph[N, V]
               ) -> Generator[Sequence[GraphOp[N, V]], NestingGraph[N, V], None]:
        '''Yield a stage's ops; receive the graph they leave.'''

    def staged(self, graph: NestingGraph[N, V]) -> tuple[Stage[N, V], ...]:
        '''Every stage, with the graph it left.'''
        out, decide, g = [], self.stages(graph), graph
        try:
            ops = next(decide)
            while True:
                ops = tuple(ops)
                for op in ops:
                    g = op.apply(g)
                out.append(Stage(ops, g))
                ops = decide.send(g)
        except StopIteration:
            pass
        return tuple(out)

    def _apply[G: NestingGraph[N, V]](self, graph: G) -> G:
        stages = self.staged(graph)
        return stages[-1].graph if stages else graph


# ---------------------------------------------------------------- plan

class Plan[N: Label, V: Place](Frozen):
    '''A base graph plus the ops applied to it. It pickles as `base` and `ops`; `graph` is
    rebuilt on demand.'''

    base : NestingGraph[N, V]
    ops  : tuple[GraphOp[N, V], ...] = ()

    @cached_property
    def trunk(self) -> 'Plan[N, V] | None':
        '''This plan minus its last op; `None` for the base. Cached, so each prefix is built
        once.'''
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
        # Both are already known: do not rebuild them.
        plan.__dict__.update(trunk=self, graph=graph)
        return plan

    def __repr__(self) -> str:
        return f'Plan({len(self.base.nodes)} nodes | {" ".join(map(repr, self.ops)) or "-"})'


