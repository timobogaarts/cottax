'''
`Plan`: a base graph plus the ops applied to it, in order.
'''

from functools import cached_property

from ..frozen import Frozen
from ..frozen import first
from ..maths.graph import Graph
from ..maths.places import Label, Place
from ..maths.rules import AnyGraph, Rules
from .op import GraphOp


class Plan[N: Label, V: Place](Frozen):
    '''A base graph plus the ops applied to it, each graph they leave holding `keeps`. It
    pickles as `base`, `ops` and `keeps`; `graphs` is rebuilt on demand, by a loop.'''

    base  : Graph[N, V]
    ops   : tuple[GraphOp[N, V], ...] = ()
    keeps : type[Rules] = AnyGraph
    '''What every graph the ops leave holds: nothing asked by default; `Valid` for a plan of
    a problem, which an assembly asks.'''

    @cached_property
    def graphs(self) -> tuple[Graph[N, V], ...]:
        '''`base`, then the graph each op leaves, in order.'''
        out = [self.base]
        for op in self.ops:
            out.append(self._kept(op.apply(out[-1])))
        return tuple(out)

    @property
    def graph(self) -> Graph[N, V]:
        '''`base` with every op applied.'''
        return self.graphs[-1]

    def __add__(self, op: GraphOp[N, V]) -> 'Plan[N, V]':
        '''`op` applied and appended: a refused op, or one leaving a graph that does not hold
        `keeps`, raises, and never enters the plan.'''
        if not isinstance(op, GraphOp):
            raise TypeError(f'cannot add {type(op).__name__} to a Plan')
        graph = self._kept(op.apply(self.graph))
        plan  = Plan(self.base, self.ops + (op,), self.keeps)
        plan.__dict__['graphs'] = (*self.graphs, graph)    # known: not rebuilt
        return plan

    def _kept(self, graph: Graph[N, V]) -> Graph[N, V]:
        first(self.keeps.check(graph))
        return graph

    def __repr__(self) -> str:
        return f'Plan({len(self.base.nodes)} nodes | {" ".join(map(repr, self.ops)) or "-"})'
