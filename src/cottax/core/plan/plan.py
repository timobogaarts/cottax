from collections.abc import Iterable
from functools import cached_property

from ..frozen import Frozen, Violation
from ..frozen import first
from ..maths.graph import Graph
from ..maths.places import Label, Place
from ..maths.rules import AnyGraph, Rules
from .op import GraphOp

def check_base_kept(plan: 'Plan') -> Iterable[Violation]:
    '''The base holds `keeps`, as every graph an op leaves must.'''
    return plan.keeps.check(plan.base)


class Plan[N: Label, V: Place](Frozen):
    '''A base graph plus the ops applied to it.'''

    base  : Graph[N, V]
    ops   : tuple[GraphOp[N, V], ...] = ()
    keeps : type[Rules] = AnyGraph
    '''Requirement on the base and after every Op (default = No checks beyond a what a Graph implies)'''

    checks = (check_base_kept,)

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
        graph = self._kept(op.apply(self.graph))
        plan  = Plan(self.base, self.ops + (op,), self.keeps)
        plan.__dict__['graphs'] = (*self.graphs, graph)    # known: not rebuilt
        return plan

    def _kept(self, graph: Graph[N, V]) -> Graph[N, V]:
        first(self.keeps.check(graph))
        return graph

    def __repr__(self) -> str:
        return f'Plan({len(self.base.nodes)} nodes | {" ".join(map(repr, self.ops)) or "-"})'
