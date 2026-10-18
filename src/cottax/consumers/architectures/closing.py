'''Every group of cuts closed by a `Closing`, each at a name of its own; `CloseFunctionalCycles`, the op recording a rule.'''

from collections.abc import Iterable, Sequence

from ...core import (
    CutRule, GraphOp, Label, NestingGraph, Opened, Place, first, cuts_per_cycle, ComposedOp,
)
from .interface import Closing
from .refusals import ClosuresCollide


def closed[N: Label, V: Place](opened: Iterable[Opened[N, V]], closing: Closing[N, V]
                               ) -> tuple[GraphOp[N, V], ...]:
    '''Every group on every cycle closed, each bound at a name of its own.'''
    groups = tuple((cycle, group) for cycle, gs in opened for group in gs)
    names = [closing.bound_at(cycle, group) for cycle, group in groups]
    first(() if len(set(names)) == len(names)
          else (ClosuresCollide(tuple(str(n) for n in dict.fromkeys(
              n for n in names if names.count(n) > 1))),))
    return tuple(closing(cycle, group) for cycle, group in groups)


class CloseFunctionalCycles[N: Label, V: Place](ComposedOp[N, V]):
    '''
    Every cycle of the function nodes closed by the rule, over the graph this op is
    applied to. Records the rule, where a closure records a decision.
    '''

    rule    : CutRule[N, V]
    closing : Closing[N, V]

    def composed(self, graph: NestingGraph[N, V]) -> Sequence[GraphOp[N, V]]:
        return closed(cuts_per_cycle(graph, self.rule), self.closing)

    def __repr__(self) -> str:
        return f'close_functional_cycles({self.rule!r})'
