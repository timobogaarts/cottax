'''An executable graph read as steps: `Run` a function node, `Solve` a cycle.'''

from collections.abc import Callable, Iterator
from functools import cached_property
from typing import Self

from .definition import NodeDefinition
from .executable import ExecutableGraph
from ..frozen import Frozen, PerLevel, Violation, unchecked
from .places import Label, Place


def check_every_node(g: 'SteppedExecutableGraph') -> Iterator[Violation]:
    '''Every level's `checks_node`, root first, each asked of every node.'''
    for check in type(g).per_level('checks_node'):
        for name, node in g.definitions.items():
            yield from check(name, node)


class Run[N: Label](Frozen):
    '''A function node, run.'''

    node : N


class Solve[N: Label, G](Frozen):
    '''A cycle: its outermost problem, and what one iteration of it runs.'''

    problem  : N
    interior : G
    '''The cycle without `problem`, proved at the level `steps` was asked of.'''


type Step[N: Label, G] = Run[N] | Solve[N, G]
'''What a run does with one component of an executable graph, by label.'''


class SteppedExecutableGraph[N: Label, V: Place](ExecutableGraph[N, V]):
    '''
    An executable graph read as steps, and the proofs over it that keep them. A level
    below adds `checks_node` alone, which holds of every subset, so every interior is
    proved at the level `steps` was asked of.
    '''

    checks = (check_every_node,)

    checks_node : PerLevel[tuple[Callable[[N, NodeDefinition[V]], Iterator[Violation]], ...]] = ()
    '''`(name, node) -> violations`: the one thing a level below adds.'''

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if 'checks' in cls.__dict__:
            raise TypeError(
                f'{cls.__name__} declares `checks`, a check of the whole graph: a level '
                f'below `SteppedExecutableGraph` says something of each node, so declare '
                f'`checks_node`, or wrap the proof and forward')

    @cached_property
    def steps(self) -> tuple[Step[N, Self], ...]:
        '''
        One per component, in dependency order. One node is a `Run` -- a problem is never
        alone on its cycle -- and several are a cycle, a `Solve` of its one outermost
        problem over the interior.
        '''
        outer = set(self.outermost_problems)
        steps : list[Step[N, Self]] = []
        for component in self.graph.graph.components:
            if len(component) == 1:
                steps.append(Run(component[0]))
            else:
                problem, = (p for p in component if p in outer)
                with unchecked(check_every_node):
                    interior = type(self)(self.interior(problem))
                steps.append(Solve(problem, interior))
        return tuple(steps)
