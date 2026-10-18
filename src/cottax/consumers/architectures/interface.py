'''
Interface for the architectures.

1. How a problem is placed: `ModelReading`, three questions, registered per condition class
2. How a group of cuts is closed: `Closing`, chosen

```
ModelReading  not_implicit(graph, problem)      why it is not an implicit model: refusals, none if it is
              body(graph, problem)              the function nodes that are its own: boxed with it, never cut
              combines_as(condition)            the kind whose `+` folds it (`Combine`), or None
Closing       __call__(cycle, group)            the op closing a group of cuts on a cycle
              bound_at(cycle, group)            the node name of the problem that op binds
```

A problem is asked by its node name in the graph handed; the answer is found by its
condition's class, a driven node's through its condition (`condition_of`).
'''

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

from ...core import ConditionalNode, GraphOp, Label, NestingGraph, Place, ToCut, Violation
from ...core.frozen import Answer, Choice, Frozen

# ============================================================================ registered

class ModelReading(Answer):
    '''
    How a problem reads as a model:

    1. Whether it is an implicit model
    2. Its body, the function nodes that are its own
    3. How it combines with other problems
    '''

    about = ConditionalNode

    @abstractmethod
    def not_implicit[N: Label, V: Place](self, graph: NestingGraph[N, V], problem: N
                                         ) -> Iterable[Violation]:
        '''Why the problem bound at `problem` is no implicit model in its language; nothing
        if it is.'''

    @abstractmethod
    def body[N: Label, V: Place](self, graph: NestingGraph[N, V], problem: N) -> tuple[N, ...]:
        '''The function nodes that are its own, in binding order: what an architecture boxes
        with it, and never cuts inside.'''

    @abstractmethod
    def combines_as(self, condition: Any) -> type | None:
        '''The kind whose `+` folds it (`Combine`'s `kind`), or `None`: it is folded into
        nothing.'''


# ============================================================================ chosen

class Closing[N: Label, V: Place](Frozen, Choice):
    '''
    How a group of cuts is closed: the op closing it, and the name of the problem that op
    binds. (i.e. conventionally a fixed point cut)

    Handed in as:

    MDF(GaussSeidel(), cx.closing, top), an architecture's second argument; or
    CloseFunctionalCycles(GaussSeidel(), cx.closing).
    '''

    @abstractmethod
    def __call__(self, cycle: tuple[N, ...], group: ToCut[N, V]) -> GraphOp[N, V]:
        '''The op closing `group`, cut on `cycle` (its members in binding order).'''

    @abstractmethod
    def bound_at(self, cycle: tuple[N, ...], group: ToCut[N, V]) -> N:
        '''Where that op binds the problem closing `group`.'''


__all__ = ['ModelReading', 'Closing']
