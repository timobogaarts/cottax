'''
What a run takes, a `Model`; what a driver's implementation is handed and returns: the problem
`Posed`, the `ConditionMap` it tries values through, and the values it is written against.
'''

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol

from ...core import ConditionalNode, NestingGraph, Place
from ...core.frozen import Frozen
from ...core.driven import Role


class Model(Protocol):
    '''A declared model as a run takes it: its plan's graph and the values given with it.'''

    graph : NestingGraph
    given : Mapping[Place, Any]


type Unknowns = tuple[Any, ...]
'''One value per unknown of the node answered, in `node.unknowns` order.'''

type ConditionValues = tuple[Any, ...]
'''One value per condition of the node answered, in `node.conditions` order.'''

type Context = tuple[Any, ...]
'''What a condition map needs besides the unknowns: opaque, and passed on unchanged --
`args` in optimistix and scipy.'''

type ByRole = Mapping[type[Role], tuple[Any, ...]]
'''Per role of the driver, one value per place of the role's slot: the data it takes, or
the reports it makes.'''


class ConditionMap(Frozen):
    '''
    A condition node's values as a function: unknowns tuple first, then context tuple.

    See `InteriorConditionMap` for the common case.
    '''

    @abstractmethod
    def __call__(self, unknowns: Unknowns, context: Context) -> ConditionValues:
        '''The value at each of the node's reads: its conditions, then its parameters.'''


class Posed[V: Place](Frozen):
    '''
    The problem a driver implementation solves, all static: the condition node, the reports
    asked back, the condition map it tries values through, and where the context is read.
    '''

    condition  : ConditionalNode[V]
    '''The condition node answered, without its driver: its unknowns, its conditions, and
    what they mean.'''

    reports    : tuple[type[Role], ...]
    '''The roles asked back: the driver's `reports`.'''
    
    conditions : ConditionMap
    '''`conditions(unknowns, context)`: the value at each of `condition.reads`, the
    conditions then the parameters, which the solve holds fixed.'''

    context_places : tuple[V, ...]
    '''Where each value of the context is read: a parameter among them.'''

    def fixed(self, places: tuple[V, ...], context: Context) -> tuple[Any, ...]:
        '''The value of each of `places` in `context`, read with no unknown: a parameter an
        implementation needs before it tries any (an initial value, a span).'''
        at = dict(zip(self.context_places, context, strict=True))
        return tuple(at[p] for p in places)


__all__ = ['Model', 'Unknowns', 'ConditionValues', 'Context', 'ByRole', 'ConditionMap', 'Posed']
