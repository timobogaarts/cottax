'''What a driver is: what it accepts, and the roles its places play.'''

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

from ...core import ConditionalNode, Place
from ...core.frozen import AbstractVar, Frozen


class Role(ABC):
    '''
    What a place is to a driver: a class, never a value. Free to extend.
    '''

    def __init__(self) -> None:
        raise TypeError(f'{type(self).__name__} is a role, not a value')

    @classmethod
    @abstractmethod
    def arity(cls, node: ConditionalNode) -> int:
        '''How many places the role takes on `node`.'''


class PerUnknown(Role):
    '''Role per unknown (e.g. initial guesses)'''

    @classmethod
    def arity(cls, node: ConditionalNode) -> int:
        return len(node.unknowns)


class PerProblem(Role):
    '''Role about solve (e.g. global tolerance)'''

    @classmethod
    def arity(cls, node: ConditionalNode) -> int:
        return 1


class DriverContract[V: Place](Frozen):
    '''
    Algorithm answering a ConditionalNode's problem.

    takes (input roles) is not exhaustive: certain choices can not be made from the graph
    such as a Linear Solver type for a Newton driver.
    '''

    accepts : AbstractVar[Callable[[ConditionalNode[V]], bool]]
    '''
    Whether it accepts that ConditionalNode
    '''

    takes : AbstractVar[tuple[type[Role], ...]]
    '''Input roles'''

    reports : AbstractVar[tuple[type[Role], ...]]
    '''Output roles.'''


def entry_for[T, D](table: Mapping[type[DriverContract], T], contract: DriverContract,
                    default: D = None) -> T | D:
    '''The entry of `table` for `contract`'s most specific class along its ancestry,
    whatever order the table is written in; `default` if none covers it.'''
    return next((table[k] for k in type(contract).__mro__ if k in table), default)
