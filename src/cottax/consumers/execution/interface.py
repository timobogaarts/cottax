'''
Execution asks of a place:

1. What a data structure holding it is, and how it is read and written (Answer)

Of a feature:

2. Driver implementations, each implementing a DriverContract (Choice)
'''

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from types import get_original_bases
from typing import Any, TypeVar, get_args, get_origin

from ...core import Place
from ...core.frozen import AbstractClassVar, Answer, Choice, Frozen, OrderedMap
from ...core.driven import DriverContract, Role
from .interface_types import ByRole, Context, Posed, Unknowns


# ============================================================================ registered

class DataStructure[V: Place](Answer):
    '''
    A caller's data, indexed by places: registered for the place classes it indexes, and
    handed the data on each call. A slot holding `None` holds no value.
    '''

    about = Place

    @abstractmethod
    def holds(self, data: Any, place: V) -> bool:
        '''Whether `data` has a slot at `place`.'''

    @abstractmethod
    def read(self, data: Any, places: Iterable[V]) -> OrderedMap[V, Any]:
        '''The value in `data` at each of `places`, in that order, `None` where its slot
        holds none; refuses a place it does not hold.'''

    @abstractmethod
    def write(self, data: Any, values: Mapping[V, Any]) -> Any:
        '''`data` with `values` written; refuses a place it cannot hold.'''


# ============================================================================ chosen

class DriverImplementation[D: DriverContract](Frozen, Choice):
    '''
    Implementation for a DriverContract, named per contract in `ImplementDrivers`, per
    problem in `Drive` or `Implement`.

    The type parameter is the contract it implements: `Newton(DriverImplementation[RootFinder])`
    implements a `RootFinder`.

    Needs and provides are with respect to that contract.

    Handed in as:

    Plan(graph) + ImplementDrivers({RootFinder : Newton()}) where the first is a DriverContract type (contract) and 
    the latter is a DriverImplementation for that contract.

    Plan(graph) + Implement({node_name : Newton()}) where the first is a node name and the latter is a DriverImplementation for that node.
    
    '''

    implements : AbstractClassVar[type[DriverContract]]    
    needs      : AbstractClassVar[frozenset[type[Role]]]
    provides   : AbstractClassVar[frozenset[type[Role]]]    

    def __init_subclass__(cls, **kwargs: Any) -> None:
        '''
        Implementation of the type-annotation as `implements`.
        '''
        if 'implements' in cls.__dict__:
            raise TypeError(f'{cls.__name__} binds `implements`, which is read off the type '
                            f'parameter: write `DriverImplementation[<driver>]` instead')
        for base in get_original_bases(cls):
            origin = get_origin(base)
            if not (isinstance(origin, type) and issubclass(origin, DriverImplementation)):
                continue
            for arg in get_args(base):
                if isinstance(arg, type) and issubclass(arg, DriverContract):
                    cls.implements = arg
                elif not isinstance(arg, TypeVar):
                    raise TypeError(f'{cls.__name__} implements {arg!r}, which is not a '
                                    f'driver class')
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def __call__[V: Place](self, posed: Posed[V], context: Context,
                           data: ByRole) -> tuple[Unknowns, ByRole]:
        '''
        Use the driver to solve.

        Parameters
        ----------

        posed : Posed
            What the values mean, the reports asked back, and the condition map they are
            tried through. Static.
        context : Context
            Values that the ConditionMap needs (e.g. boundary inputs of interior graph)
        data  : ByRole
            Values that the driver needs (tolerance, initial guess) keyed by role: one entry
            per role the driver takes, so `role in data` says whether it takes that role.

        Returns
        -----------
        [tuple[Any], Mapping[type[Role], Any]] = [Unknowns, ByRole]
        Unknown values and, per role in `posed.reports`, its values (e.g. trajectory.)
        '''


__all__ = ['DataStructure', 'DriverImplementation']
