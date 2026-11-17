'''
What a driver is: the algorithm answering a condition node, what it accepts, the roles its
places play, and where each sits on the node. A condition node carries one as its `driver`
field, an `Attachment`: its places are edges, so it is not an attribute, and the maths knows
it by that protocol alone.
'''

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping

from ...core import Attachment, Condition, Place, Renaming, Violation, relabelled
from ...core.frozen import AbstractVar, Frozen


class Role(ABC):
    '''
    What a place is to a driver: a class, never a value. Free to extend.
    '''

    def __init__(self) -> None:
        raise TypeError(f'{type(self).__name__} is a role, not a value')

    @classmethod
    @abstractmethod
    def arity(cls, condition: Condition) -> int:
        '''How many places the role takes on `condition`.'''


class PerUnknown(Role):
    '''Role per unknown (e.g. initial guesses)'''

    @classmethod
    def arity(cls, condition: Condition) -> int:
        return len(condition.owns_inside)


class PerProblem(Role):
    '''Role about solve (e.g. global tolerance)'''

    @classmethod
    def arity(cls, condition: Condition) -> int:
        return 1


class DriverContract[V: Place](Frozen):
    '''
    Algorithm answering a condition node's problem.

    takes (input roles) is not exhaustive: certain choices can not be made from the graph
    such as a Linear Solver type for a Newton driver.
    '''

    accepts : AbstractVar[Callable[[Condition[V]], bool]]
    '''
    Whether it accepts that condition
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


class Driver[V: Place](Attachment[V]):
    '''
    The contract answering a condition node, and where each of its roles sits on that node.
    What it takes is read outside the node's cycle, what it reports owned outside; whether
    the slots fit the node is asked by `violations_on`, a role's arity read off the node.
    '''

    contract : DriverContract[V]
    data     : tuple[tuple[V, ...], ...]
    '''One slot per role in `contract.takes`, each as many places as its role's `arity`.'''
    reports  : tuple[tuple[V, ...], ...]
    '''One slot per role in `contract.reports`, the same way.'''

    @property
    def reads(self) -> tuple[V, ...]:
        '''Every place it takes: read outside the node's cycle.'''
        return _joined(self.data)

    @property
    def owns(self) -> tuple[V, ...]:
        '''Every place it reports at: owned outside the node's cycle.'''
        return _joined(self.reports)

    def data_of(self, role: type) -> tuple[V, ...] | None:
        '''The places it takes `role` at: one per unknown for a per-unknown role, one for a
        per-problem one; `None` where it takes no such role.'''
        takes = self.contract.takes
        return self.data[takes.index(role)] if role in takes else None

    def places_of(self, role: type) -> tuple[V, ...] | None:
        '''The places it takes or reports `role` at, laid out as `data_of`; `None` where it
        has no such role.'''
        reports = self.contract.reports
        return self.data_of(role) if role not in reports else self.reports[reports.index(role)]

    def relabel(self, renaming: Renaming[V]) -> 'Driver[V]':
        return Driver(self.contract,
                      tuple(relabelled(slot, renaming) for slot in self.data),
                      tuple(relabelled(slot, renaming) for slot in self.reports))

    def violations_on(self, node: Condition[V]) -> Iterator[Violation]:
        '''It accepts `node`, it has one slot per role it declares, and each slot as many
        places as its role takes on `node`.'''
        name = repr(self)
        if not self.contract.accepts(node):
            accepts = self.contract.accepts
            yield NotAccepted(name, node, getattr(accepts, '__name__', repr(accepts)))
        for side, roles, slots in (('takes', self.contract.takes, self.data),
                                   ('reports', self.contract.reports, self.reports)):
            if len(slots) != len(roles):
                yield SlotsMiscounted(name, side, roles, len(slots))
                continue
            if twice := tuple(dict.fromkeys(r.__name__ for r in roles if roles.count(r) > 1)):
                yield RoleTwice(name, side, twice)        # a run hands data back by role
                continue
            for role, places in zip(roles, slots):
                if inspect.isabstract(role):
                    yield RoleUnsized(role.__name__)
                elif len(places) != (wanted := role.arity(node)):
                    yield RoleMisplaced(role.__name__, wanted, places)

    def join_violations(self, other: 'Driver[V]', node: Condition[V],
                        other_node: Condition[V]) -> Iterator[Violation]:
        '''Two contracts (which algorithm answers the join is not read off the two), or a
        per-problem role at two places.'''
        if other.contract != self.contract:
            yield DriversDiffer(self.contract, other.contract)
            return
        for roles, mine, theirs in ((self.contract.takes, self.data, other.data),
                                    (self.contract.reports, self.reports, other.reports)):
            for role, here, there in zip(roles, mine, theirs):
                if not issubclass(role, PerUnknown) and here != there:
                    yield RoleAtTwoPlaces(role.__name__, here, there)

    def joined(self, other: 'Driver[V]', node: Condition[V],
               other_node: Condition[V], joined: Condition[V]
                         ) -> 'Driver[V]':
        '''The driver of a join under one contract: a per-unknown role at each unknown's
        place, in the join's order of unknowns; a per-problem role at the one place both
        take it.'''
        def slot(role: type, here: tuple[V, ...], there: tuple[V, ...]) -> tuple[V, ...]:
            if issubclass(role, PerUnknown):
                at = dict(zip(node.owns_inside, here)) | dict(zip(other_node.owns_inside, there))
                return tuple(at[u] for u in joined.owns_inside)
            return here
        return Driver(self.contract,
                      tuple(map(slot, self.contract.takes, self.data, other.data)),
                      tuple(map(slot, self.contract.reports, self.reports, other.reports)))

    def __repr__(self) -> str:
        return type(self.contract).__name__


def _joined[V](slots: tuple[tuple[V, ...], ...]) -> tuple[V, ...]:
    return tuple(p for slot in slots for p in slot)


# ----------------------------------------------------------------- violation finals

class NotAccepted(Violation):
    driver    : str
    problem   : object
    predicate : str
    error    = ValueError
    template = ('{driver} does not answer {problem!r}: its `accepts` is `{predicate}` -- what '
                'makes an answer right is not negotiable, and what a driver takes is what '
                'that predicate says')


class SlotsMiscounted(Violation):
    driver : str
    side   : str
    roles  : tuple
    slots  : int
    error    = ValueError
    template = ('{driver} {side} {roles!r}, one slot each, and is handed {slots} slots')


class RoleTwice(Violation):
    driver : str
    side   : str
    roles  : tuple
    error    = ValueError
    template = ('{driver} {side} {roles!r} more than once: a role is one slot, since a run '
                'hands its data over by role')


class RoleUnsized(Violation):
    role : str
    error    = TypeError
    template = ('{role} leaves `arity` abstract, so nothing says how many places it takes: '
                'a role is per unknown, per problem, or binds its own')


class RoleMisplaced(Violation):
    role   : str
    wanted : object
    places : tuple
    error    = ValueError
    template = '{role} sits at {places!r}, but takes {wanted} on this node'


class DriversDiffer(Violation):
    driver : object
    other  : object
    error    = ValueError
    template = ('{driver!r} and {other!r} answer the two: which algorithm answers their join '
                'is not read off the two -- drive them alike, or `Undrive` and join the conditions')


class RoleAtTwoPlaces(Violation):
    role   : str
    places : tuple
    other  : tuple
    error    = ValueError
    template = ('{role} sits at {places!r} and at {other!r}: one problem takes it once -- '
                'name one place for both before joining them')
