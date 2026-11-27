'''
A driver: its Roles of its extra data, the contract (what it *may take* and *can* report), 
and the driver itself (which picks out a particular choice).
'''
import dataclasses
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import ClassVar

from ..frozen import AbstractClassVar, Frozen, OrderedMap, Violation, name_of
from ..maths import Place, Renaming, relabelled
from .condition import Condition


# ---------------------------------------------------------------- roles

class Role(ABC):
    '''
    Role of driver I/O and how it maps to the condition itself (i.e. its Places or just the total Condition)
    '''
    def __init__(self) -> None:
        raise TypeError(f'{type(self).__name__} is a role, not a value')
    
    @classmethod
    @abstractmethod
    def index[V: Place](cls, condition: Condition[V]
              ) -> tuple[V | None, ...]:
        '''
        What each Role-keyed place stands for: a {Start : (^start.u,^start.v)} will map to the conditions's
        unknowns as (u,v). None is the condition as a whole (a e.g. a NumberOfIterations).'''

    @classmethod
    def arity(cls, condition: Condition) -> int:
        '''How many places the role requires.'''
        return len(cls.index(condition))

class PerUnknown(Role):
    '''One place per unknown (e.g. where each starts).'''

    @classmethod
    def index[V: Place](cls, condition: Condition[V]
              ) -> tuple[V | None, ...]:
        return condition.owns_inside
class PerCondition(Role):
    '''One place for the condition as a whole (e.g. a tolerance, the steps taken).'''

    @classmethod
    def index[V: Place](cls, condition: Condition[V]
              ) -> tuple[V | None, ...]:
        return (None,)


# ---------------------------------------------------------------- the contract

def check_contract_coherent(c: 'DriverContract') -> Iterator[Violation]:
    '''What a contract declares holds together, whatever it drives: each role it names has an
    `index` and is named once on its side; a report's needs are for a report it makes; no
    role is both required and optional, nor both data and a report.'''
    name = type(c).__name__
    data_side = (*c.takes, *c.may_take, *(r for rs in c.report_needs.values() for r in rs))
    if unsized := tuple(r.__name__ for r in c.roles if inspect.isabstract(r)):
        yield RoleUnsized(unsized)
    for listed in (c.takes, c.may_take, c.may_report):
        if twice := tuple(dict.fromkeys(r.__name__ for r in listed if list(listed).count(r) > 1)):
            yield RoleListedTwice(name, twice)
    if stray := tuple(r.__name__ for r in c.report_needs if r not in c.may_report):
        yield NeedsUnreported(name, stray)
    if both := tuple(r.__name__ for r in c.takes if r in c.may_take):
        yield RequiredAndOptional(name, both)
    if both := tuple(dict.fromkeys(r.__name__ for r in data_side if r in c.may_report)):
        yield RoleOnBothSides(name, both)


class DriverContract[V: Place](Frozen):
    '''
    The algorithm answering a condition node: 
    '''

    accepts : AbstractClassVar[Callable[[Condition[V]], bool]]
    '''Whether it answers that condition'''

    takes   : AbstractClassVar[Collection[type[Role]]]
    '''The data it must be given.'''

    may_take     : ClassVar[Collection[type[Role]]] = ()
    '''The data it may be given besides: none, unless a final says.'''

    may_report   : ClassVar[Collection[type[Role]]] = ()
    '''What it may be asked to report: nothing, unless a final says.'''
    report_needs : ClassVar[Mapping[type[Role], tuple[type[Role], ...]]] = {}
    '''Per report, the data it needs (e.g. trajectory requires places to save at)'''

    checks = (check_contract_coherent,)

    @property
    def options(self) -> tuple[type[Role], ...]:
        '''What a driver of it may ask for: each report it may make, each datum it may take.'''
        return (*self.may_report, *self.may_take)

    @property
    def roles(self) -> tuple[type[Role], ...]:
        '''Every role it names, in the order a driver holds them: its data as `roles_for`
        orders it, then its reports.'''
        needed = (r for rs in self.report_needs.values() for r in rs)
        return tuple(dict.fromkeys((*self.takes, *needed, *self.may_take, *self.may_report)))

    def roles_for(self, asks: Collection[type[Role]]
                  ) -> tuple[tuple[type[Role], ...], tuple[type[Role], ...]]:
        '''The roles a driver asking `asks` (of its `options`) has, in order: its data -- what
        it takes, what the reports asked need, the data asked -- and its reports. The one
        place a choice turns into roles: everything placing or judging a driver asks this.'''
        reports = tuple(r for r in self.may_report if r in asks)
        needed = (r for report in reports for r in self.report_needs.get(report, ()))
        data = tuple(dict.fromkeys((*self.takes, *needed, *(r for r in self.may_take if r in asks))))
        return data, reports


def entry_for[T, D](table: Mapping[type[DriverContract], T], contract: DriverContract,
                    default: D = None) -> T | D:
    '''The entry of `table` for `contract`'s most specific class along its ancestry,
    whatever order the table is written in; `default` if none covers it.'''
    return next((table[k] for k in type(contract).__mro__ if k in table), default)


# ---------------------------------------------------------------- the driver

def check_roles_offered(d: 'Driver') -> Iterator[Violation]:
    '''Its roles are those of one choice: what it reports the contract may report, and its
    data exactly what `roles_for` gives for that choice -- what it asks being its reports and
    the optional data it holds. Asked of the driver alone: none of it needs the condition.'''
    name, contract = repr(d), d.contract
    if extra := tuple(r.__name__ for r in d.reports if r not in contract.may_report):
        yield NotOffered(name, 'reports', extra)
    asks = (*d.reports, *(r for r in d.data if r in contract.may_take))
    data, _ = contract.roles_for(asks)
    if extra := tuple(r.__name__ for r in d.data if r not in data):
        yield NotOffered(name, 'data', extra)
    if missing := tuple(r.__name__ for r in data if r not in d.data):
        yield RoleMissing(name, missing)


class Driver[V: Place](Frozen):
    '''
    The contract answering a condition node, and where each of its roles sits on that node:
    per role, its places, as many as the role's `index` has entries on the node and in that
    order. What it takes is read outside the node's cycle, what it reports owned outside.
    It holds only its own places: what each stands for is read off the current condition,
    never stored, so a rename cannot leave it stale.
    '''

    contract : DriverContract[V]
    data     : OrderedMap[type[Role], tuple[V, ...]] = OrderedMap()
    '''Per role it is given, its places.'''
    reports  : OrderedMap[type[Role], tuple[V, ...]] = OrderedMap()
    '''Per role it is asked to report, its places: a report is asked for by placing it.'''

    checks = (check_roles_offered,)

    @property
    def reads(self) -> tuple[V, ...]:
        '''Every place it takes: read outside the node's cycle.'''
        return tuple(p for places in self.data.values() for p in places)

    @property
    def owns(self) -> tuple[V, ...]:
        '''Every place it reports at: owned outside the node's cycle.'''
        return tuple(p for places in self.reports.values() for p in places)

    def places_for(self, role: type[Role], condition: Condition[V]) -> dict[V | None, V] | None:
        '''What each place of `role` stands for on `condition`, to the place (`{u: s_u, v:
        s_v}` for a start); `None` where it neither takes nor reports `role`.'''
        side = self.data if role in self.data else self.reports if role in self.reports else None
        return None if side is None else dict(zip(role.index(condition), side[role]))

    def relabel(self, renaming: Renaming[V]) -> 'Driver[V]':
        moved = lambda side: OrderedMap((r, relabelled(ps, renaming)) for r, ps in side.items())
        return dataclasses.replace(self, data=moved(self.data), reports=moved(self.reports))

    def violations_on(self, condition: Condition[V]) -> Iterator[Violation]:
        '''What needs the condition: the contract accepts it, and each role's `index` on it
        names no entry twice and has as many entries as the role has places.'''
        if not self.contract.accepts(condition):
            yield NotAccepted(repr(self), condition, name_of(self.contract.accepts))
        for role, places in (*self.data.items(), *self.reports.items()):
            subjects = role.index(condition)
            if len(set(subjects)) != len(subjects):
                yield IndexRepeats(role.__name__, subjects)
            elif len(places) != len(subjects):
                yield RoleMisplaced(role.__name__, len(subjects), places)

    def __repr__(self) -> str:
        return type(self.contract).__name__


# ----------------------------------------------------------------- violation finals

class NotAccepted(Violation):
    driver    : str
    problem   : object
    predicate : str
    error    = ValueError
    template = ('{driver} does not answer {problem!r}: its `accepts` is `{predicate}` -- what '
                'makes an answer right is not negotiable, and what a driver takes is what '
                'that predicate says')


class NotOffered(Violation):
    driver : str
    side   : str
    roles  : tuple
    error    = ValueError
    template = '{driver} is handed {side} for {roles!r}, which its contract does not offer'


class RoleOnBothSides(Violation):
    contract : str
    roles    : tuple
    error    = TypeError
    template = ('{contract} names {roles!r} as data and as a report: a guess and an answer are '
                'two roles')


class RoleListedTwice(Violation):
    contract : str
    roles    : tuple
    error    = TypeError
    template = '{contract} lists {roles!r} twice on one side'


class NeedsUnreported(Violation):
    contract : str
    roles    : tuple
    error    = TypeError
    template = ('{contract} says what {roles!r} need, but may not report them: the needs would '
                'never apply')


class RequiredAndOptional(Violation):
    contract : str
    roles    : tuple
    error    = TypeError
    template = '{contract} both requires and may take {roles!r}'


class RoleMissing(Violation):
    driver : str
    roles  : tuple
    error    = ValueError
    template = '{driver} must be given {roles!r}, and is not'


class RoleUnsized(Violation):
    roles : tuple
    error    = TypeError
    template = ('{roles!r} leave `index` abstract, so nothing says what their places stand '
                'for: a role is per unknown, per condition, or says its own')


class IndexRepeats(Violation):
    role     : str
    subjects : tuple
    error    = ValueError
    template = ('{role} stands for {subjects!r} on this node, an entry twice: a place stands '
                'for one thing')


class RoleMisplaced(Violation):
    role   : str
    wanted : int
    places : tuple
    error    = ValueError
    template = '{role} sits at {places!r}, but takes {wanted} on this node'
