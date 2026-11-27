'''
What a driver is: the algorithm answering a condition node -- its contract, the roles its
places play -- and where each of those places sits. A condition node carries one as its
`driver` field: its places are edges, so it is not an attribute. Judged against the bare
condition, so below the node.
'''

import dataclasses
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import ClassVar

from ..frozen import AbstractVar, Frozen, OrderedMap, Violation, name_of
from ..maths import Place, Renaming, relabelled
from .condition import Condition


# ---------------------------------------------------------------- roles

class Role(ABC):
    '''
    What a set of places is to a driver: a class, never a value, and the key a run hands the
    data over by. Its `index` says what each of its places stands for on a problem answered
    by a contract: one entry per place, in order, no entry twice. Free to extend: a feature
    defines a role over its own conditions (a multiplier per inequality).
    '''

    def __init__(self) -> None:
        raise TypeError(f'{type(self).__name__} is a role, not a value')

    @classmethod
    @abstractmethod
    def index(cls, condition: Condition, contract: 'DriverContract') -> tuple[object, ...]:
        '''What each place stands for: an unknown, `None` for the problem as a whole, or what
        the role reads off the condition or the contract (a segment, a constraint).'''

    @classmethod
    def arity(cls, condition: Condition, contract: 'DriverContract') -> int:
        '''How many places the role takes: one per entry of its `index`.'''
        return len(cls.index(condition, contract))


class PerUnknown(Role):
    '''One place per unknown (e.g. where each starts).'''

    @classmethod
    def index(cls, condition: Condition, contract: 'DriverContract') -> tuple[object, ...]:
        return condition.owns_inside


class PerProblem(Role):
    '''One place for the problem as a whole (e.g. a tolerance, the steps taken).'''

    @classmethod
    def index(cls, condition: Condition, contract: 'DriverContract') -> tuple[object, ...]:
        return (None,)


# ---------------------------------------------------------------- the contract

class DriverContract[V: Place](Frozen):
    '''
    The algorithm answering a condition node: what it accepts, the data it must be given
    (`takes`), the data it may be given (`may_take`), what it may report (`may_report`), and
    the data a report needs of its own (`report_needs`). A report, or an optional datum, is
    asked for by giving it places: nothing is switched on by a flag. A final binds each as a
    value, or as a property read off its own fields. Not exhaustive: a choice the graph
    cannot make (a Newton driver's linear solver) is a field of the final.
    '''

    accepts : AbstractVar[Callable[[Condition[V]], bool]]
    '''Whether it answers that condition.'''

    takes   : AbstractVar[Collection[type[Role]]]
    '''The data it must be given, whatever it is asked to report.'''

    may_take     : ClassVar[Collection[type[Role]]] = ()
    '''The data it may be given besides: none, unless a final says.'''
    may_report   : ClassVar[Collection[type[Role]]] = ()
    '''What it may be asked to report: nothing, unless a final says.'''
    report_needs : ClassVar[Mapping[type[Role], tuple[type[Role], ...]]] = {}
    '''Per report, the data it needs of its own (an integrator's trajectory, the times it is
    saved at).'''

    def required(self, reports: Collection[type[Role]]) -> tuple[type[Role], ...]:
        '''The data it must be given when asked for `reports`, in order: `takes`, then what
        each of those reports needs.'''
        return tuple(dict.fromkeys((*self.takes, *(r for report in reports
                                                     for r in self.report_needs.get(report, ())))))

    @property
    def may_be_given(self) -> tuple[type[Role], ...]:
        '''Every role it may be given data for: what the most it can report requires, then
        `may_take`.'''
        return tuple(dict.fromkeys((*self.required(tuple(self.may_report)), *self.may_take)))


def entry_for[T, D](table: Mapping[type[DriverContract], T], contract: DriverContract,
                    default: D = None) -> T | D:
    '''The entry of `table` for `contract`'s most specific class along its ancestry,
    whatever order the table is written in; `default` if none covers it.'''
    return next((table[k] for k in type(contract).__mro__ if k in table), default)


# ---------------------------------------------------------------- the driver

def check_roles_offered(d: 'Driver') -> Iterator[Violation]:
    '''Each role it is given the contract may be given, each it reports the contract may
    report, none on both sides, none left without an `index`; and what the contract requires
    for those reports is given. Asked of the driver alone: none of it needs the condition.'''
    name, contract = repr(d), d.contract
    for side, allowed, given in (('data', contract.may_be_given, d.data),
                                 ('reports', tuple(contract.may_report), d.reports)):
        if extra := tuple(r.__name__ for r in given if r not in allowed):
            yield NotOffered(name, side, extra)
    if both := tuple(r.__name__ for r in d.data if r in d.reports):
        yield RoleOnBothSides(name, both)
    if unsized := tuple(r.__name__ for r in (*d.data, *d.reports) if inspect.isabstract(r)):
        yield RoleUnsized(unsized)
    if missing := tuple(r.__name__ for r in contract.required(tuple(d.reports))
                        if r not in d.data):
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

    def places_for(self, role: type[Role], condition: Condition[V]) -> dict[object, V] | None:
        '''What each place of `role` stands for on `condition`, to the place (`{u: s_u, v:
        s_v}` for a start); `None` where it neither takes nor reports `role`.'''
        side = self.data if role in self.data else self.reports if role in self.reports else None
        return None if side is None else dict(zip(role.index(condition, self.contract), side[role]))

    def relabel(self, renaming: Renaming[V]) -> 'Driver[V]':
        moved = lambda side: OrderedMap((r, relabelled(ps, renaming)) for r, ps in side.items())
        return dataclasses.replace(self, data=moved(self.data), reports=moved(self.reports))

    def violations_on(self, condition: Condition[V]) -> Iterator[Violation]:
        '''What needs the condition: the contract accepts it, and each role's `index` on it
        names no entry twice and has as many entries as the role has places.'''
        if not self.contract.accepts(condition):
            yield NotAccepted(repr(self), condition, name_of(self.contract.accepts))
        for role, places in (*self.data.items(), *self.reports.items()):
            subjects = role.index(condition, self.contract)
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
    driver : str
    roles  : tuple
    error    = ValueError
    template = ('{driver} both takes and reports {roles!r}: a guess and an answer are two '
                'roles')


class RoleMissing(Violation):
    driver : str
    roles  : tuple
    error    = ValueError
    template = '{driver} must be given {roles!r}, and is not'


class RoleUnsized(Violation):
    roles : tuple
    error    = TypeError
    template = ('{roles!r} leave `index` abstract, so nothing says what their places stand '
                'for: a role is per unknown, per problem, or says its own')


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
