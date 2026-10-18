'''
What attaching a driver makes: a conditional node with the algorithm answering it, and
the places its roles sit at. Nothing here knows what runs it: `realisation/` sits above
this and wraps a driven node like any other.
'''

import inspect
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

from ...core import ConditionalNode, Label, NestingGraph, Place, Renaming, Violation, relabelled
from ...core.frozen import AbstractVar
from .driver import DriverContract, PerUnknown, Role


def check_pairing(driven: 'Driven') -> Iterator[Violation]:
    '''Check that a driver can be paired with the ConditionalNode. '''
    driver = type(driven.contract).__name__
    if isinstance(driven.condition, DrivenNode):
        yield DrivenTwice(driven.condition, driver)
        return
    if not driven.contract.accepts(driven.condition):
        accepts = driven.contract.accepts
        yield NotAccepted(driver, driven.condition, getattr(accepts, '__name__', repr(accepts)))

def check_slots(driven: 'Driven') -> Iterator[Violation]:
    '''One slot per role the driver declares, and each slot as many places as its role
    takes on this node.'''
    driver = type(driven.contract).__name__
    for side, roles, slots in (('takes', driven.contract.takes, driven.data),
                               ('reports', driven.contract.reports, driven.reports)):
        if len(slots) != len(roles):
            yield SlotsMiscounted(driver, side, roles, len(slots))
            continue
        if twice := tuple(dict.fromkeys(r.__name__ for r in roles if roles.count(r) > 1)):
            yield RoleTwice(driver, side, twice)        # a run hands data back by role
            continue
        for role, places in zip(roles, slots):
            if inspect.isabstract(role):
                yield RoleUnsized(role.__name__)
            elif len(places) != (wanted := role.arity(driven.condition)):
                yield RoleMisplaced(role.__name__, wanted, places)

class DrivenNode[V: Place](ConditionalNode[V]):
    '''
    A conditional node with a driver answering it. Abstract like FunctionalNode & ConditionalNode;
    so an implementation or no implementation is possible.
    '''

    condition : AbstractVar[ConditionalNode[V]]
    '''The node answered, without the driver.'''
    contract    : AbstractVar[DriverContract[V]]
    data      : AbstractVar[tuple[tuple[V, ...], ...]]
    '''One slot per role in `driver.takes`, each as many places as its role's `arity`.'''
    reports   : AbstractVar[tuple[tuple[V, ...], ...]]
    '''One slot per role in `driver.reports`, the same way.'''

    @abstractmethod
    def undriven(self) -> ConditionalNode[V]:
        '''The condition node with the driver taken off: what `Undrive` edits. A final
        that cannot have it taken off refuses here.'''

    def data_of(self, role: type) -> tuple[V, ...] | None:
        '''The places its driver takes `role` at, the slot read off the node: one per unknown
        for a per-unknown role, one for a per-problem one; `None` where the driver takes no
        such role.'''
        takes = self.contract.takes
        return self.data[takes.index(role)] if role in takes else None

    def places_of(self, role: type) -> tuple[V, ...] | None:
        '''The places its driver takes or reports `role` at, laid out as `data_of`; `None`
        where it has no such role.'''
        reports = self.contract.reports
        return self.data_of(role) if role not in reports else self.reports[reports.index(role)]

#-------------------------------------------------------- final with no implementation.

class Driven[V: Place](DrivenNode[V]):
    '''
    Final of DrivenNode that has no implementation.
    '''

    condition : ConditionalNode[V]
    contract    : DriverContract[V]
    data      : tuple[tuple[V, ...], ...]
    reports   : tuple[tuple[V, ...], ...]

    checks = (check_pairing, check_slots)

    @property
    def unknowns(self) -> tuple[V, ...]:    return self.condition.unknowns
    @property
    def conditions(self) -> tuple[V, ...]:  return self.condition.conditions
    @property
    def reads(self) -> tuple[V, ...]:       return self.condition.reads + _joined(self.data)
    @property
    def owns(self) -> tuple[V, ...]:        return self.condition.owns + _joined(self.reports)

    def determined(self, unknowns: tuple[V, ...]) -> 'Driven[V]':
        '''Refused: an algorithm was chosen for the node as it stands, and its
        per-unknown slots are laid out against the unknowns it had.'''
        raise DeterminedWhileDriven(type(self.contract).__name__, self.unknowns,
                                    tuple(unknowns)).raised

    def undriven(self) -> ConditionalNode[V]:
        return self.condition

    def __add__(self, other: 'Driven[V]') -> 'Driven[V]':
        '''The join under one contract: the conditions joined by their own `+`, a per-unknown
        role at each unknown's place, a per-problem role at the one place both take it.
        Refused: two contracts (which algorithm answers the join is not read off the two),
        and a per-problem role at two places.'''
        if not isinstance(other, Driven):
            return NotImplemented
        if other.contract != self.contract:
            raise DriversDiffer(self.contract, other.contract).raised
        joined = self.condition + other.condition
        def slot(role: type, mine: tuple[V, ...], theirs: tuple[V, ...]) -> tuple[V, ...]:
            if issubclass(role, PerUnknown):
                at = dict(zip(self.unknowns, mine)) | dict(zip(other.unknowns, theirs))
                return tuple(at[u] for u in joined.unknowns)
            if mine != theirs:
                raise RoleAtTwoPlaces(role.__name__, mine, theirs).raised
            return mine
        return Driven(joined, self.contract,
                      tuple(map(slot, self.contract.takes, self.data, other.data)),
                      tuple(map(slot, self.contract.reports, self.reports, other.reports)))

    def relabel(self, renaming: Renaming[V]) -> 'Driven[V]':
        return Driven(self.condition.relabel(renaming), self.contract,
                      tuple(relabelled(slot, renaming) for slot in self.data),
                      tuple(relabelled(slot, renaming) for slot in self.reports))

    def __repr__(self) -> str:
        return f'Driven({self.condition!r} by {type(self.contract).__name__})'


def _joined[V](slots: tuple[tuple[V, ...], ...]) -> tuple[V, ...]:
    return tuple(p for slot in slots for p in slot)

def is_driven(d: object) -> bool:
    '''A node carrying a driver, bare or under what runs it.'''
    return isinstance(d, DrivenNode)

def condition_of(node: Any) -> Any:
    '''A driven node's condition, else the node itself.'''
    return node.condition if is_driven(node) else node

def role_slots[N: Label, V: Place](graph: NestingGraph[N, V], role: type[Role]
                                   ) -> tuple[tuple[V, N, V | None], ...]:
    '''Each place a driven problem takes or reports `role` at, in binding order: the place,
    its problem, and for a per-unknown role the unknown it pairs with (`None` for a
    per-problem one). A place two problems take is there twice.'''
    return tuple((place, name, unknown)
                 for name in graph.problems if is_driven(node := graph[name])
                 and (places := node.places_of(role)) is not None
                 for place, unknown in (zip(places, node.unknowns) if issubclass(role, PerUnknown)
                                        else ((places[0], None),)))


# ----------------------------------------------------------------- violation finals

class DeterminedWhileDriven(Violation):
    driver   : str
    unknowns : tuple
    asked    : tuple
    error    = TypeError
    template = ('a {driver} answers this problem over {unknowns!r}, so it cannot be made to '
                'determine {asked!r} -- the algorithm was chosen for the problem as it '
                'stands; `Undrive` it first')


class DrivenTwice(Violation):
    problem : object
    driver  : str
    error    = TypeError
    template = ('{problem!r} already carries an algorithm, so a {driver} cannot answer it '
                'too -- `Undrive` it first')


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
