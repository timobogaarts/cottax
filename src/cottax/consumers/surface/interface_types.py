from collections.abc import Callable, Iterable, Mapping
import dataclasses
import functools
import inspect
from types import UnionType
from typing import TYPE_CHECKING, Any, Self, Union, get_args, get_origin

from ...core import Label, Mint, Place, PlaceToPlace
from ...core.driven import DriverContract, PerProblem, PerUnknown, Role
from ...core.driven import roles
from ...core.frozen import Frozen, OrderedMap, answer, answer_for_class
from .refusals import NoPlaceFor, NotANode, NotAPlace, StandsForSeveral

if TYPE_CHECKING:                                   # the interface is built on this
    from .interface import DerivedNames, Locator


# ============================================================================ what is written

type Named[P] = str | P
'''A place as written: a name, or a path.'''


# ============================================================================ a step down a tree of models

class Step(Frozen):
    '''One step from a container of models to what it holds: a name its place gives the node
    (`DerivedNames.node_in_tree`).'''


class Field(Step):
    '''A dataclass's public field, or a named tuple's.'''
    name : str


class Key(Step):
    '''A mapping's key; a loose wrapper's own name.'''
    key : Any


class Index(Step):
    '''A tuple's or a list's index.'''
    index : int


# ============================================================================ the surface

class Surface[N: Label, V: Place](Frozen):
    '''
    What a model is written through: `locator` says what a bare string means, and what else is
    written means what its `Written` answer says. A part is a level under it adding methods; a
    surface a study uses joins parts, a class. A method's parameter annotated `Place`, `Label`
    or a tuple of either (alone, `| None`, varargs, or in a `Mapping`) is read through the
    surface before the method runs: a caller writes it any way the surface reads, and the
    method is handed it exact.
    '''

    locator : 'Locator[V, N]'

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if shadowing := [n for n in vars(cls).get('__annotations__', {})
                         if callable(vars(Surface).get(n))]:
            raise TypeError(f'{cls.__qualname__}: {shadowing} would shadow the surface\'s own '
                            f'method -- name the field otherwise')
        for name, method in list(vars(cls).items()):
            if not name.startswith('_') and inspect.isfunction(method):
                readings = {p: reading for p, written in inspect.get_annotations(method).items()
                            if p != 'return' and (reading := _reading(written)) is not None}
                if readings:
                    setattr(cls, name, _reading_first(method, readings))

    # ------------------------------------------------------------ another surface

    def located(self, locator: 'Locator[V, N]') -> Self:
        '''The same surface, strings meaning what `locator` says.'''
        return dataclasses.replace(self, locator=locator)

    # ------------------------------------------------------------ what is written

    def _stood(self, written: Any, role: str) -> tuple[V, ...] | None:
        '''The places what is written stands for, `None` where nothing answers for its class;
        refused where its answer says it is no place of this surface's.'''
        if (says := _written(written)) is None:
            return None
        stood = says.places(written)
        if stood is None or not all(isinstance(p, self.locator.places) for p in stood):
            raise NotAPlace(role, repr(written)).raised
        return stood

    def place(self, written: Any, *, role: str = 'what is written') -> V:
        '''A place: a string, as the locator says, an exact place, or what it stands for
        (`Written`), if one place: a `V` path, a wrapper's one output.'''
        if isinstance(written, str):
            return self.locator.place(written)
        if isinstance(written, self.locator.places):
            return written
        if isinstance(written, Derived):
            under = self.place(written.written, role=role)
            for name in written.beside:
                under = self.derived.beside(under, name)
            return written.mint(under)
        if (stood := self._stood(written, role)) is None:
            raise NotAPlace(role, repr(written)).raised
        if len(stood) != 1:
            raise StandsForSeveral(role, repr(written), stood).raised
        return stood[0]

    def places(self, written: Any, *, role: str = 'what is written') -> tuple[V, ...]:
        '''Places: one written alone, or several; what stands for places, all of them.'''
        def each(x: Any) -> tuple[V, ...] | None:
            if isinstance(x, (str, Derived, self.locator.places)):
                return (self.place(x, role=role),)
            return self._stood(x, role)
        if (one := each(written)) is not None:
            return one
        if not isinstance(written, Iterable):
            raise NotAPlace(role, repr(written)).raised
        return tuple(p for x in written for p in (each(x) or (self.place(x, role=role),)))

    def node(self, written: Any, *, role: str = 'what is written') -> N:
        '''A node name: a string, as the locator says, an exact name, or the name it takes
        (`Written`): an `N` path, a wrapper's function's name.'''
        if isinstance(written, str):
            return self.locator.node(written)
        if isinstance(written, self.locator.nodes):
            return written
        if isinstance(written, Derived) and not written.beside:
            return written.mint(self.node(written.written, role=role))
        if (says := _written(written)) is not None and isinstance(
                at := says.node(written, self.derived), self.locator.nodes):
            return at
        raise NotANode(role, repr(written)).raised

    def nodes(self, written: Any, *, role: str = 'what is written') -> tuple[N, ...]:
        '''Node names: one written alone, or several.'''
        one = (isinstance(written, (str, Derived, self.locator.nodes, self.locator.places))
               or _written(written) is not None or not isinstance(written, Iterable))
        return (self.node(written, role=role),) if one else tuple(
            self.node(x, role=role) for x in written)

    @property
    def derived(self) -> 'DerivedNames[N, V]':
        '''The names derived from the class of place the locator locates.'''
        from .interface import DerivedNames                     # the interface is built on this
        return answer_for_class(DerivedNames, self.locator.places)


# ============================================================================ a derived name, as written

class Derived(Frozen):
    '''The name `mint` derives from one as written, read where the surface reads it and never
    stored: `cx.copy('y')`, `cx.solve('f')`, `cx.given.beside('y', 'k')` (`beside` the names
    one deeper first, a place only).'''

    mint    : Mint
    written : Any
    beside  : tuple[str, ...]


class Purpose(Frozen):
    '''A purpose as a study writes it: called on a name as written, the `Derived` name.'''

    mint : Mint

    def __call__(self, written: Any) -> Derived:
        return Derived(self.mint, written, ())

    def beside(self, written: Any, *names: str) -> Derived:
        '''The place for this purpose `names` deeper than the place written.'''
        return Derived(self.mint, written, names)

    def __repr__(self) -> str:
        return repr(self.mint)


def _written(x: Any) -> Any:
    '''The `Written` answer for the class of what is written, `None` where nothing answers.'''
    from .interface import Written                              # the interface is built on this
    return answer(Written, x, None)


# ---------------------------------------------------------------- the purposes the surface derives names for

class Given(Mint):
    '''A read written as a value, beside the first output: `^given.<output>.<parameter>`.'''


class Start(Mint):
    '''Where each unknown starts: `^start.u`.'''


class Steps(Mint):
    '''How many iterations a solve took: `^steps.<the problem>`.'''


# ---------------------------------------------------------------- where a driver's places are named

class RolePlaces(Frozen):
    '''
    Where a driver's data and reports go, a rule per role (`entries`), found along the role's
    ancestry, the most specific first: a `PerUnknown` role's applied to each unknown, a
    `PerProblem` one's to the problem's name read as a place. A role with no entry is refused:
    nothing is named after a role unless a rule says so. `a | b` joins two, `b`'s winning.
    '''

    entries : OrderedMap[type[Role], PlaceToPlace]

    def rule(self, role: type[Role]) -> PlaceToPlace:
        '''The rule for `role`: its own entry, else its nearest ancestor's.'''
        for level in role.__mro__:
            if level in self.entries:
                return self.entries[level]
        raise NoPlaceFor(role.__qualname__).raised

    def __call__[N, V](self, at: N, unknowns: tuple[V, ...], driver: DriverContract
                       ) -> tuple[tuple[tuple[V, ...], ...], tuple[tuple[V, ...], ...]]:
        '''`driver`'s data and reports, for the problem at `at` determining `unknowns`.'''
        from .interface import DerivedNames                 # the interface is built on this
        def slot(role: type[Role]) -> tuple[V, ...]:
            rule = self.rule(role)
            if issubclass(role, PerUnknown):
                return tuple(rule(u) for u in unknowns)
            if issubclass(role, PerProblem):
                return (rule(answer(DerivedNames, at).place_at(at)),)
            raise TypeError(f'{role.__name__} is neither per unknown nor per problem: pass its '
                            f'places to `Assign`')
        return tuple(map(slot, driver.takes)), tuple(map(slot, driver.reports))

    def __or__(self, other: 'RolePlaces') -> 'RolePlaces':
        return RolePlaces(self.entries | other.entries)


core_role_places = RolePlaces(OrderedMap(((roles.Start, Start()),
                                          (roles.Steps, Steps()))))
'''Core's roles, each at its purpose: a start at `^start.u`, the steps at `^steps.<problem>`.'''


# ---------------------------------------------------------------- read before a method runs

type Reading = Callable[[Surface, Any, str], Any]

_READ: dict[Any, str] = {Place: 'place', Label: 'node'}
_READ_MANY: dict[Any, str] = {Place: 'places', Label: 'nodes'}


def _reading(exact: Any) -> Reading | None:
    '''How a parameter a method takes as `exact` is read through a surface from what a caller
    wrote; `None` where it is taken as it is.'''
    if (how := _READ.get(exact)) is not None:
        return lambda surface, x, role: getattr(surface, how)(x, role=role)
    origin, args = get_origin(exact), get_args(exact)
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis \
            and (how := _READ_MANY.get(args[0])) is not None:
        return lambda surface, x, role: getattr(surface, how)(x, role=role)
    if origin is Mapping:
        key, value = (_reading(a) for a in args)
        if key is value is None:
            return None
        return lambda surface, x, role: {
            (key(surface, k, role) if key else k): (value(surface, v, role) if value else v)
            for k, v in x.items()}
    if origin in (Union, UnionType):
        arms = [a for a in args if a is not type(None)]
        mapping = next((r for a in arms if get_origin(a) is Mapping and (r := _reading(a))), None)
        other = next((r for a in arms if get_origin(a) is not Mapping and (r := _reading(a))), None)
        if mapping is other is None:
            return None
        def read(surface: Surface, x: Any, role: str) -> Any:
            if x is None:
                return None
            if mapping is not None and isinstance(x, Mapping):
                return mapping(surface, x, role)
            return other(surface, x, role) if other is not None else x
        return read
    return None


def _reading_first(method: Callable[..., Any], readings: Mapping[str, Reading]) -> Callable[..., Any]:
    '''`method`, each parameter of `readings` read through the surface first, named in a
    refusal as the method's.'''
    signature = inspect.signature(method)
    @functools.wraps(method)
    def read_first(self: Surface, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(self, *args, **kwargs)
        for name, read in readings.items():
            if name in bound.arguments:
                role, given = f"`{method.__name__}`'s `{name}`", bound.arguments[name]
                many = signature.parameters[name].kind is inspect.Parameter.VAR_POSITIONAL
                bound.arguments[name] = (tuple(read(self, x, role) for x in given) if many
                                         else read(self, given, role))
        return method(*bound.args, **bound.kwargs)
    return read_first


__all__ = [
    'Named', 'Step', 'Field', 'Key', 'Index', 'Surface', 'Purpose', 'Derived',
    'Given', 'Start', 'Steps', 'RolePlaces', 'core_role_places',
]
