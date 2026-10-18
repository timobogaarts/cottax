'''
`Frozen`, the base every frozen dataclass of the graph layer inherits from: `AbstractVar`
and `AbstractClassVar` as in equinox, without jax; abstract-or-final enforced for them and
for every method; `checks` accumulated per level rather than overridden.
'''

import inspect
from abc import ABC
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from collections.abc import Callable, Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self, get_origin

if TYPE_CHECKING:
    type AbstractVar[T] = Annotated[T, 'AbstractVar']
    from typing import ClassVar as AbstractClassVar
    from typing import ClassVar as PerLevel
else:
    class AbstractVar[T]:
        '''Annotation only, on a level: `fn: AbstractVar[Callable]`.'''

    class AbstractClassVar[T]:
        '''Annotation only, on a level: `fn: AbstractClassVar[Callable]`. A class variable'''

    class PerLevel[T]:
        '''Annotation only, on the level that declares it: `checks: PerLevel[tuple] = ()`.
        A class variable each level binds for itself, read with `per_level`, which
        accumulates them root first -- so a level adds and never replaces.'''


class _Unbound:
    '''What the level binds in place of the annotation: abstract to ABC until a subclass
    binds the name, and absent on a class that declares it as a field.'''

    __isabstractmethod__ = True

    def __init__(self, name: str, level: type) -> None:
        self.name, self.level = name, level

    def __get__(self, obj, owner=None):
        for cls in owner.__mro__:
            if cls is self.level:
                break
            annotation = inspect.get_annotations(cls).get(self.name)
            if annotation is not None and not _is_class_var(annotation):
                raise AttributeError(f'{owner.__name__}.{self.name} is a field, set per instance')
        return self

    def __repr__(self) -> str:
        return f'<AbstractVar {self.level.__name__}.{self.name}>'


def _is_class_var(annotation) -> bool:
    return (annotation is ClassVar or get_origin(annotation) is ClassVar
            or (isinstance(annotation, str) and annotation.startswith('ClassVar')))


def _abstract_kind(annotation) -> 'type | None':
    '''`AbstractVar`, `AbstractClassVar`, or `None` for an ordinary annotation.'''
    if isinstance(annotation, str):
        if annotation.startswith(('AbstractVar', 'AbstractClassVar')):
            raise NotImplementedError('a stringified abstract annotation is not supported')
        return None
    if annotation in (AbstractVar, AbstractClassVar):
        raise TypeError(f'`{annotation.__name__}` must be subscripted with the type a final binds')
    origin = get_origin(annotation)
    return origin if origin in (AbstractVar, AbstractClassVar) else None


_skipped : ContextVar['frozenset | None'] = ContextVar('skipped', default=frozenset())
'''The checks not asked at construction: none, some by identity, or `None` for all.'''


@contextmanager
def unchecked(*skipped: Callable[[Any], object]):
    '''
    Disable Frozen `checks` for the scope of `with`: the callables in `skipped`, matched by
    identity, or all if `skipped` is empty.
    '''
    token = _skipped.set(None if not skipped else _skipped.get() | frozenset(skipped)
                         if _skipped.get() is not None else None)
    try:
        yield
    finally:
        _skipped.reset(token)


class Frozen(ABC):
    '''
    Every subclass is a frozen dataclass whose `checks`, each level's added to its
    ancestors', run at post-init.
    '''

    checks : PerLevel[tuple[Callable[[Any], object], ...]] = ()
    '''
    Callable [self] -> iterable of violations, which are raised at construction unless skipped (see `unchecked`).
    '''

    __per_level__ = frozenset({'checks'})
    '''The names declared `PerLevel` here and above: what `per_level` may be asked.'''

    @classmethod
    def per_level(cls, name: str, derived_first: bool = False) -> 'Iterator[Callable]':
        '''
        Get the attribute `name` from each level in the MRO, in order from base to
        implementation, or the reverse. `name` is one a level declared `PerLevel`.
        '''
        if name not in cls.__per_level__:
            raise TypeError(f'{cls.__name__}.{name} is not declared `PerLevel`, so no level '
                            f'accumulates it')
        for level in (cls.__mro__ if derived_first else reversed(cls.__mro__)):
            yield from level.__dict__.get(name, ())

    def __post_init__(self) -> None:
        '''
        Run the checks, unless skipped by `unchecked`.
        '''
        skipped = _skipped.get()
        if skipped is None:
            return
        for check in type(self).per_level('checks'):
            if check in skipped:
                continue
            for violation in check(self) or ():
                raise violation.raised

    def __getstate__(self) -> dict:
        '''The fields alone: a cached answer in `__dict__` is a memo, not the value, so a
        pickle or a copy comes back cold -- the same rule the cache follows for
        equality and the jit key.'''
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def _carried(self, **changes) -> 'Self':
        '''
        The same value with `changes` in place of those fields, and no rule asked.
        For delta checking.

        i.e. ._carried(x=1) is like `replace(self, x=1)` but without running checks.
        '''
        with unchecked():
            return replace(self, **changes)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        if '__post_init__' in cls.__dict__:
            raise TypeError(
                f'{cls.__name__} defines `__post_init__`, which would bypass `Frozen`\'s '
                f'runner: declare `checks = (...)` instead'
            )

        # As equinox's `better_dataclass` does it: read through `inspect.get_annotations`
        # and *assign* the filtered dict back, since Python 3.14 (PEP 649) evaluates class
        # annotations lazily and keeps nothing in the class `__dict__` to edit in place.
        annotations = inspect.get_annotations(cls)
        per_level = {n for n, a in annotations.items() if get_origin(a) is PerLevel}
        cls.__per_level__ = frozenset(per_level).union(
            *(getattr(k, '__per_level__', ()) for k in cls.__mro__[1:]))
        kinds = {n: k for n, a in annotations.items() if (k := _abstract_kind(a)) is not None}
        for name in kinds:
            if name in cls.__dict__:
                raise TypeError(f'{cls.__name__}.{name} is abstract and cannot have a value')
            setattr(cls, name, _Unbound(name, cls))
        cls.__annotations__ = {n: a for n, a in annotations.items()
                               if n not in kinds and n not in per_level}

        # `dataclass` writes a `__hash__` on every class without one in its own body, so
        # a hash a level wrote would be shadowed on each final. Carry it down instead: required
        # for properly writing custom jax-allowable hashes.
        if '__hash__' in cls.__dict__:
            cls.__written_hash__ = cls.__dict__['__hash__']
        elif (written := getattr(cls, '__written_hash__', None)) is not None:
            cls.__hash__ = written
        # A `__repr__` a level wrote is carried down the same way: `dataclass` leaves it alone.
        written_repr = any('__repr__' in k.__dict__ for k in cls.__mro__[:-1] if k is not Frozen)
        dataclass(frozen=True, repr=not written_repr)(cls)
        # The hash `dataclass` generated, over the eq fields, computed once per instance:
        # a value is frozen, so the memo can never be stale, and it lives in `__dict__`
        # like every cached query. A hash a level wrote is left as it is.
        if getattr(cls, '__written_hash__', None) is None and '__hash__' in cls.__dict__:
            cls.__field_hash__ = cls.__dict__['__hash__']
            cls.__hash__ = _cached_hash

        cls.__annotations__ = annotations           # restored for readers; no longer a field
        per_class = {n for k in cls.__mro__[1:] for n in getattr(k, '__abstract_class_vars__', ())}
        per_class |= {n for n, k in kinds.items() if k is AbstractClassVar}
        cls.__abstract_class_vars__ = frozenset(per_class)
        if as_field := [n for n in per_class if n in {f.name for f in fields(cls)}]:
            raise TypeError(
                f'{cls.__name__} binds {", ".join(sorted(as_field))} as a field, but an '
                f'`AbstractClassVar` is per class: bind it as a class attribute'
            )
        _refuse_overrides(cls)
        _refuse_joined_twice(cls)
        # Abstract or final: bound once in a line of descent, never rebound below it.
        if rebound := [n for n in per_class if n in cls.__dict__ and any(
                n in k.__dict__ and not isinstance(k.__dict__[n], _Unbound) for k in cls.__mro__[1:])]:
            raise TypeError(
                f'{cls.__name__} rebinds {", ".join(sorted(rebound))}, which a level above it '
                f'already bound: an `AbstractClassVar` is bound once, by a final'
            )


def _cached_hash(self) -> int:
    '''`__field_hash__` once, then the memo; nothing is memoised when hashing raises.'''
    memo = self.__dict__
    try:
        return memo['_frozen_hash']
    except KeyError:
        memo['_frozen_hash'] = value = self.__field_hash__()
        return value


def _implementation(value: object) -> 'Callable | None':
    '''The function behind a method, property, cached property, static or class method, or
    what a descriptor names as its `__implementation__`; `None` for anything else -- a plain
    value, which accumulates or is bound once.'''
    if isinstance(value, type):
        return None
    if (implementation := getattr(value, '__implementation__', None)) is not None:
        return implementation
    if isinstance(value, (staticmethod, classmethod)):
        return value.__func__
    if isinstance(value, property):
        return value.fget
    if isinstance(value, cached_property):
        return value.func
    return value if inspect.isfunction(value) else None


def _refuse_overrides(cls: type) -> None:
    '''Abstract or final, for methods: a class may implement a name only where no level
    above it already did -- binding an abstract one is what a final is for. Dunders are
    exempt: they are Python's protocols, cooperative or written by `dataclass`.'''
    for name, value in cls.__dict__.items():
        if name.startswith('__') and name.endswith('__') or (f := _implementation(value)) is None:
            continue
        for level in cls.__mro__[1:]:
            if name not in level.__dict__:
                continue
            above = level.__dict__[name]
            g = _implementation(above)
            if g is None or g is f or getattr(g, '__isabstractmethod__', False):
                break                          # a value, the same one, or abstract: fine
            raise TypeError(
                f'{cls.__name__}.{name} overrides the one {level.__name__} implements: a '
                f'level implements a name once. Subclass the level that leaves `{name}` '
                f'abstract, or make it abstract on {level.__name__} with finals beside '
                f'{cls.__name__}'
            )


def _refuse_joined_twice(cls: type) -> None:
    '''Abstract or final, across a join: two levels of `cls`, neither above the other, may
    not implement one name with different functions, which the order of the bases would
    otherwise settle silently. Dunders and abstract methods are exempt, as for an override.'''
    found: dict[str, tuple[type, Callable]] = {}
    for level in cls.__mro__:
        for name, value in level.__dict__.items():
            if (name.startswith('__') and name.endswith('__')
                    or (f := _implementation(value)) is None
                    or getattr(f, '__isabstractmethod__', False)):
                continue
            first, g = found.setdefault(name, (level, f))
            if g is not f and not issubclass(first, level):
                raise TypeError(
                    f'{cls.__name__} joins {first.__name__} and {level.__name__}, which both '
                    f'implement `{name}`: a level implements a name once. Make it abstract on '
                    f'a level both share, or implement it on one of them only'
                )
