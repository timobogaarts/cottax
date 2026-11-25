'''
How a feature's class gets into a consumer:
1. An `Answer`: one per class per interface, registered with `@answers` and found along the
   class's ancestry.
2. A `Choice`: several fit, so a study names one; each final is marked `@choice(interface)`,
   which lists it among the interface's `choices`.
'''

import inspect
from abc import ABC
from collections.abc import Callable, Iterable
from dataclasses import fields
from typing import Any

from .base import AbstractClassVar, Frozen, PerLevel, _Unbound
from .violation import Violation, dotted, first


class Answer(Frozen):
    '''
    Answer: one class.

    E.g. for a Place->str function:

    `class PlaceToStr(Answer):
        about = Place        
        checks_answer = ()

        @abstractmethod 
        def write_place(self, place : V) -> str:
            ....
    
    @answers(SomePath)
    class SomePathToStr(PlaceToStr[SomePath]):    
        def write_place(self, place: SomePath) -> str:
            return place.get_str()    
    '''

    about : AbstractClassVar[Any]
    '''What it needs answering for (above, a Place). A Protocol is checked by shape.'''

    checks_answer : PerLevel[tuple[Callable[[Any, tuple[Any, ...]], Iterable[Violation]], ...]] = ()
    '''`check(answer, registered)`: why `answer` is refused beside the answers already
    registered for its interface. Example: a JSON codec needs words to be unique to be invertible, so we have to check that they are.'''


class Choice(ABC):
    '''
    Several fit. Marked choice and registered as options.
    
    class Formatter(Choice):
        @abstractmethod 
        def spell(self, path : Path) -> str 
            ...

    @choice(Formatter)
    class NoFormatter(Formatter)        
        def spell(self, path : Path) -> str
            return path.spelling
    '''

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if issubclass(cls, Answer):
            raise TypeError(f'{dotted(cls)} is both an `Answer` and a `Choice`: one right answer is '
                            f'registered, several are named by a study, and nothing is both')


def interface_of(cls: type[Answer]) -> type[Answer]:
    '''The interface `cls` answers: its level directly under `Answer`.'''
    found = [k for k in cls.__mro__ if Answer in k.__bases__]
    if len(found) != 1:
        raise TypeError(f'{dotted(cls)} is under the interfaces '
                        f'{tuple(k.__qualname__ for k in found)}: an answer is registered under '
                        f'one, a level directly under `Answer`')
    return found[0]

# ---------------------------------------------------------------------- registries



# ----------------- answer registering
_ANSWERS : dict[tuple[type, type], Answer] = {}
'''The answer registered per interface and class, in registration order.'''
_FOUND   : dict[tuple[type, type], Answer | None] = {}
'''The answer found per interface and class along its ancestry; emptied on registration.'''


def answers[A: type[Answer]](*classes: type) -> Callable[[A], A]:
    '''Register an `Answer` final for each of `classes`. A class is answered once per
    interface: a second answer is refused, so nothing depends on import order, unless it
    has the first's module and qualified name (a module reloaded), when it replaces it.'''
    if not classes or not all(isinstance(c, type) for c in classes):
        raise TypeError(f'`@answers` takes the classes answered for, got {classes!r}')

    def register(cls: A) -> A:
        if not (isinstance(cls, type) and issubclass(cls, Answer)):
            raise TypeError(f'{getattr(cls, "__qualname__", cls)!r} is not an `Answer`: a '
                            f'`Choice` is named by a study in its slot, never registered')
        interface = interface_of(cls)
        if 'about' not in vars(interface) or isinstance(vars(interface)['about'], _Unbound):
            raise TypeError(f'{interface.__qualname__} does not say what it is answered per: '
                            f'bind `about` on it, a class, a type alias, a protocol or a tuple of those')
        if strays := tuple(dotted(c) for c in classes if not is_about(c, interface.about)):
            raise TypeError(f'{interface.__qualname__} is answered per {_spelled(interface.about)}, '
                            f'and {strays!r} is not: an answer is registered for what its '
                            f'interface is about')
        if held := tuple(f.name for f in fields(cls)):
            raise TypeError(f'{dotted(cls)} holds the fields {held}: an answer is built by the '
                            f'registry, `cls()`, so whatever it needs is a class attribute or a method')
        found = cls()                      # a method left abstract is refused here
        for c in classes:
            earlier = _ANSWERS.get((interface, c))
            if earlier is not None and dotted(type(earlier)) != dotted(cls):
                raise TypeError(f'{dotted(c)} has the {interface.__qualname__} answer '
                                f'{dotted(type(earlier))} already, so {dotted(cls)} is refused: '
                                f'which one answers would be chosen by import order')
        registered = tuple(dict.fromkeys(a for (i, _), a in _ANSWERS.items()
                                         if i is interface and dotted(type(a)) != dotted(cls)))
        for check in cls.per_level('checks_answer'):
            first(check(found, registered))
        _ANSWERS.update(((interface, c), found) for c in classes)
        _FOUND.clear()
        return cls
    return register


def is_about(cls: type, about: Any) -> bool:
    '''Whether `cls` is one of what `about` says: a subclass of a class (or of what a type
    alias names), or for a protocol, met by shape: each of its members an attribute of `cls`
    or annotated on a level of it (a field without a default).'''
    if isinstance(about, tuple):
        return any(is_about(cls, a) for a in about)
    about = getattr(about, '__value__', about)
    if getattr(about, '_is_protocol', False):
        annotated = {n for k in cls.__mro__ for n in inspect.get_annotations(k)}
        return all(hasattr(cls, m) or m in annotated for m in about.__protocol_attrs__)
    return issubclass(cls, about)


def _spelled(about: Any) -> str:
    if isinstance(about, tuple):
        return ' or '.join(map(_spelled, about))
    return getattr(about, '__qualname__', None) or getattr(about, '__name__', repr(about))

# ----------------- choice registering
_CHOICES : dict[type, dict[str, type]] = {}
'''The finals marked per `Choice` interface, by dotted name, in marking order.'''


def choice[C: type](interface: type) -> Callable[[C], C]:
    '''Mark a final as one a study may name for the `Choice` `interface`: it is checked to
    be a final of `interface`, listed among its `choices`, and returned as it is. Marked again
    under its own module and qualified name (a module reloaded), it replaces the first.'''
    if not (isinstance(interface, type) and Choice in interface.__bases__):
        raise TypeError(f'{getattr(interface, "__qualname__", repr(interface))} is not a '
                        f'`Choice` interface: `@choice` names a level directly under `Choice`')

    def marked(cls: C) -> C:
        if not (isinstance(cls, type) and issubclass(cls, interface)):
            raise TypeError(f'{getattr(cls, "__qualname__", repr(cls))} is marked a '
                            f'{interface.__qualname__} choice but does not extend '
                            f'{interface.__qualname__}')
        if inspect.isabstract(cls):
            raise TypeError(f'{dotted(cls)} leaves {tuple(sorted(cls.__abstractmethods__))} '
                            f'abstract, so it is not a {interface.__qualname__} a study can '
                            f'name: `@choice` marks a final')
        _CHOICES.setdefault(interface, {})[dotted(cls)] = cls
        return cls
    return marked
# ----------------- queries
def choices(interface: type) -> tuple[type, ...]:
    '''The finals marked `@choice(interface)` so far, in marking order: those of the
    modules imported.'''
    return tuple(_CHOICES.get(interface, {}).values())


_REFUSE = object()

def answer_for_class[A: Answer](interface: type[A], cls: type, default: Any = _REFUSE) -> A:
    '''The `interface` answer registered for `cls`, looked up along its ancestry, most
    specific first. With none: `default`, or else refused.'''
    key = (interface, cls)
    if key not in _FOUND:
        _FOUND[key] = next((_ANSWERS[interface, k] for k in cls.__mro__
                            if (interface, k) in _ANSWERS), None)
    if (found := _FOUND[key]) is not None:
        return found
    if default is _REFUSE:
        raise Unanswered(interface.__qualname__, dotted(cls)).raised
    return default


def answer[A: Answer](interface: type[A], obj: object, default: Any = _REFUSE) -> A:
    '''The `interface` answer registered for the class of `obj` (`answer_for_class`).'''
    return answer_for_class(interface, type(obj), default)


def answered() -> dict[type[Answer], dict[type, Answer]]:
    '''What has been answered: per interface, the answer registered for each class.'''
    table: dict[type[Answer], dict[type, Answer]] = {}
    for (interface, c), found in _ANSWERS.items():
        table.setdefault(interface, {})[c] = found
    return table

# ----------------- violation finals
class Unanswered(Violation):
    interface : str
    about     : str
    error    = TypeError
    template = ('{about} has no {interface} answer: import the module that registers one '
                '(`@answers`)')


__all__ = ['Answer', 'Choice', 'answers', 'answer', 'answer_for_class', 'answered', 'choice', 'choices',
           'is_about', 'Unanswered']
