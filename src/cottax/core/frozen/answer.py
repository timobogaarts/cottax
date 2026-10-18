'''
How a feature's class gets into a consumer:
1. An `Answer`: one per class per interface, registered with `@answers` and found along the
   class's ancestry.
2. A `Choice`: several fit, so a study names one; each final is marked `@choice(interface)`,
   which registers nothing.
'''

import inspect
from abc import ABC
from collections.abc import Callable, Iterable
from dataclasses import fields
from typing import Any

from .base import AbstractClassVar, Frozen, PerLevel, _Unbound
from .violation import Violation, dotted


class Answer(Frozen):
    '''
    The one right answer per node or place class. A level directly under `Answer` is a
    consumer's interface, saying what it is answered per (`about`); the owner of a class it is
    about writes a final and registers it with `@answers(*classes)`: a feature for its own, a
    consumer for its own and for core's. A final holds no field: the registry builds it,
    `cls()`.
    '''

    about : AbstractClassVar[Any]
    '''What the interface is answered per, bound by the interface: a class, a type alias of
    one, a protocol, or a tuple of those. `@answers` refuses a class that is none; a protocol
    is met by shape, which a class cannot be asked.'''

    checks_answer : PerLevel[tuple[Callable[[Any, tuple[Any, ...]], Iterable[Violation]], ...]] = ()
    '''`check(answer, registered)`: why `answer` is refused beside the answers already
    registered for its interface (a word two would share).'''


class Choice(ABC):
    '''
    Several fit, so none is registered: a study names one in a slot, which the interface
    class's docstring states in a `Handed in as:` block, an example of the call. Never an
    `Answer` as well.
    '''

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if issubclass(cls, Answer):
            raise AnswerAndChoice(dotted(cls)).raised


def interface_of(cls: type[Answer]) -> type[Answer]:
    '''The interface `cls` answers: its level directly under `Answer`.'''
    found = [k for k in cls.__mro__ if Answer in k.__bases__]
    if len(found) != 1:
        raise NotOneInterface(dotted(cls), tuple(k.__qualname__ for k in found)).raised
    return found[0]


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
        if isinstance(vars(interface).get('about'), _Unbound) or 'about' not in vars(interface):
            raise AboutNothing(interface.__qualname__).raised
        if strays := tuple(dotted(c) for c in classes if not is_about(c, interface.about)):
            raise NotAbout(interface.__qualname__, strays, _spelled(interface.about)).raised
        if held := tuple(f.name for f in fields(cls)):
            raise AnswerHolds(dotted(cls), held).raised
        found = cls()                      # a method left abstract is refused here
        for c in classes:
            first = _ANSWERS.get((interface, c))
            if first is not None and dotted(type(first)) != dotted(cls):
                raise AnsweredTwice(interface.__qualname__, dotted(c), dotted(type(first)),
                                    dotted(cls)).raised
        registered = tuple(dict.fromkeys(a for (i, _), a in _ANSWERS.items()
                                         if i is interface and dotted(type(a)) != dotted(cls)))
        for check in cls.per_level('checks_answer'):
            for violation in check(found, registered):
                raise violation.raised
        _ANSWERS.update(((interface, c), found) for c in classes)
        _FOUND.clear()
        return cls
    return register


def is_about(cls: type, about: Any) -> bool:
    '''Whether `cls` is one of what `about` says: a subclass of a class (or of what a type
    alias names), any class for a protocol, met by shape.'''
    if isinstance(about, tuple):
        return any(is_about(cls, a) for a in about)
    about = getattr(about, '__value__', about)
    return getattr(about, '_is_protocol', False) or issubclass(cls, about)


def _spelled(about: Any) -> str:
    if isinstance(about, tuple):
        return ' or '.join(map(_spelled, about))
    return getattr(about, '__qualname__', None) or getattr(about, '__name__', repr(about))


def choice[C: type](interface: type) -> Callable[[C], C]:
    '''Mark a final as one a study may name for the `Choice` `interface`. Nothing is
    registered and the class is returned as it is: only that it is a final of
    `interface` is checked, so every choice on offer is found by searching for the mark.'''
    if not (isinstance(interface, type) and Choice in interface.__bases__):
        raise NotAChoice(getattr(interface, '__qualname__', repr(interface))).raised

    def marked(cls: C) -> C:
        if not (isinstance(cls, type) and issubclass(cls, interface)):
            raise ChoiceElsewhere(getattr(cls, '__qualname__', repr(cls)),
                                  interface.__qualname__).raised
        if inspect.isabstract(cls):
            raise ChoiceAbstract(dotted(cls), interface.__qualname__,
                                 tuple(sorted(cls.__abstractmethods__))).raised
        return cls
    return marked


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


class AboutNothing(Violation):
    interface : str
    error    = TypeError
    template = ('{interface} does not say what it is answered per: bind `about` on it, a class, '
                'a type alias, a protocol or a tuple of those')


class NotAbout(Violation):
    interface : str
    classes   : tuple
    about     : str
    error    = TypeError
    template = ('{interface} is answered per {about}, and {classes!r} is not: an answer is '
                'registered for what its interface is about')


class AnsweredTwice(Violation):
    interface : str
    about     : str
    first     : str
    second    : str
    error    = TypeError
    template = ('{about} has the {interface} answer {first} already, so {second} is refused: '
                'which one answers would be chosen by import order')


class AnswerAndChoice(Violation):
    about : str
    error    = TypeError
    template = ('{about} is both an `Answer` and a `Choice`: one right answer is registered, '
                'several are named by a study, and nothing is both')


class AnswerHolds(Violation):
    answer : str
    held   : tuple[str, ...]
    error    = TypeError
    template = ('{answer} holds the fields {held}: an answer is built by the registry, '
                '`cls()`, so whatever it needs is a class attribute or a method')


class ChoiceAbstract(Violation):
    choice    : str
    interface : str
    abstract  : tuple[str, ...]
    error    = TypeError
    template = ('{choice} leaves {abstract} abstract, so it is not a {interface} a study can '
                'name: `@choice` marks a final')


class ChoiceElsewhere(Violation):
    choice    : str
    interface : str
    error    = TypeError
    template = '{choice} is marked a {interface} choice but does not extend {interface}'


class NotAChoice(Violation):
    interface : str
    error    = TypeError
    template = ('{interface} is not a `Choice` interface: `@choice` names a level directly '
                'under `Choice`')


class NotOneInterface(Violation):
    answer     : str
    interfaces : tuple[str, ...]
    error    = TypeError
    template = ('{answer} is under the interfaces {interfaces}: an answer is registered under '
                'one, a level directly under `Answer`')


class Unanswered(Violation):
    interface : str
    about     : str
    error    = TypeError
    template = ('{about} has no {interface} answer: import the module that registers one '
                '(`@answers`)')


__all__ = ['Answer', 'Choice', 'answers', 'answer', 'answer_for_class', 'answered', 'choice', 'is_about',
           'AnsweredTwice', 'AnswerAndChoice', 'AnswerHolds', 'ChoiceAbstract', 'ChoiceElsewhere',
           'NotAChoice', 'AboutNothing', 'NotAbout', 'NotOneInterface', 'Unanswered']
