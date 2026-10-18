'''Why a picture is not drawn: two names, or two categories, it would draw as one.'''

from collections.abc import Iterable, Iterator

from ...core import Label, Violation
from .lookup import spell_plain


def spelled_apart(names: Iterable[Label]) -> Iterator[Violation]:
    '''Two different names with one plain spelling: a picture keys a name by it.'''
    seen : dict[str, Label] = {}
    for name in names:
        if (first := seen.setdefault(spelled := spell_plain(name), name)) != name:
            yield SpelledAlike(first, name, spelled)


class SpelledAlike(Violation):
    first   : Label
    second  : Label
    spelled : str
    error    = ValueError
    template = ('{first!r} and {second!r} are both spelled {spelled!r}: a picture keys a name '
                'by its plain spelling, so it would draw one as the other')


class NamedAlike(Violation):
    name   : str
    first  : object
    second : object
    error    = TypeError
    template = 'two categories are named {name!r}: {first!r} and {second!r}'


__all__ = ['NamedAlike', 'SpelledAlike', 'spelled_apart']
