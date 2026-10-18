'''
Why a graph is not written or read as JSON, and why a spelling or a codec is refused: a word
another one is registered under (`one_per_kind`, `one_per_tag`).
'''

from collections.abc import Iterator
from typing import TYPE_CHECKING

from ...core import Violation
from ...core.frozen import dotted

if TYPE_CHECKING:
    from .interface import LabelCodec, PlaceCodec, Spelling


def one_per_kind(spelling: 'Spelling', registered: tuple['Spelling', ...]) -> Iterator[Violation]:
    '''A `kind` names one spelling: a file is read by it.'''
    for first in registered:
        if first.kind == spelling.kind:
            yield SpelledTwice(repr(spelling.kind), dotted(type(first)), dotted(type(spelling)))


def one_per_tag(codec: 'PlaceCodec | LabelCodec', registered: tuple['PlaceCodec | LabelCodec', ...]
                ) -> Iterator[Violation]:
    '''A `tag` names one codec of its interface: a file's header is read by it.'''
    for first in registered:
        if first.tag == codec.tag:
            yield CodedTwice(repr(codec.tag), dotted(type(first)), dotted(type(codec)))


class SpelledTwice(Violation):
    what   : str
    first  : str
    second : str
    error    = TypeError
    template = ('{what} is spelled by {first} already, so {second} is refused: which one reads '
                'it would be chosen by import order')


class CodedTwice(Violation):
    what   : str
    first  : str
    second : str
    error    = TypeError
    template = ('{what} is written by the codec {first} already, so {second} is refused: which '
                'one reads it would be chosen by import order')


class Unexportable(Violation):
    node   : object
    what   : str
    reason : str
    error    = TypeError
    template = '{node!r}: {what} is not written as JSON -- {reason}'


class NotAGraph(Violation):
    format   : object
    version  : object
    expected : str
    current  : int
    error    = ValueError
    template = 'a {format!r} file of version {version!r}, not {expected!r} version {current}'


class UnknownCodec(Violation):
    interface : str
    written   : object
    error    = ValueError
    template = ('written by the {interface} {written!r}, which is not registered: import the '
                'module that registers it (a feature registers its own when its `consumers` is '
                'imported)')


class UnknownKind(Violation):
    node : object
    kind : str
    error    = ValueError
    template = ('{node!r}: {kind!r} is no node kind with a spelling registered -- import the '
                'module that registers it (a feature registers its own when its `consumers` is '
                'imported)')


class Inconsistent(Violation):
    node : object
    side : str
    error    = ValueError
    template = '{node!r}: the file\'s {side} roles disagree with its driver\'s'


class NotLoaded(Violation):
    name   : str
    module : str
    error    = ImportError
    template = ('{name!r}: {module!r} is not imported. Loading imports nothing: import it first, '
                'or pass a `resolve` of your own')


__all__ = [
    'one_per_kind', 'one_per_tag', 'SpelledTwice', 'CodedTwice', 'Unexportable',
    'NotAGraph', 'UnknownCodec', 'UnknownKind', 'Inconsistent', 'NotLoaded',
]
