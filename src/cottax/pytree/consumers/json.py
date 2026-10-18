'''Registers the pytree place's and node name's JSON codecs: a key path written and read, a
`Mint` segment as the value it is.'''

from functools import singledispatch
from collections.abc import Callable
from typing import Any

from jax.tree_util import FlattenedIndexKey

from ...core import Mint
from ...core.frozen import answers, dotted
from ...consumers.json.interface import LabelCodec, PlaceCodec
from ...consumers.json.interface_types import Reader, Writer
from ..path import DictKey, GetAttrKey, NodePath, SequenceKey, VarPath


SEGMENTS: dict[str, Callable[[Any], object]] = {}
'''Each tag's key kind, built from the one value it holds (`spelled`).'''


@singledispatch
def write_segment(key: object) -> dict[str, Any]:
    raise TypeError(f'{type(key).__name__} is a key kind with no JSON spelling: give it one '
                    'with `spelled`')


def spelled(cls: type, tag: str, value: Callable[[Any], Any],
            build: Callable[[Any], object] | None = None) -> None:
    '''A key kind's JSON spelling: a key of `cls` written `{tag: value(key)}`, read back by
    `build` (`cls` itself by default). A tag spells one kind.'''
    build = cls if build is None else build
    if tag in SEGMENTS and dotted(SEGMENTS[tag]) != dotted(build):  # the same, reloaded
        raise ValueError(f'{tag!r} already spells {dotted(SEGMENTS[tag])}')
    SEGMENTS[tag] = build
    write_segment.register(cls)(lambda key: {tag: value(key)})


def _dict_key(key: DictKey) -> str | int:
    if not isinstance(key.key, (str, int)):
        raise TypeError(f'{key!r}: only a `str` or `int` dict key is written as JSON')
    return key.key


spelled(DictKey, 'dict', _dict_key)
spelled(GetAttrKey, 'attr', lambda key: key.name)
spelled(SequenceKey, 'seq', lambda key: key.idx)
spelled(FlattenedIndexKey, 'flat', lambda key: key.key)


def write_key(key: object, w: Writer) -> dict[str, Any]:
    '''A segment: a purpose as the value it is (its class, by name), any other by its kind's
    spelling.'''
    return {'mint': w.value(key)} if isinstance(key, Mint) else write_segment(key)


def read_key(data: dict[str, Any], r: Reader) -> object:
    (tag, value), = data.items()
    return r.value(value) if tag == 'mint' else SEGMENTS[tag](value)


def segments_of(path: VarPath | NodePath, cls: type) -> tuple:
    '''`path`'s segments, refused unless it is a `cls`: another class's with the same keys
    would be read back as a `cls`.'''
    if not isinstance(path, cls):
        raise TypeError(f'{path!r} is a {dotted(type(path))}, not a {cls.__qualname__}: '
                        f'its keys would be read back as a {cls.__qualname__}')
    return path.segments


@answers(VarPath)
class PytreeCodec(PlaceCodec[VarPath]):
    '''A variable is a `VarPath`, a list of segments.'''

    tag = 'pytree'

    def write_place(self, place: VarPath, w: Writer) -> list:
        return [write_key(k, w) for k in segments_of(place, VarPath)]

    def read_place(self, data: list, r: Reader) -> VarPath:
        return VarPath(tuple(read_key(k, r) for k in data))


@answers(NodePath)
class PytreeLabelCodec(LabelCodec[NodePath]):
    '''A node name is a `NodePath`, a list of segments.'''

    tag = 'pytree'

    def write_label(self, name: NodePath, w: Writer) -> list:
        return [write_key(k, w) for k in segments_of(name, NodePath)]

    def read_label(self, data: list, r: Reader) -> NodePath:
        return NodePath(tuple(read_key(k, r) for k in data))
