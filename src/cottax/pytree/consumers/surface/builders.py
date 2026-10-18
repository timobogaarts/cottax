'''
Writing a place using `V` or a node using `N`:

V.plasma.ip[0] = GetAttrKey('plasma'), GetAttrKey('ip'), SequenceKey(0)
'''

import operator
from typing import Any

from ....pytree.crossings import is_key_entry
from ....core import Mint
from ....pytree.path import (
    DictKey, GetAttrKey, KeyEntry, KeyPath, NodePath, Path, SequenceKey, VarPath,
)

class Keys:
    '''
    Key Path builder: `V` or `N` followed by attribute and index access.

    WARNING: a [0] always is a SequenceKey (not a DictKey). Use [DictKey(0)] instead.

    Refused: a private or dunder name, which keeps `copy`, `pickle` and IPython's probes
    from becoming keys.
    '''

    __slots__ = ('_into', '_keys')

    def __init__(self, into: type[Path], keys: KeyPath = ()) -> None:
        object.__setattr__(self, '_into', into)
        object.__setattr__(self, '_keys', keys)

    def __getattr__(self, name: str) -> 'Keys':
        if name.startswith('_'):
            raise AttributeError(f'{name!r}: a private or dunder name is not a key')
        if name in _PATH_NAMES:
            raise AttributeError(f'{self!r}.{name}: {self!r} is a key path under construction, '
                                 f'not a path -- pass it to `place` or `node` first')
        return Keys(self._into, self._keys + (GetAttrKey(name),))

    def __getitem__(self, key: Any) -> 'Keys':
        return Keys(self._into, self._keys + (entry_of(key),))

    def __iter__(self):
        '''Refused: iteration would otherwise index `[0]`, `[1]`, ... without end.'''
        raise TypeError(f'{self!r} is a key path under construction, not a path: pass it to '
                        f'`place` or `node` first')

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError('a key path is immutable')

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Path):
            raise TypeError(f'{self!r} is a key path under construction, compared to a path: '
                            f'pass it to `place` or `node` first')
        return (isinstance(other, Keys)
                and (self._into, self._keys) == (other._into, other._keys))

    def __hash__(self) -> int:
        '''Its path's: looked up among paths it meets one, and `__eq__` refuses there.'''
        return hash(self._into(self._keys))

    def __reduce__(self):
        return Keys, (self._into, self._keys)

    def __repr__(self) -> str:
        return ('V' if self._into is VarPath else 'N') + self._into(self._keys).spelling


def entry_of(key: Any) -> KeyEntry:
    '''The segment an index writes: a key of a registered key kind (`key_kind`) as it is, an `int` a `SequenceKey`, any
    other key a `DictKey`. An index that reads as a number or a slice is refused, and so is
    a minted segment, which only a mint writes.'''
    if isinstance(key, Mint):
        raise TypeError(f'{key!r}: a minted segment is written by its mint (`Copy()(V.y)`, `cx.copy(...)`)')
    if is_key_entry(key):
        return key
    if isinstance(key, bool) or isinstance(key, (float, complex, slice, Keys, Path)):
        raise TypeError(f'{key!r}: an index is a key entry, an int >= 0 (a sequence index) or '
                        f'a dict key')
    if isinstance(key, int) or hasattr(key, '__index__'):
        key = operator.index(key)
        if key < 0:
            raise TypeError(f'{key!r}: an index is a key entry, an int >= 0 (a sequence '
                            f'index) or a dict key')
        return SequenceKey(key)
    try:
        hash(key)
    except TypeError:
        raise TypeError(f'{key!r}: a dict key is hashable') from None
    return DictKey(key)


_PATH_NAMES = frozenset(name for name in dir(VarPath) if not name.startswith('_'))
'''What a `Path` answers, so what a builder must not quietly answer too.'''

# ---------------------------------------------------------------- the roots, and a builder converted
V = Keys(VarPath)
'''The root of every place.'''
N = Keys(NodePath)
'''The root of every node name.'''


def path_of[P: Path](x: Keys | P, into: type[P]) -> P:
    '''`x` as a `into`: a builder in that namespace, or a path already one. A root names
    nothing, and a builder of the other namespace is refused.'''
    if isinstance(x, Keys):
        if x._into is not into:
            raise TypeError(f'{x!r} is not a {into.__name__}')
        if not x._keys:
            raise ValueError(f'{x!r} is the root, which names no {into.__name__}')
        return into(x._keys)
    if isinstance(x, into):
        return x
    if isinstance(x, str):
        raise TypeError(f'{x!r}: a string is a name only under a root, which only a surface '
                        f'holds -- write the {into.__name__} in full, with '
                        f'{"V" if into is VarPath else "N"}')
    raise TypeError(f'{x!r} names no {into.__name__}: write it with '
                    f'{"V" if into is VarPath else "N"}, or pass a {into.__name__}')

type Var  = Keys | VarPath
type Node = Keys | NodePath


def place_of(x: Var) -> VarPath:
    return path_of(x, VarPath)


def node_of(x: str | Node) -> NodePath:
    '''A node name: an `N` path or a `NodePath`, or a string, a `DictKey` at the root, as a
    slot in a mapping is.'''
    return NodePath((DictKey(x),)) if isinstance(x, str) else path_of(x, NodePath)


def keys_of(x: Var) -> KeyPath:
    '''The keys of a place, or of a builder, the root included: what a scope extends.'''
    return x._keys if isinstance(x, Keys) and x._into is VarPath else place_of(x).segments
