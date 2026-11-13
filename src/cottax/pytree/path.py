'''
PyTree path: KeyPath from Jax with KeyEntries.
'''

from abc import ABC
import dataclasses

# jax's key vocabulary, re-exported: the `as` spelling says these are part of this module's
# surface and not its plumbing.
from jax.tree_util import (
    DictKey as DictKey, GetAttrKey as GetAttrKey, KeyEntry as KeyEntry,
    KeyPath as KeyPath, SequenceKey as SequenceKey, keystr, register_pytree_with_keys,
)

from ..core.frozen import OrderedMap


@dataclasses.dataclass(frozen=True, slots=True)
class Path(ABC):
    '''
    Place in a Pytree    
    '''

    segments : KeyPath
    _hash    : int = dataclasses.field(init=False, repr=False, compare=False)
    '''The hash of `segments`, taken once: a place keys every map a graph derives, and a
    graph derives them anew after every edit. Not part of the value, nor of a pickle (a
    hash does not outlive its process).'''
    _prefixes : tuple | None = dataclasses.field(init=False, repr=False, compare=False)
    '''`prefixes`, once asked: kept as the hash is, and for the same reason.'''

    def __hash__(self) -> int:
        return self._hash

    @property
    def prefixes(self) -> tuple:
        '''Every strict prefix, root first: the root, `.x`, `.x.y` of `.x.y.z`. Built on the
        first ask and kept, each with its own hash taken.'''
        if (kept := self._prefixes) is None:
            cls, segments = type(self), self.segments
            kept = tuple(cls(segments[:i]) for i in range(len(segments)))
            object.__setattr__(self, '_prefixes', kept)
        return kept

    def __reduce__(self):
        return type(self), (self.segments,)

    @property
    def spelling(self) -> str:
        '''
        jax's own spelling, root dot and all: `.mesh.nodes`, `['eq'][0].psi`. What a
        renderer draws and what `str(path)` gives.

        A label, not an identity: `DictKey(0)` and `SequenceKey(0)` both write `[0]`,
        so nothing keys, orders or compares by it.
        '''
        return keystr(self.segments, simple = False)

    def __post_init__(self) -> None:
        '''Abstract behaviour: we do not want a Path that is not a variable or node path.'''
        if type(self) is Path:
            raise TypeError(
                'Path is abstract: a place is either a variable or node (use `NodePath` or `VarPath`)'
            )
        object.__setattr__(self, '_hash', hash(self.segments))
        object.__setattr__(self, '_prefixes', None)

    def __repr__(self) -> str:
        '''
        The identity: the namespace and the place. What a violation's `{place!r}` prints,
        because the one thing a reader debugging a name needs is which world it is in.
        '''
        return f'{type(self).__name__}({self.spelling})'

    def __str__(self) -> str:
        '''The name: the spelling alone. What an op's repr and a drawing want, where the
        namespace is already said by the slot the name sits in.'''
        return self.spelling

class VarPath(Path):
    '''Where a value lives.'''
    __slots__ = ()


class NodePath(Path):
    '''Where a computation lives.'''
    __slots__ = ()

# `OrderedMap` as a pytree: its keys ride in the treedef, so named values cross a
# `filter_jit` with the names static. Here because core does not import jax.
register_pytree_with_keys(
    OrderedMap,
    lambda m: (tuple((DictKey(k), v) for k, v in zip(m._keys, m._values)), m._keys),
    lambda keys, values: OrderedMap.trusted(keys, values),
)
