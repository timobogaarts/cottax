'''
Frozen ordered mapping.

Order is needed for tiebreaks in networkx algorithms: 
this requires the order to be stable and part of the equality. 
Two graphs with different orderings execute differently and thus they shouldn't hash equivalently.
'''

from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping
from functools import cached_property


class OrderedMap[K: Hashable, V](Mapping[K, V]):
    '''
    A frozen mapping in the order it was written. Two tuples; the lookup table is built
    on first use and cached, the hash on first call.
    '''

    _keys   : tuple[K, ...]
    _values : tuple[V, ...]

    def __init__(self, entries: 'Mapping[K, V] | Iterable[tuple[K, V]]' = ()) -> None:
        items = tuple(entries.items() if isinstance(entries, Mapping) else entries)
        keys  = tuple(key for key, _ in items)
        # A repeated key would iterate twice and look up once: refused, and the `Counter`
        # naming it is paid only on the way to raising. `trusted` is the door
        # that skips it.
        if len(set(keys)) != len(keys):
            repeated = Counter(keys)
            raise ValueError(f'duplicate key(s) {[k for k, n in repeated.items() if n > 1]!r}')
        object.__setattr__(self, '_keys',   keys)
        object.__setattr__(self, '_values', tuple(value for _, value in items))

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(f'{type(self).__name__} is immutable')

    @cached_property
    def _index(self) -> dict[K, int]:
        return {key: i for i, key in enumerate(self._keys)}

    def __getitem__(self, key: K) -> V:
        return self._values[self._index[key]]

    def __iter__(self) -> Iterator[K]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    # The tuples are already here: `Mapping`'s mixins would go through `__getitem__`.
    def keys(self) -> 'tuple[K, ...]':                  # type: ignore[override]
        return self._keys

    def values(self) -> 'tuple[V, ...]':                # type: ignore[override]
        return self._values

    def items(self) -> 'tuple[tuple[K, V], ...]':       # type: ignore[override]
        return tuple(zip(self._keys, self._values))

    def __eq__(self, other) -> bool:
        if not isinstance(other, OrderedMap):
            return NotImplemented
        return self._keys == other._keys and self._values == other._values

    def __hash__(self) -> int:
        '''`Mapping` sets `__hash__ = None`. Computed once and kept.'''
        try:
            return self._hash
        except AttributeError:
            object.__setattr__(self, '_hash', value := hash((self._keys, self._values)))
            return value

    def __getstate__(self) -> tuple:
        '''The two tuples alone: `_index` and the hash are memos, rebuilt on demand.'''
        return self._keys, self._values

    def __setstate__(self, state: tuple) -> None:
        object.__setattr__(self, '_keys',   state[0])
        object.__setattr__(self, '_values', state[1])

    def __or__(self, other: 'Mapping[K, V]') -> 'OrderedMap[K, V]':
        '''`dict`'s `|`: a key already here keeps its position and takes the new value,
        a new key is appended.'''
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = dict(zip(self._keys, self._values))
        merged.update(other.items())
        return OrderedMap.trusted(tuple(merged), tuple(merged.values()))

    def __ror__(self, other: 'Mapping[K, V]') -> 'OrderedMap[K, V]':
        '''`other | self`, in `other`'s order: still an `OrderedMap`.'''
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = dict(other.items())
        merged.update(zip(self._keys, self._values))
        return OrderedMap.trusted(tuple(merged), tuple(merged.values()))

    def __repr__(self) -> str:
        return f'{{{", ".join(f"{k}: {v!r}" for k, v in self.items())}}}'

    @classmethod
    def trusted(cls, keys: 'tuple[K, ...]', values: 'Iterable[V]') -> 'OrderedMap[K, V]':
        '''Built without the duplicate check, `keys` already distinct: a dict's, or a flatten's.'''
        m = object.__new__(cls)
        object.__setattr__(m, '_keys',   keys)
        object.__setattr__(m, '_values', tuple(values))
        return m
