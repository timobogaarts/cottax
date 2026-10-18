'''
Reading, writing and completing a pytree at bare key paths.
'''

from collections.abc import Callable, Iterable, Mapping
from typing import Any
from functools import singledispatch

from jax.tree_util import (
    DictKey, FlattenedIndexKey, GetAttrKey, KeyEntry, KeyPath, SequenceKey, keystr,
    tree_flatten_with_path,
    tree_unflatten,
)


# ---------------------------------------------------------------------- reading
# A walk, one key at a time: O(depth) for a key kind with a step of its own, a one-level
# flatten matched by equality for any other.

def read_at(tree: Any, keys: KeyPath) -> Any:
    '''The value at `keys`, stepped to one key at a time.'''
    node = tree
    for key in keys:
        node = _step(key, node)
    return node


@singledispatch
def _step(key: KeyEntry, node: Any) -> Any:
    '''One step in `node` by `key`: by default the child a one-level flatten yields at it.'''
    for keys, child in tree_flatten_with_path(
        node, is_leaf=lambda k, _: len(k) == 1, is_leaf_takes_path=True
    )[0]:
        if len(keys) == 1 and keys[0] == key:
            return child
    raise KeyError(key)


KEY_KINDS: set[type] = set()
'''Every class registered as a key kind (`key_kind`): what a place's segment may be.'''


def key_kind[K: type](cls: K, step: Callable[[Any, Any], Any] | None = None) -> K:
    '''Registers `cls` as a key kind, so a key of it is a segment as written; `step(key, node)`
    is its O(1) step, the one-level flatten without one.'''
    KEY_KINDS.add(cls)
    if step is not None:
        _step.register(cls)(step)
    return cls


def is_key_entry(key: object) -> bool:
    '''Whether `key` is of a registered key kind.'''
    return isinstance(key, tuple(KEY_KINDS))


key_kind(GetAttrKey, lambda key, node: getattr(node, key.name))
key_kind(DictKey, lambda key, node: node[key.key])
key_kind(SequenceKey, lambda key, node: node[key.idx])
key_kind(FlattenedIndexKey)


# ---------------------------------------------------------------- writing

def write_at(tree: Any, values: Mapping[KeyPath, Any]) -> Any:
    '''
    `tree` with the value at each key path in `values` replaced.

    One flatten, with each path as a leaf: the treedef is `tree`'s shape with a slot at
    each path, so a path may hold a whole subtree and `None` stays a place. Refuses a path
    inside another (it would be written twice) and a path the flatten does not reach (it
    would be written nowhere, in silence).
    '''
    if not values:
        return tree
    check_antichain(values)
    leaves, treedef = tree_flatten_with_path(
        tree, is_leaf=lambda keys, _: tuple(keys) in values, is_leaf_takes_path=True)
    reached = {k for keys, _ in leaves if (k := tuple(keys)) in values}
    for keys in values:
        if keys not in reached:
            check_pytree_readable(tree, [keys])     # a name that does not resolve at all
            raise KeyError(
                f'{keystr(keys)} resolves but is not a place this pytree flattens to, so '
                f'nothing could be written back there -- register the container as a '
                f'pytree node, name a place inside one, or check the key kinds against '
                f'what it yields'
            )
    return tree_unflatten(treedef, [values[k] if (k := tuple(keys)) in values else value
                                    for keys, value in leaves])


# ---------------------------------------------------------------- completing

def complete_at(tree: Any, names: Iterable[KeyPath]) -> Any:
    '''
    `tree` with every one of `names` it lacks added, holding `None`. Only a `dict` grows a
    key, a missing `dict` along the way made empty; any other missing key is refused, as
    writing refuses it. `tree` itself is not changed.
    '''
    for keys in sorted(names, key=len, reverse=True):   # `.s.u` makes `.s` before `.s` is asked
        tree = _completed(tree, tuple(keys), tuple(keys))
    return tree


def _completed(node: Any, keys: KeyPath, whole: KeyPath) -> Any:
    if not keys:
        return node
    key, rest = keys[0], keys[1:]
    try:
        child = _step(key, node)
    except (AttributeError, IndexError, KeyError, TypeError):
        if not (type(node) is dict and isinstance(key, DictKey)):
            raise KeyError(
                f'{keystr(whole)} names no place in this data structure, and only a dict '
                f'grows a key: {keystr(whole[:len(whole) - len(rest)])} cannot be added'
            ) from None
        return {**node, key.key: _completed({}, rest, whole) if rest else None}
    new = _completed(child, rest, whole)
    if new is child:
        return node
    if type(node) is dict and isinstance(key, DictKey):
        return {**node, key.key: new}
    return write_at(node, {(key,): new})


# ---------------------------------------------------------------- the checks

def _stops_at(tree: Any, keys: KeyPath) -> int | None:
    '''
    Index of the first of `keys` that does not resolve, or `None` if all of them do.
    '''
    node = tree
    for i, key in enumerate(keys):
        try:
            node = _step(key, node)
        except (AttributeError, IndexError, KeyError, TypeError):
            return i
    return None


def check_antichain(names: Iterable[KeyPath]) -> None:
    '''
    Refuse a name that contains another: naming both writes one place twice.

    A write precondition only -- two nested names read fine, one value read twice.
    `core.maths.places.comparables_in` is the same question over `Place`s; this takes raw key
    tuples.
    '''
    names = tuple(names)
    seen  = set(names)
    for keys in names:
        for i in range(1, len(keys)):
            if (head := keys[:i]) in seen:
                raise ValueError(
                    f'{keystr(keys)} lies inside {keystr(head)}, so naming both writes '
                    f'one place twice -- name the outer one, or the parts of it'
                )


def resolves(tree: Any, keys: KeyPath) -> bool:
    '''Whether `read_at` resolves `keys` in `tree`.'''
    return _stops_at(tree, keys) is None


def check_pytree_readable(tree: Any, names: Iterable[KeyPath]) -> None:
    '''
    Raise unless `read_at` resolves every one of `names`, naming the first key that does
    not.
    '''
    for keys in names:
        stop = _stops_at(tree, keys)
        if stop is not None:
            raise KeyError(
                f'{keystr(keys)} names no place in this data structure: '
                f'{keystr(keys[:stop + 1])} does not resolve'
            )
