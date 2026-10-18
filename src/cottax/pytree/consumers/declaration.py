'''A declared model across a jax trace: its plan, `Static` values and tags the key, every other value traced.'''

import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from ...core.frozen import OrderedMap
from ...consumers.declaration import Declared
from ...consumers.declaration.interface_types import Tagged


class Static(Tagged):
    '''A given value that sizes or selects rather than computes (a count, a switch): kept as
    written and part of the compiled program's key, so a new one compiles anew. Every other
    value given is an array, traced.'''


def _flatten(d: Declared):
    '''A `Declared` across a jit: its plan, static values and tags the key, every other value
    an array of its own dtype, traced.'''
    fixed = {p for p, tags in d.place_tags.items() if any(isinstance(t, Static) for t in tags)}
    traced = OrderedMap((p, jnp.asarray(v)) for p, v in d.given.items() if p not in fixed)
    kept = tuple((p, v) for p, v in d.given.items() if p in fixed)
    return (traced,), (d.plan, kept, tuple(d.given), d.place_tags, d.node_tags)


def _unflatten(key, leaves) -> Declared:
    plan, kept, order, place_tags, node_tags = key
    values = dict(leaves[0].items()) | dict(kept)
    return Declared(plan, OrderedMap((p, values[p]) for p in order), place_tags, node_tags)


register_pytree_node(Declared, _flatten, _unflatten)


__all__ = ['Static']
