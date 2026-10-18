'''
A place as a jax key path, the one implementation of `core.Place` cottax ships: `VarPath` /
`NodePath` (`path.py`), its derived names, a `Mint` as the root segment (`mint.py`), and indexing a pytree by key path
(`crossings.py`), where a key kind of one's own is registered.
'''

from .path import NodePath, Path, VarPath
from . import mint as mint
from .crossings import (
    check_antichain, check_pytree_readable, complete_at, key_kind, read_at, write_at,
)

__all__ = [
    'NodePath', 'Path', 'VarPath',
    'check_antichain', 'check_pytree_readable', 'complete_at', 'key_kind', 'read_at', 'write_at',
]
