'''
A key path's derived names: the `Mint` itself is the root segment, a key kind of its own, so
a flatten, which only yields the kinds a pytree node registers, never yields a minted name.
'''

from ..core import Mint, Minting
from ..core.frozen import answers
from .crossings import key_kind
from .path import NodePath, Path, VarPath


key_kind(Mint)


@answers(VarPath, NodePath)
class PathMinting(Minting[Path]):
    '''A purpose as the root segment: `^copy.y`, and over a minted name `^start.^copy.y`.'''

    def minted[P: Path](self, m: Mint, name: P) -> P:
        return type(name)((m, *name.segments))

    def mint_of(self, name: Path) -> Mint | None:
        first = name.segments[:1]
        return first[0] if first and isinstance(first[0], Mint) else None

    def unminted[P: Path](self, name: P) -> P:
        return type(name)(name.segments[1:]) if self.mint_of(name) is not None else name
