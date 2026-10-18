'''Registers how a key path is drawn (`PlaceDrawing`); offers `NoFormat`, jax's spelling,
chosen as a drawing's `formatter=`.'''

from ...core import Mint, mint_of, unminted
from ...core.frozen import answers, choice
from ...consumers.visualization.interface import Formatter, PlaceDrawing
from ...consumers.visualization.interface_types import Around, Notation, Text
from ..path import DictKey, Path


@answers(Path)
class PathDrawing(PlaceDrawing):
    '''A key path, written plainly, as jax spells it, minted under a `Mint`, grouped by
    its leading string keys.'''

    def spell_plain(self, path: Path) -> str:
        '''
        A string key as itself, dot-joined -- `['mda']['y1']` is `mda.y1`, `^cond['y1']` is
        `^cond.y1`. Any other key (an index, an attribute, a kind no surface mints) is
        written as jax writes it, so a name is never mis-written.
        '''
        out : list[str] = []
        for k in path.segments:
            if isinstance(k, DictKey) and isinstance(k.key, str):
                out.append(f'.{k.key}' if out else k.key)
            elif isinstance(k, Mint):
                out.append(f'.{k}' if out else str(k))
            else:
                out.append(str(k))
        return ''.join(out)

    def spelling(self, path: Path) -> str:
        return path.spelling

    def minted_from(self, path: Path) -> tuple[str, Path] | None:
        '''Its purpose and the path under it.'''
        m = mint_of(path)
        return None if m is None else (m, unminted(path))

    def namespace_of(self, path: Path, depth: int | None = None) -> tuple[str, ...]:
        '''
        Its leading string keys (a `DictKey` of a string, or an attribute), without its own
        last key, at most `depth` of them. A minted name is read through its namespace, so
        `^solve.physics.x` is in `physics`.
        '''
        keys: list[str] = []
        for key in unminted(path).segments:
            name = key.key if isinstance(key, DictKey) and isinstance(key.key, str) \
                else getattr(key, 'name', None)
            if not isinstance(name, str):
                break
            keys.append(name)
        own = max(len(keys) - 1, 0)
        return tuple(keys[:own if depth is None else min(depth, own)])


@choice(Formatter)
class NoFormat(Formatter):
    '''The name as jax spells it, notation included.'''

    def spell(self, path: Path) -> str:
        return path.spelling

    def set(self, source: str) -> str:
        return source

    def label(self, text: str) -> str:
        return text

    def text(self, text: Text) -> str:
        return text.plain

    def outright(self, path: Path) -> bool:
        return True                     # the spelling as it stands, minted or not

    def notation(self, m: Mint) -> Notation:
        return Around('{}')
