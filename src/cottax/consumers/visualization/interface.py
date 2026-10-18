'''
What drawing asks. Of a node class: how a condition node is drawn (`NodeDrawing`) [answer].
Of a place class: how a place is written (`PlaceDrawing`) [answer]. Of a purpose: how a name
derived for it is written (`MintDrawing`) [answer]. Of a study: how a name is written
(`Formatter`) [choice].
'''

from abc import abstractmethod
from typing import Any

from ...core.frozen import AbstractVar, Answer, Choice
from ...core import ConditionalNode, Label, Mint, Place
from .interface_types import Drawn, Notation, Text


# ============================================================================ registered

class NodeDrawing(Answer):
    '''How a condition node is drawn.'''

    about = ConditionalNode

    @abstractmethod
    def drawn(self, node: Any) -> Drawn:
        '''Its category and slots.'''


class PlaceDrawing(Answer):
    '''How a place is written in a drawing, and a node name of its kind.'''

    about = (Place, Label)

    @abstractmethod
    def spell_plain(self, place: Label) -> str:
        '''A name written plainly: a key within one picture, its identity there whatever it
        shows. A picture refuses two places, or two node names, sharing one
        (`refusals.SpelledAlike`).'''

    @abstractmethod
    def spelling(self, place: Label) -> str:
        '''A name as its own kind writes it in full, in a one-line summary or a refusal.'''

    @abstractmethod
    def minted_from(self, place: Label) -> tuple[Mint, Place] | None:
        '''`(purpose, what it stands for)` for a minted name, else `None`.'''

    @abstractmethod
    def namespace_of(self, place: Label, depth: int | None = None) -> tuple[str, ...]:
        '''The group a name declares: the names it sits under, at most `depth` of them.'''


class MintDrawing(Answer):
    '''How a name derived for a purpose is written over what it stands for: `plain` in plain
    text, `latex` in LaTeX. A purpose nobody answers for is written by its spelling,
    `copy(·)`.'''

    about = Mint

    plain : AbstractVar[Notation]
    latex : AbstractVar[Notation]


# ============================================================================ chosen

class Formatter(Choice):
    '''
    How a drawing writes a name. `spell(path)` is the name of a place in the formatter's
    own source, `set(source)` that source as what a drawing sets (`MARKUP` runs),
    `label(text)` a text naming no place written in its own source (a boundary box), and
    `text(text)` one written in each (a slot's, `Text`), set from its own. A minted name
    is written by `shown`: `outright(path)` says the formatter names it itself, and
    otherwise `notation(m)` is how its purpose is written over what it stands for: its
    `MintDrawing`, unless the formatter overrides it. Named by a drawing's `formatter=`.

    Handed in as:

    render_xdsm_svg(graph, formatter=PlainFormat(notation={Copy: Around('{}′')})) where the
    argument is a Formatter: names as words, a cut's copy primed.
    '''

    @abstractmethod
    def spell(self, path: Place) -> str:
        ...

    @abstractmethod
    def set(self, source: str) -> str:
        ...

    @abstractmethod
    def label(self, text: str) -> str:
        ...

    @abstractmethod
    def text(self, text: Text) -> str:
        ...

    @abstractmethod
    def outright(self, path: Place) -> bool:
        ...

    @abstractmethod
    def notation(self, m: Mint) -> Notation:
        ...


__all__ = ['NodeDrawing', 'PlaceDrawing', 'MintDrawing', 'Formatter']
