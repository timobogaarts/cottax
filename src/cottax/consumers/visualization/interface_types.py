'''
What a drawing is answered with: a condition node's `Drawn`, its `Kind` and `Slot`s; and
the `Notation`s a `Formatter` writes a minted name in.
'''

from __future__ import annotations

from ...core.frozen import Frozen
from ...core import Place


class Kind(Frozen):
    '''
    A category of problem: a language's own, which the drawing names nowhere. `name` is
    its short name, what a struct calls it; two categories with one name are refused in
    one picture. `legend` is what a key says, `title` what hovering says. `slotted`: its
    reads are of mixed kind, so its slots are drawn by default. `colour` is its
    `(fill, stroke)` on a light ground, which a style draws as it is or derives its own
    from (`Style.ground`), written on each shape drawn in it.
    '''

    name    : str
    legend  : str
    title   : str
    slotted : bool
    colour  : tuple[str, str]


class Text(Frozen):
    '''A text naming no place, written in each source a `Formatter` sets from: `plain`,
    and `latex` -- `Text('≤ 0', r'\\leq 0')`.'''

    plain : str
    latex : str


class Slot(Frozen):
    '''
    Some of a problem's reads under one label. `label` is texts and places in order --
    `(Text('min', r'\\min'),)`, `(Text('= ', '= '), u)`, `(lhs, Text(' = ', ' = '), rhs)` --
    set by the drawing: a text as `Formatter.text`, a place as its shown name.
    '''

    label : tuple[Text | Place, ...]
    reads : tuple[Place, ...]


class Drawn(Frozen):
    '''How one condition node is drawn: its category (`None`: none, a grey problem) and
    its slots, in the order its driver is handed them.'''

    kind  : Kind | None
    slots : tuple[Slot, ...]


class Notation(Frozen):
    '''How a name minted in a namespace is written over the place it stands for.'''


class Around(Notation):
    '''
    Text around the inner name: `template` in the formatter's own source (plain text for
    `PlainFormat`, LaTeX for a `LatexFormatter`), its first `{}` the inner --
    `Around('solve({})')`, `Around('{}′')`, `Around(r'S_{{}}')`.
    '''

    template : str


class Accent(Notation):
    '''
    A mark over the inner name, whatever it is: `'hat'`, `'bar'`, `'tilde'`, `'dot'`.
    `spelled` (an `Around` template) is how it is written where no mark can be drawn --
    inside another notation, or in plain text.
    '''

    mark    : str
    spelled : str


__all__ = ['Kind', 'Text', 'Slot', 'Drawn', 'Notation', 'Around', 'Accent']
