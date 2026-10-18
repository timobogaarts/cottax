'''How a drawing writes a name: the plain formatter, a name as shown, and its markup.'''

from __future__ import annotations

import html
import re
from collections.abc import Mapping

from ...core import Copy, Label, Mint
from ...core.frozen import answer, answers, choice
from .interface import Formatter, MintDrawing
from .interface_types import Accent, Around, Notation, Text
from .lookup import minted_from, spell_plain


type Notations = Mapping[type[Mint], Notation]
'''A formatter's own notation per purpose, by its `Mint` class, over the `MintDrawing`s.'''


def notation_of(notations: Notations, m: Mint, medium: str, otherwise: str) -> Notation:
    '''`m`'s notation in `medium` (`'plain'`, `'latex'`): the formatter's own along its class's
    ancestry, else its `MintDrawing`'s, else `otherwise` around its spelling.'''
    for level in type(m).__mro__:
        if level in notations:
            return notations[level]
    if (drawing := answer(MintDrawing, m, None)) is not None:
        return getattr(drawing, medium)
    return Around(otherwise.format(str(m).removeprefix('^')))


@answers(Copy)
class CopyDrawing(MintDrawing):
    r'''A cut's copy wears a hat: `\hat{y}`.'''

    plain = Around('hat({})')
    latex = Accent('hat', r'\hat{{}}')


@choice(Formatter)
class PlainFormat(Formatter):
    '''
    Every name by `spell_plain`, a minted one in its purpose's notation (`notation` keyed by
    the `Mint` class, else its `MintDrawing`), else as `<spelling>(·)` -- what a drawing
    writes unless told otherwise.
    '''

    def __init__(self, notation: Notations | None = None) -> None:
        self.notations = dict(notation or {})

    def spell(self, path: Label) -> str:
        return spell_plain(path)

    def set(self, source: str) -> str:
        return source

    def label(self, text: str) -> str:
        return text

    def text(self, text: Text) -> str:
        return text.plain

    def outright(self, path: Label) -> bool:
        return False

    def notation(self, m: Mint) -> Notation:
        return notation_of(self.notations, m, 'plain', '{}({{}})')


# ============================================================================ a name, shown
type Shown = tuple[str, str | None]
'''What a drawing sets for a name: its markup, and the mark drawn over it (or `None`).'''

MARKS = {'hat': '\u0302', 'bar': '\u0304', 'tilde': '\u0303', 'dot': '\u0307'}
'''Each mark as a combining character, for a medium that cannot draw it.'''


def _around(template: str, inner: str) -> str:
    i = template.find('{}')
    return template[:i] + inner + template[i + 2:] if i >= 0 else template + inner


def _source(path: Label, formatter: Formatter, outer: bool) -> tuple[str, str | None]:
    minted = None if formatter.outright(path) else minted_from(path)
    if minted is None or not minted[1].segments:
        return formatter.spell(path), None
    m, under = minted
    note = formatter.notation(m)
    inner, _ = _source(under, formatter, outer=False)
    if isinstance(note, Accent):
        return (inner, note.mark) if outer else (_around(note.spelled, inner), None)
    return _around(note.template, inner), None


def shown(path: Label, formatter: Formatter) -> Shown:
    '''
    `path` as a drawing sets it: the formatter's own name for it where it gives one
    (`outright`), else its namespace's notation over what it stands for (recursively: a
    mint over a mint), else its spelling. Decided once, here, for every drawing.
    '''
    source, mark = _source(path, formatter, outer=True)
    return formatter.set(source), mark


def marked(name: Shown) -> str:
    '''A shown name as one run of markup, its mark a combining character on the first
    letter of its text -- for a medium that cannot draw the mark itself.'''
    markup, mark = name
    if mark is None:
        return markup
    char = MARKS.get(mark, '\u0302')
    parts = markup.split(MARKUP)
    for k, part in enumerate(parts):
        if k % 2:                                            # a marked run: past its tags
            m = re.search(r'(?:^|>)([^<&])', part)
            if m:
                parts[k] = part[:m.end(1)] + char + part[m.end(1):]
                break
        elif part:
            parts[k] = part[0] + char + part[1:]
            break
    return MARKUP.join(parts)


# ------------------------------------------------------- a label a formatter marked up
# A label is text; a `Formatter` wraps what it set as markup in `MARKUP`. The tag subset is
# graphviz's HTML-like one, so one formatter serves the SVG and the DOT pictures.

MARKUP = '\x01'
'''What a `Formatter` wraps around a spelling it has already set as markup.

`<I>`, `<B>`, `<SUB>`, `<SUP>`, `<FONT FACE="...">` and the XML entities are what a
marked run may hold; any other tag is dropped and its text kept, so an unknown one is
legible rather than fatal. `\\x01` cannot occur in a name and never reaches the drawing.'''


def markup_html(s: str) -> str:
    '''
    Markup as HTML, for a page's tooltip and key: a name escaped as the text it is, a
    marked run's `<I>`, `<B>`, `<SUB>`, `<SUP>` kept as the HTML they already are, and a
    named face a `<span>` in it.
    '''
    out = []
    for k, part in enumerate(s.split(MARKUP)):
        if not k % 2:
            out.append(html.escape(part, quote=False))
            continue
        part = re.sub(r'<FONT\s+FACE\s*=\s*"([^"]*)"\s*>', r'<span style="font-family:\1">', part,
                      flags=re.IGNORECASE)
        part = re.sub(r'</FONT>', '</span>', part, flags=re.IGNORECASE)
        out.append(re.sub(r'</?(I|B|SUB|SUP)>', lambda m: m.group(0).lower(), part))
    return ''.join(out)
