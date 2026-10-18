r'''
Names set as LaTeX: a `Formatter` whose spellings are LaTeX rendered into the markup every
drawing here reads (`naming.MARKUP`), so one formatter serves the XDSMs, `dot` and the N2.

The supported LaTeX is the subset such markup can carry -- sub- and superscripts, greek,
upright / bold / script alphabets, accents as combining marks, the common relations and
operators. Anything else is written through as upright text rather than raising, so an
unknown macro is legible in the picture instead of failing the build.

A name is read as math (`y_1` is already what LaTeX would set), unless it holds a `$`, in
which case it alternates text / math as a LaTeX line does (`Discipline $1$`).
'''

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping

from ....core import Label, Mint
from ....core.frozen import choice
from ..interface import Formatter
from ..lookup import spell_plain
from ..interface_types import Around, Notation, Text
from ..naming import MARKUP, Notations, notation_of
from ..svg import xesc

GREEK = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ϵ',
    'varepsilon': 'ε', 'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'vartheta': 'ϑ',
    'iota': 'ι', 'kappa': 'κ', 'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ',
    'pi': 'π', 'varpi': 'ϖ', 'rho': 'ρ', 'varrho': 'ϱ', 'sigma': 'σ',
    'varsigma': 'ς', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'ϕ', 'varphi': 'φ',
    'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ', 'Pi': 'Π',
    'Sigma': 'Σ', 'Upsilon': 'Υ', 'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
}

# An accent is a *combining* mark after the glyph it sits on: it needs no element at all.
ACCENT = {
    'hat': '̂', 'widehat': '̂', 'bar': '̄', 'overline': '̄',
    'tilde': '̃', 'widetilde': '̃', 'dot': '̇', 'ddot': '̈',
    'vec': '⃗', 'check': '̌', 'breve': '̆', 'acute': '́',
    'grave': '̀',
}

SYMBOL = {
    'cdot': '·', 'times': '×', 'ast': '∗', 'star': '⋆', 'circ': '∘', 'pm': '±',
    'mp': '∓', 'to': '→', 'rightarrow': '→', 'longrightarrow': '⟶',
    'leftarrow': '←', 'Rightarrow': '⇒', 'mapsto': '↦', 'leq': '≤', 'le': '≤',
    'geq': '≥', 'ge': '≥', 'neq': '≠', 'ne': '≠', 'approx': '≈', 'equiv': '≡',
    'sim': '∼', 'propto': '∝', 'in': '∈', 'notin': '∉', 'subset': '⊂',
    'subseteq': '⊆', 'cup': '∪', 'cap': '∩', 'forall': '∀', 'exists': '∃',
    'partial': '∂', 'nabla': '∇', 'infty': '∞', 'sum': '∑', 'prod': '∏',
    'int': '∫', 'sqrt': '√', 'hbar': 'ℏ', 'ell': 'ℓ', 'Re': 'ℜ', 'Im': 'ℑ',
    'prime': '′', 'dots': '…', 'ldots': '…', 'cdots': '⋯', 'langle': '⟨',
    'rangle': '⟩', 'lVert': '‖', 'rVert': '‖', 'emptyset': '∅', 'oplus': '⊕',
    'otimes': '⊗', 'perp': '⊥', 'angle': '∠',
}

FALLBACK = frozenset('′ℏℑℓℜ↦⇒∀∂∃∅∇∈∉∏∑∓∘∝∠∩∪∫∼≈≠≡≤≥⊂⊆⊕⊗⊥⋆⋯⟨⟩⟶')
'''The symbols Latin Modern Roman does not carry, set in `MATH_FACE` as `\\mathcal` is: a
renderer that cannot find a glyph draws nothing (`g \\leq 0` once printed as `g   0`).'''

SPACE = {',': ' ', ':': ' ', ';': ' ', ' ': ' ', 'quad': ' ',
         'qquad': '  ', '!': ''}
'''Spacing macros. `\\!` is negative and has no counterpart, so it is dropped.'''

_CAL = {
    'A': '\U0001d49c', 'B': 'ℬ', 'C': '\U0001d49e', 'D': '\U0001d49f', 'E': 'ℰ',
    'F': 'ℱ', 'G': '\U0001d4a2', 'H': 'ℋ', 'I': 'ℐ', 'J': '\U0001d4a5',
    'K': '\U0001d4a6', 'L': 'ℒ', 'M': 'ℳ', 'N': '\U0001d4a9', 'O': '\U0001d4aa',
    'P': '\U0001d4ab', 'Q': '\U0001d4ac', 'R': 'ℛ', 'S': '\U0001d4ae',
    'T': '\U0001d4af', 'U': '\U0001d4b0', 'V': '\U0001d4b1', 'W': '\U0001d4b2',
    'X': '\U0001d4b3', 'Y': '\U0001d4b4', 'Z': '\U0001d4b5',
}

_BB = {c: chr(0x1d538 + ord(c) - ord('A')) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
_BB.update({'C': 'ℂ', 'H': 'ℍ', 'N': 'ℕ', 'P': 'ℙ', 'Q': 'ℚ', 'R': 'ℝ', 'Z': 'ℤ'})

MATH_FACE = 'STIXGeneral, serif'
'''The face `\\mathcal`, `\\mathbb`, lowercase greek and `FALLBACK` are set in: a text face
carries none of them, and a renderer that cannot find a glyph draws nothing rather than
reaching for another font.'''

ARGUMENT = frozenset(ACCENT) | {
    'mathrm', 'text', 'textrm', 'operatorname', 'mathsf', 'mathbf', 'bm',
    'boldsymbol', 'mathit', 'mathnormal', 'mathcal', 'mathscr', 'mathbb',
}
'''The macros that take a braced argument, so `_\\text{max}` reads as one token.'''

def _group(s: str, i: int) -> tuple[str, int]:
    '''The argument starting at `i` -- a brace group, a macro, or one character, as TeX
    reads `y_2` -- and where it ends.'''
    if i >= len(s):
        return '', i
    if s[i] == '{':
        depth, j = 1, i + 1
        while j < len(s) and depth:
            depth += (s[j] == '{') - (s[j] == '}')
            j += 1
        return s[i + 1:j - 1], j
    if s[i] == '\\':
        name, j = _macro(s, i)
        if name in ARGUMENT and j < len(s) and s[j] == '{':
            _, j = _group(s, j)
        return s[i:j], j
    return s[i], i + 1


def _macro(s: str, i: int) -> tuple[str, int]:
    m = re.match(r'\\([A-Za-z]+|.)', s[i:])
    if m is None:
        return '', i + 1
    return m.group(1), i + m.end()


def _accented(inner: str, mark: str) -> str:
    '''`mark` on `inner`'s first glyph -- past any opening tag -- composed where unicode
    has a composed form, since cairo's toy API draws a bare combining mark as a blank.'''
    marked = re.sub(r'^((?:<[^>]*>)*(?:&[A-Za-z]+;|.))', r'\1' + mark, inner, count=1)
    return unicodedata.normalize('NFC', marked)


def _ident(word: str) -> str:
    '''One letter is a variable, set italic; several are a name, set upright.'''
    return f'<I>{xesc(word)}</I>' if len(word) == 1 else xesc(word)


def _alphabet(arg: str, table: Mapping[str, str]) -> str:
    return (f'<FONT FACE="{MATH_FACE}">'
            + xesc(''.join(table.get(c, c) for c in arg)) + '</FONT>')


def _math(s: str) -> str:
    '''One math-mode string as markup.'''
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '\\':
            name, i = _macro(s, i)
            if name in ACCENT:
                arg, i = _group(s, i)
                out.append(_accented(_math(arg), ACCENT[name]))
            elif name in ('mathrm', 'text', 'textrm', 'operatorname', 'mathsf'):
                arg, i = _group(s, i)
                out.append(xesc(arg))
            elif name in ('mathbf', 'bm', 'boldsymbol'):
                arg, i = _group(s, i)
                out.append(f'<B>{_math(arg)}</B>')
            elif name in ('mathit', 'mathnormal'):
                arg, i = _group(s, i)
                out.append(f'<I>{xesc(arg)}</I>')
            elif name in ('mathcal', 'mathscr'):
                arg, i = _group(s, i)
                out.append(_alphabet(arg, _CAL))
            elif name == 'mathbb':
                arg, i = _group(s, i)
                out.append(_alphabet(arg, _BB))
            elif name in ('left', 'right', 'big', 'Big', 'bigg', 'Bigg'):
                continue
            elif name in GREEK:
                g = GREEK[name]
                out.append(xesc(g) if not name[:1].islower()
                           else f'<FONT FACE="{MATH_FACE}"><I>{g}</I></FONT>')
            elif name in SYMBOL:
                sym = SYMBOL[name]
                out.append(xesc(sym) if sym not in FALLBACK
                           else f'<FONT FACE="{MATH_FACE}">{xesc(sym)}</FONT>')
            elif name in SPACE:
                out.append(xesc(SPACE[name]))
            else:
                out.append(xesc(name))                      # legible, rather than raising
        elif c in '_^':
            arg, i = _group(s, i + 1)
            tag = 'SUB' if c == '_' else 'SUP'
            out.append(f'<{tag}>{_math(arg)}</{tag}>')
        elif c == '{':
            arg, i = _group(s, i)
            out.append(_math(arg))
        elif c == '}':
            i += 1
        elif c == '~':
            out.append(' ')
            i += 1
        elif c.isalpha():
            j = i
            while j < n and s[j].isalpha():
                j += 1
            out.append(_ident(s[i:j]))
            i = j
        else:
            out.append(xesc(c) if c not in FALLBACK
                       else f'<FONT FACE="{MATH_FACE}">{xesc(c)}</FONT>')
            i += 1
    return ''.join(out)


def latex_to_markup(s: str) -> str:
    '''
    A LaTeX-ish string as markup: all math without a `$` (a variable name already is),
    alternating text / math from text mode with one.
    '''
    if '$' not in s:
        return _math(s)
    parts = s.split('$')
    return ''.join(_math(p) if k % 2 else xesc(p) for k, p in enumerate(parts))


_DIGITS = re.compile(r'(?<=[A-Za-z])(\d+)')


def subscripted(name: str) -> str:
    '''
    A declared name as the notation it stands for: `V1` is `V_{1}`, `S1S2` is
    `S_{1}S_{2}`. An identifier cannot carry a subscript, so a digit against a letter is
    one; anything holding LaTeX of its own is the caller's spelling and passes through.
    '''
    return name if set(name) & set('\\$_^{}') else _DIGITS.sub(r'_{\1}', name)


def tex(s: str | None) -> str | None:
    '''A LaTeX string as a label a drawing sets rather than prints.'''
    return s if s is None or MARKUP in s else MARKUP + latex_to_markup(s) + MARKUP


@choice(Formatter)
class LatexFormatter(Formatter):
    r'''
    Every name spelt as LaTeX: `names` maps a plain spelling (`spell`, `spell_plain` by
    default) to the LaTeX that replaces it -- `{'V1': r'\mathcal{D}_1'}` -- and anything
    not in it is read as LaTeX as it stands, digits made subscripts (`subscripted`).

    A minted name given in `names` is named **outright**: drawn as given, and not under
    the notation of its purpose -- `{'^solve.S1': 'S_1'}` draws `S_1`. Every other minted
    name is written in its purpose's notation, in LaTeX: `notation` by `Mint` class --
    `{Solve: Around(r'S_{{}}')}` -- else its `MintDrawing`'s, else as
    `\mathrm{<spelling>}(·)`.
    '''

    def __init__(self, names: Mapping[str, str] | None = None,
                 notation: Notations | None = None,
                 spell: Callable[[Label], str] = spell_plain) -> None:
        self.names = dict(names or {})
        self.notations = dict(notation or {})
        self.written = spell

    def spell(self, path: Label) -> str:
        written = self.written(path)
        given = self.names.get(written)
        return subscripted(written) if given is None else given

    def set(self, source: str) -> str:
        return MARKUP + latex_to_markup(source) + MARKUP

    def label(self, text: str) -> str:
        return tex(text)

    def text(self, text: Text) -> str:
        return tex(text.latex)

    def outright(self, path: Label) -> bool:
        return self.written(path) in self.names

    def notation(self, m: Mint) -> Notation:
        return notation_of(self.notations, m, 'latex', r'\mathrm{{{}}}({{}})')
