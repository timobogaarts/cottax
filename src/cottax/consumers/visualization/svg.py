'''The SVG every picture draws with: a label's runs and width, the shapes, escaping.'''

from __future__ import annotations

import re
from collections.abc import Callable

from .naming import MARKUP


class SvgDoc(str):
    '''
    SVG markup as a `str`, which a notebook also renders inline (`_repr_svg_`). `path` is
    the `.svg` file it was written to, or `None`; `cell` (the widest step column and
    tallest row) and `size` (width, height) are in px, for another drawing's `cell` and
    `canvas`.
    '''

    path : str | None = None
    cell : tuple[float, float] | None = None
    size : tuple[float, float] | None = None

    def _repr_svg_(self) -> str:                  # Jupyter rich display hook
        return str(self)


def xesc(s: str) -> str:
    '''XML-escape text content.'''
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def attr(s: str) -> str:
    '''Text as an attribute value, a newline kept as one.'''
    return xesc(s).replace('"', '&quot;').replace('\n', '&#10;')


BASE_FONT_SIZE = 13.0
'''The size a diagonal label is set at, and the unit every distance here is a multiple of.'''


CHAR_W     = 0.5615     # what one character measures, as a fraction of the font size
BOLD_W     = 1.09       # ... and how much wider it is at the weight a box name is set in
SHIFT_SIZE = 0.75       # a sub- or superscript, relative to the text it sits on
SHIFT_DY   = 0.33       # ... and how far off the baseline it moves


_TAG    = re.compile(r'</?([A-Za-z]+)(\s[^>]*)?>')
_FACE   = re.compile(r'FACE\s*=\s*"([^"]*)"', re.IGNORECASE)
ENTITY = re.compile(r'&[A-Za-z]+;|&#\d+;')


def markup_runs(s: str) -> list[tuple[str, bool, bool, int, str | None]]:
    '''
    `s` as `(text, italic, bold, shift, face)` runs, `shift` -1 for a subscript and +1
    for a superscript. Text outside a marked run is escaped here, since it is a name;
    text inside one was written as markup and is already escaped.
    '''
    out : list[tuple[str, bool, bool, int, str | None]] = []
    for k, part in enumerate(s.split(MARKUP)):
        if not part:
            continue
        if not k % 2:                                    # a name, drawn as it stands
            out.append((xesc(part), False, False, 0, None))
            continue
        italic = bold = shift = i = 0
        faces : list[str] = []
        for m in _TAG.finditer(part):
            if m.start() > i:
                out.append((part[i:m.start()], italic > 0, bold > 0,
                            max(-1, min(1, shift)), faces[-1] if faces else None))
            close = part[m.start() + 1] == '/'
            d     = -1 if close else 1
            tag   = m.group(1).upper()
            if   tag == 'I':   italic += d
            elif tag == 'B':   bold   += d
            elif tag == 'SUB': shift  -= d
            elif tag == 'SUP': shift  += d
            elif tag == 'FONT':
                # a stack: a face may be named inside another
                face = _FACE.search(m.group(2) or '')
                if close:
                    faces and faces.pop()
                else:
                    faces.append(face.group(1) if face else '')
            i = m.end()
        if i < len(part):
            out.append((part[i:], italic > 0, bold > 0, max(-1, min(1, shift)),
                        faces[-1] if faces else None))
    return out


def weight(run_bold: bool, face: str | None, bold: bool) -> bool:
    '''
    Whether one run is drawn bold, as `label` sets it: `bold` is the surrounding
    `<text>`'s weight, `run_bold` the run's own markup's. A run naming a face takes only
    its own: the named face may carry no bold.
    '''
    return run_bold if face else (bold or run_bold)


def tw(s: str, size: float = BASE_FONT_SIZE, *, bold: bool = False) -> float:
    '''
    The estimated width of `s` drawn at `size`, its markup counted as it is set; `bold`
    for a label set bold.
    '''
    return sum(len(ENTITY.sub('.', t)) * CHAR_W * size
               * (BOLD_W if weight(b, f, bold) else 1.0)
               * (SHIFT_SIZE if sh else 1.0)
               for t, _i, b, sh, f in markup_runs(s))


def label(s: str, size: float) -> str:
    '''
    `s` as the content of a `<text>`: a tspan per marked run, plain text where there is
    no markup. A sub- or superscript is a `dy` off the running offset (converters drop
    `baseline-shift`).
    '''
    runs = markup_runs(s)
    if all(not (i or b or sh or f) for _t, i, b, sh, f in runs):
        return ''.join(t for t, *_ in runs)
    out, cur = [], 0.0
    for t, italic, bold, sh, face in runs:
        if not (italic or bold or sh or face) and not cur:
            out.append(t)
            continue
        a = ''
        if face:
            # a face run at its own weight (`weight`)
            a += f' font-family="{face}"' + ('' if bold else ' font-weight="normal"')
        if italic:
            a += ' font-style="italic"'
        if bold:
            a += ' font-weight="bold"'
        if sh:
            a += f' font-size="{size * SHIFT_SIZE:.1f}"'
        dy   = -sh * SHIFT_DY * size - cur          # y grows down: a superscript is up
        cur += dy
        if abs(dy) > 0.01:
            a += f' dy="{dy:.2f}"'
        out.append(f'<tspan{a}>{t}</tspan>')
    return ''.join(out)


def hexagon(cx: float, cy: float, w: float, h: float, scale: float = 1.0,
             cap: float | None = None) -> str:
    '''
    The points of a `w` by `h` box with chevron sides, centred on `(cx, cy)`: the chevron
    is 12% of the width, at most `11 * scale` px (`scale`: the type's size over
    `BASE_FONT_SIZE`), or `cap` px when given.
    '''
    k = min(11.0 * scale, w * 0.12) if cap is None else cap
    pts = [(cx - w / 2 + k, cy - h / 2), (cx + w / 2 - k, cy - h / 2), (cx + w / 2, cy),
           (cx + w / 2 - k, cy + h / 2), (cx - w / 2 + k, cy + h / 2), (cx - w / 2, cy)]
    return ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)


def cut_hexagon(points: str, head: float, half: float) -> tuple[str, str]:
    '''
    The contour of `hexagon`'s `points` as two open paths, the left chevron and the
    right, stopping `half` either side of `head` (the middle of the shape's own column)
    on the top and bottom edges.
    '''
    (x0, yt), (x1, _), (xr, ym), (x3, yb), (x4, _), (xl, _) = (
        tuple(map(float, p.split(','))) for p in points.split())
    return (f'M{head - half:.1f},{yt:.1f} L{x0:.1f},{yt:.1f} L{xl:.1f},{ym:.1f} '
            f'L{x4:.1f},{yb:.1f} L{head - half:.1f},{yb:.1f}',
            f'M{head + half:.1f},{yt:.1f} L{x1:.1f},{yt:.1f} L{xr:.1f},{ym:.1f} '
            f'L{x3:.1f},{yb:.1f} L{head + half:.1f},{yb:.1f}')


def cut_rect(x: float, y: float, w: float, h: float, r: float,
              half: float) -> tuple[str, str]:
    '''A rounded rect's contour as two open halves, stopping `half` either side of the
    middle of its top and bottom edges.'''
    c = x + w / 2
    return (f'M{c - half:.1f},{y:.1f} L{x + r:.1f},{y:.1f} A{r:.1f},{r:.1f} 0 0 0 '
            f'{x:.1f},{y + r:.1f} L{x:.1f},{y + h - r:.1f} A{r:.1f},{r:.1f} 0 0 0 '
            f'{x + r:.1f},{y + h:.1f} L{c - half:.1f},{y + h:.1f}',
            f'M{c + half:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} A{r:.1f},{r:.1f} 0 0 1 '
            f'{x + w:.1f},{y + r:.1f} L{x + w:.1f},{y + h - r:.1f} A{r:.1f},{r:.1f} 0 0 1 '
            f'{x + w - r:.1f},{y + h:.1f} L{c + half:.1f},{y + h:.1f}')


type Measure = Callable[[str, float, bool], tuple[float, float]]
'''What `render_xdsm_svg`'s `measure` is: `(label, size, bold) -> (ink_left, ink_width)`,
in px and relative to the pen origin, for a label already carrying its `MARKUP` runs.'''


# how far a glyph's ink reaches from the baseline, as a fraction of the font size, between
# a serif face's and a sans one's
X_HEIGHT = 0.50         # `a`, `x`: a lowercase with nothing sticking up
ASCENDER = 0.71         # `b`, `V`, `1`: anything that does
DESCENDER = 0.21        # `y`, `(`, `,`: and anything hanging below

# the letters stopping at the x-height, and those hanging below the baseline; any other
# reaches the ascender
SHORT = frozenset('acemnorsuvwxzgpqy.·' + 'αεικνοπστυωγημρχς')
DESCEND = frozenset('gjpqy(),;[]{}/' + 'βγηζμξρςφχψ')


def _ink(s: str, size: float) -> tuple[float, float]:
    '''
    How far `s`'s ink reaches above and below the baseline, in px, estimated from its
    letters: what a label is centred on.
    '''
    top = bot = 0.0
    for t, _i, _b, sh, _f in markup_runs(s):
        text = ENTITY.sub('X', t)
        if not text:
            continue
        scale = SHIFT_SIZE if sh else 1.0
        shift = sh * SHIFT_DY               # up is positive here: a subscript is below
        a = X_HEIGHT if all(c in SHORT for c in text) else ASCENDER
        d = DESCENDER if any(c in DESCEND for c in text) else 0.0
        top = max(top, (a * scale + shift) * size)
        bot = max(bot, (d * scale - shift) * size)
    return top, bot


def _baseline(s: str, y: float, size: float) -> float:
    '''The baseline that puts `s`'s ink on `y`. See `_ink`.'''
    top, bot = _ink(s, size)
    return y + (top - bot) / 2


def _text(text: str, x: float, y: float, size: float, cls: str, *,
          anchor: str = 'middle', bold: bool = False, extra: str = '',
          span: tuple[float, float] | None = None) -> str:
    '''
    One `<text>` of class `cls`, anchored on `(x, y)` by `anchor`. A label holding tspans,
    or any label given its `span` (`(ink_left, ink_width)`, `_span`), is placed from its
    start instead, since cairosvg anchors each tspan apart.
    '''
    body = label(text, size)
    if span is not None or '<tspan' in body:
        ink, w = span if span is not None else (0.0, tw(text, size, bold=bold))
        x = (x - w / 2 if anchor == 'middle' else x - w if anchor == 'end' else x) - ink
        anchor = 'start'
    at = '' if anchor == 'start' else f' text-anchor="{anchor}"'
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}"{at}{extra}>{body}</text>'


def _plain(s: str) -> str:
    '''`s` without its markup, for a `<title>`.'''
    return ''.join(t for t, *_ in markup_runs(s))


def _span(s: str, size: float = BASE_FONT_SIZE, *, bold: bool = False,
          measure: Measure | None = None) -> tuple[float, float]:
    '''
    `(ink_left, ink_width)` of `s` drawn at `size`, relative to the pen origin: by
    `measure` when given, else `(0, tw(...))`.
    '''
    if measure is not None:
        return measure(s, size, bold)
    return 0.0, tw(s, size, bold=bold)
