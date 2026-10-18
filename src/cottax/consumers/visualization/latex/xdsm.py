r'''
XDSM figures for a LaTeX document: `render_xdsm_svg` set in the document's face, measured
in the face it is drawn in, and written as the PDF the document includes -- at a size
said as the page does (`page='half', text_pt=9`).

No TeX engine reads SVG, so the drawing becomes a PDF here with `cairosvg`, which needs no
Inkscape and keeps it vector. An SVG unit is 1/96 in and a PDF point 1/72 in, so a figure
included at natural size sets its text at `font_size` px = `Layout.body_pt`: **include it
without a `width=`**, or use `save_figure` and the width it hands back.

Needs `cairosvg` (and `cairocffi`, which it brings).
'''

from __future__ import annotations

import functools
import html
import pathlib
import re
from collections.abc import Callable, Sequence

from ....core.frozen import Frozen
from ..style import TERRACOTTA, Style, style_of
from ..svg import SHIFT_SIZE, SvgDoc, markup_runs, weight
from ..xdsm import render_xdsm_svg
from .text import LatexFormatter, tex

LATIN_MODERN = 'LMRoman10, serif'
'''Latin Modern, which is Computer Modern, as **fontconfig** names it -- the converter asks
fontconfig, and `fc-match 'Latin Modern Roman'` answers another face entirely.'''

PX_PER_PT = 96 / 72.27
'''A CSS px per TeX point.'''


class Layout(Frozen):
    '''The page a figure is printed on: `\\linewidth` and the body size in TeX points, and
    the face the body is set in.'''

    linewidth_pt : float
    body_pt      : float
    font         : str


REVTEX = Layout(linewidth_pt=510.0, body_pt=10.0, font=LATIN_MODERN)
'''REVTeX 4-2, one column, at 10 pt.'''

PAGE = {'full': 1.0, 'half': 0.48, 'third': 0.32, 'quarter': 0.24}
'''A named printed width as a fraction of `\\linewidth` -- what to write in the
`\\includegraphics[width=...]`. `half` is `0.48`: two panels need a gutter.'''

PANEL_FONT_SIZE = 20.0
'''What a panel printed at a fraction of the line is drawn at, before `font_for` solves.'''

INK_PER_EM = 0.10
'''`line_scale` per px of `font_size`, so a figure keeps one weight at any size.'''

RUNAWAY_ROUNDS = 40
RUNAWAY_FONT = 200.0
'''`font_for` gives up after this many rounds, or once the font passes this size: the
printed type asymptotes (the boxes grow with the font, `gap` does not), and a size past
the asymptote is walked towards forever.'''


# ------------------------------------------------------------------ measuring the type

@functools.cache
def _context():
    import cairocffi
    return cairocffi.Context(cairocffi.SVGSurface(None, 8, 8))


@functools.lru_cache(maxsize=4096)
def _extents(text: str, face: str, size: float, italic: bool, bold: bool):
    '''One run's `(x_bearing, width, x_advance)`, through the cairo toy API cairosvg draws
    with: what this measures is what that will draw.'''
    ctx = _context()
    ctx.select_font_face(face.split(',')[0].strip().strip('\'"'), int(italic), int(bold))
    ctx.set_font_size(size)
    x_bearing, _, width, _, x_advance, _ = ctx.text_extents(text)
    return x_bearing, width, x_advance


class CairoMeasure(Frozen):
    '''
    `render_xdsm_svg`'s `measure`: `(label, size, bold) -> (ink_left, ink_width)` for a
    label carrying its markup, in `font` as cairo will draw it. A box is cut to that and
    the label placed on it -- an italic capital's side bearing is several px.
    '''

    font : str

    def __call__(self, label: str, size: float, bold: bool) -> tuple[float, float]:
        left, right, pen = None, 0.0, 0.0
        for text, italic, run_bold, shift, face in markup_runs(label):
            text = html.unescape(text)
            if not text:
                continue
            run_size = size * SHIFT_SIZE if shift else size
            x_bearing, width, x_advance = _extents(text, face or self.font, run_size,
                                                   italic, weight(run_bold, face, bold))
            # A space has no ink, but air asked for is kept: a space extends the ink to
            # its advance, at either end of a run too (the separator `, `).
            if text.isspace():
                left = pen if left is None else left
                right = max(right, pen + x_advance)
            else:
                left = pen + x_bearing if left is None else left
                right = max(right, pen + x_bearing + width)
                if text[-1].isspace():
                    right = max(right, pen + x_advance)
                if text[0].isspace():
                    left = min(left, pen)
            pen += x_advance
        return (0.0, 0.0) if left is None else (left, right - left)


# ------------------------------------------------------------------------- the drawing

INPUTS, OUTPUTS = r'\mathcal{I}', r'\mathcal{O}'
'''The two boundary boxes: what the model is given, and what it produces.'''


def to_svg(graph, *, layout: Layout = REVTEX, **kw) -> SvgDoc:
    '''
    `render_xdsm_svg` with a figure's defaults over its own: `layout`'s face at its body
    size, measured (`CairoMeasure`), a `LatexFormatter`, the boundary boxes `\\mathcal{I}`
    and `\\mathcal{O}`, and no key (the caption says what a shape means). Every argument is
    `render_xdsm_svg`'s.
    '''
    kw.setdefault('font', layout.font)
    kw.setdefault('font_size', layout.body_pt * PX_PER_PT)
    kw.setdefault('formatter', LatexFormatter())
    kw.setdefault('measure', CairoMeasure(kw['font']))
    kw.setdefault('legend', False)
    kw.setdefault('input_label', INPUTS)
    kw.setdefault('output_label', OUTPUTS)
    return render_xdsm_svg(graph, **kw)


def _write(svg: str | bytes, path: pathlib.Path) -> pathlib.Path:
    import cairosvg
    data = svg.encode() if isinstance(svg, str) else svg
    fmt = path.suffix.lstrip('.').lower() or 'svg'
    if fmt == 'svg':
        path.write_bytes(data)
    elif fmt == 'pdf':
        cairosvg.svg2pdf(bytestring=data, write_to=str(path))
    elif fmt == 'png':
        cairosvg.svg2png(bytestring=data, write_to=str(path), scale=3)
    else:
        raise ValueError(f'{path.suffix!r}: write a .pdf, .svg or .png')
    return path


def save(graph, path, **kw) -> pathlib.Path:
    '''`to_svg` written as the format `path`'s suffix names: `.pdf`, `.svg` or `.png`.'''
    return _write(str(to_svg(graph, **kw)), pathlib.Path(path))


def _panel_kw(kw: dict, rest: Sequence[dict]) -> dict:
    return {**kw, **(rest[0] if rest else {})}


def save_panels(panels, *, formats=('pdf', 'svg'), **kw) -> list[pathlib.Path]:
    '''
    Several drawings as one figure's panels, at **one zoom**: every panel padded to the
    largest (`canvas`), so two printed at `0.48\\linewidth` set their type alike.
    `panels` is `(stem, graph)` or `(stem, graph, kwargs)`; `kw` is `to_svg`'s for every
    panel, a panel's own merged over it. Returns every path written.
    '''
    kw.setdefault('font_size', PANEL_FONT_SIZE)
    kw.setdefault('line_scale', INK_PER_EM * kw['font_size'])
    canvas = kw.pop('canvas', None)
    prepared = [(pathlib.Path(stem), g, _panel_kw(kw, rest)) for stem, g, *rest in panels]
    if canvas is None:
        sizes = [to_svg(g, **k).size for _, g, k in prepared]
        canvas = (max(s[0] for s in sizes), max(s[1] for s in sizes))
    return [save(g, stem.with_suffix('.' + fmt), **{**k, 'canvas': canvas})
            for stem, g, k in prepared for fmt in formats]


def _settled(width_at: Callable[[float], float], text_pt: float, printed: float, start: float,
             refused: Callable[[float], Exception]) -> float:
    '''The font, from `start`, at which a drawing `width_at(font)` px wide printed `printed`
    pt wide sets its type at `text_pt`: a fixed point, `refused(font)` raised once the font
    runs past `RUNAWAY_FONT`; after `RUNAWAY_ROUNDS`, where it has got to.'''
    font = start
    for _ in range(RUNAWAY_ROUNDS):
        nxt = text_pt * width_at(font) / printed
        if abs(nxt - font) < 0.01:
            return font
        font = nxt
        if font > RUNAWAY_FONT:
            raise refused(font)
    return font                                      # settling, just slowly


def _solved_font(panels, text_pt: float, page, layout: Layout, kw: dict) -> dict:
    '''`kw` with `layout`, the font solved for unless given (`font_for`), and the ink
    following it.'''
    kw = {**kw, 'layout': layout}
    if 'font_size' not in kw:
        kw['font_size'] = font_for(panels, text_pt, page, **kw)
    kw.setdefault('line_scale', INK_PER_EM * kw['font_size'])
    return kw


def font_for(panels, text_pt: float, page='half', *, layout: Layout = REVTEX,
             start: float = PANEL_FONT_SIZE, **kw) -> float:
    '''
    The `font_size` to draw at so the labels print at `text_pt` once scaled to `page` of
    the line. The printed type is `font_size * printed / drawn width`, and the drawn width
    follows the font, so it is solved for; a size past what the panels asymptote to (the
    boxes grow with the font, `gap` does not) raises, naming the way out.
    '''
    printed = PAGE.get(page, page) * layout.linewidth_pt

    def widest(font):
        return max(to_svg(g, layout=layout, **{**_panel_kw(kw, rest), 'font_size': font}).size[0]
                   for _s, g, *rest in panels)

    return _settled(widest, text_pt, printed, start, lambda font: ValueError(
        f'text_pt={text_pt} is out of reach at page={page!r}: these panels print their '
        f'labels at about {font * printed / widest(font):.1f} pt however large they are '
        f'drawn. Give the figure more width or ask for smaller labels.'))


def save_figure(panels, *, page='half', text_pt=9.0, align='top-left', match_height=False,
                layout: Layout = REVTEX, formats=('pdf', 'svg'), **kw):
    '''
    `save_panels` told what the figure is *for*: `page` (a name in `PAGE` or a fraction of
    `\\linewidth`) and the `text_pt` its labels print at there. The font is solved for
    (`font_for`), the ink follows it, and the panels share one canvas **width** --
    `\\includegraphics` scales by width -- aligned top-left; `match_height` pins the
    height too, for a row that must line up on the page. Returns the paths and the
    `width=` to print them at.
    '''
    kw = _solved_font(panels, text_pt, page, layout, kw)
    if 'canvas' not in kw:
        sizes = [to_svg(g, **_panel_kw(kw, rest)).size for _s, g, *rest in panels]
        kw['canvas'] = (max(w for w, _h in sizes),
                        max(h for _w, h in sizes) if match_height else 0.0)
    return save_panels(panels, formats=formats, align=align, **kw), PAGE.get(page, page)


def save_grid(rows, *, page='half', text_pt=9.0, align='top-left',
              layout: Layout = REVTEX, formats=('pdf', 'svg'), **kw):
    '''`save_figure` for panels in several rows: one font over all of them, one width, and
    each row padded to its own tallest panel only.'''
    panels = [p for row in rows for p in row]
    kw = _solved_font(panels, text_pt, page, layout, kw)
    sizes = {stem: to_svg(g, **_panel_kw(kw, rest)).size for stem, g, *rest in panels}
    width = max(w for w, _h in sizes.values())
    paths = []
    for row in rows:
        tallest = max(sizes[stem][1] for stem, *_ in row)
        paths += save_panels(row, formats=formats, align=align, canvas=(width, tallest), **kw)
    return paths, PAGE.get(page, page)


# ------------------------------------------------------------ several panels, one file

_INK  = re.compile(r'<(?:rect|polygon)\b[^>]*>')
_XYWH = re.compile(r'\b(x|y|width|height)="(-?[\d.]+)"')
_PTS  = re.compile(r'\bpoints="([^"]+)"')


def ink_box(body: str) -> tuple[float, float, float, float]:
    '''`(x0, y0, x1, y1)` of every box and hexagon in one drawing's body: where a panel's
    ink is, inside the margins the renderer keeps around it.'''
    xs, ys = [], []
    for m in _INK.finditer(body):
        tag = m.group(0)
        if pts := _PTS.search(tag):
            nums = [float(v) for v in re.split(r'[ ,]+', pts.group(1).strip())]
            xs += nums[0::2]
            ys += nums[1::2]
        else:
            d = {k: float(v) for k, v in _XYWH.findall(tag)}
            if {'x', 'y', 'width', 'height'} <= d.keys():
                xs += [d['x'], d['x'] + d['width']]
                ys += [d['y'], d['y'] + d['height']]
    return min(xs), min(ys), max(xs), max(ys)


def _split(doc) -> tuple[str, str]:
    '''A drawing as `(head, body)`: what goes once into a shared file, and what is placed.'''
    s = str(doc)
    start = s.index('>', s.index('<svg')) + 1
    end = s.rindex('</defs>') + len('</defs>')
    return s[start:end], s[end:s.rindex('</svg>')]


def printed_ink(sty: Style) -> tuple[str, str]:
    '''The ink and muted ink a figure's own text and arrows are set in: black on a light
    ground, the style's on a dark one.'''
    return (sty.ink, sty.muted) if sty.ground.scheme == 'dark' else ('#000', '#444')


def op_label(text: str, size: float, xc: float, y: float, font: str,
             lead: tuple[str, float] | None = None, ink: str = '#000') -> str:
    '''
    An operation named as `\\op{}` sets it -- small caps, an initial at `size` and the
    rest at 0.78 of it -- with its argument from the first `(` in the text face; several
    `;`-separated, in `ink`. `lead` is `(text, size)` set in front (a panel's `(a)`). Every
    piece is measured and placed: cairosvg reads neither `font-variant` nor an anchor over
    tspans.
    '''
    pieces = [(lead[0] + ' ', lead[1])] if lead else []
    for j, op in enumerate(text.split(';')):
        if j:
            pieces.append((', ', size))
        head, sep, arg = op.strip().partition('(')
        for i, word in enumerate(head.split(' ')):
            if i:
                pieces.append((' ', size))
            if word:
                pieces.append((word[0].upper(), size))
                if word[1:]:
                    pieces.append((word[1:].upper(), size * 0.78))
        if sep:
            pieces.append((sep + arg, size))
    advance = [_extents(t, font, sz, False, False)[2] for t, sz in pieces]
    x = xc - sum(advance) / 2
    spans = []
    for (t, sz), a in zip(pieces, advance):
        spans.append(f'<tspan x="{x:.2f}" y="{y:.2f}" font-size="{sz:.2f}">'
                     f'{html.escape(t)}</tspan>')
        x += a
    return (f'<text text-anchor="start" font-family="{font}" fill="{ink}">'
            f'{"".join(spans)}</text>')


def op_arrow(op: str, xm: float, y: float, gutter: float, size: float, font: str,
             ink: str = '#000', muted: str = '#444') -> str:
    '''An operation between two panels: an arrow across the gutter centred on `xm` at
    height `y`, in `muted`, the operation named over it in `ink` (`op_label`), for labels
    `size` px.'''
    L, h = max(gutter * 0.8, 1.8 * size), 0.34 * size
    return (f'<line x1="{xm - L / 2:.2f}" y1="{y:.2f}" x2="{xm + L / 2 - h:.2f}" y2="{y:.2f}" '
            f'stroke="{muted}" stroke-width="{max(0.08 * size, 0.5):.2f}"/>'
            f'<path d="M {xm + L / 2:.2f},{y:.2f} L {xm + L / 2 - h:.2f},{y - h * 0.45:.2f} '
            f'L {xm + L / 2 - h:.2f},{y + h * 0.45:.2f} Z" fill="{muted}"/>'
            + op_label(op, 0.90 * size, xm, y - 0.42 * size, font, ink=ink))


def save_strip(panels, path, *, text_pt=7.0, page='full', gutter_em=1.5, gap_em=None,
               gap_y_em=None, ring_gap_em=None, labels=None, label_pt=9.0, label_gap_em=0.6,
               ops=None, sublabels=None, titles=None, layout: Layout = REVTEX,
               formats=('pdf',), **kw):
    '''
    Several panels as **one drawing**, side by side by their ink (`ink_box`),
    `gutter_em` apart and top-aligned, with their labels set into it: `labels` under
    each (`'(a)'`), `titles` over each, and either `ops` -- `n - 1` operations, one per
    gutter, as an arrow with the name over it: a chain of states -- or `sublabels` --
    `n`, one beside each letter: branches off one graph. One font for the strip, solved
    so labels print at `text_pt` at `page`; with the gaps in ems (`gap_em`, `gap_y_em`,
    `ring_gap_em`) nothing is left to solve and the printed size is reported instead.

    Returns the paths written, the font drawn at, and the size the labels print at.
    '''
    printed = PAGE.get(page, page) * layout.linewidth_pt
    font_face = kw.get('font', layout.font)
    sty = style_of(kw.get('style', TERRACOTTA))
    n = len(panels)

    def drawn(font):
        per = {} if gap_y_em is None else {'gap_y': gap_y_em * font}
        if gap_em is not None:
            per['gap'] = gap_em * font
        if ring_gap_em is not None:
            per['ring_gap'] = ring_gap_em * font
        docs = [to_svg(g, layout=layout, **_panel_kw({**kw, 'font_size': font,
                                                       'line_scale': INK_PER_EM * font, **per},
                                                      rest))
                for _s, g, *rest in panels]
        parts = [_split(d) for d in docs]
        return parts, [ink_box(b) for _h, b in parts]

    last = {}

    def width_at(font):
        '''The strip's ink width at `font`, the drawing kept: the last one is the strip.'''
        last['parts'], last['boxes'] = drawn(font)
        last['width'] = (sum(x1 - x0 for x0, _y0, x1, _y1 in last['boxes'])
                         + (n - 1) * gutter_em * font)
        return last['width']

    given = kw.pop('font_size', None)
    font = given or PANEL_FONT_SIZE
    if given is None and gap_em is None:
        font = _settled(width_at, text_pt, printed, font, lambda _font: ValueError(
            f'text_pt={text_pt} is out of reach at page={page!r} for '
            f'{n} panels in a strip: ask for smaller labels or a wider page.'))
    else:
        width_at(font)
    parts, boxes, width = last['parts'], last['boxes'], last['width']

    prints = font * printed / width
    pad = 0.1 * font
    lab_px = label_pt * font / prints
    ink_h = max(y1 - y0 for _x0, y0, _x1, y1 in boxes)
    sub_px = 0.90 * lab_px
    sub_lead = 1.25 * sub_px
    sub_lines = max((s.count('\n') for s in sublabels if s), default=0) if sublabels else 0
    title_h = lab_px + label_gap_em * font if titles else 0
    H = (pad + title_h + ink_h + (label_gap_em * font + lab_px if labels or sublabels else 0)
         + sub_lines * sub_lead + pad)
    W = pad + width + pad
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="0 0 {W:.1f} {H:.1f}">', parts[0][0]]
    x, top = pad, pad + title_h
    y_lab = top + ink_h + label_gap_em * font + lab_px * 0.75
    gutter = gutter_em * font
    y_arrow = y_lab - 0.30 * lab_px

    ink, muted = printed_ink(sty)

    def text(s, xc, y, size):
        return (f'<text x="{xc:.2f}" y="{y:.2f}" text-anchor="middle" font-family="{font_face}" '
                f'font-size="{size:.2f}" fill="{ink}">{html.escape(s)}</text>')

    for i, ((_head, body), (x0, y0, x1, y1), label) in enumerate(
            zip(parts, boxes, labels or [None] * n)):
        out.append(f'<g transform="translate({x - x0:.2f},{top - y0:.2f})">{body}</g>')
        xc = x + (x1 - x0) / 2
        if titles and i < len(titles) and titles[i]:
            out.append(text(titles[i], xc, pad + lab_px * 0.75, lab_px))
        sub = sublabels[i] if sublabels and i < len(sublabels) else None
        if sub:
            for k, line in enumerate(sub.split('\n')):
                out.append(op_label(line, sub_px, xc, y_lab + k * sub_lead, font_face,
                                    lead=(label, lab_px) if label and not k else None,
                                    ink=ink))
        elif label:
            out.append(text(label, xc, y_lab, lab_px))
        x += (x1 - x0)
        op = ops[i] if ops and i < len(ops) else None
        if op:
            out.append(op_arrow(op, x + gutter / 2, y_arrow, gutter, lab_px, font_face,
                                ink, muted))
        x += gutter
    out.append('</svg>')
    svg = ''.join(out)

    path = pathlib.Path(path)
    paths = [_write(svg, path.with_suffix('.' + fmt)) for fmt in formats]
    return paths, font, prints


__all__ = ['CairoMeasure', 'INPUTS', 'LATIN_MODERN', 'Layout', 'OUTPUTS', 'PAGE', 'REVTEX',
           'font_for', 'ink_box', 'op_arrow', 'printed_ink', 'save', 'save_figure', 'save_grid', 'save_panels',
           'save_strip', 'tex', 'to_svg']
