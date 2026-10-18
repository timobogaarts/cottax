r'''
N2 figures for a LaTeX document: `render_matrix_svg` set in the document's face at the
size it prints -- a px of the drawing is a px of the page, so include it **without a
`width=`** -- and several set as one figure (`save_rows`): rows of panels, a label under
each, an operation between two, and one key under them all.

Needs `cairosvg`.
'''

from __future__ import annotations

import html
import pathlib

from ..matrix import CELL_EM, draw_key, key, render_matrix_svg
from ..style import TERRACOTTA, Style, style_of
from ..svg import SvgDoc, hexagon
from .xdsm import PAGE, PX_PER_PT, REVTEX, Layout, _write, op_arrow, printed_ink


def to_svg(graph, *, layout: Layout = REVTEX, text_pt: float = 7.0, **kw) -> SvgDoc:
    '''`render_matrix_svg` in `layout`'s face, its type printing at `text_pt`, without a
    key (a figure draws one for all its panels). Every argument is `render_matrix_svg`'s.'''
    kw.setdefault('font', layout.font)
    kw.setdefault('font_size', text_pt * PX_PER_PT)
    kw.setdefault('legend', False)
    return render_matrix_svg(graph, **kw)


def _panels(row) -> list[tuple[object, dict]]:
    return [(p[0], dict(p[1])) if isinstance(p, tuple) else (p, {}) for p in row]


def _filled(row: list[tuple[object, dict]], width: float, gutter: float, **kw) -> None:
    '''Give every panel of `row` without a `cell` one pitch, the same for all, so the row
    is `width` wide: the drawing grows about linearly in the pitch, so a few secant steps.'''
    free = [k for _g, k in row if 'cell' not in k]
    if not free:
        return

    def drawn(cell):
        for k in free:
            k['cell'] = cell
        w = sum(to_svg(g, **{**kw, **k}).size[0] for g, k in row) + (len(row) - 1) * gutter
        for k in free:
            del k['cell']
        return w

    a, b = 1.0, 4.0
    wa, wb = drawn(a), drawn(b)
    for _ in range(8):
        if abs(wb - width) < 0.05 or wb == wa:
            break
        a, wa, b = b, wb, max(b + (width - wb) * (b - a) / (wb - wa), 0.05)
        wb = drawn(b)
    for k in free:
        k['cell'] = b


def _key(entries, sty: Style, y0: float, right: float, size: float) -> tuple[str, float]:
    '''The key (`matrix.draw_key`) at a figure's size, in ink, from `(0, y0)` to `right`.'''
    side = 0.72 * size
    return draw_key(entries, x0=0.0, y0=y0, right=right, size=size, side=side, air=0.4 * size,
                    after=2.2 * size, line=1.5 * size, end=1.5 * size, stroke=side / 10,
                    ink=sty.ink, hexagon_at=lambda cx, cy: hexagon(
                        cx, cy, side * 1.25, side, side / 12, cap=side * 0.3))


def save_rows(rows, path, *, page='full', text_pt: float = 7.0, labels=None, ops=None,
              legend: bool = True, columns: bool = False, gutter_em: float = 2.0,
              row_gap_em: float = 1.2, align: str = 'top', layout: Layout = REVTEX,
              formats=('pdf', 'svg'), **kw):
    '''
    Rows of N2 panels as **one drawing**, its type printing at `text_pt`. A panel is a
    graph or `(graph, kwargs)`, `kw` `render_matrix_svg`'s for every panel and a panel's
    own over it. A panel given a `cell` is drawn at it; the others in a row share one
    pitch, so the row fills `page` of the line (a name in `PAGE` or a fraction). Per row,
    `labels` names each panel under it and `ops` the `n - 1` operations between them (an
    arrow, `op_arrow`). `columns` gives the names of the panels at one position across the
    rows one column, flush left, so the rows' matrices start alike. `legend`
    keys every subsystem and problem type drawn, once. `align` ('top' or 'bottom') sets
    the panels of a row against its top or its bottom edge: 'bottom' lines a cut's rows up
    with the rows it was cut from when what it adds sits above them.

    Returns the paths written.
    '''
    if align not in ('top', 'bottom'):
        raise ValueError(f"align={align!r}: 'top' or 'bottom'")
    sty = style_of(kw.get('style', TERRACOTTA))
    size = text_pt * PX_PER_PT
    kw = {'font': layout.font, 'font_size': size, **kw}
    width = PAGE.get(page, page) * layout.linewidth_pt * PX_PER_PT
    gutter = gutter_em * size
    rows = [_panels(r) for r in rows]
    for r in rows:
        _filled(r, width, gutter, **kw)

    drawings = [[to_svg(g, **{**kw, **k}) for g, k in row] for row in rows]
    if columns:                                   # one name column per column of panels
        for j in range(max(len(r) for r in rows)):
            widest = max((ds[j].names_width for ds in drawings if j < len(ds)), default=0)
            for r in rows:
                if j < len(r) and widest and r[j][1].get('names', True):
                    r[j][1]['name_column'] = widest
        drawings = [[to_svg(g, **{**kw, **k}) for g, k in row] for row in rows]
    out, keyed, y = [], {}, 0.0
    right = 0.0
    for i, (row, docs) in enumerate(zip(rows, drawings)):
        h = max(d.size[1] for d in docs)
        names = labels[i] if labels and i < len(labels) and labels[i] else [None] * len(row)
        between = ops[i] if ops and i < len(ops) and ops[i] else [None] * (len(row) - 1)
        x = 0.0
        for j, (d, (g, k)) in enumerate(zip(docs, row)):
            s = str(d)
            body = s[s.index('>', s.index('<svg')) + 1:s.rindex('</svg>')]
            top = y + (h - d.size[1] if align == 'bottom' else 0.0)
            out.append(f'<g transform="translate({x:.2f},{top:.2f})">{body}</g>')
            if names[j]:
                out.append(f'<text x="{x + d.size[0] / 2:.2f}" y="{y + h + size:.2f}" '
                           f'text-anchor="middle" font-size="{size:.2f}" fill="{sty.ink}">'
                           f'{html.escape(names[j])}</text>')
            if legend:                                    # a subsystem, not its tints
                for e in key(d.struct, sty, outermost=True):
                    keyed.setdefault((e['shape'], e['text']), e)
            x += d.size[0]
            if j < len(between) and between[j]:
                out.append(op_arrow(between[j], x + gutter / 2, y + h / 2, gutter, size,
                                    kw['font'], *printed_ink(sty)))
            x += gutter
        right = max(right, x - gutter)
        y += h + (1.6 * size if any(names) else 0) + row_gap_em * size
    if legend and keyed:
        groups = [e for e in keyed.values() if e['shape'] != 'hex']
        kinds = [e for e in keyed.values() if e['shape'] == 'hex']
        svg, y = _key(groups + kinds, sty, y, right, size)
        out.append(svg)
    W, H = right, y
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="0 0 {W:.1f} {H:.1f}" font-family="{kw["font"]}">'
           f'<rect width="100%" height="100%" fill="{sty.surface}"/>{"".join(out)}</svg>')
    path = pathlib.Path(path)
    return [_write(doc, path.with_suffix('.' + fmt)) for fmt in formats]


__all__ = ['CELL_EM', 'save_rows', 'to_svg']
