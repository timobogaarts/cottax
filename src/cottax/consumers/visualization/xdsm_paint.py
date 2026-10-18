'''The XDSM's paint: each part of a laid-out struct as SVG elements, in the order drawn.'''

from __future__ import annotations

from collections.abc import Sequence

from .kinds import STATED_TITLE, answered, ring_stroke
from .naming import Shown, marked, markup_html
from .style import Style
from .svg import (Measure, _baseline, _ink, _plain, _span, _text, attr, cut_hexagon,
                  cut_rect, hexagon, xesc)
from .xdsm_layout import SEP, VAR_FONT_SIZE, Layout
from .xdsm_struct import ring_class


RING_NEST = 0.25
'''How far a `run` ring steps in from the cell boundary per level of depth, as a fraction
of the half-gap, capped at 0.75. (A `solve` ring has a lane of its own, `ring_gap`.)'''


def _var_line(names: Sequence[Shown], xc: float, y: float,
              *, size: float = VAR_FONT_SIZE, bold: bool = True,
              measure: Measure | None = None) -> list[str]:
    '''
    One line of a coupling box: the shown names centred on `(xc, y)` on one baseline, each
    its own `<text>` anchored at its centre with its mark over it (a caret for `'hat'`, a
    rule for `'bar'`, any other a combining character), a comma flush after each but the
    last.
    '''
    shown  = [(t, m) if m in ('hat', 'bar', None) else (marked((t, m)), None) for t, m in names]
    spans  = [_span(t, size, bold=bold, measure=measure) for t, _ in shown]
    widths = [w for _, w in spans]
    sepw   = _span(SEP, size, bold=bold, measure=measure)[1]
    x      = xc - (sum(widths) + sepw * (len(names) - 1)) / 2
    k      = size / VAR_FONT_SIZE
    base   = _baseline(''.join(t for t, _ in shown), y, size)

    out : list[str] = []
    for i, ((text, mark), w) in enumerate(zip(shown, widths)):
        mid = x + w / 2
        out.append(_text(text, mid, base, size, 'dlbl', span=spans[i]))
        if mark:
            hw = min(w * 0.5, 9.0 * k)        # flat and wide: a caret, not a roof
            top = base - _ink(text, size)[0]  # clear of the glyphs it sits on
            peak = top - (4.2 if mark == 'hat' else 2.8) * k
            low = top - (1.4 if mark == 'hat' else 2.8) * k
            out.append(f'<polyline class="deco" fill="none" points="'
                       f'{mid - hw:.1f},{low:.1f} {mid:.1f},{peak:.1f} '
                       f'{mid + hw:.1f},{low:.1f}"/>')
        if i + 1 < len(names):
            out.append(f'<text class="dlbl" x="{x + w:.1f}" y="{base:.1f}" '
                       f'text-anchor="start">,</text>')
        x += w + sepw
    return out


def paint_couplings(lay: Layout, edge_w: float) -> list[str]:
    '''Each coupling's line: along its row out of the source's box, then down its column
    into the consumer's. No arrowheads.'''
    g, cx, cy = lay.grid, lay.cx, lay.cy
    parts : list[str] = []
    for (r, c) in g.nodes:
        dst = g.diag_row(c)
        pts = [(cx(g.diag_col(r)), cy(r)), (cx(c), cy(r))] \
            + ([(cx(c), cy(dst))] if dst is not None else [])
        parts.append(
            f'<polyline class="edge" fill="none" stroke-width="{edge_w:.2f}" points="'
            + ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts) + '"/>')
    return parts


def paint_io(lay: Layout, io_w: float) -> list[str]:
    '''The bare arrows in and out, drawn when there is no boundary box.'''
    g, cx, cy, k, vsize = lay.grid, lay.cx, lay.cy, lay.setting.k, lay.setting.vsize
    mT, mL = lay.m_top, lay.m_left
    parts : list[str] = []
    if (in_target := lay.in_target) is not None:
        x, ytop = cx(in_target), cy(in_target) - lay.bh_of(in_target) / 2
        parts.append(f'<line class="edge" x1="{x:.1f}" y1="{mT - 24 * k:.1f}" '
                     f'x2="{x:.1f}" y2="{ytop:.1f}" stroke-width="{io_w:.2f}" '
                     f'marker-end="url(#arr)"/>')
        parts.append(_text(lay.inputs_label, x, mT - 30 * k, vsize, 'io arrow'))
    if (out_arrow := lay.out_arrow) is not None:
        y, xleft = cy(out_arrow), cx(g.diag_col(out_arrow)) - lay.bw_of(g.diag_col(out_arrow)) / 2
        parts.append(f'<line class="edge" x1="{xleft:.1f}" y1="{y:.1f}" '
                     f'x2="{mL - 44 * k:.1f}" y2="{y:.1f}" '
                     f'stroke-width="{io_w:.2f}" marker-end="url(#arr)"/>')
        parts.append(_text(lay.outputs_label, mL - 50 * k,
                           _baseline(lay.outputs_label, y, vsize), vsize, 'io arrow',
                           anchor='end'))
    return parts


def paint_boundary(lay: Layout) -> list[str]:
    '''The two boundary boxes.'''
    g, t = lay.grid, lay.setting
    parts : list[str] = []

    # `Coor` spanning the input rows, top left; the results box at the foot of its column
    def _ext_box(col: int, y_mid: float, height: float, width: float, text: str) -> None:
        bw = width - lay.gap
        parts.append(f'<rect class="ext" x="{lay.cx(col) - bw / 2:.1f}" '
                     f'y="{y_mid - height / 2:.1f}" width="{bw:.1f}" '
                     f'height="{height:.1f}" rx="{6 * t.k:.1f}"/>')
        parts.append(_text(text, lay.cx(col), _baseline(text, y_mid, t.size), t.size, 'lbl',
                           bold=t.bold, span=t.spanl(text)))

    if g.n_ext_col:
        rowY = lay.row_y
        _ext_box(0, (rowY[0] + rowY[g.n_in]) / 2, rowY[g.n_in] - rowY[0] - lay.gap_y,
                 lay.ext_w, lay.inputs_label)
    if g.n_out:
        _ext_box(g.out_col, lay.cy(g.out_row), lay.bh_of(g.out_row), lay.out_w,
                 lay.outputs_label)
    return parts


def paint_rings(struct: dict, lay: Layout, sty: Style, cut: float) -> list[str]:
    '''Each block's ring around its square of the diagonal.'''
    g, size = lay.grid, lay.setting.size
    head_col, span, n_in = g.head_col, g.span, g.n_in
    colX, rowY, gap, gap_y = lay.col_x, lay.row_y, lay.gap, lay.gap_y
    parts : list[str] = []
    for b in struct.get('blocks', []):
        kind, depth, btype = b['ring'], b['depth'], b['type']
        ds = [g.idx[s] for s in b['steps'] if s in g.idx]
        if not ds or (kind == 'run' and sty.run_ring is None):    # a style drawing none
            continue
        lo, hi = min(ds), max(ds)
        if kind == 'solve':                     # in its lanes (`Layout.ring_at`)
            lane = lay.ring_at
            gx = lane['x', head_col[lo], 'L', depth]
            gy = lane['y', n_in + lo, 'L', depth]
            gw = lane['x', head_col[hi] + span[hi], 'R', depth] - gx
            gh = lane['y', n_in + hi + 1, 'R', depth] - gy
        else:
            # on the cell boundary, stepped in by depth
            nest = min(RING_NEST * depth, 0.75)
            padx, pady = nest * gap / 2, nest * gap_y / 2
            gx = colX[head_col[lo]] + padx
            gy = rowY[n_in + lo] + pady
            gw = colX[head_col[hi] + span[hi]] - padx - gx
            gh = rowY[n_in + hi + 1] - pady - gy
        # a solve's contour in the hue of the problem answering it, grey where none does
        ring_cls = ring_class(kind, depth)
        contour  = '' if kind != 'solve' else ring_stroke(btype, struct['kinds'], sty)
        drawn_as = f' stroke="{contour}"' if contour else ''
        rx   = min(sty.ring_radius_em * size, gw / 2, gh / 2)
        rect = (f'<rect class="{ring_cls}" x="{gx:.1f}" y="{gy:.1f}" '
                f'width="{gw:.1f}" height="{gh:.1f}" rx="{rx:.1f}"{drawn_as}')
        # no driver: the band unedged, the edge as two halves stopping short in the middle
        if kind == 'solve' and sty.cuts and not answered(btype):
            parts.append(f'{rect} style="stroke:none"/>')
            parts += [f'<path class="{ring_cls}" d="{d}"{drawn_as} style="fill:none"/>'
                      for d in cut_rect(gx, gy, gw, gh, rx, cut)]
        else:
            parts.append(f'{rect}/>')
    return parts


def paint_diagonal(struct: dict, lay: Layout, sty: Style, cut: float, hex_w: float,
                   line_scale: float) -> list[str]:
    '''Each step on the diagonal: a box, or a problem's hexagon with its slots.'''
    g, t = lay.grid, lay.setting
    size, ssize, k, chip_h = t.size, t.ssize, t.k, t.chip_h
    cx, cy, bw_of, bh_of = lay.cx, lay.cy, lay.bw_of, lay.bh_of
    steps, slots, n_in = g.steps, g.slots, g.n_in
    problem_types, named = struct['problem_types'], struct['kinds']
    stated_shapes, driven = struct['stated_shapes'], struct['driven']
    diag_labels = lay.labels
    parts : list[str] = []
    # a step's box is one line tall (or its floor), whatever its row holds
    diag_h = max(t.line_h, lay.box_floor)
    for i in range(g.n):
        c0, sp  = g.head_col[i], g.span[i]
        by      = cy(n_in + i)
        if slots[i]:                            # one shape over the slots, two bands tall
            x0 = cx(c0) - bw_of(c0) / 2
            x1 = cx(c0 + sp - 1) + bw_of(c0 + sp - 1) / 2
            bx, bw = (x0 + x1) / 2, x1 - x0
            # ... or as tall as its row, if that is taller
            bh  = max(diag_h + chip_h + 5 * k, bh_of(n_in + i))
            cap = min(4.0 * k, bw * 0.12)       # short of the chips' corners
        else:
            bx, bw = cx(c0), bw_of(c0)
            bh  = min(bh_of(n_in + i), diag_h)
            cap = None
        bx_head = bx
        ptype  = problem_types.get(steps[i])
        hover  = STATED_TITLE if ptype == 'stated' else \
            named[ptype].title if ptype in named else None
        title  = f'<title>{hover}</title>' if hover else ''
        at     = f' data-i="{i}"'               # the step a page hovers
        if ptype is None:
            box = (f'<rect class="diag"{at} x="{bx - bw / 2:.1f}" y="{by - bh / 2:.1f}" '
                   f'width="{bw:.1f}" height="{bh:.1f}" rx="{6 * k:.1f}"')
            parts.append(f'{box}>{title}</rect>' if title else f'{box}/>')
        else:
            # a problem in its category's colour, filled as far as it is decided; a
            # requirement hollow in the colour it will take, else grey
            if ptype in named:
                colour = named[ptype].colour
                fill   = sty.fill(colour, steps[i] in driven)
                stroke = sty.paint(colour)[1]
            elif (w := stated_shapes.get(steps[i])) in named:
                fill, stroke = sty.surface, sty.paint(named[w].colour)[1]
            else:
                fill, stroke = sty.stated
            cls   = 'prob prob-stated' if ptype == 'stated' else 'prob'
            coloured = f' fill="{fill}" stroke="{stroke}" stroke-width="{hex_w:.2f}"'
            pts   = hexagon(bx, by, bw, bh, k, cap)
            if sty.cuts and steps[i] not in driven:
                # no driver: the fill unedged, the edge as two chevrons cut at the head
                parts.append(f'<polygon class="{cls}"{at} points="{pts}"{coloured} '
                             f'style="stroke:none">{title}</polygon>')
                parts += [f'<path class="{cls}" d="{d}"{coloured} style="fill:none"/>'
                          for d in cut_hexagon(pts, bx_head, cut)]
            else:
                parts.append(f'<polygon class="{cls}"{at} points="{pts}"{coloured}>'
                             f'{title}</polygon>')
        # the name: in the top band of a slotted shape, the middle of a plain one
        y_name = by - bh / 2 + (bh - chip_h - 5 * k) / 2 if slots[i] else by
        parts.append(_text(diag_labels[i], bx, _baseline(diag_labels[i], y_name, size),
                           size, 'lbl', bold=t.bold,
                           span=t.spanl(diag_labels[i])))
        # the slots: a chip per kind in the bottom band, over its own column
        for s_i, (lab, _v) in enumerate(slots[i]):
            c  = c0 + s_i
            # inside the shape: clear of the chevron at either end, and of the neighbour
            xl = max(cx(c) - bw_of(c) / 2 + 1.5 * k, bx - bw / 2 + cap + 1.5 * k)
            xr = min(cx(c) + bw_of(c) / 2 - 1.5 * k, bx + bw / 2 - cap - 1.5 * k)
            cw = xr - xl
            cy_chip = by + bh / 2 - 3 * k - chip_h / 2
            parts.append(f'<rect class="slot" x="{xl:.1f}" y="{cy_chip - chip_h / 2:.1f}" '
                         f'width="{cw:.1f}" height="{chip_h:.1f}" rx="{2.5 * k:.1f}" '
                         f'fill="{sty.chip[0]}" fill-opacity="0.6" stroke="{sty.chip[1]}" stroke-opacity="0.45" '
                         f'stroke-width="{0.7 * line_scale:.2f}"/>')
            # `.lbl` carries no size of its own, so the chip's is said on the text
            parts.append(_text(lab, (xl + xr) / 2, _baseline(lab, cy_chip, ssize), ssize, 'lbl',
                               extra=f' font-size="{ssize:.2f}px"',
                               span=_span(lab, ssize, measure=t.measure)))
    return parts


def paint_data_boxes(struct: dict, lay: Layout, collapse_names: bool) -> list[str]:
    '''Each coupling box and its names.'''
    g, t = lay.grid, lay.setting
    k, lineH, n_in = t.k, t.line_h, g.n_in
    cx, cy = lay.cx, lay.cy
    shown_vars = struct['shown_vars']

    def sv(f: str) -> Shown:
        return tuple(shown_vars[f])

    parts : list[str] = []
    for (r, c), fs in g.nodes.items():
        lines = lay.lines[r, c]
        bw, bh = lay.box_size[r, c]
        # for a page's hover: its two ends' steps (-1 the boundary) and its names
        dst  = g.diag_row(c)
        ends = (f' data-r="{r - n_in if r >= n_in else -1}"'
                f' data-c="{dst - n_in if dst is not None and dst - n_in < g.n else -1}"'
                f' data-v="{attr(chr(10).join(markup_html(marked(sv(f))) for f in fs))}"')
        box = (f'<rect class="data"{ends} x="{cx(c) - bw / 2:.1f}" y="{cy(r) - bh / 2:.1f}" '
               f'width="{bw:.1f}" height="{bh:.1f}" rx="{4 * k:.1f}"')
        if collapse_names:                     # the names the count stands for
            names = SEP.join(_plain(marked(sv(f)))
                             for f in fs)
            parts.append(f'{box}><title>{names}</title></rect>')
        else:
            parts.append(f'{box}/>')
        y0 = cy(r) - (len(lines) - 1) * lineH / 2
        for kk, line in enumerate(lines):
            parts += _var_line(line, cx(c), y0 + kk * lineH, size=t.vsize,
                               bold=t.bold, measure=t.measure)
    return parts


def paint_key(keys: list[dict], lay: Layout, sty: Style, cut: float,
              hex_w: float) -> list[str]:
    '''The key under the matrix, each sample in the class and colour of what it stands for.'''
    t = lay.setting
    k, vsize = t.k, t.vsize
    keyH, keyW, keyG, mB = lay.key_h, lay.key_w, lay.key_g, lay.m_bottom
    parts : list[str] = []
    if keys:
        ky = lay.height - mB / 2 - keyH / 2 + 6 * k
        kx = lay.m_left
        for entry in keys:
            text, cls = entry['text'], entry['class']
            sample = (f'<rect class="key {cls}" x="{kx:.1f}" y="{ky - 7 * k:.1f}" '
                      f'width="{18 * k:.1f}" height="{14 * k:.1f}" rx="{4 * k:.1f}"')
            if entry['shape'] == 'open' and sty.cuts:
                parts.append(f'{sample} style="stroke:none"/>')
                parts += [f'<path class="key {cls}" d="{d}" stroke="{entry["stroke"]}" '
                          f'style="fill:none"/>'
                          for d in cut_rect(kx, ky - 7 * k, 18 * k, 14 * k, 4 * k, cut)]
            elif entry['shape'] != 'hex':
                contour = '' if cls == 'group-ff' else f' stroke="{entry["stroke"]}"'
                parts.append(f'{sample}{contour}/>')
            else:
                parts.append(f'<polygon class="key {cls}" '
                             f'points="{hexagon(kx + 9 * k, ky, 18 * k, 14 * k)}" '
                             f'fill="{entry["fill"]}" stroke="{entry["stroke"]}" '
                             f'stroke-width="{hex_w:.2f}"><title>{xesc(entry["title"])}'
                             f'</title></polygon>')
            parts.append(f'<text class="io" x="{kx + keyW:.1f}" '
                         f'y="{_baseline(xesc(text), ky, vsize):.1f}">'
                         f'{xesc(text)}</text>')
            kx += keyW + t.twv(text) + keyG
    return parts
