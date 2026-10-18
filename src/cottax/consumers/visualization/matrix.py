'''
A graph as an N2 matrix (a DSM) in SVG: a row and a column per node, a mark where the row's
node produces what the column's reads (below the diagonal is feedback). Optional on top: groups
(`groups`: a mapping of group to nodes, or a function of a node, `by_namespace`; a group is a
tuple and nests, each level a band), the solves boxed at every depth, and the row names.
'''

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence

from ...core import NestingGraph, Label
from ...core.driven import is_driven
from .lookup import minted_from, namespace_of
from .sequencing import Block, graph_drawn, ordered_graph, problems_first, walk
from .style import TERRACOTTA, Style, mix, style_of
from .page import TIP_VARS, Pictures, changed
from .interface import Formatter
from .interface_types import Kind
from .kinds import (answered, block_type, by_name, category_of, key_entry, problem_type_of,
                    ring_stroke)
from .naming import PlainFormat, marked, markup_html, shown
from .svg import (BASE_FONT_SIZE, SvgDoc, attr, cut_hexagon, cut_rect, hexagon, label, tw,
                  xesc)

type Group = tuple[str, ...]
type Grouping = Mapping[str | Group, Iterable[Label]] | Callable[[Label], str | Group | None]

TINTS = (0.0, 0.34, -0.26, 0.52, -0.44, 0.18, -0.13, 0.62, -0.55, 0.44, -0.35)
'''How far each group inside an outermost one is shaded from that one's hue: towards
white where positive, towards black where negative.'''

UNGROUPED = '#9a9a9a'

FINE_CELL = 4.0
'''The least pitch, in px, a cell is read at: below it the matrix is a texture, drawn with
no grid and its diagonal plain, since a hairline per row and an outline per node would
only grey it over.'''

CELL_EM = 1.4
'''A row's pitch in ems where a panel names its rows: a little over the type.'''


# ---------------------------------------------------------------------------- groups

def by_namespace(depth: int | None = None) -> Callable[[Label], Group]:
    '''The grouping a graph's own names declare (`PlaceDrawing.namespace_of`: its leading names
    without its own, a minted one read through its namespace), `depth` levels deep.'''
    return lambda path: namespace_of(path, depth)


def _as_group(g: str | Group | None) -> Group:
    return () if g is None else (g,) if isinstance(g, str) else tuple(g)


def _grouping(graph: NestingGraph, groups: Grouping | None) -> dict[Label, Group]:
    '''Every node's group, `()` for none.'''
    if groups is None:
        return {n: () for n in graph.nodes}
    if callable(groups):
        return {n: _as_group(groups(n)) for n in graph.nodes}
    at: dict[Label, Group] = {}
    for group, members in groups.items():
        for n in members:
            if n in at:
                raise ValueError(f'{n!r} is in two groups, {at[n]!r} and {_as_group(group)!r}')
            at[n] = _as_group(group)
    return {n: at.get(n, ()) for n in graph.nodes}


def _palette(at: Mapping[Label, Group], order: Sequence[Label], sty: Style,
             colours: Mapping[str, str] | None = None) -> dict[Group, tuple[str, str]]:
    '''Each group and every group above one: `(colour, hue)` -- its own tint, and the
    undiluted hue of its outermost group, which a label is written in. The hues go round
    `sty.groups` in `order`, the graph's own, unless `colours` names an outermost one's.'''
    seen = [g for g in dict.fromkeys(at[n] for n in order) if g]
    closure = list(dict.fromkeys(g[:i + 1] for g in seen for i in range(len(g))))
    tops = list(dict.fromkeys(g[:1] for g in closure))
    out: dict[Group, tuple[str, str]] = {(): (UNGROUPED, UNGROUPED)}
    for i, top in enumerate(tops):
        hue = (colours or {}).get(top[0]) or sty.groups[i % len(sty.groups)]
        family = [g for g in closure if g[:1] == top]
        for rank, g in enumerate(family):
            t = TINTS[rank % len(TINTS)]
            out[g] = (mix(hue, '#ffffff', t) if t >= 0 else mix(hue, '#000000', -t), hue)
    return out


# ---------------------------------------------------------------------------- orders

def _structure(graph: NestingGraph) -> tuple[list[Label], tuple[Block, ...]]:
    '''The run order at every level, each solve contiguous and its problem first -- the
    XDSMs' diagonal -- and the blocks of the graph ordered so.'''
    ordered = ordered_graph(graph)
    blocks = walk(ordered)
    names = problems_first(ordered.nodes, [(b.nodes, b.problem) for b in blocks],
                           lambda n: problem_type_of(ordered[n]) is not None)
    return names, blocks


def _order(graph: NestingGraph, order, at: Mapping[Label, Group],
           structure: list[Label]) -> list[Label]:
    if order == 'structure':
        return structure
    if order == 'binding':
        return list(graph.nodes)
    if order == 'grouped':
        # the structure order, stably regrouped: a group where its first member runs, a
        # nested group right after the one around it
        first: dict[Group, int] = {}
        for i, n in enumerate(structure):
            for k in range(len(at[n]) + 1):
                first.setdefault(at[n][:k], i)
        return sorted(structure, key=lambda n: tuple(first[at[n][:k + 1]]
                                                     for k in range(len(at[n]))) + (len(at[n]),))
    names = list(order)
    if set(names) != set(graph.nodes) or len(names) != len(graph.nodes):
        raise ValueError('an order is a permutation of every node of the graph')
    return names


# ---------------------------------------------------------------------------- the struct

def matrix_struct(graph, *, order: str | Sequence[Label] = 'structure',
                  groups: Grouping | None = None, formatter: Formatter = PlainFormat(),
                  style: Style | str = TERRACOTTA,
                  colours: Mapping[str, str] | None = None) -> dict:
    '''
    The plain data every matrix renderer draws: `rows` (name, group, colour, problem
    type, driven, minted, depth, reads, writes), `cells` (row, column, the variables, the
    source's colour), `bands` (a contiguous run of one group, per nesting level of the
    groups), `boxes` (a solve contiguous in this order: rows, depth, the type answering
    it), `legend` (each group and its count), `kinds` (each category drawn, its name to
    its `Kind`, in row order), `nodes` (the order) and `summary` counts. A group's hue is fixed by the graph's binding order, so one graph
    is coloured alike in every order; `colours` pins an outermost group's hue by name,
    for several graphs drawn alike.
    '''
    g = graph_drawn(graph)
    sty = style_of(style)
    at = _grouping(g, groups)
    structure, blocks = _structure(g)
    names = _order(g, order, at, structure)
    index = {n: i for i, n in enumerate(names)}
    palette = _palette(at, g.nodes, sty, colours)

    depth = {n: 0 for n in names}
    boxes = []
    for b in blocks:
        if not b.solved:
            continue
        for m in b.nodes:
            depth[m] = max(depth[m], b.depth + 1)
        rows = sorted(index[m] for m in b.nodes)
        if rows[-1] - rows[0] + 1 == len(rows):             # boxed where contiguous
            boxes.append({'lo': rows[0], 'hi': rows[-1], 'depth': b.depth, 'type': block_type(b)})

    def ports(vars_) -> tuple[list[str], int]:
        seen = list(dict.fromkeys(vars_))
        return [marked(shown(v, formatter)) for v in seen[:TIP_VARS]], len(seen)

    # every category drawn, in the order the rows first show it
    kinds = by_name(category_of(g[n]) for n in names)

    rows = []
    for n in names:
        node = g[n]
        kind = problem_type_of(node)
        colour, hue = palette[at[n]]
        if kind is not None:
            colour, hue = sty.paint(colour_of(kinds, kind))
        reads, nr = ports(node.reads)
        writes, nw = ports(node.owns)
        rows.append({'name': marked(shown(n, formatter)), 'group': '.'.join(at[n]),
                     'colour': colour, 'hue': hue, 'kind': kind, 'driven': is_driven(node),
                     'minted': minted_from(n) is not None, 'depth': depth[n],
                     'reads': reads, 'nr': nr, 'writes': writes, 'nw': nw})

    owners = g.graph.owners
    flows: dict[tuple[int, int], list] = {}
    for n in names:
        for v in dict.fromkeys(g[n].reads):
            src = owners.get(v)
            if src is not None and src != n:
                flows.setdefault((index[src], index[n]), []).append(v)
    cells = [{'r': r, 'c': c, 'n': len(vs), 'v': [marked(shown(v, formatter)) for v in vs[:TIP_VARS]],
              # a problem's marks in its driver's stroke: the fill is too pale to mark with
              'colour': rows[r]['hue'] if rows[r]['kind'] else rows[r]['colour']}
             for (r, c), vs in flows.items()]

    bands = []
    for level in range(max((len(at[n]) for n in names), default=0)):
        run, start = None, 0
        for i, n in enumerate([*names, None]):
            key = at[n][:level + 1] if n is not None and len(at[n]) > level else None
            if key != run:
                if run is not None:
                    bands.append({'level': level, 'lo': start, 'hi': i - 1,
                                  'label': '.'.join(run) if level == 0 else run[-1],
                                  'full': '.'.join(run), 'colour': palette[run][0]})
                run, start = key, i

    legend = [{'group': '.'.join(grp), 'colour': palette[grp][0],
               'count': sum(1 for n in names if at[n] == grp)}
              for grp in dict.fromkeys(at[n] for n in names) if grp]
    return {'rows': rows, 'cells': cells, 'bands': bands, 'boxes': boxes,
            'legend': legend, 'kinds': kinds, 'nodes': names,
            'summary': {'nodes': len(rows), 'marks': len(cells),
                        'backward': sum(1 for c in cells if c['r'] > c['c']),
                        'solves': len(boxes)}}


def grouped_struct(struct: dict) -> dict:
    '''
    `struct` collapsed to its groups, in the order they first appear: a row per group, a
    cell counting the variables flowing from one to another (the diagonal, within one).
    '''
    order = list(dict.fromkeys(r['group'] for r in struct['rows']))
    gi = {g: i for i, g in enumerate(order)}
    of = [gi[r['group']] for r in struct['rows']]
    colour = {r['group']: r['colour'] for r in struct['rows'] if r['kind'] is None}
    counts: dict[tuple[int, int], int] = {}
    for c in struct['cells']:
        key = (of[c['r']], of[c['c']])
        counts[key] = counts.get(key, 0) + c['n']
    rows = [{'name': g or '(ungrouped)', 'group': g, 'colour': colour.get(g, UNGROUPED),
             'hue': colour.get(g, UNGROUPED), 'kind': None, 'driven': False, 'minted': False,
             'depth': 0, 'reads': [], 'nr': 0, 'writes': [], 'nw': 0,
             'size': of.count(i)} for i, g in enumerate(order)]
    cells = [{'r': r, 'c': c, 'n': n, 'v': [], 'colour': rows[r]['colour']}
             for (r, c), n in counts.items()]
    return {**struct, 'rows': rows, 'cells': cells, 'bands': [], 'boxes': [], 'kinds': {},
            'summary': {**struct['summary'], 'groups': len(rows)}}


# ---------------------------------------------------------------------------- the picture

def colour_of(kinds: Mapping[str, Kind], name: str | None) -> tuple[str, str] | None:
    '''The colour of the category named `name` among a struct's `kinds`, `None` for none.'''
    return kinds[name].colour if name in kinds else None


def key(struct: dict, sty: Style, *, outermost: bool = False) -> list[dict]:
    '''The key of one N2: each group as a square of its colour, with its count -- or each
    outermost group, its name alone (`outermost`, a figure keying several N2s) -- then each
    category drawn (`key_entry`).'''
    if outermost:
        groups = list({b['label']: {'shape': 'square', 'fill': b['colour'], 'text': b['label']}
                       for b in struct['bands'] if b['level'] == 0}.values())
    else:
        groups = [{'shape': 'square', 'fill': e['colour'], 'stroke': e['colour'],
                   'text': f'{e["group"]} ({e["count"]})'} for e in struct['legend']]
    return groups + [key_entry(kind, sty) for kind in struct['kinds'].values()]


def draw_key(entries: Sequence[dict], *, x0: float, y0: float, right: float, size: float,
             side: float, air: float, after: float, line: float, end: float,
             hexagon_at: Callable[[float, float], str], stroke: float, ink: str,
             rx: float | None = None, titled: bool = False, digits: int = 2
             ) -> tuple[str, float]:
    '''
    A key's entries (`key`) set in lines from `(x0, y0)` up to `right`: a group a square
    `side` across (corners `rx`), a category its hexagon (`hexagon_at(cx, cy)`, outlined
    `stroke`, hovered for its `title` if `titled`), each named in `ink` at `size`, `air`
    after the mark and `after` after the name, a line `line` below the last. Returns the
    SVG and the y `end` below the last line.
    '''
    parts, x, y = [], x0, y0
    n = lambda v: f'{v:.{digits}f}'
    for e in entries:
        text = e['text']
        w = side + air + tw(text, size) + after
        if x + w > right and x > x0:
            x, y = x0, y + line
        if e['shape'] == 'hex':
            close = f'><title>{xesc(e["title"])}</title></polygon>' if titled else '/>'
            parts.append(f'<polygon points="{hexagon_at(x + side / 2, y + side / 2)}" '
                         f'fill="{e["fill"]}" stroke="{e["stroke"]}" stroke-width="{n(stroke)}"'
                         + close)
        else:
            corner = '' if rx is None else f' rx="{n(rx)}"'
            parts.append(f'<rect x="{n(x)}" y="{n(y)}" width="{n(side)}" height="{n(side)}"'
                         f'{corner} fill="{e["fill"]}"/>')
        parts.append(f'<text x="{n(x + side + air)}" y="{n(y + side * 0.9)}" '
                     f'font-size="{n(size)}" fill="{ink}">{xesc(text)}</text>')
        x += w
    return ''.join(parts), y + end


def names_width(struct: dict, font_size: float) -> float:
    '''The column the row names take, in px: the widest, and an em of air before the matrix.'''
    return max((tw(r['name'], font_size) for r in struct['rows']), default=0) + font_size


def _svg(struct: dict, sty: Style, *, names: bool, colour: bool, cell: float,
         font_size: float, font: str | None, counts: bool, legend: bool,
         problem_mark: float = 0.0, name_column: float | None = None) -> tuple[str, float, float]:
    rows, n = struct['rows'], len(struct['rows'])
    if counts:                     # the widest count fits its mark, which is 0.64 of a cell
        widest = max((tw(str(c['n']), font_size) for c in struct['cells']), default=0)
        cell = max(cell, (widest + font_size / 2) / 0.64)
    lanes = 1 + max((b['level'] for b in struct['bands']), default=-1)
    lane = max(0.6 * cell, font_size / 2)                  # a band stays seen at small cells
    label_w = names_width(struct, font_size) if names else 0
    label_w = max(label_w, name_column or 0)
    x0 = 8 + label_w + lanes * (lane + 2) + 4
    y0 = 8 + lanes * (lane + 2) + 4
    ink = sty.ink
    parts: list[str] = []

    # One pitch everywhere: a solve's box runs along the cell boundaries, and the node in
    # its corner overruns it a little -- accepted, for a grid that stays one grid.
    bsw  = max(cell / 14, 0.8)                             # a box's stroke
    at   = [k * cell for k in range(n + 1)]                # where each cell starts
    L    = n * cell
    W, H = x0 + L + 8, y0 + L + 8
    air  = cell * 0.05                                     # a node's own margin in its cell

    def box(b):
        lo = at[b['lo']]
        return x0 + lo, y0 + lo, at[b['hi']] + cell - lo

    for b in struct['boxes']:                              # the solves, as ground
        fill = sty.band(b['depth']) or 'none'
        x, y, w = box(b)
        parts.append(f'<rect class="area" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                     f'height="{w:.1f}" fill="{fill}"/>')
    # the grid: a hairline at every boundary, in translucent ink (see the stylesheet)
    fine = cell >= FINE_CELL
    for k in (range(n + 1) if fine else ()):
        t = at[k]
        parts.append(f'<line class="grid" x1="{x0:.1f}" y1="{y0 + t:.1f}" '
                     f'x2="{x0 + L:.1f}" y2="{y0 + t:.1f}"/>'
                     f'<line class="grid" x1="{x0 + t:.1f}" y1="{y0:.1f}" '
                     f'x2="{x0 + t:.1f}" y2="{y0 + L:.1f}"/>')
    # the diagonal: a model a rounded box, a problem a hexagon in its type's hue, tinted
    # and cut open until driven, at least `problem_mark` across and drawn last
    hexagons: list[str] = []
    for i, r in enumerate(rows):
        x, y = x0 + at[i], y0 + at[i]
        sw = f'{max(cell / 12, 0.6):.2f}'
        pad, side = air, cell - 2 * air                    # the whole cell, less its margin
        if r['kind']:
            fill, stroke = sty.fill(colour_of(struct['kinds'], r['kind']), r['driven']), r['hue']
            side = max(side, problem_mark)
            sw = f'{max(side / 12, 0.6):.2f}'
            pts = hexagon(x + cell / 2, y + cell / 2, side, side * 0.8, side / BASE_FONT_SIZE,
                           cap=side * 0.22)
            if sty.cuts and not r['driven']:
                hexagons.append(f'<polygon class="diag" data-i="{i}" points="{pts}" '
                                f'fill="{fill}" stroke="none"/>')
                hexagons += [f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"/>'
                             for d in cut_hexagon(pts, x + cell / 2, side * sty.cut_em / 3)]
            else:
                hexagons.append(f'<polygon class="diag" data-i="{i}" points="{pts}" '
                                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
            continue
        grouped = ' grouped' if colour and r['group'] else ''
        fill = mix(r['colour'], sty.surface, 0.6) if grouped else sty.model[0]
        outline = f'rx="{side / 5:.1f}" stroke="{sty.model[1]}" stroke-width="{sw}"' if fine else ''
        parts.append(f'<rect class="diag{grouped}" data-i="{i}" x="{x + pad:.1f}" y="{y + pad:.1f}" '
                     f'width="{side:.1f}" height="{side:.1f}" fill="{fill}" {outline}/>')
    # the marks
    for c in struct['cells']:
        x, y = x0 + at[c['c']], y0 + at[c['r']]
        fill = c['colour'] if colour else ink
        inset = cell * 0.18 if fine else 0.0             # a texture's mark is its whole cell
        carried = attr('\n'.join(markup_html(v) for v in c['v'])) if c['v'] else str(c['n'])
        parts.append(f'<rect class="mark{" fb" if c["r"] > c["c"] else ""}" data-r="{c["r"]}" '
                     f'data-c="{c["c"]}" data-v="{carried}" x="{x + inset:.1f}" y="{y + inset:.1f}" '
                     f'width="{cell - 2 * inset:.1f}" height="{cell - 2 * inset:.1f}" '
                     f'fill="{fill}"/>')
        if counts:
            parts.append(f'<text class="count" x="{x + cell / 2:.1f}" '
                         f'y="{y + cell / 2 + font_size * 0.35:.1f}" text-anchor="middle" '
                         f'font-size="{font_size:.1f}">{c["n"]}</text>')
    # the solves' edges, over the marks: the driver's hue, cut open where none answers
    for b in struct['boxes']:
        x, y, w = box(b)
        stroke = ring_stroke(b['type'], struct['kinds'], sty)
        sw = f'{bsw:.2f}'
        if answered(b['type']) or not sty.cuts:
            parts.append(f'<rect class="solve" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                         f'height="{w:.1f}" fill="none" stroke="{stroke}" stroke-width="{sw}"/>')
        else:                          # no driver: cut open, as a ring or a hexagon is
            parts += [f'<path class="solve" d="{d}" fill="none" stroke="{stroke}" '
                      f'stroke-width="{sw}"/>' for d in cut_rect(x, y, w, w, 0.0, sty.cut_em * cell / 3)]
    parts += hexagons
    # the group bands, beside and over the matrix, outermost nearest the labels
    for b in struct['bands']:
        off = 8 + label_w + b['level'] * (lane + 2)
        lo, span = at[b['lo']], at[b['hi']] + cell - at[b['lo']]
        fill = b['colour'] if colour else sty.boundary[0]
        title = f'<title>{xesc(b["full"])}</title>'
        parts.append(f'<rect class="band" x="{off:.1f}" y="{y0 + lo:.1f}" width="{lane:.1f}" '
                     f'height="{span:.1f}" fill="{fill}">{title}</rect>')
        parts.append(f'<rect class="band" x="{x0 + lo:.1f}" y="{8 + b["level"] * (lane + 2):.1f}" '
                     f'width="{span:.1f}" height="{lane:.1f}" fill="{fill}">{title}</rect>')
    # the names
    if names:
        for i, r in enumerate(rows):
            weight = ' font-weight="600"' if r['kind'] else ''
            fill = r['hue'] if (r['kind'] or (colour and r['group'])) else ink
            x, anchor = (8, 'start') if name_column else (8 + label_w - font_size / 2, 'end')
            parts.append(f'<text class="mlbl" data-i="{i}" x="{x:.1f}" '
                         f'y="{y0 + at[i] + cell / 2 + font_size * 0.35:.1f}" '
                         f'text-anchor="{anchor}" font-size="{font_size:.1f}" fill="{fill}"'
                         f'{weight}>{label(r["name"], font_size)}</text>')
    # the key: the groups, then the problem types drawn
    if legend:
        entries = key(struct, sty)
        drawn, bottom = draw_key(
            entries, x0=8.0, y0=H + 4, right=W, size=font_size, side=10, air=4, after=12,
            line=font_size * 1.6, end=font_size * 1.8, stroke=0.8, ink=sty.muted, rx=2,
            titled=True, digits=1,
            hexagon_at=lambda cx, cy: hexagon(cx, cy, 10, 8, 10 / BASE_FONT_SIZE, cap=2.2))
        parts.append(drawn)
        H = bottom if entries else H
    face = f'font-family:{font};' if font else 'font-family:ui-sans-serif,system-ui,sans-serif;'
    style_css = (f'svg{{{face}background:{sty.surface}}}'
                 # translucent ink: a shade darker than whatever it lies on, white or a
                 # band at any depth, so it reads the same everywhere
                 f'.grid{{stroke:{sty.ink};stroke-opacity:0.1;stroke-width:0.5}}'
                 f'.count{{fill:{sty.surface}}}')
    body = f'<style>{style_css}</style>' + ''.join(parts)
    return body, W, H


def render_matrix_svg(graph, *, order: str | Sequence[Label] = 'structure',
                      groups: Grouping | None = None, grouped: bool = False,
                      names: bool = True, colour: bool = True, legend: bool = True,
                      bands: bool = True, name_column: float | None = None,
                      cell: float | None = None, font_size: float = 9.0,
                      problem_mark: float = 0.0, colours: Mapping[str, str] | None = None,
                      font: str | None = None, style: Style | str = TERRACOTTA,
                      formatter: Formatter = PlainFormat(), write: bool = False,
                      file_name: str = 'matrix', outdir: str = '.') -> SvgDoc:
    '''
    The N2 matrix of `graph` (a `NestingGraph`, an `ExecutableGraph` or a `Plan`) as a
    self-contained SVG; `.size`, `.names_width` and `.struct` (`matrix_struct`) beside it.

    `order`: `'structure'` (run order at every level, each solve contiguous and boxed),
    `'binding'`, `'grouped'` (structure, regrouped so each group is contiguous) or a
    permutation of the nodes.
    `groups`: a mapping of group to nodes, or a function of a node (`by_namespace`).
    `grouped`: draw the groups themselves, a cell counting the variables between two.
    `names`, `colour`, `legend`: the row names, the group colours, the key.
    `bands`: the group bars beside and over the matrix.
    `name_column`: the names' column width in px, flush left, so several matrices line up
    (default: their own width, flush right).
    `cell`: a row's pitch in px (`CELL_EM * font_size`).
    `font_size`, `font`: the type (`font` default: the style's).
    `problem_mark`: the least width of a problem's hexagon in px, drawn over its neighbours.
    `colours`: an outermost group's hue by name, to colour several graphs alike.
    `style`, `formatter`: as every picture's.
    `write`, `file_name`, `outdir`: with `write=True`, also write `<outdir>/<file_name>.svg`.
    '''
    sty = style_of(style)
    struct = drawn = matrix_struct(graph, order=order, groups=groups, formatter=formatter,
                                   style=sty, colours=colours)
    if grouped:
        struct = grouped_struct(struct)
    if not bands:
        struct = {**struct, 'bands': []}
    cell = cell if cell is not None else CELL_EM * font_size
    body, W, H = _svg(struct, sty, names=names, colour=colour, cell=cell, font_size=font_size,
                      font=font or sty.font, counts=grouped, legend=legend,
                      problem_mark=problem_mark, name_column=name_column)
    doc = SvgDoc(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
                 f'viewBox="0 0 {W:.0f} {H:.0f}">{body}</svg>')
    doc.size = (W, H)
    doc.names_width = names_width(struct, font_size) if names else 0.0
    doc.struct = drawn
    if write:
        os.makedirs(outdir, exist_ok=True)
        doc.path = os.path.join(outdir, file_name + '.svg')
        with open(doc.path, 'w') as fh:
            fh.write(doc)
    return doc


# ---------------------------------------------------------------------------- the page

def n2_view(graph, pictures: Pictures, before=None, *, style: Style | str = TERRACOTTA,
            order: str | Sequence[Label] = 'structure', groups: Grouping | None = None,
            grouped: bool = False, formatter: Formatter = PlainFormat(), cell: float | None = None,
            font_size: float = 9.0, legend: bool = True,
            colours: Mapping[str, str] | None = None) -> dict:
    '''One frame's N2, as a page view (`page.page`): its picture added to `pictures`, what
    hovering a row says, and the rows of the nodes changed since `before`.'''
    sty = style_of(style)
    g = graph_drawn(graph)
    struct = matrix_struct(g, order=order, groups=groups, formatter=formatter, style=sty,
                           colours=colours)
    moved = changed(before, g)
    marked = [i for i, n in enumerate(struct['nodes']) if n in moved]
    if grouped:
        struct, marked = grouped_struct(struct), []
    cell = cell if cell is not None else CELL_EM * font_size
    body, W, H = _svg(struct, sty, names=True, colour=True, cell=cell, font_size=font_size,
                      font=sty.font, counts=grouped, legend=False)   # the page draws the key
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="0 0 {W:.0f} {H:.0f}">{body}</svg>')
    rows = [{'name': markup_html(r['name']), 'group': markup_html(r['group']), 'kind': r['kind'],
             'driven': r['driven'], 'reads': [markup_html(v) for v in r['reads']], 'nr': r['nr'],
             'writes': [markup_html(v) for v in r['writes']], 'nw': r['nw']} for r in struct['rows']]
    return {'svg': pictures.add(svg), 'rows': rows, 'changed': marked,
            'legend': key(struct, sty) if legend else []}


def colour_toggle(sty: Style, on: bool) -> tuple[str, str, bool, str]:
    '''The **Colour** toggle: the group colours off, over one picture.'''
    return ('colour', 'Colour', on,
            f':host([data-nocolour]) .mark{{fill:{sty.ink}}}'
            f':host([data-nocolour]) .band{{fill:{sty.boundary[0]}}}'
            f':host([data-nocolour]) .diag.grouped{{fill:{sty.model[0]}}}'
            f':host([data-nocolour]) .mlbl{{fill:{sty.ink}}}')


NAMES_CSS = ':host([data-nonames]) .mlbl{display:none}'


__all__ = ['by_namespace', 'grouped_struct', 'matrix_struct', 'render_matrix_svg']
