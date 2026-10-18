'''
A graph as an XDSM (Lambe & Martins, 2012) in SVG: the stylesheet and the painting in order.
'''

from __future__ import annotations

import os
from collections.abc import Mapping
from ...core import NestingGraph, ExecutableGraph, Plan

from .interface import Formatter
from .lookup import spell_plain  # noqa: F401  (imported from here by the docs)
from .naming import PlainFormat
from .style import DARK, TERRACOTTA, Style, style_of
from .svg import Measure, SvgDoc, _baseline, _text  # noqa: F401  (`_baseline`, `_text`: the tests)
from .xdsm_layout import CELL_GAP, layout  # noqa: F401  (`CELL_GAP`: the tests)
from .xdsm_paint import (paint_boundary, paint_couplings, paint_data_boxes, paint_diagonal,
                         paint_io, paint_key, paint_rings)
from .xdsm_struct import RING_LEVELS, key_entries, xdsm_struct


# the structure of the stylesheet, whatever the colours: weights, dashes, the type
BASE_CSS = (
    'svg{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;font-size:13px}'
    '.diag,.ext{stroke-width:1.4}.data{stroke-width:1}.dlbl,.io{font-size:12px}'
    '.deco{stroke-width:1.1;stroke-linecap:round;stroke-linejoin:round}'  # the hat
    '.group{stroke-width:1.7}'                                    # a block with a cycle
    '.group-ff{stroke-width:1.3;stroke-dasharray:1 4;stroke-linecap:round}'
)


def xdsm_css(sty: Style, levels: int = RING_LEVELS) -> str:
    '''
    `sty` as the XDSM's stylesheet, colours only: `BASE_CSS` states the structure. The
    class names are `render_xdsm_svg`'s -- `.diag` a model, `.data` a coupling, `.ext`
    a boundary box, `.group` a solve ring's band (`.group-l<d>` its depth,
    `.group-nested` below the top), `.group-ff` a block run in order. A problem's
    hexagon and a solve ring's contour are coloured on the shape, not here.
    '''
    weight = 600 if sty.bold else 'normal'
    css = [
        f'svg{{background:{sty.surface}}}' if sty.surface != '#ffffff' else '',
        f'.diag{{fill:{sty.model[0]};stroke:{sty.model[1]}}}',
        f'.ext{{fill:{sty.boundary[0]};stroke:{sty.boundary[1]}}}',
        f'.data{{fill:{sty.variable[0]};stroke:{sty.variable[1]}}}',
        f'.lbl{{fill:{sty.ink};font-weight:{weight}}}',
        f'.dlbl{{fill:{sty.variable_ink};font-weight:{weight}}}',
        f'.io{{fill:{sty.muted}}}',
        f'.deco{{stroke:{sty.ink}}}',
        f'.edge{{stroke:{sty.edge}}}.arr{{fill:{sty.edge}}}',
    ]
    top = sty.band(0)
    css.append(f'.group{{fill:{top or "none"};'
               f'stroke-dasharray:{"6 4" if sty.dashed else "none"}}}')
    if sty.dashed:
        css.append('.group-l1{stroke-dasharray:10 4}.group-l2{stroke-dasharray:2 3}')
    for depth in range(1, levels):
        if (fill := sty.band(depth)) is not None:
            css.append(f'.group.group-nested.group-l{depth}{{fill:{fill}}}')
    css.append(f'.group-ff{{fill:none;stroke:{sty.run_ring or "none"}}}')
    return ''.join(css)


LINE_WEIGHTS = {'.diag,.ext': 1.4, '.data': 1.0, '.deco': 1.1,
                '.group': 1.7, '.group-ff': 1.3, '.edge': 2.0, '.io': 1.4}
'''Every line weight in px, by class: `.diag,.ext` a model box and the boundary boxes,
`.data` a coupling box, `.deco` a minted name's caret, `.group` a solve's ring, `.group-ff`
a block run in order, `.edge` a coupling line, `.io` an external arrow. `line_weights`
overrides one, `line_scale` multiplies all.'''

RING_THICK = 4.0
'''A ring's weight under `thick_blocks=True`, in px.'''

_ALIGN = {'center': (0.5, 0.5), 'top-left': (0.0, 0.0),
          'top': (0.5, 0.0), 'left': (0.0, 0.5)}
'''`render_xdsm_svg`'s `align`, as the fraction of the canvas's slack to the drawing's
left and above it.'''


def stylesheet(sty: Style, *, follow_system: bool, css: str, font: str | None,
               font_size: float | None, size: float, vsize: float,
               weights: Mapping[str, float], line_scale: float,
               line_weights: Mapping[str, float] | None, thick_blocks: bool | float) -> str:
    '''The whole of one drawing's `<style>`: `BASE_CSS`, `sty`'s colours, and every
    setting of `render_xdsm_svg` that is said in CSS.'''
    sheet = BASE_CSS + xdsm_css(sty)
    if follow_system:                                         # the viewer decides
        sheet += '@media (prefers-color-scheme:dark){' + xdsm_css(DARK) + '}'
    sheet += css

    # the type last, to win over `BASE_CSS`
    if font:
        sheet += f'svg{{font-family:{font}}}'
    if font_size:
        sheet += (f'svg{{font-size:{size:.2f}px}}'
                  f'.dlbl,.io{{font-size:{vsize:.2f}px}}')
    if line_scale != 1 or line_weights:
        sheet += ''.join(f'{sel}{{stroke-width:{w * line_scale:.2f}}}'
                         for sel, w in weights.items() if sel.startswith('.')
                         and sel not in ('.edge', '.io'))
    if thick_blocks:
        ring = RING_THICK if isinstance(thick_blocks, bool) else float(thick_blocks)
        sheet += ('.group,.group-ff{stroke-width:%.2f;stroke-dasharray:none}'
                  % (ring * line_scale))
    return sheet


def render_xdsm_svg(
    graph : NestingGraph | ExecutableGraph | Plan | dict,
    *,
    blocks : bool | ExecutableGraph = False,
    file_name      : str = 'xdsm',
    outdir         : str = '.',
    style          : Style | str = TERRACOTTA,
    follow_system  : bool = False,
    css            : str = '',
    input_label    : str | None = None,
    output_label   : str | None = None,
    write          : bool = False,
    legend         : bool = True,
    font           : str | None = None,
    font_size      : float | None = None,
    names          : bool = True,
    edges          : bool = False,
    thick_blocks   : bool | float = False,
    line_scale     : float = 1.0,
    line_weights   : Mapping[str, float] | None = None,
    box_scale      : float = 1.0,
    var_box_scale  : float = 1.0,
    condition_slots: bool | str = False,
    problem_first  : bool = True,
    stack_inputs   : bool = False,
    collapse_names : bool = False,
    inline_names   : bool = False,
    square_names   : bool = False,
    gap            : float | None = None,
    gap_y          : float | None = None,
    ring_gap       : float | None = None,
    nest_gap       : float | None = None,
    head_gap       : float | None = None,
    cell           : tuple[float, float] | None = None,
    canvas         : tuple[float, float] | None = None,
    align          : str = 'center',
    formatter      : Formatter = PlainFormat(),
    measure        : Measure | None = None,
) -> SvgDoc:
    '''
    `graph` as a self-contained XDSM in SVG: the steps on the diagonal, a coupling's box at
    (producer's row, consumer's column), a free input entering from the top-left box and a
    result leaving to the bottom-right one. Returns an `SvgDoc`, whose `.cell` is the widest
    step column and tallest row used and `.size` the drawing's `(width, height)`.

    `graph`: a `NestingGraph`, a `Plan` (its graph), a proof (its graph, ringed as proved)
    or an `xdsm_struct` already built.
    `blocks`: ring every solve, and order the diagonal to run order (an `ExecutableGraph`:
    ring as that proof has it); otherwise the diagonal is binding order.
    `file_name`, `outdir`, `write`: with `write=True`, also write
    `<outdir>/<file_name>.svg` (`.path`).
    `style`: a `Style` or a name in `STYLES`.
    `follow_system`: add `DARK` under the viewer's `prefers-color-scheme` (the default
    style only); a hexagon and a ring's contour keep the light ground's colours.
    `css`: appended to the stylesheet.
    `input_label`, `output_label`: the boundary boxes' text (`None`: `'Coor'`, `'Out'`).
    `legend`: draw the key beneath the matrix.
    `font`: the face, a CSS family list (`None`: the style's).
    `font_size`: a step label's size in px; every other size and distance follows it.
    `names`: draw the text; off, the boxes keep the size of the names they would hold.
    `edges`: also draw the line from producer through a coupling box to its consumer, and
    with no boundary variable the labelled arrows in and out.
    `thick_blocks`: rings at `RING_THICK` px and solid, or at the number given.
    `line_scale`: multiplies every line weight.
    `line_weights`: merged over `LINE_WEIGHTS` before `line_scale`, per class.
    `box_scale`: the air around a step's label and its row's floor, times this.
    `var_box_scale`: the air around a coupling box's names, times this.
    `condition_slots`: a condition node's reads in a column per kind (`min`, `= 0`, `≤ 0`,
    `= û`, `≤ u`, `g ≤ g_max`); `True` for optimise and feasibility problems and every
    requirement (what it requires), `'all'` for every condition node.
    `problem_first`: a problem at the head of the block it drives; off keeps the graph's
    order within a block.
    `stack_inputs`: the free inputs on one row, not one row each.
    `collapse_names`: a coupling box shows a count, `N vars`, its names in its `<title>`.
    `inline_names`: a coupling box's names on one line.
    `square_names`: a coupling box's names in the fewest lines keeping it about square.
    `gap`: px between neighbouring boxes, both ways (`CELL_GAP`); `gap_y`: the vertical
    one alone.
    `ring_gap`: px from a ring to the boxes and rings beside it (`gap / 2`).
    `nest_gap`, `head_gap`: px from a ring to a ring nested in it and to its problem's
    hexagon (`ring_gap`).
    `cell`: `(w, h)` in px, every step column and every row at this pitch (another
    drawing's `.cell`), so two drawings share a grid.
    `canvas`: `(w, h)` in px to pad the drawing to (another's `.size`), so two share a
    scale; a smaller canvas is ignored.
    `align`: where the drawing sits in `canvas`: `'center'`, `'top-left'`, `'top'`, `'left'`.
    `formatter`: how a name is written; markup (`MARKUP`) is measured as it is set.
    `measure`: `(label, size, bold) -> (ink_left, ink_width)` in px, replacing the width
    estimate (`latex.xdsm.CairoMeasure`).
    '''
    sty = style_of(style)
    if follow_system and sty is not TERRACOTTA:
        raise ValueError(f"style {sty.name!r} has no reading on a dark system: follow_system "
                         f"draws the default style's, DARK")
    if font is None:
        font = sty.font
    if isinstance(graph, dict):                       # a struct, already built
        struct = graph
    else:
        struct = xdsm_struct(graph, blocks=blocks, formatter=formatter,
                             problem_first=problem_first)
    keys = key_entries(struct, sty) if legend else []
    lay = layout(struct, sty, keys, formatter=formatter, font_size=font_size, measure=measure,
                 condition_slots=condition_slots, stack_inputs=stack_inputs,
                 collapse_names=collapse_names, inline_names=inline_names,
                 square_names=square_names, input_label=input_label,
                 output_label=output_label, edges=edges, box_scale=box_scale,
                 var_box_scale=var_box_scale, gap=gap, gap_y=gap_y, ring_gap=ring_gap,
                 nest_gap=nest_gap, head_gap=head_gap, cell=cell)
    W0, H0 = W, H = lay.width, lay.height   # the drawing's own size, before any padding

    # line weights are scaled by `line_scale` alone, not by the type
    weights = dict(LINE_WEIGHTS, **(line_weights or {}))
    edge_w  = weights['.edge'] * line_scale
    io_w    = weights['.io'] * line_scale
    hex_w   = weights['.diag,.ext'] * line_scale       # a hexagon weighs what a model does

    cut = sty.cut_em * lay.setting.size / 2      # half the gap an open contour leaves
    # the rings first, under everything: a band is filled
    parts = (paint_rings(struct, lay, sty, cut)
             + (paint_couplings(lay, edge_w) if edges else [])
             + paint_io(lay, io_w)
             + paint_boundary(lay)
             + paint_diagonal(struct, lay, sty, cut, hex_w, line_scale)
             + paint_data_boxes(struct, lay, collapse_names)
             + paint_key(keys, lay, sty, cut, hex_w))

    sheet = stylesheet(sty, follow_system=follow_system, css=css, font=font,
                       font_size=font_size, size=lay.setting.size, vsize=lay.setting.vsize,
                       weights=weights, line_scale=line_scale, line_weights=line_weights,
                       thick_blocks=thick_blocks)

    # names off takes the text (and a minted name's caret) out, after layout: the geometry
    # is the same either way
    if not names:
        parts = [p for p in parts
                 if not p.startswith(('<text', '<polyline class="deco"'))]

    # a canvas is padding around the drawing, placed by `align`
    body = ''.join(parts)
    if canvas is not None:
        W, H = max(W, canvas[0]), max(H, canvas[1])
        fx, fy = _ALIGN[align]
        body = (f'<g transform="translate({(W - W0) * fx:.1f},{(H - H0) * fy:.1f})">'
                f'{body}</g>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
        f'viewBox="0 0 {W:.0f} {H:.0f}">'
        f'<style>{sheet}</style>'
        f'<defs>'
        f'<marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" '
        f'orient="auto"><path class="arr" d="M0,0 L6,3 L0,6 z"/></marker>'
        f'</defs>'
        + body + '</svg>'
    )

    doc = SvgDoc(svg)
    # `.cell` is what `cell=` pins: the widest step column, the tallest row
    g = lay.grid
    doc.cell, doc.size = (max(lay.col_w[g.n_ext_col:g.out_col], default=0.0), max(lay.row_h)), (W, H)
    if write:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, file_name + '.svg')
        with open(path, 'w') as fh:
            fh.write(svg)
        doc.path = path
    return doc
