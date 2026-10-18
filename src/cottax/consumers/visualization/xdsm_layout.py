'''The XDSM's layout: where each step, coupling and ring of a struct stands, in cells and in px.'''

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

from .interface import Formatter
from .naming import Shown, marked
from .style import Style
from .svg import BASE_FONT_SIZE, Measure, _span


SEP = ', '
'''What separates two variable names sharing one line of a data box.'''


# ------------------------------------------------------------------- how wide a box is
# A column is its label, plus air in em around it, plus the gap in px to the next box; a
# row is its lines (or a step box's floor) plus `gap_y`.
BOX_AIR              = 1.0   # em inside a step's box, around its label (`box_scale`)
MODEL_BOX_MIN_HEIGHT = 24    # a step box's floor, without the gap (`box_scale`)
CELL_GAP             = 16    # px from a box to the next (`gap`)
MODEL_CELL_MIN_WIDTH = 44    # a step column's floor
VAR_BOX_AIR          = 1.0   # em inside a coupling box, around its widest line
VAR_BOX_MIN_WIDTH    = 36    # ... and its floor
EXT_BOX_AIR          = 1.0   # em inside `Coor` and the results box, around their names


VAR_FONT_SIZE = 12.0
'''The size a variable name, an i/o label and the key are set at, at the base font size.'''


class Setting(NamedTuple):
    '''How the text is set: the sizes, the scale `k` every distance is a multiple of, and
    how a label is measured.'''

    size    : float
    vsize   : float
    ssize   : float
    k       : float
    bold    : bool
    measure : Measure | None

    @property
    def chip_h(self) -> float:
        return self.ssize + 5 * self.k

    @property
    def line_h(self) -> float:
        return 17 * self.k

    def spanl(self, s: str) -> tuple[float, float]:   # a diagonal label, or a box name
        return _span(s, self.size, bold=self.bold, measure=self.measure)

    def twl(self, s: str) -> float:
        return self.spanl(s)[1]

    def twv(self, s: str) -> float:                    # an i/o label, or a key entry
        return _span(s, self.vsize, measure=self.measure)[1]

    def twd(self, s: str) -> float:                    # a name in a coupling box
        return _span(s, self.vsize, bold=self.bold, measure=self.measure)[1]


def setting(font_size: float | None, bold: bool, measure: Measure | None) -> Setting:
    '''The text set at `font_size`, `bold` where a name is, measured by `measure`.'''
    # every distance is a multiple of `k`
    size  = float(font_size) if font_size else BASE_FONT_SIZE
    vsize = size * VAR_FONT_SIZE / BASE_FONT_SIZE
    k     = size / BASE_FONT_SIZE
    ssize = vsize                                    # a condition slot's label
    return Setting(size, vsize, ssize, k, bold, measure)


class Grid(NamedTuple):
    '''Where each step and each coupling box stands, in rows and columns.'''

    steps     : list[str]
    idx       : dict[str, int]
    slots     : list[list]
    span      : list[int]
    head_col  : list[int]
    n_in      : int
    n_ext_col : int
    n_out     : int
    out_col   : int
    out_row   : int
    nrows     : int
    ncols     : int
    nodes     : dict[tuple[int, int], list[str]]

    @property
    def n(self) -> int:
        return len(self.steps)

    # the diagonal sits at (n_in + i, head_col[i])
    def row_of(self, s: str) -> int:                     # a step's row
        return self.n_in + self.idx[s]

    def col_of(self, s: str) -> int:                     # a step's column
        return self.head_col[self.idx[s]]

    def diag_col(self, r: int) -> int:
        '''The column of the box that sources row r: a step's own, or the `Coor` block.'''
        return self.head_col[r - self.n_in] if r >= self.n_in else 0

    def diag_row(self, c: int) -> int | None:
        '''
        The row of the box column c feeds into: a step's, or for the results column the
        results box at its foot.
        '''
        if self.n_out and c == self.out_col:
            return self.out_row
        for i in range(self.n):
            if self.head_col[i] <= c < self.head_col[i] + self.span[i]:
                return self.n_in + i
        return None


def grid(struct: dict, *, condition_slots: bool | str, stack_inputs: bool,
         collapse_names: bool) -> Grid:
    '''Every step's columns and row, and the coupling boxes between them.'''
    steps    = list(struct['steps'])
    idx      = {name: i for i, name in enumerate(steps)}
    n        = len(steps)
    producer = struct['producer']
    problem_types = struct['problem_types']
    named         = struct['kinds']

    # which condition nodes draw their slots: under `True` a `stated` one's (a requirement's say
    # what it requires) and a `slotted` category's; all under `'all'`
    def draws_slots(s: str) -> bool:
        if not condition_slots:
            return False
        found = named.get(problem_types.get(s))
        return (condition_slots == 'all' or problem_types.get(s) == 'stated'
                or (found is not None and found.slotted))
    slots = [list(struct.get('slots', {}).get(s, [])) if draws_slots(s) else [] for s in steps]
    # a slotted step's columns are its slots, its name over them
    span  = [len(sl) or 1 for sl in slots]

    # -- the external variables: a row per input, one results column on the right ----
    # A box stands at (row = source, column = consumer): an input in a row of its own,
    # sourced from `Coor`; a result in the results column on its producer's row, one box
    # per producing step.
    coupled_vars = {f for dst in steps for f in struct['reads'][dst]
                    if (p := producer.get(f)) is not None and p != dst and p in idx}
    ext_in  : list[str] = []
    ext_out : list[str] = []
    for s in steps:                      # in diagonal order
        for f in struct['reads'][s]:
            if producer.get(f) is None and f not in ext_in:
                ext_in.append(f)
        for f in struct['writes'][s]:
            if f not in coupled_vars and f not in ext_out:
                ext_out.append(f)
    # `Coor` is drawn whenever there is a boundary, with a row of its own when nothing is
    # free; `stack_inputs` or `collapse_names` folds the input rows into one.
    fold_in   = collapse_names or stack_inputs
    n_ext_col = 1 if (ext_in or ext_out) else 0
    n_in      = ((1 if ext_in else 0) if fold_in else len(ext_in)) or n_ext_col
    n_out     = 1 if (n_ext_col and ext_out) else 0
    out_row   = n_in + n                    # the row and column the results box stands on
    head_col : list[int] = []               # where each step's own column is ...
    _c = n_ext_col
    for i in range(n):
        head_col.append(_c)
        _c += span[i]
    total_span = _c - n_ext_col             # ... and how many columns the steps take in all
    out_col   = n_ext_col + total_span
    nrows     = n_in + n + n_out
    ncols     = n_ext_col + total_span + n_out
    g = Grid(steps, idx, slots, span, head_col, n_in, n_ext_col, n_out, out_col, out_row,
             nrows, ncols, {})

    def in_row_of(f: str) -> int:
        return 0 if fold_in else ext_in.index(f)

    def slot_col(dst: str, f: str) -> int:
        '''The column a read of `f` by `dst` lands in: the slot that types it, else its own.'''
        for s_i, (_lab, vars_) in enumerate(slots[idx[dst]]):
            if f in vars_:
                return head_col[idx[dst]] + s_i
        return g.col_of(dst)

    # -- data nodes: (row=source, col=consumer) -> ordered list of variable names -----
    nodes = g.nodes

    def add_node(r: int, c: int, fields) -> None:
        if g.diag_col(r) == c or not fields:       # the diagonal itself holds no data
            return
        bucket = nodes.setdefault((r, c), [])
        for f in fields:
            if f not in bucket:
                bucket.append(f)

    for dst in steps:                                    # step -> step: couplings
        for f in struct['reads'][dst]:
            src = producer.get(f)
            if src is not None and src != dst and src in idx:
                add_node(g.row_of(src), slot_col(dst, f), [f])
    for s in steps:                                      # Coor -> step, step -> results
        for f in struct['reads'][s]:
            if f in ext_in:
                add_node(in_row_of(f), slot_col(s, f), [f])
        add_node(g.row_of(s), out_col, [f for f in struct['writes'][s]
                                        if f in ext_out])
    return g


def line_width(line: Sequence[Shown], t: Setting) -> float:
    '''What one drawn line measures: the base names, plus the separators.'''
    return sum(t.twd(text) for text, _mark in line) + t.twd(SEP) * (len(line) - 1)


class Layout(NamedTuple):
    '''Where everything stands, in px: every column's and row's size and start, every
    ring's edges, the two boundary boxes, the margins, the key's strip and the canvas.'''

    grid          : Grid
    setting       : Setting
    gap           : float
    gap_y         : float
    box_floor     : float
    labels        : dict[int, str]
    lines         : dict[tuple[int, int], list[list[Shown]]]
    box_size      : dict[tuple[int, int], tuple[float, float]]
    inputs_label  : str
    outputs_label : str
    col_w         : list[float]
    row_h         : list[float]
    col_x         : list[float]
    row_y         : list[float]
    ring_at       : dict[tuple[str, int, str, int], float]
    ext_w         : float
    out_w         : float
    in_target     : int | None
    out_arrow     : int | None
    m_top         : float
    m_left        : float
    m_bottom      : float
    key_h         : float
    key_w         : float
    key_g         : float
    width         : float
    height        : float

    def cx(self, c: int) -> float:
        return self.col_x[c] + self.col_w[c] / 2

    def cy(self, r: int) -> float:
        return self.row_y[r] + self.row_h[r] / 2

    def bw_of(self, c: int) -> float:
        '''The box in column `c`: its column, less the gap to the next one.'''
        return self.col_w[c] - self.gap

    def bh_of(self, r: int) -> float:
        '''... and the same down the rows, less the vertical gap.'''
        return self.row_h[r] - self.gap_y


def layout(struct: dict, sty: Style, keys: list[dict], *, formatter: Formatter,
           font_size: float | None, measure: Measure | None, condition_slots: bool | str,
           stack_inputs: bool, collapse_names: bool, inline_names: bool,
           square_names: bool, input_label: str | None, output_label: str | None,
           edges: bool, box_scale: float, var_box_scale: float, gap: float | None,
           gap_y: float | None, ring_gap: float | None, nest_gap: float | None,
           head_gap: float | None, cell: tuple[float, float] | None) -> Layout:
    '''`struct` laid out, under every argument of `render_xdsm_svg` that moves anything.'''
    t = setting(font_size, sty.bold, measure)
    size, vsize, ssize, k = t.size, t.vsize, t.ssize, t.k
    gap   = CELL_GAP if gap is None else float(gap)
    gap_y = gap if gap_y is None else float(gap_y)   # ... down the rows, if said apart
    chip_h = t.chip_h                                # ... and the chip it sits in
    twl, twv = t.twl, t.twv

    g = grid(struct, condition_slots=condition_slots, stack_inputs=stack_inputs,
             collapse_names=collapse_names)
    n, n_in, n_ext_col, n_out = g.n, g.n_in, g.n_ext_col, g.n_out
    head_col, span, slots, nodes = g.head_col, g.span, g.slots, g.nodes
    out_col, nrows, ncols = g.out_col, g.nrows, g.ncols
    shown_vars = struct['shown_vars']

    # -- labels + sizing -------------------------------------------------------------
    # a diagonal label is drawn with its mark as a combining character
    diag_labels = {i: marked(tuple(struct['shown_steps'][s])) for s, i in g.idx.items()}

    # A coupling box's lines, each a list of names: one per line, the results' along one
    # line, `inline_names` / `square_names` as said; `collapse_names` a count.
    def label_of(r: int, c: int) -> list[list[Shown]]:
        fs = nodes[(r, c)]
        if collapse_names:
            return [[(f'{len(fs)} var' + ('' if len(fs) == 1 else 's'), None)]]
        shown = [tuple(shown_vars[f]) for f in fs]
        if c == out_col or inline_names:
            return [shown]
        if square_names and len(fs) > 1:
            per = math.ceil(math.sqrt(len(fs)))
            return [shown[i:i + per] for i in range(0, len(fs), per)]
        return [[f] for f in shown]

    lines_of = {(r, c): label_of(r, c) for (r, c) in nodes}

    def linew(line: Sequence[Shown]) -> float:
        return line_width(line, t)

    inputs_label  = formatter.label(input_label  or 'Coor')
    outputs_label = formatter.label(output_label or 'Out')
    # each column cut to its diagonal label and the coupling boxes in it, each row to the
    # most lines a box in it stacks
    lineH   = t.line_h
    colW    = [0.0] * ncols
    rowlines = [1] * nrows
    is_slot = [False] * ncols
    for i in range(n):
        if not slots[i]:
            colW[head_col[i]] = max(colW[head_col[i]], twl(diag_labels[i]))
        for s_i, (lab, _v) in enumerate(slots[i]):
            c = head_col[i] + s_i
            colW[c] = max(colW[c], _span(lab, ssize, measure=measure)[1])
            is_slot[c] = True
    for (r, c) in nodes:
        lines = lines_of[r, c]
        rowlines[r] = max(rowlines[r], len(lines))
        if n_ext_col <= c < out_col:
            colW[c] = max(colW[c], max(linew(l) for l in lines))
    for c in range(n_ext_col, out_col):
        if is_slot[c]:                          # a slot is cut to its chip, and no wider
            colW[c] = max(colW[c] + 0.6 * BOX_AIR * box_scale * vsize + 7 * k + gap,
                          VAR_BOX_MIN_WIDTH * var_box_scale + gap)
        else:
            colW[c] = max(colW[c] + BOX_AIR * box_scale * size + gap,
                          MODEL_CELL_MIN_WIDTH * box_scale)
    # ... and a slotted step's name has to fit over its slots: the last slot pays.
    for i in range(n):
        if slots[i]:
            c0, sp = head_col[i], span[i]
            need = twl(diag_labels[i]) + BOX_AIR * box_scale * size + gap
            have = sum(colW[c0:c0 + sp])
            if have < need:
                colW[c0 + sp - 1] += need - have
    box_floor = MODEL_BOX_MIN_HEIGHT * k * box_scale
    rowH = [max(ln * lineH, box_floor) + gap_y for ln in rowlines]
    # an input row holds no step box, so no floor
    for r in range(n_in):
        rowH[r] = rowlines[r] * lineH + 8 * k + gap_y      # ... the data box, and the gap
    for i in range(n):                          # a slotted step is two bands tall
        if slots[i]:
            rowH[n_in + i] = max(rowH[n_in + i], max(lineH, box_floor) + chip_h + 5 * k + gap_y)
    floor = box_floor + gap_y
    if n_in and sum(rowH[:n_in]) < floor:         # ... but `Coor` is still a box
        rowH[0] += floor - sum(rowH[:n_in])
    if cell is not None:            # another drawing's grid, so the two share one
        for c in range(n_ext_col, out_col):
            colW[c] = cell[0]
        rowH = [cell[1]] * nrows
    # the boundary columns: `Coor` its label, the results column its results
    extW = max(twl(inputs_label) + EXT_BOX_AIR * size + gap,
               VAR_BOX_MIN_WIDTH) if n_ext_col else 0.0
    outW = max(max([linew(l) for (r, c) in nodes for l in lines_of[r, c] if c == out_col]
                   + [twl(outputs_label)]) + EXT_BOX_AIR * size + gap,
               VAR_BOX_MIN_WIDTH) if n_out else 0.0
    if n_ext_col:
        colW[0] = extW
    if n_out:
        colW[out_col] = outW

    # with no boundary box and `edges`, a bare arrow in and out
    in_target = None if n_ext_col or input_label is None or not edges else g.col_of(g.steps[0])
    out_arrow = None if n_ext_col or output_label is None or not edges else g.row_of(g.steps[-1])
    mT     = (46 if in_target is not None else 16) * k
    mL     = (88 if out_arrow is not None else 16) * k
    mR, mB = 20 * k, 24 * k

    # -- the key's strip -----------------------------------------------------------------
    keyH  = 30.0 * k if keys else 0.0
    keyW  = 26.0 * k                          # the sample, plus the gap to its text
    keyG  = 22.0 * k                          # between one entry and the next

    # Where each column and row starts, widened at each boundary a solve ring reaches by
    # what its rings need beyond the clearance there. Clearance is measured to the boxes
    # as drawn: a coupling box may overhang its cell (`over_col`, `over_row`).
    # `lanes[axis, b, side]`: the depths of the rings with an edge at boundary `b` on that
    # side, outermost first.
    over_col = [0.0] * ncols
    over_row = [0.0] * nrows
    box_size : dict[tuple[int, int], tuple[float, float]] = {}
    for (r, c) in nodes:
        lines = lines_of[r, c]
        bw = max(max(linew(l) for l in lines) + VAR_BOX_AIR * var_box_scale * vsize,
                 VAR_BOX_MIN_WIDTH * var_box_scale)
        bh = len(lines) * lineH + 8 * k * var_box_scale
        box_size[r, c] = (bw, bh)
        if n_ext_col <= c < out_col:
            over_col[c] = max(over_col[c], (bw - (colW[c] - gap)) / 2)
        over_row[r] = max(over_row[r], (bh - (rowH[r] - gap_y)) / 2)
    extraX = [0.0] * (ncols + 1)
    extraY = [0.0] * (nrows + 1)
    lanes : dict[tuple[str, int, str], list[int]] = {}
    ring_gap = sty.ring_gap * gap if ring_gap is None else float(ring_gap)
    own      = max(ring_gap, sty.nest_em * size)       # what a ring's own stand off it
    nest_gap = own if nest_gap is None else float(nest_gap)
    head_gap = own if head_gap is None else float(head_gap)
    # per boundary side, the innermost ring's clearance: `ring_gap`, or `head_gap` from a
    # problem's chevron
    inner : dict[tuple[str, int, str], float] = {}
    # only a solve ring takes a lane
    for b in struct.get('blocks', []):
        ds, depth = [g.idx[s] for s in b['steps'] if s in g.idx], b['depth']
        if not ds or b['ring'] != 'solve':
            continue
        lo, hi = min(ds), max(ds)
        for key, end in ((('x', head_col[lo], 'L'), lo), (('x', head_col[hi] + span[hi], 'R'), hi),
                         (('y', n_in + lo, 'L'), lo),      (('y', n_in + hi + 1, 'R'), hi)):
            lanes.setdefault(key, []).append(depth)
            at = head_gap if g.steps[end] in struct['problem_types'] else ring_gap
            inner[key] = max(inner.get(key, 0.0), at)
    for depths in lanes.values():
        depths.sort()

    def drawn_edge(axis, b, side):
        '''How far the box nearest boundary `b` on `side` stands from the boundary line.'''
        g, over = (gap, over_col) if axis == 'x' else (gap_y, over_row)
        return g / 2 - (over[b] if side == 'L' else over[b - 1])

    def stack(axis, b, side):
        '''What the rings with an edge at boundary `b` on `side` take, from the box on
        that side to the outermost of them: the innermost's own clearance and `nest_gap`
        for each ring outside it.'''
        depths = lanes.get((axis, b, side), ())
        return inner[axis, b, side] + nest_gap * (len(depths) - 1) if depths else 0.0

    for axis in ('x', 'y'):
        n_bounds = ncols if axis == 'x' else nrows
        for bnd in range(n_bounds + 1):
            sides = [s for s in ('L', 'R') if (axis, bnd, s) in lanes]
            if not sides:
                continue
            avail = ((drawn_edge(axis, bnd, 'L') if bnd < n_bounds else 0.0)
                     + (drawn_edge(axis, bnd, 'R') if bnd > 0 else 0.0))
            # each side's stack, plus `ring_gap` between the outermost ring and what faces it
            need = sum(stack(axis, bnd, s) for s in sides) + ring_gap
            (extraX if axis == 'x' else extraY)[bnd] += max(0.0, need - avail)

    def lane(axis, b, side, depth):
        '''Where a ring of `depth` with an edge at boundary `b` on `side` sits: the
        innermost's clearance in from the box on that side, and `nest_gap` more per ring
        inside it.'''
        pos, extra = (colX, extraX) if axis == 'x' else (rowY, extraY)
        depths = lanes[axis, b, side]
        rank   = depths.index(depth)                    # 0 is the outermost ring here
        off    = inner[axis, b, side] + nest_gap * (len(depths) - 1 - rank)
        if side == 'L':                                 # a block starting at b
            return pos[b] + drawn_edge(axis, b, 'L') - off
        return pos[b] - extra[b] - drawn_edge(axis, b, 'R') + off
    colX, rowY = [mL + extraX[0]], [mT + extraY[0]]
    for c, w in enumerate(colW):
        colX.append(colX[-1] + w + extraX[c + 1])
    for r, h in enumerate(rowH):
        rowY.append(rowY[-1] + h + extraY[r + 1])
    ring_at = {(axis, b, side, depth): lane(axis, b, side, depth)
               for (axis, b, side), depths in lanes.items() for depth in depths}

    W, H   = colX[-1] + mR, rowY[-1] + mB + keyH
    W      = max(W, mL + sum(keyW + twv(e['text']) + keyG for e in keys) + mR)

    return Layout(g, t, gap, gap_y, box_floor, diag_labels, lines_of, box_size, inputs_label,
                  outputs_label, colW, rowH, colX, rowY, ring_at, extW, outW, in_target,
                  out_arrow, mT, mL, mB, keyH, keyW, keyG, W, H)
