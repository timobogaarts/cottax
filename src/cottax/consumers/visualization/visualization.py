'''
A graph as nodes and links: DOT source (`to_dot`, `render`) or a matplotlib axis (`draw`).
A box is a model, an ellipse a source, a hexagon a problem; dashed is no body, or no driver.
'''

from __future__ import annotations

import networkx as nx

from ...core import ExecutableGraph, NestingGraph, Plan
from ...core.driven import is_driven
from ...core.realisation import is_implemented
from .lookup import spelling
from .sequencing import Block, graph_drawn, walk
from .style import TERRACOTTA, Style, style_of
from .interface import Formatter
from .interface_types import Kind
from .kinds import answered, block_type, by_name, category_of, problem_type_of, ring_stroke
from .naming import MARKUP, PlainFormat, marked, shown
from .svg import xesc


def _kinds(nodes) -> dict[str, tuple[str, str]]:
    '''Every category among `nodes` (definitions, or `None` for a boundary vertex), its
    name to its colour.'''
    return {s: k.colour for s, k in by_name(category_of(node) for node in nodes
                                             if node is not None).items()}


def _palette(style: 'Style | str', kinds: dict[str, tuple[str, str]]) -> dict:
    '''
    The colours to draw in: `surface`, `text`, `muted` (an edge's label), `edge`, `font`,
    and per kind of vertex its stroke under its name and its fill under `<kind>_fill`:
    `node` a model, `free` / `sink` a boundary variable facing in / out, `ext` a collapsed
    boundary, `stated` a problem of no category, and one per category in `kinds` (with
    `<kind>_driven`, its driven fill).
    '''
    sty = style_of(style)
    c = {'surface': sty.surface, 'text': sty.ink, 'muted': sty.variable_ink,
         'edge': sty.variable[1],                   # an edge *is* a variable, as a data box is
         'node': sty.model[1], 'node_fill': sty.model[0],
         'free': sty.muted, 'free_fill': sty.surface,
         'sink': sty.variable[1], 'sink_fill': sty.variable[0],
         'ext': sty.boundary[1], 'ext_fill': sty.boundary[0],
         'stated': sty.stated[1], 'stated_fill': sty.stated[0],
         'font': sty.font}
    for name, colour in kinds.items():
        c[name] = sty.paint(colour)[1]
        c[f'{name}_fill'] = sty.fill(colour, driven=False)
        c[f'{name}_driven'] = sty.fill(colour, driven=True)
    return c

_BOUNDARY = ('free', 'sink')


def networkx_graph(graph: NestingGraph, *, boundary: bool = False) -> nx.DiGraph:
    '''
    The graph as a drawing sees it: a vertex per node (`node`, its definition) and, with
    `boundary`, one per boundary variable (`kind` `'free'` or `'sink'`); each edge carries
    the variables along it (`vars`). Refuses a boundary variable spelt as a node is.
    '''
    owner = graph.graph.owners
    g = nx.DiGraph()
    for name in graph.nodes:
        g.add_node(name, node=graph[name])

    if boundary:
        labels = {spelling(name) for name in graph.nodes}
        clash  = {spelling(var) for var in graph.graph.boundary_inputs + graph.graph.boundary_outputs} & labels
        if clash:
            raise ValueError(
                f'boundary variable(s) {sorted(clash)} are written the same as a node; '
                f'they cannot both be drawn'
            )
        for var in graph.graph.boundary_inputs:
            g.add_node(var, kind='free', node=None)
        for var in graph.graph.boundary_outputs:
            g.add_node(var, kind='sink', node=None)

    for name in graph.nodes:
        for var in dict.fromkeys(graph[name].reads):
            if var in owner:
                src = owner[var]
            elif boundary:
                src = var
            else:
                continue
            if g.has_edge(src, name):
                g[src][name]['vars'] += (var,)
            else:
                g.add_edge(src, name, vars=(var,))

    if boundary:
        # One edge per sink, never merged: a sink vertex *is* one variable.
        for var in graph.graph.boundary_outputs:
            g.add_edge(owner[var], var, vars=(var,))
    return g


def _unpack(graph, boundary: bool) -> nx.DiGraph:
    '''An already-built `nx.DiGraph` as it is, else what a drawing is handed
    (`graph_drawn`) as `networkx_graph` reads it.'''
    if isinstance(graph, nx.DiGraph):
        return graph
    return networkx_graph(graph_drawn(graph), boundary=boundary)


class _Ext:
    '''
    A collapsed boundary's vertex, carrying its own label: not a place, so it never
    collides with a vertex the graph named. One instance per drawing.
    '''

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f'_Ext({self.label!r})'


def _collapsed(g: nx.DiGraph, input_label: str, output_label: str) -> nx.DiGraph:
    '''
    A copy of `g` with the boundary vertices merged into one at each end, labelled
    `input_label` and `output_label`; the variable names stay on the edges.
    '''
    kinds = {n: d.get('kind') for n, d in g.nodes(data=True) if d.get('node') is None}
    h = g.copy()

    for kind, label, out in (('free', input_label, True), ('sink', output_label, False)):
        group = [n for n, k in kinds.items() if k == kind]
        if not group:
            continue
        ext = _Ext(label)
        h.add_node(ext, kind='ext', node=None)
        for var in group:
            edges = h.out_edges(var, data=True) if out else h.in_edges(var, data=True)
            for u, v, d in list(edges):
                pair = (ext, v) if out else (u, ext)
                if h.has_edge(*pair):
                    h.edges[pair]['vars'] += d['vars']
                else:
                    h.add_edge(*pair, vars=d['vars'])
        h.remove_nodes_from(group)

    return h


def _is_place(n) -> bool:
    '''Whether a vertex is a place, which is all `core.Place` says one is: its segments.'''
    return isinstance(getattr(n, 'segments', None), tuple)


def _label(n, formatter: Formatter) -> str:
    '''
    The written name of one vertex, as the XDSMs show it (`shown`); a collapsed boundary's
    own label, or `str` of a vertex that is not a place.
    '''
    if isinstance(n, _Ext):
        return n.label
    if not _is_place(n):
        return str(n)
    return marked(shown(n, formatter))


def _vars(d: dict, formatter: Formatter) -> str:
    '''The variables along one edge, written.'''
    return ', '.join(marked(shown(v, formatter)) if _is_place(v) else str(v)
                     for v in d['vars'])


def _dot_label(s: str) -> str:
    '''`s` as a DOT label: quoted text, or where a formatter marked it up (`MARKUP`)
    graphviz's HTML-like label -- a marked run as the tags it holds, the text around it
    escaped as the text it is.'''
    if MARKUP not in s:
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return '<' + ''.join(p if k % 2 else xesc(p) for k, p in enumerate(s.split(MARKUP))) + '>'


def _style(data: dict, c: dict) -> tuple[str, str, bool]:
    '''
    `(colour, shape, solid)` for one vertex: a boundary variable coloured by the way it
    faces, a collapsed boundary `ext`, a problem a hexagon in its type's colour (solid once
    driven), any other node `node` (solid once implemented).
    '''
    node = data.get('node')
    if data.get('kind') == 'ext':
        return 'ext', 'ext', True
    if node is None:
        return data.get('kind', 'free'), 'boundary', False
    if (kind := problem_type_of(node)) is not None:
        return kind if kind in c else 'stated', 'h', is_driven(node)

    shape = 's' if node.reads else 'o'
    return 'node', shape, is_implemented(node)


def layered_pos(g: nx.DiGraph, *, horizontal: bool = True, sweeps: int = 4) -> dict:
    '''
    Positions in layers, sources first, laid out on the condensation so a cycle's members
    share a layer. A node with no predecessors sits one layer before its first reader; each
    layer is ordered by barycentre over `sweeps` passes. `horizontal`: layers run left to
    right. For a large graph, `to_dot` lays out better.
    '''
    if g.number_of_nodes() == 0:
        return {}

    condensed = nx.condensation(g)
    layer : dict[str, int] = {}
    for i, component in enumerate(nx.topological_generations(condensed)):
        for c in component:
            for n in condensed.nodes[c]['members']:
                layer[n] = i

    for n in g:
        if g.in_degree(n) == 0 and g.out_degree(n) > 0:
            layer[n] = min(layer[m] for m in g.successors(n)) - 1

    shift = min(layer.values())
    layer = {n: i - shift for n, i in layer.items()}

    layers : list[list[str]] = [[] for _ in range(max(layer.values()) + 1)]
    for n in g:
        layers[layer[n]].append(n)

    def sweep(i: int, against: int, neighbours) -> None:
        '''Order layer `i` by the mean position of its neighbours in layer `against`.'''
        rank = {n: k for k, n in enumerate(layers[against])}
        fixed = {n: k for k, n in enumerate(layers[i])}     # keep unattached nodes put
        layers[i].sort(key=lambda n: (
            sum(rank[m] for m in neighbours(n) if m in rank) / c
            if (c := sum(1 for m in neighbours(n) if m in rank)) else fixed[n]
        ))

    for _ in range(sweeps):
        for i in range(1, len(layers)):
            sweep(i, i - 1, g.predecessors)
        for i in range(len(layers) - 2, -1, -1):
            sweep(i, i + 1, g.successors)

    widest = max(len(col) for col in layers)
    pos = {}
    for i, col in enumerate(layers):
        spread = widest / len(col)                  # thin layers fill the same height
        for k, n in enumerate(col):
            across = (k - (len(col) - 1) / 2) * spread
            pos[n] = (i, -across) if horizontal else (across, -i)

    return nx.rescale_layout_dict(pos)


def draw(
    graph            : NestingGraph | ExecutableGraph | Plan | nx.DiGraph,
    ax               = None,
    *,
    boundary         : bool = True,
    collapse_boundary: bool = False,
    input_label      : str | None = None,
    output_label     : str | None = None,
    pos              : dict | None = None,
    layout           : str = 'auto',
    names            : bool = True,
    node_size        : int = 2400,
    font_size        : int = 8,
    style            : Style | str = TERRACOTTA,
    formatter        : Formatter = PlainFormat(),
):
    '''
    Draw a `NestingGraph`, a `Plan`, a proof (each as `graph_drawn` reads it) or an
    already-built `nx.DiGraph` onto a matplotlib axis (a new one if `ax` is `None`);
    returns the axis.

    `boundary`, `collapse_boundary`, `input_label`, `output_label`, `style`, `formatter`:
    as `to_dot`. A vertex that is not a place (a caller's `nx.DiGraph`) is written as
    `str`.
    `pos`: positions given; else `layout`, `'dot'` (needs graphviz), `'layered'`
    (`layered_pos`) or `'auto'` (`dot` when installed).
    `names`: the variable names on the edges.
    `node_size`, `font_size`: matplotlib's.
    '''
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    g = _unpack(graph, boundary)
    c = _palette(style, _kinds(d.get('node') for _, d in g.nodes(data=True)))
    if collapse_boundary:
        g = _collapsed(g, input_label or 'In', output_label or 'Out')

    if pos is None:
        if layout in ('auto', 'dot'):
            try:
                pos = _dot_pos(g, rankdir='LR', formatter=formatter)
            except Exception:
                if layout == 'dot':
                    raise
                pos = None
        if pos is None:
            pos = layered_pos(g)

    if ax is None:
        depth = len({round(x, 6) for x, _ in pos.values()}) or 1
        _, ax = plt.subplots(figsize=(2.5 + 1.8 * depth, 4.5))
    ax.set_facecolor(c['surface'])
    ax.figure.set_facecolor(c['surface'])

    groups: dict[tuple[str, str, bool], list[str]] = {}
    for n, d in g.nodes(data=True):
        groups.setdefault(_style(d, c), []).append(n)

    for (colour, shape, has_body), names in groups.items():
        if shape == 'boundary':
            continue
        fill = c[f'{colour}_driven'] if has_body and f'{colour}_driven' in c else c[f'{colour}_fill']
        marks = nx.draw_networkx_nodes(
            g, pos, nodelist=names, ax=ax,
            node_shape='s' if shape == 'ext' else shape, node_size=node_size,
            node_color=fill, edgecolors=c[colour], linewidths=1.6,
        )
        # no body (or no driver): dashed
        marks.set_linestyle('solid' if has_body else 'dashed')

    bound = {k[0]: v for k, v in groups.items() if k[1] == 'boundary'}
    for colour, names in bound.items():
        marks = nx.draw_networkx_nodes(
            g, pos, nodelist=names, ax=ax, node_shape='o', node_size=node_size * 0.45,
            node_color=c[f'{colour}_fill'], edgecolors=c[colour], linewidths=1.0,
        )
        marks.set_linestyle('dashed')

    nx.draw_networkx_edges(
        g, pos, ax=ax, edge_color=c['edge'], width=1.4, arrowsize=11,
        node_size=node_size, connectionstyle='arc3,rad=0.06',
    )
    nx.draw_networkx_labels(
        g, pos, ax=ax, font_size=font_size, font_color=c['text'],
        labels={n: _label(n, formatter) for n in g.nodes},
    )

    if names:
        nx.draw_networkx_edge_labels(
            g, pos, ax=ax, font_size=font_size - 1, font_color=c['muted'],
            edge_labels={(u, v): _vars(d, formatter) for u, v, d in g.edges(data=True)},
            bbox={'boxstyle': 'round,pad=0.15', 'fc': c['surface'], 'ec': 'none'},
            rotate=False, connectionstyle='arc3,rad=0.06',
        )

    # a key of the boundary's two colours, when both are drawn
    handles = [
        mpatches.Patch(facecolor=c[f'{b}_fill'], edgecolor=c[b], linestyle='dashed', label=b)
        for b in _BOUNDARY if b in bound
    ]
    if len(handles) > 1:
        leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0),
                        frameon=False, fontsize=font_size)
        for t in leg.get_texts():
            t.set_color(c['muted'])

    ax.set_axis_off()
    ax.margins(0.12)
    return ax


def _dot_source(g: nx.DiGraph, *, rankdir: str, formatter: Formatter,
                style: 'Style | str' = TERRACOTTA, nesting: NestingGraph | None = None,
                font: str | None = None, font_size: int = 10, edge_font_size: int = 9,
                nodesep: float = 0.35, ranksep: float = 0.6, node_margin: str = '0.14,0.08',
                margin: float | None = None) -> tuple[str, dict[str, object]]:
    '''
    The DOT source, and the map from each vertex's id back to the vertex. A vertex's id is
    its index (`n0`, `n1`, ...) and its name a `label=`, since two vertices may be spelt
    alike. `nesting` is the graph whose solves are drawn as clusters (`to_dot`'s
    `blocks`); the rest are `to_dot`'s.
    '''
    present = [d.get('node') for _, d in g.nodes(data=True)]
    if nesting is not None:
        present += [nesting[n] for n in nesting.nodes]
    sty = style_of(style)
    kinds = by_name(category_of(node) for node in present if node is not None)
    c = _palette(sty, {s: k.colour for s, k in kinds.items()})
    font = font or c['font'] or 'Helvetica'
    ids = {f'n{i}': n for i, n in enumerate(g.nodes())}
    key = {n: i for i, n in ids.items()}

    lines = [
        'digraph cottax {',
        *([f'  margin={margin};'] if margin is not None else []),
        f'  rankdir={rankdir};',
        f'  bgcolor="{c["surface"]}";',
        f'  nodesep={nodesep}; ranksep={ranksep};',
        f'  node [fontname="{font}" fontsize={font_size} penwidth=1.6 '
        f'fontcolor="{c["text"]}" margin="{node_margin}"];',
        f'  edge [fontname="{font}" fontsize={edge_font_size} color="{c["edge"]}" '
        f'fontcolor="{c["muted"]}" penwidth=1.2 arrowsize=0.7];',
    ]

    def vertex(n, d) -> str:
        colour, shape, solid = _style(d, c)
        if shape == 'boundary':
            attrs = (f'shape=ellipse style="filled,dashed" fillcolor="{c[colour + "_fill"]}" '
                     f'color="{c[colour]}" fontcolor="{c["muted"]}" penwidth=1.0')
        elif shape == 'ext':
            attrs = (f'shape=box style="filled" '
                     f'fillcolor="{c["ext_fill"]}" color="{c["ext"]}"')
        elif shape == 'h':
            fill = c.get(f'{colour}_driven' if solid else f'{colour}_fill', c['stated_fill'])
            attrs = (f'shape=hexagon style="filled{"" if solid else ",dashed"}" '
                     f'fillcolor="{fill}" color="{c.get(colour, c["stated"])}"')
        else:
            attrs = (f'shape={"box" if shape == "s" else "ellipse"} '
                     f'style="{"filled,rounded" if shape == "s" else "filled"}'
                     f'{"" if solid else ",dashed"}" '
                     f'fillcolor="{c["node_fill"]}" color="{c[colour]}"')
        return f'{key[n]} [label={_dot_label(_label(n, formatter))} {attrs}];'

    placed : set = set()
    if nesting is not None:
        lines += _clusters(nesting, g, vertex, placed, kinds, sty)
    for n, d in g.nodes(data=True):
        if n not in placed:
            lines.append('  ' + vertex(n, d))

    for u, v, d in g.edges(data=True):
        lines.append(f'  {key[u]} -> {key[v]} [label={_dot_label(f" {_vars(d, formatter)} ")}];')

    lines.append('}')
    return '\n'.join(lines), ids


def _clusters(graph: NestingGraph, g: nx.DiGraph, vertex, placed: set,
              kinds: dict[str, Kind], sty: Style) -> list[str]:
    '''
    Every solve of `graph` as a `cluster`, nested as the solves nest: filled in its band,
    edged in its ring colour (`ring_stroke`), dashed where no driver answers it. Its
    members are declared in it and added to `placed`.
    '''
    out : list[str] = []
    open_ : list[Block] = []                            # the clusters open, outermost first

    def close(depth: int) -> None:
        # what the interior did not ring (acyclic entries) belongs to the cluster around it
        while open_ and open_[-1].depth >= depth:
            done = open_.pop()
            pad = '  ' * (done.depth + 1)
            for name in (done.held.nodes if done.held is not None else ()):
                if name in g and name not in placed:
                    out.append(f'{pad}  ' + vertex(name, g.nodes[name]))
                    placed.add(name)
            out.append(f'{pad}}}')

    count = 0
    for b in walk(graph):
        close(b.depth)
        if not b.solved:                               # a `Run`: nothing solved together
            continue
        btype = block_type(b)
        fill = sty.band(b.depth) or sty.surface
        edge = ring_stroke(btype, kinds, sty)
        count += 1
        pad = '  ' * (b.depth + 1)
        out.append(f'{pad}subgraph cluster_{count} {{')
        out.append(f'{pad}  style="filled,rounded{"" if answered(btype) else ",dashed"}"; '
                   f'fillcolor="{fill}"; color="{edge}"; penwidth=1.6; label="";')
        inner = set(b.held.nodes) if b.held is not None else set()
        for name in b.nodes:
            if name in g and name not in inner and name not in placed:
                out.append(f'{pad}  ' + vertex(name, g.nodes[name]))
                placed.add(name)
        open_.append(b)
    close(0)
    return out


def to_dot(
    graph            : NestingGraph | ExecutableGraph | Plan | nx.DiGraph,
    *,
    boundary         : bool = True,
    collapse_boundary: bool = False,
    input_label      : str | None = None,
    output_label     : str | None = None,
    rankdir          : str = 'LR',
    style            : Style | str = TERRACOTTA,
    blocks           : bool = False,
    formatter        : Formatter = PlainFormat(),
    font             : str | None = None,
    font_size        : int = 10,
    edge_font_size   : int = 9,
    nodesep          : float = 0.35,
    ranksep          : float = 0.6,
    node_margin      : str = '0.14,0.08',
    margin           : float | None = None,
) -> str:
    '''
    `graph` as Graphviz DOT source (producing it needs nothing installed). `graph` is read
    as `draw` reads it.

    `boundary`: a vertex per boundary variable.
    `collapse_boundary`: those merged into two boxes, one every free input comes from and
    one every result goes into, labelled `input_label` / `output_label` through
    `formatter.label` (`None`: `'In'` / `'Out'`).
    `rankdir`: graphviz's.
    `style`: a `Style` or a name in `STYLES`.
    `blocks`: every solve a cluster, nested as the solves nest, in its band and ring colours.
    `formatter`: how a name is written; a marked-up label becomes an HTML-like one.
    `font`: the face (the style's, else Helvetica); `font_size`, `edge_font_size`: a
    vertex's and an edge's type.
    `nodesep`, `ranksep`: the spacing; `node_margin`: a vertex's (`"x,y"`, inches);
    `margin`: the drawing's (none by default).
    '''
    g = _unpack(graph, boundary)
    if collapse_boundary:
        g = _collapsed(g, formatter.label(input_label or 'In'),
                       formatter.label(output_label or 'Out'))
    nesting = None
    if blocks:
        if isinstance(graph, nx.DiGraph):
            raise TypeError('blocks=True draws the solves of a graph, not of an nx.DiGraph')
        nesting = graph_drawn(graph)
    return _dot_source(g, rankdir=rankdir, formatter=formatter, style=style, nesting=nesting,
                       font=font, font_size=font_size, edge_font_size=edge_font_size,
                       nodesep=nodesep, ranksep=ranksep, node_margin=node_margin,
                       margin=margin)[0]


def render(graph: NestingGraph | ExecutableGraph | Plan | nx.DiGraph, **kw):
    '''
    `to_dot(graph, **kw)` as a `graphviz.Source`, which Jupyter displays inline as SVG.
    '''
    import graphviz
    return graphviz.Source(to_dot(graph, **kw))


def _dot_pos(g: nx.DiGraph, *, rankdir: str, formatter: Formatter) -> dict:
    '''
    Positions from `dot`'s `plain` output, keyed by the vertices of `g` (already
    unpacked: the ids `dot` reports are `_dot_source`'s for this `g`).
    '''
    import shlex

    import graphviz

    src, ids = _dot_source(g, rankdir=rankdir, formatter=formatter)
    plain = graphviz.Source(src).pipe(format='plain').decode()

    pos = {}
    for line in plain.splitlines():
        fields = shlex.split(line)
        if fields and fields[0] == 'node' and fields[1] in ids:
            pos[ids[fields[1]]] = (float(fields[2]), float(fields[3]))
    return nx.rescale_layout_dict(pos)
