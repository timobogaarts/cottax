'''
The node-link drawing: what a vertex is called, and what `dot` is told to call it.

The layout itself is not tested here -- `layered_pos` is a heuristic and `dot` is not
installed everywhere. What is tested is the seam these two share with the XDSMs: one
`Formatter` spells every name in the package, and the id a vertex is given in the DOT is
not that name.
'''

import re
import sys
import types

import networkx as nx
import pytest
from jax.tree_util import DictKey, GetAttrKey

from cottax.pytree.path import NodePath, VarPath
from cottax.core import ExecutableGraph, Function, SteppedExecutableGraph
from cottax.core.maths.plan import Cut, Plan
from cottax.core.realisation import ImplementedFunction
from cottax.core.frozen import OrderedMap
from cottax.consumers.visualization.style import DARK, PYXDSM, TERRACOTTA
from cottax.pytree.consumers.visualization import NoFormat
from cottax.consumers.visualization.interface_types import Around
from cottax.consumers.visualization.interface import Formatter
from cottax.consumers.visualization.lookup import spell_plain
from cottax.consumers.visualization import render_html, render_matrix_svg, render_xdsm_svg
from cottax.consumers.visualization.naming import PlainFormat
import purposes
from cottax.relational.rewrites import FixedPointCut
from cottax.consumers.visualization.visualization import (
    _dot_pos,
    _dot_source,
    _unpack,
    networkx_graph,
    to_dot,
)
from graphs import graph_of


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def DV(*keys) -> VarPath:
    return VarPath(tuple(DictKey(k) for k in keys))


def DN(*keys) -> NodePath:
    return NodePath(tuple(DictKey(k) for k in keys))


def call(reads, owns):
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), lambda *a: None)


@pytest.fixture
def sellar():
    return graph_of(OrderedMap({
        N('z'): call([], [V('z')]),
        N('d1'): call([V('z'), V('y2')], [V('y1')]),
        N('d2'): call([V('z'), V('y1')], [V('y2')]),
        N('obj'): call([V('y1'), V('y2')], [V('f')]),
    }))


@pytest.fixture
def flat():
    return graph_of(OrderedMap({
        DN('z'): call([], [DV('z')]),
        DN('d1'): call([DV('z')], [DV('y1')]),
    }))


@pytest.fixture
def open_model():
    '''Two free inputs and two results -- what a collapsed boundary has to merge.'''
    return graph_of(OrderedMap({
        DN('d1'): call([DV('x'), DV('z'), DV('y2')], [DV('y1')]),
        DN('d2'): call([DV('z'), DV('y1')], [DV('y2')]),
        DN('obj'): call([DV('y1'), DV('y2')], [DV('f'), DV('g')]),
    }))


class Rude(Formatter):
    '''A formatter that spells every node the same, quotes and all.'''

    def spell(self, path):
        return 'he said "hi"' if isinstance(path, NodePath) else spell_plain(path)

    def set(self, source):
        return source

    def label(self, text):
        return text

    def text(self, text):
        return text.plain

    def outright(self, path):
        return False

    def notation(self, ns):
        return Around(f'{ns}({{}})')


def labels(dot: str) -> list[str]:
    return re.findall(r'label="\s*(.*?)\s*"', dot)


def vertices(dot: str) -> dict[str, str]:
    '''Written name -> the DOT attributes it was drawn with, for the vertices alone.

    An edge label is a `label="..."` too, so `labels` sees both; a vertex is the one
    that carries attributes after it.
    '''
    return dict(re.findall(r'\[label="(.*?)" (.*?)\];', dot))


def test_a_name_is_written_not_repred(sellar):
    # `str(Path)` is the repr (`NodePath(.d1)`) on purpose -- printing a name says which
    # namespace it is in. A drawing wants the written name, which is `Formatter`'s job.
    assert '.d1' in labels(to_dot(sellar))
    assert 'NodePath(.d1)' not in to_dot(sellar)


def test_the_same_formatter_the_xdsms_take(flat):
    assert 'd1' in labels(to_dot(flat))                          # plain, by default
    assert "['d1']" in labels(to_dot(flat, formatter=NoFormat()))
    assert "['d1']" not in labels(to_dot(flat, formatter=PlainFormat()))


def test_the_formatter_reaches_the_edge_labels_too(flat):
    # an edge is labelled by the variables along it, and a variable is a name like any
    assert " ['z'] " in to_dot(flat, formatter=NoFormat())
    assert ' z ' in to_dot(flat)


def test_the_dot_id_is_not_the_name(sellar):
    '''A `Path` is not a string, and a formatter may spell two vertices the same.

    Writing the name as the id did both wrong at once: `_dot_pos` handed back positions
    keyed by strings while the graph is keyed by paths, and two identical spellings would
    have silently merged into one node.
    '''
    src, ids = _dot_source(_unpack(sellar, True), rankdir='LR',
                           formatter=PlainFormat())
    assert set(ids) == {f'n{i}' for i in range(len(ids))}
    assert all(isinstance(n, (NodePath, VarPath)) for n in ids.values())
    for i in ids:
        assert f'  {i} [label=' in src


def test_dot_positions_come_back_keyed_by_the_vertices(sellar, monkeypatch):
    '''What `dot` reports is keyed by the id it was given, and a caller needs vertices.'''
    g = _unpack(sellar, True)
    _, ids = _dot_source(g, rankdir='LR', formatter=PlainFormat())
    plain = '\n'.join(f'node {i} {k}.0 {k}.5 1 1 "" "" "" ""'
                      for k, i in enumerate(ids))

    class Source:
        def __init__(self, src): pass
        def pipe(self, format): return plain.encode()

    monkeypatch.setitem(sys.modules, 'graphviz', types.SimpleNamespace(Source=Source))
    pos = _dot_pos(g, rankdir='LR', formatter=PlainFormat())
    assert set(pos) == set(g.nodes())
    assert all(isinstance(n, (NodePath, VarPath)) for n in pos)


def test_a_caller_s_own_digraph_keeps_its_own_names():
    # `_unpack` accepts an already-built graph, and a name in one was never ours to spell
    g = nx.DiGraph()
    g.add_edge('a', 'b', vars=('x',))
    assert 'a' in labels(to_dot(g))
    assert 'x' in to_dot(g)


def test_the_collapsed_boundary_is_one_vertex_at_each_end(open_model):
    '''Two vertices, whatever the variable count: it is the *model's* boundary drawn.

    A vertex per variable says which variables are free; one vertex per end says the same
    of the model, which is what the XDSM's two boxes at opposite corners say -- and a
    figure printed beside one has to agree with it.
    '''
    src = to_dot(open_model, collapse_boundary=True, formatter=PlainFormat())
    # x and z leave one vertex, f and g enter one -- and neither is a vertex of its own
    assert set(vertices(src)) == {'In', 'Out', 'd1', 'd2', 'obj'}


def test_collapsing_keeps_every_variable_on_an_edge(open_model):
    '''The names are what a reader takes off the picture, and they were already there.

    A boundary vertex *was* one variable, so merging the vertices loses the name unless
    the edge each left along keeps it.
    '''
    src = to_dot(open_model, collapse_boundary=True, formatter=PlainFormat())
    on_edges = {v for e in re.findall(r'-> n\d+ \[label=" (.*?) "\]', src)
                for v in e.split(', ')}
    assert on_edges == {'x', 'z', 'y1', 'y2', 'f', 'g'}


def test_a_vertex_per_boundary_variable_is_still_what_boundary_draws(open_model):
    '''The default is not the new mode: `boundary=True` still names the free variables.

    Which variables are free is a question the collapsed drawing cannot answer -- the
    names sit on the edges, and an edge names a pair -- so these are two modes and not
    one replacing the other.
    '''
    written = vertices(to_dot(open_model, formatter=PlainFormat()))
    assert {'x', 'z', 'f', 'g'} <= set(written)
    assert 'In' not in written


def test_a_boundary_label_is_not_a_name(open_model):
    '''It stands for the world outside the model, which is no place inside it.

    So it is written as the caller gave it -- no `Formatter` is asked to spell it, which
    is also what lets a paper hand it markup its own text is set in.
    '''
    src = to_dot(open_model, collapse_boundary=True, input_label='I', output_label='O',
                 formatter=Rude())
    assert {'I', 'O'} <= set(labels(src))
    assert src.count(r'he said \"hi\"') == 3            # every node, escaped, and neither vertex


def test_the_two_boundary_vertices_are_drawn_alike_and_unlike_a_step(open_model):
    '''Both ends are the same thing -- outside -- and neither of them runs.'''
    attrs = vertices(to_dot(open_model, collapse_boundary=True))
    assert attrs['In'] == attrs['Out']
    assert attrs['In'] not in {v for k, v in attrs.items() if k not in ('In', 'Out')}


def test_the_colours_are_a_style(flat):
    '''By name or as a value: the graphs and the matrices are drawn in one style, which
    is the whole point of there being one.'''
    assert TERRACOTTA.model[1] in to_dot(flat)
    assert PYXDSM.model[1] in to_dot(flat, style='pyxdsm')
    assert TERRACOTTA.model[1] not in to_dot(flat, style=PYXDSM)


def test_a_marked_up_label_is_an_html_like_one_and_the_attributes_are_arguments(sellar):
    '''Written as graphviz reads markup, never as text holding `MARKUP`; and the type and
    spacing said where the source is written, not substituted into it after.'''
    from cottax.consumers.visualization.latex import LatexFormatter
    from cottax.consumers.visualization.naming import MARKUP
    src = to_dot(sellar, formatter=LatexFormatter())
    assert MARKUP not in src and 'label=< .<I>y</I><SUB>1</SUB> >' in src
    tuned = to_dot(sellar, font='Serif', font_size=12, nodesep=0.5, margin=0)
    assert tuned.startswith('digraph cottax {\n  margin=0;\n')
    assert 'nodesep=0.5;' in tuned and 'node [fontname="Serif" fontsize=12 ' in tuned


def test_dark_is_a_style(flat):
    '''The style is the one switch: a dark ground is a style of its own.'''
    assert TERRACOTTA.model[1] not in to_dot(flat, style=DARK)
    assert DARK.model[1] in to_dot(flat, style='dark')


# ---------------------------------------------------------------- the drawing's graph

def test_the_vertices_are_the_names_themselves(sellar):
    g = networkx_graph(sellar)
    assert set(g) == {N(n) for n in ('z', 'd1', 'd2', 'obj')}
    assert g[N('d1')][N('d2')]['vars'] == (V('y1'),)


def test_an_edge_keeps_every_variable_along_it():
    # Cutting acts on a variable, not on a node pair, so the pair keeps them all.
    g = networkx_graph(graph_of(OrderedMap({
        N('a'): call([], [V('p'), V('q')]),
        N('b'): call([V('p'), V('q')], [V('r')]),
    })))
    assert g[N('a')][N('b')]['vars'] == (V('p'), V('q'))


def test_the_boundary_enters_as_variables(sellar):
    g = networkx_graph(sellar, boundary=True)
    assert g.nodes[V('f')]['kind'] == 'sink'
    assert set(g.pred[V('f')]) == {N('obj')}


def test_a_variable_and_a_node_of_one_spelling_are_different_vertices():
    # `str(Path)` is not injective, so identity by spelling would be a lie -- but a
    # drawing labels both by spelling, which is what `boundary=True` refuses.
    assert V('q') != N('q')
    g = graph_of(OrderedMap({N('x'): call([V('x')], [V('q')])}))
    with pytest.raises(ValueError, match='written the same'):
        networkx_graph(g, boundary=True)


def test_the_drawings_graph_is_not_the_algebras(sellar):
    # A caller may decorate what it is handed; the core's bare projection is never it.
    handed = networkx_graph(sellar)
    handed.nodes[N('obj')]['pos'] = (0, 0)
    assert 'pos' not in sellar.graph._nx_dependencies.nodes[N('obj')]
    assert 'kind' not in sellar.graph._nx_dependencies.nodes[N('obj')]


def test_every_picture_draws_a_graph_a_plan_and_any_proof_alike(sellar):
    '''A `Plan` is drawn as its graph and a proof as its graph, ringed as proved in an XDSM.'''
    closed = FixedPointCut((Cut(V('y2'), (N('d1'),), purposes.Copy()),), purposes.Solve()(N('y2')))
    graph = closed(sellar)
    handed = (graph, Plan(sellar) + closed, ExecutableGraph(graph), SteppedExecutableGraph(graph))
    for each in handed:
        assert to_dot(each, blocks=True) == to_dot(graph, blocks=True)
        assert render_matrix_svg(each) == render_matrix_svg(graph)
        assert 'cx-wrap' in render_html(each)
    assert render_xdsm_svg(handed[1]) == render_xdsm_svg(graph)
    for proof in handed[2:]:
        assert render_xdsm_svg(proof) == render_xdsm_svg(graph, blocks=True)
    with pytest.raises(TypeError, match='got dict'):
        to_dot({})
