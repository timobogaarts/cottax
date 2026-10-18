'''
The interactive XDSM: the static picture framed. What is pinned here is the Python side
of the contract -- the page holds `render_xdsm_svg`'s own drawing, a frame per plan prefix
with shared pictures shipped once, the toggles as CSS, what hovering reads, and that a
name cannot break out of the page. The interaction runs in a browser.
'''

import json
import re

import pytest
from jax.tree_util import GetAttrKey

from cottax.core import Function
from cottax.core.frozen import OrderedMap
from cottax.core.maths.plan import Cut, Plan
import purposes
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedFunction
from cottax.consumers.visualization.page import HtmlDoc
from cottax.consumers.visualization.style import DARK
from cottax.consumers.visualization.xdsm import render_xdsm_svg, xdsm_struct
from cottax.consumers.visualization.interactive import render_html
from graphs import graph_of

HAT = purposes.Copy()


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def call(reads, owns):
    return ImplementedFunction(Function(tuple(V(r) for r in reads), tuple(V(o) for o in owns)),
                               lambda *a: None)


@pytest.fixture
def sellar():
    return graph_of(OrderedMap({
        N('z'): call([], ['z']),
        N('d1'): call(['z', 'y2'], ['y1']),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
        N('d2'): call(['z', 'y1'], ['y2']),
    }))




def _svgs(doc: str) -> list[str]:
    return json.loads(re.search(r'const SVGS = (\[.*?\]), FRAMES', doc).group(1))


def _frames(doc: str) -> list[dict]:
    return json.loads(re.search(r'FRAMES = (\[.*\]), VIEWS', doc).group(1))


def _views(doc: str) -> list[str]:
    return json.loads(re.search(r'VIEWS = (\[.*?\]);\n', doc).group(1))


def _shell(doc: str) -> str:
    return json.loads(re.search(r'ROOT\.innerHTML = (.*?);\n', doc).group(1))


# ---------------------------------------------------------------- one drawing

def test_the_page_holds_the_static_picture(sellar):
    '''No layout of its own: what the page shows is `render_xdsm_svg` with the same
    arguments, so the page and a figure cannot drift.'''
    doc = render_html(sellar, blocks=True, xdsm={'condition_slots': True})
    assert _svgs(doc)[0] == str(render_xdsm_svg(sellar, blocks=True, condition_slots=True,
                                                legend=False))


def test_the_key_is_the_pages_and_does_not_pan(sellar):
    '''The pictures on a page carry no key; each view carries its key as data, drawn in a
    strip outside the stage.'''
    doc = render_html(_cut_closed(sellar), blocks=True)
    assert all('class="key' not in x for x in _svgs(doc))
    xdsm, n2 = _frames(doc)[0]['views']
    assert [e['text'] for e in xdsm['legend']] == ['fixed-point (u = g(u))', 'solved together']
    assert [e['shape'] for e in n2['legend']] == ['hex']
    assert '<div class="key"></div><div class="stage">' in _shell(doc)


def _cut_closed(sellar):
    from cottax.relational.rewrites import FixedPointCut
    return (Plan(sellar) + FixedPointCut((Cut(V('y2'), (N('d1'),), HAT),), purposes.Solve()(N('y2')))).graph


def test_a_plan_is_a_frame_per_prefix(sellar):
    plan = Plan(sellar) + Cut(V('y2'), (N('d1'),), HAT)
    frames = _frames(render_html(plan, blocks=True))
    assert [f['label'] for f in frames] == ['base', repr(plan.ops[0])]
    assert frames[0]['views'][0]['changed'] == []
    steps = xdsm_struct(plan.graph, blocks=True)['steps']
    assert [steps[i] for i in frames[1]['views'][0]['changed']] == ['.d1']   # the cut rewired it


def test_a_picture_two_frames_share_is_shipped_once(sellar):
    '''An op that changes nothing drawn -- here, the same cut undone and redone would --
    costs a frame, not a picture: the frame points at the one already shipped.'''
    plan = Plan(sellar) + Cut(V('y2'), (N('d1'),), HAT)
    doc = render_html(plan)
    frames, svgs = _frames(doc), _svgs(doc)
    used = {v['svg'] for f in frames for v in f['views']}
    assert len(svgs) == len(used) <= sum(len(f['views']) for f in frames)


def test_a_graph_is_one_frame_and_no_slider(sellar):
    doc = render_html(sellar)
    assert len(_frames(doc)) == 1
    assert 'data-act=\\"frame\\"' not in doc


# ---------------------------------------------------------------- what the browser does

def test_the_names_are_a_css_toggle(sellar):
    shell = _shell(render_html(sellar))
    assert 'data-toggle="names"' in shell
    assert ':host([data-nonames]) .lbl' in shell
    assert ' data-nonames' in render_html(sellar, names=False)


def test_the_bar_is_fit_the_other_picture_and_the_toggles(sellar):
    shell = _shell(render_html(sellar))
    assert re.findall(r'data-(?:act|toggle)="([a-z]+)"', shell) == ['fit', 'view', 'names', 'colour']
    assert re.findall(r'data-(?:act|toggle)="([a-z]+)"', _shell(render_html(sellar, views=('xdsm',)))) \
        == ['fit', 'names']


def test_views_say_which_pictures_and_which_first(sellar):
    '''`('n2',)` renders no XDSM at all; the first view is the one the page opens on.'''
    n2 = render_html(sellar, views=('n2',))
    assert _views(n2) == ['N2'] and all('class="mark' in x for x in _svgs(n2))
    assert _views(render_html(sellar, views=('n2', 'xdsm'))) == ['N2', 'XDSM']
    with pytest.raises(ValueError, match='views'):
        render_html(sellar, views=('dsm',))


def test_every_frame_carries_its_n2(sellar):
    '''The same graph as a matrix, a button away: its own picture, its own row order.'''
    plan = Plan(sellar) + Cut(V('y2'), (N('d1'),), HAT)
    doc = render_html(plan, blocks=True)
    assert _views(doc) == ['XDSM', 'N2']
    for f in _frames(doc):
        xdsm, n2 = f['views']
        assert 'class="mark' in _svgs(doc)[n2['svg']]
        assert sorted(r['name'] for r in xdsm['rows']) == sorted(r['name'] for r in n2['rows'])
    assert len(_frames(doc)[1]['views'][1]['changed']) == 1


def test_hovering_reads_what_the_picture_says(sellar):
    '''A step carries its index, a coupling box its two ends and its names; the frame
    carries each step's ports for the tooltip.'''
    doc = render_html(sellar)
    svg, frame = _svgs(doc)[0], _frames(doc)[0]['views'][0]
    assert svg.count('data-i="') == len(frame['rows']) == 4
    assert re.search(r'class="data" data-r="\d" data-c="\d" data-v="', svg)
    obj = next(r for r in frame['rows'] if r['name'] == '.obj')
    assert obj['reads'] == ['.z', '.y1', '.y2'] and obj['writes'] == ['.f']


def test_the_surface_is_the_styles(sellar):
    '''A dark page is not framed in white.'''
    shell = _shell(render_html(sellar, style=DARK))
    assert DARK.surface in shell and 'color-scheme:dark' in shell


# ---------------------------------------------------------------- the document

def test_it_is_a_str_that_also_renders_inline(sellar):
    doc = render_html(sellar)
    assert isinstance(doc, HtmlDoc) and doc._repr_html_() == str(doc)
    assert doc.path is None


def test_write_drops_a_page_that_fills_the_window(sellar, tmp_path):
    doc = render_html(sellar, write=True, outdir=str(tmp_path), file_name='g')
    page = (tmp_path / 'g.html').read_text()
    assert doc.path == str(tmp_path / 'g.html')
    assert page.startswith('<!doctype html>') and ' data-fill>' in page
    assert ' data-fill' not in str(doc)


def test_two_renders_get_two_ids(sellar):
    ids = {re.search(r'id="(cx-[0-9a-f]+)"', render_html(sellar)).group(1)
           for _ in range(2)}
    assert len(ids) == 2


def test_an_empty_graph_raises():
    with pytest.raises(ValueError, match='no nodes'):
        render_html(graph_of(OrderedMap()))


def test_a_script_closing_tag_in_a_name_cannot_break_out():
    evil = graph_of(OrderedMap({
        NodePath((GetAttrKey('</script><script>window.pwned=1</script>'),)): call([], ['x']),
    }))
    doc = render_html(evil)
    assert doc.count('</script>') == 1
    # a name is HTML now, escaped as the text it is
    assert any('&lt;/script&gt;' in r['name'] for r in _frames(doc)[0]['views'][0]['rows'])


def test_a_nested_namespace_gets_step_groups():
    g = graph_of(OrderedMap({
        N('fem', 'mesh'): call([], ['x']),
        N('fem', 'solve'): call(['x'], ['u']),
        N('post', 'lift'): call(['u'], ['L']),
        N('flat'): call(['L'], ['J']),
    }))
    groups = xdsm_struct(g)['step_groups']
    assert groups == {'.fem.mesh': 'fem', '.fem.solve': 'fem', '.post.lift': 'post'}
    assert 'step_groups' not in xdsm_struct(graph_of(OrderedMap({N('a'): call([], ['x'])})))
    assert {r['group'] for r in _frames(render_html(g))[0]['views'][0]['rows']} \
        == {'fem', 'post', ''}


def test_a_step_is_grouped_as_the_n2_groups_it():
    '''One rule, `namespace_of`: `['a']['b'][0].c` is in `a`, in the XDSM's hover as in
    the N2, and not in `a.b[0]`.'''
    from jax.tree_util import DictKey, SequenceKey
    deep = NodePath((DictKey('a'), DictKey('b'), SequenceKey(0), GetAttrKey('c')))
    g = graph_of(OrderedMap({deep: call([], ['x']), N('c', 'd'): call(['x'], ['y'])}))
    xdsm, n2 = (view['rows'] for view in _frames(render_html(g))[0]['views'])
    assert xdsm_struct(g)['step_groups'] == {'a.b[0].c': 'a', '.c.d': 'c'}
    assert sorted(r['group'] for r in xdsm) == sorted(r['group'] for r in n2) == ['a', 'c']
