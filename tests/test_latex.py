'''Figures for a LaTeX document: set in the document's face, at the size the page says.'''

import pytest
from jax.tree_util import GetAttrKey

from cottax.core import Function
from cottax.core.frozen import OrderedMap
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedFunction
from graphs import graph_of

cairosvg = pytest.importorskip('cairosvg')
from cottax.consumers.visualization.latex import xdsm as lx  # noqa: E402


def V(k):
    return VarPath((GetAttrKey(k),))


def N(k):
    return NodePath((GetAttrKey(k),))


def call(reads, owns):
    return ImplementedFunction(Function(tuple(map(V, reads)), tuple(map(V, owns))),
                               lambda *a: None)


@pytest.fixture
def sellar():
    return graph_of(OrderedMap({N('d1'): call(['z', 'y2'], ['y1']),
                                N('d2'): call(['z', 'y1'], ['y2'])}))


def test_a_figure_is_set_in_the_page_face_and_measured(sellar):
    svg = lx.to_svg(sellar, blocks=True)
    assert 'LMRoman10' in svg and '<tspan' in svg                    # y with a subscript
    assert lx.CairoMeasure(lx.LATIN_MODERN)('\x01<I>y</I>\x01', 10, False)[1] > 0


def test_a_figure_prints_at_the_size_asked(sellar, tmp_path):
    paths, width = lx.save_figure([(tmp_path / 'a', sellar)], page='half', text_pt=9,
                                  formats=('pdf',))
    assert width == 0.48 and paths[0].stat().st_size > 0
    font = lx.font_for([(tmp_path / 'a', sellar)], 9, 'half')
    drawn = lx.to_svg(sellar, font_size=font).size[0]
    assert font * 0.48 * lx.REVTEX.linewidth_pt / drawn == pytest.approx(9, abs=0.05)


def test_a_strip_names_the_ops_between_its_panels(sellar, tmp_path):
    paths, _font, prints = lx.save_strip([('a', sellar), ('b', sellar)], tmp_path / 's',
                                         ops=['Cut(y2)'], labels=['(a)', '(b)'],
                                         formats=('svg',))
    assert 'UT' in paths[0].read_text() and prints == pytest.approx(7, abs=0.05)


def test_a_row_of_n2_panels_fills_the_page(sellar, tmp_path):
    '''Panels without a pitch share one, so the row is the page's width; a key once.'''
    from cottax.consumers.visualization.latex import matrix as lm
    [svg] = lm.save_rows([[sellar, (sellar, {'order': 'binding'})]], tmp_path / 'row',
                         labels=[['(a)', '(b)']], ops=[['Cut']], formats=('svg',))
    text = svg.read_text()
    width = float(text.split('width="', 1)[1].split('"', 1)[0])
    assert width == pytest.approx(lx.REVTEX.linewidth_pt * lx.PX_PER_PT, abs=2)
    assert text.count('(a)') == 1 and 'UT' in text                   # Cut, in small caps


def test_a_dark_strip_and_row_set_their_ops_and_labels_in_the_styles_ink(sellar, tmp_path):
    from cottax.consumers.visualization.latex import matrix as lm
    from cottax.consumers.visualization.style import DARK
    arrow = lx.op_arrow('Cut(y2)', 50, 20, 30, 10, lx.LATIN_MODERN, DARK.ink, DARK.muted)
    assert f'fill="{DARK.ink}"' in arrow and f'stroke="{DARK.muted}"' in arrow
    strip, _font, _prints = lx.save_strip(
        [('a', sellar), ('b', sellar)], tmp_path / 's', ops=['Cut(y2)'],
        labels=['(a)', '(b)'], titles=['one', 'two'], sublabels=None, style='dark',
        formats=('svg',))
    [row] = lm.save_rows([[sellar, sellar]], tmp_path / 'row', labels=[['(a)', '(b)']],
                         ops=[['Cut']], style='dark', formats=('svg',))
    for text in (strip[0].read_text(), row.read_text()):
        assert '"#000"' not in text and '"#444"' not in text


def test_a_light_strip_sets_its_ops_and_labels_black(sellar, tmp_path):
    strip, _font, _prints = lx.save_strip(
        [('a', sellar), ('b', sellar)], tmp_path / 's', ops=['Cut(y2)'],
        labels=['(a)', '(b)'], titles=['one', 'two'], sublabels=None, formats=('svg',))
    text = strip[0].read_text()
    assert 'fill="#000">one<' in text and 'stroke="#444"' in text
    assert lx.printed_ink(lx.style_of('dark')) == (lx.style_of('dark').ink, lx.style_of('dark').muted)
