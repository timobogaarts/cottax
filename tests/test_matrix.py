'''The N2 matrix: an order, groups of nodes with a colour, and the solves boxed.'''

import re

import pytest
from jax.tree_util import DictKey, GetAttrKey

from cottax.core import Function
from cottax.core.frozen import OrderedMap
from cottax.core.maths.plan import Cut
import purposes
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedFunction
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.consumers.visualization.matrix import (by_namespace, grouped_struct, matrix_struct,
                                         namespace_of, render_matrix_svg)
from cottax.consumers.visualization.interactive import render_html
from cottax.consumers.visualization.style import TERRACOTTA
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
    '''`obj` bound between the two it reads, so binding order is not run order.'''
    return graph_of(OrderedMap({
        N('d', 'one'): call(['z', 'y2'], ['y1']),
        N('obj')     : call(['y1', 'y2'], ['f']),
        N('d', 'two'): call(['z', 'y1'], ['y2']),
    }))


@pytest.fixture
def closed(sellar):
    cut = Cut(V('y2'), (N('d', 'one'),), HAT)(sellar)
    fp = RelationalCondition((Relation(HAT(V('y2')), V('y2'), Eq),), (), (HAT(V('y2')),), ())
    return graph_of(OrderedMap({**cut.definitions, N('fp'): fp}))


def test_a_namespace_is_the_leading_keys_without_the_own():
    assert namespace_of(N('d', 'one')) == ('d',)
    assert namespace_of(NodePath((DictKey('a'), DictKey('b'), DictKey('c'))), depth=1) == ('a',)
    assert namespace_of(HAT(N('d', 'one'))) == ('d',)                  # read through the mint
    assert namespace_of(N('obj')) == ()


def test_the_structure_order_runs(sellar):
    rows = [r['name'] for r in matrix_struct(sellar)['rows']]
    assert rows.index('.obj') > rows.index('.d.two')                   # it reads both
    assert [r['name'] for r in matrix_struct(sellar, order='binding')['rows']] \
        == ['.d.one', '.obj', '.d.two']


def test_a_mark_below_the_diagonal_is_feedback(sellar):
    s = matrix_struct(sellar)
    assert s['summary']['backward'] == 1                              # y2 back into one
    assert {(c['r'], c['c']) for c in s['cells']} == {(0, 1), (1, 0), (0, 2), (1, 2)}


def test_groups_are_given_or_read_off_the_names(sellar):
    by_hand = matrix_struct(sellar, groups={'disc': [N('d', 'one'), N('d', 'two')]})
    by_name = matrix_struct(sellar, groups=by_namespace())
    assert [r['group'] for r in by_hand['rows']] == ['disc', 'disc', '']
    assert [r['group'] for r in by_name['rows']] == ['d', 'd', '']
    assert by_hand['bands'] == [{'level': 0, 'lo': 0, 'hi': 1, 'label': 'disc', 'full': 'disc',
                                 'colour': TERRACOTTA.groups[0]}]
    with pytest.raises(ValueError, match='two groups'):
        matrix_struct(sellar, groups={'a': [N('obj')], 'b': [N('obj')]})


def test_a_nested_group_is_a_tint_of_its_outermost(sellar):
    s = matrix_struct(sellar, groups={('d', 'x'): [N('d', 'one')], ('d',): [N('d', 'two')]})
    hues = {r['hue'] for r in s['rows'] if r['group']}
    assert len(hues) == 1                                             # one family
    assert len({r['colour'] for r in s['rows'] if r['group']}) == 2   # two tints of it
    assert {b['level'] for b in s['bands']} == {0, 1}


def test_a_solve_is_boxed_in_the_structure_order(closed):
    s = matrix_struct(closed)
    assert [(b['depth'], b['type']) for b in s['boxes']] == [(0, 'fixed-point')]
    first = s['rows'][s['boxes'][0]['lo']]
    assert first['kind'] == 'fixed-point' and not first['driven']      # the problem first


def test_an_order_is_a_permutation(sellar):
    with pytest.raises(ValueError, match='permutation'):
        matrix_struct(sellar, order=[N('obj')])


def test_the_grouped_view_counts(sellar):
    g = grouped_struct(matrix_struct(sellar, groups=by_namespace()))
    assert [r['name'] for r in g['rows']] == ['d', '(ungrouped)']
    counts = {(c['r'], c['c']): c['n'] for c in g['cells']}
    assert counts == {(0, 0): 2, (0, 1): 2}                           # y1, y2 each way


def _hexagon_width(svg: str) -> float:
    pts = re.search(r'<polygon class="diag" data-i="\d+" points="([^"]+)"', svg).group(1)
    xs = [float(p.split(',')[0]) for p in pts.split()]
    return max(xs) - min(xs)


def test_a_problem_mark_is_at_least_its_size(closed):
    '''At a few hundred rows a cell is too small to tell a hexagon in.'''
    assert _hexagon_width(render_matrix_svg(closed, cell=3)) < 3
    assert _hexagon_width(render_matrix_svg(closed, cell=3, problem_mark=12)) == pytest.approx(12, abs=0.2)


def test_a_count_fits_its_cell(sellar):
    '''The grouped view's cell is at least as wide as its widest count's mark.'''
    svg = render_matrix_svg(sellar, groups=by_namespace(), grouped=True, font_size=9, cell=4)
    mark = float(re.search(r'class="mark[^"]*"[^>]*width="([\d.]+)"', svg).group(1))
    assert mark >= 9 * 0.5


def test_a_group_is_coloured_alike_in_every_order(sellar):
    '''A hue is the graph's, not the drawing's: reordering keeps it; `colours` pins it.'''
    hue = lambda st: {r['name']: r['colour'] for r in st['rows']}
    assert hue(matrix_struct(sellar, groups=by_namespace())) \
        == hue(matrix_struct(sellar, groups=by_namespace(), order='binding'))
    pinned = matrix_struct(sellar, groups=by_namespace(), colours={'d': '#123456'})
    assert hue(pinned)['.d.one'] == '#123456'


def test_a_texture_has_no_grid(closed):
    '''Below a readable cell a hairline per row only greys the matrix over.'''
    assert 'class="grid"' in render_matrix_svg(closed)
    assert 'class="grid"' not in render_matrix_svg(closed, cell=2)


def test_the_picture_says_what_it_was_asked(closed):
    svg = render_matrix_svg(closed, groups=by_namespace())
    assert svg.count('class="mark') == 5 and 'class="solve"' in svg
    assert 'class="mlbl"' not in render_matrix_svg(closed, names=False)
    assert TERRACOTTA.groups[0] not in render_matrix_svg(closed, groups=by_namespace(),
                                                          colour=False, legend=False)


def test_the_page_is_one_picture_with_its_toggles(closed):
    '''The colours and the names are CSS over one picture, not two pictures.'''
    doc = render_html(closed, views=('n2',), n2={'groups': by_namespace()})
    assert doc.count('\\u003csvg xmlns') == 1
    assert ':host([data-nocolour]) .mark' in doc and ':host([data-nonames]) .mlbl' in doc
    assert 'data-nonames' in render_html(closed, views=('n2',), names=False)
