'''
What a drawing says with its style: a problem's type as a hue, how far it is decided as its
fill and contour, a solve as a band -- the same in the static XDSM, the interactive one and
the node-link drawing.
'''

import re

import pytest
from jax.tree_util import GetAttrKey

from cottax.core import Function
from cottax.relational import Min, Objective
from cottax.core.frozen import OrderedMap
from cottax.core.maths.plan import Cut
import purposes
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedFunction
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.relational.consumers.visualization import FIXED_POINT, KINDS, ROOT_FIND
from cottax.consumers.visualization.latex import LatexFormatter, tex
from cottax.consumers.visualization.naming import PlainFormat, shown
from cottax.consumers.visualization.interface_types import Text
import cottax.assemblies.jax as cx
from cottax.consumers.visualization.style import (DARK, PYXDSM, STYLES, TERRACOTTA, mix,
                                        style_of)
from cottax.consumers.visualization.visualization import to_dot
from cottax.consumers.visualization.naming import MARKUP
from cottax.consumers.visualization.xdsm import render_xdsm_svg, xdsm_css, xdsm_struct
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
        N('z') : call([], ['z']),
        N('d1'): call(['z', 'y2'], ['y1']),
        N('d2'): call(['z', 'y1'], ['y2']),
    }))


@pytest.fixture
def closed(sellar):
    '''`d1` / `d2` closed by a fixed point on `^copy.y2`: a problem with no driver.'''
    cut = Cut(V('y2'), (N('d1'),), HAT)(sellar)
    fp = RelationalCondition((Relation(HAT(V('y2')), V('y2'), Eq),), (), (HAT(V('y2')),), ())
    return graph_of(OrderedMap({**cut.definitions, N('fp'): fp}))


# ------------------------------------------------------------------------- the style

def test_every_style_is_found_by_name():
    assert style_of('terracotta') is TERRACOTTA
    assert set(STYLES) == {'terracotta', 'pyxdsm', 'dark'}
    with pytest.raises(ValueError, match='no style named'):
        style_of('sage')


def test_a_problem_without_a_driver_is_a_tint_of_one_with():
    fill = ROOT_FIND.colour[0]
    rf = ROOT_FIND.colour
    assert TERRACOTTA.fill(rf, driven=False) == mix(fill, '#ffffff', 1 - TERRACOTTA.tint)
    assert TERRACOTTA.fill(rf, driven=True) != TERRACOTTA.fill(rf, driven=False)
    assert PYXDSM.fill(rf, False) == PYXDSM.fill(rf, True)            # 'solid'


def test_a_ring_takes_its_drivers_hue_and_a_band_by_depth(closed):
    svg = render_xdsm_svg(closed, blocks=True)                       # `.fp` solves d1 / d2
    assert re.search(r'<rect class="group group-l0" [^>]*'
                     rf'stroke="{FIXED_POINT.colour[1]}"/>', svg)    # on the shape
    css = xdsm_css(TERRACOTTA)
    assert f'.group.group-nested.group-l1{{fill:{TERRACOTTA.bands[1]}}}' in css
    assert '.group-ff{fill:none;stroke:none}' in css                  # a run draws nothing


def test_a_category_is_coloured_once_and_each_ground_paints_it():
    for kind in KINDS:
        assert TERRACOTTA.paint(kind.colour) == PYXDSM.paint(kind.colour) == kind.colour
        fill, stroke = DARK.paint(kind.colour)
        assert (fill, stroke) != kind.colour and DARK.paint(kind.colour) == (fill, stroke)
    assert TERRACOTTA.paint(None) == TERRACOTTA.stated                # no category: grey


# ---------------------------------------------------------------- the open contour

def test_a_stylesheet_holds_no_rule_naming_a_category(closed):
    '''A category's colour is written on the shapes drawn in it, never as a class.'''
    for kw in ({}, {'style': 'pyxdsm'}, {'style': 'dark'}, {'follow_system': True}):
        svg = render_xdsm_svg(closed, blocks=True, **kw)
        sheet = re.search('<style>(.*?)</style>', svg).group(1)
        assert not any(k.name in sheet for k in KINDS) and 'ring-' not in sheet
        assert not any(f'prob-{k.name}' in svg for k in KINDS)
    assert FIXED_POINT.colour[1] in render_xdsm_svg(closed, blocks=True)


def test_a_problem_without_a_driver_is_drawn_open(closed):
    svg = render_xdsm_svg(closed, blocks=True)
    tint, hue = TERRACOTTA.fill(FIXED_POINT.colour, False), FIXED_POINT.colour[1]
    assert len(re.findall(r'<polygon class="prob" data-i="\d+" points=', svg)) == 1
    assert f'fill="{tint}" stroke="{hue}" stroke-width="1.40" style="stroke:none"' in svg
    chevrons = f'stroke="{hue}" stroke-width="1.40" style="fill:none"/>'
    assert svg.count(chevrons) == 2


def test_a_driven_problem_is_drawn_closed(closed):
    struct = xdsm_struct(closed, blocks=True)
    struct['driven'] = {'.fp'}
    svg = render_xdsm_svg(struct)
    fill, hue = TERRACOTTA.fill(FIXED_POINT.colour, True), FIXED_POINT.colour[1]
    assert re.search(rf'<polygon class="prob" data-i="\d+" points="[^"]*" fill="{fill}" '
                     rf'stroke="{hue}" stroke-width="1.40">', svg)
    assert '<path class="prob' not in svg


def test_a_solid_style_never_opens_a_contour(closed):
    assert '<polygon class="prob" ' in render_xdsm_svg(closed, style='pyxdsm')
    assert '<path class="prob' not in render_xdsm_svg(closed, style='pyxdsm')


def test_the_bands_are_the_ground(sellar):
    '''A band is filled, so it goes under the couplings and the chain, not over them.'''
    svg = render_xdsm_svg(sellar, blocks=True, edges=True)
    assert svg.index('class="group') < svg.index('class="edge"')


def test_the_weight_is_the_styles(sellar):
    assert '.lbl{fill:#3d3330;font-weight:normal}' in render_xdsm_svg(sellar)
    assert 'font-weight:600' in render_xdsm_svg(sellar, style='pyxdsm')


# ------------------------------------------------------------- names said outright

def test_a_minted_name_given_outright_is_drawn_as_given(closed):
    plain = xdsm_struct(closed)['shown_vars']['^copy.y2']
    named = xdsm_struct(closed, formatter=LatexFormatter({'^copy.y2': 'u'}))['shown_vars']['^copy.y2']
    assert plain == ('hat(.y2)', None)
    assert named == (LatexFormatter().set('u'), None)                  # no caret over `u`


def test_a_latex_drawing_writes_a_symbols_copy_with_a_caret():
    f = LatexFormatter()
    assert shown(HAT(V('y2')), f) == (f.set('.y_{2}'), 'hat')
    assert shown(purposes.Cond()(V('a')), f) == (f.set(r'\mathrm{R}(.a)'), None)
    assert shown(purposes.Start()(HAT(V('y'))), f) == (f.set(r'\mathrm{start}(\hat{.y})'), None)
    assert tex(r'\leq') == f'{MARKUP}≤{MARKUP}' or '≤' in tex(r'\leq')


def test_a_slot_kind_is_a_text_the_formatter_sets_from_its_own_source():
    assert LatexFormatter().text(Text('≤ 0', r'\leq 0')) == tex(r'\leq 0')
    assert LatexFormatter().text(Text('min', r'\min')).startswith(MARKUP)
    assert PlainFormat().text(Text('≤ 0', r'\leq 0')) == '≤ 0'


def test_a_latex_drawing_writes_a_copy_with_a_hat_and_an_unanswered_purpose_by_its_spelling():
    f = LatexFormatter()
    assert shown(purposes.Copy()(V('y2')), f) == (f.set(r'.y_{2}'), 'hat')
    assert shown(purposes.Guess()(V('y2')), f) == (f.set(r'\mathrm{guess}(.y_{2})'), None)


# ---------------------------------------------------------------- the node-link

def test_a_problem_is_a_hexagon_in_the_graph_too(closed):
    src = to_dot(closed)
    assert 'shape=hexagon style="filled,dashed"' in src             # no driver: broken
    assert FIXED_POINT.colour[1] in src


def test_a_solve_is_a_cluster(closed, sellar):
    assert 'subgraph cluster_1' in to_dot(closed, blocks=True)
    assert 'dashed' in to_dot(sellar, blocks=True).split('subgraph cluster_1')[1].split(';')[0]
    assert 'subgraph' not in to_dot(closed)


def test_a_hand_order_survives_blocks_when_problem_first_is_off(closed):
    '''`problem_first=False` says the caller ordered the graph: `blocks` rings it as it
    stands, rather than re-sorting it into run order first.'''
    hand = [r'.fp', '.z', '.d2', '.d1']
    order = graph_of(OrderedMap({n: closed[n] for n in (N('fp'), N('z'), N('d2'), N('d1'))}))
    assert xdsm_struct(order, blocks=True, problem_first=False)['steps'] == hand
    assert xdsm_struct(order, blocks=True)['steps'] != hand


def test_the_bare_arrows_are_edges_too(closed):
    '''With no boundary box the model's ends are bare arrows, drawn with the edges.'''
    svg = render_xdsm_svg(closed, blocks=True, input_label=None)
    assert '<line class="edge"' not in svg and 'class="io arrow"' not in svg


def test_a_relation_between_two_unknowns_is_slotted_as_written(closed):
    '''A problem owning a copy and its original holds `ŷ = y` between two of its own
    unknowns: nothing to receive, so the slot says the relation.'''
    fp = closed[N('fp')]
    both = RelationalCondition(fp.relations, (Objective(V('z'), Min),), (HAT(V('y2')), V('y2')), ())
    g = graph_of(OrderedMap({N('z'): call([], ['z']), N('fp'): both}))
    assert xdsm_struct(g)['slots']['.fp'][-1] == ('hat(.y2) = .y2', [])


def test_every_picture_rings_a_solve_alike(sellar, closed):
    '''A ring's contour, in the XDSM, the N2 and the node-link: its driver's hue, grey
    for a problem with no category, `unanswered` where no driver answers it.'''
    from cottax.consumers.visualization.matrix import render_matrix_svg
    fp = closed[N('fp')]
    two = RelationalCondition(fp.relations, (Objective(V('y1'), Min), Objective(V('z'), Min)),
                              fp.unknowns, ())
    uncategorised = graph_of(OrderedMap({**closed.definitions, N('fp'): two}))
    hex_ = '(#[0-9a-fA-F]+)'
    for graph, contour in ((closed, FIXED_POINT.colour[1]), (sellar, TERRACOTTA.unanswered),
                           (uncategorised, TERRACOTTA.stated[1])):
        xdsm = re.search(rf'class="group group-l0"[^>]* stroke="{hex_}"',
                         str(render_xdsm_svg(graph, blocks=True))).group(1)
        n2 = re.search(rf'class="solve"[^>]* stroke="{hex_}"',
                       str(render_matrix_svg(graph))).group(1)
        dot = re.search(rf'subgraph cluster_1 {{\s*style="[^"]*"; fillcolor="[^"]*"; '
                        rf'color="{hex_}"', to_dot(graph, blocks=True)).group(1)
        assert xdsm == n2 == dot == contour
