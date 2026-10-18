'''
What a drawing asks of what it draws (`consumers/visualization/interface.py`): a condition
node or a place nobody answered for still draws, and one answered for draws as answered --
with no language and no place of cottax's anywhere in the graph. Every class answered here
is this module's own, since a registration lasts the session.
'''

import re
from dataclasses import dataclass

import pytest

from cottax.core import ConditionalNode, FunctionalNode, Mint
from cottax.core.frozen import Choice, OrderedMap, answers
from cottax.consumers.visualization import DARK, TERRACOTTA, render_matrix_svg, render_xdsm_svg
from cottax.consumers.visualization.interface import Formatter, NodeDrawing, PlaceDrawing
from cottax.consumers.visualization.interface_types import Drawn, Kind, Slot, Text
from cottax.consumers.visualization.matrix import by_namespace, matrix_struct
from cottax.consumers.visualization.kinds import by_name
from cottax.consumers.visualization.naming import PlainFormat
from cottax.consumers.visualization.xdsm import xdsm_struct
from graphs import graph_of


# ---------------------------------------------------------------- a place of the test's own

@dataclass(frozen=True)
class Term:
    '''The whole of `Place`: segments, and a constructor taking them. Nobody answers
    for it, so it is drawn by the defaults.'''
    segments : tuple

    def __str__(self) -> str:
        return '/'.join(self.segments)


@dataclass(frozen=True)
class Dotted:
    '''A place that answers for itself: dot-joined, a `Mint` first segment minted.'''
    segments : tuple

    def __str__(self) -> str:
        return f'Dotted({self.segments})'


@answers(Dotted)
class DottedDrawing(PlaceDrawing):
    def spell_plain(self, place):
        return '.'.join(map(str, place.segments))

    def spelling(self, place):
        return str(place)

    def minted_from(self, place):
        first = place.segments[0] if place.segments else None
        return (first, Dotted(place.segments[1:])) if isinstance(first, Mint) else None

    def namespace_of(self, place, depth=None):
        _, under = self.minted_from(place) or (None, place)
        own = under.segments[:-1]
        return own if depth is None else own[:depth]


class Tilde(Mint):
    '''The test's own purpose, drawn by no notation: `tilde(·)`.'''


class Body(FunctionalNode):
    reads : tuple
    owns  : tuple
    def relabel(self, renaming): return self


class Unanswered(ConditionalNode):
    '''A condition node no language answers for.'''
    conditions : tuple
    unknowns   : tuple
    reads      = property(lambda self: self.conditions)
    owns       = property(lambda self: self.unknowns)
    def relabel(self, renaming): return self
    def determined(self, unknowns): return Unanswered(self.conditions, unknowns)


class Answered(ConditionalNode):
    '''One that is: a category of the test's own, and a slot per condition.'''
    conditions : tuple
    unknowns   : tuple
    reads      = property(lambda self: self.conditions)
    owns       = property(lambda self: self.unknowns)
    def relabel(self, renaming): return self
    def determined(self, unknowns): return Answered(self.conditions, unknowns)


TOY = Kind('toy', 'toy (r ~ 0)', 'a toy category: r near zero', True, ('#a1b2c3', '#123456'))
'''Named by no style: it draws by what it carries alone.'''


@answers(Answered)
class AnsweredDrawing(NodeDrawing):
    def drawn(self, node):
        r, s = node.conditions
        return Drawn(TOY, (Slot((Text('~ 0', r'\sim 0'),), (r,)),
                           Slot((Text('= ', '= '), node.unknowns[0]), (s,))))


def hexagon(svg: str, cls: str = 'prob') -> tuple[str, str]:
    '''The `(fill, stroke)` written on the one problem hexagon of class `cls`.'''
    (found,) = re.findall(rf'<polygon class="{cls}" data-i="\d+" points="[^"]*" '
                          r'fill="([^"]*)" stroke="([^"]*)"', svg)
    return found


def coupled(place, kind):
    '''A body and a condition node closing a cycle on `u`.'''
    p = lambda *s: place(s)
    return graph_of(OrderedMap({
        p('m', 'body'): Body((p('u'),), (p('r'), p('s'))),
        p('m', 'cond'): kind((p('r'), p('s')), (p('u'),)),
    }))


# ---------------------------------------------------------------- a condition node

def test_a_condition_node_nobody_answers_for_is_a_grey_problem_with_no_slots():
    g = coupled(Term, Unanswered)
    struct = xdsm_struct(g, blocks=True)
    assert struct['problem_types'] == {'m/cond': 'stated'}
    assert 'slots' not in struct and struct['stated_shapes'] == {}
    svg = render_xdsm_svg(g, blocks=True, condition_slots='all')
    assert hexagon(svg, 'prob prob-stated') == TERRACOTTA.stated     # grey, on the shape
    assert 'requirement, no unknowns named' in svg                   # the grey key entry


def test_an_answered_category_and_its_slots_are_drawn():
    g = coupled(Term, Answered)
    struct = xdsm_struct(g)
    assert struct['problem_types'] == {'m/cond': 'toy'}
    assert struct['slots'] == {'m/cond': [('~ 0', ['r']), ('= u', ['s'])]}
    assert struct['kinds'] == {'toy': TOY}
    # no style names it, and it is drawn all the same: coloured, keyed, hovered, slotted
    svg = render_xdsm_svg(g, condition_slots=True)                   # `slotted`: drawn
    assert '>~ 0<' in svg and '>= u<' in svg
    assert 'toy (r ~ 0)' in svg and '<title>a toy category: r near zero</title>' in svg
    assert hexagon(svg) == (TERRACOTTA.fill(TOY.colour, False), '#123456')   # never grey
    assert 'toy' not in re.search('<style>(.*?)</style>', svg).group(1)  # no rule names it
    ringed = render_xdsm_svg(g, blocks=True)
    assert '<rect class="group group-l0" ' in ringed and 'stroke="#123456"/>' in ringed
    assert '>~ 0<' not in render_xdsm_svg(g)                         # slots asked for
    assert matrix_struct(g)['kinds'] == {'toy': TOY}
    rows = matrix_struct(g)['rows']
    assert [r['kind'] for r in rows] == ['toy', None] and rows[0]['hue'] == '#123456'
    n2 = render_matrix_svg(g)
    assert f'fill="{TERRACOTTA.fill(TOY.colour, True)}"' in n2 and 'stroke="#123456"' in n2
    assert 'toy (r ~ 0)' in n2 and '<title>a toy category: r near zero</title>' in n2  # its key


def test_a_category_draws_in_its_own_colour_on_a_dark_ground():
    '''The dark style names no category either: it paints the one the category carries.'''
    g = coupled(Term, Answered)
    fill, stroke = DARK.paint(TOY.colour)
    assert (fill, stroke) != TOY.colour and stroke != DARK.stated[1]
    dark = render_xdsm_svg(g, blocks=True, style=DARK)
    assert hexagon(dark) == (DARK.fill(TOY.colour, False), stroke)
    assert '<rect class="group group-l0" ' in dark and f'stroke="{stroke}"/>' in dark
    auto = render_xdsm_svg(g, blocks=True, follow_system=True)       # the shapes: the light
    assert hexagon(auto) == (TERRACOTTA.fill(TOY.colour, False), '#123456')
    assert f'stroke="{stroke}"' not in auto and '@media (prefers-color-scheme:dark)' in auto
    assert matrix_struct(g, style=DARK)['rows'][0]['hue'] == stroke
    assert f'stroke="{stroke}"' in render_matrix_svg(g, style=DARK)


class Twin(ConditionalNode):
    '''One answered with another category under `TOY`'s name.'''
    conditions : tuple
    unknowns   : tuple
    reads      = property(lambda self: self.conditions)
    owns       = property(lambda self: self.unknowns)
    def relabel(self, renaming): return self
    def determined(self, unknowns): return Twin(self.conditions, unknowns)


TWIN = Kind('toy', 'another toy', 'not the toy category', False, ('#ffffff', '#000000'))


@answers(Twin)
class TwinDrawing(NodeDrawing):
    def drawn(self, node):
        return Drawn(TWIN, ())


def test_two_categories_with_one_name_are_refused_in_one_picture():
    '''A picture would draw one as the other: refused where its struct is built.'''
    p = lambda *s: Term(s)
    g = graph_of(OrderedMap({
        p('m', 'body'): Body((p('u'),), (p('r'), p('s'))),
        p('m', 'cond'): Answered((p('r'), p('s')), (p('u'),)),
        p('n', 'body'): Body((p('v'),), (p('a'), p('b'))),
        p('n', 'cond'): Twin((p('a'), p('b')), (p('v'),)),
    }))
    for build in (xdsm_struct, matrix_struct):
        with pytest.raises(TypeError, match="two categories are named 'toy'") as e:
            build(g)
        assert 'another toy' in str(e.value) and 'a toy category' in str(e.value)
    assert by_name([TOY, None, TOY]) == {'toy': TOY}                  # one category, twice


# ---------------------------------------------------------------- a place

def test_a_place_nobody_answers_for_is_drawn_as_its_str():
    g = coupled(Term, Answered)
    assert xdsm_struct(g)['steps'] == ['m/cond', 'm/body']
    assert 'm/body' in render_xdsm_svg(g, blocks=True)
    assert 'm/body' in render_matrix_svg(g)
    assert [r['group'] for r in matrix_struct(g, groups=by_namespace())['rows']] == ['', '']


def test_a_place_that_answers_is_spelt_minted_and_grouped_by_its_answers():
    g = graph_of(OrderedMap({
        Dotted(('m', 'body')): Body((Dotted((Tilde(), 'u')),), (Dotted(('u',)),)),
        Dotted(('n', 'use')) : Body((Dotted(('u',)),), (Dotted(('f',)),)),
    }))
    struct = xdsm_struct(g, formatter=PlainFormat())
    assert struct['steps'] == ['m.body', 'n.use']
    assert struct['shown_vars']['^tilde.u'] == ('tilde(u)', None)        # its word, no notation
    assert struct['step_groups'] == {'m.body': 'm', 'n.use': 'n'}
    assert 'tilde(u)' in render_xdsm_svg(g)
    assert [r['group'] for r in matrix_struct(g, groups=by_namespace())['rows']] == ['m', 'n']
    assert [r['minted'] for r in matrix_struct(g)['rows']] == [False, False]
    assert 'm.body' in render_matrix_svg(g, groups=by_namespace())


# ---------------------------------------------------------------- one answer per class

def test_a_class_is_answered_once():
    '''A second answer would replace the first silently, and what draws would depend on
    the order modules were imported in.'''
    class Again(NodeDrawing):
        def drawn(self, node):
            return Drawn(None, ())
    with pytest.raises(TypeError, match=r'test_drawn.Answered has the NodeDrawing answer '
                                        r'test_drawn.AnsweredDrawing already'):
        answers(Answered)(Again)


def test_an_answer_leaving_a_question_unanswered_is_refused():
    class Partial(PlaceDrawing):
        def spell_plain(self, place):
            return ''
    with pytest.raises(TypeError, match='abstract'):
        answers(Term)(Partial)


def test_a_choice_is_never_registered():
    with pytest.raises(TypeError, match='`Choice` is named by a study'):
        answers(Term)(Formatter)
    assert issubclass(Formatter, Choice)
