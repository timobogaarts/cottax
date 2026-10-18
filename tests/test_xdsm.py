'''
The XDSM drawing: the order it puts on the diagonal, and the notation it puts on a
variable name.

Names are built by hand from key entries, as in `test_algebra.py`.
'''

import re

import pytest
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

from cottax.core.maths.executable import ExecutableGraph
from cottax.relational import Min, Objective
from cottax.core.maths.plan import Plan
from cottax.core.frozen import OrderedMap
from cottax.core.maths.plan import Nest
from cottax.relational import Eq, Le, Relation, RelationalCondition
from cottax.core.maths.plan import Cut
from cottax.relational.rewrites import FixedPointCut
from cottax.pytree.path import NodePath, VarPath
from cottax.core import Function
from cottax.core.realisation import ImplementedFunction
import purposes
from cottax.relational.consumers.visualization import FIXED_POINT, OPTIMISE
from cottax.consumers.visualization.style import DARK, PYXDSM, TERRACOTTA
from cottax.pytree.consumers.visualization import NoFormat
from cottax.consumers.visualization.naming import MARKUP, PlainFormat, marked, markup_html, shown
from cottax.consumers.visualization.svg import BASE_FONT_SIZE, label, tw
from cottax.consumers.visualization.xdsm import (
    CELL_GAP, _baseline, _text, render_xdsm_svg, xdsm_struct,
)
from cottax.consumers.visualization.interface_types import Accent, Around
import cottax.assemblies.jax as cx
from cottax.consumers.visualization.refusals import SpelledAlike
from cottax.consumers.visualization.interface import Formatter
from graphs import graph_of

HAT, PROBLEM, START = purposes.Copy(), purposes.Solve(), purposes.Start()


def optimise(objective, unknowns, inequalities=()) -> RelationalCondition:
    '''Minimise one variable over `unknowns`, subject to `g <= 0`.'''
    return RelationalCondition(tuple(Relation(g, None, Le) for g in inequalities),
                               (Objective(objective, Min),), unknowns, ())


def root_find(conditions, unknowns) -> RelationalCondition:
    '''Every condition vanishes, over as many unknowns.'''
    return RelationalCondition(tuple(Relation(c, None, Eq) for c in conditions), (), unknowns, ())


def fixed_point(conditions, unknowns) -> RelationalCondition:
    '''`u = g(u)` per pair: the condition *is* the next iterate.'''
    return RelationalCondition(tuple(Relation(u, c, Eq) for u, c in zip(unknowns, conditions)),
                               (), unknowns, ())


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def call(reads, owns):
    return ImplementedFunction(Function(tuple(V(r) for r in reads), tuple(V(o) for o in owns)),
                       lambda *a: None)


@pytest.fixture
def sellar():
    '''`d1` and `d2` are coupled, and are bound with `obj` between them.'''
    return graph_of(OrderedMap({
        N('z')  : call([], ['z']),
        N('d1') : call(['z', 'y2'], ['y1']),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
        N('d2') : call(['z', 'y1'], ['y2']),
    }))


# ---------------------------------------------------------------- the diagonal order

def test_the_diagonal_is_binding_order(sellar):
    assert xdsm_struct(sellar)['steps'] == ['.z', '.d1', '.obj', '.d2']


def test_blocks_rebind_into_run_order(sellar):
    '''
    A block's box is the square [lo..hi] on the diagonal, so a scattered block would
    swallow whatever sat between its members: `blocks` draws the run order instead.
    '''
    struct = xdsm_struct(sellar, blocks=True)
    assert struct['steps'] == ['.z', '.d1', '.d2', '.obj']
    assert [(b['steps'], b['ring'], b['depth']) for b in struct['blocks']] \
        == [(['.z'], 'run', 0), (['.d1', '.d2'], 'solve', 0), (['.obj'], 'run', 0)]
    assert [b['type'] for b in struct['blocks']] == [None, 'cycle', None]


def test_a_problem_leads_the_block_it_drives(sellar):
    '''
    A rewrite mints its problem last, so every order the package reports puts it at the end
    of its block. The XDSM convention is a driver at the head of what it drives, so the
    drawing permutes it there -- within the block's own slots, so `.obj` (which sits
    between `.d1` and `.d2` in binding order) does not move.
    '''
    g = FixedPointCut((Cut(V('y2'), (N('d1'),), HAT),), PROBLEM(N('y2')))(sellar)
    assert [n.spelling for n in g.nodes] == ['.z', '.d1', '.obj', '.d2', '^solve.y2']

    assert xdsm_struct(g)['steps'] == ['.z', '^solve.y2', '.obj', '.d1', '.d2']

    struct = xdsm_struct(g, blocks=True)
    assert struct['steps'] == ['.z', '^solve.y2', '.d1', '.d2', '.obj']
    assert [(b['steps'], b['ring'], b['depth']) for b in struct['blocks']] \
        == [(['.z'], 'run', 0), (['^solve.y2', '.d1', '.d2'], 'solve', 0), (['.obj'], 'run', 0)]


def test_the_blocking_is_the_solver_blocking(sellar):
    blocked = [b['steps'] for b in xdsm_struct(sellar, blocks=True)['blocks']]
    assert blocked == [[n.spelling for n in c]
                       for c in sellar.graph.components]


def test_only_a_coupled_block_is_ringed_as_one(sellar):
    svg = render_xdsm_svg(sellar, blocks=True)
    grey = f'stroke="{TERRACOTTA.unanswered}"'
    ring = r'<{} class="group group-l0" [^>]*' + grey
    assert len(re.findall(ring.format('rect'), svg)) == 1              # d1 / d2
    assert len(re.findall(ring.format('path'), svg)) == 2              # ... cut: nobody solves it
    assert 'class="group-ff"' not in svg          # a style with no `run_ring` draws none
    pyxdsm = render_xdsm_svg(sellar, blocks=True, style='pyxdsm')
    assert pyxdsm.count('class="group-ff"') == 2  # z, obj: run, not solved
    assert '<path class="group' not in pyxdsm


def test_the_figure_and_the_page_key_alike_and_once(sellar, nestable):
    '''One key, each entry once: under `PYXDSM` run rings at two depths are one entry,
    and a ring no driver answers is keyed in either picture.'''
    from cottax.consumers.visualization.page import Pictures
    from cottax.consumers.visualization.xdsm import key_entries
    from cottax.consumers.visualization.xdsm_html import xdsm_view
    two = graph_of(OrderedMap({**nestable.definitions, N('pre'): call([], ['k'])}))
    nested = ExecutableGraph((Plan(two) + Nest(N('h'), N('opt'))).graph)
    assert {(b['ring'], b['depth']) for b in xdsm_struct(two, blocks=nested)['blocks']
            if b['ring'] == 'run'} == {('run', 0), ('run', 1)}
    for graph, blocks in ((two, nested), (sellar, True)):
        for style in (PYXDSM, TERRACOTTA):
            struct = xdsm_struct(graph, blocks=blocks)
            texts = [e['text'] for e in key_entries(struct, style)]
            assert len(texts) == len(set(texts))
            assert ('run in order' in texts) == (style is PYXDSM)
            assert ('no driver' in texts) == (graph is sellar)
            page = xdsm_view(graph, Pictures(), blocks=blocks, style=style)['legend']
            assert [e['text'] for e in page] == texts
            svg = render_xdsm_svg(graph, blocks=blocks, style=style)
            assert [t for t in texts if svg.count(f'>{t}</text>') != 1] == []


def test_without_blocks_nothing_is_ringed(sellar):
    svg = render_xdsm_svg(sellar)
    assert 'class="group' not in svg


def test_an_empty_graph_has_nothing_to_draw():
    with pytest.raises(ValueError, match='no nodes'):
        xdsm_struct(graph_of(OrderedMap()))


# ---------------------------------------------------------------- the problem types

@pytest.fixture
def mdf():
    '''`d1` / `d2` closed by a fixed point on `^copy.y2`, under an optimiser on `.z`.'''
    hat_y2 = Cut(V('y2'), [N('d1')], HAT).copy
    return graph_of(OrderedMap({
        N('opt'): optimise(V('f'), (V('z'),)),
        N('d1') : ImplementedFunction(Function((V('z'), hat_y2), (V('y1'),)), lambda *a: None),
        N('d2') : call(['z', 'y1'], ['y2']),
        N('mda'): fixed_point((V('y2'),), (hat_y2,)),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
    }))


def test_a_problem_says_which_type_it_is_and_a_body_has_none(mdf, sellar):
    assert xdsm_struct(mdf)['problem_types'] == {'.opt': 'optimise', '.mda': 'fixed-point'}
    assert xdsm_struct(sellar)['problem_types'] == {}


def test_a_problems_conditions_are_slotted_by_what_each_one_says():
    """One slot per kind, in the order a driver is handed them: the objective, then the
    relations as written -- against zero, bounded, and a two-sided one named for the side
    the problem owns."""
    hat_y2 = HAT(V('y2'))
    g = graph_of(OrderedMap({
        N('opt'): RelationalCondition(
            (Relation(V('h'), None, Eq), Relation(V('g'), None, Le),
             Relation(hat_y2, V('y2'), Eq)),
            (Objective(V('f'), Min),), (V('z'), hat_y2), ()),
        N('m')  : ImplementedFunction(Function((V('z'), hat_y2),
                                       (V('f'), V('g'), V('h'), V('y2'))),
                              lambda *a: None),
    }))
    assert xdsm_struct(g)['slots']['.opt'] == [
        ('min', ['.f']), ('= 0', ['.h']), ('\u2264 0', ['.g']), ('= hat(.y2)', ['.y2']),
    ]


def test_an_inequality_between_two_places_is_slotted_as_one():
    """A bound is named for the unknown it bounds, and a limit held fixed is written as it
    stands: neither is a read at most zero."""
    g = graph_of(OrderedMap({
        N('opt'): RelationalCondition(
            (Relation(V('lo'), V('u'), Le), Relation(V('u'), V('hi'), Le),
             Relation(V('g'), V('g_max'), Le)),
            (Objective(V('f'), Min),), (V('u'),), (V('lo'), V('hi'), V('g_max'))),
        N('m')  : call(['u'], ['f', 'g']),
    }))
    assert xdsm_struct(g)['slots']['.opt'] == [
        ('min', ['.f']), ('\u2264 .u', ['.lo']), ('.u \u2264', ['.hi']),
        ('.g \u2264 .g_max', ['.g', '.g_max']),
    ]


def test_a_requirement_says_what_it_requires():
    """A requirement is slotted as a problem is, and its slots drawn under `True`, whatever
    category it will take: they are all it says of what it holds."""
    g = graph_of(OrderedMap({
        N('a')  : call([], ['x']),
        N('req'): RelationalCondition((Relation(V('x'), None, Eq),), (), (), ()),
    }))
    assert xdsm_struct(g)['slots'] == {'.req': [('= 0', ['.x'])]}
    assert 'class="slot"' in render_xdsm_svg(g, condition_slots=True)
    assert 'class="slot"' not in render_xdsm_svg(g)


def test_a_statement_with_no_unknowns_draws_as_stated():
    '''A shape is what the relations *and* the unknowns make, so one that determines
    nothing has none yet -- the fallthrough, and what `Determine` gives it a shape.'''
    g = graph_of(OrderedMap({
        N('a')   : call([], ['x']),
        N('prob'): RelationalCondition((Relation(V('x'), None, Eq),), (), (), ()),
        N('b')   : call(['u'], ['x2']),
    }))
    assert xdsm_struct(g)['problem_types'] == {'.prob': 'stated'}
    assert 'prob-stated' in render_xdsm_svg(g)


def test_a_problem_is_drawn_apart_from_a_body(mdf):
    '''Shape as well as colour: a problem is a chevron box, an analysis a rect.'''
    svg = render_xdsm_svg(mdf)
    hexagons = re.findall(r'<polygon class="prob" [^>]* stroke="([^"]*)"', svg)
    assert hexagons == [OPTIMISE.colour[1], FIXED_POINT.colour[1]]
    assert svg.count('class="diag"')                  == 3      # d1, d2, obj


def test_the_key_names_only_what_was_drawn(mdf, sellar):
    svg = render_xdsm_svg(mdf)
    assert '>optimise<' in svg and 'fixed-point (u = g(u))' in svg
    assert 'root-find (r = 0)' not in svg              # the palette carries every rule,
    assert 'solved together' not in svg                # only the key says what was drawn
    assert 'solved together' in render_xdsm_svg(sellar, blocks=True)


def test_a_key_sample_is_not_counted_as_a_drawn_thing(mdf):
    '''The sample leads with `key`, so it carries the same classes without being one.'''
    svg = render_xdsm_svg(mdf)
    samples = re.findall(r'<polygon class="key prob" [^>]* stroke="([^"]*)"', svg)
    assert samples == [OPTIMISE.colour[1], FIXED_POINT.colour[1]]       # coloured as drawn
    assert svg.count('<polygon class="prob" ') == 2                    # ... and not counted
    assert not render_xdsm_svg(mdf, legend=False).count('class="key')


# ---------------------------------------------------------------- the notation

PLAIN = PlainFormat()
CARET = PlainFormat({purposes.Copy: Accent('hat', 'hat({})')})


@pytest.mark.parametrize('path, drawn', [
    (V('y2'),                 ('.y2',        None)),
    (HAT(V('y2')),            ('hat(.y2)',   None)),     # plain text takes no caret by default
    (purposes.Cond()(V('y2')), ('R(.y2)',     None)),
    (purposes.Guess()(V('y2')), ('guess(.y2)', None)),   # a namespace nothing names
    (HAT(VarPath(())),        ('^copy',       None)),     # nothing to write it over
])
def test_a_minted_name_is_its_namespaces_notation_over_what_it_stands_for(path, drawn):
    assert shown(path, PLAIN) == drawn


def test_a_purpose_is_drawn_by_its_answer_else_by_its_spelling():
    '''A notation is the purpose owner's `MintDrawing`; one nobody answers for is written
    around its spelling.'''
    assert shown(purposes.Cond()(V('y2')), PlainFormat()) == ('R(.y2)', None)
    assert shown(PROBLEM(N('d1')), PlainFormat()) == ('solve(.d1)', None)


def test_a_mint_over_a_mint_is_notated_all_the_way_down():
    assert shown(START(HAT(V('y2'))), PLAIN) == ('start(hat(.y2))', None)


def test_an_accent_is_a_mark_and_spelled_where_none_can_be_drawn():
    '''The mark goes over the outermost name only; inside another notation it is
    written out, by the accent's own spelling.'''
    assert shown(HAT(V('y2')), CARET) == ('.y2', 'hat')
    assert shown(START(HAT(V('y2'))), CARET) == ('start(hat(.y2))', None)


def test_a_formatter_chooses_a_namespaces_notation():
    prime = PlainFormat({purposes.Copy: Around('{}′'), purposes.Solve: Around('S[{}]')})
    assert shown(HAT(V('y2')), prime) == ('.y2′', None)
    assert shown(PROBLEM(N('d1')), prime) == ('S[.d1]', None)


def test_no_format_is_the_jax_spelling_minted_or_not():
    assert shown(HAT(V('y2')), NoFormat()) == (HAT(V('y2')).spelling, None)


def test_a_mark_that_cannot_be_drawn_is_a_combining_character():
    assert marked(('y', 'hat')) == 'ŷ'[0] + '\u0302' or marked(('y', 'hat')) == 'y\u0302'
    assert marked((f'{MARKUP}<I>y</I><SUB>2</SUB>{MARKUP}', 'hat')) == \
        f'{MARKUP}<I>y\u0302</I><SUB>2</SUB>{MARKUP}'
    assert markup_html(f'a<b {MARKUP}<I>y</I>{MARKUP}') == 'a&lt;b <i>y</i>'


def test_the_struct_keys_by_spelling_and_carries_what_is_drawn(sellar):
    '''Every name keyed by its plain spelling, whatever a formatter makes of it, and what
    is drawn for each carried beside it -- so no renderer spells.'''
    op     = FixedPointCut((Cut(V('y2'), (N('d1'),), HAT),), PROBLEM(N('y2')))
    struct = xdsm_struct(op(sellar), formatter=PLAIN)
    assert '^solve.y2' in struct['steps'] and '^copy.y2' in struct['producer']
    assert struct['shown_steps']['^solve.y2'] == ('solve(.y2)', None)
    assert struct['shown_vars']['^copy.y2'] == ('hat(.y2)', None)
    assert xdsm_struct(op(sellar), formatter=CARET)['shown_vars']['^copy.y2'] == ('.y2', 'hat')


def test_two_places_spelled_alike_are_refused_naming_both():
    '''`['a']['b']` and `['a'].b` both spell `a.b`: keyed by it, the coupling would be
    drawn from whichever owner came last.'''
    ab, a_b = VarPath((DictKey('a'), DictKey('b'))), VarPath((DictKey('a'), GetAttrKey('b')))
    g = graph_of(OrderedMap({
        N('p'): Function((), (ab,)), N('q'): Function((), (a_b,)),
        N('r'): Function((ab,), (V('z'),)),
    }))
    for draw in (xdsm_struct, render_xdsm_svg):
        with pytest.raises(ValueError) as e:
            draw(g)
        assert e.value.args[0] == SpelledAlike(ab, a_b, 'a.b').message


def test_two_node_names_spelled_alike_are_refused_naming_both():
    n0, s0 = NodePath((DictKey(0),)), NodePath((SequenceKey(0),))
    g = graph_of(OrderedMap({n0: Function((), (V('x'),)), s0: Function((V('x'),), (V('y'),))}))
    with pytest.raises(ValueError) as e:
        render_xdsm_svg(g)
    assert e.value.args[0] == SpelledAlike(n0, s0, '[0]').message


def test_a_cut_copy_is_drawn_with_a_hat_when_the_formatter_says_so(sellar):
    cut = Cut(V('y2'), [N('d1')], HAT)(sellar)
    svg = render_xdsm_svg(cut, formatter=CARET)
    assert '^copy' not in svg                    # the namespace is a spelling, not a label
    assert svg.count('<polyline class="deco"') == 1
    plain = render_xdsm_svg(cut)
    assert 'hat(.y2)' in plain and 'class="deco"' not in plain


@pytest.fixture
def nestable():
    '''An optimiser around a root-find, both in one block: two problems, so it can be
    nested either way round.'''
    return graph_of(OrderedMap({
        N('opt'): optimise(V('f'), (V('z'),)),
        N('d')  : call(['z', 'r'], ['u']),
        N('h')  : root_find((V('u'),), (V('r'),)),
        N('obj'): call(['u'], ['f']),
    }))


def test_a_nested_solve_is_a_ring_inside_a_ring(nestable):
    '''The interior draws its own blocks, so nesting shows as containment and the colour
    is left to say *which level* -- level 0 the solver colour, level 1 the next.'''
    nested = ExecutableGraph((Plan(nestable) + Nest(N('h'), N('opt'))).graph)

    # Un-nested, the two problems share one block and no blocking exists; `blocks=True`
    # still rings the component, since the drawing walks `entries(graph.graph)`.
    assert [(b['ring'], b['depth']) for b in xdsm_struct(nestable, blocks=True)['blocks']] \
        == [('solve', 0)]
    assert [(b['ring'], b['depth']) for b in xdsm_struct(nestable, blocks=nested)['blocks']] \
        == [('solve', 0), ('solve', 1), ('run', 1)]

    svg = render_xdsm_svg(nestable, blocks=nested)
    assert 'group-l0' in svg and 'group-l1' in svg
    assert 'solved together' in svg and 'solved one level in' in svg


def test_the_outer_problem_leads_so_every_interior_is_contiguous(nestable):
    '''A ring is a rect over `[lo..hi]` on the diagonal, so an interior has to be a
    contiguous run. Pinning the problem answered *outside* to the head is what leaves the
    interior occupying the rest of it -- at every depth.'''
    nested = ExecutableGraph((Plan(nestable) + Nest(N('h'), N('opt'))).graph)
    struct = xdsm_struct(nestable, blocks=nested)

    at = {s: i for i, s in enumerate(struct['steps'])}
    assert struct['steps'][0] == '.opt'                 # the outer problem leads
    for b in struct['blocks']:
        slots = sorted(at[s] for s in b['steps'])
        assert slots == list(range(slots[0], slots[-1] + 1))


def test_nesting_the_other_way_leads_with_the_other_problem(nestable):
    nested = ExecutableGraph((Plan(nestable) + Nest(N('opt'), N('h'))).graph)
    assert xdsm_struct(nestable, blocks=nested)['steps'][0] == '.h'


@pytest.fixture
def prelude():
    """
    An interior that has to run something *before* the problem it holds.

    `con` reads only what the outer problem owns, so it is not on the inner optimiser's
    cycle -- but the optimiser reads what it writes, so inside one outer iterate it runs
    first. Nothing of that is visible in the block minus its problems: dropping `opt` and
    `h` drops every edge `con` has.
    """
    return graph_of(OrderedMap({
        N('opt'): optimise(V('f'), (V('x'),), (V('g'),)),
        N('r')  : call(['a', 'x'], ['res']),
        N('h')  : root_find((V('res'),), (V('a'),)),
        N('obj'): call(['a', 'x'], ['f']),
        N('con'): call(['a'], ['g']),
    }))


def test_ordered_graph_is_the_binding_in_run_order():
    """Run order is a presentation, so it is spelled here and not on `ExecutableGraph` --
    a diagonal is one line, and a run never needs the whole graph in one order."""
    from cottax.consumers.visualization.sequencing import ordered_graph

    g = graph_of(OrderedMap({
        N('c'): call(['b'], ['c']),
        N('a'): call([],    ['a']),
        N('b'): call(['a'], ['b']),
    }))
    assert [n.spelling for n in ordered_graph(ExecutableGraph(g)).nodes] == ['.a', '.b', '.c']


def test_an_interior_runs_in_its_own_order(prelude):
    """The interior is a blocking, so it *states* the order it runs in -- `con`, then the
    optimiser it feeds -- and the diagonal is that order and not the one the block was
    written in."""
    # `con` is constant for the inner optimiser, so this nesting has no blocking -- and
    # is drawn anyway, `blocks=True` walking `entries(graph)`.
    steps = xdsm_struct((Plan(prelude) + Nest(N('opt'), N('h'))).graph, blocks=True)['steps']

    assert steps[0] == '.h'                          # the outer problem still leads
    assert steps.index('.con') < steps.index('.opt')  # ... and the interior runs in order


def test_a_nested_problem_leads_its_own_level_and_not_the_one_above(prelude):
    """Pinned to the head of the block that *contains* its level, an inner problem is
    dealt in ahead of a step that runs before it, and the ring over its level then spans a
    step the level does not hold."""
    struct = xdsm_struct((Plan(prelude) + Nest(N('opt'), N('h'))).graph, blocks=True)

    at   = {s: i for i, s in enumerate(struct['steps'])}
    ring = [b['steps'] for b in struct['blocks'] if (b['ring'], b['depth']) == ('solve', 1)]
    (inner,) = ring
    assert set(inner) == {'.opt', '.obj'}
    slots = sorted(at[s] for s in inner)
    assert slots == list(range(slots[0], slots[-1] + 1))   # contiguous: a rect over its own
    assert at['.con'] < slots[0]


# --------------------------------------------------------------- the boundary block

def test_the_boundary_boxes_are_coor_and_out_and_the_caller_can_name_them(sellar):
    '''The environment the run starts and ends in -- what the DSM calls the
    coordinator -- not merely a bag of inputs.'''
    svg = render_xdsm_svg(sellar)
    assert '>Coor<' in svg and 'Inputs' not in svg and '>Out<' in svg
    assert '>start<' in render_xdsm_svg(sellar, input_label='start')
    assert '>results<' in render_xdsm_svg(sellar, output_label='results')


def test_every_result_leaves_down_the_lane_into_the_results_box(sellar):
    """
    A result sat in the *first* column with a line coming in from its producer and nothing
    leaving, so the picture never said where it went -- and it sat there **below the
    diagonal**, in the half a reordered matrix is drawn to have nothing in.

    So it leaves the way a coupling does: right along its producer's row, into a column of
    its own past the diagonal, and down that one lane into a box in the far corner. Drawn
    in `Coor`'s own colour, because it is the same world outside the model. The run's own
    return ends there too: sourced top left, spent bottom right.
    """
    struct = xdsm_struct(sellar)
    steps  = struct['steps']
    coupled = {f for dst in steps for f in struct['reads'][dst]
               if (p := struct['producer'].get(f)) is not None and p != dst}
    results = {f for s in steps for f in struct['writes'][s] if f not in coupled}
    assert results                                    # the fixture has some

    svg = render_xdsm_svg(sellar, edges=True)
    ext = [tuple(map(float, m)) for m in re.findall(
        r'<rect class="ext" x="([\d.]+)" y="([\d.]+)" '
        r'width="([\d.]+)" height="([\d.]+)"', svg)]
    assert len(ext) == 2                              # `Coor`, and the results box
    (ix, _, _, _), (ox, oy, ow, _) = ext
    lane, foot = ox + ow / 2, oy
    assert lane > ix                                  # the far corner, not `Coor`'s column

    # past the diagonal, which is the whole of why it moved: nothing a result draws is in
    # the lower triangle
    diag = [float(m) for m in re.findall(r'<rect class="diag"[^>]* x="([\d.]+)"', svg)]
    assert diag
    assert all(x < lane for x in diag)

    # every result box is in that column, above the box, with its own leg down the lane
    # into it -- one lane, drawn once per result and every copy on the same x
    on_lane = [m for m in re.finditer(
        r'<rect class="data"[^>]* x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg)
        if abs(float(m.group(1)) + float(m.group(3)) / 2 - lane) < 1]
    assert on_lane
    assert all(float(m.group(2)) + float(m.group(4)) / 2 < foot for m in on_lane)

    def ends(kind):
        return [tuple(map(float, m.group(1).split()[-1].split(',')))
                for m in re.finditer(rf'<polyline class="{kind}"[^>]*points="([^"]+)"', svg)]

    down = [p for p in ends('edge') if abs(p[0] - lane) < 1]
    assert len(down) == len(on_lane)


def test_with_nothing_leaving_the_run_still_closes(sellar):
    '''A model whose every write is read by another step has no results box, and the
    return closes back into `Coor` -- there is nowhere else for it to end.'''
    inner = sellar.subgraph([N('d1'), N('d2')])

    svg = render_xdsm_svg(inner)
    assert svg.count('<rect class="ext"') == 1
    assert '>Out<' not in svg


# ------------------------------------------------------------- the type, and the switches

@pytest.fixture
def boundary():
    '''Three free inputs, one result: a graph with both bands to draw.'''
    return graph_of(OrderedMap({
        N('a'): call(['p', 'q', 'r'], ['u']),
        N('b'): call(['u'], ['v']),
    }))


class _Marked(Formatter):
    '''A formatter that sets `y2` as `y` with a subscript, the way a paper would.'''

    def spell(self, path):
        return path.spelling

    def set(self, source):
        return f'{MARKUP}<I>{source[1]}</I><SUB>{source[2:]}</SUB>{MARKUP}' \
            if len(source) > 2 else source

    def label(self, text):
        return text

    def text(self, text):
        return text.plain

    def outright(self, path):
        return False

    def notation(self, ns):
        return Around(f'{ns}({{}})')


def test_a_marked_run_is_set_as_markup(sellar):
    svg = render_xdsm_svg(sellar, formatter=_Marked())

    assert '<tspan font-style="italic">y</tspan>' in svg
    assert MARKUP not in svg and '<SUB>' not in svg


def test_an_unmarked_name_is_drawn_as_the_text_it_is():
    '''Markup is what a formatter *says* is markup -- a name holding a tag is a name.'''
    graph = graph_of(OrderedMap({N('a<b'): call([], ['c'])}))

    assert 'a&lt;b' in render_xdsm_svg(graph)


def test_a_named_face_is_set_at_its_own_weight():
    """
    A face is named where the drawing face has no such glyph (`\\mathcal{D}` is one), and
    asking the named face for a *weight* it does not carry loses the glyph all over
    again -- a diagonal label is set at 600, and a script capital is a regular-weight
    thing in most faces.
    """
    assert label(f'{MARKUP}<FONT FACE="STIX">D</FONT>{MARKUP}', 13.0) \
        == '<tspan font-family="STIX" font-weight="normal">D</tspan>'


def test_a_subscript_is_measured_as_it_is_set():
    '''The box is cut to what the label draws, so a subscript may not measure as a letter.'''
    assert tw(f'{MARKUP}<I>y</I><SUB>2</SUB>{MARKUP}') < tw('y2')


def test_the_face_and_the_size_are_the_callers(sellar):
    set_up = render_xdsm_svg(sellar, font='Times New Roman, serif', font_size=20)

    assert 'font-family:Times New Roman, serif' in set_up
    assert 'font-size:20.00px' in set_up


def test_a_box_is_its_label_plus_an_em_of_air(sellar):
    """
    Which is the whole of the horizontal rule: type-sized air inside the box, a fixed gap
    outside it. So doubling the type doubles the box and the label in it and leaves the
    gap where it stood -- a box is never held open by a floor cut for some other size,
    which is what had a short name sitting in a third of its own box.

    Down the rows there is no label to be cut for, so the box *is* the floor and follows
    the type outright, with the gap outside it exactly as across -- which is what leaves
    `gap_y` something to close.
    """
    plain = render_xdsm_svg(sellar)
    big   = render_xdsm_svg(sellar, font_size=2 * BASE_FONT_SIZE)

    assert big.cell[0] - plain.cell[0] == pytest.approx(plain.cell[0] - CELL_GAP)
    ph, bh = _diag_boxes(plain)[0][1], _diag_boxes(big)[0][1]
    assert bh == pytest.approx(2 * ph)
    assert big.cell[1] - plain.cell[1] == pytest.approx(plain.cell[1] - CELL_GAP)


def _width(svg):
    return float(re.search(r'<svg[^>]* width="([\d.]+)"', svg).group(1))


def _geometry(svg):
    '''The drawing with its stylesheet taken off -- what a CSS switch may not touch.'''
    return re.sub(r'<style>.*?</style>', '', svg)


def test_hidden_labels_cost_no_geometry(sellar):
    """
    It takes ink away and never a distance, so what is left is the rest of the same
    picture -- which is what makes it free.

    Taken away and not merely styled away: a converter that honours `display:none` on a
    polyline but still draws its `marker-end` leaves a figure with arrowheads floating
    over nothing, and cairosvg is one.
    """
    gone = r'<text[^>]*>.*?</text>|<polyline class="deco"[^>]*/>'
    svg  = render_xdsm_svg(sellar, names=False)

    assert not re.search(gone, svg)
    assert _geometry(svg) == re.sub(gone, '', _geometry(render_xdsm_svg(sellar)))


def test_the_edges_are_ink_on_the_same_picture(sellar):
    '''A coupling's box already says where it comes from and goes; `edges` draws the
    line through it as well, and moves nothing.'''
    plain, lined = render_xdsm_svg(sellar), render_xdsm_svg(sellar, edges=True)
    assert 'class="edge"' not in plain and 'class="proc"' not in plain
    assert _geometry(plain) == re.sub(r'<polyline class="edge"[^>]*/>', '', _geometry(lined))


def test_a_thickened_ring_is_drawn_solid_at_the_weight_it_is_given(sellar):
    '''At that weight a dash pattern reads as a broken line, and the colour already says
    solver from run.'''
    svg = render_xdsm_svg(sellar, blocks=True, thick_blocks=True)

    assert 'stroke-dasharray:none' in svg
    assert '.group,.group-ff{stroke-width:4.00' in svg
    assert '.group,.group-ff{stroke-width:2.50' in render_xdsm_svg(
        sellar, blocks=True, thick_blocks=2.5)
    assert _geometry(svg) == _geometry(render_xdsm_svg(sellar, blocks=True))


def test_stacked_inputs_are_one_row(boundary):
    """
    A row per boundary variable is the default and is what gives each one a track across
    the matrix; on a large model that band is most of the picture's height, and this folds
    it onto one row -- one box holding every name, where there were as many boxes as names.
    """
    spread  = render_xdsm_svg(boundary)
    stacked = render_xdsm_svg(boundary, stack_inputs=True)

    assert all(f'>.{v}<' in stacked for v in 'pqr')       # all three still drawn
    assert stacked.count('<rect class="data"') == spread.count('<rect class="data"') - 2


def test_collapsed_names_count_what_they_stand_for(boundary):
    svg = render_xdsm_svg(boundary, collapse_names=True)

    assert '>3 vars<' in svg and '>1 var<' in svg
    assert '<title>.p, .q, .r</title>' in svg             # the names, one hover away
    assert '>.p<' not in svg


# --------------------------------------------------------------- two panels, one zoom

def _diag_boxes(svg: str) -> list[tuple[float, float]]:
    '''Every ordinary diagonal box as `(width, height)`, in the order they were drawn.'''
    return [(float(w), float(h)) for w, h in
            re.findall(r'<rect class="diag" [^>]*width="([\d.]+)" height="([\d.]+)"', svg)]


def _diag_widths(svg: str) -> list[float]:
    '''The width of every ordinary diagonal box, in the order they were drawn.'''
    return [w for w, _ in _diag_boxes(svg)]


def test_a_column_is_cut_for_its_own_content(sellar):
    '''`obj` is a longer name than `z`, and only `obj`'s column is cut for it: the whole
    matrix used to be as wide as its longest label, everywhere.'''
    widths = _diag_widths(render_xdsm_svg(sellar))

    assert len(widths) == 4
    assert len(set(widths)) > 1


def test_a_pinned_cell_is_the_grid_the_drawing_uses(sellar):
    '''Each column is cut for its own content, so two panels of one figure are on two
    grids unless one is told the other's -- which is what pinning a cell says: every model
    column on one pitch, and every row on one too.'''
    plain  = render_xdsm_svg(sellar)
    w, h   = plain.cell
    pinned = render_xdsm_svg(sellar, cell=(w, h))
    svg    = render_xdsm_svg(sellar, cell=(w + 40, h + 10))
    n      = len(xdsm_struct(sellar)['steps'])

    assert svg.cell == (w + 40, h + 10)
    assert len(set(_diag_widths(pinned))) == 1                # one pitch, every column
    # `.cell` is the widest column the drawing used, so putting every column there is at
    # least as wide as cutting each to its own -- and 40 more per column is 40n more.
    assert pinned.size[0] >= plain.size[0]
    assert svg.size[0] == pinned.size[0] + 40 * n             # a column per step, wider
    assert svg.size[1] > pinned.size[1]


def test_a_pinned_canvas_pads_centres_or_aligns_and_never_shrinks(sellar):
    '''What a larger box buys is that two panels printed to one width on a page are at
    one scale; centred in it by default, since a panel in the corner of its box reads as
    misplaced rather than as smaller -- or aligned, as a matrix of panels wants, with the
    slack at the bottom right. Never shrinking: a picture cut down would lose ink.'''
    plain = render_xdsm_svg(sellar)
    box   = (plain.size[0] + 100, plain.size[1] + 40)
    svg   = render_xdsm_svg(sellar, canvas=box)

    assert svg.size == box
    assert '<g transform="translate(50.0,20.0)">' in svg
    assert '<g transform="translate(0.0,0.0)">' in render_xdsm_svg(
        sellar, canvas=box, align='top-left')
    assert '<g transform="translate(50.0,0.0)">' in render_xdsm_svg(
        sellar, canvas=box, align='top')
    assert '<g transform="translate(0.0,20.0)">' in render_xdsm_svg(
        sellar, canvas=box, align='left')
    assert render_xdsm_svg(sellar, canvas=(10, 10)).size == plain.size


def test_a_label_is_centred_on_its_own_ink(sellar):
    """
    The font's vertical middle leaves room above for an ascender a lowercase has not, so a
    box holding `a` reads as holding it low. Each label is centred on what it actually
    draws instead -- so `a` and `b` do *not* share a baseline, which is the point.
    """
    assert _baseline('b', 100, 20) > _baseline('a', 100, 20)   # the ascender sits it down
    assert _baseline('y', 100, 20) < _baseline('a', 100, 20)   # the descender lifts it
    assert _baseline('a', 100, 20) == 105                      # half its x-height, down

    # and nothing is left to the renderer's own idea of a middle
    assert 'dominant-baseline' not in render_xdsm_svg(sellar)


def test_every_line_can_be_thickened_at_once(sellar):
    '''A figure printed at half its size wants heavier ink, since scaling the picture
    scales the ink with it.'''
    svg = render_xdsm_svg(sellar, line_scale=2, edges=True)

    assert '.diag,.ext{stroke-width:2.80}' in svg
    assert 'stroke-width="4.00"' in svg               # a coupling's polyline, doubled


def test_a_label_with_markup_is_placed_and_not_anchored():
    '''`text-anchor` on a text made of tspans is honoured per-tspan by at least one
    converter, which sets `V` and its subscript with a hole between them. Only a plain
    label -- where nothing has to be estimated -- keeps the anchor.'''
    marked = f'{MARKUP}<I>V</I><SUB>1</SUB>{MARKUP}'

    assert 'text-anchor' not in _text(marked, 100, 50, 20, 'lbl')
    assert 'text-anchor="middle"' in _text('V1', 100, 50, 20, 'lbl')
    # placed half its own width left of where it would have been anchored
    assert 'x="90.2"' in _text(marked, 100, 50, 20, 'lbl')


def test_the_gap_between_boxes_is_the_callers(sellar, boundary):
    '''What is left over once the boxes are cut to their labels, so closing it up makes a
    drawing compact without taking anything out of it -- and it closes both ways at once,
    the floor under a step's row included: that floor is the box, and the gap is outside
    it.'''
    plain = render_xdsm_svg(sellar)
    tight = render_xdsm_svg(sellar, gap=6)

    assert tight.cell[0] == plain.cell[0] - 10
    assert tight.cell[1] == plain.cell[1] - 10
    assert tight.size[0] < plain.size[0] and tight.size[1] < plain.size[1]

    # ... vertically too, wherever a row is taller than its floor
    tall = render_xdsm_svg(boundary, stack_inputs=True)
    assert render_xdsm_svg(boundary, stack_inputs=True, gap=6).cell[1] == tall.cell[1] - 10


def test_the_vertical_gap_can_be_said_on_its_own(sellar):
    '''`gap` is both ways, `gap_y` the rows alone -- which is the half a figure usually
    has to tune, since the horizontal one is spoken for by the width it prints at while
    nothing downstream is watching the height. Saying it moves the pitch and the box, and
    leaves every column exactly where it was.'''
    plain = render_xdsm_svg(sellar, gap=10)
    short = render_xdsm_svg(sellar, gap=10, gap_y=4)

    assert short.cell[0] == plain.cell[0] and short.size[0] == plain.size[0]
    assert short.cell[1] == plain.cell[1] - 6
    assert short.size[1] < plain.size[1]
    # and with nothing said it is the gap, both ways
    assert render_xdsm_svg(sellar, gap=10, gap_y=10).size == plain.size


def test_a_style_replaces_the_default_one(sellar):
    '''By name or as a value, and a stylesheet appended over either -- the paper wants one
    reading and the notebook another, and neither should have to be the drawing's own.'''
    pyxdsm = render_xdsm_svg(sellar, style='pyxdsm')

    assert '.diag{fill:#e4f2e0' in pyxdsm and '.diag{fill:#f8ece7' not in pyxdsm
    assert '.diag{fill:#e4f2e0' in render_xdsm_svg(sellar, style=PYXDSM)
    assert '.diag{fill:#abcdef}' in render_xdsm_svg(sellar, css='.diag{fill:#abcdef}')


def test_a_style_is_drawn_on_its_own_ground(sellar):
    '''Following the system draws the default style's dark reading, so another style
    is refused it rather than quietly drawn half in each.'''
    with pytest.raises(ValueError, match='follow_system'):
        render_xdsm_svg(sellar, style='pyxdsm', follow_system=True)


def test_a_slots_chip_is_the_styles(mdf):
    '''A chip is coloured by the style it is drawn in, a dark one included.'''
    for sty in (TERRACOTTA, DARK):
        svg = render_xdsm_svg(mdf, condition_slots='all', style=sty)
        assert 'class="slot" ' in svg and f'fill="{sty.chip[0]}" fill-opacity' in svg
        assert f'stroke="{sty.chip[1]}" stroke-opacity' in svg


def test_the_cell_is_the_widest_step_column_not_a_boundary_one():
    '''A result spelled long makes the results column the widest; pinning a drawing to
    its own `cell` must not widen its steps to it.'''
    g = graph_of(OrderedMap({N('a'): call(['p'], ['a_result_spelled_at_great_length_indeed'])}))
    plain = render_xdsm_svg(g)
    assert render_xdsm_svg(g, cell=plain.cell).size[0] == plain.size[0]
