"""
`entries(graph)`: what each component is -- a `Run`, a `Cycle`, a `Solve` or an
`Unresolved` -- and `ExecutableGraph`, the graph proved to hold only the first and third, at
every depth. A graph always has entries; a blocking exists where they are executable.
"""

from itertools import permutations

import pytest
from jax.tree_util import GetAttrKey

from cottax.core.frozen import OrderedMap
from cottax.relational import Min, Objective
from cottax.core import ExecutableGraph, Plan
from cottax.consumers.visualization.sequencing import Cycle, Run, Solve, Unresolved, walk, entries
from cottax.pytree.path import NodePath, VarPath
from cottax.core import Function
from cottax.core.maths.plan import Nest
from cottax.core.realisation import ImplementedFunction
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.consumers.visualization.sequencing import ordered_graph, sequence
from cottax.consumers.visualization.xdsm import xdsm_struct
from cottax.consumers.visualization.kinds import block_type
from graphs import (NESTED, NESTED_BESIDE_P, NOTHING_NESTED, UNORDERED_IN_P, graph_of,
                    three_problems)
from cottax.core.maths.nesting import NestingGraph


def V(k): return VarPath((GetAttrKey(k),))
def N(k): return NodePath((GetAttrKey(k),))
def call(reads, owns):
    return ImplementedFunction(Function(tuple(V(r) for r in reads), tuple(V(o) for o in owns)),
                       lambda *a: 0.0)


def drawn(g):
    '''Each block `walk` gives: its kind, the problem answered there, its depth, its nodes.'''
    return [(type(b.entry).__name__, b.problem, b.depth, b.nodes) for b in walk(g)]


EVERY = tuple(N(k) for k in ('p', 'q', 'r', 'a', 'b', 'obj'))
'''`three_problems`' nodes, in binding order.'''


@pytest.fixture
def two_problems():
    '''An optimiser and a root find on one cycle, nothing saying which is outer.'''
    return graph_of(OrderedMap({
        N('opt'): RelationalCondition((), (Objective(V('f'), Min),), (V('z'),), ()),
        N('h')  : RelationalCondition((Relation(V('r'), None, Eq),), (), (V('u'),), ()),
        N('d')  : call(['z', 'u'], ['r']),
        N('obj'): call(['z', 'u'], ['f']),
    }))


@pytest.fixture
def cycle_nobody_solves():
    return graph_of(OrderedMap({
        N('A'): call(['b'], ['a']),
        N('B'): call(['a'], ['b']),
        N('C'): call(['a'], ['c']),
    }))


def test_a_blocking_wraps_a_graph_and_nothing_else():
    assert [f for f in ExecutableGraph.__dataclass_fields__] == ['graph']


def test_a_graph_has_entries_where_a_blocking_refuses(two_problems, cycle_nobody_solves):
    for g, message, kind in ((two_problems, 'several problems', Unresolved),
                             (cycle_nobody_solves, 'no problem', Cycle)):
        with pytest.raises(ValueError, match=message):
            ExecutableGraph(g)
        assert [e.nodes for e in entries(g)] == list(g.graph.components)
        # No single problem is answered at the coupled component's level: nothing to
        # answer, or two with nothing saying which -- an entry that says so, not a refusal.
        coupled, = [e for e in entries(g) if not e.subgraph.graph.is_acyclic]
        assert isinstance(coupled, kind)


def test_an_entry_is_the_components_type(two_problems, cycle_nobody_solves):
    '''Four kinds, derived and never declared, each carrying what only it has: a
    `Solve` its problem and interior, an `Unresolved` the problems nothing ranks.'''
    (unresolved,) = entries(two_problems)
    assert isinstance(unresolved, Unresolved) and unresolved.problems == (N('opt'), N('h'))
    assert not hasattr(unresolved, 'interior')

    a_b, c = entries(cycle_nobody_solves)
    assert isinstance(a_b, Cycle) and a_b.nodes == (N('A'), N('B'))
    assert isinstance(c, Run) and c.nodes == (N('C'),)

    nested = (Plan(two_problems) + Nest(N('h'), N('opt'))).graph
    (solve,) = entries(nested)
    assert isinstance(solve, Solve) and solve.problem == N('opt')
    assert isinstance(solve.interior, NestingGraph)
    assert [type(e).__name__ for e in entries(solve.interior)] == ['Solve', 'Run']
    assert entries(solve.interior)[0].problem == N('h')
    # `walk`: every block a drawing rings, outermost first, with the depth it sits at --
    # not `h`'s body, which holds no problem and is drawn without a ring
    assert [(b.depth, type(b.entry).__name__) for b in walk(nested)] \
        == [(0, 'Solve'), (1, 'Solve'), (1, 'Run')]


def test_a_blocking_is_runs_and_solves_all_the_way_down(two_problems):
    '''That is what the executable rules say: every entry at every depth a `Run` or a
    `Solve`. The graph is the same graph -- the proof adds no data.'''
    nested = (Plan(two_problems) + Nest(N('h'), N('opt'))).graph
    b = ExecutableGraph(nested)
    assert b.graph is nested
    assert all(isinstance(blk.entry, (Run, Solve)) for blk in walk(b.graph))
    # three deep: a solve per level, each holding its interior; the innermost's body runs
    deep = ExecutableGraph(three_problems(*NESTED)).graph
    assert drawn(deep) == [('Solve', N('p'), 0, EVERY),
                           ('Solve', N('q'), 1, EVERY[1:]),
                           ('Run', None, 2, (N('a'),)),
                           ('Solve', N('r'), 2, (N('r'), N('b'), N('obj')))]
    p, q, run, r = walk(deep)
    assert p.held == p.entry.interior and q.held == q.entry.interior
    assert run.held is None and r.held is None


def test_problems_with_nothing_nested_are_one_leaf():
    '''Three problems on one cycle, none nested: one entry naming all three, and one block
    with nothing drawn inside it.'''
    g = three_problems(*NOTHING_NESTED)
    (entry,) = entries(g)
    assert isinstance(entry, Unresolved) and entry.problems == (N('p'), N('q'), N('r'))
    assert drawn(g) == [('Unresolved', None, 0, EVERY)]
    assert walk(g)[0].held is None


def test_problems_unordered_inside_a_solve_are_a_leaf_one_level_in():
    '''`q` and `r` both in `p`: `p` alone is outermost, so its block is a solve, and its
    interior holds two problems nothing orders, a leaf at depth 1.'''
    g = three_problems(*UNORDERED_IN_P)
    assert drawn(g) == [('Solve', N('p'), 0, EVERY), ('Unresolved', None, 1, EVERY[1:])]
    p, inner = walk(g)
    assert p.held == p.entry.interior and inner.held is None
    assert inner.entry.problems == (N('q'), N('r'))


def test_an_unresolved_diagonal_is_its_problems_then_its_body(two_problems):
    '''No problem is answered there, so the diagonal puts its problems first, in binding
    order, then its body: as bound, or in `components_order` where as bound it would draw
    feedback.'''
    assert sequence(two_problems) == (N('opt'), N('h'), N('d'), N('obj'))
    g = three_problems(*NOTHING_NESTED)
    assert sequence(g) == EVERY
    obj_first = graph_of(OrderedMap((n, g[n]) for n in (*EVERY[:3], N('obj'), N('b'), N('a'))))
    assert sequence(obj_first) == EVERY        # `obj` reads `b` and `a`, `b` reads `a`
    # one level in, the same: `p` leads, then the level nothing orders
    assert sequence(three_problems(*UNORDERED_IN_P)) == EVERY


def test_a_proof_spells_where_a_rewrite_stands(two_problems):
    nested = (Plan(two_problems) + Nest(N('h'), N('opt'))).graph
    assert repr(ExecutableGraph(nested)) == 'ExecutableGraph(solve(.opt)[.opt, .h, .d, .obj])'


def test_the_drawing_takes_a_graph_or_a_blocking(two_problems):
    nested = (Plan(two_problems) + Nest(N('h'), N('opt'))).graph
    by_graph    = xdsm_struct(nested, blocks=True)
    by_blocking = xdsm_struct(nested, blocks=ExecutableGraph(nested))
    assert by_graph['blocks'] == by_blocking['blocks']
    # and the un-nested graph is still drawn, its coupled component ringed as unanswered
    struct = xdsm_struct(two_problems, blocks=True)
    assert [(b['ring'], b['depth'], b['type']) for b in struct['blocks']] \
        == [('solve', 0, 'unanswered')]                                  # `obj` is on the cycle too


def test_a_blocks_problem_is_the_entries_answer_said_out_loud(two_problems, cycle_nobody_solves):
    assert [b.problem for b in walk(two_problems)] == [None]
    assert [b.problem for b in walk(cycle_nobody_solves)] == [None, None]
    nested = (Plan(two_problems) + Nest(N('h'), N('opt'))).graph
    assert [(b.problem, b.depth) for b in walk(nested)] == [(N('opt'), 0), (N('h'), 1), (None, 1)]


def test_a_ring_is_drawn_only_where_a_statement_encloses_another(two_problems):
    nested = (Plan(two_problems) + Nest(N('h'), N('opt'))).graph
    outer, h, run = walk(nested)
    assert outer.held == entries(nested)[0].interior
    assert h.held is None and run.held is None       # `h`'s body is runs: no ring inside it


def test_the_diagonal_is_the_entries_in_run_order_problem_first(two_problems):
    nested = (Plan(two_problems) + Nest(N('h'), N('opt'))).graph
    assert [n.spelling for n in sequence(nested)] == ['.opt', '.h', '.d', '.obj']
    assert ordered_graph(nested).nodes == ordered_graph(ExecutableGraph(nested)).nodes == sequence(nested)


# ---------------------------------------------------------------- inside a grey ring
# A problem's parent is the problem it is nested in, its children those nested in it; an
# outermost problem has no parent, and a leaf no children.

def walked(g):
    '''Each block `walk` gives: its kind, the problem answered there, its depth.'''
    return [(type(b.entry).__name__, b.problem, b.depth) for b in walk(g)]


def tree(t, into=(), out=()):
    '''A fixed point `q<t>`, the parent of a minimiser `r<t>`, on their own cycle, as `q` and
    `r` of `three_problems`: `a<t>` also reads `into`, `obj<t>` also owns `out`.'''
    return {N(f'q{t}')  : RelationalCondition((Relation(V(f'yh{t}'), V(f'y1{t}'), Eq),), (),
                                               (V(f'yh{t}'),), ()),
            N(f'r{t}')  : RelationalCondition((), (Objective(V(f'f{t}'), Min),), (V(f'x{t}'),), ()),
            N(f'a{t}')  : call([f'yh{t}', *into], [f'y2{t}']),
            N(f'b{t}')  : call([f'x{t}', f'y2{t}'], [f'y1{t}']),
            N(f'obj{t}'): call([f'x{t}', f'y1{t}', f'y2{t}'], [f'f{t}', *out])}


TREES = ((N('r1'), N('q1')), (N('r2'), N('q2')))
'''Each `tree`'s `r` a child of its `q`.'''


def root_find(condition, unknown):
    '''A problem holding `condition = 0` over `unknown`.'''
    return RelationalCondition((Relation(V(condition), None, Eq),), (), (V(unknown),), ())


def four_problems():
    '''`three_problems` inside a fourth, `s`, which `a` reads too: `p` and `q` children of
    `s`, `r` of `q`.'''
    three = three_problems()
    return graph_of({N('s'): RelationalCondition((), (Objective(V('f'), Min),), (V('w'),), ()),
                     **three.definitions, N('a'): call(['z', 'yh', 'w'], ['y2'])},
                    ((N('p'), N('s')), (N('q'), N('s')), (N('r'), N('q'))))


def joined_by_leaf():
    '''Two trees that only the leaf `l` joins into one cycle.'''
    return graph_of({N('l'): root_find('c', 'v'), **tree(1, into=['v'], out=['o1']),
                     **tree(2, into=['o1'], out=['c'])}, TREES)


def trees_on_one_cycle():
    '''Two trees on one cycle through their outermost problems `q1` and `q2`.'''
    return graph_of({**tree(1, into=['yh2']), **tree(2, into=['yh1'])}, TREES)


def trees_and_leaf():
    '''`trees_on_one_cycle` with the leaf `l` on the cycle too.'''
    return graph_of({N('l'): root_find('c', 'v'), **tree(1, into=['yh2', 'v']),
                     **tree(2, into=['yh1'], out=['c'])}, TREES)


def coupled_below():
    '''Two trees whose `r`s share a cycle even without the `q`s.'''
    return graph_of({**tree(1, into=['o2'], out=['o1']), **tree(2, into=['o1'], out=['o2'])},
                    TREES)


def link_through_leaf():
    '''`i` a child of `o`, on `o`'s cycle only through the leaf `l`.'''
    return graph_of({N('o'): root_find('c', 'u'), N('l'): root_find('d', 'v'),
                     N('i'): root_find('e', 'w'),
                     N('A'): call(['u'], ['d']), N('B'): call(['v', 'w'], ['c', 'e'])},
                    ((N('i'), N('o')),))


ARRANGEMENTS = {
    'nothing nested': lambda: three_problems(*NOTHING_NESTED),
    'unordered in p': lambda: three_problems(*UNORDERED_IN_P),
    'nested': lambda: three_problems(*NESTED),
    'nested beside p': lambda: three_problems(*NESTED_BESIDE_P),
    'four problems': four_problems, 'joined by a leaf': joined_by_leaf,
    'trees on one cycle': trees_on_one_cycle, 'trees and a leaf': trees_and_leaf,
    'coupled below': coupled_below, 'link through a leaf': link_through_leaf,
}
'''Every arrangement of problems the tests below draw, by name.'''


def removed(block):
    '''The problems a grey block removes from the rest of its cycle: drawn on its ring.'''
    return tuple(n for n in block.nodes if n not in block.held)


def test_a_tree_beside_a_leaf_is_ringed_inside_the_grey_ring():
    '''`r` a child of `q`, `p` a leaf: no single problem answers the cycle, but `q` and `r`
    have an order, so with `p` removed the walk rings them inside the grey ring, and `p`
    sits in no inner block.'''
    g = three_problems(*NESTED_BESIDE_P)
    assert drawn(g) == [('Unresolved', None, 0, EVERY),
                        ('Solve', N('q'), 1, EVERY[1:]),
                        ('Run', None, 2, (N('a'),)),
                        ('Solve', N('r'), 2, (N('r'), N('b'), N('obj')))]
    grey = walk(g)[0]
    assert removed(grey) == (N('p'),)
    assert grey.held == grey.entry.held and grey.held.nodes == EVERY[1:]


def test_what_each_kind_of_entry_holds(two_problems, cycle_nobody_solves):
    '''A run and a cycle no problem answers hold nothing; a solve holds its interior; a grey
    cycle with nothing nested in it holds nothing.'''
    for g in (two_problems, three_problems(*NOTHING_NESTED)):
        (entry,) = entries(g)
        assert isinstance(entry, Unresolved) and entry.held is None and walk(g)[0].held is None
    assert {type(e) for e in entries(cycle_nobody_solves)} == {Cycle, Run}
    assert all(e.held is None for e in entries(cycle_nobody_solves))
    (solve,) = entries(three_problems(*NESTED))
    assert isinstance(solve, Solve) and solve.held == solve.interior


def test_a_grey_level_inside_a_solve_rings_what_is_nested_in_it():
    '''`four_problems`: `s` answers the cycle; inside it `p` and `q` have no order, and with
    the leaf `p` removed `q` and `r` are ringed one and two levels further in.'''
    assert drawn(four_problems()) == [('Solve', N('s'), 0, (N('s'), *EVERY)),
                                      ('Unresolved', None, 1, EVERY),
                                      ('Solve', N('q'), 2, EVERY[1:]),
                                      ('Run', None, 3, (N('a'),)),
                                      ('Solve', N('r'), 3, (N('r'), N('b'), N('obj')))]


def test_trees_joined_only_by_a_leaf_are_ringed_apart():
    '''Two trees that only the leaf `l` joins into one cycle: with `l` removed they fall
    apart, each ringed at depth 1, in dependency order.'''
    assert walked(joined_by_leaf()) == [
        ('Unresolved', None, 0),
        ('Solve', N('q1'), 1), ('Run', None, 2), ('Solve', N('r1'), 2),
        ('Solve', N('q2'), 1), ('Run', None, 2), ('Solve', N('r2'), 2)]


def test_trees_on_one_cycle_ring_the_children_of_their_parents():
    '''Two trees on one cycle through their outermost problems `q1` and `q2`, neither a leaf:
    neither `q` can be ringed apart from the other, so both are removed and drawn on the grey
    ring, and their children `r1` and `r2` are ringed inside it. With a leaf `l` on the
    cycle too, `l` is removed first; the trees then share a grey ring inside it, and the
    `r`s are ringed two in.'''
    roots = trees_on_one_cycle()
    assert walked(roots) == [('Unresolved', None, 0),
                             ('Run', None, 1), ('Solve', N('r1'), 1),
                             ('Run', None, 1), ('Solve', N('r2'), 1)]
    assert removed(walk(roots)[0]) == (N('q1'), N('q2'))
    assert [n.spelling for n in sequence(roots)][:2] == ['.q1', '.q2']
    with_l = trees_and_leaf()
    assert walked(with_l) == [('Unresolved', None, 0), ('Unresolved', None, 1),
                              ('Run', None, 2), ('Solve', N('r1'), 2),
                              ('Run', None, 2), ('Solve', N('r2'), 2)]
    assert [removed(b) for b in walk(with_l)[:2]] == [(N('l'),), (N('q1'), N('q2'))]


def test_the_walk_stops_where_nothing_is_nested():
    '''Two trees whose `r`s share a cycle even with both `q`s removed: a grey ring inside the
    grey ring, holding `r1` and `r2`, which have no children, so nothing further.'''
    shared = coupled_below()
    assert walked(shared) == [('Unresolved', None, 0), ('Unresolved', None, 1)]
    assert walk(shared)[1].held is None


def test_a_link_whose_cycle_runs_through_a_removed_leaf_is_dropped_inside():
    '''`i` a child of `o`, on `o`'s cycle only through the leaf `l`: with `l` removed, `o`
    is on no cycle, so the link means nothing in the rest of the cycle and is dropped there.
    `i` keeps its ring, `o` is drawn as a plain node inside the grey ring, and the drawing
    refuses nothing.'''
    g = link_through_leaf()
    grey = walk(g)[0]
    assert isinstance(grey.entry, Unresolved) and grey.entry.problems == (N('o'), N('l'))
    assert removed(grey) == (N('l'),) and not grey.held.within
    assert [(b.problem, b.depth) for b in walk(g) if b.solved] == [(None, 0), (N('i'), 1)]
    assert sequence(g) == (N('l'), N('i'), N('B'), N('o'), N('A'))


def test_a_ring_is_unsettled_where_its_problem_is_part_of_the_grey_rings_decision():
    '''A ring directly inside a grey ring whose problem is one of that ring's outermost
    problems could run as it stands, but its place is still open: it is unsettled. The rings
    of its children, and of children whose parents the grey ring carries, are settled.'''
    unsettled = {name: [b.problem for b in walk(build()) if b.unsettled]
                 for name, build in ARRANGEMENTS.items()}
    assert {name: found for name, found in unsettled.items() if found} == {
        'nested beside p': [N('q')], 'four problems': [N('q')],
        'joined by a leaf': [N('q1'), N('q2')]}
    for name, build in ARRANGEMENTS.items():
        for b in walk(build()):
            assert (block_type(b) == 'unsettled') == b.unsettled


def rebound(g, order):
    '''`g` with its problems dealt into the binding positions they hold, in `order`.'''
    dealt, problems = iter(order), set(g.problems)
    return graph_of({(m := next(dealt) if n in problems else n): g[m] for n in g.nodes},
                    tuple(g.within.items()))


def boxes(g):
    '''What `walk` draws, apart from order: each box's nodes, depth and type.'''
    return {(frozenset(b.nodes), b.depth, block_type(b)) for b in walk(g)}


@pytest.mark.parametrize('name', ARRANGEMENTS)
def test_the_boxes_do_not_depend_on_which_problem_is_bound_first(name):
    '''A grey ring defers a nesting decision, so the drawing makes none of its own: dealing
    the problems into their binding positions in every order draws the same boxes, at the
    same depths, of the same types. Only the diagonal changes: the problems on each grey ring
    lead its stretch in binding order, so whichever is bound first comes first.'''
    g = ARRANGEMENTS[name]()
    for order in permutations(g.problems):
        other = rebound(g, order)
        assert boxes(other) == boxes(g)
        diagonal = sequence(other)
        for b in walk(other):
            if isinstance(b.entry, Unresolved):
                on = removed(b) if b.entry.held is not None else b.entry.problems
                assert tuple(n for n in diagonal if n in b.nodes)[:len(on)] == on
