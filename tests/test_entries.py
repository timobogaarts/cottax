"""
`entries(graph)`: what each component is -- a `Run`, a `Cycle`, a `Solve` or an
`Unresolved` -- and `ExecutableGraph`, the graph proved to hold only the first and third, at
every depth. A graph always has entries; a blocking exists where they are executable.
"""

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
from graphs import graph_of
from cottax.core.maths.nesting import NestingGraph


def V(k): return VarPath((GetAttrKey(k),))
def N(k): return NodePath((GetAttrKey(k),))
def call(reads, owns):
    return ImplementedFunction(Function(tuple(V(r) for r in reads), tuple(V(o) for o in owns)),
                       lambda *a: 0.0)


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
    # `walk`: every block a drawing rings, outermost first, with the problems it sits
    # inside -- not `h`'s body, which holds no problem and is drawn without a ring
    assert [(b.within, type(b.entry).__name__) for b in walk(nested)] \
        == [((), 'Solve'), ((N('opt'),), 'Solve'), ((N('opt'),), 'Run')]


def test_a_blocking_is_runs_and_solves_all_the_way_down(two_problems):
    '''That is what the executable rules say: every entry at every depth a `Run` or a
    `Solve`. The graph is the same graph -- the proof adds no data.'''
    nested = (Plan(two_problems) + Nest(N('h'), N('opt'))).graph
    b = ExecutableGraph(nested)
    assert b.graph is nested
    assert all(isinstance(blk.entry, (Run, Solve)) for blk in walk(b.graph))


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
