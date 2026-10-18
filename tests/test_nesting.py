"""
Nesting: which problem's iteration answers which.

The claim under test is that recording it on the graph -- as `within`, a partial function
inner -> outer -- makes every consumer read one answer instead of being told, and makes
the ways a rewrite could quietly void it refusals rather than surprises.
"""

import pytest
from jax.tree_util import GetAttrKey as G

from cottax.core.maths.executable import ExecutableGraph
from cottax.relational import Min, Objective
from cottax.core.driven import Assign, DriverContract
from cottax.core import Function
from cottax.core.maths.plan import Delete, Insert, Nest, Plan, Unnest
from cottax.core.realisation import ImplementedFunction
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.core.maths.plan import Cut, Determine, Undetermine
from cottax.core import Combine
from cottax.pytree.path import NodePath, VarPath
from cottax.core.frozen import OrderedMap
import purposes
from graphs import graph_of
from cottax.core.maths.nesting import NestingGraph

HAT, PROBLEM = purposes.Copy(), purposes.Solve()


def optimise(objective, unknowns) -> RelationalCondition:
    '''Minimise one variable over some unknowns, and no relation.'''
    return RelationalCondition((), (Objective(objective, Min),), unknowns, ())


def fixed_point(unknown, condition) -> RelationalCondition:
    '''`unknown = condition`: the problem a cut closes a cycle with.'''
    return RelationalCondition((Relation(unknown, condition, Eq),), (), (unknown,), ())


def requirement(condition) -> RelationalCondition:
    '''`condition = 0`, determining nothing -- so it owns nothing and lies on no cycle.'''
    return RelationalCondition((Relation(condition, None, Eq),), (), (), ())


def root_find(conditions, unknowns) -> RelationalCondition:
    '''Every condition vanishes, over as many unknowns.'''
    return RelationalCondition(tuple(Relation(c, None, Eq) for c in conditions), (), unknowns, ())


def v(*names: str) -> VarPath:
    return VarPath(tuple(G(x) for x in names))


def n(*names: str) -> NodePath:
    return NodePath(tuple(G(x) for x in names))


def call(reads, owns):
    return ImplementedFunction(Function(tuple(v(r) for r in reads), tuple(v(o) for o in owns)),
                       lambda *a: tuple(a[:1]) * len(owns))


@pytest.fixture
def mdf() -> NestingGraph:
    """An optimiser over `.z` and a fixed point on `^copy.y`, on one cycle."""
    return graph_of(OrderedMap({
        n('opt'): optimise(v('f'), (v('z'),)),
        n('mda'): fixed_point(v('hy'), v('y')),
        n('d')  : call(['z', 'hy'], ['y']),
        n('obj'): call(['y'],       ['f']),
    }))


@pytest.fixture
def nested(mdf) -> NestingGraph:
    return (Plan(mdf) + Nest(n('mda'), n('opt'))).graph


# ---------------------------------------------------------------- the field

def test_a_redefine_elsewhere_does_not_void_the_nesting(nested):
    # `Redefine.apply` rescans when an output moved, and a rescan must not be what drops
    # the graph's second component: an unrelated `Determine` once erased `within` whole.
    unrelated = (Plan(nested)
                 + Insert(OrderedMap({n('src'): call([], ['w']),
                                    n('req'): root_find((v('w'),), ())}))
                 + Determine(n('req'), (v('q'),))).graph
    assert dict(unrelated.within) == {n('mda'): n('opt')}


def test_undetermining_one_end_is_refused_rather_than_dropped(nested):
    # The same answer `Delete` gives: refuse, and name `Unnest` -- never quietly void
    # what an earlier op recorded.
    with pytest.raises(ValueError, match='Unnest'):
        Plan(nested) + Undetermine(n('mda'))


def test_the_nesting_is_a_forest_on_the_value_not_only_on_the_op(mdf):
    '''A loop is refused wherever it arrives: built from nothing, or closed by a `Nest`
    -- at length one (`x` inside itself) and at any length. The rule is the graph's
    (`check_no_nesting_loops`, `check_nesting`), so an op never has to say it.'''
    with pytest.raises(ValueError, match='closes a loop'):
        graph_of(mdf.definitions, OrderedMap({n('mda'): n('opt'), n('opt'): n('mda')}))
    with pytest.raises(ValueError, match='inside itself'):
        graph_of(mdf.definitions, OrderedMap({n('opt'): n('opt')}))
    # A loop the walk does not start on, at length three, and one reached only by a
    # tail: both terminate and both are reported from where the loop closes.
    three = OrderedMap({n('d'): n('obj'), n('obj'): n('mda'), n('mda'): n('opt'), n('opt'): n('obj')})
    with pytest.raises(ValueError, match='closes a loop'):
        graph_of(mdf.definitions, three)
    with pytest.raises(ValueError, match='inside itself'):
        Plan(mdf) + Nest(n('opt'), n('opt'))
    with pytest.raises(ValueError, match='closes a loop'):
        Plan(mdf) + Nest(n('mda'), n('opt')) + Nest(n('opt'), n('mda'))


def test_a_nested_pair_shares_a_cycle_at_its_own_level():
    '''`x` inside `mda` inside `opt`, with `x`'s only way back to `mda` through what
    `opt` owns: in the whole graph the pair shares a cycle, in `opt`'s interior -- one
    iterate of `mda`, `z` held -- it does not, so nothing `mda` iterates re-runs `x`
    with feedback. Refused where the nesting lives, and `levels` is total because of
    it: every interior it builds is well-formed.'''
    g = graph_of(OrderedMap({
        n('opt'): optimise(v('f'), (v('z'),)),
        n('mda'): fixed_point(v('hy'), v('y')),
        n('d')  : call(['hy', 'w2'], ['y']),             # x feeds mda ...
        n('x')  : fixed_point(v('hw'), v('w2')),
        n('e')  : call(['hw', 'w'],  ['w2']),
        n('k')  : call(['z'],        ['w']),             # ... and mda reaches x only through opt's z
        n('obj'): call(['y'],        ['f']),
    }))
    assert n('x') in g.graph.descendants([n('mda')]) and n('mda') in g.graph.descendants([n('x')])   # one cycle, in the graph
    outer = Plan(g) + Nest(n('mda'), n('opt'))
    with pytest.raises(ValueError, match='not on one cycle'):
        outer + Nest(n('x'), n('mda'))
    # Built from nothing, the same: the base case walks the same tree.
    with pytest.raises(ValueError, match='not on one cycle'):
        graph_of(g.definitions, OrderedMap({n('mda'): n('opt'), n('x'): n('mda')}))
    # Nested in `opt` directly, `x` is at opt's level, where the cycle is: fine, and the
    # tree has two levels under opt's block.
    both = (outer + Nest(n('x'), n('opt'))).graph
    interior = both.interior(n('opt'))
    assert set(interior.outermost_problems) == {n('mda'), n('x')}     # two, side by side


def test_a_pair_whose_inner_is_outside_its_outers_component_is_refused(mdf):
    '''`p` in its own component, nothing to or from the rest; `p` inside `mda` inside
    `opt`. The level-by-level form of the rule descended into `opt`'s interior keeping
    only the pairs with both ends inside, so this pair was dropped on the way down and
    the nesting accepted -- while `Nest(p, mda)` before `Nest(mda, opt)` was refused.
    Stated per pair it is refused either way round.'''
    g = graph_of(OrderedMap({**dict(mdf.definitions.items()),
                        n('p'): root_find((v('s'),), (v('r'),)),
                        n('e'): call(['r'], ['s'])}))
    assert n('p') not in g.graph.descendants([n('mda')]) and n('mda') not in g.graph.descendants([n('p')])
    with pytest.raises(ValueError, match='not on one cycle'):
        Plan(g) + Nest(n('mda'), n('opt')) + Nest(n('p'), n('mda'))
    with pytest.raises(ValueError, match='not on one cycle'):
        Plan(g) + Nest(n('p'), n('mda')) + Nest(n('mda'), n('opt'))
    with pytest.raises(ValueError, match='not on one cycle'):
        graph_of(g.definitions, OrderedMap({n('mda'): n('opt'), n('p'): n('mda')}))


def test_nesting_changes_equality_and_the_hash(mdf, nested):
    assert nested != mdf and hash(nested) != hash(mdf)
    assert (Plan(nested) + Unnest(n('mda'))).graph == mdf


def test_roots_is_what_a_block_answers(nested):
    assert nested.outermost_problems == (n('opt'),)
    assert nested.interior(n('opt')).outermost_problems == (n('mda'),)   # the outer gone, the inner leads


def test_a_statement_that_determines_nothing_cannot_be_an_end(mdf):
    stated = root_find((v('y'),), ())
    g = graph_of(OrderedMap({**dict(mdf.definitions.items()), n('c'): stated}))
    with pytest.raises(ValueError, match='determines nothing'):
        Plan(g) + Nest(n('c'), n('opt'))


# ---------------------------------------------------------------- the blocking reads it


def test_two_statements_with_nothing_said_have_no_blocking_but_still_draw(mdf):
    # A blocking is executable by construction, so a graph halfway through a rewrite has
    # none -- and is drawn regardless, the drawing walking `entries(graph)`.
    from cottax.consumers.visualization.xdsm import xdsm_struct
    with pytest.raises(ValueError, match='several problems'):
        ExecutableGraph(mdf)
    assert [(b['ring'], b['depth']) for b in xdsm_struct(mdf, blocks=True)['blocks']] == [('solve', 0)]


def test_severing_a_nested_pair_is_refused(nested):
    '''A `Cut` that pulls the two ends onto different components is refused like a
    `Delete` of an end: the pair's cycle is an invariant of `within`, not a query, so
    there is no state in which the graph holds a nesting that means nothing. `Unnest`
    first, then cut.'''
    with pytest.raises(ValueError, match='not on one cycle'):
        Plan(nested) + Cut(v('f'), [n('opt')], HAT)
    unnested = Plan(nested) + Unnest(n('mda')) + Cut(v('f'), [n('opt')], HAT)
    assert dict(unnested.graph.within) == {}
    # And from nothing: the base case is the same rule.
    with pytest.raises(ValueError, match='not on one cycle'):
        graph_of(unnested.graph.definitions, OrderedMap({n('mda'): n('opt')}))
    with pytest.raises(ValueError, match='determines nothing'):
        graph_of(nested.definitions, OrderedMap({n('d'): n('opt')}))


# ---------------------------------------------------------------- rewrites over a pair

def test_deleting_one_end_is_refused(nested):
    with pytest.raises(ValueError, match='lose one end'):
        Plan(nested) + Delete((n('opt'),))
    # Both ends together is fine: nothing is left dangling.
    assert (Plan(nested) + Delete((n('opt'), n('mda'),))).graph.within == OrderedMap()


def test_restricting_to_what_contributes_keeps_both_ends_or_neither(nested):
    # Both ends are on one cycle, so each is an ancestor of the other: a restriction to
    # what a variable needs keeps both or drops both, and the state
    # `check_delete_keeps_nesting_whole` would refuse is unreachable this way. (It was
    # reachable after a `Cut` split the pair; that `Cut` is now the refusal.)
    contributing = lambda g, var: g.subgraph(g.graph.ancestors((g.graph.owners[var],)))
    assert dict(contributing(nested, v('f')).within) == {n('mda'): n('opt')}
    beside = (Plan(nested) + Insert(OrderedMap({n('src'): call([], ['w'])}))).graph
    assert dict(contributing(beside, v('w')).within) == {}


def test_a_subgraph_is_the_induced_one_and_a_delete_may_not_strand(nested):
    # A restriction restricts every relation: a pair with an end not kept goes, which
    # is what promotes an interior's children. That a *Delete* may not do that is the
    # op's check (`Stranded`), since a nesting is a decision the plan recorded.
    assert dict(nested.subgraph([n('mda'), n('d')]).within) == {}
    assert dict(nested.subgraph(nested.nodes).within) == {n('mda'): n('opt')}
    assert dict(nested.interior(n('opt')).within) == {}
    # Both ends kept and the cycle between them not: the pair survives the restriction
    # and stands on nothing, the one invariant a restriction can break. Refused, not
    # dropped -- the graph does not decide a stated nesting no longer applies.
    with pytest.raises(ValueError, match='not on one cycle'):
        nested.subgraph([n('mda'), n('opt')])


def test_combining_both_ends_folds_the_nesting_away(nested):
    # As deleting both ends together is fine: the pair goes with the problems it
    # related, and nothing is left dangling. One end alone is `check_delete_keeps_nesting_whole`'s refusal.
    op   = Combine(PROBLEM(n('both')), (n('opt'), n('mda')), RelationalCondition)
    both = (Plan(nested) + op).graph
    assert dict(both.within) == {} and op.at in both


def test_a_fold_at_the_outer_name_keeps_the_nesting(nested):
    # What the binding buys, and the only difference between the two spellings of a fold:
    # a requirement folded into `.opt` lands on `Replace`, so the pair `.opt` is the outer
    # end of survives. The same join bound at a name of its own deletes `.opt` and strands
    # the pair -- `Unnest` first, or fold in place.
    g    = (Plan(nested) + Insert(OrderedMap({n('req'): requirement(v('f'))}))).graph
    kept = (Plan(g) + Combine(n('opt'), (n('opt'), n('req')), RelationalCondition)).graph

    assert dict(kept.within) == {n('mda'): n('opt')}
    assert list(kept.nodes) == [x for x in g.nodes if x != n('req')]
    with pytest.raises(ValueError, match='lose one end'):
        Plan(g) + Combine(PROBLEM(n('both')), (n('opt'), n('req')), RelationalCondition)


def test_driving_and_nesting_are_independent_decisions(nested):
    class Solver(DriverContract):
        accepts = staticmethod(lambda node: True)
        takes   = ()
        reports = ()

    a = (Plan(nested) + Assign(n('mda'), Solver(), (), ())).graph
    b = (Plan(mdf_of(nested)) + Assign(n('mda'), Solver(), (), ())
         + Nest(n('mda'), n('opt'))).graph
    assert a == b
    assert dict(a.within) == {n('mda'): n('opt')}


def mdf_of(nested: NestingGraph) -> NestingGraph:
    return (Plan(nested) + Unnest(n('mda'))).graph


# ---------------------------------------------------------------- several

def test_two_siblings_share_one_outer(mdf):
    g = graph_of(OrderedMap({
        n('opt') : optimise(v('f'), (v('z'),)),
        n('mda') : fixed_point(v('hy'), v('y')),
        n('mda2'): fixed_point(v('hw'), v('w')),
        n('d')   : call(['z', 'hy'],  ['y']),
        n('e')   : call(['z', 'hw'],  ['w']),
        n('obj') : call(['y', 'w'],   ['f']),
    }))
    g = (Plan(g) + Nest(n('mda'), n('opt')) + Nest(n('mda2'), n('opt'))).graph
    assert g.outermost_problems == (n('opt'),)
    assert set(g.within) == {n('mda'), n('mda2')}


def test_three_levels_are_two_pairs_in_either_order():
    '''Deeper nesting is more `Nest`s. A pair is checked at the level its outer lives
    at, so the inner pair may be stated before or after the outer one.'''
    g = graph_of(OrderedMap({
        n('opt') : optimise(v('f'), (v('z'),)),
        n('mda') : fixed_point(v('hy'), v('y')),
        n('mda2'): fixed_point(v('hw'), v('w')),
        n('d')   : call(['z', 'hy', 'hw'], ['y']),
        n('e')   : call(['z', 'hy', 'hw'], ['w']),
        n('obj') : call(['y', 'w'],       ['f']),
    }))
    up   = (Plan(g) + Nest(n('mda2'), n('mda')) + Nest(n('mda'), n('opt'))).graph
    down = (Plan(g) + Nest(n('mda'), n('opt')) + Nest(n('mda2'), n('mda'))).graph
    assert dict(up.within) == dict(down.within) == {n('mda2'): n('mda'), n('mda'): n('opt')}
    assert up.outermost_problems == (n('opt'),)
    assert up.interior(n('opt')).outermost_problems == (n('mda'),)
    assert up.interior(n('opt')).interior(n('mda')).outermost_problems == (n('mda2'),)


# ---------------------------------------------------------------- interior

def test_interior_is_the_block_without_the_problem_and_is_valid(mdf, nested):
    '''Any problem, at any depth: a nested one is asked with the problems enclosing it
    taken out; a function node iterates nothing; a name not here is not here. What
    comes back is a valid graph, the nesting rule re-asked of it as `subgraph` does.'''
    with pytest.raises(ValueError, match='not a problem'):
        nested.interior(n('d'))
    with pytest.raises(KeyError, match='not a node'):
        nested.interior(n('nope'))
    interior = nested.interior(n('opt'))
    assert n('opt') not in interior and dict(interior.within) == {}
    assert graph_of(interior.definitions, interior.within) == interior  # rebuilt with every check, the same
    # The nested one, asked of the whole graph, is what asking it of opt's interior gives.
    assert nested.interior(n('mda')) == interior.interior(n('mda'))
    assert set(nested.cycle_of(n('mda'))) == {n('mda'), n('d')}


def test_interior_refuses_to_sever_a_siblings_nesting():
    '''The one case the graph's own invariant does not vouch for: an outermost problem
    that encloses nothing, beside a sibling that does, with the sibling pair's cycle
    running through it. Taking it out would leave a pair on no cycle -- refused, not
    carried.'''
    g = graph_of(OrderedMap({
        n('o')  : fixed_point(v('hi'), v('i')),
        n('i')  : fixed_point(v('hp'), v('p')),
        n('p')  : fixed_point(v('ho'), v('o')),
        n('fo') : call(['hi'], ['o']),      # i -> o -> p -> i, and only that way round
        n('fp') : call(['ho'], ['p']),
        n('fi') : call(['hp'], ['i']),
    }))
    g = (Plan(g) + Nest(n('i'), n('o'))).graph
    assert set(g.outermost_problems) == {n('o'), n('p')}
    g.interior(n('o'))                              # the outer: vouched for
    with pytest.raises(ValueError, match='not on one cycle'):
        g.interior(n('p'))                          # the sibling: severs (i, o)


def test_contract_drops_a_pair_with_an_end_in_a_block_and_keeps_the_rest(nested):
    # A box is no problem, so a pair touching one goes. A pair outside keeps its
    # cycle, since a contraction only adds paths.
    boxed = nested.contract(((n('mda'), n('d')),))
    assert dict(boxed.within) == {}
    assert boxed.nodes == (n('opt'), n('mda'), n('obj'))
    assert boxed[n('mda')] == Function((v('z'),), (v('hy'), v('y')))
    kept = nested.contract(((n('d'), n('obj')),))
    assert dict(kept.within) == {n('mda'): n('opt')}
    from cottax.core.maths.nesting import NestingGraph
    assert isinstance(nested.contract(()), NestingGraph)
