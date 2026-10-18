'''
`relational/rewrites.py`: the rewrites that write or join a relation, and what each one
does to the graph it is applied to. `Cut` itself is `core`'s, and `test_core_ops.py`'s.

Names are built by hand from key entries, as in `test_algebra.py` and `test_xdsm.py`.
The minted ones are asked of the op (`op.copies`, `op.at`) and never spelled here:
which namespace a rewrite opened is a field of the rewrite.
'''

import pytest
from jax.tree_util import GetAttrKey

from cottax.core.frozen import OrderedMap
from cottax.relational import Min, Objective
from cottax.core.maths.plan import Cut, Insert, Plan
from cottax.core import ConditionalNode, Function, body_of, problems
from cottax.core.driven import Assign, Driven, DriverContract, is_driven
from cottax.core.driven import Start
from cottax.assemblies.jax import absolute

assign = absolute.assign
from cottax.core.realisation import ImplementedDriven, ImplementedFunction
from cottax.relational import (
    Eq, Le, Relation, RelationalCondition, is_fixed_point, is_optimise, is_root_find,
)
from cottax.core import Combine
from cottax.relational.rewrites import FixedPointCut
from cottax.pytree.path import NodePath, VarPath
import purposes
from graphs import graph_of


HAT, PROBLEM, START = purposes.Copy(), purposes.Solve(), purposes.Start()


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


X, Z, M, Y1, Y2, F, C, R = (V(s) for s in 'x z M y1 y2 f c r'.split())


def call(reads, owns):
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), lambda *a: None)


def source(var):
    return call([], [var])


def root_find(conditions, unknowns) -> RelationalCondition:
    return RelationalCondition(tuple(Relation(c, None, Eq) for c in conditions),
                               (), tuple(unknowns), ())


def fixed_point(conditions, unknowns) -> RelationalCondition:
    return RelationalCondition(
        tuple(Relation(u, c, Eq) for u, c in zip(unknowns, conditions)), (), tuple(unknowns), ())


def optimise(objective, unknowns, equalities=(), inequalities=()) -> RelationalCondition:
    return RelationalCondition(
        tuple(Relation(h, None, Eq) for h in equalities)
        + tuple(Relation(g, None, Le) for g in inequalities),
        (Objective(objective, Min),), tuple(unknowns), ())


def names(graph):
    '''The graph's node names as they are written, sorted -- `written`'s own order.'''
    return sorted(n.spelling for n in graph.nodes)       # a set of names, compared as such


@pytest.fixture
def sellar():
    '''`d1` and `d2` are coupled; `obj` and `con` hang off them.'''
    return graph_of(OrderedMap({
        N('z')  : source(Z),
        N('d1') : call([Z, Y2],     [Y1]),
        N('d2') : call([Z, Y1],     [Y2]),
        N('obj'): call([Z, Y1, Y2], [F]),
        N('con'): call([Y1],        [C]),
    }))


@pytest.fixture
def looped():
    '''`d2` is the *only* reader of `y1`, so cutting it leaves `y1` unread.'''
    return graph_of(OrderedMap({
        N('d1') : call([Y2], [Y1]),
        N('d2') : call([Y1], [Y2]),
        N('obj'): call([Y2], [F]),
    }))


# ---------------------------------------------------------------- the cut, closed

def opening(graph, var):
    '''`Cut(var, ..., HAT)` aimed at whatever closes the cycle -- what a caller writes.
    The cut's own tests are `test_core_ops.py`'s; here it is what a closure is made of.'''
    return Cut(var, graph.graph.closing_readers(var), HAT)


def closure(graph, var, place=None):
    '''What a caller writes: the cut and the label the closure binds at, as one op. The
    label is the crossing, spelled by whoever knows both namespaces -- here, the test.'''
    return FixedPointCut((opening(graph, var),),
                         PROBLEM(NodePath(var.segments) if place is None else place))


def test_a_closure_inherits_the_cut_preconditions(sellar, looped):
    with pytest.raises(ValueError, match='no readers'):
        FixedPointCut((Cut(Y1, (), HAT),), PROBLEM(N('y1')))
    with pytest.raises(KeyError, match='nothing to cut'):
        closure(sellar, Y1)(graph_of(OrderedMap({N('d2'): call([Y1], [Y2])})))
    with pytest.raises(KeyError, match='not a node of this graph'):
        FixedPointCut((Cut(Y1, (N('nope'),), HAT),), PROBLEM(N('y1')))(looped)


def test_a_fixed_point_cut_is_one_node_and_stays_cyclic(looped):
    '''The loop is re-closed through the problem, and the relation's right side *is*
    the next iterate, so there is nothing to subtract.'''
    o = closure(looped, Y1)
    g = o(looped)
    assert names(g) == ['.d1', '.d2', '.obj', '^solve.y1']
    assert not g.graph.is_acyclic
    assert problems(g.definitions) == (o.at,)
    assert is_fixed_point(g[o.at])
    assert g[o.at].relations == (Relation(o.copies[0], Y1, Eq),)   # `^copy.y1 = y1`
    assert g[o.at].owns == o.copies and g[o.at].reads == o.vars
    assert body_of(g).graph.is_acyclic


def test_the_closure_is_exactly_its_parts(looped):
    '''Spelled as a plan, since an op is what a plan records: the closure is one entry
    where the parts are two.'''
    cut = opening(looped, Y1)
    fp  = FixedPointCut((cut,), PROBLEM(NodePath(cut.var.segments)))
    assert fp(looped) == (looped + cut + Insert(OrderedMap({fp.at: fp.definition}))).graph


def test_the_minted_names_follow_their_own_roots(looped):
    '''The copy follows the cut's own `copy_of`; the problem is named from the label the
    closure was handed, so the cut namespace stays plumbing and renames nothing.'''
    cut = Cut(Y1, (N('d2'),), purposes.Guess())
    fp  = FixedPointCut((cut,), PROBLEM(NodePath(cut.var.segments)))
    assert cut.copy.spelling == '^guess.y1'
    assert fp.at.spelling == '^solve.y1'
    assert fp(looped).graph.owners[cut.copy] == fp.at


def test_closure_repr(looped):
    assert repr(closure(looped, Y1)) == 'fixed_point_cut(.y1->^copy.y1 @ ^solve.y1)'


# ---------------------------------------------------------------- one closure, several cuts

@pytest.fixture
def two_cycles():
    '''`d1`/`d2` coupled through `y1`, and `d3` closing a second cycle through `z`.'''
    return graph_of(OrderedMap({
        N('d1') : call([Z, Y2], [Y1]),
        N('d2') : call([Z, Y1], [Y2]),
        N('d3') : call([Y2],    [Z]),
        N('obj'): call([Y1],    [F]),
    }))


def test_a_closure_over_several_cuts_is_one_problem(two_cycles):
    '''One problem owning every copy and reading every original, so both are converged
    by one driver in one iteration -- which is the whole difference from two closures.'''
    op = FixedPointCut((Cut(Y2, (N('d1'),), HAT), Cut(Z, (N('d1'), N('d2'),), HAT)),
                       PROBLEM(N('mda')))
    g  = op(two_cycles)

    assert op.at.spelling == '^solve.mda'
    assert [v.spelling for v in op.copies] == ['^copy.y2', '^copy.z']
    assert g[op.at].owns  == op.copies
    assert g[op.at].reads == op.vars == (Y2, Z)
    assert g[op.at].relations == tuple(Relation(m, v, Eq)
                                            for m, v in zip(op.copies, op.vars))


def test_that_is_the_graph_two_cuts_and_a_combine_reach(two_cycles):
    '''Same graph, said in one op instead of three -- so the pairing is what is recorded
    rather than something a reader has to spot across three entries in `Plan.ops`.'''
    cuts = (Cut(Y2, (N('d1'),), HAT), Cut(Z, (N('d1'), N('d2'),), HAT))
    one  = FixedPointCut(cuts, PROBLEM(N('mda')))(two_cycles)

    a = FixedPointCut((cuts[0],), PROBLEM(NodePath(cuts[0].var.segments)))
    b = FixedPointCut((cuts[1],), PROBLEM(NodePath(cuts[1].var.segments)))
    comb  = Combine(PROBLEM(N('mda')), (a.at, b.at), RelationalCondition)
    long_ = (two_cycles + a + b + comb).graph

    assert comb.at == FixedPointCut(cuts, PROBLEM(N('mda'))).at
    assert set(one.nodes) == set(long_.nodes)
    assert one[comb.at] == long_[comb.at]


def test_a_closure_is_always_told_where_it_binds():
    '''A variable never names a node, so where the closure binds is a decision the op
    records rather than one it remakes -- the convenient default (the one cut variable's
    own place) lives on the surfaces that know both namespaces.'''
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'at'"):
        FixedPointCut((Cut(Y2, (N('d1'),), HAT),))


def test_one_variable_may_not_be_cut_twice_in_one_closure():
    with pytest.raises(ValueError, match='cut twice in one closure'):
        FixedPointCut((Cut(Y2, (N('d1'),), HAT), Cut(Y2, (N('d2'),), HAT)),
                      PROBLEM(N('mda')))


# ---------------------------------------------------------------- Combine

@pytest.fixture
def two_problems():
    '''Two independent root-finds, so two components.'''
    return graph_of(OrderedMap({
        N('fa'): call([M],  [Y1]),
        N('ha'): root_find((Y1,), (M,)),
        N('fb'): call([Z],  [Y2]),
        N('hb'): root_find((Y2,), (Z,)),
    }))


def test_combine_is_one_problem_over_every_unknown(two_problems):
    op = Combine(PROBLEM(N('mda')), (N('ha'), N('hb'),), RelationalCondition)
    g  = op(two_problems)

    assert names(g) == ['.fa', '.fb', '^solve.mda']
    assert g[op.at].owns  == (M, Z)                 # concatenated, in `problems` order
    assert g[op.at].reads == (Y1, Y2)
    assert is_root_find(g[op.at])


def test_combine_is_the_join_written_by_hand(two_problems):
    op = Combine(PROBLEM(N('mda')), (N('ha'), N('hb'),), RelationalCondition)
    assert op.definition(two_problems) == two_problems[N('ha')] + two_problems[N('hb')]
    assert op(two_problems)[op.at] == two_problems[N('ha')] + two_problems[N('hb')]


def test_combining_makes_one_component(two_problems):
    '''The whole point: the blocking follows, so nothing has to state it.'''
    assert len(two_problems.graph.cycles) == 2
    g = Combine(PROBLEM(N('mda')), (N('ha'), N('hb'),), RelationalCondition)(two_problems)
    assert [[n.spelling for n in c] for c in g.graph.cycles] == [['.fa', '.fb', '^solve.mda']]


def test_combining_pulls_in_what_lies_between():
    '''Strong connectivity does the closure a partition value computed by hand.'''
    g = graph_of(OrderedMap({
        N('fa')  : call([M],  [Y1]),
        N('ha')  : root_find((Y1,), (M,)),
        N('glue'): call([M],  [C]),
        N('fb')  : call([C, Z], [Y2]),
        N('hb')  : root_find((Y2,), (Z,)),
    }))
    h = Combine(PROBLEM(N('mda')), (N('ha'), N('hb'),), RelationalCondition)(g)
    assert sorted(n.spelling for n in h.graph.cycles[0]) == ['.fa', '.fb', '.glue', '^solve.mda']


@pytest.fixture
def mdf():
    '''An optimisation outside a root-find: `x` designed, `u` solved, `f` minimised.'''
    return graph_of(OrderedMap({
        N('an')  : call([X, M],  [Y1, F, C]),
        N('h')   : root_find((Y1,), (M,)),
        N('opt') : optimise(F, (X,), inequalities=(C,)),
    }))


def test_an_optimise_absorbs_a_root_find(mdf):
    '''SAND: the unknowns join the design, the vanishing relations join the constraints.'''
    op = Combine(PROBLEM(N('sand')), (N('opt'), N('h'),), RelationalCondition)
    g  = op(mdf)
    joined = g[op.at]

    assert is_optimise(joined)
    assert joined.objectives  == (Objective(F, Min),)
    assert joined.owns        == (X, M)          # design, then the absorbed unknowns
    assert joined.conditions  == (F, C, Y1)      # objective, then the relations as written
    assert joined.reads       == joined.conditions          # on a bare condition node, every read is one
    assert [r.op for r in joined.relations] == [Le, Eq]


def test_two_fixed_points_stay_a_fixed_point():
    g = graph_of(OrderedMap({
        N('fa'): call([M],  [Y1]),
        N('ha'): fixed_point((Y1,), (M,)),
        N('fb'): call([Z],  [Y2]),
        N('hb'): fixed_point((Y2,), (Z,)),
    }))
    op = Combine(PROBLEM(N('mda')), (N('ha'), N('hb'),), RelationalCondition)
    assert is_fixed_point(op(g)[op.at])


def test_a_fold_refuses_a_node_that_is_not_the_kind_it_declared(two_problems):
    '''The kind is asked where the graph is read, so the refusal names which node and
    what it found -- against `+`'s own `TypeError`, which names only two types.'''
    with pytest.raises(TypeError, match='well-defined over one kind only') as e:
        Combine(PROBLEM(N('mda')), (N('ha'), N('fb'),), RelationalCondition)(two_problems)
    assert '.fb' in str(e.value) and 'Implemented' in str(e.value)
    with pytest.raises(KeyError, match='not a node of this graph'):
        Combine(PROBLEM(N('mda')), (N('ha'), N('nope'),), RelationalCondition)(two_problems)


def test_a_fold_refuses_a_driven_problem(two_problems):
    '''A `Driven` is not a `RelationalCondition`, so the fold has no `+` to reach: the
    algorithm answering each would be discarded, and which should answer the join cannot
    be read off the two.'''
    driven = (Plan(two_problems) + assign(N('ha'), Newton())).graph
    with pytest.raises(TypeError, match='well-defined over one kind only') as e:
        Combine(PROBLEM(N('mda')), (N('ha'), N('hb'),), RelationalCondition)(driven)
    assert 'Driven' in str(e.value)


def test_a_fold_declares_what_it_folds(two_problems):
    '''The kind is the op's intent, so it is a field and not something read back off the
    graph: a replay against a changed model refuses rather than folding what it finds.
    A kind with no `+` at all is refused when the op is built, before any graph.'''
    with pytest.raises(TypeError, match='defines no `\\+`'):
        Combine(PROBLEM(N('mda')), (N('ha'), N('hb'),), ImplementedDriven)
    # A level defines no `+` either -- it leaves one to its finals, so it is refused too
    # and the kind names the final that is actually folded.
    with pytest.raises(TypeError, match='defines no `\\+`'):
        Combine(PROBLEM(N('mda')), (N('ha'), N('hb'),), ConditionalNode)


def test_combine_reads_written():
    assert (repr(Combine(PROBLEM(N('mda')), (N('ha'), N('hb'),), RelationalCondition))
            == 'combine(^solve.mda <- .ha, .hb)')
    assert repr(Combine(N('ha'), (N('ha'), N('hb'),), RelationalCondition)) == 'combine(.ha <- .ha, .hb)'


# ------------------------------------------------- Combine, bound at one it folds

def test_a_fold_at_a_name_it_folds_keeps_that_binding(two_problems):
    '''The one decision the op records beside the join: naming one of the problems it
    folds lands on `Replace`, so the binding -- name and place in the order -- survives.'''
    op = Combine(N('ha'), (N('ha'), N('hb'),), RelationalCondition)
    g  = op(two_problems)

    assert [n.spelling for n in g.nodes] == ['.fa', '.ha', '.fb']   # `.ha` where it was
    assert g[N('ha')] == two_problems[N('ha')] + two_problems[N('hb')]


def test_a_fold_at_a_name_it_does_not_fold_is_refused(two_problems):
    '''A binding of its own is an insertion, so it may not land on a node that stays --
    the join would replace something it never read.'''
    with pytest.raises(ValueError, match='already bound'):
        Combine(N('fa'), (N('ha'), N('hb'),), RelationalCondition)(two_problems)


def test_a_fold_needs_two_nodes_counting_the_one_it_binds_at(two_problems):
    with pytest.raises(ValueError, match='fewer than two nodes'):
        Combine(N('ha'), (N('ha'),), RelationalCondition)
    with pytest.raises(ValueError, match='fewer than two nodes'):
        Combine(PROBLEM(N('mda')), (N('ha'),), RelationalCondition)
    with pytest.raises(ValueError, match='names a node twice'):
        Combine(N('ha'), (N('hb'), N('hb'),), RelationalCondition)
    with pytest.raises(ValueError, match='names a node twice'):
        Combine(N('ha'), (N('ha'), N('ha'),), RelationalCondition)


# ---------------------------------------------------------------- rewiring any node shape

def test_a_rewire_works_on_a_statement_that_groups_its_places():
    '''A relational condition derives `reads` / `owns` from its relations, objectives and unknowns,
    so `Rewire` renames the place-holding fields and works on any definition -- which is
    what an MDF or SAND plan needs, since every one of them has an optimiser in it.'''
    g = graph_of(OrderedMap({
        N('opt'): optimise(F, (Z,), equalities=(C,)),
        N('m')  : call([Z, Y1], [F, C]),
        N('d')  : call([Z], [Y1]),
    }))
    # The optimiser is not even a reader here: every node of the graph is rebuilt.
    cut = Cut(Y1, (N('m'),), HAT)
    out = cut(g)
    assert out[N('m')].reads == (Z, cut.copy)
    assert out[N('opt')] is g[N('opt')]              # untouched, and the same object

    # ... and when it *is* the reader, the relation's operator survives the rename.
    op    = Cut(C, (N('opt'),), HAT)
    moved = op(g)
    assert is_optimise(moved[N('opt')])
    assert moved[N('opt')].relations == (Relation(op.copy, None, Eq),)
    assert moved[N('opt')].objectives == (Objective(F, Min),) and moved[N('opt')].unknowns == (Z,)


# ---------------------------------------------------------------- assign

class Started(DriverContract):
    '''A contract taking a start, answering whichever shape its final names. Nothing
    runs: what is under test is what the op writes into the graph.'''

    takes   = (Start,)
    reports = ()


def assigned(plan: Plan, problem, driver) -> Plan:
    '''`plan` with `driver` assigned at its default places.'''
    return plan + assign(problem, driver)


class Picard(Started):  accepts = staticmethod(is_fixed_point)
class Newton(Started):  accepts = staticmethod(is_root_find)
class Descent(Started): accepts = staticmethod(is_optimise)


def test_assign_derives_one_start_per_unknown(sellar):
    # The algorithm choice recorded as an op, its places with it: one `Start` per
    # unknown, named from the unknown -- the copy -- so it never shares the original's
    # start, and an ordinary boundary input like any other read.
    cut = closure(sellar, Y2)
    g   = assigned(Plan(sellar) + cut, cut.at, Picard()).graph
    node = g[cut.at]

    assert is_driven(node) and is_fixed_point(node.condition)
    assert node.data == ((START(HAT(Y2)),),)
    assert START(HAT(Y2)) in g.graph.boundary_inputs


def test_assign_derives_nothing_for_a_driver_requiring_nothing(sellar):
    # The other half: what a solve needs supplied follows from its algorithm, so a
    # driver that starts itself leaves the problem's reads alone.
    class SelfStarting(DriverContract):
        accepts = staticmethod(is_fixed_point)
        takes   = ()                        # starts itself from zeros
        reports = ()

    cut = closure(sellar, Y2)
    g   = assigned(Plan(sellar) + cut, cut.at, SelfStarting()).graph
    assert g[cut.at].data == ()
    assert g[cut.at].reads == g[cut.at].conditions


def test_assign_refuses_a_body_and_a_second_time(sellar):
    # A body answers nothing, so there is no condition node to attach an algorithm to; and
    # replacing one algorithm with another silently is not a rewrite -- `Undrive` says it.
    with pytest.raises(TypeError, match='not a condition node'):
        Plan(sellar) + Assign(N('d2'), Picard(), ((START(Y2),),), ())
    cut  = closure(sellar, Y2)
    plan = assigned(Plan(sellar) + cut, cut.at, Picard())
    with pytest.raises(TypeError, match='already carries a'):
        plan + Assign(cut.at, Picard(), ((START(Y2),),), ())


def test_assign_refuses_a_driver_that_answers_another_shape(sellar):
    # What makes an answer right is the problem's, not the driver's: the pairing is
    # checked when the driven node is minted, so a mismatch never reaches a schedule.
    cut = closure(sellar, Y2)
    with pytest.raises(ValueError, match='does not answer'):
        assigned(Plan(sellar) + cut, cut.at, Newton())


def test_assign_keeps_an_optimise_start_apart_from_its_conditions():
    # A `Start` among the equalities would claim to constrain, so it is derived beside
    # them and `conditions` never counts it.
    g = graph_of(OrderedMap({
        N('model'): call([V('x')], [V('f')]),
        N('opt')  : optimise(V('f'), (V('x'),)),
    }))
    opt = assigned(Plan(g), N('opt'), Descent()).graph[N('opt')]

    assert is_driven(opt) and is_optimise(opt.condition)
    assert opt.data == ((START(V('x')),),)
    assert opt.conditions == (V('f'),)


def test_a_rewire_reaches_the_statement_a_driver_wraps():
    '''
    `Driven` wraps the problem it answers, so a driven problem's *conditions* sit one
    definition down while its driver's data is derived on the wrapper. Without the
    wrapper relabelling through, a `Cut` whose reader is driven would rename nothing and
    say nothing, which is a wrong answer and not an error.
    '''
    g = graph_of(OrderedMap({
        N('f'): call([V('u')], [V('r')]),
        N('h'): root_find((V('r'),), (V('u'),)),
    }))
    driven = assigned(Plan(g), N('h'), Newton())
    after  = (driven + Cut(V('r'), readers=(N('h'),), copy_of=HAT)).graph[N('h')]

    assert [p.spelling for p in after.reads] == ['^copy.r', '^start.u']
    assert [p.spelling for p in after.conditions] == ['^copy.r']     # renamed, still a condition
    assert [p.spelling for slot in after.data for p in slot] == ['^start.u']   # untouched


def test_every_rewrite_case_has_wording():
    """The ops' own checks, each formatting: what `__post_init__` used to raise."""
    from cottax.core import violations
    from cottax.relational.rewrites import CutTwice, NothingCut
    cut = Cut(Y1, (N('d2'),), HAT)
    produced = set()
    for vs in (violations(FixedPointCut, (), N('mda')),
               violations(FixedPointCut, (cut, cut), N('mda'))):
        for x in vs:
            produced.add(type(x)); x.message
    assert produced == {NothingCut, CutTwice}
