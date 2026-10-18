'''
`core/maths/plan/ops.py`'s edits on what a node reads, owns and determines: `Cut`, which
names no relation, and `Determine` / `Undetermine`, which ask the node itself
(`ConditionalNode.determined`) for the condition node over other unknowns.

A condition node is written as a `RelationalCondition`, since `core` ships no final; what is
under test is what the op writes into the graph, and never how the condition node is spelled.

Names are built by hand from key entries, as in `test_algebra.py` and `test_rewrites.py`.
'''

import pytest
from jax.tree_util import GetAttrKey

from cottax.core import Function, problems
from cottax.core.maths.plan import Cut, Determine, Insert, Plan, Undetermine, refusals
from cottax.core.driven import Undrive
from cottax.relational.contracts import RootFinding
from cottax.assemblies.jax import absolute

assign = absolute.assign
import purposes
from cottax.core.frozen import OrderedMap
from cottax.pytree.path import NodePath, VarPath
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.consumers.visualization.kinds import problem_type_of
from graphs import graph_of
from cottax.core.maths.nesting import NestingGraph


HAT = purposes.Copy()


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


X, Z, Y1, Y2, F, C, R, U, GU = (V(s) for s in 'x z y1 y2 f c r u gu'.split())


def call(reads, owns) -> Function:
    '''A computation: what it reads and what it owns, which is all an edit ever sees.'''
    return Function(tuple(reads), tuple(owns))


def source(var) -> Function:
    return call([], [var])


def stated(*relations) -> RelationalCondition:
    '''A requirement: relations alone.'''
    return RelationalCondition(relations, (), (), ())


def names(graph):
    '''The graph's node names as they are written, sorted -- a set of names.'''
    return sorted(n.spelling for n in graph.nodes)


def contributing(graph, *wanted):
    '''What a run to `wanted` needs -- the ancestors of their producers -- as a graph.
    The hazard below is about what this drops.'''
    owners = graph.graph.owners
    return graph.subgraph(graph.graph.ancestors(owners[v] for v in wanted if v in owners))


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


# ---------------------------------------------------------------- Cut, and its hazard

def opening(graph, var):
    '''`Cut(var, ..., HAT)` aimed at whatever closes the cycle -- what a caller writes.'''
    return Cut(var, graph.graph.closing_readers(var), HAT)


def test_a_bare_cut_leaves_the_producer_droppable(looped):
    '''The hazard: `d2` was the only reader of `y1`, so after the cut nothing reads it
    and restricting to what `f` needs drops `d1` -- the cut has silently become a delete.'''
    assert names(contributing(opening(looped, Y1)(looped), F)) == ['.d2', '.obj']


def test_a_residual_makes_the_pair_an_edge(looped):
    '''Reading both halves puts the producer back in the structure, where a restriction sees it.'''
    op = opening(looped, Y1)
    g  = (looped + op + Insert(OrderedMap({N('res'): call([Y1, op.copy], [R])}))).graph
    assert g.graph.is_acyclic
    assert names(contributing(g, F, R)) == ['.d1', '.d2', '.obj', '.res']


def test_a_source_that_reads_nothing_leaves_the_producer_droppable(looped):
    '''Driving from outside: acyclic, but nothing inside needs the computed value.'''
    op = opening(looped, Y1)
    g  = (looped + op + Insert(OrderedMap({N('guess'): source(op.copy)}))).graph
    assert g.graph.is_acyclic
    assert '.d1' not in names(contributing(g, F))


def test_a_source_that_reads_the_computed_value_keeps_the_cycle(looped):
    '''`d1 -> mda -> d2 -> d1`. The loop is closed inside the graph, so cutting bought
    nothing that assigning a driver would not have.'''
    op = opening(looped, Y1)
    g  = (looped + op + Insert(OrderedMap({N('mda'): call([Y1], [op.copy])}))).graph
    assert not g.graph.is_acyclic
    assert '.d1' in names(contributing(g, F))


def test_a_variable_is_cut_at_most_once(sellar):
    '''Two cuts would assume two values for one place. `obj` still reads the computed
    `y1`, so the second cut is well-formed as a *rewire* and has to be caught on the
    name it would mint twice.'''
    plan = Plan(sellar) + opening(sellar, Y1)
    with pytest.raises(ValueError, match='already cut'):
        plan + Cut(Y1, (N('obj'),), HAT)


@pytest.fixture
def boundary_fed():
    """`.x` is a boundary input, read by `.a` and produced by nobody."""
    return graph_of(OrderedMap({N('a'): call([X], [Y1]), N('b'): call([Y1], [F])}))


def test_what_an_op_needs_of_the_graph_can_be_asked_for(boundary_fed, sellar):
    """
    The third family: `checks_graph`, what an op needs of the graph before it can compute
    anything. `refusals` reports them as data where `apply` raises the first, so one is
    matched by kind exactly as a rule's is.
    """
    unproduced = Cut(X, (N('a'),), HAT)
    assert [type(v).__name__ for v in refusals(unproduced, boundary_fed)] == ['NotProduced']
    assert list(refusals(Cut(Y1, (N('d2'),), HAT), sellar)) == []   # applies: nothing to say
    with pytest.raises(KeyError, match='nothing to cut'):
        unproduced.apply(boundary_fed)


def test_an_ops_own_precondition_is_asked_before_the_one_it_inherits(boundary_fed):
    """`Rewire`'s check reads the renaming `Cut` computes, so the derived level is asked
    first -- otherwise a cut of a place the reader has not got reports the inherited
    refusal and never says nothing produces it."""
    off = Cut(V('nope'), (N('a'),), HAT)          # `.a` does not read `.nope` either
    assert [type(v).__name__ for v in refusals(off, boundary_fed)] == ['NotProduced',
                                                                       'DoesNotRead',
                                                                       'NotRenamed']


def test_a_cut_names_its_copy_by_the_function_it_carries():
    cut = Cut(Y1, (N('d2'),), purposes.Guess())
    assert cut.copy.spelling == '^guess.y1'
    assert repr(cut) == 'cut(.y1->^guess.y1)'


# ---------------------------------------------------------------- the unknowns

def graph_with(statement: RelationalCondition) -> NestingGraph:
    '''One model owning everything a condition node below relates, and the condition node.'''
    return graph_of(OrderedMap({N('m'): call([U], [GU, R, C]), N('c'): statement}))


def test_determine_names_the_unknowns_and_keeps_the_binding():
    p = Plan(graph_with(stated(Relation(R, None, Eq)))) + Determine(N('c'), (U,))
    assert p.graph.nodes == (N('m'), N('c'))            # in place: the binding did not move
    assert problem_type_of(p.graph[N('c')]) == 'root-find'
    assert p.graph[N('c')].unknowns == (U,)
    assert problems(p.graph.definitions) == (N('c'),)


def test_determine_takes_no_target_type_because_the_shape_follows():
    '''The op edits the unknowns and nothing else, so what makes an answer right is read
    back off the relations the condition node already carries.'''
    p = Plan(graph_with(stated(Relation(U, GU, Eq)))) + Determine(N('c'), (U,))
    assert problem_type_of(p.graph[N('c')]) == 'fixed-point'
    assert p.graph[N('c')].reads == (GU,)               # the term moved to an unknown


def test_undetermine_is_the_inverse():
    for req in (stated(Relation(R, None, Eq)), stated(Relation(U, GU, Eq))):
        p = Plan(graph_with(req)) + Determine(N('c'), (U,))
        assert (p + Undetermine(N('c'))).graph[N('c')] == req


def test_determine_refuses_a_statement_that_already_determines_something():
    p = Plan(graph_with(stated(Relation(R, None, Eq)))) + Determine(N('c'), (U,))
    with pytest.raises(TypeError, match='`Undetermine` it first'):
        p + Determine(N('c'), (GU,))


def test_undetermine_refuses_a_statement_that_determines_nothing():
    with pytest.raises(TypeError, match='nothing to take off'):
        Plan(graph_with(stated(Relation(R, None, Eq)))) + Undetermine(N('c'))


def test_the_ops_read_written():
    assert repr(Determine(N('c'), (U,))) == 'determine(.c <- .u)'
    assert repr(Undetermine(N('c'))) == 'undetermine(.c)'


# ------------------------------------------- what a node answered by an algorithm says

def driven() -> Plan:
    '''A root find with a contract on it; nothing runs, what is under test is the
    refusal.'''
    g = graph_of(OrderedMap({N('m'): call([U], [R]),
                          N('c'): RelationalCondition((Relation(R, None, Eq),), (), (U,), ())}))
    return Plan(g) + assign(N('c'), RootFinding(steps=False))


def test_what_a_driven_statement_determines_cannot_be_edited():
    '''The algorithm was chosen for the problem as it stands and its data is derived
    from the unknowns it had, so the node itself refuses -- `Determine` never gets a
    kind to test, and `Undetermine` reaches the same refusal one step later.'''
    p = driven()
    with pytest.raises(TypeError, match='`Undrive` it first'):
        p.graph[N('c')].determined((Z,))
    with pytest.raises(TypeError, match='`Undrive` it first'):
        p + Undetermine(N('c'))
    with pytest.raises(TypeError, match='`Undetermine` it first'):
        p + Determine(N('c'), (Z,))


def test_undriving_first_is_what_makes_it_editable():
    p = driven() + Undrive(N('c')) + Undetermine(N('c'))
    assert p.graph[N('c')].unknowns == ()
    assert (p + Determine(N('c'), (U,))).graph[N('c')] == driven().graph[N('c')].condition
