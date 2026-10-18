"""
The relational condition: a `Relation` between two places, the `RelationalCondition`
over a tuple of them, and the shapes read back off it.

The claim under test is that one final over relations says everything the four problem
types said -- and two things they could not: a condition node whose unknowns are not chosen
yet, and one that mixes an equality with an inequality.
"""

import pytest
from jax.tree_util import DictKey

from cottax.core.maths.executable import ExecutableGraph
from cottax.relational import Min, Objective
from cottax.consumers.execution import CallableGraph
from cottax.core.maths.plan import Delete, Plan
from cottax.relational import (
    Eq, Le, Rel, Relation, RelationalCondition, is_constrained_optimise, is_equalities,
    is_feasibility, is_fixed_point, is_optimise, is_root_find, is_unconstrained_optimise,
)
from cottax.relational.relational_condition import (
    FixedAndMoved, FixedRelation, NotARel, ParameterUnrelated, RelatesNothing,
)
from cottax.pytree.path import NodePath, VarPath
from cottax.core import Function, problems, violations
from cottax.core.maths.definition import ReadsOwn
from cottax.core.realisation import ImplementedFunction
from cottax.core.driven import Driven
from cottax.relational.contracts import RootFinding, is_iteration
from cottax.assemblies.jax import absolute

assign = absolute.assign
import purposes
from cottax.core.frozen import OrderedMap
from cottax.consumers.visualization.sequencing import entries
from cottax.consumers.visualization.kinds import problem_type_of
from cottax.relational.consumers.visualization import kind_of
from graphs import graph_of
from cottax.core.maths.nesting import NestingGraph


def v(*names: str) -> VarPath:
    return VarPath(tuple(DictKey(x) for x in names))


def n(*names: str) -> NodePath:
    return NodePath(tuple(DictKey(x) for x in names))


def root_find(conditions, unknowns) -> RelationalCondition:
    return RelationalCondition(tuple(Relation(c, None, Eq) for c in conditions),
                               (), tuple(unknowns), ())


def fixed_point(conditions, unknowns) -> RelationalCondition:
    return RelationalCondition(tuple(Relation(u, c, Eq) for u, c in zip(unknowns, conditions)),
                               (), tuple(unknowns), ())


def optimise(objective, unknowns, equalities=(), inequalities=()) -> RelationalCondition:
    return RelationalCondition(
        tuple(Relation(h, None, Eq) for h in equalities)
        + tuple(Relation(g, None, Le) for g in inequalities),
        (Objective(objective, Min),), tuple(unknowns), ())


def stated(*relations) -> RelationalCondition:
    '''A requirement: relations alone.'''
    return RelationalCondition(relations, (), (), ())


def model() -> dict:
    return {n('m'): ImplementedFunction(
        Function((v('u'),), (v('gu'), v('r'), v('cost'), v('g1'))),
        lambda u: (0.5 * u + 1.0, u, u, u),
    )}


def graph_with(statement: RelationalCondition) -> NestingGraph:
    return graph_of(OrderedMap([*model().items(), (n('c'), statement)]))


# ---------------------------------------------------------------- relations

def test_zero_is_the_absence_of_a_side():
    # The one literal the language admits, spelled as absence -- so nothing numeric
    # enters the core at all.
    r = Relation(v('r'), None, Eq)
    assert r.rhs is None and r.terms == (v('r'),)
    assert Relation(v('a'), v('b'), Eq).terms == (v('a'), v('b'))
    assert Relation(None, v('r'), Le).terms == (v('r'),)         # and may stand on the left


def test_a_relation_with_zero_on_both_sides_relates_nothing():
    with pytest.raises(ValueError, match='relates nothing'):
        Relation(None, None, Eq)


def test_a_relation_relabels_both_sides():
    assert (Relation(v('a'), v('b'), Le).relabel({v('a'): v('x')})
            == Relation(v('x'), v('b'), Le))
    # zero has nothing to rename, and an untouched place is kept
    assert (Relation(v('a'), None, Eq).relabel({v('b'): v('x')})
            == Relation(v('a'), None, Eq))
    assert (Relation(None, v('b'), Le).relabel({v('b'): v('x')})
            == Relation(None, v('x'), Le))


def test_a_relation_is_written_as_it_reads():
    assert repr(Relation(v('a'), v('b'), Eq)) == f"{v('a')} = {v('b')}"
    assert repr(Relation(v('a'), None, Eq))   == f"{v('a')} = 0"
    assert repr(Relation(v('a'), v('b'), Le)) == f"{v('a')} <= {v('b')}"
    assert repr(Relation(None, v('r'), Le))   == f"0 <= {v('r')}"


def test_zero_on_the_left_is_a_root_find_like_zero_on_the_right():
    # Which side the zero is written on is a fact about the *writing*: an unknown is
    # never `None`, so this is no fixed point, and `0 = r` is `c(x) = 0` like any other
    # equality. It was a `'feasibility'` while `is_root_find` asked for a `None` right
    # side -- an artefact of that test, and a word the paper keeps for an inequality.
    c = RelationalCondition((Relation(None, v('r'), Eq),), (), (v('u'),), ())
    assert is_root_find(c) and not is_fixed_point(c) and not is_feasibility(c)
    assert problem_type_of(c) == 'root-find'


def test_an_operator_is_a_marker_and_never_a_value():
    for op in (Eq, Le):
        with pytest.raises(TypeError, match='marker, not a value'):
            op()


def test_a_term_may_name_an_unknown():
    # `u = g(u)` says `u` on the left, and a node may not *read* what it owns -- so a
    # term is a place, and one that is among the unknowns is what this node fixes;
    # every other term is a read. Membership, not a class.
    c = RelationalCondition((Relation(v('u'), v('gu'), Eq),), (), (v('u'),), ())
    assert c.reads == (v('gu'),) and c.owns == (v('u'),)
    d = RelationalCondition((Relation(v('z'), v('gu'), Eq),), (), (v('u'),), ())
    assert d.reads == (v('z'), v('gu')) and d.owns == (v('u'),)


# ------------------------------------------------------ the relational condition

def test_every_field_is_written():
    # No defaults on the relational finals: a caller says which operator, says `None` for
    # zero, and writes `()` where there is nothing.
    with pytest.raises(TypeError):
        Relation(v('a'), v('b'))
    with pytest.raises(TypeError):
        RelationalCondition((Relation(v('r'), None, Eq),))
    with pytest.raises(TypeError):
        RelationalCondition((Relation(v('r'), None, Eq),), ())


def test_a_statement_relates_something():
    with pytest.raises(ValueError, match='relates nothing'):
        RelationalCondition((), (), (v('u'),), ())
    found, = violations(RelationalCondition, (), (), (v('u'),), ())
    assert isinstance(found, RelatesNothing) and found.message


def test_conditions_are_the_objectives_then_the_terms_that_are_not_unknowns():
    c = RelationalCondition(
        (Relation(v('u'), v('gu'), Eq), Relation(v('g1'), None, Le)),
        (Objective(v('cost'), Min),), (v('u'),), ())
    assert c.conditions == (v('cost'), v('gu'), v('g1'))
    assert c.reads == c.conditions and c.owns == c.unknowns


@pytest.mark.parametrize('op', ['eq', int])
def test_a_relation_holds_a_rel_class(op):
    '''The op is asked by identity, so anything but a `Rel` class is refused, not written
    as some other symbol.'''
    with pytest.raises(TypeError, match='is not a `Rel` class'):
        Relation(v('a'), v('b'), op)
    assert [type(x) for x in violations(Relation, v('a'), v('b'), op)] == [NotARel]


def test_a_new_rel_writes_its_own_symbol():
    class Ge(Rel):
        symbol = '>='
    assert repr(Relation(v('a'), None, Ge)) == f"{v('a')} >= 0"


def test_a_statement_is_written_as_it_reads():
    c = RelationalCondition((Relation(v('r'), None, Eq),), (Objective(v('cost'), Min),), (v('u'),), ())
    assert repr(c) == f"RelationalCondition(min {v('cost')}; {v('r')} = 0 over {v('u')})"
    assert '(requirement)' in repr(stated(Relation(v('r'), None, Eq)))


# ------------------------------------------------------ parameters

def test_a_parameter_is_read_and_is_no_condition():
    '''`lo <= u`, `g <= g_max` with `lo`, `g_max` held fixed: they are read, after the
    conditions, and only `g` must be moved by the unknowns.'''
    n = RelationalCondition((Relation(v('lo'), v('u'), Le), Relation(v('g'), v('g_max'), Le)),
                            (), (v('u'),), (v('lo'), v('g_max')))
    assert n.conditions == (v('g'),)
    assert n.reads == (v('g'), v('lo'), v('g_max'))
    assert n.owns == (v('u'),)


def test_a_parameter_is_held_fixed_related_and_never_alone():
    u, g, lo, hi = v('u'), v('g'), v('lo'), v('hi')
    found = lambda relations, objectives, parameters: [type(x) for x in violations(
        RelationalCondition, relations, objectives, (u,), parameters)]
    assert found((Relation(lo, u, Le),), (), (u,)) == [ReadsOwn, FixedAndMoved]
    assert found((Relation(g, u, Le),), (Objective(g, Min),), (g,)) == [FixedAndMoved]
    assert found((Relation(g, None, Le),), (), (lo,)) == [ParameterUnrelated]
    assert found((Relation(lo, hi, Le), Relation(lo, None, Le)), (), (lo, hi)) == [FixedRelation]
    assert found((Relation(lo, u, Le), Relation(u, hi, Le)), (), (lo, hi)) == []


def test_the_join_holds_fixed_what_either_side_does_once():
    a = RelationalCondition((Relation(v('lo'), v('u'), Le),), (), (v('u'),), (v('lo'),))
    b = RelationalCondition((Relation(v('lo'), v('w'), Le),), (), (v('w'),), (v('lo'),))
    assert (a + b).parameters == (v('lo'),)
    assert (a + b).relabel({v('lo'): v('floor')}).parameters == (v('floor'),)


def test_a_statement_relabels_relations_objectives_and_unknowns():
    c = RelationalCondition((Relation(v('u'), v('gu'), Eq),), (Objective(v('cost'), Min),), (v('u'),), ())
    back = c.relabel({v('u'): v('x'), v('cost'): v('f')})
    assert back.unknowns == (v('x'),) and back.objectives == (Objective(v('f'), Min),)
    assert back.relations == (Relation(v('x'), v('gu'), Eq),)


# ---------------------------------------------------------------- the shapes

@pytest.mark.parametrize('statement, shape', [
    (root_find((v('r'),), (v('u'),)),                                    'root-find'),
    (fixed_point((v('gu'),), (v('u'),)),                                 'fixed-point'),
    (optimise(v('cost'), (v('u'),), inequalities=(v('g1'),)),            'optimise'),
    (RelationalCondition((Relation(v('g1'), None, Le),), (), (v('u'),), ()), 'feasibility'),
])
def test_a_shape_is_read_off_the_relations(statement, shape):
    assert problem_type_of(statement) == shape


def test_each_predicate_says_what_it_says():
    constrained = optimise(v('cost'), (v('u'),), inequalities=(v('g1'),))
    assert is_optimise(constrained) and not is_unconstrained_optimise(constrained)
    assert is_unconstrained_optimise(optimise(v('cost'), (v('u'),)))
    assert is_constrained_optimise(constrained)
    # a feasibility is an *inequality* with no objective, never the whole no-objective
    # branch: a root find no longer answers to it
    assert not is_feasibility(root_find((v('r'),), (v('u'),)))
    assert is_feasibility(RelationalCondition((Relation(v('g'), None, Le),), (), (v('u'),), ()))
    assert is_fixed_point(fixed_point((v('gu'),), (v('u'),)))
    assert is_root_find(root_find((v('r'),), (v('u'),)))
    # a root find is every relation against zero, however many unknowns
    assert is_root_find(root_find((v('r'),), (v('u'), v('x'))))
    assert not is_root_find(fixed_point((v('gu'),), (v('u'),)))
    # a fixed point's left side is an unknown of its own; anything else root-finds
    assert is_root_find(
        RelationalCondition((Relation(v('z'), v('gu'), Eq),), (), (v('u'),), ()))
    # and an objective disqualifies every one of the no-objective finals
    assert not is_feasibility(constrained) and not is_equalities(constrained)


def test_a_statement_with_no_unknowns_has_no_shape_yet():
    # A shape is what the relations *and* the unknowns make, so every predicate is false
    # until `Determine` names them.
    req = stated(Relation(v('r'), None, Eq))
    assert not any(p(req) for p in (is_root_find, is_fixed_point, is_optimise,
                                    is_feasibility, is_unconstrained_optimise))
    assert problem_type_of(req) == 'stated'


def test_a_statement_matching_no_shape_is_declared():
    two = optimise(v('cost'), (v('a'),)) + optimise(v('g1'), (v('b'),))
    assert len(two.objectives) == 2 and kind_of(two) is None


def test_a_requirement_draws_as_the_problem_its_left_sides_make_it():
    '''`kind_of` of a requirement is that of the same relations determining their left
    sides (a side of zero's other side), as the predicates read it.'''
    g, h, r, u, w = (v(x) for x in 'ghruw')
    corpus = [
        (Relation(r, None, Eq),),                               # r = 0
        (Relation(None, h, Eq),),                               # 0 = h
        (Relation(u, w, Eq),),                                  # u = w
        (Relation(None, h, Eq), Relation(u, w, Eq)),            # 0 = h, u = w
        (Relation(r, None, Eq), Relation(u, w, Eq)),            # r = 0, u = w
        (Relation(g, None, Le),),                               # g <= 0
        (Relation(g, None, Le), Relation(u, w, Eq)),            # g <= 0, u = w
    ]
    for rels in corpus:
        lefts = tuple(dict.fromkeys(x.lhs if x.lhs is not None else x.rhs for x in rels))
        problem = RelationalCondition(rels, (), lefts, ())
        assert kind_of(stated(*rels)) == kind_of(problem) is not None, rels


def test_the_five_finals_are_disjoint():
    '''`tab:conditions`: every condition node that determines something with at most one
    objective is exactly one of them, so the order `kind_of` asks them in settles
    nothing. Two objectives match none, and `kind_of` gives them no category.'''
    finals = (is_unconstrained_optimise, is_constrained_optimise, is_feasibility,
              is_fixed_point, is_root_find)
    corpus = {
        'unconstrained optimise': optimise(v('f'), (v('x'),)),
        'constrained optimise'  : optimise(v('f'), (v('x'),), equalities=(v('h'),)),
        'optimise with both'    : optimise(v('f'), (v('x'),), equalities=(v('h'),),
                                           inequalities=(v('g'),)),
        'feasibility'           : RelationalCondition((Relation(v('g'), None, Le),),
                                                      (), (v('u'),), ()),
        'feasibility + equality': RelationalCondition(
            (Relation(v('g'), None, Le), Relation(v('h'), None, Eq)), (), (v('u'),), ()),
        'root find'             : root_find((v('r'),), (v('u'),)),
        'fixed point'           : fixed_point((v('g'),), (v('u'),)),
        'the join of the two'   : root_find((v('r'),), (v('x'),))
                                  + fixed_point((v('g'),), (v('u'),)),
        'zero on the left'      : RelationalCondition((Relation(None, v('r'), Eq),),
                                                      (), (v('u'),), ()),
    }
    for name, c in corpus.items():
        assert sum(f(c) for f in finals) == 1, f'{name} is not exactly one final'
    two = optimise(v('f'), (v('x'),)) + optimise(v('g'), (v('y'),))
    assert not any(f(two) for f in finals)
    # and each level is the union of the two finals under it
    for c in corpus.values():
        assert is_optimise(c) == (is_unconstrained_optimise(c) or is_constrained_optimise(c))
        assert is_equalities(c) == (is_fixed_point(c) or is_root_find(c))


def test_a_root_find_joined_with_a_fixed_point_is_a_root_find():
    # What `Close` leaves: one relation against zero and one with two sides. Not every
    # relation puts an unknown on the left, so it is not the fixed point -- and the
    # paper's footnote is that anything else is `c(x) = 0` WLOG, which a Newton takes.
    joined = root_find((v('r'),), (v('x'),)) + fixed_point((v('g'),), (v('u'),))
    assert is_equalities(joined)
    assert is_root_find(joined) and not is_fixed_point(joined)
    assert (is_equalities(root_find((v('r'),), (v('u'),)))
            and is_equalities(fixed_point((v('g'),), (v('u'),))))
    assert not is_equalities(optimise(v('f'), (v('x'),), equalities=(v('h'),)))


def test_a_shape_reads_the_operators_and_the_sides_and_counts_nothing():
    # A place names one value of any shape, so as many relations as unknowns says
    # nothing about the system being square -- that is a fact about the values, and the
    # driver that inverts them refuses it per solve.
    two = root_find((v('r'), v('s')), (v('u'),))
    assert is_root_find(two) and is_equalities(two) and problem_type_of(two) == 'root-find'


def test_a_shape_predicate_answers_false_off_a_relational_condition():
    """
    A function, or a driven node, has no relations to read: every predicate answers
    False, and a drawing gives it no category.
    """
    body = ImplementedFunction(Function((v('u'),), (v('r'),)), lambda u: u)
    driven = Driven(root_find((v('r'),), (v('u'),)), RootFinding(steps=False),
                    ((purposes.Start()(v('u')),),), ())
    for node in (body, driven):
        for p in (is_root_find, is_fixed_point, is_optimise, is_feasibility, is_equalities,
                  is_unconstrained_optimise, is_constrained_optimise, is_iteration):
            assert p(node) is False
    assert problem_type_of(body) is None


def test_a_shape_predicate_reads_through_a_driven_node():
    """A driven node's shape is its condition's, so it is the same before and after
    `Assign`; a drawing reads through, a predicate is asked of the condition."""
    driven = Driven(root_find((v('r'),), (v('u'),)), RootFinding(steps=False),
                    ((purposes.Start()(v('u')),),), ())
    assert is_root_find(driven.condition) and problem_type_of(driven) == 'root-find'


# ---------------------------------------------------------------- the join

def test_the_join_concatenates_in_order_and_is_associative():
    a = root_find((v('r'),), (v('u'),))
    b = RelationalCondition((Relation(v('g1'), None, Le),), (), (v('x'),), ())
    c = RelationalCondition((), (Objective(v('cost'), Min),), (v('y'),), ())
    assert (a + b) + c == a + (b + c)
    joined = a + b
    assert (joined.relations   == a.relations   + b.relations
            and joined.objectives == a.objectives + b.objectives
            and joined.unknowns   == a.unknowns   + b.unknowns)
    assert (b + a).relations == b.relations + a.relations    # this side first


def test_the_join_refuses_anything_but_a_relational_condition_on_either_side():
    """Joining a problem with a driven one discards the algorithm answering it, and which
    should answer the join cannot be read off the two (two driven alike join, in
    `test_driven_join`). A node with no `__add__` already raises, so a function says so by
    not implementing it."""
    plain  = root_find((v('r'),), (v('u'),))
    driven = Driven(root_find((v('r2'),), (v('x'),)), RootFinding(steps=False),
                    ((purposes.Start()(v('x')),),), ())
    body   = ImplementedFunction(Function((v('u'),), (v('r'),)), lambda u: u)
    for a, b in ((plain, driven), (driven, plain), (body, body),
                 (optimise(v('f'), (v('x'),)), driven)):
        with pytest.raises(TypeError, match='unsupported operand'):
            a + b


def test_driven_join():
    """Two problems driven alike join under that contract: the conditions by their own `+`,
    a per-unknown role at each unknown's place, a per-problem one at the one place both take.
    Two contracts, or a per-problem role at two places, are refused."""
    from cottax.core.driven.nodes import DriversDiffer, RoleAtTwoPlaces
    start = purposes.Start()
    def driven(r, u, steps=None):
        contract = RootFinding(steps=steps is not None)
        return Driven(root_find((v(r),), (v(u),)), contract, ((start(v(u)),),),
                      ((v(steps),),) if steps else ())
    joined = driven('r1', 'u1') + driven('r2', 'u2')
    assert joined == Driven(root_find((v('r1'), v('r2')), (v('u1'), v('u2'))),
                            RootFinding(steps=False), ((start(v('u1')), start(v('u2'))),), ())
    assert (driven('r1', 'u1', 'n') + driven('r2', 'u2', 'n')).reports == ((v('n'),),)
    with pytest.raises(ValueError) as refused:
        driven('r1', 'u1') + driven('r2', 'u2', 'n')
    assert refused.value.violation == DriversDiffer(RootFinding(steps=False), RootFinding(steps=True))
    with pytest.raises(ValueError) as refused:
        driven('r1', 'u1', 'n') + driven('r2', 'u2', 'm')
    assert refused.value.violation == RoleAtTwoPlaces('Steps', (v('n'),), (v('m'),))


# ---------------------------------------------------------------- running is refused

def test_a_proof_refuses_a_statement_that_determines_nothing():
    g = graph_with(stated(Relation(v('r'), None, Eq)))
    assert problems(g.definitions) == ()
    with pytest.raises(ValueError, match='holds .* against nothing'):
        ExecutableGraph(g)


def test_drawing_blocking_and_pruning_are_all_fine():
    from cottax.consumers.visualization.xdsm import xdsm_struct
    g = graph_with(stated(Relation(v('r'), None, Eq)))
    assert tuple(e.nodes for e in entries(g)) == ((n('m'),), (n('c'),))
    assert xdsm_struct(g)['problem_types'] == {'c': 'stated'}
    assert g.graph.ancestors((g.graph.owners[v('cost')],)) == (n('m'),)


def test_deleting_it_is_how_a_graph_asserting_nothing_runs():
    p = Plan(graph_with(stated(Relation(v('r'), None, Eq)))) + Delete((n('c'),))
    CallableGraph(p.graph)                                # builds


def test_a_problem_with_no_conditions_is_off_its_cycle():
    """
    `u = 0` with `u` its own unknown builds, takes a driver -- it is square and vanishes
    -- and reads nothing. Nothing the driver could move depends on `u`, so the blocking
    refuses it as a problem outside every cycle, not by blaming a missing driver. Asked
    of the problem and not of its conditions: "every condition is constant" is
    vacuously true here, and once let it through.
    """
    p = Plan(graph_with(RelationalCondition((Relation(v('u'), None, Eq),), (), (v('u'),), ())))
    p = p + assign(n('c'), RootFinding(steps=False))
    assert p.graph[n('c')].conditions == ()
    with pytest.raises(ValueError, match='in no cycle'):
        ExecutableGraph(p.graph)
