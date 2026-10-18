'''
Safe cutting (`core/maths/cut_rules.py`, `consumers/architectures/closing.py`): `per_functional_cycle` is
the decision over the graph as handed, the condition nodes dropped and each cycle handed to
the rule; a `Closing` makes one closure of one group, and `CloseFunctionalCycles` is the op recording a rule. The
fixtures and helpers here are shared with `test_mdao_architectures.py`.
'''

import pytest
from jax.tree_util import GetAttrKey

from cottax.core import Function, is_problem
from cottax.relational import Min, Objective
from cottax.core.maths.executable import ExecutableGraph
from cottax.core.maths.plan import Cut, Nest, Plan
from cottax.core.frozen import violations
from cottax.core.realisation import ImplementedFunction
from cottax.consumers.architectures import IDF, MDF, SAND
from cottax.core.maths.cut_rules import (
    ByName, CutRule, GaussSeidel, GaussSeidelMinimal, GaussSeidelOrdered, Jacobi, NoCuts,
    coupling_reads, cuts_per_cycle, gauss_seidel, minimal_feedback_order, per_functional_cycle,
)
from cottax.consumers.architectures import AtFirstMember, CloseFunctionalCycles, closed
from cottax.relational.consumers.architectures import FixedPointClosing
import purposes
from cottax.core.frozen import OrderedMap
from cottax.pytree.path import NodePath, VarPath
from cottax.relational import Eq, Le, Relation, RelationalCondition, is_optimise
from cottax.core.maths.plan import Combine
from cottax.relational.rewrites import FixedPointCut
from cottax.consumers.visualization.sequencing import Run, Solve, entries
from graphs import graph_of

HAT, MINT_MDA = purposes.Copy(), purposes.Closure()


def optimise(objective, unknowns, inequalities=()) -> RelationalCondition:
    return RelationalCondition(tuple(Relation(g, None, Le) for g in inequalities),
                               (Objective(objective, Min),), unknowns, ())


def root_find(conditions, unknowns) -> RelationalCondition:
    return RelationalCondition(tuple(Relation(c, None, Eq) for c in conditions), (), unknowns, ())


def fixed_point(conditions, unknowns) -> RelationalCondition:
    return RelationalCondition(tuple(Relation(u, c, Eq) for u, c in zip(unknowns, conditions)),
                               (), unknowns, ())


def requirement(condition, op=Le) -> RelationalCondition:
    return RelationalCondition((Relation(condition, None, op),), (), (), ())


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def call(reads, owns):
    return ImplementedFunction(Function(tuple(V(r) for r in reads), tuple(V(o) for o in owns)),
                       lambda *a: None)


def names(nodes):
    return sorted(n.spelling for n in nodes)


def blocks(g):
    return [names(e.nodes) for e in entries(g)]


def cyc(g, *nodes):
    '''A block as the graph a rule reads: owns and reads, no nesting.'''
    return g.graph.subgraph(nodes)


def bodies(g):
    '''The function nodes of `g`, as a rule sees them under `Bodies()`.'''
    return g.graph.subgraph(n for n, d in g.definitions.items() if not is_problem(d))


OPT, D1, D2 = N('opt'), N('D1'), N('D2')
H12 = (N('h1'), N('h2'))
CLOSING = FixedPointClosing(HAT, AtFirstMember(MINT_MDA))
__all__ = ['optimise', 'root_find', 'fixed_point', 'requirement', 'V', 'N', 'call', 'names',
           'blocks', 'cyc', 'bodies', 'OPT', 'D1', 'D2', 'H12', 'HAT', 'MINT_MDA', 'CLOSING', 'Y2_MDA',
           'mdf', 'idf', 'sand', 'sellar', 'implicit', 'two_cycles', 'through_explicit']

Y2_MDA = MINT_MDA(D1)                  # what a Gauss-Seidel over (D1, D2) binds, named by `at`


def mdf(rule=GaussSeidel()) -> MDF:
    return MDF(rule, CLOSING, OPT)


def idf(rule=GaussSeidel()) -> IDF:
    return IDF(rule, CLOSING, OPT)


def sand(rule=GaussSeidel()) -> SAND:
    return SAND(rule, CLOSING, OPT)


@pytest.fixture
def sellar():
    '''Explicit disciplines, coupled through `y1` / `y2`, under an optimiser.'''
    return graph_of(OrderedMap({
        OPT     : optimise(V('f'), (V('z'),), (V('g'),)),
        D1      : call(['z', 'y2'], ['y1']),
        D2      : call(['z', 'y1'], ['y2']),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
        N('con'): call(['y1'], ['g']),
    }))


@pytest.fixture
def implicit():
    '''Implicit disciplines: each a body computing a residual and a problem owning the
    state, coupled through those states.'''
    return graph_of(OrderedMap({
        OPT     : optimise(V('f'), (V('z'),), (V('g'),)),
        D1      : call(['z', 'y1', 'y2'], ['r1']),
        N('h1') : root_find((V('r1'),), (V('y1'),)),
        D2      : call(['z', 'y1', 'y2'], ['r2']),
        N('h2') : root_find((V('r2'),), (V('y2'),)),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
        N('con'): call(['y1'], ['g']),
    }))


@pytest.fixture
def two_cycles():
    '''Two disjoint coupled pairs beneath one optimiser, B downstream of A.'''
    return graph_of(OrderedMap({
        OPT     : optimise(V('f'), (V('z'),)),
        N('A1') : call(['z', 'a2'], ['a1']),
        N('A2') : call(['a1'], ['a2']),
        N('B1') : call(['a2', 'b2'], ['b1']),
        N('B2') : call(['b1'], ['b2']),
        N('obj'): call(['z', 'b2'], ['f']),
    }))


@pytest.fixture
def through_explicit():
    '''An implicit `A1` fed back through the explicit `A2`, and a lone implicit `B1`
    whose own problem closes its cycle.'''
    return graph_of(OrderedMap({
        OPT     : optimise(V('f'), (V('z'),)),
        N('A1') : call(['z', 'a2', 'ua'], ['ra', 'a1']),
        N('hA') : root_find((V('ra'),), (V('ua'),)),
        N('A2') : call(['a1'], ['a2']),
        N('B1') : call(['a2', 'ub'], ['rb', 'b1']),
        N('hB') : root_find((V('rb'),), (V('ub'),)),
        N('obj'): call(['z', 'b1'], ['f']),
    }))


# ================================================================ the rules

def test_gauss_seidel_opens_the_back_edges_of_the_order(sellar):
    assert gauss_seidel(cyc(sellar, D1, D2), (D1, D2)) == {V('y2'): (D1,)}
    assert gauss_seidel(cyc(sellar, D1, D2), (D2, D1)) == {V('y1'): (D2,)}


def test_jacobi_is_coupling_reads_itself(sellar):
    cycle = cyc(sellar, D1, D2)
    assert Jacobi().to_cuts(cycle) == (coupling_reads(cycle),) == ({V('y1'): (D2,), V('y2'): (D1,)},)
    assert NoCuts().to_cuts(cycle) == ()


def test_coupling_reads_holds_nothing_back(implicit):
    '''Raw, every read of an owned variable is a coupling, a problem's included. What
    a rule sees is `per_functional_cycle`'s decision, never a kind read in the rule.'''
    cycle = cyc(implicit, D1, N('h1'), D2, N('h2'))
    assert coupling_reads(cycle) == {V('r1'): (N('h1'),), V('y1'): (D1, D2),
                                     V('r2'): (N('h2'),), V('y2'): (D1, D2)}
    folded = implicit.contract(((N('h1'), D1), (N('h2'), D2)))
    assert coupling_reads(folded.graph) == {V('z'): (D1, D2, N('obj')),
                                            V('y1'): (D2, N('obj'), N('con')),
                                            V('y2'): (D1, N('obj')),
                                            V('f'): (OPT,), V('g'): (OPT,)}


def test_gauss_seidel_takes_one_order(two_cycles):
    '''Three sweeps, one abstract `order_of`. An order naming a cycle's nodes wrongly
    is refused rather than falling back to binding order.'''
    default = per_functional_cycle(two_cycles, GaussSeidel())
    flipped = per_functional_cycle(two_cycles, ByName(((N('A1'), GaussSeidelOrdered((N('A2'), N('A1')))),
                                                   (N('B1'), GaussSeidel()))))
    assert [set(g) for g in default] == [{V('a2')}, {V('b2')}]
    assert [set(g) for g in flipped] == [{V('a1')}, {V('b2')}]
    for order in ((N('A2'), N('A1')), (N('A1'), N('A1'), N('A2')), (N('A1'),)):
        with pytest.raises(ValueError, match='exactly once each'):
            per_functional_cycle(two_cycles, GaussSeidelOrdered(order))
    assert repr(GaussSeidelOrdered((N('A2'), N('A1')))) == 'gauss_seidel(.A2, .A1)'


def test_by_name_refuses_a_cycle_it_does_not_name(two_cycles):
    half = ByName(((N('A1'), GaussSeidel()),))
    with pytest.raises(ValueError, match='no rule keyed by a node of'):
        per_functional_cycle(two_cycles, half)
    whole = ByName(((N('A1'), GaussSeidel()), (N('B1'), NoCuts())))
    assert [set(g) for g in per_functional_cycle(two_cycles, whole)] == [{V('a2')}]


def test_by_name_is_a_mapping_with_one_key_per_cycle(two_cycles):
    with pytest.raises(ValueError, match='key more than one rule'):
        ByName(((N('A1'), Jacobi()), (N('A1'), NoCuts())))
    # only the repeated key, once
    found, = violations(ByName, ((N('A1'), Jacobi()), (N('A2'), Jacobi()),
                                 (N('A1'), NoCuts()), (N('A1'), GaussSeidel())))
    assert found.keys == (str(N('A1')),)
    both = ByName(((N('A1'), Jacobi()), (N('A2'), NoCuts())))
    with pytest.raises(ValueError, match='several rules keyed on one cycle'):
        per_functional_cycle(two_cycles, both)


def test_a_rule_per_cycle_is_a_rule_of_its_own(two_cycles):
    both = per_functional_cycle(two_cycles, ByName(((N('A1'), GaussSeidel()), (N('B1'), Jacobi()))))
    assert [set(g) for g in both] == [{V('a2')}, {V('b1'), V('b2')}]


def test_the_minimal_order_opens_fewest():
    '''a -> b -> c -> a, and c reads a as well: binding order (c, a, b) opens two
    variables, the minimal order (a, b, c) one.'''
    g = graph_of(OrderedMap({
        N('c'): call(['vb', 'va'], ['vc']),
        N('a'): call(['vc'], ['va']),
        N('b'): call(['va'], ['vb']),
    }))
    assert len(gauss_seidel(g.graph, g.nodes)) == 2
    assert len(gauss_seidel(g.graph, minimal_feedback_order(g.graph, 18))) == 1
    (group,) = per_functional_cycle(g, GaussSeidelMinimal(18))
    assert len(group) == 1
    with pytest.raises(ValueError, match='exact order search'):
        per_functional_cycle(g, GaussSeidelMinimal(limit=1))
    assert per_functional_cycle(g, GaussSeidelMinimal(3)) == per_functional_cycle(g, GaussSeidelMinimal(18))


# ================================================================ per_functional_cycle


def test_bodies_is_the_rule_over_the_function_nodes(sellar, implicit):
    '''`Bodies()` hands the rule the function nodes alone, so what it opens is what
    `check_body_is_dag` forces and nothing an architecture could nest or combine.'''
    assert per_functional_cycle(sellar, Jacobi()) == ({V('y1'): (D2,), V('y2'): (D1,)},)
    assert per_functional_cycle(implicit, Jacobi()) == ()          # all of it is state coupling
    g = graph_of(OrderedMap({**dict(sellar.definitions),
                          N('req'): RelationalCondition((Relation(V('y1'), None, Le),), (), (), ())}))
    assert per_functional_cycle(g, GaussSeidel()) == per_functional_cycle(sellar, GaussSeidel())


def test_a_leaf_rule_takes_one_cycle_as_one_closure(two_cycles):
    '''`per_functional_cycle` is where the cycles live; a rule answers per cycle.'''
    assert [set(g) for g in per_functional_cycle(two_cycles, Jacobi())] == [{V('a1'), V('a2')},
                                                                        {V('b1'), V('b2')}]


# ================================================================ closing

def test_a_closing_makes_one_closure_of_one_group():
    op = CLOSING((D1, D2), {V('y1'): (D2,), V('y2'): (D1,)})
    assert op == FixedPointCut((Cut(V('y1'), (D2,), HAT), Cut(V('y2'), (D1,), HAT)), MINT_MDA(D1))
    assert repr(CLOSING) == 'closing(^copy, at_first_member(^closure))'


def test_closures_records_a_rule(sellar):
    '''Replayed on a model whose coupling changed, it derives its cuts again -- where a
    `Cut` names its readers and would refuse.'''
    op = CloseFunctionalCycles(GaussSeidel(), CLOSING)
    other = graph_of(OrderedMap({**dict(sellar.definitions), D2: call(['z', 'y1', 'w'], ['y2']),
                              N('W'): call(['y2'], ['w'])}))
    assert [len(c.cuts) for c in op.composed(sellar)] == [1]
    assert [len(c.cuts) for c in op.composed(other)]  == [2]
    g = (Plan(sellar) + op).graph
    assert names(g.nodes) == names(sellar.nodes) + ['^closure.D1']
    assert g[Y2_MDA].unknowns == (HAT(V('y2')),)
    assert repr(op) == 'close_functional_cycles(gauss_seidel)'


def test_a_closure_is_named_after_its_cycle_not_its_rule(sellar):
    '''Whichever reader a rule cuts first, the closure binds at the cycle's first member.'''
    for rule in (GaussSeidel(), Jacobi(), GaussSeidelOrdered((D2, D1))):
        (op,) = CloseFunctionalCycles(rule, CLOSING).composed(sellar)
        assert op.at == Y2_MDA


def test_two_groups_on_one_cycle_are_refused_by_name():
    '''A rule closing one cycle as several groups: `AtFirstMember` would bind them all at
    the cycle's first member.'''
    class OnePerVariable(CutRule):
        def to_cuts(self, graph):
            return tuple({v: r} for v, r in coupling_reads(graph).items())
    g = graph_of(OrderedMap({
        D1: call(['a', 'b'], ['c']),
        D2: call(['c'], ['a', 'b']),
    }))
    assert len(per_functional_cycle(g, OnePerVariable())) == 3
    with pytest.raises(ValueError, match='two closures bound at') as caught:
        closed(cuts_per_cycle(g, OnePerVariable()), CLOSING)
    # the name bound twice, once
    assert f"{(str(MINT_MDA(D1)),)!r}" in str(caught.value)
