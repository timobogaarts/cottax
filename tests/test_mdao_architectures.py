'''
A `Monolithic` architecture is the requirements absorbed, the cycles under its top cut
and closed, then its final's own combining. Every shipped final reaches an executable
graph on every fixture it is meant for, a study's own final is a few lines on a level,
and MDF and IDF driven agree. Fixtures and helpers are `test_cutting.py`'s.
'''

import pytest

from cottax.core import Function
from cottax.core.maths.executable import ExecutableGraph
from cottax.core.maths.plan import Combine, Nest, Plan
from cottax.core.frozen import OrderedMap, violations
from cottax.core.realisation import ImplementedFunction
from cottax.consumers.architectures import (
    IDF, MDA, MDF, Analysis, Architecture, Bodies, Boxes, Coupled, Disciplines, IndividualFeasible,
    Monolithic, MultidisciplinaryFeasible, OnCycle, Simultaneous, Solves,
)
from cottax.consumers.architectures.architectures import body_of_problem, cuts_per_cycle_boxed, per_functional_cycle_boxed
import purposes
from cottax.pytree.path import NodePath, VarPath
from cottax.relational import Eq, Le, Relation, RelationalCondition, is_optimise
from cottax.core.maths.cut_rules import (
    ByName, CutRule, GaussSeidel, GaussSeidelMinimal, Jacobi, coupling_reads, per_functional_cycle,
)
from cottax.consumers.architectures import CloseFunctionalCycles, closed
from cottax.relational.rewrites import FixedPointCut
from cottax.relational.consumers.architectures import FixedPointClosing
from cottax.consumers.architectures.interface_types import At
from cottax.consumers.visualization.sequencing import Run, Solve, entries
from test_cutting import *
from test_cutting import sellar, implicit, two_cycles, through_explicit     # the fixtures, by name
from graphs import graph_of


# ================================================================ boxes

def test_a_sweep_over_bodies_does_not_depend_on_where_a_problem_is_bound(implicit):
    '''Raw, a state coupling is owned by the *problem*, so the cut set depends on where
    a problem happens to be declared. Over the bodies there are no problems to order,
    and over the boxes neither.'''
    orders = ([D1, N('h1'), D2, N('h2')], [D1, D2, N('h1'), N('h2')],
              [N('h1'), N('h2'), D1, D2])
    rebound = [graph_of(OrderedMap({k: implicit[k] for k in o})).subgraph(o) for o in orders]
    raw = {frozenset(GaussSeidel().to_cuts(g.graph)[0].items()) for g in rebound}
    assert len(raw) == 3
    assert [per_functional_cycle(g, GaussSeidel()) for g in rebound] == [()] * 3
    folded = [per_functional_cycle_boxed(g, Solves(H12), GaussSeidel()) for g in rebound]
    assert folded == [({V('y2'): (D1,)},)] * 3


def test_a_box_is_a_problem_with_its_body(implicit):
    '''The owners of the problem's conditions and nothing else: no block is computed.
    The contraction is `NestingGraph.contract`'s, bound at the block's first member in
    binding order, the body here.'''
    assert body_of_problem(implicit, N('h1')) == (D1,)
    folded = implicit.contract(((N('h1'), D1), (N('h2'), D2)))
    assert folded[D1] == Function((V('z'), V('y2')), (V('r1'), V('y1')))
    assert folded[D2] == Function((V('z'), V('y1')), (V('r2'), V('y2')))
    assert N('h1') not in folded and folded.nodes == (OPT, D1, D2, N('obj'), N('con'))


def test_a_cut_on_a_box_lands_on_the_body_inside(implicit):
    assert per_functional_cycle_boxed(implicit, Solves(H12), Jacobi()) == \
        ({V('y1'): (D2,), V('y2'): (D1,)},)


def test_disciplines_is_every_problem_of_the_graph_handed(implicit, through_explicit):
    '''Under an architecture the graph handed is the top's interior, so this is every
    implicit model beneath the top. `hA` is a box even though its loop runs through
    `A2`: the box is `A1` with `hA`, `A2` stays outside, and the loop is still cut.'''
    assert Disciplines().folded(implicit.interior(OPT)) == H12
    assert Disciplines().folded(through_explicit.interior(OPT)) == (N('hA'), N('hB'))
    assert per_functional_cycle_boxed(through_explicit.interior(OPT), Disciplines(), GaussSeidel()) == \
        ({V('a2'): (N('A1'),)},)


def test_one_folded_and_one_not_finds_nothing_and_the_proof_says_why(implicit):
    '''The unfolded solve's state is owned by a dropped condition node, so its
    coupling is invisible to the rule; the architecture then leaves two declared
    problems on one cycle, which the proof refuses.'''
    assert per_functional_cycle_boxed(implicit, Solves((N('h1'),)), Jacobi()) == ()

    class Partial(MultidisciplinaryFeasible):
        boxes = Solves((N('h1'),))
    g = (Plan(implicit) + Partial(GaussSeidel(), CLOSING, OPT)).graph
    assert [type(v).__name__ for v in violations(ExecutableGraph, g)] == ['SeveralOutermost']


def test_a_fixed_point_folds_as_a_root_find_does():
    '''A converged solve is a function of its outside reads whatever its shape.'''
    g = graph_of(OrderedMap({
        OPT     : optimise(V('f'), (V('z'),)),
        D1      : call(['z', 'y1', 'y2'], ['v1']),
        N('h1') : fixed_point((V('v1'),), (V('y1'),)),
        D2      : call(['z', 'y1', 'y2'], ['v2']),
        N('h2') : fixed_point((V('v2'),), (V('y2'),)),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
    }))
    assert per_functional_cycle_boxed(g.interior(OPT), Disciplines(), GaussSeidel()) == ({V('y2'): (D1,)},)
    m = (Plan(g) + mdf()).graph
    assert dict(m.within) == {Y2_MDA: OPT, N('h1'): Y2_MDA, N('h2'): Y2_MDA}
    ExecutableGraph(m)
    lone = graph_of(OrderedMap({OPT: optimise(V('f'), (V('z'),)),
                             D1: call(['z', 'u'], ['v']), N('h'): fixed_point((V('v'),), (V('u'),)),
                             N('obj'): call(['z', 'u'], ['f'])}))
    assert per_functional_cycle_boxed(lone.interior(OPT), Disciplines(), GaussSeidel()) == ()


# ================================================================ the one kind of model

def test_a_loop_through_an_explicit_and_a_state_coupling_at_once(sellar):
    '''The case the block-based fold got wrong: `hA`'s body loops through `A2` *and*
    reads `hB`'s state. A box is the body with its problem, so both couplings are
    seen and cut in one closure, and MDF and IDF are executable.'''
    mixed = graph_of(OrderedMap({
        OPT     : optimise(V('f'), (V('z'),)),
        N('A1') : call(['z', 'a2', 'ua', 'ub'], ['ra', 'a1']),
        N('hA') : root_find((V('ra'),), (V('ua'),)),
        N('A2') : call(['a1'], ['a2']),
        N('B1') : call(['ua', 'ub'], ['rb', 'b1']),
        N('hB') : root_find((V('rb'),), (V('ub'),)),
        N('obj'): call(['z', 'b1'], ['f']),
    }))
    (group,) = per_functional_cycle_boxed(mixed.interior(OPT), Disciplines(), GaussSeidel())
    assert group == {V('a2'): (N('A1'),), V('ub'): (N('A1'),)}
    for arch in (mdf(), idf(), sand()):
        ExecutableGraph((Plan(mixed) + arch).graph)
    m = (Plan(mixed) + mdf()).graph
    a = MINT_MDA(N('A1'))
    assert dict(m.within) == {a: OPT, N('hA'): a, N('hB'): a}


def test_a_residual_through_a_chain_is_refused():
    '''Its body does not read its unknown: a chain is not a discipline.'''
    chain = graph_of(OrderedMap({
        OPT: optimise(V('f'), (V('z'),)),
        N('A'): call(['z', 'y'], ['m']), N('B'): call(['m'], ['r']),
        N('h'): root_find((V('r'),), (V('y'),)),
        N('obj'): call(['y'], ['f'])}))
    with pytest.raises(ValueError, match='does not read'):
        Plan(chain) + idf()


def test_two_problems_sharing_a_body_are_refused():
    shared = graph_of(OrderedMap({
        OPT: optimise(V('f'), (V('z'),)),
        N('D'): call(['z', 'y1', 'y2'], ['r1', 'r2']),
        N('h1'): root_find((V('r1'),), (V('y1'),)),
        N('h2'): root_find((V('r2'),), (V('y2'),)),
        N('obj'): call(['y1'], ['f'])}))
    with pytest.raises(ValueError, match='share the body'):
        Plan(shared) + sand()


def test_a_problem_lying_inside_anothers_body_is_nested_in_it():
    '''`h` drives `r2` by `u3` and `H` both `r1` and `r2` by `u1, u2`: `h` and its body
    lie on `H`'s cycle, inside `H`'s body, so it can be placed nowhere else and is nested
    there; with `D3` beside, the bodies only overlap, and that is two solves sharing a body.'''
    from cottax.consumers.architectures.refusals import BodyShared
    from cottax.core.maths.plan import refusals
    model = {OPT: optimise(V('f'), (V('z'),)),
             N('D1'): call(['z', 'u1', 'u2'], ['r1']), N('D2'): call(['u2', 'u3'], ['r2']),
             N('H'): root_find((V('r1'), V('r2')), (V('u1'), V('u2'))),
             N('obj'): call(['u1'], ['f'])}
    inside = graph_of(OrderedMap({**model, N('h'): root_find((V('r2'),), (V('u3'),))}))
    for arch in (mdf(), idf()):
        placed = (Plan(inside) + arch).graph
        assert dict(placed.within) == {N('h'): N('H'), N('H'): OPT}
        ExecutableGraph(placed)
    analysis = inside.subgraph(n for n in inside.nodes if n != OPT)
    assert dict((Plan(analysis) + MDA(GaussSeidel(), CLOSING)).graph.within) == {N('h'): N('H')}
    apart = graph_of(OrderedMap({**model, N('D3'): call(['u3'], ['r3']),
                                 N('h'): root_find((V('r2'), V('r3')), (V('u3'),))}))
    assert [(type(v), v.body) for v in refusals(mdf(), apart)] == [(BodyShared, str(N('D2')))]
    with pytest.raises(ValueError, match='share the body .*: one solve written as two -- `Combine`'):
        Plan(apart) + mdf()


def test_a_condition_owned_by_a_problem_is_refused():
    g = graph_of(OrderedMap({
        OPT: optimise(V('f'), (V('z'),)),
        D1: call(['z', 'u'], ['v']), N('h'): fixed_point((V('v'),), (V('u'),)),
        N('k'): root_find((V('u'),), (V('w'),)),          # a condition that is another problem's unknown
        N('obj'): call(['w'], ['f'])}))
    with pytest.raises(ValueError, match='not owned by a function node'):
        Plan(g) + mdf()


def test_a_problem_of_another_language_is_held_to_its_own_picture():
    '''An ODE is no relational condition: whether it is an implicit model is its own answer
    (a function writes every rate), its body every function on its cycle, so a rate through a
    chain is a model; two sharing a body, neither inside the other, are refused as any.'''
    from cottax.ode import Dt, OdeCondition
    from cottax.ode.consumers.architectures import RateUnwritten
    from cottax.consumers.architectures.refusals import BodyShared
    from cottax.core.maths.plan import refusals
    ode = lambda x, r: OdeCondition((Dt(V(x), V(r), V(f'{x}0')),), V('span'))    # noqa: E731
    top = {OPT: optimise(V('f'), (V('z'),)), N('obj'): call(['z', 'x'], ['f'])}
    off = graph_of(OrderedMap({**top, N('h'): ode('x', 'r')}))
    chain = graph_of(OrderedMap({**top, N('A'): call(['z', 'x'], ['m']), N('B'): call(['m'], ['r']),
                                 N('h'): ode('x', 'r')}))
    shared = graph_of(OrderedMap({**top, N('D'): call(['z', 'x', 'y'], ['r', 's']),
                                  N('h'): ode('x', 'r'), N('k'): ode('y', 's')}))
    for g, kinds in ((off, [RateUnwritten]), (chain, []), (shared, [BodyShared])):
        assert [type(v) for v in refusals(mdf(), g)] == kinds
    with pytest.raises(ValueError, match='neither lies inside the other'):
        Plan(shared) + mdf()


def test_what_is_combined_into_the_top_is_a_relational_condition():
    '''A requirement the design reaches is combined into the top under every final, and
    the top with it: neither may be of another language.'''
    from cottax.ode import Dt, OdeCondition
    from cottax.consumers.architectures.refusals import NotCombinable
    from cottax.core.maths.plan import refusals
    from test_core import States                 # an ODE is never a requirement
    held = graph_of(OrderedMap({OPT: optimise(V('f'), (V('z'),)), N('A'): call(['z'], ['f', 'r']),
                                N('req'): States((V('r'),), ())}))
    assert [(type(v), v.problem) for v in refusals(mdf(), held)] == [(NotCombinable, str(N('req')))]
    top = graph_of(OrderedMap({OPT: OdeCondition((Dt(V('z'), V('r'), V('z0')),), V('span')),
                               N('A'): call(['z'], ['r'])}))
    assert [(type(v), v.problem) for v in refusals(mdf(), top)] == [(NotCombinable, str(OPT))]


def test_a_nested_model_is_refused(implicit):
    '''An architecture places every problem itself, so it starts from a model with no
    nesting: its own output is refused, not re-resolved.'''
    resolved = (Plan(implicit) + mdf()).graph
    for arch in (mdf(), idf(), sand()):
        with pytest.raises(ValueError, match='already nested'):
            Plan(resolved) + arch


# ================================================================ the architectures

def test_mdf_nests_the_coupling_in_the_optimiser(sellar):
    g = (Plan(sellar) + mdf()).graph
    top, = entries(ExecutableGraph(g).graph)
    assert isinstance(top, Solve) and top.problem == OPT
    inner = entries(top.interior)
    assert isinstance(inner[0], Solve) and inner[0].problem == Y2_MDA
    assert blocks(top.interior) == [['.D1', '.D2', '^closure.D1'], ['.obj'], ['.con']]
    assert all(isinstance(e, Run) for e in entries(inner[0].interior))


def test_the_resolution_is_the_derivation_kept(sellar, implicit):
    '''What the op did, as a value: the three stages, the graph between the cutting
    and the placing, and the result -- which is what `apply` returns.'''
    with_req = graph_of(OrderedMap({**dict(sellar.definitions.items()),
                                 N('lim'): call(['y2'], ['m']), N('req'): requirement(V('m'))}))
    r = mdf().resolution(with_req)
    assert r.absorbed  == (Combine(OPT, (OPT, N('req')), RelationalCondition),)
    assert [c.at for c in r.closures] == [Y2_MDA]
    assert r.combining == (Nest(Y2_MDA, OPT),)
    assert r.graph == (Plan(with_req) + mdf()).graph
    assert N('req') not in r.opened and Y2_MDA in r.opened and not r.opened.within
    # the stages applied in order are the op
    step = Plan(with_req)
    for op in (*r.absorbed, *r.outside, *r.closures, *r.combining):
        step = step + op
    assert step.graph == r.graph
    # SAND on implicit disciplines: nothing cut, everything placed
    r = sand().resolution(implicit)
    assert r.closures == () and r.opened == implicit
    assert [type(op).__name__ for op in r.combining] == ['Combine']


def test_the_paper_recipe_is_two_ops(sellar):
    '''On explicit disciplines with no requirement: the closures, then one `Nest` or
    one `Combine`.'''
    opened = Plan(sellar) + CloseFunctionalCycles(GaussSeidel(), CLOSING)
    assert (Plan(sellar) + mdf()).graph  == (opened + Nest(Y2_MDA, OPT)).graph
    assert (Plan(sellar) + idf()).graph  == (opened + Combine(OPT, (OPT, Y2_MDA), RelationalCondition)).graph
    assert (Plan(sellar) + sand()).graph == (Plan(sellar) + idf()).graph
    assert mdf().absorbing(sellar) == ()
    assert mdf().cycles_under(opened.graph, (Y2_MDA,)) == (OnCycle((Y2_MDA,), ()),)
    assert mdf().combining(opened.graph, (Y2_MDA,)) == [Nest(Y2_MDA, OPT)]


def test_every_shipped_final_is_executable_where_it_is_meant(sellar, implicit, through_explicit, two_cycles):
    '''The table: the disciplines under MDF and IDF, the bodies under SAND, every cell
    executable. Neither view alone would do -- `implicit` needs the boxes, and
    `through_explicit` needs `hA` left as bodies.'''
    for g in (sellar, implicit, through_explicit, two_cycles):
        for arch in (mdf(), idf(), sand()):
            ExecutableGraph((Plan(g) + arch).graph)


def test_the_three_on_implicit_disciplines(implicit):
    m = (Plan(implicit) + mdf()).graph
    assert dict(m.within) == {Y2_MDA: OPT, N('h1'): Y2_MDA, N('h2'): Y2_MDA}
    i = (Plan(implicit) + idf()).graph
    assert dict(i.within) == {N('h1'): OPT, N('h2'): OPT}
    assert i[OPT].relations == (Relation(V('g'), None, Le),
                                Relation(HAT(V('y2')), V('y2'), Eq))
    s = (Plan(implicit) + sand()).graph
    assert not s.within and N('h1') not in s and N('h2') not in s
    assert s[OPT].unknowns == (V('z'), V('y1'), V('y2'))
    assert s[OPT].relations == (Relation(V('g'), None, Le),
                                Relation(V('r1'), None, Eq),
                                Relation(V('r2'), None, Eq))
    assert HAT(V('y2')) not in s.graph.variables


def test_a_declared_problem_nests_in_the_mda_on_its_cycle_else_in_the_optimiser(through_explicit):
    g = (Plan(through_explicit) + mdf()).graph
    a = MINT_MDA(N('A1'))
    assert dict(g.within) == {a: OPT, N('hA'): a, N('hB'): OPT}
    assert HAT(V('a2')) in g.graph.variables
    i = (Plan(through_explicit) + idf()).graph
    assert dict(i.within) == {N('hA'): OPT, N('hB'): OPT}


def test_two_cycles_are_two_solves_under_mdf(two_cycles):
    g = (Plan(two_cycles) + mdf()).graph
    top, = entries(ExecutableGraph(g).graph)
    solves = [e for e in entries(top.interior) if isinstance(e, Solve)]
    assert [names(e.nodes) for e in solves] == [['.A1', '.A2', '^closure.A1'], ['.B1', '.B2', '^closure.B1']]


def test_idf_keeps_the_optimiser_where_it_was_bound(sellar):
    g = (Plan(sellar) + idf()).graph
    assert g.nodes[0] == OPT and is_optimise(g[OPT])
    assert g[OPT].unknowns == (V('z'), HAT(V('y2')))
    assert g[OPT].conditions == (V('f'), V('g'), V('y2'))


def test_sand_absorbs_a_declared_fixed_point_as_written():
    g = graph_of(OrderedMap({
        OPT     : optimise(V('f'), (V('z'),)),
        D1      : call(['z', 'u'], ['v']),
        N('h')  : fixed_point((V('v'),), (V('u'),)),
        N('obj'): call(['z', 'u'], ['f']),
    }))
    s = (Plan(g) + sand()).graph
    assert s[OPT].relations == (Relation(V('u'), V('v'), Eq),)
    assert s[OPT].unknowns  == (V('z'), V('u'))
    ExecutableGraph(s)


class AbsorbedMDF(Monolithic):
    '''The other MDF: the full model solved inside each optimiser iteration as one
    root find over the states and the residuals. Same boxes, the declared problems
    folded into their MDA instead of nested in it.'''
    boxes = Disciplines()

    def combining(self, opened, closures):
        ops = []
        for cycle in self.cycles_under(opened, closures):
            if not cycle.closures:
                ops += [Nest(p, self.top) for p in cycle.declared]
                continue
            mda = cycle.closures[0]
            ops += [Combine(mda, (*cycle.closures, *cycle.declared), RelationalCondition), Nest(mda, self.top)]
        return ops


def test_mdf_with_the_models_absorbed_into_the_mda(implicit):
    g = (Plan(implicit) + AbsorbedMDF(GaussSeidel(), CLOSING, OPT)).graph
    assert dict(g.within) == {Y2_MDA: OPT}
    assert N('h1') not in g and N('h2') not in g
    assert g[Y2_MDA].unknowns  == (HAT(V('y2')), V('y1'), V('y2'))
    assert g[Y2_MDA].relations == (Relation(HAT(V('y2')), V('y2'), Eq),
                                   Relation(V('r1'), None, Eq),
                                   Relation(V('r2'), None, Eq))
    ExecutableGraph(g)


class KeepSolves(Monolithic):
    '''A study's own: the named solves kept as solves of their own, nested in the top
    and seen as boxes by the cutting; everything else absorbed. `boxes` bound as a
    property, off a field.'''
    held : tuple[NodePath, ...]

    @property
    def boxes(self) -> Boxes:
        return Solves(self.held)

    def combining(self, opened, closures):
        cycles   = self.cycles_under(opened, closures)
        absorbed = [p for c in cycles for p in (*c.closures, *c.declared) if p not in self.held]
        return (*(Nest(p, self.top) for c in cycles for p in c.declared if p in self.held),
                *((Combine(self.top, (self.top, *absorbed), RelationalCondition),) if absorbed else ()))


def test_a_study_keeps_one_solve_and_absorbs_the_rest(implicit, sellar):
    '''MDF / IDF / SAND are defaults. A large implicit model kept under its own solver
    with the rest simultaneous is one final on `Monolithic`.'''
    g = (Plan(implicit) + KeepSolves(GaussSeidel(), CLOSING, OPT, (N('h1'),))).graph
    assert dict(g.within) == {N('h1'): OPT}
    assert N('h2') not in g
    assert g[OPT].unknowns == (V('z'), V('y2'))
    assert Relation(V('r2'), None, Eq) in g[OPT].relations
    ExecutableGraph(g)
    # a declared fixed point beside sellar's disciplines, kept while the coupling is
    # absorbed
    own = purposes.Solve()(N('con'))
    with_fp = graph_of(OrderedMap({**dict(sellar.definitions),
                                N('con'): call(['y1', 'u'], ['g', 'v']),
                                own: fixed_point((V('v'),), (V('u'),))}))
    k = (Plan(with_fp) + KeepSolves(GaussSeidel(), CLOSING, OPT, (own,))).graph
    assert dict(k.within) == {own: OPT}
    assert HAT(V('y2')) in k[OPT].unknowns and V('u') not in k[OPT].unknowns
    ExecutableGraph(k)


def test_a_requirement_the_design_reaches_is_absorbed(sellar):
    with_req = graph_of(OrderedMap({
        **dict(sellar.definitions.items()),
        N('lim'): call(['y2'], ['m']),
        N('req'): requirement(V('m')),
    }))
    assert mdf().absorbing(with_req) == (Combine(OPT, (OPT, N('req')), RelationalCondition),)
    g = (Plan(with_req) + mdf()).graph
    assert N('req') not in g
    assert Relation(V('m'), None, Le) in g[OPT].relations
    ExecutableGraph(g)


def test_a_requirement_the_design_does_not_reach_is_refused(sellar):
    with_req = graph_of(OrderedMap({
        **dict(sellar.definitions.items()),
        N('lim'): call(['w'], ['m']),
        N('req'): requirement(V('m')),
    }))
    assert mdf().absorbing(with_req) == ()
    g = (Plan(with_req) + mdf()).graph
    assert N('req') in g
    with pytest.raises(ValueError, match='`Determine` it by a variable on its cycle'):
        ExecutableGraph(g)


def test_a_problem_reached_only_through_a_requirement_is_combined():
    g = graph_of(OrderedMap({
        OPT     : optimise(V('f'), (V('x'),)),
        N('obj'): call(['x'], ['f']),
        D1      : call(['x', 'y2'], ['y1']),
        D2      : call(['y1'], ['y2']),
        N('lim'): call(['y1'], ['c']),
        N('req'): requirement(V('c')),
    }))
    assert per_functional_cycle_boxed(g.interior(OPT), Disciplines(), GaussSeidel()) == ()      # off the optimiser's cycle
    out = (Plan(g) + mdf()).graph
    assert N('req') not in out
    assert Relation(V('c'), None, Le) in out[OPT].relations
    assert dict(out.within) == {Y2_MDA: OPT}
    ExecutableGraph(out)


def test_the_combining_is_written_by_hand_the_same_way(implicit):
    (op,) = closed(cuts_per_cycle_boxed(implicit, Solves(H12), GaussSeidel()), CLOSING)
    cut = (Plan(implicit) + op).graph
    assert idf().combining(cut, (Y2_MDA,)) == \
        (Combine(OPT, (OPT, Y2_MDA), RelationalCondition), Nest(N('h1'), OPT), Nest(N('h2'), OPT))
    by_hand = Plan(cut) + Combine(OPT, (OPT, Y2_MDA), RelationalCondition) + Nest(N('h1'), OPT) + Nest(N('h2'), OPT)
    assert by_hand.graph == (Plan(implicit) + idf()).graph


def test_a_problem_the_declaration_surface_minted_is_declared(sellar):
    '''A closure is what the architecture's own cutting bound; a namespace says
    nothing.'''
    own = purposes.Solve()(N('con'))
    g = graph_of(OrderedMap({**dict(sellar.definitions),
                          N('con'): call(['y1', 'u'], ['g', 'v']),
                          own: fixed_point((V('v'),), (V('u'),))}))
    i = (Plan(g) + idf()).graph
    assert dict(i.within) == {own: OPT}
    assert V('u') not in i[OPT].unknowns


def test_a_combining_the_graph_cannot_answer_is_refused_by_the_proof(implicit):
    class Beside(Monolithic):
        boxes = Disciplines()
        def combining(self, opened, closures):
            return [Nest(p, self.top) for c in self.cycles_under(opened, closures)
                    for p in (*c.closures, *c.declared)]
    g = (Plan(implicit) + Beside(GaussSeidel(), CLOSING, OPT)).graph
    with pytest.raises(ValueError):
        ExecutableGraph(g)


def test_top_names_a_bound_problem(sellar):
    with pytest.raises(KeyError, match='is no node of the model'):
        Plan(sellar) + MDF(GaussSeidel(), CLOSING, N('nope'))
    with pytest.raises(ValueError, match='is not a problem'):
        Plan(sellar) + MDF(GaussSeidel(), CLOSING, D1)


def test_a_final_binds_boxes_or_is_abstract():
    class Forgot(Simultaneous):
        pass
    with pytest.raises(TypeError):
        Forgot(GaussSeidel(), CLOSING, OPT)
    for level in (Monolithic, MultidisciplinaryFeasible, IndividualFeasible, Simultaneous):
        with pytest.raises(TypeError):
            level(GaussSeidel(), CLOSING, OPT)
    for level in (Architecture, Coupled, Analysis):
        with pytest.raises(TypeError):
            level(GaussSeidel(), CLOSING)


def test_two_closures_on_one_cycle_are_one_mda_under_mdf(implicit):
    '''A rule closing one cycle as two groups collides under `AtFirstMember`; an `At`
    naming them apart closes both, and MDF makes them one MDA.'''
    class OnePerVariable(CutRule):
        def to_cuts(self, graph):
            return tuple({v: r} for v, r in coupling_reads(graph).items())
    class AtVariable(At):
        def __call__(self, cycle, group):
            return purposes.Closure()(NodePath(next(iter(group)).segments))
    assert len(per_functional_cycle_boxed(implicit.interior(OPT), Disciplines(), OnePerVariable())) == 2
    with pytest.raises(ValueError, match='two closures bound at'):
        Plan(implicit) + mdf(OnePerVariable())
    apart = FixedPointClosing(HAT, AtVariable())
    m = (Plan(implicit) + MDF(OnePerVariable(), apart, OPT)).graph
    mda, = [p for p in m.problems if p not in implicit.problems]
    assert dict(m.within) == {mda: OPT, N('h1'): mda, N('h2'): mda}
    assert set(m[mda].unknowns) == {HAT(V('y1')), HAT(V('y2'))}
    ExecutableGraph(m)
    i = (Plan(implicit) + IDF(OnePerVariable(), apart, OPT)).graph
    assert HAT(V('y1')) in i[OPT].unknowns and HAT(V('y2')) in i[OPT].unknowns
    ExecutableGraph(i)


def test_a_closure_written_before_the_architecture_is_refused(sellar):
    '''A closure is a problem whose copy is read by one node and whose condition is
    owned by another, so it is not an implicit model: an architecture starts from the
    model, and the cutting is its own.'''
    before = Plan(sellar) + CloseFunctionalCycles(GaussSeidel(), CLOSING)
    with pytest.raises(ValueError, match='not an implicit model'):
        before + mdf()


def test_reprs():
    assert repr(mdf()) == 'mdf(disciplines, gauss_seidel, closing(^copy, at_first_member(^closure)), .opt)'
    assert repr(sand(Jacobi())) == 'sand(bodies, jacobi, closing(^copy, at_first_member(^closure)), .opt)'
    assert repr(mdf()) != repr(MDF(GaussSeidel(), FixedPointClosing(purposes.Twin(), CLOSING.at), OPT))
    assert repr(CloseFunctionalCycles(ByName(((D1, Jacobi()),)), CLOSING)) == \
        'close_functional_cycles(by_name(.D1: jacobi))'
    assert repr(GaussSeidelMinimal(18)) == 'gauss_seidel_minimal(limit=18)'
    assert repr(Disciplines()) == 'disciplines' and repr(Bodies()) == 'bodies'
    assert repr(Solves(H12)) == 'solves(.h1, .h2)'


def upstream():
    '''Cycles off the optimiser's: a coupled pair of functions, two implicit models
    coupled through their states, and one implicit model alone, all read by the
    objective and none reading the design.'''
    return graph_of(OrderedMap({
        OPT     : optimise(V('f'), (V('z'),)),
        N('A1') : call(['a2'], ['a1']),
        N('A2') : call(['a1'], ['a2']),
        D1      : call(['y1', 'y2'], ['r1']),
        N('h1') : root_find((V('r1'),), (V('y1'),)),
        D2      : call(['y1', 'y2'], ['r2']),
        N('h2') : root_find((V('r2'),), (V('y2'),)),
        N('L')  : call(['u'], ['r']),
        N('h')  : root_find((V('r'),), (V('u'),)),
        N('obj'): call(['z', 'a1', 'y1', 'u'], ['f']),
    }))


def test_a_cycle_off_the_top_is_resolved_by_an_mda(sellar):
    '''Every cycle off the top's is an MDA's, with the top's rule unless another
    analysis is named: closed, the models nested in their closure, a model alone left
    alone. Under every final, and nothing of it nested in the top.'''
    g = upstream()
    r = mdf().resolution(g)
    assert [op.at for op in r.outside if isinstance(op, FixedPointCut)] == [MINT_MDA(N('A1')), Y2_MDA]
    assert dict(r.graph.within) == {N('h1'): Y2_MDA, N('h2'): Y2_MDA}
    assert r.closures == () and r.combining == ()
    for arch in (mdf(), idf(), sand()):
        ExecutableGraph((Plan(g) + arch).graph)
    assert mdf().resolution(sellar).outside == ()
    named = MDF(GaussSeidel(), CLOSING, OPT, analysis=AbsorbedMDA(GaussSeidel(), CLOSING)).resolution(g)
    assert named.graph[N('h1')].unknowns == (V('y1'), V('y2')) and not named.graph.within
    ExecutableGraph(named.graph)


# ================================================================ an analysis, no top

def without_top(g):
    '''The disciplines alone: the optimiser and what only it reads taken out.'''
    return g.subgraph(n for n in g.nodes if n not in (OPT, N('obj'), N('con')))


def test_mda_is_mdf_without_the_top(implicit, sellar):
    '''The whole graph is the scope, nothing is absorbed, and per cycle the closure
    holds the declared problems -- the MDF's inner MDA on its own.'''
    disciplines = without_top(implicit)
    r = MDA(GaussSeidel(), CLOSING).resolution(disciplines)
    assert r.absorbed == () and [c.at for c in r.closures] == [Y2_MDA]
    assert dict(r.graph.within) == {N('h1'): Y2_MDA, N('h2'): Y2_MDA}
    ExecutableGraph(r.graph)
    explicit = (Plan(without_top(sellar)) + MDA(GaussSeidel(), CLOSING)).graph
    assert explicit.problems == (Y2_MDA,) and not explicit.within
    ExecutableGraph(explicit)
    assert repr(MDA(GaussSeidel(), CLOSING)) == 'mda(disciplines, gauss_seidel, closing(^copy, at_first_member(^closure)))'


def test_the_bare_op_and_the_mda_differ_exactly_on_state_coupling(implicit):
    '''Over the bodies alone the states are the problems' own and nothing is cut, so
    two root finds share one cycle and the proof refuses; the MDA folds them.'''
    disciplines = without_top(implicit)
    bare = (Plan(disciplines) + CloseFunctionalCycles(GaussSeidel(), CLOSING)).graph
    assert bare == disciplines
    assert [type(v).__name__ for v in violations(ExecutableGraph, bare)] == ['SeveralOutermost']
    ExecutableGraph((Plan(disciplines) + MDA(GaussSeidel(), CLOSING)).graph)


class AbsorbedMDA(Coupled):
    '''The SAND-style analysis: per cycle, every problem on it combined into one root
    find over all the states and residuals. Over the bodies, so two implicit models
    coupled through their states need no copy at all.'''
    boxes = Bodies()

    def combining(self, opened, closures):
        ops = []
        for cycle in self.cycles_under(opened, closures):
            on = (*cycle.closures, *cycle.declared)
            if len(on) > 1:
                ops.append(Combine(on[0], on, RelationalCondition))
        return ops


def test_an_absorbed_mda_needs_no_copy(implicit):
    r = AbsorbedMDA(GaussSeidel(), CLOSING).resolution(without_top(implicit))
    assert r.closures == ()
    assert r.graph.problems == (N('h1'),) and N('h2') not in r.graph
    assert r.graph[N('h1')].unknowns == (V('y1'), V('y2'))
    assert HAT(V('y2')) not in r.graph.graph.variables
    ExecutableGraph(r.graph)


def test_an_analysis_refuses_a_design(implicit):
    '''An optimise problem is not an implicit model, and an analysis names no top.'''
    with pytest.raises(ValueError, match='states an objective'):
        Plan(implicit) + MDA(GaussSeidel(), CLOSING)
    with pytest.raises(ValueError, match='states an objective'):    # read without applying
        MDA(GaussSeidel(), CLOSING).resolution(implicit)


def test_two_cycles_are_two_closures_under_mda(two_cycles):
    g = (Plan(without_top(two_cycles)) + MDA(GaussSeidel(), CLOSING)).graph
    assert names(g.problems) == ['^closure.A1', '^closure.B1']
    ExecutableGraph(g)


# ---------------------------------------------------------------- the same optimum

@pytest.mark.slow
def test_mdf_and_idf_reach_the_same_optimum():
    '''The architectures are different derivations of one model: driven, they agree.'''
    pytest.importorskip('scipy')
    import jax
    import jax.numpy as jnp
    from jax.tree_util import DictKey
    from cottax.relational.contracts import (
        ConstrainedMinimisation, ConstrainedMinimiser, FixedPointIteration, FixedPointIterator,
    )
    from cottax.consumers.execution import CallableGraph
    from cottax.assemblies.jax import Start, absolute, flat
    assign = absolute.assign
    from cottax.relational.consumers.execution.jax_solvers import SLSQP, Picard
    from cottax.core.realisation import ImplementDrivers

    def D(name):
        return VarPath((DictKey(name),))

    def body(reads, owns, fn):
        return ImplementedFunction(Function(tuple(D(r) for r in reads),
                                            tuple(D(o) for o in owns)), fn)

    model = graph_of(OrderedMap({
        OPT     : optimise(D('f'), (D('z'),), (D('g'),)),
        D1      : body(['z', 'y2'], ['y1'], lambda z, y2: z + 0.2 * y2),
        D2      : body(['z', 'y1'], ['y2'], lambda z, y1: jnp.sqrt(jnp.abs(y1)) + 0.1 * z),
        N('obj'): body(['z', 'y1', 'y2'], ['f'],
                       lambda z, y1, y2: (z - 2.0) ** 2 + 0.1 * y1 + 0.1 * y2),
        N('con'): body(['z'], ['g'], lambda z: z - 1.0),
    }))
    guess = {Start: {D('z'): jnp.asarray(0.0), absolute.copy(D('y2')): jnp.asarray(1.0)}}
    mda = MINT_MDA(D1)

    optimiser = ConstrainedMinimisation(steps=False)
    backend   = ImplementDrivers(OrderedMap({ConstrainedMinimiser: SLSQP(),
                                      FixedPointIterator: Picard(rtol=1e-12, atol=1e-12)}))

    def assigned(plan, problem, contract):
        return plan + assign(problem, contract)

    def ran(plan):
        # One start for both: under MDF the MDA owns the copy of `y2`, under IDF the top.
        graph = CallableGraph((plan + backend).graph)
        with jax.enable_x64(True):
            return flat.run(graph, guess)

    m = assigned(assigned(Plan(model) + mdf(), OPT, optimiser),
                 mda, FixedPointIteration(steps=False))
    out_mdf = ran(m)
    out_idf = ran(assigned(Plan(model) + idf(), OPT, optimiser))

    assert float(out_mdf['z']) == pytest.approx(1.0, abs=1e-6)
    assert float(out_idf['z']) == pytest.approx(float(out_mdf['z']), abs=1e-6)
    assert float(out_idf['y1']) == pytest.approx(float(out_mdf['y1']), abs=1e-5)
    assert float(out_idf['y2']) == pytest.approx(float(out_mdf['y2']), abs=1e-5)
