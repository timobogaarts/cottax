'''
`ExecutableGraph`: the proof between a graph and a run -- a graph executable at every depth.

Names are built by hand from key entries, as everywhere below the declaration surface.
'''

import pytest
from jax.tree_util import GetAttrKey

from cottax.core.maths import steps as stepped
from cottax.relational import Min, Objective
from cottax.core.maths.executable import ExecutableGraph
from cottax.core.maths.steps import SteppedExecutableGraph
from cottax.core.driven import Assign, DriverContract
from cottax.consumers.execution import CallableGraph, run
from cottax.consumers.execution.interface import DriverImplementation
from cottax.core.realisation import Implement, ImplementDrivers, ImplementedFunction, is_implemented
from cottax.consumers.visualization.sequencing import Run, Solve, entries
from cottax.consumers.visualization.kinds import problem_type_of
from cottax.core.maths.plan import Nest, Plan
from cottax.core.frozen import OrderedMap
from cottax.relational import (
    Eq, Le, Relation, RelationalCondition, is_fixed_point, is_root_find,
)
from cottax.core import Combine, GraphWithValues
from cottax.pytree.path import NodePath, VarPath
from cottax.core import Function, problems
import purposes
from graphs import graph_of
from cottax.core.maths.nesting import NestingGraph

HAT, PROBLEM = purposes.Copy(), purposes.Solve()


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def call(reads, owns, fn=lambda *a: None):
    return ImplementedFunction(
        Function(reads=tuple(V(r) for r in reads), owns=tuple(V(o) for o in owns)), fn)


def problem(reads, owns):
    '''A root find: one relation per condition, each against zero.'''
    return RelationalCondition(tuple(Relation(V(r), None, Eq) for r in reads),
                               (), tuple(V(o) for o in owns), ())


def fixed_point(conditions, unknowns):
    '''`u = g(u)` per pair, in the order given.'''
    return RelationalCondition(
        tuple(Relation(V(u), V(c), Eq) for u, c in zip(unknowns, conditions)),
        (), tuple(V(u) for u in unknowns), ())


def names(paths):
    return [p.spelling for p in paths]


def blocks(g):
    '''Each entry's nodes, spelt -- of a graph, or of a blocking's graph.'''
    g = g.graph if isinstance(g, ExecutableGraph) else g
    return [names(e.nodes) for e in entries(g)]


@pytest.fixture
def sellar():
    '''`d1` and `d2` are coupled, closed by the fixed point `mda` on `y2`; `obj` and
    `con` hang off them. A blocking is executable by construction, so the cycle carries
    its problem.'''
    return graph_of(OrderedMap({
        N('z')  : call([], ['z']),
        N('d1') : call(['z', 'hy2'], ['y1']),
        N('d2') : call(['z', 'y1'], ['y2']),
        N('mda'): fixed_point(['y2'], ['hy2']),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
        N('con'): call(['y1'], ['c']),
    }))


@pytest.fixture
def uncut():
    '''`d1` and `d2` coupled with no problem declared: a graph no blocking exists for.'''
    return graph_of(OrderedMap({
        N('z')  : call([], ['z']),
        N('d1') : call(['z', 'y2'], ['y1']),
        N('d2') : call(['z', 'y1'], ['y2']),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
        N('con'): call(['y1'], ['c']),
    }))


@pytest.fixture
def two_solves():
    '''Two coupled pairs in sequence, with a glue node between them.'''
    return graph_of(OrderedMap({
        N('a1')  : call(['x', 'ha2'], ['a1']),
        N('a2')  : call(['a1'],       ['a2']),
        N('pa')  : fixed_point(['a2'], ['ha2']),
        N('glue'): call(['a1'],       ['g']),
        N('b1')  : call(['g', 'hb2'], ['b1']),
        N('b2')  : call(['b1'],       ['b2']),
        N('pb')  : fixed_point(['b2'], ['hb2']),
    }))


# ---------------------------------------------------------------- the default

def test_the_entries_are_the_components(sellar):
    assert blocks(ExecutableGraph(sellar)) == blocks(sellar) == [
        ['.z'], ['.d1', '.d2', '.mda'], ['.obj'], ['.con'],
    ]


# ---------------------------------------------------------------- the steps

def every_interior(graph):
    '''Every interior at every depth, top first.'''
    for step in graph.steps:
        if isinstance(step, stepped.Solve):
            yield step.interior
            yield from every_interior(step.interior)


def test_the_steps_are_a_run_per_node_and_a_solve_per_cycle(sellar):
    steps = SteppedExecutableGraph(sellar).steps
    assert [type(s) for s in steps] == [stepped.Run, stepped.Solve, stepped.Run, stepped.Run]
    assert [s.node for s in steps if isinstance(s, stepped.Run)] == [N('z'), N('obj'), N('con')]
    solve = steps[1]
    assert solve.problem == N('mda')
    assert solve.interior.nodes == (N('d1'), N('d2'))
    assert [type(s) for s in solve.interior.steps] == [stepped.Run, stepped.Run]


def test_a_nested_problem_is_a_solve_inside_the_interior(mdf):
    solve, = SteppedExecutableGraph((Plan(mdf) + Nest(MDA, N('opt'))).graph).steps
    assert solve.problem == N('opt')
    inner = [s for s in solve.interior.steps if isinstance(s, stepped.Solve)]
    assert [s.problem for s in inner] == [MDA]
    assert [s.node for s in inner[0].interior.steps] == [N('D1'), N('D2')]


def test_every_interior_at_every_depth_is_the_same_proof(sellar, two_solves, mdf):
    class Stronger(SteppedExecutableGraph):
        checks_node = ()
    mda, = [n for n in problems(mdf.definitions) if n != N('opt')]
    for graph in (sellar, two_solves, (Plan(mdf) + Nest(mda, N('opt'))).graph,
                  (Plan(mdf) + Nest(N('opt'), mda)).graph):
        for level in (SteppedExecutableGraph, Stronger):
            interiors = list(every_interior(level(graph)))
            assert interiors and all(type(i) is level for i in interiors)


def test_a_level_below_says_something_of_each_node_and_nothing_else():
    with pytest.raises(TypeError, match='declares `checks`'):
        class WholeGraph(SteppedExecutableGraph):
            checks = (lambda graph: (),)


# ---------------------------------------------------------------- into a run

@pytest.fixture
def two_problems():
    '''Two independent root-finds and an objective reading both.'''
    return graph_of(OrderedMap({
        N('fa')  : call(['ua'], ['ra'], lambda u: u - 1.0),
        N('ha')  : problem(['ra'], ['ua']),
        N('fb')  : call(['ub'], ['rb'], lambda u: u - 2.0),
        N('hb')  : problem(['rb'], ['ub']),
        N('obj') : call(['ua', 'ub'], ['f'], lambda a, b: a + b),
    }))


class SelfStarting(DriverContract):
    '''A root find taking no data: the start is the algorithm's own business, so a run
    below needs nothing supplied.'''
    accepts = staticmethod(is_root_find)
    takes   = ()
    reports = ()


class Subtract(DriverImplementation[SelfStarting]):
    '''`r' = 1` for every body below, so plain subtraction converges.'''
    needs    = frozenset()
    provides = frozenset()

    def __call__(self, posed, context, data):
        u = (0.0,) * len(posed.condition.unknowns)
        for _ in range(64):
            r = posed.conditions(u, context)
            u = tuple(ui - ri for ui, ri in zip(u, r))
        return u, {}


def solving(problem, contract=SelfStarting()) -> Assign:
    '''`contract` on `problem`, taking no data and reporting nothing.'''
    return Assign(problem, contract, (), ())


def driven(graph, *ops, implementations=OrderedMap({SelfStarting: Subtract()})) -> NestingGraph:
    '''
    The graph these ops leave, every contract then implemented by kind. A driver is part
    of the problem it answers, so an algorithm is chosen with `Assign` and recorded in
    `Plan.ops` like every other modelling decision -- a run is handed no drivers at all.
    '''
    plan = Plan(graph)
    for op in ops:
        plan = plan + op
    if any(not is_implemented(plan.graph[n]) for n in problems(plan.graph.definitions)):
        plan = plan + ImplementDrivers(implementations)
    return plan.graph


def ran(graph) -> OrderedMap:
    return run(GraphWithValues(CallableGraph(graph), OrderedMap()))


def test_combining_the_problems_is_the_merged_block(two_problems):
    '''The blocking follows the rewrite: one problem, so one component, and one `Solve`
    over both unknowns.'''
    op = Combine(PROBLEM(N('mda')), (N('ha'), N('hb')), RelationalCondition)
    graph = CallableGraph(driven(two_problems, op, solving(op.at)))

    solve, run_ = graph.steps
    assert isinstance(solve, stepped.Solve) and isinstance(run_, stepped.Run)
    assert names(graph[solve.problem].unknowns) == ['.ua', '.ub']
    assert ran(graph.graph)[V('f')] == pytest.approx(3.0)


def test_the_finer_blocking_is_two_solves(two_problems):
    graph = CallableGraph(driven(two_problems, solving(N('ha')), solving(N('hb'))))
    assert [type(s).__name__ for s in graph.steps] == ['Solve', 'Solve', 'Run']
    assert graph.steps is graph.steps                               # derived once
    assert ran(graph.graph)[V('f')] == pytest.approx(3.0)


def test_the_driver_rides_on_the_problem_through_a_re_blocking(two_problems):
    '''A rewrite re-blocks constantly, and a driver is part of the node it answers --
    so there is nothing to resolve onto the new blocks and no position to go stale.'''
    graph  = driven(two_problems, solving(N('ha')), solving(N('hb')))
    graph  = graph_of(OrderedMap({**dict(graph.definitions.items()),
                            N('post'): call(['f'], ['g'], lambda f: 2 * f)}), graph.within)

    assert len(CallableGraph(graph).steps) == 4
    assert ran(graph)[V('g')] == pytest.approx(6.0)


# ------------------------------------------------------- one block, one problem type

def test_a_driver_may_answer_a_whole_family(two_problems):
    class Any_(DriverContract):
        accepts = staticmethod(lambda node: True)
        takes   = ()
        reports = ()

    class AnySubtract(DriverImplementation[Any_]):
        needs    = frozenset()
        provides = frozenset()

        def __call__(self, posed, context, data):
            return Subtract()(posed, context, data)

    graph = driven(two_problems, solving(N('ha'), Any_()), solving(N('hb'), Any_()),
                   implementations=OrderedMap({Any_: AnySubtract()}))
    assert ran(graph)[V('f')] == pytest.approx(3.0)


def test_a_block_of_two_problems_is_refused():
    '''An optimisation around a root-find in one block: the conditions would reach the
    driver as one flat list. `Combine` them, or nest one in the other.'''
    g = graph_of(OrderedMap({
        N('f')  : call(['u', 'd'], ['r'], lambda u, d: u - d),
        N('h')  : problem(['r'], ['u']),
        N('c')  : call(['u'], ['obj'], lambda u: u),
        N('opt'): RelationalCondition((), (Objective(V('obj'), Min),), (V('d'),), ()),
    }))
    with pytest.raises(ValueError, match='declares several problems'):
        CallableGraph(g)


@pytest.fixture
def coupled_problems():
    '''Two `RootFind`s coupled through their bodies, so one SCC holds both.'''
    return graph_of(OrderedMap({
        N('fa'): call(['ua', 'ub'], ['ra'], lambda a, b: a - b),
        N('ha'): problem(['ra'], ['ua']),
        N('fb'): call(['ua', 'ub'], ['rb'], lambda a, b: b - a - 1.0),
        N('hb'): problem(['rb'], ['ub']),
    }))


def test_two_problems_of_one_type_are_refused_too(coupled_problems):
    '''Same type is no less a decision: solving both with one driver is `Combine`, and
    used to happen silently because the unknowns simply concatenated.'''
    with pytest.raises(ValueError, match='declares several problems'):
        ExecutableGraph(coupled_problems)


def test_combining_them_is_how_that_is_said(coupled_problems):
    op    = Combine(PROBLEM(N('mda')), (N('ha'), N('hb')), RelationalCondition)
    graph = CallableGraph(driven(coupled_problems, op, solving(op.at)))
    solve, = graph.steps
    assert names(graph[solve.problem].unknowns) == ['.ua', '.ub']
    assert solve.problem == op.at and problem_type_of(graph[solve.problem]) == 'root-find'


# ---------------------------------------------------------------- the blocking's own view

def test_the_subgraphs_are_one_pass_over_the_parent(two_problems):
    '''
    The blocks partition the nodes, so all of the subgraphs together *are* the parent's
    bindings redistributed -- one walk, in parent order, bucketed by `index`. Asking
    `NestingGraph.subgraph` per block was O(V) each and quadratic over the blocking.

    What is pinned here is the result, not the implementation: every block a graph of its
    own, in the parent's order, and their nodes partitioning the parent's exactly.
    '''
    subs = [e.subgraph for e in entries(two_problems)]

    assert [s.nodes for s in subs] == list(two_problems.graph.components)
    assert [n for s in subs for n in s.nodes] == sorted(
        (n for s in subs for n in s.nodes), key=two_problems.nodes.index)
    assert {n for s in subs for n in s.nodes} == set(two_problems.nodes)


def test_a_driven_block_still_says_what_makes_its_answer_right(two_problems):
    '''The problem is what makes each block right, and the driver is only *how*
    -- so the shape is read off the problem's own relations, which a driver being a
    field beside them leaves exactly where they were.'''
    graph    = driven(two_problems, solving(N('ha')))
    blocking = ExecutableGraph(graph)
    assert problem_type_of(blocking[N('ha')]) == 'root-find'
    assert [e.problem for e in entries(ExecutableGraph(graph.subgraph([N('fa'), N('ha')])).graph)] == [N('ha')]
    assert is_root_find(graph[N('ha')].condition)


def test_a_blocking_is_executable_by_construction(uncut):
    '''`d1` and `d2` are a cycle with no problem in it. It partitions and it orders, but
    there is no answer to what makes its answer right -- so no `ExecutableGraph` exists for it,
    and the refusal is at construction rather than when a run is built. A drawing
    still rings it, walking `entries(graph)`.'''
    with pytest.raises(ValueError, match='declares no problem'):
        ExecutableGraph(uncut)


def test_a_block_still_cyclic_without_its_problem_is_not_drivable():
    '''The trap counting problems used to miss: one problem is declared, so counting them
    said `RootFind` -- but the bodies alone are still a cycle, so no condition map can
    be built and no driver could have answered it.'''
    g = graph_of(OrderedMap({
        N('a') : call(['ua', 'b'], ['ra'], lambda u, b: u - b),
        N('b') : call(['ra'],      ['b'],  lambda r: r),        # closes back on `a`
        N('ha'): problem(['ra'], ['ua']),
    }))
    with pytest.raises(ValueError, match=r'inside .*\.ha.*still cyclic with no problem'):
        ExecutableGraph(g)


# ---------------------------------------------------------------- nesting: derived

def test_entries_are_runs_and_flat_solves_where_nothing_is_nested(two_problems):
    '''One entry per component, parallel to `components`. A
    flat blocking is the no-nesting case, not a different shape: a component nobody
    answers is a `Run`, and a `Solve` with nothing nested iterates its body, whose
    entries are runs.'''
    g = two_problems
    assert len(entries(g)) == len(g.graph.components)
    assert [[x.nodes for x in entries(e.interior)] if isinstance(e, Solve) else None for e in entries(g)] == [
        [(n,) for n in e.nodes if n != e.problem] if isinstance(e, Solve) else None
        for e in entries(g)]


def test_a_proof_reads_as_the_graph_it_proves(two_problems):
    '''A proof adds no field, so it has the graph's whole interface; a restriction
    is not the proven thing, and comes back as the graph.'''
    blocking = ExecutableGraph(two_problems)
    g        = two_problems
    assert blocking.definitions is g.definitions and blocking.nodes == g.nodes
    assert blocking[g.nodes[0]] is g[g.nodes[0]] and g.nodes[0] in blocking
    assert blocking.within is g.within
    assert blocking.outermost_problems == g.outermost_problems
    p = g.outermost_problems[0]
    assert blocking.cycle_of(p) == g.cycle_of(p) and blocking.enclosing(p) == g.enclosing(p)
    assert blocking.interior(p) == g.interior(p) and type(blocking.interior(p)) is type(g)
    assert blocking.subgraph(g.nodes) == g and type(blocking.subgraph(g.nodes)) is type(g)
    with pytest.raises(TypeError):
        iter(blocking)


# ---------------------------------------------------------------- nesting: the invariant

MDA = purposes.Solve()(N('y2'))     # the problem `FixedPointCut` mints below


@pytest.fixture
def mdf():
    '''Sellar under MDF, with the MDA cut so that it declares a problem of its own.

    One SCC holding two problems: the optimiser and the MDA. Which is outer cannot be
    read off the *structure* -- dropping either leaves something drivable -- so it is
    the case nesting exists to state, and `Nest` is where it is stated.
    '''
    from cottax.core.maths.plan import Plan
    from cottax.core.maths.plan import Cut
    from cottax.relational.rewrites import FixedPointCut
    g = graph_of(OrderedMap({
        N('opt'): RelationalCondition((Relation(V('g'), None, Le),),
                                      (Objective(V('f'), Min),), (V('z'),), ()),
        N('D1') : call(['z', 'y2'], ['y1']),
        N('D2') : call(['z', 'y1'], ['y2']),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
        N('con'): call(['y1'], ['g']),
    }))
    return (Plan(g) + FixedPointCut((Cut(V('y2'), (N('D1'),), HAT),),
                                    PROBLEM(N('y2')))).graph


def test_either_problem_may_be_the_outer_one(mdf):
    '''Both orderings are structurally legal, which is why nesting is *stated* -- on the
    graph, by `Nest`, and read from there.'''
    mda, = [n for n in problems(mdf.definitions) if n != N('opt')]

    outer_opt = ExecutableGraph((Plan(mdf) + Nest(mda, N('opt'))).graph)
    outer_mda = ExecutableGraph((Plan(mdf) + Nest(N('opt'), mda)).graph)

    assert blocks(entries(outer_opt.graph)[0].interior) == [['.D1', '.D2', '^solve.y2'], ['.obj'], ['.con']]
    assert blocks(entries(outer_mda.graph)[0].interior) == [['.opt', '.D1', '.D2', '.obj', '.con']]


def test_a_problem_in_no_cycle_of_its_block_cannot_be_answered_there():
    '''The nested mirror of `driven`'s refusal: nothing comes back to it, so nothing
    determines what it owns and no interior could. Asked where answerability is asked.'''
    g = graph_of(OrderedMap({
        N('src'): call([],    ['r']),
        N('h')  : problem(['r'], ['u']),          # nothing comes back to it
        N('k')  : call(['u'], ['w']),
    }))
    with pytest.raises(ValueError, match='in no cycle'):
        ExecutableGraph(g)


# ---------------------------------------------------------------- nesting: `nest`

def test_enclosing_walks_up_the_forest(mdf):
    '''`enclosing(q)` is every problem `q` is nested in, empty before anything is nested.'''
    assert mdf.enclosing(N('opt')) == () and mdf.enclosing(MDA) == ()
    nested = (Plan(mdf) + Nest(MDA, N('opt'))).graph
    assert nested.enclosing(MDA) == (N('opt'),) and nested.enclosing(N('opt')) == ()
    with pytest.raises(ValueError, match='not a problem'):
        nested.enclosing(N('D1'))


def test_a_level_exists_only_where_a_statement_encloses_another(mdf):
    '''
    Two problems are two levels, and no more: `inner` is stated where a problem is
    nested inside another, and nowhere else. A level whose interior holds only function
    nodes needs no entry, because `Drive.body` already *is* the block minus its problem.
    '''
    deep = ExecutableGraph((Plan(mdf) + Nest(MDA, N('opt'))).graph)
    # The interior's own blocking states nothing further: `subgraph` dropped the pair
    # when `opt` left, so `.mda`'s block has no nested problem under it.
    assert all(isinstance(e, Run) for e in entries(entries(entries(deep.graph)[0].interior)[0].interior))   # the MDA's interior runs


def test_a_level_answers_one_problem(mdf):
    # A problem has one enclosing iteration, so `within` cannot key it twice.
    nested = (Plan(mdf) + Nest(MDA, N('opt'))).graph
    mda, = [n for n in problems(nested.definitions) if n != N('opt')]
    with pytest.raises(ValueError, match='already answered inside'):
        Plan(nested) + Nest(mda, N('opt'))


def test_nest_refuses_a_name_that_is_not_here(mdf):
    with pytest.raises(KeyError, match='not a node'):
        ExecutableGraph((Plan(mdf) + Nest(MDA, N('nope'))).graph)


def test_problems_names_what_each_level_answers(mdf):
    nested = ExecutableGraph((Plan(mdf) + Nest(MDA, N('opt'))).graph)
    assert [e.problem.spelling for e in entries(nested.graph)] == ['.opt']
    assert problem_type_of(nested[N('opt')]) == 'optimise'

    interior = entries(nested.graph)[0].interior
    assert [e.problem.spelling if isinstance(e, Solve) else None for e in entries(interior)] \
        == ['^solve.y2', None, None]
    assert problem_type_of(interior[MDA]) == 'fixed-point'


def test_answerability_is_asked_at_every_depth():
    '''An interior that no schedule could be built for is the whole blocking's refusal,
    however deep it sits: every interior is a blocking, checked as it is made.'''
    g = graph_of(OrderedMap({
        N('opt'): RelationalCondition((), (Objective(V('f'), Min),), (V('z'),), ()),
        N('h')  : problem(['r'], ['u']),
        N('a')  : call(['z', 'u', 'b'], ['a']),
        N('b')  : call(['a'],           ['b', 'r']),   # a cycle with no problem in it
        N('obj'): call(['a'],           ['f']),
    }))
    nested = (Plan(g) + Nest(N('h'), N('opt'))).graph
    with pytest.raises(ValueError, match=r'inside .*\.h.*still cyclic with no problem'):
        ExecutableGraph(nested)                                     # the interior is not
    # And as data: `violations` reports it rather than raising -- the rules walk the
    # entries and never build a `ExecutableGraph`.
    from cottax.core import violations
    found = list(violations(ExecutableGraph, nested))
    assert [type(v).__name__ for v in found] == ['StillCyclic']


# ---------------------------------------------------------------- a solve is its cycle

def test_a_condition_produced_off_the_cycle_is_a_constant():
    '''With the invariant above, "inside the block" is "on the cycle" -- so a condition
    a run block produces cannot depend on the unknowns, and is refused as a constant.'''
    g = graph_of(OrderedMap({
        N('a'): call(['u'], ['ra']),
        N('p'): problem(['ra', 'x'], ['u']),
        N('c'): call([], ['x']),
    }))
    with pytest.raises(ValueError, match="a constant is not a condition"):
        ExecutableGraph(Assign(N('p'), SelfStarting(), (), ())(g))


def test_a_parameter_is_a_read_held_fixed_off_the_cycle():
    '''A term the problem declares a parameter is no condition: produced off the cycle it
    is held fixed, and produced on it the proof refuses it, as it refuses driver data.'''
    def with_limit(limit_from):
        return graph_of(OrderedMap({
            N('a'): call(['u'], ['ra']),
            N('p'): RelationalCondition((Relation(V('ra'), V('x'), Eq),), (), (V('u'),),
                                        (V('x'),)),
            N('c'): call(limit_from, ['x']),
        }))
    ExecutableGraph(Assign(N('p'), SelfStarting(), (), ())(with_limit([])))
    with pytest.raises(ValueError, match='produced inside the block it drives'):
        ExecutableGraph(Assign(N('p'), SelfStarting(), (), ())(with_limit(['u'])))


# ---------------------------------------------------------------- the proof a run takes

def test_a_callable_graph_is_an_executable_one_with_more_proved(two_problems, coupled_problems):
    '''A proof strengthens by subclassing: the executable checks run first, and the
    stronger proof is the weaker one wherever that is asked.'''
    g = driven(two_problems, solving(N('ha')), solving(N('hb')))
    assert isinstance(CallableGraph(g), ExecutableGraph)
    assert CallableGraph(g) != ExecutableGraph(g)                 # a different proof
    with pytest.raises(ValueError, match='no driver'):              # undriven
        CallableGraph(two_problems)
    with pytest.raises(ValueError, match='several problems'):       # executable's, first
        CallableGraph(coupled_problems)


def test_a_declared_function_is_executable_but_not_callable(two_problems):
    '''A bare `Function` is drawn, rewritten and proved executable; only running it is
    refused, and the refusal names `Implement`.'''
    g = graph_of(OrderedMap({**dict(two_problems.definitions.items()),
                          N('post'): Function((V('f'),), (V('g'),))}))
    ExecutableGraph(g)
    with pytest.raises(ValueError, match=r'\.post.*`Implement` it'):
        CallableGraph(driven(g, solving(N('ha')), solving(N('hb'))))


def test_implement_fills_the_slot_the_proof_refuses(two_problems):
    '''`Implement` wraps the declaration, so the same graph proves callable.'''
    g = graph_of(OrderedMap({**dict(two_problems.definitions.items()),
                          N('post'): Function((V('f'),), (V('g'),))}))
    g = driven(g, solving(N('ha')), solving(N('hb')),
               Implement(OrderedMap({N('post'): lambda f: 2 * f})))
    assert ran(g)[V('g')] == pytest.approx(6.0)


# ---------------------------------------------------------------- a fact is computed once

def test_proving_a_graph_condenses_each_block_a_bounded_number_of_times():
    '''A block's graph is held by the blocking and read by every rule, so its components
    condense once however many rules ask. Linear in the blocks -- 77 before that.'''
    import networkx as nx
    from cottax.core.maths.plan import Cut
    from cottax.relational.rewrites import FixedPointCut
    from cottax.relational import is_optimise

    class Opt(DriverContract):
        accepts = staticmethod(is_optimise); takes = (); reports = ()

    class Iterate(DriverContract):
        accepts = staticmethod(is_fixed_point); takes = (); reports = ()

    g = graph_of(OrderedMap({
        N('opt'): RelationalCondition((), (Objective(V('f'), Min),), (V('z'),), ()),
        N('D1') : call(['z', 'y2'], ['y1']),
        N('D2') : call(['z', 'y1'], ['y2']),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
    }))
    g = (Plan(g) + FixedPointCut((Cut(V('y2'), (N('D1'),), HAT),),
                                 PROBLEM(N('y2'))) + Nest(MDA, N('opt'))
         + solving(N('opt'), Opt()) + solving(purposes.Solve()(N('y2')), Iterate())).graph

    count = 0
    real  = nx.condensation
    def counting(*a, **k):
        nonlocal count; count += 1
        return real(*a, **k)
    nx.condensation = counting
    try:
        steps = SteppedExecutableGraph(g).steps
    finally:
        nx.condensation = real

    # Every entry at every depth, the flat solve's interior included.
    def every(graph, depth=0):
        for entry in entries(graph):
            yield depth, entry
            if isinstance(entry, Solve):
                yield from every(entry.interior, depth + 1)
    found  = list(every(g))
    depths = {d for d, _ in found}
    assert len(steps) == 1 and (len(depths), len(found)) == (3, 5)
    # A constant per entry and per level is the claim.
    assert count <= 4 * len(found) + 4 * len(depths), (count, found, depths)
