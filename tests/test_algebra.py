'''
The structural algebra on `NestingGraph`: binding, ordering, condensation, reachability, reduction.

Names are built by hand from key entries, as in `test_paths.py`: the core takes a
`VarPath` or a `NodePath` and lifts nothing, so a test that wrote `'d1'` would be
testing a convenience layer that does not live here.
'''

import jax
import pytest
from jax.tree_util import GetAttrKey

from cottax.core.frozen import OrderedMap
from cottax.pytree.path import NodePath, VarPath
from cottax.core import Function, body_of, problems
from cottax.core.maths.nesting import BlockTooSmall, NodeInTwoBlocks, check_blocks_partition
from cottax.core.realisation import ImplementedFunction
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.consumers.visualization.sequencing import entries
from graphs import graph_of


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def call(reads, owns):
    return ImplementedFunction(Function(tuple(V(r) for r in reads), tuple(V(o) for o in owns)),
                       lambda *a: None)


def source(var):
    '''A node that reads nothing. Since `InputNode` went, that is all a source is.'''
    return call([], [var])


def names(paths):
    '''Names as they are written -- the queries deal in names, never in nodes.'''
    return [p.spelling for p in paths]


@pytest.fixture
def sellar():
    '''`d1` and `d2` are coupled; `obj` and `con` hang off them.'''
    return graph_of(OrderedMap({
        N('z')  : source('z'),
        N('d1') : call(['z', 'y2'], ['y1']),
        N('d2') : call(['z', 'y1'], ['y2']),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
        N('con'): call(['y1'], ['c']),
    }))


@pytest.fixture
def chain():
    '''`a -> b -> c`, bound out of order, so binding order is not dependency order.'''
    return graph_of(OrderedMap({
        N('c'): call(['b'], ['y']),
        N('b'): call(['a'], ['b']),
        N('a'): source('a'),
    }))


# ---------------------------------------------------------------- binding

def test_a_graph_binds_names_to_nodes(sellar):
    # A node carries no name of its own; the graph is what binds one.
    assert sellar[N('d1')].reads == (V('z'), V('y2'))
    assert not hasattr(sellar[N('d1')], 'name')
    assert N('d1') in sellar and V('d1') not in sellar


def test_a_variable_still_has_exactly_one_producer():
    with pytest.raises(ValueError, match='owned by both'):
        graph_of(OrderedMap({N('a'): source('x'), N('b'): source('x')}))


def nested(reads, owns):
    '''`call`, but each name is a *sequence* of keys, so paths can nest.'''
    return ImplementedFunction(Function(tuple(V(*r) for r in reads), tuple(V(*o) for o in owns)),
                       lambda *a: None)


def test_one_producer_is_by_containment_not_only_by_equality():
    # `.s` already holds `.s.u`'s storage, so two nodes would determine one place. The
    # same violation as owning the same name, one step up the prefix order.
    with pytest.raises(ValueError, match='lies inside'):
        graph_of(OrderedMap({N('a'): nested([], [('s',)]),
                        N('b'): nested([], [('s', 'u')])}))


def test_a_read_inside_an_output_is_refused():
    # Reads and owns are matched by equality, so this would silently be a boundary input
    # and `b`'s dependency on `a` would vanish rather than be refused. Projection is
    # what would make it legal, and it is not implemented.
    with pytest.raises(ValueError, match='would silently be a boundary input'):
        graph_of(OrderedMap({N('a'): nested([], [('s',)]),
                        N('b'): nested([('s', 'u')], [('y',)])}))


def test_an_output_inside_a_read_is_refused():
    # The mirror: `b` reads `.s` wholesale, so it would silently miss `.s.u`.
    with pytest.raises(ValueError, match='would silently miss this output'):
        graph_of(OrderedMap({N('a'): nested([], [('s', 'u')]),
                        N('b'): nested([('s',)], [('y',)])}))


def test_resolving_an_unknown_name_says_so(sellar):
    with pytest.raises(KeyError, match='not a node of this graph'):
        sellar[N('nope')]


def test_a_graph_is_not_a_mapping(sellar):
    # `nodes` is the sequence and `definitions` the mapping, so an iterable graph would
    # have to choose between them.
    with pytest.raises(TypeError):
        iter(sellar)


# ---------------------------------------------------------------- the binding

def test_a_graph_is_bound_in_the_order_written(chain):
    # `chain` is bound c, b, a -- so binding order is not dependency order, and it is
    # binding order the queries report in.
    assert list(chain.nodes) == [N('c'), N('b'), N('a')]


def test_a_graph_is_ordered_in_equality():
    # It must be: the order decides what every query reports.
    assert (graph_of(OrderedMap({N('a'): source('x'), N('b'): source('y')}))
            != graph_of(OrderedMap({N('b'): source('y'), N('a'): source('x')})))


def test_a_node_name_is_structure_not_a_leaf(sellar):
    # Keys are static, so they ride in the treedef and jax never sorts them.
    leaves = jax.tree_util.tree_leaves(sellar.definitions)
    assert not any(isinstance(leaf, NodePath) for leaf in leaves)
    assert repr(N('d1')) in str(jax.tree_util.tree_structure(sellar.definitions))


def test_a_graph_is_hashable(sellar):
    # A graph is a jit cache key.
    assert hash(sellar) == hash(graph_of(OrderedMap(dict(sellar.definitions))))


# ---------------------------------------------------------------- ordering

def test_topological_order_is_dependency_order(chain):
    assert names(chain.graph.topological_order) == ['.a', '.b', '.c']


def test_topological_order_raises_on_a_cycle_by_name(sellar):
    with pytest.raises(ValueError, match=r"\(NodePath\(\.d1\), NodePath\(\.d2\)\)"):
        sellar.graph.topological_order


def test_condensation_groups_the_cycle(sellar):
    assert [names(c) for c in sellar.graph.components] == [
        ['.z'], ['.d1', '.d2'], ['.obj'], ['.con']]


def test_condensation_of_an_acyclic_graph_is_a_topological_order(chain):
    assert chain.graph.is_acyclic
    assert [names(c) for c in chain.graph.components] == [['.a'], ['.b'], ['.c']]


def test_run_order_is_a_query(sellar):
    # Every node once, in dependency order -- what a drawing puts on its diagonal
    # (`sequencing.ordered_graph` rebinds a graph in it); the graph itself keeps binding order.
    assert names(sellar.graph.components_order) == ['.z', '.d1', '.d2', '.obj', '.con']


# ---------------------------------------------------------------- coupling

def test_cycles_are_the_coupled_components(sellar):
    assert [names(c) for c in sellar.graph.cycles] == [['.d1', '.d2']]
    assert not sellar.graph.is_acyclic


def test_a_node_may_not_read_what_it_owns():
    # One name would mean the value assumed and the value produced at once. A fixed
    # point is written by minting the assumed copy, which is what `Cut` does -- so a
    # cycle is always at least two nodes.
    with pytest.raises(ValueError, match='which it also owns'):
        call(['a', 'x'], ['x'])


# ---------------------------------------------------------------- reachability

def test_reach_is_transitive(chain):
    assert names(chain.graph.reach([V('a')])) == ['.c', '.b']       # binding order


def test_owning_a_varying_variable_does_not_dirty_the_owner(chain):
    # The driven case: `b`'s value changes for reasons outside `b`, so `b` itself
    # need not re-run -- only what reads it.
    assert names(chain.graph.reach([V('b')])) == ['.c']


def test_reach_reaches_round_a_cycle(sellar):
    # `d1` owns y1 and so is not dirtied directly, but `d2` reads y1 and `d1` reads
    # what `d2` owns.
    assert names(sellar.graph.reach([V('y1')])) == ['.d1', '.d2', '.obj', '.con']


def test_reach_rejects_a_name_that_is_not_a_variable(sellar):
    with pytest.raises(KeyError, match='nope'):
        sellar.graph.reach([V('nope')])


def test_ancestors_and_descendants_are_inclusive(sellar):
    assert names(sellar.graph.ancestors([N('obj')])) == ['.z', '.d1', '.d2', '.obj']
    assert names(sellar.graph.descendants([N('d1')])) == ['.d1', '.d2', '.obj', '.con']


def test_the_two_namespaces_do_not_meet_in_a_query(sellar):
    # A name that is not one is simply not in the graph, and no query may be lenient
    # about which namespace it was handed.
    with pytest.raises(KeyError, match='not node'):
        sellar.graph.ancestors([V('obj')])
    with pytest.raises(KeyError, match='not variable'):
        sellar.graph.reach([N('y1')])


# ---------------------------------------------------------------- reduction

def test_subgraph_leaves_severed_reads_as_unowned_inputs(sellar):
    sub = sellar.subgraph([N('obj')])
    assert set(sub.graph.boundary_inputs) == {V('z'), V('y1'), V('y2')}
    assert sub.graph.boundary_outputs == (V('f'),)


def test_subgraph_rejects_an_unknown_node(sellar):
    with pytest.raises(KeyError, match='nope'):
        sellar.subgraph([N('nope')])


def test_a_subgraph_is_canonical_whatever_order_it_is_asked_for(sellar):
    '''
    A subgraph of a *set* is one graph. Load-bearing twice: every query tiebreaks by
    `nodes`, and `OrderedMap` equality tracks the treedef, which carries the key order -- so
    two orderings of one set would compare unequal, retrace separately, and
    `tree_unflatten` across them would silently mismap values onto names.

    Asked for scrambled, since the ordered case is the one the callers happen to hit and
    would not catch a regression here.
    '''
    want = [N('d1'), N('obj'), N('con')]                 # in `sellar`'s binding order
    for order in ([N('obj'), N('con'), N('d1')], [N('con'), N('d1'), N('obj')]):
        sub = sellar.subgraph(order)
        assert sub.nodes == tuple(want)                  # the parent's order, not this one
        assert sub == sellar.subgraph(want)              # and therefore one graph


def test_the_levels_are_the_components_with_nothing_nested(sellar):
    # Uncut, the coupled pair has no problem and so no blocking -- and no interior
    # anywhere: nothing is answered at any block's level, so nothing iterates.
    assert [names(e.nodes) for e in entries(sellar)] == [['.z'], ['.d1', '.d2'], ['.obj'], ['.con']]
    assert [type(e).__name__ for e in entries(sellar)] == ['Run', 'Cycle', 'Run', 'Run']


def test_the_body_drops_the_problem_and_leaves_what_it_owned_unowned():
    # The unknown becomes a boundary input, which is what a residual guess is.
    problem = RelationalCondition((Relation(V('r_u'), None, Eq),), (), (V('u'),), ())
    g = graph_of(OrderedMap({N('a'): call(['u'], ['r_u']), N('close'): problem}))
    assert names(problems(g.definitions)) == ['.close']
    assert names(body_of(g).nodes) == ['.a']
    assert body_of(g).graph.boundary_inputs == (V('u'),)


def test_closing_readers_is_the_set_that_closes_the_loop(sellar):
    assert names(sellar.graph.closing_readers(V('y1'))) == ['.d2']
    assert sellar.graph.closing_readers(V('f')) == ()


def test_feedback_variables_are_the_ones_a_cut_could_name(sellar):
    # `y1` and `y2` close the d1/d2 loop; `z` feeds it and `f` hangs off it.
    assert sellar.graph.feedback_variables == (V('y1'), V('y2'))


def test_an_acyclic_graph_has_no_feedback_variables(chain):
    assert chain.graph.feedback_variables == ()


def test_a_feedback_variable_is_one_with_closing_readers(sellar):
    # The two halves of a cut: which variables, and for one of them which readers.
    assert set(sellar.graph.feedback_variables) == {
        var for var in sellar.graph.owned_variables if sellar.graph.closing_readers(var)
    }


def test_closing_readers_needs_a_producer(sellar):
    with pytest.raises(KeyError, match='no producer'):
        sellar.graph.closing_readers(V('z2'))


# ---------------------------------------------------------------- contraction

def test_contract_is_the_quotient_by_the_blocks(sellar):
    '''One node per block, reading what the block reads from outside and owning what
    it owns, bound at the first member's place; the other members gone.'''
    g = sellar.contract(((N('d2'), N('d1')),))
    assert g.nodes == (N('z'), N('d1'), N('obj'), N('con'))
    assert g[N('d1')] == Function((V('z'),), (V('y1'), V('y2')))     # y1 / y2 internal, not read
    assert g.graph.is_acyclic
    assert sellar.contract(()) == sellar


def test_contract_coarsens_and_never_refines(sellar):
    '''A path through a box may appear where none was: `obj` and `con` in one box
    puts `con` downstream of `y2`, which it never read.'''
    g = sellar.contract(((N('obj'), N('con')),))
    assert g[N('obj')] == Function((V('z'), V('y1'), V('y2')), (V('f'), V('c')))
    assert set(g.graph.descendants([N('d2')])) >= {N('obj')}
    assert set(sellar.graph.descendants([N('d2')])) >= {N('obj'), N('con')}


def test_contract_refuses_a_short_block_a_node_in_two_blocks_and_an_unknown_one(sellar):
    '''A block of one is the identity or a change of kind, neither a view's to make;
    `Combine` refuses the same.'''
    for short in ((N('d1'),), (N('d1'), N('d1')), ()):
        assert list(check_blocks_partition(sellar, (short,))) == [BlockTooSmall(short)]
        with pytest.raises(ValueError, match='is not two nodes'):
            sellar.contract((short,))
    twice = ((N('d1'), N('d2')), (N('d2'), N('obj')))
    assert list(check_blocks_partition(sellar, twice)) == [NodeInTwoBlocks((N('d2'),))]
    with pytest.raises(ValueError, match='each in two blocks'):
        sellar.contract(twice)
    with pytest.raises(KeyError, match='nope'):
        sellar.contract(((N('d1'), N('nope')),))
