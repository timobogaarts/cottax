'''
`plan/`: the ops on the bindings, and the plan as a value.

The named rewrites built out of these are `test_rewrites.py`. Names are built by hand from
key entries, as in `test_algebra.py` and `test_xdsm.py`.
'''

import pytest
from jax.tree_util import GetAttrKey

from cottax.core.frozen import OrderedMap
from cottax.core.plan import Composite, ComposedOp, Cut, Delete, Edit, GraphOp, Insert, Plan
from cottax.pytree.path import NodePath, VarPath
from cottax.core import FunctionalNode, Graph
from cottax.core.maths.executable import Executable
from cottax.core.maths.graph import NotBound
from cottax.core.maths.valid import Valid
from cottax.core.realisation import implemented
from graphs import graph_of


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


Z, M, Y1, Y2, F, C, R = (V(s) for s in 'z M y1 y2 f c r'.split())


def call(reads, owns):
    return implemented(FunctionalNode(tuple(reads), tuple(owns)), lambda *a: None)


def source(var):
    return call([], [var])


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
    return graph_of(OrderedMap({
        N('d1') : call([Y2], [Y1]),
        N('d2') : call([Y1], [Y2]),
        N('obj'): call([Y2], [F]),
    }))


# ---------------------------------------------------------------- node set

def test_delete_leaves_what_a_survivor_read_unowned(sellar):
    g = Delete((N('d1'),)).apply(sellar)
    assert names(g) == ['.con', '.d2', '.obj', '.z']
    assert Y1 in g.boundary_inputs


def test_insert_refuses_a_name_or_an_output_already_taken(looped):
    with pytest.raises(ValueError, match='already bound'):
        Insert(OrderedMap({N('d1'): source(R)})).apply(looped)
    with pytest.raises(ValueError, match='owned by both'):                    # where `Valid` is kept
        Plan(looped, keeps=Valid) + Insert(OrderedMap({N('other'): source(Y1)}))


# ---------------------------------------------------------------- composed

class Retype(ComposedOp):
    '''`d2` replaced by a node owning the same variable: the insert is well-formed only
    against the graph the delete leaves.'''

    def composed(self, graph):
        return (
            Delete((N('d2'),)),
            Insert(OrderedMap({N('d2b'): call([Y1, Z], [Y2])})),
        )


def test_each_op_is_checked_against_the_graph_it_will_meet(looped):
    '''The insert alone would be refused -- `y2` is produced -- and is not, because the
    delete has already been applied by the time it is checked.'''
    with pytest.raises(ValueError, match='owned by both'):
        Plan(looped, keeps=Valid) + Insert(OrderedMap({N('d2b'): call([Y1, Z], [Y2])}))
    assert names((Plan(looped, keeps=Valid) + Retype()).graph) == ['.d1', '.d2b', '.obj']


def test_a_composed_op_is_one_op_in_the_plan(looped):
    plan = looped + Retype()
    assert plan.ops == (Retype(),)
    assert plan.graphs[0] == looped


# ---------------------------------------------------------------- the plan

def test_a_plan_is_its_prefixes(looped):
    drop, add = Delete((N('obj'),)), Insert(OrderedMap({N('con'): call([Y2], [C])}))
    plan = looped + drop + add
    assert plan.graph == add.apply(drop.apply(looped))
    assert plan.graphs == (looped, drop.apply(looped), add.apply(drop.apply(looped)))


def test_a_plan_checks_at_append_time(looped):
    with pytest.raises(KeyError, match='not a node'):
        looped + Delete((N('nope'),))
    with pytest.raises(TypeError, match='cannot add'):
        Plan(looped) + N('d1')
    assert not isinstance(N('d1'), GraphOp)


def test_a_definition_that_cannot_rename_itself_cannot_exist():
    """
    `relabel` is abstract, not a walk over the fields.

    A generic walker's one failure mode is the worst available: a shape it does not
    recognise is *skipped*, so a `Cut` past such a node renames nothing and says nothing
    -- a wrong answer rather than an error. That bug shipped once here, for a field
    holding a tuple of definitions. Asked as an abstract method, it is unrepresentable.
    """
    from cottax.core import Condition

    class Forgetful(Condition):
        inputs  : tuple = ()
        outputs : tuple = ()

    with pytest.raises(TypeError,
                       match="'owns_inside', 'owns_outside', 'reads_inside', 'reads_outside', 'relabel'"):
        Forgetful()


# ---------------------------------------------------------------- an op's change, as data

def test_an_op_gives_its_change_as_edits_and_the_graph_they_leave(looped):
    add, drop = Insert(OrderedMap({N('d0'): source(R)})), Delete((N('obj'),))
    graph, edits = add.apply(looped), add.edits(looped)
    assert edits == (Edit(insert=add.nodes),) and graph == add.apply(looped)

    class Both(Composite):
        def _ops(self, graph): return (add, drop)
    graph, edits = Both().apply(looped), Both().edits(looped)
    assert edits == (*add.edits(looped), *drop.edits(add.apply(looped)))
    replayed = looped
    for edit in edits:
        replayed = edit.applied(replayed)
    assert replayed == graph == Both().apply(looped)


def test_an_edit_refuses_a_graph_it_does_not_fit(looped):
    with pytest.raises(KeyError, match='not a node of this graph') as caught:
        Edit(remove=frozenset({N('nope'), N('gone')})).applied(looped)
    assert [type(v) for v in caught.value.violations] == [NotBound, NotBound]   # every misfit


def test_a_name_an_op_looks_up_is_refused_by_the_lookup_noted_with_the_op(looped):
    class Both(Composite):
        def _ops(self, graph): return (Delete((N('obj'),)), Cut(Y1, (N('nope'),), lambda v: V('y1h')))
    with pytest.raises(KeyError, match='not a node of this graph') as caught:
        Both().apply(looped)
    assert [type(v) for v in caught.value.violations] == [NotBound]
    assert caught.value.__notes__[-1].startswith('refused by Both')             # outermost last
    assert caught.value.__notes__[0].startswith('refused by cut(')


def test_an_op_needs_of_its_input_what_its_checks_assume(looped):
    twice = Graph(looped.definitions | {N('d3'): source(Y1)}, OrderedMap())   # Y1 owned twice
    assert Delete((N('d3'),)).apply(twice) == looped             # needs nothing: it repairs
    with pytest.raises(ValueError, match='owned by both'):
        Cut(Y1, (N('d2'),), lambda v: v).apply(twice)            # needs `Valid`: refused first


def test_a_graph_need_not_be_valid_between_ops_and_a_plan_keeps_what_it_asks(looped):
    twice = Insert(OrderedMap({N('d3'): source(Y1)}))            # Y1 owned twice: not `Valid`
    assert (Plan(looped) + twice + Delete((N('d3'),))).graph == looped     # keeps nothing by default
    with pytest.raises(ValueError, match='owned by both'):
        Plan(looped, keeps=Valid) + twice
    add = Insert(OrderedMap({N('d0'): source(R)}))
    Plan(looped, keeps=Valid) + add
    with pytest.raises(ValueError):
        Plan(looped, keeps=Executable) + add                     # the loop has no driver
