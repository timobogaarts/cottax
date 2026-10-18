'''
`plan.py`: the primitive ops, and the plan as a value.

The named rewrites built out of these are `test_rewrites.py`. Names are built by hand from
key entries, as in `test_algebra.py` and `test_xdsm.py`.
'''

import pytest
from jax.tree_util import GetAttrKey

from cottax.core.frozen import OrderedMap
from cottax.core.maths.plan import ComposedOp, Delete, GraphOp, Insert, Plan
from cottax.pytree.path import NodePath, VarPath
from cottax.core import Function
from cottax.core.realisation import ImplementedFunction
from graphs import graph_of


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


Z, M, Y1, Y2, F, C, R = (V(s) for s in 'z M y1 y2 f c r'.split())


def call(reads, owns):
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), lambda *a: None)


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
    g = Delete((N('d1'),))(sellar)
    assert names(g) == ['.con', '.d2', '.obj', '.z']
    assert Y1 in g.graph.boundary_inputs


def test_insert_refuses_a_name_or_an_output_already_taken(looped):
    with pytest.raises(ValueError, match='already bound'):
        Insert(OrderedMap({N('d1'): source(R)}))(looped)
    with pytest.raises(ValueError, match='owned by both'):
        Insert(OrderedMap({N('other'): source(Y1)}))(looped)


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
        Insert(OrderedMap({N('d2b'): call([Y1, Z], [Y2])}))(looped)
    assert names(Retype()(looped)) == ['.d1', '.d2b', '.obj']


def test_a_composed_op_is_one_op_in_the_plan(looped):
    plan = looped + Retype()
    assert plan.ops == (Retype(),)
    assert plan.trunk.graph == looped


# ---------------------------------------------------------------- the plan

def test_a_plan_is_its_prefixes(looped):
    drop, add = Delete((N('obj'),)), Insert(OrderedMap({N('con'): call([Y2], [C])}))
    plan = looped + drop + add
    assert plan.graph == add(drop(looped))
    assert plan.trunk.trunk.graph == looped
    assert plan.trunk.trunk.trunk is None


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
    from cottax.core import ConditionalNode

    class Forgetful(ConditionalNode):
        inputs  : tuple = ()
        outputs : tuple = ()

    with pytest.raises(TypeError,
                       match="'conditions', 'determined', 'owns', 'reads', 'relabel', 'unknowns'"):
        Forgetful()
