'''
`PytreeStructure`: a caller's pytree as a data structure, the one registered for `VarPath`,
a model run on it through the surface's `run(..., read_from=)` and written back.
'''

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import DictKey

from cottax.core import Function
from cottax.core.frozen import OrderedMap
from cottax.core.driven import Driven
from cottax.relational.contracts import RootFinding
from cottax.core.driven import Start
import cottax.assemblies.jax as cx
from cottax.consumers.execution import CallableGraph
from cottax.consumers.execution.refusals import ForeignPlace, ValuedTwice
from cottax.consumers.execution.interface import DataStructure
from cottax.core import answer_for_places
from cottax.pytree.consumers.execution import PytreeStructure, completed
import purposes
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedDriven, ImplementedFunction
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.relational.consumers.execution.jax_solvers import Newton
from graphs import graph_of


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def V(*names: str) -> VarPath:
    return VarPath(tuple(DictKey(n) for n in names))


def N(name: str) -> NodePath:
    return NodePath((DictKey(name),))


START = purposes.Start()


def square(start=START(V('u'))) -> CallableGraph:
    '''`u * u == z`, a root find from a start at `start`.'''
    condition = RelationalCondition((Relation(V('r'), None, Eq),), (), (V('u'),), ())
    return CallableGraph(graph_of({
        N('res'): ImplementedFunction(Function((V('u'), V('z')), (V('r'),)),
                                      lambda u, z: u * u - z),
        N('h')  : ImplementedDriven(Driven(condition, RootFinding(steps=False),
                                           ((start,),), ()),
                                    Newton(rtol=1e-10, atol=1e-10)),
    }))


def on_tree(graph, tree, values=OrderedMap()):
    '''The caller's tree: each input and each start read there, run, and written back.'''
    return cx.flat.run(graph, values, read_from={cx.Inputs: tree, Start: tree}).into(tree)


def test_a_pytree_reads_and_writes_by_place():
    tree = {'u': 1.0, 'z': 9.0, 'r': None}
    assert PytreeStructure().read(tree, (V('z'), V('u'))) == OrderedMap({V('z'): 9.0, V('u'): 1.0})
    assert PytreeStructure().write(tree, {V('r'): 2.0}) == {'u': 1.0, 'z': 9.0, 'r': 2.0}


def test_the_pytree_structure_is_registered_for_a_var_path():
    assert type(answer_for_places(DataStructure, square().graph)) is PytreeStructure


def test_a_place_holding_none_reads_none_and_one_not_held_is_refused():
    assert PytreeStructure().read({'u': None}, (V('u'),)) == OrderedMap({V('u'): None})
    with pytest.raises(KeyError, match='names no place'):
        PytreeStructure().read({'u': 1.0}, (V('w'),))


@dataclasses.dataclass(frozen=True)
class Lookalike:
    '''Not a `VarPath`, with the segments of one.'''
    segments : tuple


def test_a_lookalike_of_a_var_path_is_refused_by_name():
    '''Read at the `VarPath` of its keys, it would read `a` before the run computes it.'''
    a     = Lookalike((DictKey('a'),))
    graph = CallableGraph(graph_of({
        N('g'): ImplementedFunction(Function((V('c'),), (V('a'),)), lambda c: 10 * c),
        N('f'): ImplementedFunction(Function((a,), (V('b'),)), lambda a: a + 1),
    }))
    with pytest.raises(TypeError) as refused:
        on_tree(graph, {'a': 1.0, 'b': 0.0, 'c': 2.0})
    assert refused.value.args[0] == ForeignPlace(
        (a,), ('test_pytree_structure.Lookalike',), 'PytreeStructure', 'VarPath').message
    for asked in (lambda s: s.holds({}, a), lambda s: s.read({'a': 1.0}, (a,)),
                  lambda s: s.write({'a': 1.0}, {a: 2.0})):
        with pytest.raises(TypeError, match='indexes a VarPath alone'):
            asked(PytreeStructure())


def test_a_place_is_held_where_its_key_path_resolves_and_a_minted_one_never():
    assert PytreeStructure().holds({'u': 1.0}, V('u'))
    assert PytreeStructure().holds({'u': None}, V('u'))
    assert not PytreeStructure().holds({'u': 1.0}, V('w'))
    assert not PytreeStructure().holds({'u': 1.0}, START(V('u')))


def test_the_tree_is_run_and_what_it_computes_written_back():
    out = on_tree(square(), {'u': 1.0, 'z': 9.0, 'r': None})
    assert out['u'] == pytest.approx(3.0, rel=1e-10)
    assert out['r'] == pytest.approx(0.0, abs=1e-10)
    assert out['z'] == 9.0


def test_a_start_given_is_used_where_the_tree_holds_none():
    out = on_tree(square(), {'u': None, 'z': 9.0, 'r': None}, {Start: {V('u'): -1.0}})
    assert out['u'] == pytest.approx(-3.0, rel=1e-10)


def test_a_start_given_by_name_and_read_from_the_tree_is_given_twice():
    with pytest.raises(ValueError) as refused:
        on_tree(square(), {'u': 1.0, 'z': 9.0, 'r': None}, {Start: {V('u'): -1.0}})
    assert refused.value.violation == ValuedTwice(
        START(V('u')), ("values[Start][VarPath(['u'])]", "read_from[Start] at VarPath(['u'])"))


def test_a_per_problem_role_or_one_no_driver_takes_is_refused_from_a_structure():
    from cottax.consumers.execution.refusals import NoneTakes, NoSlotFor
    from cottax.core.driven import Steps
    with pytest.raises(TypeError) as refused:
        cx.flat.valued(square(), read_from={Steps: {}})
    assert refused.value.violation == NoSlotFor('Steps')
    with pytest.raises(KeyError) as refused:
        cx.flat.valued(square(), read_from={cx.Trajectory: {}})
    assert refused.value.violation == NoneTakes('Trajectory')


def test_what_the_tree_gives_is_valued_where_it_was_read():
    tree   = {'u': 1.0, 'z': 9.0, 'r': None}
    valued = cx.flat.valued(square(), read_from={cx.Inputs: tree, Start: tree})
    assert valued.values == OrderedMap({V('z'): 9.0, START(V('u')): 1.0})


def test_a_missing_start_is_refused_by_name():
    with pytest.raises(KeyError, match=r"values\[Start\]` at VarPath\(\['u'\]\)"):
        on_tree(square(), {'u': None, 'z': 9.0, 'r': None})


def test_a_tree_completed_holds_a_slot_for_what_the_run_computes():
    tree = completed(square(), {'u': 1.0, 'z': 9.0})
    assert tree == {'u': 1.0, 'z': 9.0, 'r': None}             # no slot at the minted start
    assert on_tree(square(), tree)['u'] == pytest.approx(3.0, rel=1e-10)
    assert completed(square().graph, tree) == tree             # a graph, or a proof of one


def test_completing_invents_no_start():
    with pytest.raises(KeyError, match=r"values\[Start\]` at VarPath\(\['u'\]\)"):
        on_tree(square(), completed(square(), {'z': 9.0}))


def test_the_boundary_runs_under_jit_and_traces_once():
    # The graph is a static key; the tree's values are traced, so new values reuse it.
    traces = []

    @eqx.filter_jit
    def go(graph, tree):
        traces.append(1)
        return on_tree(graph, tree)

    graph = square()                       # one graph: a body built twice is two keys
    for z in (4.0, 9.0):
        out = go(graph, {'u': jnp.asarray(1.0), 'z': jnp.asarray(z),
                            'r': jnp.asarray(0.0)})
        assert float(out['u']) == pytest.approx(z ** 0.5, rel=1e-10)
    assert len(traces) == 1
