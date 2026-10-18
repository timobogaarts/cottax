'''
`execution.jax_jit.JitGraph`: every node a jit cache key -- hashable, and comparing by value
-- at every depth a solver is handed.
'''

import dataclasses
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from cottax.core import GraphWithValues
from cottax.core.driven import Start
from cottax.core.frozen import OrderedMap, violations
from cottax.core.maths.steps import Solve
from cottax.consumers.execution.interface_types import Posed
import cottax.assemblies.jax as cx
from cottax.consumers.execution import run, write_back
from cottax.consumers.execution.run import InteriorConditionMap
from cottax.consumers.execution.refusals import BodyByIdentity, PartialBody, Unhashable
from cottax.consumers.execution.jax_jit import JitGraph
from graphs import graph_of
from test_execution_jax import NESTED, SQUARE, N, V, call


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def graph_with(body):
    '''`SQUARE`'s graph, its first body replaced.'''
    return graph_of(dict(SQUARE.graph.definitions) | {N('mid'): call([V('u')], [V('w')], body)})


def kinds(graph) -> list[type]:
    return [type(v) for v in violations(JitGraph, graph)]


class Scaled(eqx.Module):
    '''A callable carrying an array: the value lives in the graph.'''
    factor : jnp.ndarray

    def __call__(self, u):
        return u * self.factor


class Times:
    def by(self, u):
        return u * 1.0


def test_a_graph_of_module_functions_and_frozen_solvers_is_a_jit_graph():
    assert kinds(SQUARE.graph) == []
    assert kinds(NESTED.graph) == []


def test_an_array_in_the_graph_is_refused_naming_its_leaves():
    found = list(violations(JitGraph, graph_with(Scaled(jnp.asarray(2.0)))))
    assert [type(v) for v in found] == [Unhashable]
    assert found[0].count == 1 and 'Array' in found[0].leaves
    with pytest.raises(TypeError, match='cannot be a jit cache key'):
        JitGraph(graph_with(Scaled(jnp.asarray(2.0))))


def test_an_array_in_a_solver_setting_is_named_by_its_leaf():
    '''The leaves are found under every field, a solver's settings included.'''
    h     = SQUARE.graph[N('h')]
    held  = dataclasses.replace(h, implementation=dataclasses.replace(
        h.implementation, atol=jnp.asarray(1e-8)))
    found = list(violations(JitGraph, graph_of(dict(SQUARE.graph.definitions) | {N('h'): held})))
    assert [type(v) for v in found] == [Unhashable]
    assert found[0].count == 1 and 'Array' in found[0].leaves


def test_a_bound_method_is_refused():
    assert kinds(graph_with(Times().by)) == [BodyByIdentity]


def test_a_partial_is_refused():
    assert kinds(graph_with(functools.partial(lambda u, k: u * k, k=1.0))) == [PartialBody]


def test_what_a_solver_is_posed_hashes_at_every_depth():
    # A solver compiles keyed on its `Posed`, whose condition map holds the interior.
    graph     = JitGraph(NESTED.graph)
    (outer,)  = [s for s in graph.steps if isinstance(s, Solve)]
    (inner,)  = [s for s in outer.interior.steps if isinstance(s, Solve)]
    for step, holder in ((outer, graph), (inner, outer.interior)):
        assert isinstance(step.interior, JitGraph)
        node = holder[step.problem]
        conditions = InteriorConditionMap.posing(node, step.interior)
        hash(Posed(node.condition, node.contract.reports, conditions,
                   conditions.context_places))


def test_a_jit_graph_runs_on_a_structure():
    tree = {'z': 9.0, 'u': None, 'w': None, 'r': None}
    graph = JitGraph(SQUARE.graph)
    out  = cx.flat.run(graph, {Start: {V('u'): 1.0}}, read_from={cx.Inputs: tree}).into(tree)
    assert out['u'] == pytest.approx(3.0, rel=1e-10)


def test_a_run_on_a_structure_is_read_outside_a_jit_run_inside_and_written_outside():
    '''The data is read and written outside the trace; the values cross it as arrays, so one
    trace serves every tree.'''
    graph  = JitGraph(SQUARE.graph)
    traces = []

    @eqx.filter_jit
    def go(graph, values):
        traces.append(1)
        return run(GraphWithValues(graph, values))

    for z in (4.0, 9.0, 16.0):
        tree   = {'z': z, 'u': 1.0, 'w': None, 'r': None}
        valued = cx.flat.valued(graph, read_from={cx.Inputs: tree, Start: tree})
        values = go(valued.graph, OrderedMap((p, jnp.asarray(v)) for p, v in valued.values.items()))
        out    = write_back(graph.graph, tree, values)
        assert out['u'] == pytest.approx(z ** 0.5, rel=1e-10) and out['z'] == z
    assert len(traces) == 1
