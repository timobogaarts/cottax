'''
A graph driven through `driven/`, realised through `realisation/` and run by
`execution`, with the default places the surface's `assign` fills in.

The model is one implicit discipline, exact to
`u = (k + sqrt(k^2 + 4k)) / 2` with `k = 1 + x` -- and the Newton is a contract taking a
start per unknown and a step budget per problem and reporting the steps it took,
implemented here in plain Python. Its data arrives keyed by role, and its places are the
ones `Assign` recorded, so a rename moves a start and the Newton never notices.
'''

import math

import pytest
from typing import Any

from jax.tree_util import DictKey

from cottax.core import Function, GraphWithValues
from cottax.core.maths.plan import Plan, Rename
from cottax.consumers.execution import CallableGraph, run as run_graph
from cottax.consumers.execution.interface import DriverImplementation
import dataclasses

from cottax.assemblies.jax import absolute
from cottax.consumers.surface import RolePlaces
from graphs import graph_from_pytree
import purposes
from cottax.core.frozen import OrderedMap
from cottax.pytree.path import NodePath, VarPath
from cottax.core.driven import DriverContract, PerProblem, PerUnknown
from cottax.core.realisation import ImplementedFunction
from cottax.relational import Eq, Relation, RelationalCondition, is_root_find


def v(name: str) -> VarPath:
    return VarPath((DictKey(name),))


X, K, U, Y, R_U, F = (v(s) for s in ('x', 'k', 'u', 'y', 'r_u', 'f'))
CLOSE = NodePath((DictKey('DiscA.close'),))


def call(reads, owns, fn) -> ImplementedFunction:
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), fn)


def model():
    '''Mesh -> [DiscA, DiscA.close, DiscB] -> Objective, the middle block coupled.'''
    return graph_from_pytree({
        'Mesh'       : call([X],       [K],   lambda x: 1.0 + x),
        'DiscA'      : call([U, K, Y], [R_U], lambda u, k, y: u * u - k * y),
        'DiscA.close': RelationalCondition((Relation(R_U, None, Eq),), (), (U,), ()),
        'DiscB'      : call([U],       [Y],   lambda u: 1.0 + u),
        'Objective'  : call([U, Y],    [F],   lambda u, y: u + y),
    })


def exact_u(x: float) -> float:
    k = 1.0 + x
    return (k + math.sqrt(k * k + 4.0 * k)) / 2.0


# ---------------------------------------------------------------- the roles and the driver

class Start(PerUnknown):
    '''Where each unknown starts.'''

class Budget(PerProblem):
    '''How many steps the solve may take.'''

class Steps(PerProblem):
    '''How many it took.'''


class Newton(DriverContract):
    '''A root find from a start, within a step budget, reporting the steps it took: the
    contract, and nothing that runs.'''

    accepts = staticmethod(is_root_find)
    takes   = (Start, Budget)
    reports = (Steps,)


class ScalarNewton(DriverImplementation[Newton]):
    '''`Newton` in plain Python: scalar, finite differences, on the one condition held
    against zero.'''

    needs    = frozenset({Start, Budget})
    provides = frozenset({Steps})

    def __call__(self, posed, context, data) -> tuple[Any, ...]:
        (x,), (budget,) = data[Start], data[Budget]
        gap = lambda x: posed.conditions((x,), context)[0]
        for step in range(budget):
            r = gap(x)
            if abs(r) < 1e-13:
                break
            h = 1e-7 * (abs(x) + 1.0)
            x -= r * h / (gap(x + h) - r)
        return (x,), {Steps: (step,)}


newton = ScalarNewton()


ROLE_PLACES = RolePlaces(OrderedMap(((Start, purposes.Start()),
                                     (Budget, purposes.Budget()),
                                     (Steps, purposes.Steps()))))
'''This driver's roles, each at its purpose: none is named after its role unless said so.'''

surface = dataclasses.replace(absolute, role_places=ROLE_PLACES)
assign, implement = surface.assign, surface.implement

START  = purposes.Start()(U)
BUDGET = purposes.Budget()(v('DiscA.close'))
STEPS  = purposes.Steps()(v('DiscA.close'))


def run(graph, given) -> OrderedMap:
    return run_graph(GraphWithValues(CallableGraph(graph), OrderedMap(given)))


def driven(implementation=newton, *after) -> Plan:
    '''The model, `Newton` assigned at its defaults and implemented by `implementation`.'''
    plan = Plan(model()) + assign(CLOSE, Newton())
    for op in (*after, implement({CLOSE: implementation})):
        plan = plan + op
    return plan


# ---------------------------------------------------------------- the run

def test_role_places_are_minted_off_each_unknown_and_off_the_problem():
    assert ROLE_PLACES(CLOSE, (U,), Newton()) == (((START,), (BUDGET,)), ((STEPS,),))


def test_a_role_with_no_rule_is_refused_by_name():
    from cottax.consumers.surface.refusals import NoPlaceFor
    with pytest.raises(KeyError, match='Budget has no place'):
        RolePlaces(OrderedMap(((Start, purposes.Start()),)))(CLOSE, (U,), Newton())


def test_a_role_takes_its_nearest_ancestors_rule():
    class Warm(Start):
        '''A start of its own kind.'''
    assert ROLE_PLACES.rule(Warm) == purposes.Start()


def test_a_start_renamed_onto_a_model_is_still_the_start():
    '''A warm start from another node's output: one place moved, the role untouched --
    the Newton reads `Start` and never learns where it came from.'''
    graph = driven(newton, Rename(CLOSE, ((START, K),))).graph
    assert graph[CLOSE].data[0] == (K,)
    out = run(graph, {X: 1.0, BUDGET: 50})
    assert math.isclose(out[U], exact_u(1.0), rel_tol=1e-12)


def test_a_realised_graph_is_drawn_by_its_statement():
    '''`driven.Driven` forwards no relation, so a drawing asks the problem.'''
    from cottax.consumers.visualization.kinds import problem_type_of
    graph = driven().graph
    assert problem_type_of(graph[CLOSE]) == problem_type_of(model()[CLOSE]) != 'stated'
