'''
The design of experiments, a third language beside `relational` and `ode`: its condition node,
proved by core with nothing edited for it, and the contract answering it.
'''

import pytest
from jax.tree_util import GetAttrKey

from cottax.core import ExecutableGraph, FunctionalNode, violations
from cottax.core.driven import Assign, Driven, DrivenGraph
from cottax.core.driven.nodes import NotAccepted
from cottax.core.refusal_kinds import Constant, OwnReadInsideCycle, ReadInsideCycle
from cottax.doe import DoeCondition, Recorded, Varied
from cottax.doe.contracts import Sampling
from cottax.pytree.path import NodePath, VarPath
from cottax.relational import Eq, Relation, RelationalCondition
from graphs import graph_of


def V(key: str) -> VarPath:
    return VarPath((GetAttrKey(key),))

def N(key: str) -> NodePath:
    return NodePath((GetAttrKey(key),))


DOE = DoeCondition((Varied(V('x'), V('P')),), (Recorded(V('h'), V('H')),))
'''`x` set to each point of `P`, `h` recorded into `H`.'''


def model(**nodes):
    '''The DOE over `h = F(x)`, `nodes` binding more or replacing one by name.'''
    return graph_of({N('doe'): DOE, N('F'): FunctionalNode((V('x'),), (V('h'),)),
                     **{N(k): d for k, d in nodes.items()}})


def test_a_doe_owns_its_factors_inside_and_its_records_outside():
    assert (DOE.owns_inside, DOE.reads_inside, DOE.reads_outside, DOE.owns_outside) == (
        (V('x'),), (V('h'),), (V('P'),), (V('H'),))
    g = ExecutableGraph(model())
    assert g.enclosers == (N('doe'),) and g.cycle_of(N('doe')) == (N('doe'), N('F'))


@pytest.mark.parametrize('nodes, kind', [
    ({'doe': DoeCondition(DOE.factors, (*DOE.responses, Recorded(V('c'), V('C')))),
      'c': FunctionalNode((), (V('c'),))}, Constant),             # a response no factor moves
    ({'p': FunctionalNode((V('x'),), (V('P'),))}, ReadInsideCycle),       # a plan on the cycle
    ({'F': FunctionalNode((V('x'), V('H')), (V('h'),))}, OwnReadInsideCycle),  # a record on it
], ids=['constant response', 'plan on the cycle', 'record on the cycle'])
def test_core_refuses_a_doe_by_its_own_rules(nodes, kind):
    assert [type(v) for v in violations(ExecutableGraph, model(**nodes))] == [kind]


def test_sampling_answers_a_doe_and_nothing_else():
    DrivenGraph(Assign(N('doe'), Sampling(), (), ())(model()))
    root_find = RelationalCondition((Relation(V('h'), None, Eq),), (), (V('x'),), ())
    assert [type(v) for v in violations(Driven, root_find, Sampling(), (), ())] == [NotAccepted]


def test_a_relabelling_moves_every_place_of_a_doe():
    assert DOE.relabel({V(k): V(k + '2') for k in 'xPhH'}) == DoeCondition(
        (Varied(V('x2'), V('P2')),), (Recorded(V('h2'), V('H2')),))
