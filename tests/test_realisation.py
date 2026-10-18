'''
`driven/` and `realisation/` alone: the driver contract, the driven node, the two
implemented finals and the edits putting each on, over `core` and nothing else.

Everything runs in a toy place of plain strings, with a condition node that holds no relation,
so the two legs are shown generic in the place *and* in the condition node: what `Driven` reads is
`ConditionalNode`'s, and nothing here is jax, a path or a `RelationalCondition`. The toy
place is `test_core`'s written again rather than imported, since that module takes jax.
'''

import copy
import pickle
import inspect
from dataclasses import dataclass

import pytest

from cottax.core import (
    AbstractGraph, ConditionalNode, Function, NestingGraph, Rename, Replace, refusals,
    violations,
)
from cottax.core.maths.definition import relabelled
from cottax.core.frozen import OrderedMap
from cottax.core.driven import (
    Assign, Driven, DrivenGraph, DriverContract, PerProblem, PerUnknown, Role, Undrive,
    is_driven,
)
from cottax.core.driven.nodes import (
    DeterminedWhileDriven, DrivenTwice, NotAccepted, RoleMisplaced, RoleTwice, RoleUnsized,
    SlotsMiscounted,
)
from cottax.core.driven.ops import NotAConditionalNode, NotDriven
from cottax.core.driven.proof import Undriven
from cottax.core.realisation import (
    Implement, ImplementedDriven, ImplementedFunction, RealisedGraph, Unimplement,
    is_implemented,
)
from cottax.core.realisation.proof import NoImplementation
from cottax.core.realisation.implemented import (
    DeterminedWhileImplemented, ImplementedTwice, NoContract, NotAFunction,
    UndrivenWhileImplemented, NodeNamedTwice, NotImplementable, NotImplementedYet,
    NothingImplemented,
)


# ---------------------------------------------------------------- a toy place

@dataclass(frozen=True)
class Word:
    '''A place as a tuple of strings: the whole of `Place`.'''

    segments : tuple[str, ...]

    def __str__(self) -> str:
        return '.'.join(self.segments)


def W(s: str) -> Word:
    return Word(tuple(s.split('.')))


class Holds(ConditionalNode):
    '''A condition node as two bare tuples, no relation anywhere -- and `reads` wider than
    `conditions`, so a wrapper building its reads off the wrong one is caught.'''

    conditions : tuple
    unknowns   : tuple
    extra      : tuple = ()

    @property
    def reads(self):
        return self.conditions + self.extra

    @property
    def owns(self):
        return self.unknowns

    def relabel(self, renaming):
        return Holds(relabelled(self.conditions, renaming),
                     relabelled(self.unknowns, renaming), relabelled(self.extra, renaming))

    def determined(self, unknowns):
        return Holds(self.conditions, tuple(unknowns), self.extra)


# ---------------------------------------------------------------- toy roles and drivers

class Start(PerUnknown):
    '''Where each unknown starts.'''

class Budget(PerProblem):
    '''How many steps the solve may take.'''

class Steps(PerProblem):
    '''How many it took.'''


def single(statement) -> bool:
    '''What `Solo` answers: exactly one unknown.'''
    return len(statement.unknowns) == 1


class Solo(DriverContract):
    '''A contract and nothing else: what it accepts and the roles it plays.'''
    accepts = staticmethod(single)
    takes   = (Start, Budget)
    reports = (Steps,)


class Bare(DriverContract):
    '''No roles: a driver that adds no edge.'''
    accepts = staticmethod(lambda statement: True)
    takes   = ()
    reports = ()


class Twice(DriverContract):
    '''A role listed twice: a run hands data over by role, so one slot would be lost.'''
    accepts = staticmethod(lambda statement: True)
    takes   = (Start, Start)
    reports = ()


class Unsized(DriverContract):
    '''A role that is neither per unknown nor per problem.'''
    accepts = staticmethod(lambda statement: True)
    takes   = (Role,)
    reports = ()


u, v, r, k, g = W('u'), W('v'), W('r'), W('k'), W('g')
NONE = ((), ())


def placed(unknowns=(u,)) -> tuple:
    '''Where `Solo`'s roles sit for a problem over `unknowns`: its data, its reports.'''
    return ((tuple(W(f'^start.{x}') for x in unknowns), (W('^budget.solve'),)),
            ((W('^steps.solve'),),))


def graph():
    '''A body owning `r` from `u`, and the problem `r` holds over `u`.'''
    return NestingGraph(AbstractGraph(OrderedMap({W('body'): Function((u,), (r,)),
                                                  W('solve'): Holds((r,), (u,))})), OrderedMap())


def kinds(found) -> list[type]:
    return [type(x) for x in found]


# ---------------------------------------------------------------- the contract

def test_a_condition_node_is_abstract():
    '''The level says what a rule reads -- `unknowns` and `conditions` -- and the finals
    are what carry them.'''
    with pytest.raises(TypeError, match='abstract'):
        ConditionalNode()
    with pytest.raises(TypeError, match='abstract'):
        ConditionalNode(reads=(r,), owns=(u,))


def test_a_driver_is_a_contract_and_nothing_runs():
    '''No call anywhere on the level, and no level is a driver on its own.'''
    assert not callable(Solo())
    with pytest.raises(TypeError, match='abstract'):
        DriverContract()


def test_a_role_is_a_symbol_and_not_a_value():
    with pytest.raises(TypeError, match='a role, not a value'):
        Start()


def test_the_places_are_one_slot_per_role():
    '''Slot `i` is role `takes[i]`; both flattened after the problem's own reads and
    owns -- `reads`, not `conditions`.'''
    node = Driven(Holds((r,), (u,), (k,)), Solo(), *placed())
    assert node.data    == ((W('^start.u'),), (W('^budget.solve'),))
    assert node.reports == ((W('^steps.solve'),),)
    assert node.reads   == (r, k, W('^start.u'), W('^budget.solve'))
    assert node.owns    == (u, W('^steps.solve'))


def test_a_driven_node_forwards_what_the_level_declares_only():
    '''`unknowns` and `conditions`, and nothing a final adds.'''
    node = Driven(Holds((r,), (u,)), Bare(), *NONE)
    assert (node.unknowns, node.conditions) == ((u,), (r,))
    assert 'relations' not in vars(Driven) and 'objectives' not in vars(Driven)
    assert (node.reads, node.owns) == ((r,), (u,))


@pytest.mark.parametrize('statement, driver, places, found', [
    (Holds((r,), ()),     Solo(), placed(()),     [NotAccepted]),
    (Holds((r,), (u, v)), Solo(), placed((u, v)), [NotAccepted]),
    (Driven(Holds((r,), (u,)), Bare(), *NONE), Bare(), NONE, [DrivenTwice]),
])
def test_the_pairing_is_refused_by_kind(statement, driver, places, found):
    assert kinds(violations(Driven, statement, driver, *places))[:1] == found


@pytest.mark.parametrize('data, reports, found', [
    (((W('^start.u'),),),               ((g,),),  [SlotsMiscounted]),
    (placed()[0],                       (),       [SlotsMiscounted]),
    (((), (W('^budget.solve'),)),       ((g,),),  [RoleMisplaced]),
    (((W('^start.u'),), (g, k)),        ((W('^steps.solve'),),), [RoleMisplaced]),
])
def test_one_slot_per_role_and_the_right_number_of_places(data, reports, found):
    assert kinds(violations(Driven, Holds((r,), (u,)), Solo(), data, reports)) == found


def test_a_role_says_how_many_places_it_takes():
    assert kinds(violations(Driven, Holds((r,), (u,)), Unsized(), ((g,),), ())) == [RoleUnsized]


def test_arity_is_asked_of_the_role_class():
    '''A role is a class, so `arity` is a class method, bound by its level.'''
    node = Holds((r,), (u, v))
    assert (PerUnknown.arity(node), Start.arity(node), Budget.arity(node)) == (2, 2, 1)
    assert inspect.isabstract(Role)


def test_the_refusal_names_the_predicate():
    with pytest.raises(ValueError, match='its `accepts` is `single`'):
        Driven(Holds((r,), (u, v)), Solo(), *placed((u, v)))


def test_a_driven_statement_is_not_redetermined():
    with pytest.raises(TypeError) as caught:
        Driven(Holds((r,), (u,)), Bare(), *NONE).determined((g,))
    assert DeterminedWhileDriven.template.split(' {')[0] in str(caught.value)


def test_a_driven_node_hashes_and_pickles():
    '''A node is a jit cache key and a plan pickles.'''
    node = Driven(Holds((r,), (u,)), Solo(), *placed())
    assert hash(node) == hash(Driven(Holds((r,), (u,)), Solo(), *placed()))
    assert pickle.loads(pickle.dumps(node)) == node == copy.deepcopy(node)


def test_the_predicates_and_the_reprs():
    body = ImplementedFunction(Function((u,), (r,)), print)
    node = Driven(Holds((r,), (u,)), Bare(), *NONE)
    assert is_implemented(body) and not is_implemented(node)
    assert is_driven(node) and not is_driven(node.condition)
    assert repr(body).endswith('by print)') and repr(node).endswith('by Bare)')


# ---------------------------------------------------------------- a rename

def test_a_rename_moves_the_place_and_keeps_the_role():
    '''A warm start pointed at another model's output is an ordinary rewire, and the
    slot it sits in is still `Start`.'''
    driven = Assign(W('solve'), Solo(), *placed()).apply(graph())
    moved  = Rename(W('solve'), ((W('^start.u'), g),)).apply(driven)[W('solve')]
    assert moved.data == ((g,), (W('^budget.solve'),))


def test_renaming_an_unknown_moves_that_place_and_no_other():
    '''The places are stored, so a rename moves exactly what it names: `^start.u` is a
    place of its own and stays where it was.'''
    moved = Driven(Holds((r,), (u,)), Solo(), *placed()).relabel({u: v})
    assert moved.unknowns == (v,)
    assert (moved.data, moved.reports) == placed()


def test_undrive_after_a_rename_takes_every_edge_with_it():
    g0     = graph()
    driven = Assign(W('solve'), Solo(), *placed()).apply(g0)
    moved  = Rename(W('solve'), ((W('^start.u'), g),)).apply(driven)
    assert Undrive(W('solve')).apply(moved)[W('solve')] == g0[W('solve')]


def driven_graph() -> NestingGraph:
    '''`body` computes `r` from `u`; `solve` determines `u`, driven; `src` is unrelated.'''
    return NestingGraph(AbstractGraph(OrderedMap({
        W('body') : Function((u, k), (r,)),
        W('solve'): Driven(Holds((r,), (u,)), Bare(), *NONE),
        W('src')  : Function((), (W('u0'),)),
    })), OrderedMap())


def test_replace_puts_a_definition_the_caller_already_holds_in_place():
    g     = driven_graph()
    fresh = Driven(Holds((r,), (u,)), Bare(), *NONE)
    after = Replace(W('solve'), fresh).apply(g)
    assert after[W('solve')] is fresh
    assert after.nodes == g.nodes                       # in place: the binding did not move


def test_replace_and_rename_refuse_an_unbound_node():
    with pytest.raises(KeyError, match='not a node'):
        Replace(W('nope'), Holds((r,), (u,))).apply(driven_graph())
    with pytest.raises(KeyError, match='not a node'):
        Rename(W('nope'), ((u, v),)).apply(driven_graph())


# ---------------------------------------------------------------- the edits

def test_assign_then_undrive_is_the_statement_again():
    g0     = graph()
    driven = Assign(W('solve'), Solo(), *placed()).apply(g0)
    assert driven[W('solve')] == Driven(g0[W('solve')], Solo(), *placed())
    assert Undrive(W('solve')).apply(driven)[W('solve')] == g0[W('solve')]


def test_implement_then_unimplement_is_the_function_again():
    g0   = graph()
    done = Implement(OrderedMap({W('body'): 'a description, not a callable'})).apply(g0)
    assert done[W('body')] == ImplementedFunction(g0[W('body')],
                                                  'a description, not a callable')
    assert Unimplement((W('body'),)).apply(done)[W('body')] == g0[W('body')]


@pytest.mark.parametrize('op, found', [
    (lambda: Assign(W('body'), Bare(), *NONE),           [NotAConditionalNode]),
    (lambda: Assign(W('solve'), Solo(), *NONE),          [SlotsMiscounted, SlotsMiscounted]),
    (lambda: Undrive(W('solve')),                        [NotDriven]),
    (lambda: Implement(OrderedMap({W('solve'): print})), [NotImplementable]),
    (lambda: Unimplement((W('body'),)),                  [NotImplementedYet]),
])
def test_each_edit_refuses_the_wrong_node_as_data(op, found):
    assert kinds(refusals(op(), graph())) == found


def test_a_requirement_is_refused_by_the_driver_before_the_graph_is_touched():
    '''What `apply` would raise from `Driven`'s constructor, `refusals` already says --
    and a node determining nothing is refused by the driver's own `accepts`.'''
    g0 = NestingGraph(AbstractGraph(OrderedMap({W('req'): Holds((r,), ())})), OrderedMap())
    assert kinds(refusals(Assign(W('req'), Solo(), *placed(())), g0)) == [NotAccepted]


def test_the_wrapper_is_not_put_on_twice():
    '''The node's own rule, which the op asks before the graph is touched.'''
    driven = Assign(W('solve'), Bare(), *NONE).apply(graph())
    assert kinds(refusals(Assign(W('solve'), Bare(), *NONE), driven)) == [DrivenTwice]
    # a driver with reports: the second driver is named, not its reports owned twice
    solo = Assign(W('solve'), Solo(), *placed())
    assert kinds(refusals(solo, solo.apply(graph()))) == [DrivenTwice]
    with pytest.raises(TypeError, match='already carries an algorithm'):
        solo.apply(solo.apply(graph()))
    body = Implement(OrderedMap({W('body'): print}))
    assert kinds(refusals(body, body.apply(graph()))) == [ImplementedTwice]
    assert kinds(violations(Unimplement, (W('body'), W('body')))) == [NodeNamedTwice]


# ---------------------------------------------------------------- what runs a driver

def test_a_driven_node_is_implemented_like_a_function():
    '''One op at both levels: a body on a function, what runs the driver on a driven
    node -- and `Unimplement` takes either off.'''
    driven = Assign(W('solve'), Solo(), *placed()).apply(graph())
    done   = Implement(OrderedMap({W('solve'): print})).apply(driven)
    node   = done[W('solve')]
    assert node == ImplementedDriven(driven[W('solve')], print)
    assert is_driven(node) and is_implemented(node)
    assert (node.reads, node.owns, node.unknowns) == (driven[W('solve')].reads,
                                                      driven[W('solve')].owns, (u,))
    assert Unimplement((W('solve'),)).apply(done)[W('solve')] == driven[W('solve')]


def test_the_outer_layer_comes_off_first():
    '''An implemented driven node is not undriven, driven again, implemented again or
    redetermined: each would change what the implementation was chosen for.'''
    driven = Assign(W('solve'), Bare(), *NONE).apply(graph())
    done   = Implement(OrderedMap({W('solve'): print})).apply(driven)
    again  = Implement(OrderedMap({W('solve'): print}))
    with pytest.raises(TypeError) as caught:
        Undrive(W('solve')).apply(done)
    assert UndrivenWhileImplemented.template.split(' {')[0] in str(caught.value)
    assert kinds(refusals(Assign(W('solve'), Bare(), *NONE), done)) == [DrivenTwice]
    assert kinds(refusals(again, done))                              == [ImplementedTwice]
    with pytest.raises(TypeError) as caught:
        done[W('solve')].determined((g,))
    assert DeterminedWhileImplemented.template.split(' {')[0] in str(caught.value)


def test_a_body_goes_on_a_function_and_nothing_else():
    '''Built by hand, the wrong level is refused by the node itself.'''
    assert kinds(violations(ImplementedFunction, Holds((r,), (u,)), print)) == [NotAFunction]
    assert kinds(violations(ImplementedDriven, Holds((r,), (u,)), print)) == [NoContract]


def test_an_implemented_driven_node_relabels_through_both_layers():
    node  = ImplementedDriven(Driven(Holds((r,), (u,)), Solo(), *placed()), print)
    moved = node.relabel({W('^start.u'): g})
    assert isinstance(moved, ImplementedDriven) and moved.data[0] == (g,)


def test_one_op_implements_several_nodes_and_takes_them_off():
    '''A backend chosen is one decision: one op, one rebuild.'''
    driven = Assign(W('solve'), Solo(), *placed()).apply(graph())
    done   = Implement(OrderedMap({W('body'): len, W('solve'): print})).apply(driven)
    assert (done[W('body')].implementation, done[W('solve')].implementation) == (len, print)
    assert Unimplement((W('body'), W('solve'))).apply(done) == driven


def test_an_op_naming_nothing_is_refused():
    assert kinds(violations(Implement, OrderedMap())) == [NothingImplemented]
    assert kinds(violations(Unimplement, ())) == [NothingImplemented]


# ---------------------------------------------------------------- the proofs

def test_driven_then_realised():
    '''A contract on every problem, then what runs every node -- each level refusing
    what it adds, by kind, and nothing a backend asks.'''
    g0     = graph()
    driven = Assign(W('solve'), Solo(), *placed()).apply(g0)
    done   = Implement(OrderedMap({W('body'): len, W('solve'): print})).apply(driven)
    assert kinds(violations(DrivenGraph, g0)) == [Undriven]
    assert kinds(violations(RealisedGraph, driven)) == [NoImplementation, NoImplementation]
    assert isinstance(RealisedGraph(done), DrivenGraph)


def test_a_role_is_one_slot():
    assert kinds(violations(Driven, Holds((r,), (u,)), Twice(), ((g,), (k,)), ())) == [RoleTwice]
