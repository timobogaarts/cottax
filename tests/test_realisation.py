'''
`conditional/`'s driver and `realisation/` alone: the driver contract, the driven node, the two
implemented finals and the edits putting each on, over `core` and nothing else.

Everything runs in a toy place of plain strings, with a condition node that holds no relation,
so the two legs are shown generic in the place *and* in the condition node: what a driven node reads is
`ConditionalNode`'s, and nothing here is jax, a path or a `RelationalCondition`. The toy
place is `test_core`'s written again rather than imported, since that module takes jax.
'''

import dataclasses
from toy_places import Segmented
from proofs import recalled

import copy
import pickle
import inspect
from dataclasses import dataclass

import pytest
from graphs import refusals

from cottax.core import (
    AbstractConditionalNode, Condition, ConditionalNode, FunctionalNode, Graph, Rename, Replace,
    first,
    violations,
)
from cottax.core.maths.definition import relabelled
from cottax.core.refusal_kinds import OwnsInsideDriven
from cottax.core.realisation.implemented import ImplementationWithoutDriver
from cottax.relational.rewrites import Undetermine
from cottax.core.frozen import OrderedMap
from cottax.core.conditional import Assign, AssignByRule, driven as with_driver, driving_violations, Driven, Undrive
from cottax.core import is_driven
from cottax.core.conditional import DriverContract, PerProblem, PerUnknown, Role
from cottax.core.conditional.driver import (
    IndexRepeats, NotAccepted, NotOffered, RoleMisplaced, RoleMissing, RoleOnBothSides, RoleUnsized,
)
from cottax.core.conditional.assign import DrivenTwice, NotAConditionalNode, NotDriven
from cottax.core.conditional.proof import Undriven
from cottax.core.realisation import (
    Implement, check_implementation_fits, implemented, Realised, Unimplement,
    implementation_of, is_implemented,
)
from cottax.core.realisation.proof import NoImplementation
from cottax.core.plan.primitive_ops import (
    AttributeAlreadySet, AttributeNotSet, NoAttributeNamed,
)


# ---------------------------------------------------------------- a toy place

@dataclass(frozen=True)
class Word(Segmented):
    '''A place as a tuple of strings: the whole of `Place`.'''

    segments : tuple[str, ...]

    def __str__(self) -> str:
        return '.'.join(self.segments)


def W(s: str) -> Word:
    return Word(tuple(s.split('.')))


class Holding(Condition):
    '''A condition as bare tuples, no relation anywhere -- and reading outside too, so a
    node building its reads off the wrong one is caught.'''

    reads_inside  : tuple
    owns_inside   : tuple
    reads_outside : tuple = ()
    owns_outside  = ()

    def relabel(self, renaming):
        return Holding(relabelled(self.reads_inside, renaming),
                       relabelled(self.owns_inside, renaming), relabelled(self.reads_outside, renaming))


def Holds(*fields, driver=None, **named) -> ConditionalNode:
    '''The condition node holding `Holding(*fields, **named)`, and `driver`.'''
    return ConditionalNode(Holding(*fields, **named), driver=driver)


# ---------------------------------------------------------------- toy roles and drivers

class Start(PerUnknown):
    '''Where each unknown starts.'''

class Budget(PerProblem):
    '''How many steps the solve may take.'''

class Steps(PerProblem):
    '''How many it took.'''


def single(statement) -> bool:
    '''What `Solo` answers: exactly one unknown.'''
    return len(statement.owns_inside) == 1


class Solo(DriverContract):
    '''A contract and nothing else: what it accepts, the data it takes, what it may report.'''
    accepts    = staticmethod(single)
    takes      = (Start, Budget)
    may_report = (Steps,)


class Bare(DriverContract):
    '''No roles: a driver that adds no edge.'''
    accepts = staticmethod(lambda statement: True)
    takes   = ()


class Echo(DriverContract):
    '''A role it may both take and report: a guess and an answer would share one key.'''
    accepts    = staticmethod(lambda statement: True)
    takes      = (Start,)
    may_report = (Start,)


class Unsized(DriverContract):
    '''A role that says nothing of what its places stand for.'''
    accepts = staticmethod(lambda statement: True)
    takes   = (Role,)


class Times(PerProblem):
    '''Where a history is saved: data a report needs of its own.'''

class History(PerUnknown):
    '''Each unknown at every one of the `Times`.'''


class Recording(DriverContract):
    '''May report a `History`, and then takes the `Times` it is saved at.'''
    accepts      = staticmethod(lambda statement: True)
    takes        = ()
    may_report   = (History,)
    report_needs = {History: (Times,)}


u, v, r, k, g = W('u'), W('v'), W('r'), W('k'), W('g')
NONE = (OrderedMap(), OrderedMap())


def by_role(*pairs) -> OrderedMap:
    return OrderedMap(pairs)


def placed(unknowns=(u,)) -> tuple:
    '''Where `Solo`'s roles sit for a problem over `unknowns`, `Steps` asked: its data, its
    reports.'''
    return (by_role((Start, tuple(W(f'^start.{x}') for x in unknowns)), (Budget, (W('^budget.solve'),))),
            by_role((Steps, (W('^steps.solve'),))))


def graph():
    '''A body owning `r` from `u`, and the problem `r` holds over `u`.'''
    return Graph(OrderedMap({W('body'): FunctionalNode((u,), (r,)),
                                                  W('solve'): Holds((r,), (u,))}), OrderedMap())


def kinds(found) -> list[type]:
    return [type(x) for x in found]


# ---------------------------------------------------------------- the contract

def test_a_condition_node_is_abstract():
    '''The level says what a rule reads -- `unknowns` and `conditions` -- and the finals
    are what carry them.'''
    with pytest.raises(TypeError, match='abstract'):
        AbstractConditionalNode()
    with pytest.raises(TypeError, match='abstract'):
        Condition()
    with pytest.raises(TypeError):
        ConditionalNode(reads=(r,), owns=(u,))


def test_a_driver_is_a_contract_and_nothing_runs():
    '''No call anywhere on the level, and no level is a driver on its own.'''
    assert not callable(Solo())
    with pytest.raises(TypeError, match='abstract'):
        DriverContract()


def test_a_role_is_a_symbol_and_not_a_value():
    with pytest.raises(TypeError, match='a role, not a value'):
        Start()


def test_the_places_are_kept_by_role():
    '''Per role, its places; both sides flattened after the problem's own reads and owns --
    `reads`, not `conditions` -- and each place paired with what it stands for.'''
    node = with_driver(Holds((r,), (u,), (k,)), Solo(), *placed())
    assert node.driver.data    == by_role((Start, (W('^start.u'),)), (Budget, (W('^budget.solve'),)))
    assert node.driver.reports == by_role((Steps, (W('^steps.solve'),)))
    assert node.driver.places_for(Start, node.condition) == {u: W('^start.u')}
    assert node.driver.places_for(Steps, node.condition) == {None: W('^steps.solve')}
    assert node.driver.places_for(Role, node.condition) is None
    assert node.reads   == (r, k, W('^start.u'), W('^budget.solve'))
    assert node.owns    == (u, W('^steps.solve'))
    # what every rule reads outside is the condition's and the driver's together
    assert (node.condition.reads_outside, node.reads_outside) == (
        (k,), (k, W('^start.u'), W('^budget.solve')))
    assert (node.condition.owns_outside, node.owns_outside) == ((), (W('^steps.solve'),))


def test_a_driven_node_is_its_condition_with_a_driver():
    '''The node keeps its condition as it was, and a driver with no roles adds no edge.'''
    node = with_driver(Holds((r,), (u,)), Bare(), *NONE)
    assert type(node) is ConditionalNode and node.condition == Holding((r,), (u,))
    assert (node.owns_inside, node.reads_inside) == ((u,), (r,))
    assert (node.reads, node.owns) == ((r,), (u,))


@pytest.mark.parametrize('statement, driver, places, found', [
    (Holds((r,), ()),     Solo(), placed(()),     [NotAccepted]),
    (Holds((r,), (u, v)), Solo(), placed((u, v)), [NotAccepted]),
    (with_driver(Holds((r,), (u,)), Bare(), *NONE), Bare(), NONE, [DrivenTwice]),
])
def test_the_pairing_is_refused_by_kind(statement, driver, places, found):
    assert kinds(driving_violations(statement, driver, *places))[:1] == found


@pytest.mark.parametrize('data, reports, found', [
    (by_role((Start, (W('^start.u'),))),                        by_role((Steps, (g,))), [RoleMissing]),
    (placed()[0],                                                by_role(),              []),
    (by_role(*placed()[0].items(), (Steps, (k,))),              by_role(),              [NotOffered]),
    (placed()[0],                                                by_role((History, (g,))), [NotOffered]),
    (by_role((Start, ()), (Budget, (W('^budget.solve'),))),     by_role((Steps, (g,))), [RoleMisplaced]),
    (by_role((Start, (W('^start.u'),)), (Budget, (g, k))),      placed()[1],            [RoleMisplaced]),
])
def test_the_required_roles_and_the_right_number_of_places(data, reports, found):
    '''Every role it takes is given; a report is optional, asked for by placing it; a role
    the contract does not offer on that side is refused; each role sits at as many places as
    its `index` has entries.'''
    assert kinds(driving_violations(Holds((r,), (u,)), Solo(), data, reports)) == found


def test_a_report_may_need_data_of_its_own():
    '''`Recording` takes nothing, until a `History` is asked for: then the `Times` too.'''
    history, times = by_role((History, (W('^history.u'),))), by_role((Times, (W('^times'),)))
    assert kinds(driving_violations(Holds((r,), (u,)), Recording(), by_role(), by_role())) == []
    assert kinds(driving_violations(Holds((r,), (u,)), Recording(), by_role(), history)) == [RoleMissing]
    assert kinds(driving_violations(Holds((r,), (u,)), Recording(), times, history)) == []
    assert Recording().required((History,)) == (Times,) and Recording().may_be_given == (Times,)


def test_a_driver_that_does_not_fit_is_never_built():
    '''Whether the driver fits is the node's own rule: built by hand, or rebuilt with a
    condition its driver no longer fits, the node is refused as it is made.'''
    with pytest.raises(ValueError, match='its `accepts` is `single`'):
        with_driver(Holds((r,), (u, v)), Solo(), *placed((u, v)))
    node = with_driver(Holds((r,), (u,)), Recording(), by_role((Times, (W('^times'),))),
                       by_role((History, (W('^history.u'),))))
    with pytest.raises(ValueError, match='takes 2 on this node'):
        dataclasses.replace(node, condition=Holding((r,), (u, v)))
    with pytest.raises(TypeError, match='already carries an algorithm'):
        with_driver(node, Bare())


def test_an_index_names_each_place_once():
    class Doubled(PerProblem):
        @classmethod
        def index(cls, condition, contract):
            return (None, None)
    class Doubling(DriverContract):
        accepts = staticmethod(lambda statement: True)
        takes   = (Doubled,)
    found = driving_violations(Holds((r,), (u,)), Doubling(), by_role((Doubled, (g, k))), by_role())
    assert kinds(found) == [IndexRepeats]


def test_a_condition_refused_for_what_it_states_is_refused_for_that_first():
    '''The driver's rules are asked once the condition passes its own: a condition
    determining and reading nothing is refused as empty when it is built, before any driver
    is attached, not as something `Solo` does not accept.'''
    from cottax.core.conditional import Driver
    from cottax.core.refusal_kinds import StatesNothing
    solo = Driver(Solo(), *placed(()))
    with pytest.raises(ValueError, match='an empty condition node'):
        Holds((), (), driver=solo)
    assert kinds(violations(Holding, (), ())) == [StatesNothing]


def test_what_a_driver_accepts_is_asked_of_the_condition_without_it():
    '''`accepts` sees the condition alone: neither the driver's places nor its name.'''
    seen = []
    class Spying(DriverContract):
        accepts    = staticmethod(lambda node: seen.append(node) or True)
        takes      = (Start, Budget)
        may_report = (Steps,)
    assert list(driving_violations(Holds((r,), (u,)), Spying(), *placed())) == []
    assert seen and all(type(n) is Holding and n.reads == (r,) for n in seen)
    with pytest.raises(ValueError) as refused:
        first(driving_violations(Holds((r,), (u, v)), Solo(), *placed((u, v))))
    assert ' by ' not in str(refused.value)


def test_a_role_says_what_its_places_stand_for():
    assert kinds(driving_violations(Holds((r,), (u,)), Unsized(), by_role((Role, (g,))), by_role())
                 ) == [RoleUnsized]


def test_index_is_asked_of_the_role_class():
    '''A role is a class, so `index` is a class method, bound by its level; `arity` is how
    many entries it has.'''
    node = Holding((r,), (u, v))
    assert (Start.index(node, Solo()), Budget.index(node, Solo())) == ((u, v), (None,))
    assert (PerUnknown.arity(node, Solo()), Start.arity(node, Solo()), Budget.arity(node, Solo())) == (2, 2, 1)
    assert inspect.isabstract(Role)


def test_the_refusal_names_the_predicate():
    with pytest.raises(ValueError, match='its `accepts` is `single`'):
        first(driving_violations(Holds((r,), (u, v)), Solo(), *placed((u, v))))


def test_a_driven_node_hashes_and_pickles():
    '''A node is a jit cache key and a plan pickles.'''
    node = with_driver(Holds((r,), (u,)), Solo(), *placed())
    assert hash(node) == hash(with_driver(Holds((r,), (u,)), Solo(), *placed()))
    assert pickle.loads(pickle.dumps(node)) == node == copy.deepcopy(node)


def test_the_predicates_and_the_reprs():
    body = implemented(FunctionalNode((u,), (r,)), print)
    node = with_driver(Holds((r,), (u,)), Bare(), *NONE)
    assert is_implemented(body) and not is_implemented(node)
    assert is_driven(node) and not is_driven(node.condition)
    assert repr(node.driver) == 'Bare'


# ---------------------------------------------------------------- a rename

def test_a_rename_moves_the_place_and_keeps_the_role():
    '''A warm start pointed at another model's output is an ordinary rewire, and the
    slot it sits in is still `Start`.'''
    driven = Assign(W('solve'), Solo(), *placed()).apply(graph())
    moved  = Rename(W('solve'), ((W('^start.u'), g),)).apply(driven)[W('solve')]
    assert moved.driver.data == by_role((Start, (g,)), (Budget, (W('^budget.solve'),)))


def test_renaming_an_unknown_moves_that_place_and_no_other():
    '''The places are stored, so a rename moves exactly what it names: `^start.u` is a
    place of its own and stays where it was.'''
    moved = with_driver(Holds((r,), (u,)), Solo(), *placed()).relabel({u: v})
    assert moved.owns_inside == (v,)
    assert (moved.driver.data, moved.driver.reports) == placed()


def test_undrive_after_a_rename_takes_every_edge_with_it():
    g0     = graph()
    driven = Assign(W('solve'), Solo(), *placed()).apply(g0)
    moved  = Rename(W('solve'), ((W('^start.u'), g),)).apply(driven)
    assert Undrive(W('solve')).apply(moved)[W('solve')] == g0[W('solve')]


def driven_graph() -> Graph:
    '''`body` computes `r` from `u`; `solve` determines `u`, driven; `src` is unrelated.'''
    return Graph(OrderedMap({
        W('body') : FunctionalNode((u, k), (r,)),
        W('solve'): with_driver(Holds((r,), (u,)), Bare(), *NONE),
        W('src')  : FunctionalNode((), (W('u0'),)),
    }), OrderedMap())


def test_replace_puts_a_definition_the_caller_already_holds_in_place():
    g     = driven_graph()
    fresh = with_driver(Holds((r,), (u,)), Bare(), *NONE)
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
    assert driven[W('solve')] == with_driver(g0[W('solve')], Solo(), *placed())
    assert Undrive(W('solve')).apply(driven)[W('solve')] == g0[W('solve')]


def test_implement_then_unimplement_is_the_function_again():
    g0   = graph()
    done = Implement(OrderedMap({W('body'): 'a description, not a callable'})).apply(g0)
    assert done[W('body')] == implemented(g0[W('body')],
                                                  'a description, not a callable')
    assert Unimplement((W('body'),)).apply(done)[W('body')] == g0[W('body')]


@pytest.mark.parametrize('op, found', [
    (lambda: Assign(W('body'), Bare(), *NONE),           [NotAConditionalNode]),
    (lambda: Assign(W('solve'), Solo(), *NONE),          [RoleMissing]),
    (lambda: Undrive(W('solve')),                        [NotDriven]),
    (lambda: Implement(OrderedMap({W('solve'): print})), [ImplementationWithoutDriver]),
    (lambda: Unimplement((W('body'),)),                  [AttributeNotSet]),
])
def test_each_edit_refuses_the_wrong_node_as_data(op, found):
    assert kinds(refusals(op(), graph())) == found


def test_a_requirement_is_refused_by_the_driver_before_the_graph_is_touched():
    '''What `apply` would raise from the driven node's constructor, `refusals` already says --
    and a node determining nothing is refused by the driver's own `accepts`.'''
    g0 = Graph(OrderedMap({W('req'): Holds((r,), ())}), OrderedMap())
    assert kinds(refusals(Assign(W('req'), Solo(), *placed(())), g0)) == [NotAccepted]


def test_a_driver_is_not_put_on_twice():
    '''The node's own rule, which the op asks before the graph is touched.'''
    driven = Assign(W('solve'), Bare(), *NONE).apply(graph())
    assert kinds(refusals(Assign(W('solve'), Bare(), *NONE), driven)) == [DrivenTwice]
    # a driver with reports: the second driver is named, not its reports owned twice
    solo = Assign(W('solve'), Solo(), *placed())
    assert kinds(refusals(solo, solo.apply(graph()))) == [DrivenTwice]
    with pytest.raises(TypeError, match='already carries an algorithm'):
        solo.apply(solo.apply(graph()))
    body = Implement(OrderedMap({W('body'): print}))
    assert kinds(refusals(body, body.apply(graph()))) == [AttributeAlreadySet]


# ---------------------------------------------------------------- what runs a driver

def test_a_driven_node_is_implemented_like_a_function():
    '''One op at both levels: a body on a function, what runs the driver on a driven
    node -- and `Unimplement` takes either off.'''
    driven = Assign(W('solve'), Solo(), *placed()).apply(graph())
    done   = Implement(OrderedMap({W('solve'): print})).apply(driven)
    node   = done[W('solve')]
    assert node == implemented(driven[W('solve')], print)
    assert is_driven(node) and is_implemented(node)
    assert (node.reads, node.owns, node.owns_inside) == (driven[W('solve')].reads,
                                                      driven[W('solve')].owns, (u,))
    assert Unimplement((W('solve'),)).apply(done)[W('solve')] == driven[W('solve')]


def test_the_outer_layer_comes_off_first():
    '''An implemented driven node is not driven again, implemented again or redetermined:
    each would change what the implementation was chosen for. Undriving it is the driven
    layer's op, which cannot ask the realised layer's rule: the node it leaves is refused
    where that rule is asked, by a realised graph.'''
    driven = Assign(W('solve'), Bare(), *NONE).apply(graph())
    done   = Implement(OrderedMap({W('solve'): print})).apply(driven)
    again  = Implement(OrderedMap({W('solve'): print}))
    undriven = Undrive(W('solve')).apply(done)
    assert ImplementationWithoutDriver in kinds(Realised.check(undriven))
    assert kinds(refusals(Assign(W('solve'), Bare(), *NONE), done)) == [DrivenTwice]
    assert kinds(refusals(again, done))                              == [AttributeAlreadySet]
    assert kinds(refusals(Undetermine(W('solve')), done)) == [OwnsInsideDriven]


def test_an_implementation_goes_on_a_function_or_a_driven_node():
    '''A bare condition node is refused one: nothing says what would run it until a driver
    is on. The realised layer's rule, asked by `Implement` before the graph is touched and by
    a realised graph; built by hand, the node exists until then.'''
    bare = implemented(Holds((r,), (u,)), print)
    assert kinds(check_implementation_fits.node(W('solve'), bare)) == [ImplementationWithoutDriver]
    assert kinds(refusals(Implement(OrderedMap({W('solve'): print})), graph())) == [ImplementationWithoutDriver]
    with pytest.raises(TypeError, match='Implementation does not fit a ConditionalNode'):
        Implement(OrderedMap({W('solve'): print})).apply(graph())


def test_a_driver_goes_on_a_node_carrying_attributes():
    '''The driver is the node's own field, so what the node carries stays where asking
    the graph's node finds it. (A fold of such a node is refused: `test_combine`.)'''
    class Note: ...
    noted = with_driver(dataclasses.replace(Holds((r,), (u,)), attributes=OrderedMap({Note: 'n'})), Bare(), *NONE)
    assert noted.attributes == OrderedMap({Note: 'n'}) and is_driven(noted)


def test_an_implemented_driven_node_keeps_what_runs_it_through_a_rename():
    '''`Holds.relabel` builds a fresh node and never mentions what it carries:
    `relabel` puts the driver and the attributes back, and the implementation holds, the
    contract being the same.'''
    driven = Assign(W('solve'), Solo(), *placed()).apply(graph())
    done   = Implement(OrderedMap({W('solve'): print})).apply(driven)
    moved  = Rename(W('solve'), ((W('^start.u'), g),)).apply(done)[W('solve')]
    assert is_implemented(moved) and moved.driver.data[Start] == (g,)


def test_one_op_implements_several_nodes_and_takes_them_off():
    '''A backend chosen is one decision: one op, one rebuild.'''
    driven = Assign(W('solve'), Solo(), *placed()).apply(graph())
    done   = Implement(OrderedMap({W('body'): len, W('solve'): print})).apply(driven)
    assert (implementation_of(done[W('body')]), implementation_of(done[W('solve')])) == (len, print)
    assert Unimplement((W('body'), W('solve'))).apply(done) == driven


def test_an_op_naming_nothing_is_refused():
    assert kinds(violations(Implement, OrderedMap())) == [NoAttributeNamed]
    assert kinds(violations(Unimplement, ())) == [NoAttributeNamed]


# ---------------------------------------------------------------- the proofs

def test_driven_then_realised():
    '''A contract on every problem, then what runs every node -- each level refusing
    what it adds, by kind, and nothing a backend asks.'''
    g0     = graph()
    driven = Assign(W('solve'), Solo(), *placed()).apply(g0)
    done   = Implement(OrderedMap({W('body'): len, W('solve'): print})).apply(driven)
    assert kinds(Driven.check(g0)) == [Undriven]
    assert kinds(Realised.check(driven)) == [NoImplementation, NoImplementation]
    Realised.prove(done)
    assert recalled(done, Driven)


def test_a_role_is_on_one_side():
    found = driving_violations(Holds((r,), (u,)), Echo(), by_role((Start, (g,))), by_role((Start, (k,))))
    assert kinds(found) == [RoleOnBothSides]


def test_assign_by_rule_places_what_is_asked_and_what_it_needs():
    '''The rule is given each role and the condition; `asks` wires a report, and a report
    the contract does not offer is refused before anything is placed.'''
    rule = lambda problem, condition, contract, role: tuple(
        W(f'^{role.__name__.lower()}.{problem if s is None else s}') for s in role.index(condition, contract))
    driven = AssignByRule(W('solve'), Solo(), rule, (Steps,)).apply(graph())[W('solve')]
    assert (driven.driver.data, driven.driver.reports) == placed()
    plain = AssignByRule(W('solve'), Solo(), rule).apply(graph())[W('solve')]
    assert plain.driver.reports == by_role() and plain.driver.data == placed()[0]
    assert kinds(violations(AssignByRule, W('solve'), Solo(), rule, (History,))) == [NotOffered]
