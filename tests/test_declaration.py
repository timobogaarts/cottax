'''
The declaration surface over the toy place `test_answers` declares in, with no pytree and no
jax: every relational kind, a constraint, a bounded optimisation subject to it, and a
declaration, all through a `Surface` of cottax's and relational's parts; a part of its own, stating two
kinds with `declares` and a contract cottax does not ship, joined to the relational one; and
conventions reading a term their way.
'''

import dataclasses
import inspect
import math
import pydoc

import pytest

from cottax.core import GraphWithValues, Mint, minted
import purposes
from cottax.core.frozen import OrderedMap, answer, choice
from cottax.core.maths.plan import Plan
from cottax.core.driven import Assign, DriverContract, PerUnknown
from cottax.core.driven import Start
from cottax.relational.contracts import CONTRACTS, FixedPointIteration
from cottax.consumers.execution import CallableGraph, Drive, run
from cottax.consumers.execution.interface import DriverImplementation
from cottax.consumers.declaration import (
    Conventions, Shipped, declare, loose_at, tags_of, term_of,
)
from cottax.consumers.surface import Purpose, RolePlaces, core_role_places
from cottax.consumers.declaration.refusals import NotAnInput, Unnamed
from cottax.consumers.declaration.interface import Declaring
from cottax.consumers.declaration.interface_types import (
    Keyword, Paired, PlaceWithValue, Tagged, Wrapper, declares,
)
from cottax.consumers.surface.interface import DerivedNames, Part
from cottax.assemblies.jax import Scope, ShippedSurface, V, absolute
from cottax.pytree.consumers.surface import Under

place_of = absolute.place
from cottax.pytree.consumers.surface import node_of
from cottax.pytree.path import GetAttrKey
from cottax.relational.consumers.declaration.vocabulary import RelationalDeclaring
from cottax.relational.consumers.declaration.declarations import Residual
from cottax.relational import (
    Eq, Le, Max, Objective, Relation, RelationalCondition, is_root_find,
)
from cottax.consumers.visualization.kinds import problem_type_of
from test_answers import T, Tok, Toks


def toy(*parts, conventions=Shipped()):
    '''The surface over the toy place, joining cottax's and relational's declarations and
    `parts`.'''
    return type('Toy', (RelationalDeclaring, *parts), {})(Toks(), conventions)


def test_the_surface_declares_over_a_place_of_its_own():
    s = toy()

    @s.function('y')
    def body(x, k):
        return x * k

    @s.residual('u')
    def balance(u, y):
        return y - u

    limit = s.inequality_constraint(lambda u: u, 'g', within=(None, 5.0))
    opt   = s.optimise(maximise='u', design={'x': (0, 10)}, subject_to=(limit,))
    d = declare({'body': body, 'balance': balance, 'limit': limit, 'opt': opt})

    assert d.graph.nodes == (T('body'), T('balance'), T('^solve.balance'), T('limit'), T('opt'))
    assert d.graph[T('^solve.balance')].unknowns == (T('u'),)
    top = d.graph[T('opt')]
    assert top.objectives == (Objective(T('u'), Max),)
    assert Relation(T('g'), T('^upper.g'), Le) in top.relations
    assert Relation(None, T('x'), Le) in top.relations
    assert dict(d.given) == {T('^upper.x'): 10, T('^upper.g'): 5.0}
    assert d.graph[T('balance')].owns == (T('^cond.u'),)


@dataclasses.dataclass(frozen=True)
class Plant:
    physics : dict
    limit   : object = None


def test_a_tree_of_models_is_declared_each_at_the_steps_to_it():
    s = toy()
    body  = s.function(lambda x: 2.0 * x, 'y')
    model = s.residual(lambda u, y: y - u, 'u')
    d = declare(Plant({'body': body, 'model': model}, s.requirement('y')))
    assert d.graph.nodes == (T('physics.body'), T('physics.model'), T('^solve.physics.model'),
                             T('limit'))


def test_each_condition_node_writes_the_shape_it_names():
    s = toy()
    residual  = s.residual(lambda u, w: (u, w), 'u', 'w').condition
    iteration = s.fixed_point(lambda u: u, 'u').condition
    assert problem_type_of(residual)                                               == 'root-find'
    assert problem_type_of(iteration)                                              == 'fixed-point'
    assert problem_type_of(s.satisfy(design={'u': (0, None)}).condition)           == 'feasibility'
    assert problem_type_of(s.optimise(minimise='f', design=['x']).condition)       == 'optimise'
    assert problem_type_of(s.requirement('h'))                                     == 'stated'
    assert iteration.relations == (Relation(T('u'), T('^cond.u'), Eq),)


def test_a_model_handed_over_loose_takes_the_name_it_says_and_stands_for_it_as_a_node_name():
    '''A wrapper says its function's name, a problem `^solve.<objective>`; one saying none
    (a `satisfy`, a lambda) goes in a slot. Where a node name is due the surface reads a
    declaration as the name it takes loose, a paired wrapper's its body's.'''
    s = toy()

    @s.residual('u')
    def balance(u, y):
        return y - u

    def body(x):
        return 2.0 * x

    opt = s.optimise(maximise='u', design={'x': (0, 10)})
    d = declare(s.function(body, 'y'), balance, opt)
    assert d.graph.nodes == (T('body'), T('balance'), T('^solve.balance'), T('^solve.u'))
    assert declare({'top': opt}).graph.nodes == (T('top'),)
    loose = lambda model: loose_at(model, s.derived)
    assert (loose(opt), loose(balance), s.node(Purpose(purposes.Solve())(loose(balance)))) == (
        T('^solve.u'), T('balance'), T('^solve.balance'))
    for nameless in (s.satisfy(design={'x': (0, 10)}), s.function(lambda x: x, 'y')):
        with pytest.raises(TypeError) as refused:
            declare(nameless)
        assert refused.value.args[0] == Unnamed(repr(nameless)).message
        with pytest.raises(TypeError, match='has no name of its own'):
            loose(nameless)


class Marked(Tagged):
    '''A toy tag on a node, read by no one but this test.'''


def test_a_tag_around_a_model_in_a_slot_is_on_the_node_name_it_takes():
    s = toy()
    d = declare({'coils': Marked(s.function(lambda x: x, 'y')),
                 'r': s.residual(lambda u, y: y - u, 'u')})
    assert d.graph.nodes == (T('coils'), T('r'), T('^solve.r'))
    assert d.node_tags == OrderedMap({T('coils'): (Marked(None),)})
    assert tags_of(d, T('coils'), Marked) == (Marked(None),)
    assert tags_of(d, T('r'), Marked) == ()


def test_using_gives_a_boundary_input_its_value_and_refuses_anything_else():
    s = toy()
    d = declare({'f': s.function(lambda x: x, 'y', reads={'x': 2.0})})
    k = s.place(Purpose(purposes.Given()).beside('y', 'x'))
    assert dict(d.given) == {k: 2.0}
    assert dict(d.using({k: 3.0}).given) == {k: 3.0} and d.using({k: 3.0}) == d
    with pytest.raises(KeyError) as refused:
        d.using({T('y'): 1.0})
    assert refused.value.args[0] == NotAnInput((T('y'),)).message


# ---------------------------------------------------------------- a vocabulary of its own

class Low(PerUnknown):
    '''Where each unknown's bracket starts.'''

class High(PerUnknown):
    '''Where it ends.'''


class Bisect(DriverContract):
    '''A root find inside a bracket: a contract cottax does not ship.'''
    accepts = staticmethod(is_root_find)
    takes   = (Low, High)
    reports = ()


@choice(DriverImplementation)
class Bisection(DriverImplementation[Bisect]):
    needs    = frozenset({Low, High})
    provides = frozenset()

    def __call__(self, posed, context, data):
        (lo,), (hi,) = data[Low], data[High]
        gap = lambda x: posed.conditions((x,), context)[0]
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if gap(lo) * gap(mid) <= 0 else (mid, hi)
        return ((lo + hi) / 2,), {}


@choice(DriverImplementation)
class Picard(DriverImplementation[FixedPointIteration]):
    needs    = frozenset({Start})
    provides = frozenset()

    def __call__(self, posed, context, data):
        (u,) = data[Start]
        for _ in range(100):
            u = posed.conditions((u,), context)[0]
        return (u,), {}


class Gap(Mint):
    '''A bracketed residual: `^gap.u`.'''

class Bracket(Mint):
    '''Where a bracketed residual's problem binds: `^bracket.at`.'''

class Scratch(Mint):
    '''What a tolerated function writes: `^scratch.y`.'''

class Tol(Mint):
    '''A tolerance written as a value: `^tol.y`.'''

class Lo(Mint):
    '''Where a bracket starts: `^lo.u`.'''

class Hi(Mint):
    '''Where it ends: `^hi.u`.'''


class Bracketed(Paired):
    '''A residual per unknown (`^gap.u`), held at zero by the problem at `^bracket.at`.'''

    terms = ()
    condition_mint = Bracket

    @property
    def function_outputs(self):
        return tuple(minted(Gap(), u) for u in self.outputs)

    def condition_node(self, at):
        return self.condition

    @property
    def condition(self):
        return RelationalCondition(tuple(Relation(r, None, Eq) for r in self.function_outputs),
                                   (), self.outputs, ())


class Tolerated(Wrapper):
    '''A function writing a scratch place per output (`^scratch.y`), with a tolerance beside
    it.'''

    tolerance : PlaceWithValue | None

    @property
    def function_outputs(self):
        return tuple(minted(Scratch(), y) for y in self.outputs)

    @property
    def terms(self):
        return (self.tolerance,)

    def nodes(self, at):
        return ((at, self.function_node()),)


def tolerance_of(surface, fn, outputs, tol):
    '''`tol=` as a term about the first output, none where not given.'''
    return None if tol is None else surface.term(outputs[0], tol, Tol())


@choice(Part)
class Bracketing(Declaring):
    '''A part of its own: a residual answered inside a bracket, and a function with its
    tolerance.'''

    bracketed = declares(Bracketed, '''A residual, answered inside a bracket.''')
    tolerated = declares(Tolerated, '''A function and its tolerance:
        `tolerated(fn, *outputs, tol=)`, or `@tolerated(*outputs, tol=)`.''',
                         tolerance=Keyword('tol', tolerance_of))


def test_a_second_vocabulary_joins_and_its_problems_are_driven_by_its_own_contract():
    s = toy(Bracketing)
    root = s.bracketed(lambda x, k: x * x - k, 'x')
    iterate = s.fixed_point(lambda u, k: 0.5 * (u + k / u), 'u')
    d = declare({'root': root, 'iterate': iterate})
    assert d.graph.nodes == (T('root'), T('^bracket.root'), T('iterate'), T('^solve.iterate'))
    assert d.graph[T('root')].owns == (T('^gap.x'),)

    places = core_role_places | RolePlaces(OrderedMap(((Low, Lo()), (High, Hi()))))
    drive = Drive((Picard(), Bisection()), (Bisect(), *CONTRACTS), places)
    assigned = [op for op in drive.composed(d.graph) if isinstance(op, Assign)]
    assert [(op.problem, op.contract) for op in assigned] == [
        (T('^bracket.root'), Bisect()), (T('^solve.iterate'), FixedPointIteration(steps=False))]
    assert assigned[0].data == ((T('^lo.x'),), (T('^hi.x'),))

    graph = (Plan(d.graph) + drive).graph
    out = run(GraphWithValues(CallableGraph(graph), OrderedMap({
        T('k'): 2.0, T('^lo.x'): 0.0, T('^hi.x'): 2.0, T('^start.u'): 1.0})))
    assert math.isclose(out[T('x')], math.sqrt(2.0), rel_tol=1e-12)
    assert math.isclose(out[T('u')], math.sqrt(2.0), rel_tol=1e-12)


def test_a_scope_joins_the_parts_a_study_names():
    class Plant(ShippedSurface, Bracketing):
        '''The shipped parts and a study's own.'''

    s = Plant(Under(V.plant, GetAttrKey), Shipped())
    assert hasattr(s, 'bracketed') and hasattr(s, 'residual')
    root = s.bracketed(lambda x, k: x * x - k, 'x')
    x, k = s.place('x'), s.place('k')
    assert (x, k) == (place_of(V.plant.x), place_of(V.plant.k))
    assert root.function_reads == (PlaceWithValue(x), PlaceWithValue(k))
    at = s.node(Purpose(Bracket())(node_of('root')))
    assert declare({'root': root}).graph[at].unknowns == (x,)


def test_a_kind_declared_is_a_method_in_both_forms_minting_by_purpose_and_reading_its_keyword():
    s = Bracketing(Toks(), Shipped())
    called = s.tolerated(lambda x, k: x * k, 'y', reads={'k': 2.0}, tol=1e-3)

    @s.tolerated('y', tol='eps')
    def settled(x):
        return x

    assert called.function_reads == (PlaceWithValue(T('x')), PlaceWithValue(T('^given.y.k'), 2.0))
    assert called.function_outputs == (T('^scratch.y'),)
    assert called.tolerance == PlaceWithValue(T('^tol.y'), 1e-3)
    assert dict(called.given) == {T('^given.y.k'): 2.0, T('^tol.y'): 1e-3}
    assert isinstance(settled, Tolerated) and settled.tolerance == PlaceWithValue(T('eps'))
    assert settled.fn.__qualname__.endswith('settled.fn')
    assert s.tolerated(lambda x: x, 'y').tolerance is None
    assert declare({'t': called}).graph[T('t')].owns == (T('^scratch.y'),)

    with pytest.raises(TypeError, match=r"Bracketing.tolerated\(\) got an unexpected keyword "
                                        r"argument 'within'"):
        s.tolerated(lambda x: x, 'y', within=1.0)


def test_a_declared_method_is_named_where_it_is_bound_and_its_help_is_its_doc():
    doc = Bracketing.tolerated.__doc__
    assert (Bracketing.tolerated.__name__, Bracketing.tolerated.__qualname__) == (
        'tolerated', 'Bracketing.tolerated')
    offered = Bracketing(Toks(), Shipped()).tolerated
    assert pydoc.getdoc(offered) == inspect.cleandoc(doc)
    assert (offered.__name__, offered.__qualname__) == ('tolerated', 'Bracketing.tolerated')


def test_two_parts_offering_one_word_are_refused_where_they_are_joined():
    class Rival(Declaring):
        residual = declares(Residual, '''Another residual.''')

    with pytest.raises(TypeError, match='both implement `residual`'):
        toy(Rival)
    with pytest.raises(TypeError, match='overrides the one RelationalDeclaring implements'):
        class Again(RelationalDeclaring):
            residual = declares(Residual, '''Again.''')


# ---------------------------------------------------------------- conventions of one's own

def limits(x, written, *, place, value_at):
    '''A term written as a string read under `limits`; anything else as cottax reads it.'''
    if isinstance(written, str):
        return PlaceWithValue(Tok(('limits', written)))
    return term_of(x, written, place=place, value_at=value_at)


def test_a_term_read_otherwise_changes_every_reading_that_takes_one():
    s = toy(conventions=Conventions(term=limits))
    within  = s.inequality_constraint(lambda x: x, 'g', within=(0.5, 'g_max'))
    against = s.equality_constraint(lambda x: x, 'h', against='h_t')
    assert within.bounds.at_most[T('g')]   == PlaceWithValue(T('limits.g_max'))
    assert within.bounds.at_least[T('g')]  == PlaceWithValue(T('^lower.g'), 0.5)
    assert against.targets                 == OrderedMap({T('h'): PlaceWithValue(T('limits.h_t'))})
    assert toy().equality_constraint(
        lambda x: x, 'h', against='h_t').targets == OrderedMap({T('h'): PlaceWithValue(T('h_t'))})
