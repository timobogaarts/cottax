'''
`interfaces`: functions wrapped as nodes, scopes, a tree of models as a graph, and the ops
with their defaults. Sellar is the end-to-end case, on a plain dict and on a structured
tree.
'''

import dataclasses
import enum
import inspect
import pickle
import posixpath
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

import cottax.assemblies.jax as cx
from cottax.consumers.declaration import declare
from cottax.core.frozen import OrderedMap, violations
from cottax.core.maths.plan import Insert, Nest, Plan, Rename
from cottax.relational.contracts import (
    ConstrainedMinimisation, ConstrainedMinimiser, FixedPointIteration, RootFinder, RootFinding,
)
from cottax.assemblies.jax import N, V, flat
from cottax.consumers.declaration.refusals import HeldNotInModel
from cottax.consumers.execution.refusals import Unvalued, ValuedTwice
from cottax.consumers.declaration.interface_types import Tagged
from cottax.consumers.declaration.refusals import (
    HeldByValue, NoOutputs, NotAnInput, NotPositional, ReadsUnlike,
)
from cottax.consumers.surface.refusals import NotANode, NotAPlace, StandsForSeveral
from cottax.relational.consumers.declaration.declarations import (
    AgainstNoOutput, BoundNotUnknown, HeldAgainstNothing, NotAPair,
)
from cottax.consumers.execution.jax_jit import JitGraph
import purposes
from cottax.pytree.path import DictKey, GetAttrKey, SequenceKey, VarPath
from cottax.core.realisation import ImplementDrivers
from cottax.relational import is_fixed_point, is_root_find


v, n = flat.place, cx.absolute.node


@pytest.fixture
def x64():
    with jax.enable_x64(True):                      # the inner Newton's tolerance is float64's
        yield


# ---------------------------------------------------------------- sellar, once

def disc1(z1, z2, x1, y1, y2):
    return z1**2 + z2 + x1 - 0.2 * y2 - y1


@flat.function('y2')
def disc2(z1, z2, y1):
    return y1**0.5 + z1 + z2


@flat.function('f')
def objective(x1, z2, y1, y2):
    return x1**2 + z2 + y1 + jnp.exp(-y2)


@flat.function('g1')
def con1(y1):
    return 1.0 - y1 / 3.16


@flat.function('g2')
def con2(y2):
    return y2 / 24.0 - 1.0


def sellar():
    return cx.absolute.declare(flat.residual(disc1, 'y1'), disc2, objective, con1, con2).graph


# ---------------------------------------------------------------- names


def test_a_builder_is_a_value():
    plasma = V.plasma
    assert plasma.ip == V.plasma.ip and hash(plasma.ip) == hash(V.plasma.ip)
    assert pickle.loads(pickle.dumps(plasma.ip)) == plasma.ip
    assert V.x != N.x


def test_what_is_not_a_key_is_refused():
    with pytest.raises(AttributeError, match='not a key'):
        V._private
    with pytest.raises(TypeError, match='not a path'):
        list(V.x)
    for index in (True, -1, 1.0, slice(1, 2), V.x):
        with pytest.raises(TypeError, match='an index is'):
            V[index]
    with pytest.raises(TypeError, match='hashable'):
        V[[1]]
    with pytest.raises(TypeError, match='minted'):
        V[purposes.Copy()]


def test_the_syntax_is_the_key_kind_and_any_other_key_a_dict_key():
    class Species(enum.Enum):
        D = 'deuterium'

    assert cx.absolute.place(V.eq[0]['psi']) == VarPath(
        (GetAttrKey('eq'), SequenceKey(0), DictKey('psi')))
    assert cx.absolute.place(V.ions[Species.D]) == VarPath((GetAttrKey('ions'), DictKey(Species.D)))
    assert cx.absolute.place(V.table['a', 1]) == VarPath((GetAttrKey('table'), DictKey(('a', 1))))
    assert cx.absolute.place(V.table[DictKey(0)]) == VarPath((GetAttrKey('table'), DictKey(0)))
    assert cx.absolute.place(V[0]) != cx.absolute.place(V[DictKey(0)])
    assert cx.absolute.node(N[GetAttrKey('x')]) == cx.absolute.node(N.x)


def test_a_builder_handed_where_a_path_is_read_says_so():
    from cottax.core import prefixes
    with pytest.raises(AttributeError, match='pass it to `place`'):
        list(prefixes(V['y1']))
    graph = sellar()
    for lookup in (lambda: N['disc2'] in graph, lambda: graph[N['disc2']],
                   lambda: V['y2'] in graph.graph.variables):
        with pytest.raises(TypeError, match='pass it to `place`'):
            lookup()


def test_a_root_and_the_other_namespace_are_refused():
    with pytest.raises(ValueError, match='root'):
        cx.absolute.place(V)
    with pytest.raises(TypeError) as refused:
        cx.absolute.place(N.x)
    assert refused.value.args[0] == NotAPlace('what is written', 'N.x').message
    with pytest.raises(TypeError) as refused:
        cx.absolute.node(V.x)
    assert refused.value.args[0] == NotANode('what is written', 'V.x').message


# ---------------------------------------------------------------- a wrapper

def test_a_wrapper_is_its_function():
    assert disc2(1.0, 2.0, 4.0) == 5.0
    assert inspect.signature(disc2) == inspect.signature(disc2.fn)


def test_a_decorated_function_is_the_node_s_body_and_still_pickles_by_reference():
    graph = cx.absolute.declare(disc2, objective).graph
    assert graph[n('disc2')].implementation is disc2.fn
    assert pickle.loads(pickle.dumps(graph)) == graph
    assert pickle.loads(pickle.dumps(disc2.fn)) is disc2.fn



def test_a_function_defined_elsewhere_is_not_decorated():
    with pytest.raises(TypeError, match='wrap it where it is used'):
        flat.function('y')(posixpath.basename)
    assert posixpath.basename.__qualname__ == 'basename'
    assert flat.function(posixpath.basename, 'y').fn is posixpath.basename


def test_a_wrapper_is_not_decorated_again_and_a_refusal_renames_nothing():
    def k(x, y=1.0):
        return x

    with pytest.raises(TypeError, match='held by value'):
        flat.function('y')(k)
    assert k.__qualname__.endswith('.k')
    with pytest.raises(TypeError, match='wrap it where it is used'):
        flat.function('z')(flat.function('y')(lambda x: x))


def test_a_wrapper_reads_one_place_per_parameter_in_signature_order():
    w = cx.absolute.function(disc2.fn, V.y2, reads={'y1': V.y1, 'z1': V.z1, 'z2': V.z2})
    assert tuple(r.place for r in w.function_reads) == tuple(map(cx.absolute.place, (V.z1, V.z2, V.y1)))



def test_a_setting_held_by_value_is_refused():
    def scaled(x, k=2.0):
        return k * x

    assert [type(found) for found in violations(
        cx.Explicit, scaled, (cx.PlaceWithValue(v('x')), cx.PlaceWithValue(v('k'))), (v('y'),))
    ] == [HeldByValue]
    with pytest.raises(TypeError, match='held by value'):
        flat.function(scaled, 'y')


def test_a_parameter_no_position_binds_is_refused():
    def keyword(x, *, k):
        return k * x

    def spread(*args):
        return args

    with pytest.raises(TypeError, match='keyword-only'):
        cx.absolute.function(keyword, V.y, reads={'x': V.x, 'k': V.k})
    assert [type(found) for found in violations(cx.Explicit, spread, (), (v('y'),))
            ] == [NotPositional, ReadsUnlike]


def test_reads_naming_another_parameter_are_refused():
    with pytest.raises(ValueError, match='one place per parameter'):
        cx.absolute.function(disc2.fn, V.y2, reads={'z1': V.z1, 'z2': V.z2, 'q': V.q})
    assert [type(found) for found in violations(
        cx.Explicit, disc2.fn, (cx.PlaceWithValue(v('z1')),), (v('y2'),))] == [ReadsUnlike]


def test_wrappers_owning_one_place_are_refused_as_a_graph():
    lo = flat.function(lambda z1: z1, 'y2')
    assert lo.function_outputs == disc2.function_outputs
    with pytest.raises(ValueError, match='owned by both'):
        cx.absolute.declare({'lo': lo}, disc2).graph
    held = flat.inequality_constraint(lambda y2: y2, 'g', within=(None, 1.0))
    opt  = flat.optimise(minimise='f', design=('z1',), subject_to=(held,))
    assert [type(f) for f in violations(declare, {'opt': opt})] == [HeldNotInModel]
    d = cx.absolute.declare({'g': held, 'opt': opt})
    assert d.graph.nodes == (n('g'), n('opt')) and dict(d.given) == {cx.absolute.place(cx.upper(v('g'))): 1.0}


def test_a_function_yet_to_be_written_is_declared_by_its_parameters():
    '''A description in place of the function: its parameters are all a wrapper needs, the
    node it declares has no implementation, and every proof short of one holds. Written,
    the function takes its place.'''
    from cottax.core import Function
    from cottax.core.maths.executable import ExecutableGraph
    from cottax.core.realisation.proof import NoImplementation, RealisedGraph
    sketch = flat.function(cx.Unimplemented(('x', 'k')), 'y', reads={'k': 2.0})     # `x` by name
    body   = cx.absolute.declare({'scale': sketch}).graph[n('scale')]
    assert type(body) is Function and body.reads == (v('x'), cx.absolute.place(cx.given.beside(V['y'], 'k')))
    ExecutableGraph(cx.absolute.declare({'scale': sketch}).graph)
    assert NoImplementation in [type(f) for f in
                                violations(RealisedGraph, cx.absolute.declare({'scale': sketch}).graph)]
    written = flat.function(lambda x, k: x * k, 'y', reads={'k': 2.0})
    assert cx.absolute.declare({'scale': written}).graph.nodes == cx.absolute.declare({'scale': sketch}).graph.nodes
    residual = flat.residual(cx.Unimplemented(('u', 'y')), 'u')         # the paired kinds too
    assert type(residual.function_node()) is Function
    with pytest.raises(ValueError, match='one place per parameter'):
        flat.function(cx.Unimplemented(('x',)), 'y', reads={'q': 'x'})


# ---------------------------------------------------------------- a wrapper where a place is due

def wrapped_sellar():
    def D1(x_1, z_1, z_2, y_2):
        return z_1**2 + z_2 + x_1 - 0.2 * y_2

    def D2(y_1, z_1, z_2):
        return jnp.sqrt(y_1) + z_1 + z_2

    def obj(x_1, z_2, y_1, y_2):
        return x_1**2 + z_2 + y_1 + jnp.exp(-y_2)

    def both(y_1, y_2):
        return y_1, y_2

    return (flat.function(D1, 'y_1'), flat.function(D2, 'y_2'), flat.function(obj, 'f'),
            flat.function(both, 'a', 'b'))


DESIGN = {'z_1': (-10, 10), 'z_2': (0, 10), 'x_1': (0, 10)}


def test_a_wrapper_where_a_place_is_due_stands_for_its_one_output():
    D1, D2, obj, _ = wrapped_sellar()
    C1 = flat.inequality_constraint(lambda y_1: 3.6 - y_1, 'g_1')
    assert flat.optimise(design=DESIGN, minimise=obj, subject_to=C1) \
        == flat.optimise(design=DESIGN, minimise='f', subject_to=C1)
    assert flat.place(D1) == v('y_1') and flat.requirement(D2) == flat.requirement('y_2')
    assert flat.node(obj) == flat.node('obj')                    # a node name is its own


def test_a_wrapper_with_several_outputs_is_refused_where_one_place_is_due():
    *_, both = wrapped_sellar()
    with pytest.raises(ValueError) as refused:
        flat.optimise(design=DESIGN, minimise=both)
    assert refused.value.violation == StandsForSeveral(
        "`optimise`'s `minimise`", repr(both), (v('a'), v('b')))


def test_a_wrapper_among_places_is_all_its_outputs():
    D1, _, _, both = wrapped_sellar()
    assert flat.places((both, 'z_1')) == (v('a'), v('b'), v('z_1'))
    assert flat.places(both) == (v('a'), v('b'))
    assert flat.optimise(design=(D1, 'z_1'), minimise='f').outputs == (v('y_1'), v('z_1'))
    assert flat.optimise(design=both, minimise='f').outputs == (v('a'), v('b'))


def test_a_wrapper_as_a_key_is_its_one_output_and_several_are_refused():
    D1, _, _, both = wrapped_sellar()
    assert flat.optimise(design={D1: (0, 1)}, minimise='f') \
        == flat.optimise(design={'y_1': (0, 1)}, minimise='f')
    with pytest.raises(ValueError) as refused:
        flat.optimise(design={both: (0, 1)}, minimise='f')
    assert isinstance(refused.value.violation, StandsForSeveral)
    with pytest.raises(ValueError) as refused:
        flat.inequality_constraint(lambda q: q, 'g', within={both: (None, 1)})
    assert isinstance(refused.value.violation, StandsForSeveral)


def test_a_problem_stands_for_no_place():
    *_, obj, _ = wrapped_sellar()
    problem = flat.optimise(design=DESIGN, minimise=obj)
    with pytest.raises(TypeError) as refused:
        flat.optimise(design=DESIGN, minimise=problem)
    assert isinstance(refused.value.violation, NotAPlace)


def test_a_wrapper_is_read_and_bounds_as_its_output():
    _, D2, _, _ = wrapped_sellar()
    def g(y_2):
        return y_2 - 24

    assert flat.function(g, 'g', reads={'y_2': D2}) == flat.function(g, 'g')
    assert flat.inequality_constraint(g, 'g', within=(None, D2)) \
        == flat.inequality_constraint(g, 'g', within=(None, 'y_2'))
    tagged = flat.function(g, 'g', reads={'y_2': cx.Static(D2)})
    assert tagged.function_reads[0].place == v('y_2') and tagged.function_reads[0].value is None


def test_bounds_are_a_pair_and_anything_else_is_refused():
    '''`(lo, hi)` per place: three sides, or a number alone, are refused naming the place,
    for a design as for a constraint's `within=`.'''
    for written in ((0.1, 0.2, 0.3), 5.0):
        with pytest.raises(TypeError) as refused:
            flat.optimise(design={'z': written}, minimise='f')
        assert refused.value.violation == NotAPair(v('z'), repr(written))
        with pytest.raises(TypeError) as refused:
            flat.inequality_constraint(lambda q: q, 'g', within=written)
        assert refused.value.violation == NotAPair(v('g'), repr(written))


# ---------------------------------------------------------------- the kinds


def test_several_outputs_are_returned_in_order():
    def both(z):
        return z, 2 * z

    assert cx.absolute.declare(flat.function(both, 'y1', 'y2')).graph[n('both')].owns == (v('y1'), v('y2'))


# ---------------------------------------------------------------- scopes

def test_a_scope_reads_each_parameter_under_its_root():
    def petty08(ip, b):
        return ip * b

    plasma = cx.Scope(V.physics.plasma, GetAttrKey)
    w = plasma.function(petty08, 'tau_e', reads={'b': V.magnets.b})
    assert w.function_outputs == (cx.absolute.place(V.physics.plasma.tau_e),)
    assert tuple(r.place for r in w.function_reads) == tuple(map(cx.absolute.place,
                                                (V.physics.plasma.ip, V.magnets.b)))


def test_a_scope_takes_a_place_or_a_builder_as_its_root():
    assert cx.Scope(cx.absolute.place(V.plasma), GetAttrKey).place('ip') == cx.absolute.place(V.plasma.ip)


def test_one_name_decorates_or_wraps_by_what_comes_first():
    '''A function first: wrapped where it is used; an output first: a decorator. The same
    wrapper either way, and a problem of the shape the kind names.'''
    plasma = cx.Scope(V.plasma, GetAttrKey)

    def relax(ip, rate):
        return (ip + rate) / 2

    for kind, shape in (('function', None), ('residual', is_root_find),
                        ('fixed_point', is_fixed_point)):
        decorated = getattr(plasma, kind)('ip')(relax)
        assert decorated == getattr(plasma, kind)(relax, 'ip')
        if shape is not None:
            problem = cx.absolute.declare(decorated).graph[cx.absolute.node(cx.solve(N['relax']))]
            assert shape(problem) and problem.unknowns == (cx.absolute.place(V.plasma.ip),)


def test_a_decorator_refuses_its_arguments_where_they_are_written():
    with pytest.raises(ValueError, match='reads are given for'):
        cx.absolute.function(V.y)(lambda x: x)
    with pytest.raises(TypeError, match='raeds'):
        flat.function('y', raeds={})
    with pytest.raises(TypeError, match='no signature'):
        cx.absolute.function(max, V.y, reads={})


def test_a_scope_writes_a_name_as_an_attribute_or_a_dict_key():
    with pytest.raises(TypeError, match='GetAttrKey'):
        cx.Scope(V.x, SequenceKey)


def test_a_strict_scope_reads_nothing_by_name():
    strict = cx.Scope(V, DictKey, by_name=False)
    reads = {'z1': 'z_1', 'z2': 'z_2'}
    with pytest.raises(ValueError, match='reads are given for'):
        strict.function(disc2.fn, 'y2', reads=reads)
    w = strict.function(disc2.fn, 'y2', reads=reads | {'y1': 'y_1'})
    assert w == flat.function(disc2.fn, 'y2', reads=reads | {'y1': 'y_1'})
    assert strict == flat.reading(cx.Shipped(by_name=False))
    assert strict.conventions.reads.by_name is False and flat.conventions.reads.by_name is True


def test_the_absolute_scope_names_nothing():
    with pytest.raises(TypeError, match='only in a scope'):
        cx.absolute.function(disc2.fn, 'y2', reads={p: V[p] for p in ('z1', 'z2', 'y1')})
    with pytest.raises(TypeError, match='only in a scope'):
        cx.absolute.optimise(minimise='f', design=(V.z1,))
    assert cx.absolute.function is not flat.function and cx.absolute.locator == cx.InFull()


def test_every_scope_writes_the_condition_nodes():
    design = cx.Scope(V.design, GetAttrKey)
    z1, z2, hi = (cx.absolute.place(p) for p in (V.design.z1, V.design.z2, V.design.hi))
    assert design.optimise(minimise=V.f, design={'z1': (0, 'hi'), 'z2': (None, None)}).condition == \
        cx.RelationalCondition((cx.Relation(None, z1, cx.Le), cx.Relation(z1, hi, cx.Le)),
                               (cx.Objective(cx.absolute.place(V.f), cx.Min),), (z1, z2), (hi,))
    assert flat.satisfy(design={'u': (0, None)}).condition.relations == \
        (cx.Relation(None, v('u'), cx.Le),)
    assert flat.requirement('c', cx.Le) == cx.absolute.requirement(V['c'], cx.Le)


def test_bounds_are_relations_over_parameters():
    '''`(lo, hi)` per unknown: a place is held fixed, `0` is zero, `None` no bound; a place
    bounding two unknowns is one parameter.'''
    o = flat.optimise(minimise='f', design={'a': ('lo', 'hi'), 'b': (0, None),
                                                   'c': ('lo', None)}).condition
    assert o.relations == (cx.Relation(v('lo'), v('a'), cx.Le), cx.Relation(v('a'), v('hi'), cx.Le),
                           cx.Relation(None, v('b'), cx.Le), cx.Relation(v('lo'), v('c'), cx.Le))
    assert o.parameters == (v('lo'), v('hi')) and o.conditions == (v('f'),)
    assert flat.satisfy(design={'a': (0, 'hi')}).condition.parameters == (v('hi'),)




# ---------------------------------------------------------------- a tree of models

@dataclasses.dataclass(frozen=True)
class Disciplines:
    first  : object
    second : object


class Pair(NamedTuple):
    first  : object
    second : object


def test_a_slot_names_what_fills_it_and_a_problem_follows_it():
    g = cx.absolute.declare({'mda': Disciplines(flat.residual(disc1, 'y1'), disc2)}).graph
    assert g.nodes == (cx.absolute.node(N['mda'].first), cx.absolute.node(cx.solve(N['mda'].first)),
                       cx.absolute.node(N['mda'].second))
    assert [name.spelling for name in g.nodes] == [
        "['mda'].first", "^solve['mda'].first", "['mda'].second"]


def test_slots_are_keyed_as_jax_keys_them():
    tree  = [Pair(disc2, None), (objective,)]
    names = cx.absolute.declare(tree).graph.nodes
    assert names == (cx.absolute.node(N[0].first), cx.absolute.node(N[1][0]))
    assert [p for p, _ in jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda x: isinstance(x, cx.Explicit))[0]] == [n.segments for n in names]


def test_a_mapping_is_bound_in_written_order_where_jax_sorts():
    assert cx.absolute.declare({'z': disc2, 'a': objective}).graph.nodes == (n('z'), n('a'))


def test_a_private_field_or_one_not_in_init_is_not_a_slot_and_holds_no_model():
    @dataclasses.dataclass(frozen=True)
    class Held:
        model : object
        _note : object

    @dataclasses.dataclass(frozen=True)
    class Derived:
        model : object
        extra : object = dataclasses.field(init=False, default=objective)

    assert cx.absolute.declare(Held(disc2, 2.0)).graph.nodes == (cx.absolute.node(N.model),)
    with pytest.raises(TypeError, match='no slot'):
        cx.absolute.declare(Held(disc2, objective)).graph
    with pytest.raises(TypeError, match='no slot'):              # ... nor anywhere under it
        cx.absolute.declare(Held(disc2, {'inner': [objective]})).graph
    with pytest.raises(TypeError, match='no slot'):
        cx.absolute.declare(Derived(disc2)).graph




def test_an_empty_slot_declares_nothing():
    assert cx.absolute.declare({'a': None, 'b': Disciplines(None, None)}).graph.nodes == ()
    assert cx.absolute.declare(Disciplines(disc2, None)).graph.nodes == (cx.absolute.node(N.first),)



def test_two_occupants_of_one_function_are_two_nodes():
    g = cx.absolute.declare(Disciplines(flat.function(disc2.fn, 'a'), flat.function(disc2.fn, 'b'))).graph
    assert g.nodes == (cx.absolute.node(N.first), cx.absolute.node(N.second))


def test_several_models_join_side_by_side_into_one_tree():
    '''`declare(a, b, ...)` reads one tree, joined: a wrapper at its function's name, every
    other tree's top-level entries merged; one tree joins to itself, a name twice is refused.'''
    assert cx.absolute.declare(disc2, {'o': objective}).graph == cx.absolute.declare({'disc2': disc2, 'o': objective}).graph
    with pytest.raises(ValueError, match='disc2'):
        cx.absolute.declare(disc2, {'disc2': objective}).graph


def test_a_loose_wrapper_is_named_by_its_function_and_a_lambda_or_a_callable_value_by_a_slot():
    @dataclasses.dataclass(frozen=True)
    class Scaled:
        k : float
        def __call__(self, y1):
            return self.k * y1

    assert cx.absolute.declare(disc2).graph.nodes == (n('disc2'),)
    for nameless in (lambda z1: z1, Scaled(2.0)):
        assert cx.absolute.declare({'copy': flat.function(nameless, 'y')}).graph.nodes == (n('copy'),)
        with pytest.raises(TypeError, match='no name of its own: put it in a slot'):
            cx.absolute.declare(flat.function(nameless, 'y')).graph


def test_a_node_definition_is_named_by_its_slot_only():
    requirement = cx.absolute.requirement(v('g'))
    assert cx.absolute.declare({'held': requirement}).graph[n('held')] == requirement
    with pytest.raises(TypeError, match='has no name'):
        cx.absolute.declare(requirement).graph


def test_a_setting_in_the_tree_is_refused():
    with pytest.raises(TypeError, match='is not a model'):
        cx.absolute.declare(Disciplines(disc2, 2.0)).graph


def test_a_mapping_slot_is_named_by_its_key():
    assert cx.absolute.declare({1: disc2}).graph.nodes == (cx.absolute.node(N[DictKey(1)]),)



def test_a_graph_is_added_to_by_inserting_anothers_definitions():
    base   = (Plan(sellar()) + cx.absolute.fixed_point_cut(v('y2'), [n('disc1')])).graph
    nested = Nest(cx.absolute.node(cx.solve(N['disc1'])), cx.absolute.node(cx.closure(N['y2']))).apply(base)
    out    = (Plan(nested) + Insert(cx.absolute.declare({'extra': flat.function(lambda f: f, 'h')}).graph
                                    .definitions)).graph
    assert out.nodes == (*nested.nodes, n('extra'))
    assert out.within == nested.within


# ---------------------------------------------------------------- ops

def test_cut_moves_the_readers_onto_a_hat():
    op = cx.absolute.cut(V['y2'], [N['disc1']])
    assert (op.var, op.readers, op.copy_of) == (v('y2'), (n('disc1'),), purposes.Copy())


def test_close_binds_one_cut_at_its_variable():
    op = cx.absolute.fixed_point_cut(V['y2'], [N['disc1']])
    assert op.at == cx.absolute.node(cx.closure(n('y2'))) and op.cuts == (cx.absolute.cut(V['y2'], [N['disc1']]),)


def test_a_scope_writes_a_path_relative_to_its_root_with_its_builder():
    '''A string is one name in the scope; `scope.locator.V` is any path under its root.'''
    tfcoil = cx.Scope(V.tfcoil, GetAttrKey)
    assert cx.absolute.place(tfcoil.locator.V.sig_tf) == tfcoil.place('sig_tf')
    assert cx.absolute.place(tfcoil.locator.V.coils[0].j) == cx.absolute.place(V.tfcoil.coils[0].j)
    assert cx.absolute.place(flat.locator.V['y2']) == flat.place('y2') and cx.absolute.locator.V is V
    w = tfcoil.function(lambda b_tf, rmajor: b_tf, tfcoil.locator.V.sig_tf, reads={'rmajor': V.physics.rmajor})
    assert tuple(r.place for r in w.function_reads) == (cx.absolute.place(V.tfcoil.b_tf), cx.absolute.place(V.physics.rmajor))


def test_read_in_reads_names_by_another_scopes_rule():
    '''`read_in={scope: names}`: those parameters read by name under that scope's rule, the
    rest by name in the declaring one; `reads` stays for explicit places. A name read twice,
    a name the function does not take, or anything but a name or a tuple of names is refused.'''
    tfcoil, physics = cx.Scope(V.tfcoil, GetAttrKey), cx.Scope(V.physics, GetAttrKey)
    def stress(b_tf, rmajor, kappa, j):
        return b_tf
    w = tfcoil.function(stress, 'sig_tf', read_in={physics: ('rmajor', 'kappa')},
                        reads={'j': tfcoil.locator.V.coils[0].j})
    assert tuple(r.place for r in w.function_reads) == tuple(map(cx.absolute.place, (
        V.tfcoil.b_tf, V.physics.rmajor, V.physics.kappa, V.tfcoil.coils[0].j)))
    one = tfcoil.function(stress, 'sig_tf', read_in={physics: 'rmajor'})
    assert one.function_reads[1].place == cx.absolute.place(V.physics.rmajor)       # a string is one name
    with pytest.raises(ValueError, match='read twice'):
        tfcoil.function(stress, 'sig_tf', read_in={physics: 'rmajor'}, reads={'rmajor': 'r'})
    with pytest.raises(ValueError, match='reads are given for'):
        tfcoil.function(stress, 'sig_tf', read_in={physics: 'nope'})
    with pytest.raises(TypeError, match='name or tuple'):
        tfcoil.function(stress, 'sig_tf', read_in={physics: ['rmajor']})
    with pytest.raises(TypeError, match='a surface'):
        tfcoil.function(stress, 'sig_tf', read_in={V.physics: 'rmajor'})


def test_a_cut_takes_a_place_written_in_full_and_refuses_a_string():
    assert cx.absolute.cut(V['y2'], ['disc1']) == cx.absolute.cut(VarPath((DictKey('y2'),)), [N['disc1']])
    assert cx.absolute.cut(V.plasma.ip, ['ohm']).var == cx.absolute.place(V.plasma.ip)
    assert cx.absolute.fixed_point_cut(V['y2'], ['disc1']).cuts == (cx.absolute.cut(V['y2'], ['disc1']),)
    with pytest.raises(TypeError, match='a string names a place only in a scope'):
        cx.absolute.cut('y2', ['disc1'])                                  # no root to read it under


def test_a_fixed_point_cut_closes_every_cut_at_one_problem():
    explicit = cx.absolute.declare({'disc1': flat.function(lambda z1, y2: z1 - 0.2 * y2, 'y1')},
                                   disc2).graph
    op  = flat.fixed_point_cut({'y1': 'disc2', 'y2': 'disc1'}, at=N.mda)
    out = (Plan(explicit) + op).graph
    assert op.at == cx.absolute.node(N.mda)
    assert out[op.at].unknowns == (cx.absolute.place(cx.copy(V['y1'])), cx.absolute.place(cx.copy(V['y2'])))
    assert out.graph.is_acyclic is False and out[n('disc2')].reads[-1] == cx.absolute.place(cx.copy(V['y1']))



def test_the_default_closing_binds_a_closure_in_its_own_namespace():
    g   = cx.absolute.declare({'disc1': flat.function(lambda z1, y2: z1 - 0.2 * y2, 'y1')}, disc2,
                      objective, {'opt': cx.absolute.optimise(minimise=v('f'), design=(v('z1'),))}).graph
    out = (Plan(g) + cx.flat.mdf(n('opt'))).graph
    closures = [p for p in out.problems if p != n('opt')]
    assert [p.segments[0] for p in closures] == [purposes.Closure()]


def test_an_architecture_takes_its_top_and_defaults_the_rest():
    from cottax.consumers.architectures import IDF, MDA, MDF, SAND
    from cottax.core.maths.cut_rules import GaussSeidel, Jacobi

    assert cx.flat.mdf(N.opt) == MDF(GaussSeidel(), cx.flat.closing, cx.absolute.node(N.opt))
    assert cx.flat.idf(N.opt) == IDF(GaussSeidel(), cx.flat.closing, cx.absolute.node(N.opt))
    assert cx.flat.sand(N.opt, Jacobi()) == SAND(Jacobi(), cx.flat.closing, cx.absolute.node(N.opt))
    assert cx.flat.mda() == MDA(GaussSeidel(), cx.flat.closing)


def test_implement_names_its_nodes_with_n():
    op = cx.absolute.implement({N['disc2']: disc2.fn})
    assert tuple(op.implementations) == (n('disc2'),)


# ---------------------------------------------------------------- end to end

# ---------------------------------------------------------------- the surface writes the ops

def test_each_edit_is_the_op_it_is_named_after_its_names_read_by_the_surface():
    g = sellar()
    at = cx.absolute.node(cx.solve(N['disc1']))
    assert flat.cut('y2', 'disc1') == cx.absolute.cut(v('y2'), [N['disc1']])
    assert flat.cut('y2', [disc2, 'obj']) == cx.absolute.cut(v('y2'), [N['disc2'], N['obj']])
    assert flat.fixed_point_cut('y2', 'disc1') == cx.absolute.fixed_point_cut(v('y2'), [N['disc1']])
    assert flat.fixed_point_cut({'y1': disc2, 'y2': 'disc1'}, at='mda') == cx.FixedPointCut(
        (cx.absolute.cut(v('y1'), ['disc2']), cx.absolute.cut(v('y2'), ['disc1'])), n('mda'))   # `at` is its name
    assert flat.assign(at, RootFinding(steps=False)) == cx.absolute.assign(at, RootFinding(steps=False))
    assert flat.nest('inner', 'outer') == Nest(n('inner'), n('outer'))
    assert flat.implement({disc2: disc2.fn}) == cx.absolute.implement({N['disc2']: disc2.fn})


def test_the_architectures_and_the_drive_are_the_surfaces_too():
    from cottax.core.maths.cut_rules import Jacobi
    assert flat.mdf('opt') == cx.flat.mdf('opt')
    assert flat.idf('opt', Jacobi()) == cx.flat.idf('opt', Jacobi())
    assert flat.mda() == cx.flat.mda()
    assert flat.drive(newton()) == cx.flat.drive(newton())


def test_a_name_of_the_wrong_kind_is_refused_naming_the_argument():
    with pytest.raises(TypeError) as refused:
        flat.fixed_point_cut((flat.place('y1'),), disc2)
    assert refused.value.args[0] == NotAPlace("`fixed_point_cut`'s `cuts`",
                                              repr((v('y1'),))).message
    with pytest.raises(TypeError) as refused:
        flat.cut('y2', flat.place('y1'))
    assert refused.value.args[0] == NotANode("`cut`'s `readers`", repr(v('y1'))).message
    with pytest.raises(TypeError, match='name the readers'):
        flat.fixed_point_cut('y2')


def newton():
    from cottax.relational.consumers.execution.jax_solvers import Newton
    return Newton(rtol=1e-10, atol=1e-10)


@pytest.mark.slow
def test_the_mda_runs_on_a_dict_to_sellars_reference_values(x64):
    g    = sellar()
    plan = Plan(g) + cx.absolute.assign(cx.absolute.node(cx.solve(N['disc1'])), RootFinding(steps=False))
    plan = plan + ImplementDrivers({RootFinder: newton()})
    graph = JitGraph(plan.graph)
    tree  = flat.run(graph, {cx.Inputs: {'z1': 5.0, 'z2': 2.0, 'x1': 1.0},
                             cx.Start: {'y1': 1.0}})
    assert tree['y1'] == pytest.approx(25.5883, abs=1e-3)
    assert tree['y2'] == pytest.approx(12.0585, abs=1e-3)
    assert tree['f']  == pytest.approx(28.5883, abs=1e-3)



@pytest.mark.slow
def test_the_mdf_absorbs_constraints_declared_where_computed(x64):
    """Sellar's two constraints as constraint wrappers, one against zero and one against
    a limit held in the tree: the MDF absorbs both requirements into the optimiser, the
    limit a parameter of it, and it reaches the reference optimum."""
    pytest.importorskip('scipy')
    from cottax.relational.consumers.execution.jax_solvers import SLSQP

    @flat.inequality_constraint('g1')
    def margin1(y1):
        return 1.0 - y1 / 3.16

    @flat.inequality_constraint('g2', within=(None, 'g2_max'))
    def margin2(y2):
        return y2 / 24.0

    opt   = flat.optimise(minimise='f', design={'z1': (-10, 10), 'z2': (0, 10), 'x1': (0, 10)})
    model = cx.absolute.declare(flat.residual(disc1, 'y1'), disc2, objective, margin1, margin2,
                       {'opt': opt}) + cx.flat.mdf(n('opt'))
    top   = model.graph[n('opt')]
    assert cx.Relation(v('g2'), v('g2_max'), cx.Le) in top.relations
    assert v('g2_max') in top.parameters and cx.absolute.node(cx.require(N['margin2'])) not in model.graph

    for p in model.graph.problems:
        driver = (ConstrainedMinimisation(steps=False) if p == n('opt')
                  else RootFinding(steps=False))
        model += cx.absolute.assign(p, driver)
    model += ImplementDrivers({RootFinder: newton(), ConstrainedMinimiser: SLSQP()})
    tree = flat.run(model, {cx.Inputs: {'g2_max': 1.0},
                            cx.Start: {'z1': 5.0, 'z2': 2.0, 'x1': 1.0, 'y1': 1.0,
                                       flat.copy('y2'): 1.0}})
    assert tree['f'] == pytest.approx(3.1834, abs=1e-3)
    assert tree['z1'] == pytest.approx(1.9776, abs=1e-3)


@pytest.mark.parametrize('kind', ['function', 'residual', 'fixed_point',
                                  'inequality_constraint', 'equality_constraint'])
def test_a_wrapper_writing_nothing_is_refused_where_it_is_written(kind):
    '''A bare `@kind` hands it the function alone: refused at once, naming the function, not
    later as an empty condition node.'''
    def g1(y1):
        return 3.16 - y1
    with pytest.raises(TypeError) as refused:
        getattr(flat, kind)(g1)
    assert refused.value.args[0] == NoOutputs('g1', type(getattr(flat, kind)(g1, 'g1')).__name__).message


def test_a_problem_is_subject_to_one_constraint_or_a_collection():
    margin = flat.inequality_constraint(lambda y1: y1, 'g1')
    one    = flat.optimise(minimise='f', design=('z1',), subject_to=margin)
    assert one == flat.optimise(minimise='f', design=('z1',), subject_to=(margin,))
    assert one.subject_to == (margin,)
    held = flat.requirement('r')
    assert flat.satisfy(design=('u',), subject_to=held).subject_to == (held,)


def test_a_problem_handed_over_loose_is_named_after_its_objective_and_a_satisfy_is_refused():
    opt = flat.optimise(minimise='f', design=('z1', 'z2', 'x1'))
    g   = cx.absolute.declare(flat.residual(disc1, 'y1'), disc2, objective, opt).graph
    at  = cx.absolute.node(cx.solve(N['f']))
    assert g.nodes[-1] == at and g[at] == opt.condition
    assert cx.absolute.node(opt) == at and cx.absolute.node(cx.solve(disc2)) == cx.absolute.node(cx.solve(N['disc2']))
    with pytest.raises(TypeError, match='has no name of its own: put it in a slot'):
        cx.absolute.declare(objective, flat.satisfy(design={'f': (0, 1)})).graph
    with pytest.raises(ValueError, match=r"duplicate key.*\^solve\['f'\]"):
        cx.absolute.declare(objective, opt, flat.optimise(minimise='f', design=('x1',))).graph


def test_a_declaration_stands_where_a_node_name_is_due():
    residual = flat.residual(disc1, 'y1')
    opt      = flat.optimise(minimise='f', design=('z1', 'z2', 'x1'))
    model    = cx.absolute.declare(residual, disc2, objective, opt)
    assert cx.flat.mdf(opt) == cx.flat.mdf(cx.absolute.node(cx.solve(N['f'])))
    assert cx.flat.idf(opt).top == cx.flat.sand(opt).top == cx.absolute.node(opt)
    assert cx.flat.mdf('opt') == cx.flat.mdf(N['opt']) and cx.flat.mdf(N.opt).top == cx.absolute.node(N.opt)
    assert cx.absolute.assign(opt, cx.Minimisation(steps=False)).problem == cx.absolute.node(opt)
    assert cx.absolute.cut(V['y2'], [residual]) == cx.absolute.cut(V['y2'], ['disc1'])
    assert cx.absolute.fixed_point_cut(V['y1'], [disc2]) == cx.absolute.fixed_point_cut(V['y1'], ['disc2'])
    assert cx.absolute.implement({disc2: disc2.fn}) == cx.absolute.implement({'disc2': disc2.fn})
    with pytest.raises(TypeError, match='has no name of its own'):
        cx.flat.mdf(flat.satisfy(design={'f': (0, 1)}))


def test_a_problem_in_a_slot_is_not_found_by_the_name_it_takes_loose():
    '''A slot's name wins: the problem itself then names a node the graph does not have.'''
    from cottax.core.maths.plan import refusals
    from cottax.consumers.architectures.refusals import TopUnbound
    opt   = flat.optimise(minimise='f', design=('z1', 'z2', 'x1'))
    model = cx.absolute.declare(flat.residual(disc1, 'y1'), disc2, objective, {'OPT': opt})
    assert TopUnbound(str(cx.absolute.node(opt))) in refusals(cx.flat.mdf(opt), model.graph)
    with pytest.raises(KeyError) as refused:
        model + cx.flat.mdf(opt)
    assert refused.value.args[0] == TopUnbound(str(cx.absolute.node(opt))).message


def test_a_problem_left_out_of_the_model_is_refused_as_the_top():
    '''A problem written but not declared: the architecture names what the model lacks.'''
    opt   = flat.optimise(minimise='f', design=('z1', 'z2', 'x1'))
    model = cx.absolute.declare(flat.residual(disc1, 'y1'), disc2, objective)
    with pytest.raises(KeyError, match='is no node of the model'):
        model + cx.flat.mdf(opt)
    with pytest.raises(KeyError, match='is no node of the model'):
        model + cx.flat.sand(opt)


@pytest.mark.slow
def test_sellar_declared_loose_reaches_the_optimum_of_the_problem_in_a_slot(x64):
    '''The study as written: one constraint bare, the problem loose and standing for its own
    node name; the same optimum as `{'OPT': opt}` and `mdf('OPT')`, under all three.'''
    pytest.importorskip('scipy')
    from cottax.assemblies.jax import SLSQP, Newton, Picard

    @flat.function('f')
    def N1(x, z_2, y_1, y_2):
        return x**2 + z_2 + y_1 + jnp.exp(-y_2)

    @flat.inequality_constraint('g')
    def N2(y_1):
        return 1 - y_1 / 3.16

    @flat.residual('y_1')
    def R3(y_1, z_1, z_2, x, y_2):
        return z_1**2 + z_2 + x - 0.2 * y_2 - y_1

    @flat.function('y_2')
    def N4(z_1, z_2, y_1):
        return jnp.sqrt(y_1) + z_1 + z_2

    bounds = {'z_1': (-10, 10), 'z_2': (0, 10), 'x': (0, 10)}
    opt    = flat.optimise(minimise='f', design=bounds, subject_to=N2)
    model  = cx.absolute.declare(N1, N2, R3, N4, opt)
    drive  = cx.flat.drive(SLSQP(), Picard(), Newton())
    design = {'z_1': 5., 'z_2': 2., 'x': 1.}
    guess  = {'mdf': {**design, 'y_1': 1., flat.copy('y_2'): 1.},    # each architecture's unknowns
              'idf': {**design, 'y_1': 1., flat.copy('y_2'): 1.},
              'sand': {**design, 'y_1': 1.}}                          # SAND cuts nothing here
    slotted = cx.absolute.declare(N1, N2, R3, N4, {'OPT': opt})
    for arch in ('mdf', 'idf', 'sand'):
        ran = model + getattr(cx.flat, arch)(opt) + drive
        ref = slotted + getattr(cx.flat, arch)('OPT') + drive
        out = flat.run(ran, {cx.Start: guess[arch]})
        ref = flat.run(ref, {cx.Start: guess[arch]})
        assert out['f'] == pytest.approx(3.1834, abs=1e-3)
        assert out['f'] == pytest.approx(ref['f'], abs=1e-6)
        for k in ('z_1', 'z_2', 'x'):
            assert out[k] == pytest.approx(ref[k], abs=1e-6)

    # One state serves all three: each start read at its unknown's place, a copy's at the
    # place of what it copies, and the answer written back.
    state = {'z_1': 5., 'z_2': 2., 'x': 1., 'y_1': 1., 'y_2': 1., 'f': None, 'g': None}
    mdf   = model + cx.flat.mdf(opt) + drive
    assert flat.valued(mdf, read_from={cx.Start: state}).values[
        flat.place(cx.start(flat.copy('y_2')))] == 1.0
    for arch in ('mdf', 'idf', 'sand'):
        ran = model + getattr(cx.flat, arch)(opt) + drive
        out = flat.run(ran, read_from={cx.Start: state})
        back = out.into(state)
        assert back['f'] == out['f'] == pytest.approx(3.1834, abs=1e-3)
        assert back['z_1'] == pytest.approx(1.9776, abs=1e-3)


def test_a_run_on_a_module_writes_back_into_its_fields_under_jit():
    import equinox as eqx

    class Design(eqx.Module):
        z : float

    class Out(eqx.Module):
        y : float | None = None

    class State(eqx.Module):
        design : Design
        out    : Out

    d, o  = cx.Scope(V.design, cx.GetAttrKey), cx.Scope(V.out, cx.GetAttrKey)
    model = cx.absolute.declare({'f': o.function(lambda z: 2 * z, 'y', reads={'z': d.place('z')})})
    state = State(Design(3.0), Out())
    out   = o.run(model, read_from={cx.Inputs: state})
    assert out.into(state) == State(Design(3.0), Out(6.0)) and out['y'] == 6.0
    jitted = eqx.filter_jit(lambda m, s: o.run(m, read_from={cx.Inputs: s}).into(s))
    assert float(jitted(model, state).out.y) == 6.0


def test_a_constraint_is_its_body_and_a_requirement_beside_it():
    def margin(y):
        return y

    for call, op, limit, wrote in ((flat.inequality_constraint, cx.Le, 'within', (None, 'g_max')),
                                   (flat.equality_constraint, cx.Eq, 'against', 'g_max')):
        g = cx.absolute.declare(call(margin, 'g')).graph
        assert g.nodes == (n('margin'), cx.absolute.node(cx.require(N['margin'])))
        assert g[n('margin')].owns == (v('g'),)
        assert g[cx.absolute.node(cx.require(N['margin']))] == cx.RelationalCondition(
            (cx.Relation(v('g'), None, op),), (), (), ())
        limited = call(margin, 'g', **{limit: wrote})
        assert cx.absolute.declare(limited).graph[cx.absolute.node(cx.require(N['margin']))].parameters == (v('g_max'),)
        assert call('g', **{limit: wrote})(lambda y: y).condition == limited.condition
        assert call(margin, 'g', **{limit: {'g': wrote}}) == limited


def test_an_inequality_is_held_within_its_bounds():
    t, lo, hi = v('t'), v('t_lo'), v('t_hi')
    window = flat.inequality_constraint(lambda t: t, 't', within=('t_lo', 't_hi')).condition
    assert window.relations == (cx.Relation(lo, t, cx.Le), cx.Relation(t, hi, cx.Le))
    assert window.parameters == (lo, hi)
    assert flat.inequality_constraint(lambda t: t, 't', within=(0, None)).condition.relations == \
        (cx.Relation(None, t, cx.Le),)
    both = flat.inequality_constraint(lambda a, b: (a, b), 'g1', 'g2', within={'g1': (None, 0), 'g2': (None, 'g2_max')})
    assert both.condition.relations == (cx.Relation(v('g1'), None, cx.Le),
                                       cx.Relation(v('g2'), v('g2_max'), cx.Le))


def test_a_number_stands_where_its_place_does():
    '''A number written at a side, a target or a read is a `PlaceWithValue`: the place minted for it
    and the value, standing where the place would. The value is data, not identity.'''
    g, g_hi, lo = v('g'), cx.absolute.place(cx.upper(v('g'))), cx.absolute.place(cx.lower(v('g')))
    window = flat.inequality_constraint(lambda g: g, 'g', within=(1.0, 2.0))
    assert window.bounds.at_most[g] == cx.PlaceWithValue(g_hi, 2.0) and window.bounds.at_most[g].value == 2.0
    assert window.bounds.parameters == (lo, g_hi) and dict(window.given) == {lo: 1.0, g_hi: 2.0}
    assert cx.PlaceWithValue(g_hi, 2.0) == cx.PlaceWithValue(g_hi, 3.0) and hash(cx.PlaceWithValue(g_hi, jnp.ones(2))) == hash(cx.PlaceWithValue(g_hi, 1.0))
    pair = flat.equality_constraint(lambda a, b: (a, b), 'h1', 'h2', against={'h1': 0, 'h2': 3.0})
    assert pair.targets == OrderedMap({v('h1'): None,
                                       v('h2'): cx.PlaceWithValue(cx.absolute.place(cx.target(v('h2'))), 3.0)})
    assert pair.condition.relations[1] == cx.Relation(v('h2'), cx.absolute.place(cx.target(v('h2'))), cx.Eq)
    stray = OrderedMap({v('h1'): None, v('h3'): None})
    assert [type(f) for f in violations(cx.EqualityConstraint, pair.fn, pair.function_reads,
                                        pair.outputs, targets=stray)] == [AgainstNoOutput,
                                                                          HeldAgainstNothing]
    with pytest.raises(ValueError, match=r"a limit given for \(VarPath\(\['h3'\]\),\)"):
        cx.EqualityConstraint(pair.fn, pair.function_reads, pair.outputs, targets=stray)
    with pytest.raises(ValueError, match='held against nothing'):
        flat.equality_constraint(lambda a, b: (a, b), 'h1', 'h2', against={'h1': 0})


def test_a_bound_s_number_is_given_as_written_and_crosses_a_jit_as_an_array():
    opt = flat.optimise(minimise='f', design={'z': (-1, 'z_hi')})
    assert isinstance(opt, cx.Problem) and opt.condition.parameters == (cx.absolute.place(cx.lower(v('z'))), v('z_hi'))
    declared = cx.absolute.declare({'opt': opt})
    assert tuple(declared.given.values()) == (-1,)             # as written
    (leaf,) = jax.tree_util.tree_leaves(declared)              # across a jit: an array, traced
    assert leaf.dtype.kind == 'i' and leaf == -1


def test_the_graph_is_the_same_for_every_value_given():
    square = flat.function(lambda z: z * z, 'f')
    graph  = lambda lo: cx.absolute.declare({'f': square,
                                     'opt': flat.optimise(minimise='f', design={'z': (lo, 9)})}).graph
    assert graph(-1) == graph(-5) and hash(graph(-1)) == hash(graph(-5))
    opt = flat.optimise(minimise='f', design={'z': (-1, 9)})
    with pytest.raises(ValueError, match='owned by both'):   # one place, one declaration
        cx.absolute.declare({'a': opt, 'b': opt})


def test_a_value_is_a_number_or_an_array_and_zero_is_no_place():
    import numpy as np
    import jax.numpy as jnp
    z, hi = v('z'), cx.absolute.place(cx.upper(v('z')))
    o = flat.optimise(minimise='f', design={'z': (np.float64(0), np.float32(2.0))})
    assert o.condition.relations == (cx.Relation(None, z, cx.Le), cx.Relation(z, hi, cx.Le))
    assert cx.absolute.declare({'o': o}).given[hi] == 2.0
    assert cx.absolute.declare({'o': flat.optimise(minimise='f', design={'z': (None, jnp.asarray(3.0))})}
                          ).given[hi] == 3.0
    box = cx.absolute.declare({'o': flat.optimise(minimise='f', design={'z': (None, np.arange(3))})}
                         ).given[hi]
    assert box.tolist() == [0, 1, 2]
    assert cx.absolute.place(V[np.int64(0)]) == cx.absolute.place(V[0])


def test_a_problem_subject_to_a_constraint_holds_it_without_an_architecture(x64):
    pytest.importorskip('scipy')
    from cottax.relational.consumers.execution.jax_solvers import SLSQP

    @flat.inequality_constraint('g', within=(None, 'g_max'))
    def margin(z):
        return z

    square = flat.function(lambda z: (z - 5.0) ** 2, 'f')
    opt    = flat.optimise(minimise='f', design=('z',), subject_to=(margin,))
    model  = cx.absolute.declare({'f': square, 'margin': margin, 'opt': opt}) + cx.flat.drive(SLSQP())
    assert cx.Relation(v('g'), v('g_max'), cx.Le) in model.graph[n('opt')].condition.relations
    assert cx.absolute.node(cx.require(N['margin'])) not in model.graph
    tree = flat.run(model, {cx.Inputs: {'g_max': 2.0}, cx.Start: {'z': 0.0}})
    assert tree['z'] == pytest.approx(2.0, abs=1e-6)

    with pytest.raises(ValueError, match='not in the model'):
        cx.absolute.declare({'f': square, 'opt': opt}).graph


def test_a_root_find_is_a_satisfy_over_equalities_held_bare():
    '''A root find is written as what it is subject to: bare requirements need no slot of
    their own, and the shape is read off the relations.'''
    held = [flat.requirement(c) for c in ('r1', 'r2')]
    g    = cx.absolute.declare({'b': flat.function(lambda u, w: (u, w), 'r1', 'r2'),
                        'top': flat.satisfy(design=('u', 'w'), subject_to=held)}).graph
    assert is_root_find(g[n('top')])
    assert g[n('top')].relations == (cx.Relation(v('r1'), None, cx.Eq), cx.Relation(v('r2'), None, cx.Eq))
    assert g.nodes == (n('b'), n('top'))


def test_a_model_runs_its_values_and_refuses_a_run_without_them(x64):
    pytest.importorskip('scipy')
    import equinox as eqx
    from cottax.relational.consumers.execution.jax_solvers import SLSQP

    square = flat.function(lambda z: (z - 5.0) ** 2, 'f')
    def model(hi):
        return cx.absolute.declare({'f': square, 'opt': flat.optimise(minimise='f', design={'z': (None, hi)})}
                          ) + cx.flat.drive(SLSQP())
    two, three = model(2.0), model(3.0)
    assert two == three                                  # one plan; the values are data
    with pytest.raises(KeyError) as refused:                # the graph alone holds no bound
        flat.run(JitGraph(two.graph), {cx.Start: {'z': 0.0}})
    assert refused.value.args[0] == Unvalued(f"`values[Inputs]` at {flat.place(cx.upper('z'))!r}").message

    traces = []
    @eqx.filter_jit
    def run(m, tree):
        traces.append(1)
        return flat.run(m, {cx.Start: {'z': tree}})['z']
    for m, hi in ((two, 2.0), (three, 3.0)):
        assert float(run(m, 0.0)) == pytest.approx(hi, abs=1e-6)
    assert len(traces) == 1


def test_drive_answers_each_problem_by_the_first_that_accepts_it(x64):
    from cottax.relational.consumers.execution.jax_solvers import Picard

    g = cx.absolute.declare({'cosine': flat.fixed_point(jnp.cos, 'u', reads={'x': 'u'})})
    driven = g + cx.flat.drive(Picard(rtol=1e-12, atol=1e-12), newton())
    assert driven.graph[cx.absolute.node(cx.solve(N['cosine']))].implementation == \
        Picard(rtol=1e-12, atol=1e-12)
    tree = flat.run(driven, {cx.Start: {'u': 1.0}})
    assert tree['u'] == pytest.approx(0.7390851332, abs=1e-8)


def test_a_tuple_of_reads_is_every_parameter_in_order():
    def d(z1, z2, coupling):
        return z1 + z2 + coupling

    w = flat.function(d, 'y', reads=('a', V['b'], 'c'))
    assert tuple(r.place for r in w.function_reads) == (v('a'), v('b'), v('c'))
    with pytest.raises(ValueError, match='names every parameter'):
        flat.function(d, 'y', reads=('a', 'b'))


def test_a_feasibility_of_constraints_alone_is_their_join_over_its_unknowns():
    c = flat.equality_constraint(lambda a: a - 1.0, 'r')
    find = flat.satisfy(design=('a',), subject_to=(c,))
    g = cx.absolute.declare({'r': c, 'find': find}).graph
    assert g.nodes == (n('r'), n('find'))                    # the requirement is held, not bound
    assert g[n('find')] == cx.RelationalCondition((cx.Relation(v('r'), None, cx.Eq),), (),
                                                  (v('a'),), ())
    with pytest.raises(ValueError, match='relates nothing'):
        flat.satisfy(design=('a',))
    boxed = flat.satisfy(design={'a': (0, 1.0)}, subject_to=(c,))
    assert [type(f) for f in violations(cx.Problem, (v('b'),), boxed.bounds, (), (c,))
            ] == [BoundNotUnknown]                                      # `a` bounded, `b` the unknown


def test_an_objective_is_minimised_or_maximised(x64):
    pytest.importorskip('scipy')
    from cottax.relational.consumers.execution.jax_solvers import SLSQP
    from cottax.relational import Max, Min, Objective

    hill = flat.function(lambda z: -(z - 5.0) ** 2, 'f')
    up   = flat.optimise(maximise='f', design={'z': (None, 2)})
    assert up.condition.objectives == (Objective(v('f'), Max),)
    assert flat.optimise(minimise='f', design=('z',)).condition.objectives == (Objective(v('f'), Min),)
    tree = flat.run(cx.absolute.declare({'f': hill, 'opt': up}) + cx.flat.drive(SLSQP()),
                    {cx.Start: {'z': 0.0}})
    assert tree['z'] == pytest.approx(2.0, abs=1e-6)
    for neither_or_both in ({}, {'minimise': 'f', 'maximise': 'f'}):
        with pytest.raises(TypeError, match='one objective'):
            flat.optimise(design=('z',), **neither_or_both)


def test_a_read_written_as_a_value_is_given():
    '''As a limit or a bound is: a place minted for it, the value beside the declaration.'''
    scale = flat.function(lambda x, k: x * k, 'y', reads={'k': 2.0})
    k = cx.absolute.place(cx.given.beside(V['y'], 'k'))
    assert scale.function_reads[1] == cx.PlaceWithValue(k, 2.0) and scale.given[k] == 2.0
    assert scale.function_node().reads == (v('x'), k)                        # the node holds places
    assert scale.inputs == (v('x'), k) and scale.outputs == (v('y'),)
    tree = flat.run(cx.absolute.declare({'scale': scale}), {cx.Inputs: {'x': 3.0}})
    assert tree['y'] == pytest.approx(6.0)
    with pytest.raises(TypeError):
        flat.function(lambda x, k: x * k, 'y', reads={'k': True})       # a bool is no value


def test_a_body_that_does_not_hash_still_runs_unjitted():
    class Body:
        __hash__ = None
        def __call__(self, x):
            return 2 * x
    d = cx.absolute.declare({'f': cx.flat.function(Body(), 'y', reads={'x': 'x'})})
    out = flat.run(d, {cx.Inputs: {'x': 1.0}})
    assert (out['x'], out['y']) == (1.0, 2.0)


def test_a_static_value_selects_and_a_traced_one_compiles_nothing():
    import equinox as eqx

    def tiled(x, n, c):
        return jnp.sum(jnp.full((n,), x)) * c

    def declared(n, c):
        return cx.absolute.declare({'t': flat.function(tiled, 'y', reads={
            'n': cx.Static(n), 'c': c})})

    traces = []
    @eqx.filter_jit
    def run(d, tree):
        traces.append(1)
        return flat.run(d, {cx.Inputs: tree})['y']

    assert float(run(declared(3, 1.0), {'x': 2.0})) == 6.0
    assert float(run(declared(3, 2.0), {'x': 2.0})) == 12.0     # traced: no trace
    assert float(run(declared(4, 1.0), {'x': 2.0})) == 8.0      # static: a new one
    assert len(traces) == 2


def test_using_replaces_or_adds_a_value_at_a_boundary_input():
    d = cx.absolute.declare({'f': flat.function(lambda z, k: z * k, 'f'),
                    'opt': flat.optimise(minimise='f', design={'z': (-1, 9)})})
    hi = flat.place(cx.upper('z'))
    wider = d.using({hi: 20.0})
    assert wider == d and wider.given[hi] == 20.0 and d.given[hi] == 9
    assert dict(d.using({v('k'): 2.0}).given) == {**d.given, v('k'): 2.0}
    with pytest.raises(KeyError) as refused:
        d.using({flat.place(cx.upper('y')): 1.0})
    assert refused.value.args[0] == NotAnInput((flat.place(cx.upper('y')),)).message
    n = cx.absolute.place(cx.given.beside(V['y'], 'n'))
    s = cx.absolute.declare({'t': flat.function(lambda x, n: x, 'y', reads={'n': cx.Static(3)})})
    assert s.using({n: 4}).given[n] == 4 and cx.tags_of(s, n, cx.Static) == (cx.Static(None),)


# ---------------------------------------------------------------- a tag on a variable

class Unit(Tagged):
    '''A toy fact about a variable, read by no one but this test.'''
    unit : str = '-'


def test_a_tag_is_written_on_a_read_by_name_or_path_on_an_output_and_on_a_value():
    import equinox as eqx

    def scaled(x, w, k):
        return x * w * k

    f = flat.function(scaled, Unit('y', 'm'),
                      reads={'x': Unit('x', 'm'), 'w': Unit(V['w'], 'kg'), 'k': Unit(2.0, 's')})
    d = cx.absolute.declare({'f': f})
    k = cx.absolute.place(cx.given.beside(V['y'], 'k'))
    assert d.graph[n('f')].reads == (v('x'), v('w'), k) and d.graph[n('f')].owns == (v('y'),)
    assert d.place_tags == OrderedMap({v('x'): (Unit(None, 'm'),), v('w'): (Unit(None, 'kg'),),
                                       k: (Unit(None, 's'),), v('y'): (Unit(None, 'm'),)})
    assert d.given == OrderedMap({k: 2.0})                           # a value given bare
    assert cx.tags_of(d, v('w'), Unit) == (Unit(None, 'kg'),)
    assert cx.tags_of(d, v('w'), cx.Static) == ()                    # a feature reads its own
    assert f == flat.function(scaled, 'y', reads={'x': 'x', 'w': 'w', 'k': Unit(2.0, 's')})

    tree = {'x': 3.0, 'w': 2.0}
    assert flat.run(d, {cx.Inputs: tree})['y'] == 12.0
    on_places = cx.absolute.declare({'f': flat.function(scaled, 'y', reads={'x': Unit('x', 'm'),
                                                                   'k': cx.Static(2.0)})})
    run = eqx.filter_jit(lambda d, tree: flat.run(d, {cx.Inputs: tree})['y'])
    assert float(run(on_places, tree)) == 12.0                 # the tags are in the jit key


def test_a_tag_on_an_implicit_model_s_output_is_on_its_unknown():
    d = cx.absolute.declare({'r': flat.residual(lambda u: u - 1.0, Unit('u', 'm'))})
    assert d.place_tags == OrderedMap({v('u'): (Unit(None, 'm'),)})


def test_a_tag_on_a_bound_a_target_or_a_design_bound_is_on_a_name_or_on_a_value():
    hi, x_hi = cx.absolute.place(cx.upper(v('g'))), cx.absolute.place(cx.upper(v('x')))
    named = flat.inequality_constraint(lambda x: x, 'g', within=(None, Unit('g_max', 'm')))
    assert named.bounds.at_most[v('g')] == cx.PlaceWithValue(v('g_max'))
    assert named.given == OrderedMap() and named.tags[v('g_max')] == (Unit(None, 'm'),)
    valued = flat.inequality_constraint(lambda x: x, 'g', within=(None, Unit(2.0, 'm')))
    assert valued.given == OrderedMap({hi: 2.0}) and valued.tags[hi] == (Unit(None, 'm'),)

    target = cx.absolute.place(cx.target(v('h')))
    named = flat.equality_constraint(lambda x: x, 'h', against=Unit('h_set', 'm'))
    assert named.targets == OrderedMap({v('h'): cx.PlaceWithValue(v('h_set'))})
    assert named.given == OrderedMap()
    assert named.tags[v('h_set')] == (Unit(None, 'm'),)
    valued = flat.equality_constraint(lambda x: x, 'h', against=Unit(3.0, 'm'))
    assert valued.given == OrderedMap({target: 3.0}) and valued.tags[target] == (Unit(None, 'm'),)

    d = cx.absolute.declare({'o': flat.optimise(minimise='f', design={'x': (None, Unit('xmax', 'm'))})})
    assert d.graph[n('o')].parameters == (v('xmax'),) and d.given == OrderedMap()
    assert cx.tags_of(d, v('xmax'), Unit) == (Unit(None, 'm'),)
    d = cx.absolute.declare({'o': flat.optimise(minimise='f', design={'x': (None, Unit(4.0, 'm'))})})
    assert d.given == OrderedMap({x_hi: 4.0}) and cx.tags_of(d, x_hi, Unit) == (Unit(None, 'm'),)


def test_a_string_minted_is_read_by_the_surface_s_own_rule():
    plasma = cx.Scope(V.plasma, GetAttrKey)
    assert plasma.place(cx.upper('ip')) == plasma.place(cx.upper(plasma.place('ip')))
    assert plasma.place(cx.upper('ip')) == cx.absolute.place(cx.upper(V.plasma.ip))
    assert plasma.place(cx.given.beside('ip', 'k')) == cx.absolute.place(cx.given.beside(V.plasma.ip, 'k'))
    with pytest.raises(TypeError, match='a string names a place only in a scope'):
        cx.absolute.place(cx.upper('ip'))


def test_a_tag_is_found_only_on_the_name_it_was_written_on():
    d = cx.absolute.declare({'a': flat.function(lambda x: x, Unit('y', 'm')),
                    'b': flat.function(lambda y: y, 'z')})
    cut  = d + cx.absolute.cut(V['y'], ['b'])
    copy = flat.place(cx.copy(v('y')))
    assert copy in cut.graph[n('b')].reads and cut.place_tags == d.place_tags
    assert cx.tags_of(cut, v('y'), Unit) == (Unit(None, 'm'),)
    assert cx.tags_of(cut, copy, Unit) == ()                         # a copy is another name


def test_a_place_renamed_is_another_place_with_no_value_or_tag_of_its_own():
    d = cx.absolute.declare({'f': flat.function(lambda x, k: x * k, 'y',
                                       reads={'x': Unit('x', 'm'), 'k': 2.0})})
    k, k2 = cx.absolute.place(cx.given.beside(V['y'], 'k')), cx.absolute.place(cx.copy(V['k']))
    renamed = d + Rename(n('f'), ((v('x'), v('x2')), (k, k2)))
    assert (renamed.given, renamed.place_tags) == (d.given, d.place_tags)
    assert cx.tags_of(renamed, v('x2'), Unit) == ()
    with pytest.raises(KeyError) as refused:
        flat.run(renamed, {cx.Inputs: {'x2': 3.0}})
    assert refused.value.args[0] == Unvalued(f'`values[Inputs]` at {k2!r}').message


def test_a_limit_is_a_place_or_zero_and_every_output_is_named_or_none_is():
    two = lambda a, b: (a, b)
    with pytest.raises(ValueError, match='held against nothing'):
        flat.inequality_constraint(two, 'g1', 'g2', within={'g2': (None, 'g2_max')})
    with pytest.raises(TypeError, match='for 2 places'):
        flat.inequality_constraint(two, 'g1', 'g2', within=(None, 'g_max'))
    with pytest.raises(ValueError, match='does not output'):
        flat.inequality_constraint(lambda y: y, 'g', within={'g': (None, 0), 'h': (None, 'h_max')})
    with pytest.raises(ValueError, match='reads are given for'):
        cx.Scope(V, DictKey, by_name=False).inequality_constraint(lambda y: y, 'g')


@dataclasses.dataclass(frozen=True)
class Plasma:
    confinement : object
    current     : object


def test_a_structured_model_runs_by_its_scoped_names(x64):
    here = cx.Scope(V.plasma, GetAttrKey)

    def square(ip, target):
        return ip * ip - target

    def double(ip):
        return 2.0 * ip

    model = {'plasma': Plasma(here.function(double, 'tau'), here.residual(square, 'ip'))}
    g     = cx.absolute.declare(model).graph
    at    = cx.absolute.node(cx.solve(N['plasma'].current))
    plan  = (Plan(g) + cx.absolute.assign(at, RootFinding(steps=False))
             + ImplementDrivers({RootFinder: newton()}))

    out = here.run(JitGraph(plan.graph), {cx.Inputs: {'target': 9.0}, cx.Start: {'ip': 1.0}})
    assert out['ip'] == pytest.approx(3.0, rel=1e-10)               # read under `.plasma`
    assert out[V.plasma.tau] == pytest.approx(6.0, rel=1e-10)


# ---------------------------------------------------------------- run on a caller's tree

def square(u, z):
    return u * u - z


def double(u):
    return 2.0 * u


def test_an_input_has_one_source():
    from cottax.relational.consumers.execution.jax_solvers import Newton
    graph   = cx.absolute.declare(flat.residual(square, 'u')).graph
    problem = cx.absolute.node(cx.solve(N['square']))
    solved  = JitGraph((Plan(graph) + cx.absolute.assign(problem, RootFinding(steps=False))
                        + cx.absolute.implement({problem: Newton(rtol=1e-10, atol=1e-10)})).graph)
    start   = purposes.Start()(flat.place('u'))          # `Start`'s place, after its role
    with pytest.raises(ValueError) as refused:
        flat.run(solved, {cx.Inputs: {'z': 9.0, start: 2.0}, cx.Start: {'u': 1.0}})
    assert refused.value.args[0] == ValuedTwice(
        start, (f'values[Inputs][{start!r}]', "values[Start]['u']")).message
    assert flat.run(solved, {cx.Inputs: {'z': 9.0}, cx.Start: {'u': 1.0}})['u'] == pytest.approx(3.0)


def test_assign_refuses_a_node_that_is_no_condition_node_by_name():
    from cottax.core.driven.ops import NotAConditionalNode
    model = flat.declare(disc2)
    with pytest.raises(TypeError) as refused:
        model + flat.assign('disc2', FixedPointIteration(steps=False))
    assert refused.value.args[0] == NotAConditionalNode(
        n('disc2'), type(model.graph[n('disc2')]).__name__).message


def test_driver_data_is_refused_where_no_driver_takes_it():
    from cottax.consumers.execution.refusals import NotAnUnknown, NotDriverData, TakesNoValue
    from cottax.relational.consumers.execution.jax_solvers import Newton
    model   = flat.declare(flat.residual(square, 'u'), {'third': flat.function(lambda z: z / 3, 'g0')})
    problem = cx.absolute.node(cx.solve(N['square']))
    driven  = model + flat.drive(Newton())
    for of, data, kind in ((model, {cx.Start: {'u': 1.0}}, NotDriverData),    # no driver
                           (driven, {cx.Times: {'u': (0., 1.)}}, NotDriverData), # not its role
                           (driven, {cx.Start: {'w': 1.0}}, NotAnUnknown),      # nothing owns it
                           (driven, {cx.Start: {'g0': 1.0}}, NotAnUnknown)):    # computed
        with pytest.raises(KeyError) as refused:
            flat.valued(of, {cx.Inputs: {'z': 9.0}, **data})
        assert type(refused.value.violation) is kind
    start = purposes.Start()(flat.place('u'))
    warm  = driven + Rename(problem, ((start, flat.place('g0')),))              # a warm start
    with pytest.raises(ValueError) as refused:
        flat.valued(warm, {cx.Inputs: {'z': 9.0}, cx.Start: {'u': 1.0}})
    assert refused.value.args[0] == TakesNoValue(problem, 'Start', flat.place('g0')).message
    assert flat.run(warm, {cx.Inputs: {'z': 9.0}})['u'] == pytest.approx(3.0)      # needs none


def test_a_purpose_is_called_on_a_name_as_written_and_the_surface_reads_what_it_derives():
    from cottax.core import Closure, Copy, mint_of
    y = flat.place('y')
    assert flat.place(cx.copy('y')) == Copy()(y) and str(Copy()(y)) == "^copy['y']"
    assert mint_of(flat.place(cx.copy('y'))) == Copy()
    assert flat.node(cx.closure('d1')) == Closure()(flat.node('d1'))
    assert flat.place(cx.start(cx.copy('y'))) == cx.start.mint(Copy()(y))      # purposes stack
    assert flat.places((cx.copy('y'), 'z')) == (Copy()(y), flat.place('z'))
    assert flat.place(cx.given.beside('y', 'k')) == cx.given.mint(flat.derived.beside(y, 'k'))
    with pytest.raises(TypeError):
        Copy()('y')                                         # core takes exact names only
    with pytest.raises(TypeError):
        flat.node(cx.given.beside('y', 'k'))                # one deeper is a place, never a node

def test_a_loose_paired_wrapper_names_its_condition_node():
    @flat.residual('u')
    def balance(u, y):
        return y - u

    limit = flat.inequality_constraint(lambda x: x, 'g')
    d = declare(balance, {'body': flat.function(lambda x: 2 * x, 'y'), 'lim': limit})
    assert balance.problem == flat.node(cx.solve('balance')) and balance.problem in d.graph.nodes
    with pytest.raises(TypeError, match='has no name of its own'):
        limit.requirement
