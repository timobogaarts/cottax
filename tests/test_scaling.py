'''
A driver's `Scaling`: one value at one place, opted into by the contract. What each key means
and which are refused, the arithmetic SLSQP is handed, and a real IDF through the surface:
written by name, given as a value, computed by a node, and written out. Float64.
'''

import json

import jax
import jax.numpy as jnp
import pytest

import cottax.assemblies.jax as cx
from cottax.assemblies.jax import flat
from cottax.consumers.json import read_json, write_json
from cottax.core.frozen import OrderedMap
from cottax.pytree.path import VarPath
from cottax.relational import Eq, Le, Max, Min, Objective, Relation, RelationalCondition, Scaling
from cottax.relational.consumers.execution.jax_solvers import scipy_minimize
from cottax.relational.consumers.execution.jax_solvers.canonical_nlp import (
    Box, Canonical, canonical, scaled, scaled_unknowns, unscaled_unknowns,
)
from cottax.relational.contracts import CONTRACTS, ConstrainedMinimisation, Scales
from cottax.core.driven import Start
from cottax.relational.programme import (
    ScaleOnABound, ScaleOnNoConstraint, ScaleOnNoUnknown, check_scaling, scales_by_group,
)
from jax.tree_util import DictKey


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def V(name: str) -> VarPath:
    return VarPath((DictKey(name),))


x, f, g, h, hi, k = map(V, ('x', 'f', 'g', 'h', 'hi', 'k'))
below = Relation(g, hi, Le)
held  = Relation(h, k, Eq)
bound = Relation(x, hi, Le)
FORM  = RelationalCondition((below, held, Relation(g, None, Le), bound), (Objective(f, Min),),
                            (x,), (hi, k))


# ---------------------------------------------------------------- what a key means

def test_what_is_left_out_has_no_scale():
    assert scales_by_group(FORM, Scaling()) == ((None,), (None,), (None, None))


def test_a_place_means_every_constraint_about_it_and_a_relation_itself():
    '''`h = k` is about `h`, a parameter being no subject; `g` has two rows, one exact.'''
    scaling = Scaling(OrderedMap({x: 3.0}),
                      constraints=OrderedMap({h: 5.0, g: 7.0, below: 11.0}))
    assert scales_by_group(FORM, scaling) == ((3.0,), (5.0,), (11.0, 7.0))


def test_a_refused_place_says_how_to_key_a_constraint_about_it_on_another_side():
    '''`g <= h` is about `g`: `h` means nothing here, and its `Relation` is the way to it.'''
    form = RelationalCondition((Relation(g, h, Le),), (), (x,), ())
    with pytest.raises(ValueError, match='keyed by its `Relation`'):
        scales_by_group(form, Scaling(constraints=OrderedMap({h: 2.0})))
    exact = Scaling(constraints=OrderedMap({Relation(g, h, Le): 2.0}))
    assert scales_by_group(form, exact)[2] == (2.0,)


def test_a_side_of_zero_or_a_parameter_is_no_subject():
    first_side_zero = Relation(None, g, Le)
    first_side_parameter = Relation(k, h, Eq)
    form = RelationalCondition((first_side_zero, first_side_parameter), (), (x,), (k,))
    scaling = Scaling(constraints=OrderedMap({g: 2.0, h: 3.0}))
    assert scales_by_group(form, scaling)[1:] == ((3.0,), (2.0,))


@pytest.mark.parametrize('scaling, refusal', [
    (Scaling(OrderedMap({V('gone'): 1.0})), ScaleOnNoUnknown),
    (Scaling(constraints=OrderedMap({V('gone'): 1.0})), ScaleOnNoConstraint),
    (Scaling(constraints=OrderedMap({Relation(V('gone'), None, Eq): 1.0})), ScaleOnNoConstraint),
    (Scaling(constraints=OrderedMap({bound: 1.0})), ScaleOnABound),
    (Scaling(constraints=OrderedMap({x: 1.0})), ScaleOnABound),
])
def test_a_key_saying_nothing_of_this_problem_is_refused(scaling, refusal):
    assert [type(v) for v in check_scaling(FORM, scaling)] == [refusal]
    with pytest.raises(ValueError, match='a scale'):
        scales_by_group(FORM, scaling)


# ---------------------------------------------------------------- the arithmetic

def test_each_is_read_over_its_scale_and_the_box_over_its_unknowns():
    form = RelationalCondition((Relation(h, None, Eq), Relation(g, None, Le), bound,
                                Relation(V('lo'), x, Le)),
                               (Objective(f, Min),), (x,), (hi, V('lo')))
    scaling = Scaling(OrderedMap({x: 2.0}), 10.0, OrderedMap({h: 4.0, g: 5.0}))
    values = (20.0, 8.0, 15.0, 6.0, -6.0)
    order = form.reads
    at = dict(zip((f, h, g, hi, V('lo')), values))
    read, limits = scaled(form, (1.0,), tuple(at[p] for p in order), scaling)
    assert read == Canonical((2.0,), (2.0,), (3.0,))
    assert limits == Box((-3.0,), (3.0,))


def test_nothing_stated_changes_nothing():
    '''The very values, not values equal to them: an int stays one, no dtype moves.'''
    form = RelationalCondition((Relation(h, None, Eq),), (Objective(f, Min),), (x,), ())
    values = tuple({f: 3, h: 4}[p] for p in form.reads)
    read, _ = scaled(form, (1,), values, Scaling())
    assert read == canonical(form, (1,), values)
    assert scaled_unknowns(form, (1,), Scaling()) == (1,)


def test_a_scale_goes_by_its_magnitude_and_keeps_the_sense_of_a_row():
    form = RelationalCondition((Relation(g, None, Le),), (Objective(f, Min),), (x,), ())
    values = tuple({f: 4.0, g: 6.0}[p] for p in form.reads)
    scaling = Scaling(objective=-2.0, constraints=OrderedMap({g: -3.0}))
    read, _ = scaled(form, (1.0,), values, scaling)
    assert read == Canonical((2.0,), (), (2.0,))
    assert scaled_unknowns(form, (6.0,), Scaling(OrderedMap({x: -2.0}))) == (3.0,)


def test_a_maximised_objective_stays_negated_under_a_negative_scale():
    '''Canonically minimised first, then over the magnitude: the sign is the sense's alone.'''
    form = RelationalCondition((), (Objective(f, Max),), (x,), ())
    read, _ = scaled(form, (1.0,), (4.0,), Scaling(objective=-2.0))
    assert read.objectives == (-2.0,)


def test_a_scale_broadcasts_to_what_it_scales_and_never_the_reverse():
    '''A scale per entry of a scalar unknown is refused here: it would turn the start into a
    vector, and fail far away in the solver.'''
    form = RelationalCondition((), (Objective(f, Min),), (x,), ())
    with pytest.raises(ValueError, match='broadcast'):
        scaled_unknowns(form, (1.0,), Scaling(OrderedMap({x: jnp.asarray([2.0, 3.0])})))


def test_a_single_bound_on_an_array_unknown_takes_each_entrys_scale():
    form = RelationalCondition((bound,), (Objective(f, Min),), (x,), (hi,))
    at = {f: 0.0, hi: 4.0}
    _, limits = scaled(form, (jnp.ones(2),), tuple(at[p] for p in form.reads),
                       Scaling(OrderedMap({x: jnp.asarray([2.0, 4.0])})))
    assert limits.upper[0].tolist() == [2.0, 1.0]
    assert limits.lower == (None,)


def test_anything_but_a_scaling_is_refused_by_name():
    with pytest.raises(TypeError, match='one `Scaling`, not a dict'):
        scales_by_group(FORM, {'unknowns': {x: 1.0}})


def test_an_unknowns_scale_is_undone_exactly_and_may_be_per_entry_or_per_leaf():
    form = RelationalCondition((), (Objective(f, Min),), (x,), ())
    array = jnp.asarray([1.0, 2.0])
    for value, scale in ((array, 2.0), (array, jnp.asarray([2.0, 4.0])),
                         ({'a': 1.0, 'b': 2.0}, 2.0), ({'a': 1.0, 'b': 2.0}, {'a': 2.0, 'b': 4.0})):
        scaling = Scaling(OrderedMap({x: scale}))
        back = unscaled_unknowns(form, scaled_unknowns(form, (value,), scaling), scaling)
        close = jax.tree.map(lambda a, b: jnp.allclose(a, b).item(), back, (value,))
        assert all(jax.tree.leaves(close))


def test_a_scale_does_not_raise_a_lower_precision():
    form = RelationalCondition((), (Objective(f, Min),), (x,), ())
    single = jnp.asarray(1.0, jnp.float32)
    (solved,) = scaled_unknowns(form, (single,), Scaling(OrderedMap({x: 2.0})))
    assert solved.dtype == jnp.float32


# ---------------------------------------------------------------- a study through the surface

M = 1e6


def study(*beside, architecture=flat.idf, **drive):
    '''A problem whose coupling has magnitude `M`, the same problem for every `M`: `z = 1`,
    `y2 = 1.21488916 M`. An IDF unless told otherwise.'''

    @flat.function('y1')
    def d1(z, y2):
        return M * (z + 0.2 * y2 / M)

    @flat.function('y2')
    def d2(z, y1):
        return M * (jnp.sqrt(jnp.abs(y1 / M)) + 0.1 * z)

    @flat.function('f')
    def objective(z, y1, y2):
        return (z - 2.0) ** 2 + 0.1 * y1 / M + 0.1 * y2 / M

    @flat.inequality_constraint('g')
    def con(z):
        return z - 1.0

    opt = flat.optimise(minimise='f', design=('z',), subject_to=con)
    return (cx.absolute.declare(d1, d2, objective, con, *beside, {'OPT': opt}) + architecture('OPT')
            + flat.drive(cx.SLSQP(), cx.Picard(), cx.Newton(), **drive))


COPY = flat.copy('y2')
STARTS = {'z': 0.0, COPY: M}
OF_M = flat.scaling(unknowns={COPY: M}, constraints={COPY: M})


def test_the_default_contract_takes_no_scales():
    assert ConstrainedMinimisation(steps=False).takes == (Start,)
    assert ConstrainedMinimisation(steps=False, scaled=True).takes == (Start, Scales)
    assert ConstrainedMinimisation(steps=False) in CONTRACTS
    with pytest.raises(KeyError, match='takes Start'):
        flat.run(study(), {cx.Start: STARTS, Scales: {'z': OF_M}})


def test_a_scaling_gets_the_answer_a_badly_scaled_idf_otherwise_misses():
    '''Unscaled, scipy calls `y2`'s relation met while it is still far off.'''
    without = flat.run(study(contracts=cx.SCALED), {cx.Start: STARTS, Scales: {'z': Scaling()}})
    plain   = flat.run(study(), {cx.Start: STARTS})
    out     = flat.run(study(contracts=cx.SCALED), {cx.Start: STARTS, Scales: {'z': OF_M}})
    assert float(without['z']) == float(plain['z'])
    assert abs(float(plain['z']) - 1.0) > 1e-3
    assert float(out['z']) == pytest.approx(1.0, abs=1e-6)
    assert float(out['y2']) / M == pytest.approx(1.21488916, abs=1e-6)
    assert out[Scales, 'z'] == OF_M


def test_a_scaling_and_its_scales_may_be_traced():
    ran = study(contracts=cx.SCALED)
    def solved(scale):
        return flat.run(ran, {cx.Start: STARTS, Scales: {'z': flat.scaling(
            unknowns={COPY: scale}, constraints={COPY: scale})}})['z']
    assert float(jax.jit(solved)(jnp.asarray(M))) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('scale', [0.0, float('nan'), float('inf')])
@pytest.mark.parametrize('traced', [False, True])
@pytest.mark.parametrize('of', ['unknown', 'constraint', 'objective'])
def test_a_scale_neither_finite_nor_non_zero_is_refused_by_name(scale, traced, of):
    '''Not a solver's failure to converge at the end of its iterations: under a jit too, where
    the scale is traced and SLSQP runs inside a callback.'''
    ran = study(contracts=cx.SCALED)
    def solved(s):
        return flat.run(ran, {cx.Start: STARTS, Scales: {'z': flat.scaling(
            unknowns={COPY: s} if of == 'unknown' else {},
            objective=s if of == 'objective' else None,
            constraints={COPY: s} if of == 'constraint' else {})}})['z']
    with pytest.raises(Exception, match='a scale must be finite and non-zero'):
        jax.block_until_ready((jax.jit(solved) if traced else solved)(jnp.asarray(scale)))


def test_scipy_is_handed_everything_in_the_scaling(monkeypatch):
    '''`min (x - 3)^2` on `1e8 (x - 1) = 0` with `x <= 4`, its unknown in units of 2: from
    `x = 0` the start is 0, the objective 9 over 3, the residual `-1e8` over `1e8`, the box 2.'''
    handed = {}
    minimize = scipy_minimize.minimize
    def recording(fun, x0, *, bounds, constraints, **options):
        handed.update(objective=fun(x0), bounds=bounds,
                      residual=[c['fun'](x0).tolist() for c in constraints])
        return minimize(fun, x0, bounds=bounds, constraints=constraints, **options)
    monkeypatch.setattr(scipy_minimize, 'minimize', recording)

    @flat.function('f', 'h')
    def model(x):
        return (x - 3.0) ** 2, 1e8 * (x - 1.0)

    opt = flat.optimise(minimise='f', design={'x': (None, 4.0)},
                        subject_to=flat.requirement(flat.place('h')))
    ran = cx.absolute.declare(model, {'OPT': opt}) + flat.drive(cx.SLSQP(), contracts=cx.SCALED)
    out = flat.run(ran, {cx.Start: {'x': 0.0}, Scales: {'x': flat.scaling(
        unknowns={'x': 2.0}, objective=3.0, constraints={'h': 1e8})}})
    assert handed == {'objective': 3.0, 'bounds': [(None, 2.0)], 'residual': [[-1.0]]}
    assert float(out['x']) == pytest.approx(1.0, abs=1e-6)


def test_a_key_of_another_architecture_is_refused_when_it_runs():
    '''Written for IDF's copy: MDF's problem has no such unknown or constraint.'''
    ran = study(architecture=flat.mdf, contracts=cx.SCALED)
    with pytest.raises(ValueError, match='a scale at'):
        flat.run(ran, {cx.Start: {'z': 0.0, COPY: M}, Scales: {'z': OF_M}})


# ---------------------------------------------------------------- at a place of the graph

def test_a_scaling_is_a_read_of_the_graph_and_drawn():
    ran = study(contracts=cx.SCALED)
    place = flat.place(cx.scales('OPT'))
    assert place in ran.graph.graph.boundary_inputs
    assert 'scales' in cx.render_xdsm_svg(ran)
    assert place not in study().graph.graph.boundary_inputs


def test_a_node_may_compute_the_scaling():
    '''A value like any other: read from the model's inputs, and rewired by what computes it.'''
    @flat.function(cx.scales('OPT'))
    def scaling_from(magnitude):
        return flat.scaling(unknowns={COPY: magnitude}, constraints={COPY: magnitude})

    ran = study(scaling_from, contracts=cx.SCALED)
    out = flat.run(ran, {cx.Inputs: {'magnitude': M}, cx.Start: STARTS})
    assert float(out['z']) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match='it takes no value'):
        flat.run(ran, {cx.Inputs: {'magnitude': M}, cx.Start: STARTS, Scales: {'z': OF_M}})


def test_a_scale_read_from_the_solve_itself_is_refused():
    '''A driver reads its data before its block runs: a scale cannot follow its own answer.'''
    @flat.function(cx.scales('OPT'))
    def scaling_from(y2):
        return flat.scaling(unknowns={COPY: y2})

    with pytest.raises(ValueError, match='driver'):
        flat.run(study(scaling_from, contracts=cx.SCALED), {cx.Start: STARTS})


# ---------------------------------------------------------------- written out

def test_a_scaled_contract_is_written_and_read_back():
    written = json.loads(json.dumps(write_json(study(contracts=cx.SCALED).graph, False)))
    back = read_json(written)
    opt = cx.absolute.node('OPT')
    assert back[opt].contract == ConstrainedMinimisation(steps=False, scaled=True)


def test_a_contract_written_before_scaled_existed_reads_back_unscaled():
    written = json.loads(json.dumps(write_json(study().graph, False)))
    def without(node):
        if isinstance(node, dict):
            node.get('fields', {}).pop('scaled', None)
            for v in node.values():
                without(v)
        elif isinstance(node, list):
            for v in node:
                without(v)
    without(written)
    back = read_json(written)[cx.absolute.node('OPT')]
    assert back.contract == ConstrainedMinimisation(steps=False)
