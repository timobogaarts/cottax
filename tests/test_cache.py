'''
Answering a query must leave what jax, equality and hashing see exactly as it was.

Every value is a plain frozen dataclass, so a `functools.cached_property` answer lives in
`__dict__`, where neither a flatten nor equality looks: a warm object and a cold one are
the same treedef, the same equality and the same jit cache key.
'''

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import GetAttrKey, tree_flatten, tree_structure

from cottax.core.frozen import OrderedMap
from cottax.pytree.path import NodePath, VarPath
from cottax.core import Function
from cottax.core.realisation import ImplementedFunction
import purposes
from graphs import graph_of
from cottax.core.maths.nesting import NestingGraph

HAT, PROBLEM = purposes.Copy(), purposes.Solve()


def V(*keys) -> VarPath:
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def body(*args):
    '''One function, shared: `fn` is a leaf, so a fresh lambda is a different graph.'''


def call(reads, owns):
    return ImplementedFunction(Function(tuple(V(r) for r in reads), tuple(V(o) for o in owns)),
                       body)


def acyclic() -> NestingGraph:
    '''A fresh graph, built twice so the two are equal without being identical.'''
    return graph_of(OrderedMap({
        N('d1') : call(['z'], ['y1']),
        N('d2') : call(['z', 'y1'], ['y2']),
        N('obj'): call(['y1', 'y2'], ['f']),
    }))


def ask_everything(g: NestingGraph) -> None:
    '''Answer enough queries that every layer of cache is warm.'''
    g.graph.topological_order, g.graph.variables, g.graph.boundary_outputs, g.graph.boundary_inputs
    g.graph.components, g.graph.owners, g.graph.readers
    g[N('d1')]                                  # OrderedMap._index, one level down


# ---------------------------------------------------------------- the invariant

def test_a_query_caches_on_the_instance_where_the_flatten_never_looks():
    g = acyclic()
    ask_everything(g)
    assert 'owners' in g.graph.__dict__                # ordinary instance state, on the dependency graph...
    leaves, treedef = tree_flatten(g)
    assert treedef == tree_flatten(acyclic())[1]       # ...that the pytree cannot see
    assert leaves  == tree_flatten(acyclic())[0]


def test_an_asked_graph_is_still_one_jit_cache_key():
    traces = []

    @eqx.filter_jit
    def run(graph, x):
        traces.append(graph)
        return x * 2

    asked, untouched = acyclic(), acyclic()
    ask_everything(asked)
    run(untouched, jnp.ones(3))
    run(asked, jnp.ones(3))
    assert len(traces) == 1


# ---------------------------------------------------------------- the mechanism

def test_a_rebuilt_instance_starts_empty():
    '''A new object gets its own cache, as `functools` does. Rebuilt directly rather than
    with `eqx.tree_at`: a `NestingGraph` registers no flatten, so nothing reaches past its
    constructor to swap a definition -- `Redefine` is how that is said.'''
    g = acyclic()
    ask_everything(g)
    rebuilt = graph_of(acyclic().definitions, g.within)
    # the binding rules read `owners` and `readers` at construction, so those two are
    # the dependency graph's constructor's own work; nothing else of `g`'s memo came across.
    assert set(rebuilt.__dict__) == {'graph', 'within'}
    assert set(rebuilt.graph.__dict__) == {'definitions', 'owners', 'readers'}


# ---------------------------------------------------------------- priming

'''
A maker may write an answer down before anything asks, because it already knew it --
a `__dict__` update in `NestingGraph`'s derived constructors and in `Plan.__add__`. Nothing at
write time checks that the answer is right, so the check is here: walk every cached
answer a corpus produced and recompute the slow way.
'''


def _cached_answers(value):
    '''(name, query function, cached answer) for every answered query on `value`.'''
    # A snapshot: recomputing one answer may warm others mid-walk.
    for name, cached in list(getattr(value, '__dict__', {}).items()):
        query = getattr(type(value), name, None)
        if isinstance(query, functools.cached_property):
            yield name, query.func, cached


def every_cached_answer_is_the_real_one(*values) -> int:
    '''
    Recompute every cached answer in `values` from scratch and compare. Returns how many
    were checked, so a test cannot pass by finding nothing.
    '''
    checked = 0
    for value in values:
        for name, func, cached in _cached_answers(value):
            if type(cached).__eq__ is object.__eq__:
                continue        # `_nx_dependencies`: networkx compares by identity, so a
                                # recomputed one never matches. Never pre-filled either.
            assert func(value) == cached, \
                f'{type(value).__name__}.{name} was primed with an answer of its own'
            checked += 1
    return checked


def rewritten_plan():
    '''A plan that primes at every op: each `FixedPointCut` is a `Cut` and an `Insert`.'''
    from cottax.core.maths.plan import Cut, Plan
    from cottax.relational.rewrites import FixedPointCut

    n = 5
    g = graph_of(OrderedMap([
        (N(f'd{i}'), call([f'y{(i - 1) % n}'], [f'y{i}'])) for i in range(n)
    ]))
    plan = Plan(g)
    for i in range(n):
        plan = plan + FixedPointCut((Cut(V(f'y{i}'), (N(f'd{(i + 1) % n}'),), HAT),),
                                    PROBLEM(NodePath(V(f'y{i}').segments)))
    return plan


def test_every_primed_answer_is_what_the_query_would_have_computed():
    plan   = rewritten_plan()
    plans  = []
    walk   = plan
    while walk is not None:
        plans.append(walk)
        walk = walk.trunk
    graphs = [p.graph for p in plans]
    for g in graphs:
        # not `topological_order`: the ring is cyclic until the last cut lands
        g.nodes, g.graph.variables, g.graph.boundary_outputs, g.graph.boundary_inputs, g.graph.is_acyclic
        g.graph.components, g.graph.owners, g.graph.readers, g.graph.owned_variables, g.graph.read_variables
    assert every_cached_answer_is_the_real_one(*plans, *graphs, *(g.graph for g in graphs)) > len(graphs)


def test_the_guard_catches_a_wrong_answer():
    g = acyclic()
    g.graph.__dict__['owners'] = {}
    with pytest.raises(AssertionError, match='AbstractGraph.owners'):
        every_cached_answer_is_the_real_one(g.graph)


def test_a_prefilled_answer_leaves_the_treedef_and_equality_alone():
    '''What the derived constructors do -- filling a memo slot early -- must be
    invisible to everything but the query it answers.'''
    g      = acyclic()
    before = tree_structure(g)
    g.graph.__dict__['owners'] = dict(type(g.graph).owners.func(g.graph))
    assert tree_structure(g) == before
    assert g == acyclic() and hash(g) == hash(acyclic())


# ---------------------------------------------------------------- what equinox charges

'''
Two costs equinox levies per instance, both opted out of, both silent if reinstated:
a generated `__init__` flattens its arguments, and a non-magic attribute holding a bound
method is wrapped in a `BoundMethod` -- an `eqx.Module` over `__self__`, so building one
flattens the whole receiver. Either one makes a graph operation walk every place of every
node for nothing.
'''


def flattens_during(action) -> int:
    '''How many times `jax.tree_util.tree_leaves` is called. Equinox reaches it as
    `jtu.tree_leaves`, so patching the module attribute is what it sees.'''
    calls   = []
    real    = jax.tree_util.tree_leaves
    jax.tree_util.tree_leaves = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        action()
    finally:
        jax.tree_util.tree_leaves = real
    return len(calls)


def test_the_cost_of_naming_a_method_does_not_grow_with_the_graph():
    '''The failure this guards is a slowdown, so the assertion is on the mechanism:
    a wide graph must flatten no more than a narrow one, which is zero for both.'''
    wide = graph_of(OrderedMap({
        N(f'd{i}'): call([f'y{i - 1}'], [f'y{i}']) for i in range(200)
    }))
    assert flattens_during(lambda: wide.subgraph) == 0
    assert flattens_during(lambda: wide.definitions.items) == 0
    assert flattens_during(lambda: graph_of(wide.definitions)) == 0


# ---------------------------------------------------------------- the hash and the pickle

def test_a_frozen_value_hashes_once_and_the_same_warm_or_cold():
    '''`Frozen` computes the field hash on the first call and keeps it in `__dict__`,
    where equality and the flatten never look; `OrderedMap` keeps its own the same way.'''
    cold, warm = acyclic(), acyclic()
    ask_everything(warm)
    assert hash(cold) == hash(warm)
    assert '_frozen_hash' in cold.__dict__ and cold.__dict__['_frozen_hash'] == hash(cold)
    assert cold == warm
    calls = []
    original = type(cold).__field_hash__
    type(cold).__field_hash__ = lambda self: calls.append(1) or original(self)
    try:
        fresh = acyclic()
        hash(fresh); hash(fresh); hash(fresh)
        assert calls == [1]
    finally:
        type(cold).__field_hash__ = original
    pm = cold.definitions
    assert hash(pm) == hash(OrderedMap(dict(pm.items()))) and pm._hash == hash(pm)


def test_a_written_hash_is_left_alone():
    '''`OrderedMap` writes its own; a `Frozen` level that writes one keeps it down the chain.'''
    from cottax.core.frozen import Frozen
    class Level(Frozen):
        x : int
        def __hash__(self): return 7
    class Final(Level):
        pass
    assert hash(Final(3)) == 7 and '_frozen_hash' not in Final(3).__dict__


def test_a_pickle_carries_the_fields_and_no_memo():
    '''A warm graph pickles as the value it is: the fields, no cached query, no hash,
    and it comes back equal and cold.'''
    import pickle
    warm = acyclic()
    ask_everything(warm); hash(warm)
    assert len(warm.__dict__) > 2
    back = pickle.loads(pickle.dumps(warm))
    assert set(back.__dict__) == {'graph', 'within'}                    # the two fields, cold
    assert set(back.graph.__dict__) == {'definitions'}
    assert set(vars(back.definitions)) == {'_keys', '_values'}          # no `_index`, no hash
    assert back == warm and hash(back) == hash(warm)                    # asked now, memoised now
    assert set(back.__dict__) == {'graph', 'within', '_frozen_hash'}


def test_a_plan_pickles_without_its_prefix_graphs():
    '''`Plan` is `Frozen`: the primed `trunk` and `graph` are memos and stay behind, and
    `graph` is rebuilt equal on the other side.'''
    import pickle
    from cottax.core.maths.plan import Delete, Plan
    plan = Plan(acyclic()) + Delete((N('obj'),))
    assert 'graph' in plan.__dict__ and 'trunk' in plan.__dict__
    back = pickle.loads(pickle.dumps(plan))
    assert set(back.__dict__) == {'base', 'ops'}
    assert back == plan and hash(back) == hash(plan)
    assert back.graph == plan.graph and N('obj') not in back.graph
