'''
Incremental construction: what must stay true when `Insert`, `Delete`, `Rewire` and
`NestingGraph.subgraph` stop rebuilding a whole `NestingGraph` and start validating only their own
delta against an already-valid parent.

Every operation is pinned three ways -- the result equals the graph built the long way
through the public constructor (warm or cold), every derived query agrees *in order*, and
a definition the op did not touch is still the same object. The refusals are pinned by
message, since a delta-only check is exactly where one goes missing.

Names are built by hand from key entries, as in `test_algebra.py` and `test_rewrites.py`.
'''

from typing import Any, NamedTuple

import pytest
from jax.tree_util import GetAttrKey, SequenceKey

from cottax.core import AbstractGraph, NestingGraph, Rename
from cottax.core.maths.plan import Combine, ComposedOp, GraphOp, Nest, Primitive, Rebind, StagedOp, Unnest
from cottax.core.maths.plan import Delete, Insert, Redefine
from cottax.core.maths.plan.primitive import _fully_checked, unbreakable
from cottax.core.maths.graph import check_one_producer_per_place
from cottax.relational import Eq, Relation, RelationalCondition
from cottax.core.maths.plan import Cut
from cottax.pytree.path import NodePath, VarPath
from cottax.core import Function
from cottax.core.realisation import ImplementedFunction
import purposes
from cottax.core.frozen import OrderedMap
from cottax.core import problems
from graphs import graph_of


# ---------------------------------------------------------------- names and nodes

def V(*keys) -> VarPath:
    '''A variable name, one `GetAttrKey` per key: `V('s', 'u')` is `.s.u`.'''
    return VarPath(tuple(GetAttrKey(k) for k in keys))


def N(*keys) -> NodePath:
    '''A node name, the same way -- node names nest for the same reason.'''
    return NodePath(tuple(GetAttrKey(k) for k in keys))


def VK(*entries) -> VarPath:
    '''A variable name out of key entries of any kind, for the nested cases.'''
    return VarPath(tuple(entries))


def body(*args):
    '''One function, shared: `fn` is a leaf, so a fresh lambda would be a different node
    and the long way round would never compare equal.'''


def call(reads, owns) -> ImplementedFunction:
    return ImplementedFunction(Function(tuple(reads), tuple(owns)), body)


def source(*owns) -> ImplementedFunction:
    return call((), owns)


A, B, C, F, G, K, M, W, X, Y, Z = (V(s) for s in 'a b c f g k m w x y z'.split())
Y1, Y2, U, R_U, NEW = V('y1'), V('y2'), V('u'), V('r_u'), V('new')

PSI  = VK(GetAttrKey('eq'), SequenceKey(0), GetAttrKey('psi'))
Q    = VK(GetAttrKey('eq'), SequenceKey(0), GetAttrKey('q'))
FLUX = VK(GetAttrKey('eq'), SequenceKey(1), GetAttrKey('psi'))
MESH = V('mesh', 'nodes')
OUT2 = VK(GetAttrKey('out'), SequenceKey(2))
FEM  = N('fem', 'solve')

HAT, COND = purposes.Copy(), purposes.Cond()
HAT_Y1, HAT_Y2 = HAT(Y1), HAT(Y2)
HAT_X,  HAT_U,  HAT_B = HAT(X), HAT(U), HAT(B)
COND_Y1   = COND(Y1)
COND_NODE = COND(N('y1'))


# ---------------------------------------------------------------- the fixture graphs

def chain() -> NestingGraph:
    '''`.a -> .b -> .c`, bound out of order, so binding order is not dependency order.'''
    return graph_of(OrderedMap({
        N('c'): call([B], [Y]),
        N('b'): call([A], [B]),
        N('a'): source(A),
    }))


def deep() -> NestingGraph:
    '''Nested paths, three keys deep, and one nested node name.'''
    return graph_of(OrderedMap({
        N('mesh'): source(MESH),
        FEM      : call([MESH],   [PSI, Q]),
        N('post'): call([PSI, Q], [FLUX]),
    }))


def wide() -> NestingGraph:
    '''A node with several inputs and several outputs.'''
    return graph_of(OrderedMap({
        N('src')  : source(Z, M),
        N('multi'): call([Z, M, W], [Y1, Y2, F]),
        N('sink') : call([Y1, F],   [C]),
    }))


def repeated() -> NestingGraph:
    '''`fn(x, x)` is legitimate; `readers` de-duplicates it and `read` does too.'''
    return graph_of(OrderedMap({
        N('src')  : source(X),
        N('twice'): call([X, X],  [Y1]),
        N('use')  : call([Y1, X], [F]),
    }))


def implicit() -> NestingGraph:
    '''A problem inside a cycle: `.prob` owns `.u`, `.body` produces the residual.'''
    return graph_of(OrderedMap({
        N('z')   : source(Z),
        N('body'): call([U, Z], [R_U]),
        N('prob'): RelationalCondition((Relation(R_U, None, Eq),), (), (U,), ()),
        N('obj') : call([U], [F]),
    }))


def minted() -> NestingGraph:
    '''A cut already opened: `^copy.y1` guessed from outside, `^cond.y1` the comparison.'''
    return graph_of(OrderedMap({
        N('d1')  : call([Y2],          [Y1]),
        N('d2')  : call([HAT_Y1],      [Y2]),
        COND_NODE: call([Y1, HAT_Y1],  [COND_Y1]),
        N('guess'): source(HAT_Y1),
    }))


# ---------------------------------------------------------------- the cases

class Case(NamedTuple):
    id       : str
    graph    : NestingGraph        # the parent, already valid
    produced : NestingGraph        # what the operation made of it
    expected : NestingGraph        # the same thing, built the long way


def op_case(id: str, graph: NestingGraph, op: Any, expected: NestingGraph) -> Case:
    return Case(id, graph, op(graph), expected)


def sub_case(id: str, graph: NestingGraph, keep, expected: NestingGraph) -> Case:
    return Case(id, graph, graph.subgraph(keep), expected)


CASES : list[Case] = [

    # -------------------------------------------------- chain
    op_case('chain-insert', chain(),
            Insert(OrderedMap({N('d'): call([Y], [W])})),
            graph_of(OrderedMap({
                N('c'): call([B], [Y]),
                N('b'): call([A], [B]),
                N('a'): source(A),
                N('d'): call([Y], [W]),          # appended: `Insert` binds at the end
            }))),

    op_case('chain-delete', chain(),
            Delete((N('b'),)),
            graph_of(OrderedMap({
                N('c'): call([B], [Y]),
                N('a'): source(A),
            }))),

    op_case('chain-delete-all', chain(),
            Delete((N('c'), N('b'), N('a'),)),
            graph_of(OrderedMap())),

    # `keep` is written in the opposite order to the binding, and the result is still
    # in binding order -- `subgraph` filters the binding, it does not re-bind.
    sub_case('chain-subgraph', chain(), [N('a'), N('c')],
             graph_of(OrderedMap({
                 N('c'): call([B], [Y]),
                 N('a'): source(A),
             }))),

    op_case('chain-cut', chain(),
            Cut(B, (N('c'),), HAT),
            graph_of(OrderedMap({
                N('c'): call([HAT_B], [Y]),
                N('b'): call([A], [B]),
                N('a'): source(A),
            }))),

    op_case('chain-rename-read', chain(),
            Rename(N('c'), ((B, A),)),
            graph_of(OrderedMap({
                N('c'): call([A], [Y]),
                N('b'): call([A], [B]),
                N('a'): source(A),
            }))),

    op_case('chain-rename-write', chain(),
            Rename(N('b'), ((B, W),)),
            graph_of(OrderedMap({
                N('c'): call([B], [Y]),          # now reads a boundary input
                N('b'): call([A], [W]),
                N('a'): source(A),
            }))),

    # -------------------------------------------------- deep
    op_case('deep-insert', deep(),
            Insert(OrderedMap({N('probe'): call([FLUX], [OUT2])})),
            graph_of(OrderedMap({
                N('mesh') : source(MESH),
                FEM       : call([MESH],   [PSI, Q]),
                N('post') : call([PSI, Q], [FLUX]),
                N('probe'): call([FLUX],   [OUT2]),
            }))),

    op_case('deep-delete', deep(),
            Delete((N('mesh'),)),
            graph_of(OrderedMap({
                FEM      : call([MESH],   [PSI, Q]),
                N('post'): call([PSI, Q], [FLUX]),
            }))),

    sub_case('deep-subgraph', deep(), [N('post'), FEM],
             graph_of(OrderedMap({
                 FEM      : call([MESH],   [PSI, Q]),
                 N('post'): call([PSI, Q], [FLUX]),
             }))),

    op_case('deep-cut', deep(),
            Cut(PSI, (N('post'),), purposes.Copy()),
            graph_of(OrderedMap({
                N('mesh'): source(MESH),
                FEM      : call([MESH], [PSI, Q]),
                N('post'): call([HAT(PSI), Q], [FLUX]),
            }))),

    # -------------------------------------------------- wide
    op_case('wide-insert', wide(),
            Insert(OrderedMap({N('extra'): call([Y2], [G])})),
            graph_of(OrderedMap({
                N('src')  : source(Z, M),
                N('multi'): call([Z, M, W], [Y1, Y2, F]),
                N('sink') : call([Y1, F],   [C]),
                N('extra'): call([Y2],      [G]),
            }))),

    op_case('wide-insert-two', wide(),
            Insert(OrderedMap({
                N('extra'): call([Y2], [G]),
                N('more') : call([G],  [K]),     # the delta reads what the delta owns
            })),
            graph_of(OrderedMap({
                N('src')  : source(Z, M),
                N('multi'): call([Z, M, W], [Y1, Y2, F]),
                N('sink') : call([Y1, F],   [C]),
                N('extra'): call([Y2],      [G]),
                N('more') : call([G],       [K]),
            }))),

    op_case('wide-delete', wide(),
            Delete((N('src'),)),
            graph_of(OrderedMap({
                N('multi'): call([Z, M, W], [Y1, Y2, F]),
                N('sink') : call([Y1, F],   [C]),
            }))),

    sub_case('wide-subgraph', wide(), [N('multi')],
             graph_of(OrderedMap({N('multi'): call([Z, M, W], [Y1, Y2, F])}))),

    sub_case('wide-subgraph-empty', wide(), [], graph_of(OrderedMap())),

    op_case('wide-cut', wide(),
            Cut(Y1, (N('sink'),), HAT),
            graph_of(OrderedMap({
                N('src')  : source(Z, M),
                N('multi'): call([Z, M, W], [Y1, Y2, F]),
                N('sink') : call([HAT_Y1, F], [C]),
            }))),

    # -------------------------------------------------- repeated reads
    op_case('repeated-insert', repeated(),
            Insert(OrderedMap({N('again'): call([X, X], [K])})),
            graph_of(OrderedMap({
                N('src')  : source(X),
                N('twice'): call([X, X],  [Y1]),
                N('use')  : call([Y1, X], [F]),
                N('again'): call([X, X],  [K]),
            }))),

    # A relabel is keyed by variable, so `fn(x, x)` has both reads renamed.
    op_case('repeated-cut', repeated(),
            Cut(X, (N('twice'),), HAT),
            graph_of(OrderedMap({
                N('src')  : source(X),
                N('twice'): call([HAT_X, HAT_X], [Y1]),
                N('use')  : call([Y1, X],        [F]),
            }))),

    op_case('repeated-delete', repeated(),
            Delete((N('src'),)),
            graph_of(OrderedMap({
                N('twice'): call([X, X],  [Y1]),
                N('use')  : call([Y1, X], [F]),
            }))),

    sub_case('repeated-subgraph', repeated(), [N('twice'), N('use')],
             graph_of(OrderedMap({
                 N('twice'): call([X, X],  [Y1]),
                 N('use')  : call([Y1, X], [F]),
             }))),

    # -------------------------------------------------- a problem in a cycle
    op_case('implicit-cut', implicit(),
            Cut(U, (N('body'),), HAT),
            graph_of(OrderedMap({
                N('z')   : source(Z),
                N('body'): call([HAT_U, Z], [R_U]),
                N('prob'): RelationalCondition((Relation(R_U, None, Eq),), (), (U,), ()),
                N('obj') : call([U], [F]),
            }))),

    op_case('implicit-delete', implicit(),
            Delete((N('prob'),)),
            graph_of(OrderedMap({
                N('z')   : source(Z),
                N('body'): call([U, Z], [R_U]),
                N('obj') : call([U], [F]),
            }))),

    op_case('implicit-insert', implicit(),
            Insert(OrderedMap({N('con'): call([U], [C])})),
            graph_of(OrderedMap({
                N('z')   : source(Z),
                N('body'): call([U, Z], [R_U]),
                N('prob'): RelationalCondition((Relation(R_U, None, Eq),), (), (U,), ()),
                N('obj') : call([U], [F]),
                N('con') : call([U], [C]),
            }))),

    sub_case('implicit-subgraph', implicit(), [N('prob'), N('body')],
             graph_of(OrderedMap({
                 N('body'): call([U, Z], [R_U]),
                 N('prob'): RelationalCondition((Relation(R_U, None, Eq),), (), (U,), ()),
             }))),

    # -------------------------------------------------- minted namespaces
    op_case('minted-insert', minted(),
            Insert(OrderedMap({N('probe'): call([COND_Y1], [NEW])})),
            graph_of(OrderedMap({
                N('d1')   : call([Y2],         [Y1]),
                N('d2')   : call([HAT_Y1],     [Y2]),
                COND_NODE : call([Y1, HAT_Y1], [COND_Y1]),
                N('guess'): source(HAT_Y1),
                N('probe'): call([COND_Y1],    [NEW]),
            }))),

    op_case('minted-delete', minted(),
            Delete((N('guess'),)),
            graph_of(OrderedMap({
                N('d1')  : call([Y2],         [Y1]),
                N('d2')  : call([HAT_Y1],     [Y2]),
                COND_NODE: call([Y1, HAT_Y1], [COND_Y1]),
            }))),

    op_case('minted-cut', minted(),
            Cut(Y2, (N('d1'),), HAT),
            graph_of(OrderedMap({
                N('d1')   : call([HAT_Y2],     [Y1]),
                N('d2')   : call([HAT_Y1],     [Y2]),
                COND_NODE : call([Y1, HAT_Y1], [COND_Y1]),
                N('guess'): source(HAT_Y1),
            }))),

    sub_case('minted-subgraph', minted(), [COND_NODE, N('d1')],
             graph_of(OrderedMap({
                 N('d1')  : call([Y2],         [Y1]),
                 COND_NODE: call([Y1, HAT_Y1], [COND_Y1]),
             }))),
]

IDS = [c.id for c in CASES]

QUERIES = ('nodes', 'owners', 'readers', 'owned_variables', 'read_variables',
           'variables', 'boundary_inputs', 'boundary_outputs')

TUPLE_QUERIES = ('nodes', 'owned_variables', 'read_variables', 'variables', 'boundary_inputs', 'boundary_outputs')


def ask_everything(g: NestingGraph) -> None:
    '''Answer every query, so every layer of cache is warm.'''
    for name in QUERIES:
        getattr(g.graph, name)
    g.graph.components, g.graph.is_acyclic, problems(g.definitions)


# ---------------------------------------------------------------- the result itself

@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_the_result_equals_the_graph_built_the_long_way(case):
    '''An incrementally built graph must be indistinguishable from the constructed one,
    down to binding order -- `NestingGraph` equality is order-sensitive, and it must be --
    and stays so once every query has warmed its caches.'''
    assert case.produced == case.expected
    assert hash(case.produced) == hash(case.expected)
    ask_everything(case.produced)
    assert case.produced == case.expected


@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_every_derived_query_agrees_in_binding_order(case):
    '''Order is load-bearing everywhere in this package, so the tuple queries are
    compared as sequences and never as sets.'''
    for name in QUERIES:
        got, exp = getattr(case.produced.graph, name), getattr(case.expected.graph, name)
        assert got == exp, name
        if name in TUPLE_QUERIES:
            assert isinstance(got, tuple) and list(got) == list(exp), name


@pytest.mark.parametrize('case', CASES, ids=IDS)
def test_a_definition_the_op_did_not_touch_is_the_same_object(case):
    '''`subgraph` and a relabel that renames nothing both preserve node identity, which
    is what keeps a definition one jit cache key across a rewrite.'''
    for name in case.produced.nodes:
        if name in case.graph and case.produced[name] == case.graph[name]:
            assert case.produced[name] is case.graph[name], name


# ---------------------------------------------------------------- what each op refuses

def test_a_rewire_refuses_a_node_that_is_not_there():
    with pytest.raises(KeyError, match='not a node of this graph'):
        Cut(B, (N('nope'),), HAT)(chain())


def test_a_rename_refuses_a_place_the_node_does_not_carry():
    with pytest.raises(KeyError, match='neither reads nor owns'):
        Rename(N('a'), ((B, W),))(chain())


# ------------------------------------------- the four refusals, reached through an op

def test_a_rewire_that_would_give_a_variable_two_producers_is_refused():
    '''Duplicate owner. `Rewire` has no check of its own for this, so the refusal is the
    constructor's and has to survive a delta-only validation.'''
    with pytest.raises(ValueError, match='owned by both'):
        Rename(N('b'), ((B, Y),))(chain())


def test_an_insert_owning_something_enclosing_an_existing_output_is_refused():
    '''Encapsulation: `.a` owns `.s.u`, so a node owning `.s` writes the same place.
    `Insert.check` intersects by equality and does not see this.'''
    base = graph_of(OrderedMap({
        N('a'): source(V('s', 'u')),
        N('b'): call([V('s', 'u')], [Y]),
    }))
    with pytest.raises(ValueError, match='lies inside'):
        Insert(OrderedMap({N('c'): source(V('s'))}))(base)


def test_an_insert_reading_inside_an_existing_output_is_refused():
    '''Reads match by equality, so `.s.u` would silently become a boundary input.'''
    base = graph_of(OrderedMap({
        N('a'): source(V('s')),
        N('b'): call([V('s')], [Y]),
    }))
    with pytest.raises(ValueError, match='would silently be a boundary input'):
        Insert(OrderedMap({N('c'): call([V('s', 'u')], [W])}))(base)


def test_a_rewire_reading_inside_an_existing_output_is_refused():
    '''The same refusal, reached by renaming a read rather than by inserting one.'''
    base = graph_of(OrderedMap({
        N('a'): source(V('s')),
        N('b'): call([V('s')], [Y]),
    }))
    with pytest.raises(ValueError, match='would silently be a boundary input'):
        Rename(N('b'), ((V('s'), V('s', 'u')),))(base)


def test_an_insert_owning_inside_an_existing_read_is_refused():
    '''The mirror: `.b` reads `.s` wholesale, so it would silently miss `.s.u`.'''
    base = graph_of(OrderedMap({N('b'): call([V('s')], [Y])}))
    with pytest.raises(ValueError, match='would silently miss this output'):
        Insert(OrderedMap({N('c'): source(V('s', 'u'))}))(base)


def test_the_proof_and_the_rule_report_one_kind():
    '''A refusal is matched by kind, so a conflict reported by the rebinding's proof is
    the kind the rule of the result reports: both ways round.'''
    from cottax.core.frozen import violations
    from cottax.core.maths.plan.primitive import Rebinding, check_rebound_reads_exact
    for standing, new in (({N('b'): call([V('s')], [Y])}, {N('c'): source(V('s', 'u'))}),
                          ({N('a'): source(V('s'))}, {N('c'): call([V('s', 'u')], [W])})):
        before = graph_of(OrderedMap(standing)).graph
        proved = check_rebound_reads_exact(before, None, Rebinding(frozenset(), new))
        whole  = violations(AbstractGraph, OrderedMap(standing | new))
        assert [type(v) for v in proved] == [type(v) for v in whole] != []


def test_an_insert_reading_around_an_existing_output_is_refused():
    '''The same conflict with the roles swapped in time: `.s.u` is already owned, and
    the *new* node reads `.s` wholesale. The downward scan (`_inside`) is the only
    check that can see it, since nothing owns a prefix of `.s`.'''
    base = graph_of(OrderedMap({
        N('a'): source(V('s', 'u')),
        N('b'): call([V('s', 'u')], [Y]),
    }))
    with pytest.raises(ValueError, match='would silently miss this output'):
        Insert(OrderedMap({N('c'): call([V('s')], [W])}))(base)


def test_a_rewire_reading_around_an_existing_output_is_refused():
    '''The same refusal, reached by renaming a read to enclose an existing output.'''
    base = graph_of(OrderedMap({
        N('a'): source(V('s', 'u')),
        N('b'): call([V('s', 'u')], [Y]),
    }))
    with pytest.raises(ValueError, match='would silently miss this output'):
        Rename(N('b'), ((V('s', 'u'), V('s')),))(base)


def test_a_rewire_owning_inside_an_existing_read_is_refused():
    base = graph_of(OrderedMap({
        N('a'): source(V('s')),
        N('b'): call([V('s')], [Y]),
    }))
    with pytest.raises(ValueError, match='would silently miss this output'):
        Rename(N('a'), ((V('s'), V('s', 'u')),))(base)


# ---------------------------------------------------------------- the induction

'''
An edit does not rescan the whole graph: each op's `covers` names the delta check that
stands in for a rule of the result, against a parent that was already valid. These pin
both halves: a covered rule is not asked again (otherwise the proof bought nothing), and a
rule a final adds, which no op can name, is asked of every result.
'''

def _rescans_during(action) -> int:
    '''How often the covered rules run in full during `action`, outside `fully_checked`:
    each is swapped for a counting stand-in, in `AbstractGraph.checks` and in the ops'
    `covers` alike, so the stand-in is what `rebound` looks up.'''
    real    = AbstractGraph.checks
    calls   = []
    counted = {rule: (lambda rule: lambda g: calls.append(rule) or rule(g))(rule) for rule in real}
    covers  = {op: op.__dict__['covers'] for op in (Rebind,)}
    AbstractGraph.checks = tuple(counted.values())
    for op, cover in covers.items():
        op.covers = {counted.get(rule, rule): proof for rule, proof in cover.items()}
    token = _fully_checked.set(False)
    try:
        action()
    finally:
        _fully_checked.reset(token)
        AbstractGraph.checks = real
        for op, cover in covers.items():
            op.covers = cover
    return len(calls)


def _probe_asked_during(action) -> int:
    '''How often a rule a final adds, on `AbstractGraph.checks`, runs during
    `action`.'''
    real  = AbstractGraph.checks
    calls = []
    AbstractGraph.checks = (lambda g: calls.append(1), *real)
    try:
        action()
    finally:
        AbstractGraph.checks = real
    return len(calls)


EDITS = [
    pytest.param(lambda g: Insert(OrderedMap({N('d'): call([Y], [W])})).apply(g),
                 id='insert'),
    pytest.param(lambda g: Delete((N('c'),)).apply(g),        id='delete'),
    pytest.param(lambda g: Cut(B, (N('c'),), HAT).apply(g),       id='rewire'),
]


@pytest.mark.parametrize('edit', [
    *EDITS,
    pytest.param(lambda g: g.subgraph(g.nodes[:-1]),         id='subgraph'),
])
def test_an_edit_does_not_rescan_the_whole_graph(edit):
    graph = chain()                       # built before, so its own scan is not counted
    assert _rescans_during(lambda: edit(graph)) == 0


@pytest.mark.parametrize('edit', EDITS)
def test_a_rule_a_final_adds_is_asked_of_every_result(edit):
    graph = chain()
    assert _probe_asked_during(lambda: edit(graph)) == 1


PRIMITIVES = (Rebind, Nest, Unnest)
'''The ops that build a graph, each through `rebound`: the binding primitive and the two
nesting ones. `Insert`, `Delete`, `Redefine` and `Combine` derive from `Rebind`.'''


@pytest.mark.parametrize('op', PRIMITIVES, ids=lambda op: op.__name__)
def test_a_primitive_says_how_every_rule_of_core_holds(op):
    """`covers` is total over core's rules -- a proof, `unbreakable` or `in_full` for
    each -- so a rule added to core fails here until every primitive has decided."""
    rules = {*AbstractGraph.per_level('checks'), *NestingGraph.per_level('checks')}
    assert set(op.covers) == rules


def test_every_op_is_a_primitive_or_composes_them():
    """An op builds a graph (a `Primitive`: the three above) or composes ops
    (`ComposedOp`, `StagedOp`); nothing else subclasses `GraphOp`. An op deriving from a
    primitive inherits its proof and never re-declares it."""
    import importlib, pkgutil, cottax
    for info in pkgutil.walk_packages(cottax.__path__, 'cottax.'):
        importlib.import_module(info.name)
    ours = lambda cls: {c for c in cls.__subclasses__() if c.__module__.startswith('cottax.')}
    assert ours(GraphOp) == {Primitive, ComposedOp, StagedOp}
    assert ours(Primitive) == set(PRIMITIVES)
    assert ours(Rebind) == {Insert, Delete, Redefine, Combine}

    def below(cls):
        return {d for c in ours(cls) for d in (c, *below(c))}
    assert [c.__name__ for p in PRIMITIVES for c in below(p) if 'covers' in c.__dict__] == []


class _CarelessInsert(Primitive):
    '''An insert claiming, wrongly, that it cannot break ownership -- a primitive of its
    own, since nothing below `Rebind` may rebind `covers`.'''
    nodes  : OrderedMap
    covers = {**Rebind.covers, check_one_producer_per_place: unbreakable}

    def edited(self, graph):
        return Insert(self.nodes).edited(graph)


def test_a_primitives_proof_is_bound_once():
    with pytest.raises(TypeError, match='rebinds covers'):
        class _Rebinding(Insert):
            covers = Rebind.covers


def test_fully_checked_catches_a_proof_that_passes_what_its_rule_refuses():
    rival = lambda: _CarelessInsert(OrderedMap({N('rival'): call([Z], [B])})).apply(chain())
    with pytest.raises(AssertionError, match='unbreakable passed what check_one_producer_per_place refuses'):
        rival()                                         # the suite runs fully checked
    token = _fully_checked.set(False)
    try:
        assert N('rival') in rival()                    # unchecked, the wrong proof is believed
    finally:
        _fully_checked.reset(token)

