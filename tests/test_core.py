'''
`core/` alone: the vocabulary, and the two laws that tie it to `Path`.

The rules are stated over protocols -- a place answers `prefixes` and `is_prefix`, a
definition answers `owns` and `reads` -- so this module builds *toy* places out of strings
to show the rules never learn what a real name is made of, and then holds the real `Path`
to the laws the vocabulary states.
'''

import itertools
import random
from dataclasses import dataclass

import pytest
from jax.tree_util import GetAttrKey, SequenceKey

from cottax.core.maths.places import comparable, is_prefix, overlaps
from cottax.core.maths.graph import (
    OwnAroundOwner, OwnInsideOwner, OwnedTwice, OwnerInsideRead, ReadInsideOwner,
)
from cottax.core import (
    AbstractGraph, NestingGraph, Combine, ConditionalNode, Cut, Determine, FunctionalNode,
    Rename, Undetermine, refusals,
    Violation, NodeDefinition, Insert, Redefine, Delete,
    violations, comparables_in, first, prefixes,
    answer_for_labels, answer_for_places, GraphWithValues,
)
from cottax.core.maths.plan.primitive import (
    Rebinding, check_delete_keeps_nesting_whole, check_nesting, check_unnesting, in_full,
)
from cottax.core.refusal_kinds import DoesNotRead
from cottax.core.maths.executable import ExecutableGraph
from cottax.core.frozen import Answer, OrderedMap, answers
from cottax.core.maths.nesting import (BoxOwnsNothing, NotAProblem, Unbound, check_blocks_own_something,
                                      check_blocks_partition)
from cottax.pytree.path import VarPath
from cottax.core import Mint, Minting
import purposes


# ---------------------------------------------------------------- a toy place

@dataclass(frozen=True)
class Word:
    '''
    A place as a word of plain strings. **The whole of `Place`**: one member, and a
    constructor taking a sequence of segments. Nothing here is a path, a pytree key or a
    tuple subclass -- which is the point, since the rules run on it unchanged.
    '''

    segments : tuple[str, ...]

    def __init__(self, segments):
        object.__setattr__(self, 'segments',
                           tuple(segments.split('.')) if isinstance(segments, str)
                           else tuple(segments))

    def __str__(self) -> str:
        '''A name is printed, and `str` is what every object answers -- so nothing
        above asks a place for a `spelling`. A minted segment prints itself.'''
        return '.'.join(str(k) for k in self.segments)


@answers(Word)
class WordMinting(Minting):
    '''A word wears a purpose as its first segment: no string is a `Mint`.'''

    def minted(self, m, word):  return Word((m, *word.segments))
    def mint_of(self, word):    return next((k for k in word.segments[:1] if isinstance(k, Mint)), None)
    def unminted(self, word):   return Word(word.segments[1:]) if self.mint_of(word) else word


class Toy(NodeDefinition):
    '''A definition as two tuples of `Word`s -- the whole of `NodeDefinition`.'''
    reads : tuple = ()
    owns  : tuple = ()
    def relabel(self, renaming): return self


def toy(reads, owns) -> Toy:
    '''`Toy` spelled with strings: `'s.u'` for a word, `()` for the root.'''
    return Toy(tuple(map(Word, reads)), tuple(map(Word, owns)))


def places(names):
    return tuple(Word(n) for n in names)


class Declared(FunctionalNode):
    '''A computation with nothing to call.'''
    reads : tuple = ()
    owns  : tuple = ()
    def relabel(self, renaming): return self


class States(ConditionalNode):
    '''A bare condition node: two bare tuples, and no relation anywhere. **The whole of
    `ConditionalNode`** -- what it holds against and what it determines -- so every rule
    in the layer runs on a condition node that was never written as `lhs op rhs`.'''
    conditions : tuple = ()
    unknowns   : tuple = ()
    reads      = property(lambda self: self.conditions)
    owns       = property(lambda self: self.unknowns)
    def relabel(self, renaming): return self
    def determined(self, unknowns): return States(self.conditions, unknowns)
    def __add__(self, other):
        '''The whole of what `Combine` asks of a kind -- no relation anywhere.'''
        if not isinstance(other, States):
            return NotImplemented
        return States(self.conditions + other.conditions, self.unknowns + other.unknowns)


class Driven(ConditionalNode):
    '''A problem with an algorithm, as the abstract sees it: a condition node
    reading more than its conditions and owning more than its unknowns. No driver
    anywhere -- what one *is* is `problem/`'s, and the rules never ask.'''
    problem : States
    data    : tuple = ()
    reports : tuple = ()
    conditions = property(lambda self: self.problem.conditions)
    unknowns   = property(lambda self: self.problem.unknowns)
    reads      = property(lambda self: self.problem.reads + tuple(v for vs in self.data for v in vs))
    owns       = property(lambda self: self.problem.owns + tuple(v for vs in self.reports for v in vs))
    def relabel(self, renaming): return self
    def determined(self, unknowns): return Driven(self.problem.determined(unknowns),
                                                  self.data, self.reports)


def body(reads, owns):          return declared(reads, owns)   # a body is no edge: the abstract sees a function
def declared(reads, owns):      return Declared(places(reads), places(owns))
def states(reads, owns):        return States(places(reads), places(owns))
def driven(reads, owns, data=(), reports=()):
    return Driven(states(reads, owns),
                  ((places(data),) if data else ()), ((places(reports),) if reports else ()))


def G(definitions, within=None):
    '''A nesting graph over two `OrderedMap`s: nothing on it is abstract.'''
    return NestingGraph(AbstractGraph(OrderedMap(definitions)), OrderedMap(within or {}))


def proof(graph, op, define):
    '''What an op's delta checks and proofs say of `graph` with `define` bound, as data.'''
    r     = Rebinding(frozenset(), define)
    after = graph.graph._carried(definitions=graph.definitions | define)
    return [v for check in op.per_level('checks_delta') for v in check(graph, r)] + \
           [v for proof in dict.fromkeys(op.covers.values()) if proof is not in_full
              for v in proof(graph.graph, after, r)]


def on_insert(new, standing=None):
    '''`new` inserted into a standing toy graph, empty by default.'''
    return [type(v) for v in proof(G(standing or {}), Insert, new)]


# ---------------------------------------------------------------- the rules, on toys

def test_the_rules_run_on_a_place_that_is_not_a_path():
    # Nothing in `core/` knows what a name is made of, so a string does.
    assert on_insert({Word('a'): toy([], ['s']), Word('b'): toy(['s'], ['t'])}) == []


@pytest.mark.parametrize('a_owns, b_owns, b_reads, rule', [
    (['s'],   ['s'],   [],      OwnedTwice),          # two producers
    (['s'],   ['s.u'], [],      OwnInsideOwner),      # the same, one step up the containment order
    (['s'],   ['t'],   ['s.u'], ReadInsideOwner),     # the read would silently vanish
    (['s.u'], ['t'],   ['s'],   OwnerInsideRead),     # the mirror, seen from the output
])
def test_the_four_rows_are_two_rules(a_owns, b_owns, b_reads, rule):
    # Which of the last two fires depends on which side is the delta -- both bindings are
    # new here, so the overlap is seen from the output. They cover one wrong answer.
    assert on_insert({Word('a'): toy([], a_owns), Word('b'): toy(b_reads, b_owns)}) == [rule]


@pytest.mark.parametrize('a, b, rule', [
    (toy([],      [()]),  toy([],    ['s']), OwnInsideOwner),
    (toy([],      ['s']), toy([],    [()]),  OwnAroundOwner),
    (toy([()],    []),    toy([],    ['s']), OwnerInsideRead),
    (toy([],      [()]),  toy(['s'], ['t']), OwnInsideOwner),
])
def test_the_root_place_needs_no_rule_of_its_own(a, b, rule):
    """
    The root is the empty word, so it is a prefix of every place. `prefixes` starts there,
    which means the two rules catch it like any other containment -- there is no separate
    *a place may not name the root* axiom, and there was one only while the prefix walk's loop
    started past the root and could not see it.
    """
    found = on_insert({Word('a'): a, Word('b'): b})
    assert rule in found


def test_a_lone_node_may_own_the_whole_structure():
    # Nothing to conflict with, so nothing is wrong: a node that computes the caller's
    # whole structure is a node like any other. The rules permit exactly what they say.
    assert on_insert({Word('a'): toy([], [()])}) == []


@pytest.mark.parametrize('bound, run', [
    ({'p': ([], ['pp']), 'q': (['pp'], ['qq']), 'r': ([], ['rr'])}, 'pqr'),
    ({'s1': ([], ['a']), 's2': ([], ['b']), 'x': (['b'], ['xx']), 'y': (['a'], ['yy'])},
     ['s1', 's2', 'x', 'y']),
    ({'x': (['q'], ['xx']), 'y': ([], ['yy']), 'z': (['yy'], ['q'])}, 'yzx'),
])
def test_components_ready_at_once_run_in_binding_order(bound, run):
    """Dependency order first, binding order among what is ready: a graph bound in
    dependency order runs as bound."""
    g = AbstractGraph(OrderedMap({Word(n): declared(*io) for n, io in bound.items()}))
    assert [c[0] for c in g.components] == [Word(n) for n in run]


def test_a_rule_yields_data_and_never_a_sentence():
    # A violation is a final per case, its fields the places by role. The wording is
    # `messages`', which is the only module in `core/` that asks a place how it is spelled.
    found = proof(G({}), Insert, {Word('a'): toy([], ['s']), Word('b'): toy([], ['s'])})
    assert found == [OwnedTwice(place=Word('s'), node=Word('b'), other=Word('a'))]
    assert found[0].where == {'place': Word('s'), 'node': Word('b'), 'other': Word('a')}
    with pytest.raises(ValueError, match='owned by both'):
        first(iter(found))


def test_a_fold_is_generic_in_the_place_and_in_the_kind():
    """
    `Combine` is `core`'s because it reads no relation: it folds by the `+` the declared
    kind provides, and knows nothing else about it. Here that is a toy condition node in a toy
    place, and the fold still binds, keeps a binding and refuses a kind that is not it.
    """
    a, b, t = Word('a'), Word('b'), Word('t')
    g = G({a: states(['y'], ['x']), b: states(['z'], ['w']), t: toy([], ['k'])})

    fresh = Combine(Word('both'), (a, b), States).apply(g)
    assert tuple(fresh.definitions) == (t, Word('both'))          # bound at the end
    assert fresh[Word('both')].conditions == places(['y', 'z'])
    assert fresh[Word('both')].unknowns == places(['x', 'w'])

    kept = Combine(a, (a, b), States).apply(g)
    assert tuple(kept.definitions) == (a, t)                      # `a` where it was
    assert kept[a] == g[a] + g[b]

    with pytest.raises(TypeError, match='one kind only'):
        Combine(a, (a, t), States).apply(g)


def test_a_cut_refuses_the_producer_among_its_readers():
    """Renaming at the producer would move its *output*, leaving `var` a boundary input."""
    a, b = Word('a'), Word('b')
    g = G({a: toy(['x'], ['y']), b: toy(['y'], ['z'])})
    hat = lambda w: Word(w.segments + ('hat',))
    assert [type(v) for v in refusals(Cut(Word('y'), (a,), hat), g)] == [DoesNotRead]
    with pytest.raises(ValueError, match='do not read'):
        Cut(Word('y'), (a,), hat).apply(g)
    assert list(refusals(Cut(Word('y'), (b,), hat), g)) == []            # the reader: fine


def test_a_fold_is_asked_of_its_result_and_never_of_a_step_toward_it():
    """`P -> h -> Q -> k -> R -> f -> P`, `Q` nested in `P`. Removing `R` alone would take
    `Q` off `P`'s cycle; folding `R` into `P` keeps it on, and the result is the graph
    written directly."""
    P, h, Q, k, R, f = map(Word, ('P', 'h', 'Q', 'k', 'R', 'f'))
    defs = {P: states(['c'], ['p']), h: toy(['p'], ['x1']), Q: states(['x1'], ['q']),
            k: toy(['q'], ['x2']), R: states(['x2'], ['r']), f: toy(['r'], ['c'])}
    g = G(defs, {Q: P})
    folded = Combine(P, (P, R), States).apply(g)
    direct = G({P: defs[P] + defs[R], h: defs[h], Q: defs[Q], k: defs[k], f: defs[f]}, {Q: P})
    assert folded == direct
    assert tuple(folded.definitions) == (P, h, Q, k, f)
    with pytest.raises(ValueError, match='not on one cycle'):
        Delete((R,)).apply(g)                                   # the step alone is refused


def test_an_ownership_violation_names_each_owner():
    """The standing owner, not a new one; and with three owners each names its own."""
    a, b, c = Word('a'), Word('b'), Word('c')
    found = proof(G({a: toy([], ['y'])}), Insert, {c: toy([], ['y'])})
    assert found == [OwnedTwice(place=Word('y'), node=c, other=a)]
    three = {a: toy([], ['y']), b: toy([], ['y']), c: toy([], ['y'])}
    assert list(violations(AbstractGraph, OrderedMap(three))) == [
        OwnedTwice(place=Word('y'), node=b, other=a), OwnedTwice(place=Word('y'), node=c, other=a)]
    around = {a: toy([], ['s.u']), b: toy([], ['s.v']), c: toy([], ['s'])}
    assert list(violations(AbstractGraph, OrderedMap(around))) == [
        OwnAroundOwner(Word('s'), Word('s.u'), c, a), OwnAroundOwner(Word('s'), Word('s.v'), c, b)]


def test_a_nesting_with_a_bad_end_is_listed_not_raised():
    """`check_nesting_ends_on_one_cycle` skips a pair another rule refuses."""
    a, t = Word('a'), Word('t')
    graph = AbstractGraph(OrderedMap({a: states([], ['x']), t: toy([], ['y'])}))
    assert [type(v) for v in violations(NestingGraph, graph, OrderedMap({Word('zz'): a}))] == [Unbound]
    assert [type(v) for v in violations(NestingGraph, graph, OrderedMap({t: a}))] == [NotAProblem]


def test_a_contraction_refuses_a_block_owning_nothing():
    a, b = Word('a'), Word('b')
    g = G({a: states(['x'], []), b: states(['y'], [])})
    assert list(check_blocks_own_something(g, ((a, b),))) == [BoxOwnsNothing((a, b))]
    with pytest.raises(ValueError, match='owns nothing'):
        g.contract(((a, b),))


class Said(Answer):
    '''A toy interface, answered for a word of its own.'''

    about = Word


class Spoken(Word): ...


@answers(Spoken)
class BySpeech(Said): ...


def test_a_graph_is_answered_for_the_class_of_its_places_or_of_its_names():
    g = G({Spoken('a'): Toy((), (Spoken('x'),))})
    assert answer_for_places(Said, g) == answer_for_labels(Said, g) == BySpeech()
    with pytest.raises(TypeError, match='Word has no Said answer'):
        answer_for_places(Said, G({Word('a'): Toy((), (Word('x'),))}))


def test_every_case_a_rule_can_name_has_wording():
    """
    A missing template is a `KeyError` raised at the moment something else went wrong,
    which is the worst time to find one. So every case the rules can produce is exercised.
    """
    produced = set()

    def seen(violations):
        for v in violations:
            produced.add(type(v))
            v.message                        # formats, so the roles match the template
        return produced

    a, b, t = Word('a'), Word('b'), Word('t')
    seen(proof(G({a: toy([], ['x'])}), Insert, {a: toy([], ['y'])}))       # name taken
    seen(proof(G({a: toy([], ['x'])}), Redefine, {b: toy([], ['y'])}))    # not bound
    seen(check_unnesting(G({a: states([], ['x'])}), a))                       # not nested
    seen(proof(G({}), Insert, {a: toy([], ['s']), b: toy([], ['s'])}))
    seen(proof(G({}), Insert, {a: toy([], ['s']), b: toy([], ['s.u'])}))
    seen(proof(G({a: toy([], ['s'])}), Insert,   {b: toy([], ['s.u'])}))
    seen(proof(G({a: toy([], ['s.u'])}), Insert, {b: toy([], ['s'])}))
    seen(proof(G({}), Insert, {a: toy([], ['s']), b: toy(['s.u'], ['t'])}))
    seen(proof(G({}), Insert, {a: toy([], ['s.u']), b: toy(['s'], ['t'])}))
    seen(proof(G({a: toy(['s'], ['x'])}), Insert,   {b: toy([], ['s.u'])}))
    seen(proof(G({a: toy(['s.u'], ['x'])}), Insert, {b: toy([], ['s'])}))
    seen(violations(Toy, places([]), places(['s', 's'])))
    seen(violations(Toy, places([]), places(['s', 's.u'])))
    seen(violations(Toy, places(['s']), places(['s'])))
    seen(violations(Toy, places(['s.u']), places(['s'])))
    seen(violations(Toy, places(['s']), places(['s.u'])))
    seen(violations(Declared, places([]), places([])))                       # owns nothing
    seen(violations(States, (), ()))                                        # an empty condition node
    seen(violations(Insert, OrderedMap({})))                                   # nothing inserted
    seen(violations(Delete, ()))                                            # nothing deleted
    seen(violations(Cut, Word('x'), (), lambda w: w))                       # a copy nothing reads
    seen(violations(Determine, a, ()))                                      # determines nothing
    seen(violations(Determine, a, places(['u', 'u'])))                      # an unknown twice
    seen(violations(Combine, a, (b,), States))                              # fewer than two
    seen(violations(Combine, a, (b, b), States))                            # a node twice
    seen(violations(Combine, a, (a, b), Toy))                               # a kind with no `+`

    # `checks_graph`: what an op needs of the graph, asked before it computes anything --
    # and asked for, which is what `refusals` is, so nothing here is built by hand.
    hat  = lambda w: Word(w.segments + ('hat',))
    onto = lambda w: Word('y')
    g = G({a: states(['y'], ['u']), b: toy([], ['x'])})
    seen(refusals(Determine(a, places(['v'])), g))                      # already determined
    seen(refusals(Undetermine(a), G({a: states(['y'], [])})))           # nothing to drop
    seen(refusals(Rename(b, ((Word('nope'), Word('q')),)), g))          # a place it has not got
    seen(refusals(Cut(Word('free'), (b,), hat), g))                     # nothing produces it
    seen(refusals(Cut(Word('x'), (a,), onto), g))                       # the minted name is taken
    seen(refusals(Cut(Word('x'), (b,), hat), g))                        # `b` produces it, reads nothing
    seen(refusals(Combine(a, (a, b), States), g))                       # `b` is a `Toy`

    cyc = {a: states(['y'], ['x']), b: states(['x'], ['y']), t: toy(['x'], ['w'])}
    with pytest.raises(KeyError, match='not a node of this graph'):     # `check_nesting_ends_bound`, at construction
        G({a: states([], ['x'])}, {a: b})
    seen(check_delete_keeps_nesting_whole(G(cyc, {a: b}), Rebinding(frozenset({a}), {})))  # one end lost
    seen(check_nesting(G({a: states([], ['x'])}), (a, b)))                    # unbound
    seen(check_nesting(G(cyc), (a, t)))                                       # t not a problem
    seen(check_nesting(G(cyc, {a: b}), (a, b)))                               # already
    seen(check_nesting(G(cyc), (a, a)))                                       # inside itself
    seen(check_nesting(G(cyc, {b: a}), (a, b)))                               # loop
    seen(violations(NestingGraph, AbstractGraph(OrderedMap({a: states([], ['x']), b: states([], ['y'])})), OrderedMap({a: b})))          # unreachable
    seen(check_blocks_own_something(G({a: states(['x'], []), b: states(['y'], [])}), ((a, b),)))  # a box owning nothing
    seen(check_blocks_partition(G({a: toy([], ['x']), b: toy([], ['y'])}), ((a,), (a, b))))  # a short block, a node twice
    S = G
    seen(violations(ExecutableGraph, S({a: body(['u'], ['x']), b: body(['x'], ['u'])})))     # coupled, no problem
    seen(violations(ExecutableGraph, S({a: body(['u'], ['x']), b: body(['x'], ['u']),
                                        t: driven(['x'], ['w'])})))                          # a-b still cyclic
    seen(violations(ExecutableGraph, S({t: driven(['x'], ['w']), a: body([], ['x'])})))   # every condition constant: off the cycle
    seen(violations(ExecutableGraph, S({t: driven([], ['w'])})))                   # no condition at all: off the cycle
    seen(violations(ExecutableGraph, S({a: states([], ['x']), b: states([], ['y'])})))
    seen(violations(ExecutableGraph, S({a: body([], ['x']), b: states(['x'], [])})))         # a bare condition
    seen(violations(ExecutableGraph, S({t: driven(['k', 'y'], ['u']), a: body(['u'], ['y'])})))   # one constant
    seen(violations(ExecutableGraph, S({t: driven(['x'], ['u'], ['g']), a: body(['u'], ['x', 'g'])})))   # g, read unheld, produced on t's cycle
    seen(violations(ExecutableGraph, S({t: driven(['x'], ['u'], reports=['r']), a: body(['u', 'r'], ['x'])})))  # r, owned unanswered, read on t's cycle
    seen(violations(ExecutableGraph, S({a: body(['u'], ['x']), b: body(['x'], ['u'])})))        # a cycle nobody answers
    seen(violations(ExecutableGraph, S(cyc)))                                                   # unresolved: two outermost
    seen(violations(ExecutableGraph, S({a: body(['u', 'w'], ['x']), b: body(['x'], ['u']),
                                 t: driven(['x'], ['w'])})))                              # a-b still cyclic inside t
    seen(violations(answer_for_labels, Said, G({})))                                          # no node to ask
    seen(violations(GraphWithValues, ExecutableGraph(G({a: body(['k'], ['x'])})),
                    OrderedMap({Word('x'): 1.0, Word('z'): 2.0})))      # k missing, x owned, z none

    def finals(cls):
        return {s for c in cls.__subclasses__() for s in ({c} | finals(c))
                if not s.__module__.startswith((              # those finals' cases are their own tests'
                    'cottax.relational', 'cottax.ode', 'cottax.core.maths.cut_rules', 'cottax.core.driven', 'cottax.core.realisation', 'cottax.consumers.execution',
                    'cottax.consumers.json', 'cottax.consumers.pickle', 'cottax.consumers.visualization',
                    'cottax.consumers.declaration', 'cottax.consumers.surface', 'cottax.consumers.architectures',
                    'cottax.assemblies.jax',
                    'cottax.pytree.consumers.surface', 'cottax.core.frozen.answer'))}
    assert finals(Violation) <= produced, f'never produced: {finals(Violation) - produced}'


def test_a_rewrite_can_mint_into_any_place():
    '''
    What the language requires of a place, in full: segments it can be rebuilt from,
    and `str`. `Word` declares nothing else -- so this is the whole obligation a second
    implementation takes on, and minting into one has to work without it knowing what a
    key path is.

    It is asserted because it was silently false: a minted `Word` constructed and
    round-tripped fine and then raised `TypeError` the first time a condition node printed
    one, from `model/`'s own `__repr__`.
    '''
    from cottax.relational import Eq, Relation, RelationalCondition

    from cottax.core import mint_of, unminted

    hat = purposes.Copy()(Word('s.u'))
    assert isinstance(hat, Word) and hat.segments[1:] == ('s', 'u')
    assert mint_of(hat) == purposes.Copy() and unminted(hat) == Word('s.u')
    assert mint_of(Word('s.u')) is None and unminted(Word('s.u')) == Word('s.u')

    problem = RelationalCondition((Relation(Word('r'), None, Eq),), (), (hat,), ())
    assert problem.owns == (hat,)
    assert '^copy.s.u' in repr(problem), repr(problem)   # and it prints


# ---------------------------------------------------------------- the laws, on `Path`

def _random_paths(n, seed=0):
    random.seed(seed)
    keys = [GetAttrKey('a'), GetAttrKey('b'), SequenceKey(0), SequenceKey(1)]
    return [VarPath(tuple(random.choice(keys) for _ in range(random.randint(0, 4))))
            for _ in range(n)]


def test_is_prefix_agrees_with_prefixes():
    # `is_prefix` is one slice over `segments`; `prefixes` builds places. Both are
    # derived from the same single member, and they must agree.
    names = _random_paths(120)
    for a, b in itertools.islice(itertools.product(names, names), 8000):
        assert is_prefix(a, b) == (a == b or a in list(prefixes(b)))


def test_the_lookup_agrees_with_the_definition():
    '''
    `comparables_in` walks the *prefixes of one name* and scans for extensions; the
    definition iterates the *members of the collection*. The lemma is that these are the
    same question:

        {x in among : x ~ p}   ==   {p if p in among} + {q in prefixes(p) : q in among} + {x in among : p < x}

    One side is bounded by the depth of a name where it can be, which is the whole reason
    the walk exists -- and the reason it once disagreed, by starting its loop past the root.
    '''
    names = _random_paths(200, seed=1)
    among = {n: 1 for n in names}
    for p in names:
        fast          = list(comparables_in(p, among))
        by_definition = [x for x in among if comparable(x, p)]
        assert sorted(fast, key=repr) == sorted(by_definition, key=repr)
        assert len(fast) == len(set(fast))


def test_overlaps_reports_the_same_pairs_either_way_round():
    a = [Word('s'), Word('t.u'), Word('x')]
    b = [Word('s.u'), Word('s.v'), Word('t'), Word('x')]
    forward  = {(p, q) for p, q in overlaps(a, b)}
    backward = {(q, p) for p, q in overlaps(b, a)}
    assert forward == backward == {
        (Word('s'), Word('s.u')), (Word('s'), Word('s.v')), (Word('t.u'), Word('t')), (Word('x'), Word('x')),
    }


# ---------------------------------------------------------------- the vocabulary's own SCC

def test_components_are_computed_from_ports_alone():
    '''`components` is a fact of the abstract graph, over its projection. So the
    projection is what to pin -- an edge is `owns` meeting `reads`, on toys.'''
    from cottax.core.maths.graph import nx_dependencies
    a, b, c = Word('a'), Word('b'), Word('c')
    defs = {a: toy(['y'], ['x']), b: toy(['x'], ['y']), c: toy(['y'], ['z'])}
    assert set(nx_dependencies(defs).edges) == {(a, b), (b, a), (b, c)}
    assert G(defs).graph.components == ((a, b), (c,))
    assert G({b: defs[b], a: defs[a]}).graph.components == ((b, a),)      # members in the order given
