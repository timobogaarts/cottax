'''
`core/` alone: the vocabulary, and the two laws that tie it to `Path`.

The rules are stated over protocols -- a place answers `prefixes` and `is_prefix`, a
definition answers `owns` and `reads` -- so this module builds *toy* places out of strings
to show the rules never learn what a real name is made of, and then holds the real `Path`
to the laws the vocabulary states.
'''

from toy_places import Segmented
from cottax.core import conditions
from cottax.core import Dependencies, Nesting

import dataclasses
import itertools
import random
from dataclasses import dataclass

import pytest
from graphs import functional, refusals
from jax.tree_util import GetAttrKey, SequenceKey

from cottax.core.maths.places import comparable, is_prefix, overlaps
from cottax.core.maths.valid import (
    OwnAroundOwner, OwnInsideOwner, OwnedTwice, OwnerInsideRead, ReadInsideOwner,
)
from cottax.core import (
    Graph, Condition, ConditionalNode, Cut, FunctionalNode,
    Rename, Replace, SetOwnsInside,
    RemoveAttribute, SetAttribute,
    Violation, NodeDefinition, Insert, Delete,
    violations, comparables_in, first,
    answer_for_labels, answer_for_places, Valid,
)
from cottax.core.plan import Edit
from cottax.core.plan.primitive_ops import Nest, Unnest
from cottax.core.refusal_kinds import DoesNotRead, ReadOnBothSides
from cottax.core.maths.executable import Executable
from cottax.core.frozen import Answer, OrderedMap, answers
from cottax.core.maths.valid import EndOwnsNothingInside, Unbound
from cottax.pytree.path import VarPath
from cottax.core import Mint, Minting
from cottax.core.conditional import Driver, DriverContract, PerCondition
from cottax.consumers.execution import GraphWithValues
import purposes


# ---------------------------------------------------------------- a toy place

@dataclass(frozen=True)
class Word(Segmented):
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


Declared = FunctionalNode
'''A computation with nothing to call.'''


class States(Condition):
    '''A bare condition: two bare tuples, and no relation anywhere. **The whole of a
    `Condition`** -- what it reads and owns inside -- so every rule in the layer runs on a
    condition node whose condition was never written as `lhs op rhs`.'''
    reads_inside  : tuple = ()
    owns_inside   : tuple = ()
    reads_outside = ()
    owns_outside  = ()
    def relabel(self, renaming): return self


class Driven(Condition):
    '''A problem with an algorithm, as the abstract sees it: a condition
    reading and owning outside as well as inside. No driver
    anywhere -- what one *is* is `problem/`'s, and the rules never ask.'''
    problem : States
    data    : tuple = ()
    reports : tuple = ()
    reads_inside  = property(lambda self: self.problem.reads_inside)
    owns_inside   = property(lambda self: self.problem.owns_inside)
    reads_outside = property(lambda self: tuple(v for vs in self.data for v in vs))
    owns_outside  = property(lambda self: tuple(v for vs in self.reports for v in vs))
    def relabel(self, renaming): return self


class Taken(PerCondition):
    '''A datum a driver may be given: one place, read outside the node's cycle.'''


class Anything(DriverContract):
    '''A contract answering any condition, taking `Taken` where it is given: a node's place
    rules are asked before whether its driver fits.'''
    accepts  = staticmethod(lambda condition: True)
    takes    = ()
    may_take = (Taken,)


def ported(reads=()):
    '''A driver taking `reads`, its places read outside the node's cycle.'''
    return Driver(Anything(), OrderedMap({Taken: tuple(reads)}) if reads else OrderedMap())


class Note:
    '''An attribute with no opinion on the node carrying it: a title, a tag.'''


class Shaped:
    '''An attribute written for a node's shape -- which the maths never asks it about.'''


class SetNote(SetAttribute):        kind = Note
class DropNote(RemoveAttribute):    kind = Note
class SetShaped(SetAttribute):      kind = Shaped


def body(reads, owns):          return declared(reads, owns)   # a body is no edge: the abstract sees a function
def declared(reads, owns):      return Declared(places(reads), places(owns))
def condition(reads, owns):     return States(places(reads), places(owns))
def states(reads, owns):        return ConditionalNode(condition(reads, owns))
def driven(reads, owns, data=(), reports=()):
    return ConditionalNode(Driven(condition(reads, owns), ((places(data),) if data else ()),
                                  ((places(reports),) if reports else ())))


def G(definitions, within=None):
    '''A nesting graph over two `OrderedMap`s: nothing on it is abstract.'''
    return Graph(OrderedMap(definitions), OrderedMap(within or {}))


def proof(graph, op, define):
    '''Why `op` of `define` does not apply to `graph`, or what `Valid` refuses of the graph
    it leaves, as data -- the rules of the bindings that would result among them.'''
    edit = op(OrderedMap(define))
    return list(refusals(edit, graph)) or list(Valid.check(edit.apply(graph)))


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
    g = Graph(OrderedMap({Word(n): declared(*io) for n, io in bound.items()}))
    assert [c[0] for c in Dependencies.of(g).components] == [Word(n) for n in run]


def test_a_rule_yields_data_and_never_a_sentence():
    # A violation is a final per case, its fields the places by role. The wording is
    # `messages`', which is the only module in `core/` that asks a place how it is spelled.
    found = proof(G({}), Insert, {Word('a'): toy([], ['s']), Word('b'): toy([], ['s'])})
    assert found == [OwnedTwice(place=Word('s'), node=Word('b'), other=Word('a'))]
    assert found[0].where == {'place': Word('s'), 'node': Word('b'), 'other': Word('a')}
    with pytest.raises(ValueError, match='owned by both'):
        first(iter(found))


@pytest.mark.parametrize('inside, outside', [(['r'], ['r']), (['r'], ['r.a']), (['r.a'], ['r'])])
def test_a_read_inside_is_apart_from_every_read_outside_in_the_prefix_order(inside, outside):
    """A read inside is produced on the cycle, and with it every place inside it and every
    place it lies inside: so reading `r` inside and `r.a` outside reads part of what the
    cycle produces as if it were not -- the same place, in the order that matters. Asked of
    the condition's own sides, and of a driver's place against the condition."""
    refused = [ReadOnBothSides(places(inside), places(outside))]
    assert list(violations(Driven, condition(inside, ['u']), (places(outside),))) == refused
    assert list(violations(ConditionalNode, condition(inside, ['u']),
                           driver=ported(places(outside)))) == refused


def test_reads_apart_in_the_prefix_order_may_be_on_either_side():
    assert not list(violations(Driven, condition(['r.a'], ['u']), (places(['r.b']),)))


def test_a_cut_refuses_the_producer_among_its_readers():
    """Renaming at the producer would move its *output*, leaving `var` a boundary input."""
    a, b = Word('a'), Word('b')
    g = G({a: toy(['x'], ['y']), b: toy(['y'], ['z'])})
    hat = lambda w: Word(w.segments + ('hat',))
    assert [type(v) for v in refusals(Cut(Word('y'), (a,), hat), g)] == [DoesNotRead]
    with pytest.raises(ValueError, match='do not read'):
        Cut(Word('y'), (a,), hat).apply(g)
    assert list(refusals(Cut(Word('y'), (b,), hat), g)) == []            # the reader: fine


def test_an_ownership_violation_names_each_owner():
    """The standing owner, not a new one; and with three owners each names its own."""
    a, b, c = Word('a'), Word('b'), Word('c')
    found = proof(G({a: toy([], ['y'])}), Insert, {c: toy([], ['y'])})
    assert found == [OwnedTwice(place=Word('y'), node=c, other=a)]
    three = {a: toy([], ['y']), b: toy([], ['y']), c: toy([], ['y'])}
    assert list(Valid.check(Graph(OrderedMap(three)))) == [
        OwnedTwice(place=Word('y'), node=b, other=a), OwnedTwice(place=Word('y'), node=c, other=a)]
    around = {a: toy([], ['s.u']), b: toy([], ['s.v']), c: toy([], ['s'])}
    assert list(Valid.check(Graph(OrderedMap(around)))) == [
        OwnAroundOwner(Word('s'), Word('s.u'), c, a), OwnAroundOwner(Word('s'), Word('s.v'), c, b)]


def test_a_nesting_with_a_bad_end_is_listed_not_raised():
    """A pair with a bad end is the one rule's that names it."""
    a, t = Word('a'), Word('t')
    graph = Graph(OrderedMap({a: states([], ['x']), t: toy([], ['y'])}))
    assert [type(v) for v in Valid.check(Graph(graph.definitions, OrderedMap({Word('zz'): a})))] == [Unbound]
    assert [type(v) for v in Valid.check(Graph(graph.definitions, OrderedMap({t: a})))] == [EndOwnsNothingInside]


def test_what_a_box_must_be_its_kind_refuses():
    '''A block owning nothing is no computation: the `FunctionalNode` box refuses it as it
    is built. Contraction itself asks nothing of what a box is.'''
    a, b = Word('a'), Word('b')
    g = G({a: states(['x'], []), b: states(['y'], [])})
    with pytest.raises(ValueError, match='must own at least one'):
        g.contracted({a: (a, b)}, functional)
    boxed = g.contracted({a: (a, b)}, lambda members, reads, owns: Toy(reads, owns))
    assert boxed.nodes == (a,) and boxed[a].reads == places(['x', 'y'])


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


def test_what_a_condition_node_owns_inside_is_set_where_it_is_stored():
    """`SetOwnsInside` writes the field on any kind that stores it -- no relation anywhere
    -- and both ways: some makes an encloser, none a requirement again."""
    a, b = Word('a'), Word('b')
    g = G({a: states(['y'], []), b: toy(['u'], ['y'])})
    set_ = SetOwnsInside(a, places(['u'])).apply(g)
    assert set_[a] == states(['y'], ['u']) and conditions(set_.definitions) == (a,)
    assert SetOwnsInside(a, ()).apply(set_)[a] == states(['y'], [])
    assert repr(SetOwnsInside(a, places(['u']))) == 'set_owns_inside(a <- u)'


# ---------------------------------------------------------------- what a node carries

def test_an_attribute_is_set_and_taken_off_and_is_in_equality():
    a = Word('a')
    g = G({a: toy([], ['x'])})
    noted = SetNote(OrderedMap({a: 'n'})).apply(g)
    assert noted[a].attributes == OrderedMap({Note: 'n'})           # the key, and the value as given
    assert noted[a] != g[a] and noted != g                  # a node is what it carries too
    assert DropNote((a,)).apply(noted) == g
    assert [type(v).__name__ for v in refusals(SetNote(OrderedMap({a: 'm'})), noted)] \
        == ['AttributeAlreadySet']
    assert [type(v).__name__ for v in refusals(DropNote((a,)), g)] == ['AttributeNotSet']


def test_a_replaced_definition_is_handed_over_attributes_and_all():
    """`Replace` hands a definition over, attributes and all: what the old one carried goes."""
    a = Word('a')
    g = G({a: dataclasses.replace(toy(['p'], ['q']), attributes=OrderedMap({Note: 'n'}))})
    replaced = Replace(a, toy(['p', 's'], ['q'])).apply(g)
    assert replaced[a].reads == places(['p', 's']) and replaced[a].attributes == OrderedMap()


def test_the_maths_asks_an_attribute_nothing():
    """Core cannot read an attribute, and asks nothing of one: any goes on any node, by hand
    or by an op, and is carried through a rename. Whether one fits its node is the rule of
    the layer defining it (an implementation's: `test_realisation`)."""
    a, b = Word('a'), Word('b')
    g = G({a: toy(['p'], ['q']), b: toy(['p', 's'], ['t'])})
    assert dataclasses.replace(g[b], attributes=OrderedMap({Shaped: 1})).attributes == OrderedMap({Shaped: 1})
    assert list(refusals(SetShaped(OrderedMap({b: 1})), g)) == []
    shaped = SetShaped(OrderedMap({a: 1})).apply(g)
    assert Rename(a, ((Word('p'), Word('r')),)).apply(shaped)[a].attributes == OrderedMap({Shaped: 1})
    assert SetNote(OrderedMap({b: 'n'})).apply(g)[b].attributes == OrderedMap({Note: 'n'})


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
    seen(refusals(Delete((b,)), G({a: toy([], ['x'])})))                  # not bound
    seen(refusals(Unnest(a), G({a: states([], ['x'])})))                            # not nested
    seen(refusals(Delete((a,)), G({a: states([], ['x']), b: states([], ['y'])}, {a: b})))  # one end stranded
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
    seen(violations(Driven, condition(['r'], ['u']), (places(['r']),)))     # r read inside and outside
    seen(violations(Insert, OrderedMap({})))                                   # nothing inserted
    seen(violations(Delete, ()))                                            # nothing deleted
    seen(violations(Cut, Word('x'), (), lambda w: w))                       # a copy nothing reads
    seen(violations(SetOwnsInside, a, places(['u', 'u'])))                  # a place owned inside twice
    seen(violations(SetNote, OrderedMap()))                                            # names no node

    # `checks_graph`: what an op needs of the graph, asked before it computes anything --
    # and asked for, which is what `refusals` is, so nothing here is built by hand.
    hat  = lambda w: Word(w.segments + ('hat',))
    onto = lambda w: Word('y')
    g = G({a: states(['y'], ['u']), b: toy([], ['x'])})
    seen(refusals(SetOwnsInside(a, ()), G({a: driven(['y'], ['u'])})))  # a driven node derives it: not a choice
    attached = dataclasses.replace(states(['y'], ['u']), driver=ported())
    seen(refusals(SetOwnsInside(a, ()), G({a: attached})))                   # it carries a driver
    seen(refusals(Rename(b, ((Word('nope'), Word('q')),)), g))          # a place it has not got
    seen(refusals(Cut(Word('free'), (b,), hat), g))                     # nothing produces it
    seen(refusals(Cut(Word('x'), (a,), onto), g))                       # the minted name is taken
    seen(refusals(Cut(Word('x'), (b,), hat), g))                        # `b` produces it, reads nothing
    noted = G({a: dataclasses.replace(states(['y'], ['u']), attributes=OrderedMap({Note: 'n'})), b: states(['z'], ['v']),
               t: dataclasses.replace(toy(['p'], ['q']), attributes=OrderedMap({Shaped: 1}))})
    seen(refusals(SetNote(OrderedMap({a: 'm'})), noted))                # carries one under that key already
    seen(refusals(DropNote((b,)), noted))                               # carries none to take off

    cyc = {a: states(['y'], ['x']), b: states(['x'], ['y']), t: toy(['x'], ['w'])}
    with pytest.raises(KeyError, match='not a node of this graph'):     # `check_nesting_ends_bound`
        Valid.prove(G({a: states([], ['x'])}, {a: b}))
    seen(Valid.check(Graph(G({a: states([], ['x'])}).definitions, OrderedMap({a: b}))))   # unbound
    seen(Valid.check(Graph(G(cyc).definitions, OrderedMap({a: t}))))            # t not a problem
    seen(refusals(Nest(a, b), G(cyc, {a: b})))                                  # already
    seen(Valid.check(Graph(G(cyc).definitions, OrderedMap({a: a}))))            # inside itself
    seen(Valid.check(Nest(a, b).apply(G(cyc, {b: a}))))                         # loop, on the result
    seen(Executable.check(G({a: states([], ['x']), b: states([], ['y'])}, {a: b})))   # unreachable
    seen(violations(Graph.boxed, G({a: states([], ['x']), b: states([], ['y'])}), {Word('p'): (a,), Word('q'): (a, b)}, functional))  # a short block, a node twice
    S = G
    seen(Executable.check(S({a: body(['u'], ['x']), b: body(['x'], ['u'])})))     # coupled, no problem
    seen(Executable.check(S({a: body(['u'], ['x']), b: body(['x'], ['u']),
                                        t: driven(['x'], ['w'])})))                          # a-b still cyclic
    seen(Executable.check(S({t: driven(['x'], ['w']), a: body([], ['x'])})))   # every condition constant: off the cycle
    seen(Executable.check(S({t: driven([], ['w'])})))                   # no condition at all: off the cycle
    seen(Executable.check(S({a: states([], ['x']), b: states([], ['y'])})))
    seen(Executable.check(S({a: body([], ['x']), b: states(['x'], [])})))         # a bare condition
    seen(Executable.check(S({t: driven(['k', 'y'], ['u']), a: body(['u'], ['y'])})))   # one constant
    seen(Executable.check(S({t: driven(['x'], ['u'], ['g']), a: body(['u'], ['x', 'g'])})))   # g, read unheld, produced on t's cycle
    seen(Executable.check(S({t: driven(['x'], ['u'], reports=['r']), a: body(['u', 'r'], ['x'])})))  # r, owned unanswered, read on t's cycle
    seen(Executable.check(S({a: body(['u'], ['x']), b: body(['x'], ['u'])})))        # a cycle nobody answers
    seen(Executable.check(S(cyc)))                                                   # unresolved: two outermost
    seen(Executable.check(S({a: body(['u', 'w'], ['x']), b: body(['x'], ['u']),
                                 t: driven(['x'], ['w'])})))                              # a-b still cyclic inside t
    seen(violations(answer_for_labels, Said, G({})))                                          # no node to ask
    with pytest.raises(ValueError) as twice:                            # an edit naming one node twice
        Edit(remove=frozenset({a}), insert=OrderedMap({a: toy([], ['x'])})).applied(G({a: toy([], ['x'])}))
    seen(twice.value.violations)
    seen(violations(GraphWithValues, G({a: body(['k'], ['x'])}),
                    OrderedMap({Word('x'): 1.0, Word('z'): 2.0})))      # k missing, x owned, z none

    def finals(cls):
        return {s for c in cls.__subclasses__() for s in ({c} | finals(c))
                if not s.__module__.startswith((              # those finals' cases are their own tests'
                    'cottax.relational', 'cottax.ode', 'cottax.core.conditional.driver', 'cottax.core.conditional.assign',
                    'cottax.core.conditional.proof', 'cottax.core.realisation', 'cottax.consumers.execution',
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
        assert is_prefix(a, b) == (a == b or a in b.prefixes)


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
    from cottax.core.maths.dependencies import nx_dependencies
    a, b, c = Word('a'), Word('b'), Word('c')
    defs = {a: toy(['y'], ['x']), b: toy(['x'], ['y']), c: toy(['y'], ['z'])}
    assert set(nx_dependencies(G(defs)).edges) == {(a, b), (b, a), (b, c)}
    assert Dependencies.of(G(defs)).components == ((a, b), (c,))
    assert Dependencies.of(G({b: defs[b], a: defs[a]})).components == ((b, a),)      # members in the order given
