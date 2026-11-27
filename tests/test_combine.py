'''
`Combine`, the architectures' fold: condition nodes folded into one by the join their
condition's kind answers (`ModelReading.join`). Not core's, since what joins two conditions
is said by the language stating them -- so a toy kind in a toy place answers its own join
here, and the fold reads nothing else of it.
'''

from toy_places import Segmented
from cottax.core import Place

import dataclasses
from dataclasses import dataclass

import pytest
from graphs import refusals

from cottax.consumers.architectures import Combine
from cottax.consumers.architectures.interface import ModelReading
from cottax.consumers.architectures.refusals import (
    CarriesAttributes, FoldsDriven, NodeTwice, NotAddable, NotOfKind, NothingToCombine,
)
from cottax.core import (
    Condition, ConditionalNode, Delete, Executable,
    Graph, NodeDefinition, violations,
)
from cottax.core.conditional import Driver, DriverContract
from cottax.core.frozen import OrderedMap, answers
from cottax.core.maths.definition import ReadsOwn
from cottax.core.plan.edit import NameTaken
from cottax.core.plan.primitive_ops import Stranded


# ---------------------------------------------------------------- a toy place and kind

@dataclass(frozen=True)
class Word(Segmented):
    '''A place as a word of plain strings: the whole of `Place`.'''

    segments : tuple[str, ...]

    def __init__(self, segments):
        object.__setattr__(self, 'segments',
                           tuple(segments.split('.')) if isinstance(segments, str)
                           else tuple(segments))

    def __str__(self) -> str:
        return '.'.join(self.segments)


class Toy(NodeDefinition):
    '''A definition as two tuples of `Word`s.'''
    reads : tuple = ()
    owns  : tuple = ()
    def relabel(self, renaming): return self


class States(Condition):
    '''A bare condition: two bare tuples, no relation anywhere.'''
    reads_inside  : tuple = ()
    owns_inside   : tuple = ()
    reads_outside = ()
    owns_outside  = ()
    def relabel(self, renaming): return self


class Unjoined(States):
    '''A kind no reading joins.'''


@answers(States)
class StatesReading(ModelReading):
    '''The whole of what `Combine` asks of a kind: its join -- no relation anywhere.'''

    def not_implicit(self, graph, problem): return ()
    def body(self, graph, problem):         return ()
    def combines_as(self, condition):       return States

    def join(self, condition, other):
        return States(condition.reads_inside + other.reads_inside,
                      condition.owns_inside + other.owns_inside)


@answers(Unjoined)
class UnjoinedReading(ModelReading):
    def not_implicit(self, graph, problem): return ()
    def body(self, graph, problem):         return ()
    def combines_as(self, condition):       return None


class Anything(DriverContract):
    '''A contract answering any condition, taking and reporting nothing.'''
    accepts = staticmethod(lambda condition: True)
    takes   = ()


class Note:
    '''An attribute with no opinion on the node carrying it.'''


def places(names):          return tuple(Word(n) for n in names)
def toy(reads, owns):       return Toy(places(reads), places(owns))
def condition(reads, owns): return States(places(reads), places(owns))
def states(reads, owns):    return ConditionalNode(condition(reads, owns))


def G(definitions, within=None):
    return Graph(OrderedMap(definitions), OrderedMap(within or {}))


# ---------------------------------------------------------------- the fold

def test_a_fold_is_generic_in_the_place_and_in_the_kind():
    '''The fold reads no relation: it folds by the join the kind's reading answers, and
    knows nothing else about it. Here a toy condition in a toy place: the fold still binds,
    keeps a binding and refuses a kind that is not it.'''
    a, b, t = Word('a'), Word('b'), Word('t')
    g = G({a: states(['y'], ['x']), b: states(['z'], ['w']), t: toy([], ['k'])})

    fresh = Combine(Word('both'), (a, b), States).apply(g)
    assert tuple(fresh.definitions) == (t, Word('both'))          # bound at the end
    assert fresh[Word('both')].reads_inside == places(['y', 'z'])
    assert fresh[Word('both')].owns_inside == places(['x', 'w'])

    kept = Combine(a, (a, b), States).apply(g)
    assert tuple(kept.definitions) == (a, t)                      # `a` where it was
    assert kept[a] == states(['y', 'z'], ['x', 'w'])

    with pytest.raises(TypeError, match='one kind only'):
        Combine(a, (a, t), States).apply(g)


def test_a_fold_is_asked_of_its_result_and_never_of_a_step_toward_it():
    """`P -> h -> Q -> k -> R -> f -> P`, `Q` nested in `P`. Removing `R` alone would take
    `Q` off `P`'s cycle; folding `R` into `P` keeps it on, and the result is the graph
    written directly."""
    P, h, Q, k, R, f = map(Word, ('P', 'h', 'Q', 'k', 'R', 'f'))
    defs = {P: states(['c'], ['p']), h: toy(['p'], ['x1']), Q: states(['x1'], ['q']),
            k: toy(['q'], ['x2']), R: states(['x2'], ['r']), f: toy(['r'], ['c'])}
    g = G(defs, {Q: P})
    folded = Combine(P, (P, R), States).apply(g)
    direct = G({P: states(['c', 'x2'], ['p', 'r']), h: defs[h], Q: defs[Q], k: defs[k],
                f: defs[f]}, {Q: P})
    assert folded == direct
    assert tuple(folded.definitions) == (P, h, Q, k, f)
    alone = Delete((R,)).apply(g)                               # the step alone is carried ...
    assert dict(alone.within) == {Q: P}
    with pytest.raises(ValueError, match='not on one cycle'):
        Executable.prove(alone)                                  # ... and refused as executable


def test_a_fold_joins_conditions_and_neither_what_they_carry_nor_what_drives_them():
    '''What two nodes carry has no join, and which algorithm answers a join is not read off
    the two: a fold is of bare conditions, implemented and driven after.'''
    a, b = Word('a'), Word('b')
    noted = G({a: dataclasses.replace(states(['y'], ['u']), attributes=OrderedMap({Note: 'n'})),
               b: states(['z'], ['v'])})
    assert [type(v) for v in refusals(Combine(a, (a, b), States), noted)] == [CarriesAttributes]
    driven = G({a: dataclasses.replace(states(['y'], ['u']), driver=Driver(Anything())),
                b: states(['z'], ['v'])})
    assert list(refusals(Combine(a, (a, b), States), driven)) == [FoldsDriven((a,))]


def test_every_refusal_of_a_fold_has_wording():
    '''Each refusal kind `Combine` can name, produced and formatted.'''
    a, b, c, t = Word('a'), Word('b'), Word('c'), Word('t')
    def kinds(found):
        found = list(found)
        for v in found:
            v.message                                   # formats, so the roles match the template
        return [type(v) for v in found]
    assert kinds(violations(Combine, a, (b,), States)) == [NothingToCombine]
    assert kinds(violations(Combine, a, (b, b), States)) == [NodeTwice]
    assert kinds(violations(Combine, a, (a, b), Unjoined)) == [NotAddable]
    assert kinds(violations(Combine, a, (a, b), Toy)) == [NotAddable]
    g = G({a: states(['y'], ['u']), b: toy([], ['x']), t: states(['w'], ['s'])})
    assert kinds(refusals(Combine(a, (a, b), States), g)) == [NotOfKind]
    assert kinds(refusals(Combine(b, (a, t), States), g)) == [NameTaken]
    cyc = G({a: states(['y'], ['x']), b: toy(['x'], ['y']), c: states(['x'], ['z'])}, {a: c})
    assert kinds(refusals(Combine(t, (a, b), States), cyc)) == [NotOfKind, Stranded]
    assert kinds(refusals(Combine(t, (c, a), States), cyc)) == [ReadsOwn]   # the join's own refusal
