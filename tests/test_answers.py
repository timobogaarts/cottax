'''
A toy language and a toy place through every consumer, each defined here with nothing but
core's interface. Unanswered (`Raw`, `Plain`), a graph of them is built, proved and drawn
grey in every picture, and JSON refuses it, naming what has no answer.
Answered (`Zero`, `Tok`), with a toy vocabulary, names and driver implementation chosen, it
is declared, drawn in its own category in every picture, written to JSON and read back, run
in plain Python, and placed by an architecture closing its cuts its own way. And the table of what the shipped features answer, with each cell left
empty listed.
'''

from dataclasses import dataclass
import math
import subprocess
import sys

import pytest

from cottax.core import (
    AbstractGraph, ComposedOp, ConditionalNode, Cut, ExecutableGraph, Function, GaussSeidel,
    GraphWithValues, Insert, Mint, Minting, NestingGraph, is_problem, minted,
)
from cottax.core.frozen import OrderedMap, answer, answers, choice, violations
from cottax.core.maths.definition import relabelled
from cottax.core.maths.plan import Plan
from cottax.core.driven import DriverContract, Start
from cottax.consumers.architectures import MDA
from cottax.consumers.architectures.interface import Closing, ModelReading
from cottax.consumers.execution import CallableGraph, Drive, run
from cottax.consumers.execution.interface import DriverImplementation
from cottax.consumers.declaration import declare
from cottax.consumers.surface import core_role_places
from cottax.consumers.declaration.interface import Declaring
from cottax.consumers.declaration.interface_types import Wrapper, declares
from cottax.consumers.surface.interface import DerivedNames, Locator, Part
from cottax.consumers.surface.interface_types import Field, Index, Key
import purposes
from cottax.consumers.json import read_json, write_json
from cottax.consumers.json.refusals import Unexportable
from cottax.core.frozen.answer import Unanswered
from cottax.consumers.json.interface import LabelCodec, PlaceCodec, Spelling
from cottax.consumers.visualization import render_html, render_matrix_svg, render_xdsm_svg, to_dot
from cottax.consumers.visualization.interface import NodeDrawing, PlaceDrawing
from cottax.consumers.visualization.interface_types import Drawn, Kind, Slot
from cottax.consumers.visualization.xdsm import xdsm_struct


# ================================================================ core's interface alone

@dataclass(frozen=True)
class Plain:
    '''A place: segments, and a constructor taking them. Nothing answers for it.'''
    segments : tuple

    def __str__(self) -> str:
        return '/'.join(self.segments)


class Vanishes(ConditionalNode):
    '''A condition node whose conditions vanish: the level of the two finals below.'''
    conditions : tuple
    unknowns   : tuple
    reads      = property(lambda self: self.conditions)
    owns       = property(lambda self: self.unknowns)

    def relabel(self, renaming):
        return type(self)(relabelled(self.conditions, renaming), relabelled(self.unknowns, renaming))

    def determined(self, unknowns):
        return type(self)(self.conditions, unknowns)


class Raw(Vanishes):
    '''Nothing answers for it.'''


def coupled(place, kind):
    p = lambda *s: place(s)
    return NestingGraph(AbstractGraph(OrderedMap({
        p('body'): Function((p('k'), p('u')), (p('r'),)),
        p('cond'): kind((p('r'),), (p('u'),)),
    })), OrderedMap())


def test_with_core_alone_a_graph_is_built_proved_and_drawn_grey():
    g = coupled(Plain, Raw)
    assert ExecutableGraph(g).graph == g
    assert xdsm_struct(g, blocks=True)['problem_types'] == {'cond': 'stated'}
    for picture in (render_xdsm_svg, render_matrix_svg, render_html, to_dot):
        assert 'cond' in picture(g)


def test_with_core_alone_json_refuses_naming_what_is_missing():
    refused = violations(write_json, coupled(Plain, Raw))         # names and places alike
    assert [type(v) for v in refused] == [Unanswered, Unanswered]
    assert all('test_answers.Plain' in v.message for v in refused)
    (refused,) = violations(write_json, coupled(Tok, Raw))       # the place answered
    assert isinstance(refused, Unexportable) and refused.what == 'test_answers.Raw'


# ================================================================ the toy, answered

@dataclass(frozen=True)
class Tok:
    '''A place: segments, a minted one under its purpose as the first segment.'''
    segments : tuple

    def __str__(self) -> str:
        return f'Tok{self.segments}'


class Res(Mint):
    '''The toy's residual of an unknown: `^res.u`.'''


class Zeroed(Mint):
    '''Where the toy's `Zero` beside a model binds: `^zeroed.at`.'''


class GapOf(Mint):
    '''The toy closure's gap of a copy: `^gapof.copy`.'''


class Zero(Vanishes):
    '''The toy language's condition node, answered below.'''


def _spelled(word):
    '''The one purpose defined so far spelled `^word`: the test's own reading of its spelling.'''
    every, todo = [], [Mint]
    while todo:
        todo.extend(sub := todo.pop().__subclasses__())
        every.extend(sub)
    found, = {m for m in every if not m.__abstractmethods__ and str(m()) == f'^{word}'}
    return found()


def T(s):
    '''A toy name spelled with dots, a leading `^word` the purpose spelled so.'''
    first, _, rest = s.partition('.')
    return minted(_spelled(first[1:]), T(rest)) if first.startswith('^') else Tok(tuple(s.split('.')))



# ---------------------------------------------------------------- registered

@answers(Tok)
class TokDrawing(PlaceDrawing):
    def spell_plain(self, place):
        return '.'.join(map(str, place.segments))

    def spelling(self, place):
        return '.'.join(map(str, place.segments))

    def minted_from(self, place):
        first = place.segments[0]
        return (first, Tok(place.segments[1:])) if isinstance(first, Mint) else None

    def namespace_of(self, place, depth=None):
        return ()


ZERO = Kind('zero', 'zero (r = 0)', 'every condition vanishes', False, ('#abcdef', '#123456'))


@answers(Zero)
class ZeroDrawing(NodeDrawing):
    def drawn(self, node):
        return Drawn(ZERO, (Slot(('= 0',), node.conditions),))


def tok_segments(tok, w):
    '''A segment as it is; a purpose as the value it is.'''
    return [w.value(k) if isinstance(k, Mint) else k for k in tok.segments]


def tok_read(data, r):
    return Tok(tuple(r.value(k) if isinstance(k, dict) else k for k in data))


@answers(Tok)
class TokCodec(PlaceCodec):
    tag = 'tok'
    def write_place(self, place, w): return tok_segments(place, w)
    def read_place(self, data, r):   return tok_read(data, r)


@answers(Tok)
class TokLabelCodec(LabelCodec):
    tag = 'tok'
    def write_label(self, name, w):  return tok_segments(name, w)
    def read_label(self, data, r):   return tok_read(data, r)


@answers(Tok)
class TokMinting(Minting):
    '''A purpose as the first segment: a toy's own segments are strings.'''
    def minted(self, m, tok):  return Tok((m, *tok.segments))
    def mint_of(self, tok):    return next((k for k in tok.segments[:1] if isinstance(k, Mint)), None)
    def unminted(self, tok):   return Tok(tok.segments[1:]) if self.mint_of(tok) else tok


@answers(Zero)
class ZeroSpelling(Spelling[Zero]):
    kind = 'zero'

    def write(self, node, w):
        return {'conditions': w.places(node.conditions), 'unknowns': w.places(node.unknowns)}

    def read(self, data, r):
        return Zero(r.places(data['conditions']), r.places(data['unknowns']))


# ---------------------------------------------------------------- chosen

class Secant(DriverContract):
    '''A contract of the toy's: any problem of one scalar unknown, from a start.'''
    accepts = staticmethod(is_problem)
    takes   = (Start,)
    reports = ()


@choice(DriverImplementation)
class SecantSteps(DriverImplementation[Secant]):
    needs    = frozenset({Start})
    provides = frozenset()

    def __call__(self, posed, context, data):
        (x0,) = data[Start]
        x1 = x0 + 1.0
        r = lambda x: posed.conditions((x,), context)[0]
        for _ in range(60):
            if (d := r(x1) - r(x0)) == 0:
                break
            x0, x1 = x1, x1 - r(x1) * (x1 - x0) / d
        return (x1,), {}


@answers(Tok)
class TokDerived(DerivedNames):
    '''The names derived from a toy place: a name one deeper a word more, a step in a model
    tree a word.'''

    def beside(self, place, name): return Tok((*place.segments, name))
    def node_at(self, place):    return place
    def place_at(self, node):    return node
    def node_in_tree(self, steps):
        return Tok(tuple(map(word, steps)))


@choice(Locator)
class Toks(Locator):
    '''The toy place's strings: a place or a node name is its dotted word.'''

    places = nodes = Tok

    def place(self, s):          return T(s)
    def node(self, s):           return T(s)


def word(step):
    match step:
        case Field(at) | Key(at) | Index(at):
            return str(at)


class Vanishing(Wrapper):
    '''A function writing a residual per unknown (`^res.u`), and the `Zero` problem holding
    each (at `^zeroed.at`).'''

    @property
    def function_outputs(self):
        return tuple(minted(Res(), u) for u in self.outputs)

    terms = ()

    def nodes(self, at):
        return ((at, self.function_node()),
                (minted(Zeroed(), at), Zero(self.function_outputs, self.outputs)))


@choice(Part)
class Zeroing(Declaring):
    '''The toy part: `zero`.'''

    zero = declares(Vanishing, '''A residual per unknown, each held at zero.''')


S = Zeroing(Toks())


def drive(*implementations):
    '''The toy's join, as the assembly's: every problem driven by default under `Secant`.'''
    return Drive(implementations, (Secant(),), core_role_places)


@S.zero('u')
def square(u, k):
    return u * u - k


def solved():
    declared = declare({'sq': square})
    return (Plan(declared.graph) + drive(SecantSteps())).graph


def test_the_toy_is_declared_through_its_vocabulary_and_runs_in_plain_python():
    g = solved()
    assert g.nodes == (T('sq'), Tok((Zeroed(), 'sq')))
    out = run(GraphWithValues(CallableGraph(g), OrderedMap({T('k'): 2.0,
                                                           Tok((purposes.Start(), 'u')): 1.0})))
    assert math.isclose(out[T('u')], math.sqrt(2.0), rel_tol=1e-12)


@dataclass(frozen=True)
class Plant:
    loops : dict


def test_the_toy_declares_a_tree_each_model_at_the_steps_to_it():
    cubed = S.zero(lambda w, k: w ** 3 - k, 'w')
    g = declare(Plant({'sq': square}), {'more': [cubed]}).graph
    assert g.nodes == (T('loops.sq'), Tok((Zeroed(), 'loops', 'sq')), T('more.0'),
                       Tok((Zeroed(), 'more', '0')))


def test_the_toy_is_drawn_in_its_own_category():
    g = solved()
    struct = xdsm_struct(g)
    assert struct['problem_types'] == {'^zeroed.sq': 'zero'}
    assert struct['slots'] == {'^zeroed.sq': [('= 0', ['^res.u'])]}
    assert struct['shown_vars']['^res.u'] == ('res(u)', None)          # its own minting
    assert 'zero (r = 0)' in render_xdsm_svg(g, blocks=True)
    assert 'zero (r = 0)' in render_html(g)                            # the page's key
    assert '#123456' in render_matrix_svg(g) and '#123456' in to_dot(g)   # in its colour


def test_the_toy_is_written_to_json_and_read_back():
    g = solved()
    data = write_json(g)
    assert (data['places'], data['labels']) == ('tok', 'tok') and data['nodes'][1]['definition']['node']['condition'] == {
        'kind': 'zero', 'conditions': [[{'class': 'test_answers:Res', 'fields': {}}, 'u']],
        'unknowns': [['u']]}
    assert read_json(data) == g


@answers(Zero)
class ZeroModel(ModelReading):
    '''Every `Zero` an implicit model, its body the nodes writing its conditions, folded by
    nothing.'''

    def not_implicit(self, graph, problem): return ()
    def combines_as(self, condition):       return None

    def body(self, graph, problem):
        owners = graph.graph.owners
        return tuple(dict.fromkeys(owners[c] for c in graph[problem].conditions))


class Gap(ComposedOp):
    '''The toy's closure: the cuts, a function writing each copy's gap (`^gap.copy`), and at
    `at` the `Zero` holding every gap.'''
    cuts : tuple
    at   : Tok

    def composed(self, graph):
        copies = tuple(c.copy for c in self.cuts)
        gaps   = tuple(map(GapOf(), copies))
        return (*self.cuts, Insert(OrderedMap({
            minted(GapOf(), self.at): Function((*(c.var for c in self.cuts), *copies), gaps),
            self.at: Zero(gaps, copies)})))


@choice(Closing)
class ClosedByGap(Closing):
    '''Each group closed by a `Gap` at `^closure.<the cycle's first member>`.'''

    def __call__(self, cycle, group):
        return Gap(tuple(Cut(v, readers, purposes.Copy()) for v, readers in group.items()),
                   self.bound_at(cycle, group))

    def bound_at(self, cycle, group):
        return minted(purposes.Closure(), cycle[0])


def test_the_toy_is_placed_by_an_architecture_closing_its_own_way():
    '''Two toy models coupled through their states: an MDA asks each what it is, cuts once,
    closes the cut by the toy's `Gap` and nests both models in the closure.'''
    g = NestingGraph(AbstractGraph(OrderedMap({
        T('a'): Function((T('k'), T('u'), T('v')), (T('ra'),)), T('A'): Zero((T('ra'),), (T('u'),)),
        T('b'): Function((T('u'), T('v')), (T('rb'),)),         T('B'): Zero((T('rb'),), (T('v'),)),
    })), OrderedMap())
    out = (Plan(g) + MDA(GaussSeidel(), ClosedByGap())).graph
    closure, hat = Tok((purposes.Closure(), 'a')), Tok((purposes.Copy(), 'v'))
    assert dict(out.within) == {T('A'): closure, T('B'): closure}
    assert out[closure] == Zero((minted(GapOf(), hat),), (hat,))
    assert ExecutableGraph(out).graph == out
    with pytest.raises(TypeError, match='test_answers.Raw has no ModelReading answer'):
        Plan(coupled(Tok, Raw)) + MDA(GaussSeidel(), ClosedByGap())


# ================================================================ the shipped features' table

NODE_INTERFACES  = {'NodeDrawing', 'Spelling', 'ModelReading'}
PLACE_INTERFACES = {'PlaceDrawing', 'PlaceCodec', 'LabelCodec', 'DataStructure', 'DerivedNames',
                    'Minting'}
WRITTEN_INTERFACES = {'Written'}
PURPOSE_INTERFACES = {'MintDrawing'}
'''Every interface a consumer registers answers under, by what it says it is about (`about`):
a node, a place or a node name, anything a study writes, or a purpose.'''

SHIPPED = {
    'NodeDrawing':      {'RelationalCondition': 'RelationalDrawing', 'OdeCondition': 'OdeDrawing'},
    'Spelling':         {'RelationalCondition': 'RelationalSpelling', 'OdeCondition': 'OdeSpelling'},
    'ModelReading':     {'RelationalCondition': 'RelationalModel', 'OdeCondition': 'OdeModel'},
    'PlaceDrawing':     {'Path': 'PathDrawing'},
    'PlaceCodec':       {'VarPath': 'PytreeCodec'},
    'LabelCodec':       {'NodePath': 'PytreeLabelCodec'},
    'DataStructure':    {'VarPath': 'PytreeStructure'},
    'DerivedNames':     {'VarPath': 'PytreeDerivedNames', 'NodePath': 'PytreeDerivedNames'},
    'Minting':          {'VarPath': 'PathMinting', 'NodePath': 'PathMinting'},
    'Written':          {'Keys': 'WrittenKeys'},
    'MintDrawing':      {'Cond': 'CondDrawing', 'Rate': 'RateDrawing', 'Initial': 'InitialDrawing'},
}
'''What `relational`, `ode` and `pytree` answer, per interface: the class answered for and
the answer.'''

UNANSWERED = {
    ('DataStructure', 'NodePath'): 'a node name indexes nothing in a caller\'s data',
    ('PlaceCodec', 'NodePath'):    'a node name is written by a `LabelCodec`',
    ('LabelCodec', 'VarPath'):     'a place is written by a `PlaceCodec`',
}
'''The cells left empty on purpose, and why.'''

TABLE = '''
import json
import cottax.consumers.execution, cottax.consumers.json, cottax.consumers.visualization
import cottax.pytree.consumers, cottax.relational.consumers, cottax.core
import cottax.ode.consumers
from cottax.core.frozen import Answer, answered
from cottax.ode import OdeCondition
from cottax.relational import RelationalCondition
from cottax.pytree.path import NodePath, VarPath
table = {i.__qualname__: {c.__qualname__: type(a).__qualname__ for c, a in t.items()
                          if type(a).__module__.startswith(('cottax.relational', 'cottax.ode',
                                                            'cottax.pytree'))}
         for i, t in answered().items()}
from cottax.core import Label, Mint, NodeDefinition, Place
def category(about):
    kinds = about if isinstance(about, tuple) else (about,)
    if all(isinstance(k, type) and issubclass(k, NodeDefinition) for k in kinds):
        return 'node'
    if set(kinds) <= {Place, Label}:
        return 'place'
    if kinds == (Mint,):
        return 'purpose'
    return 'written' if kinds == (object,) else repr(about)
interfaces = sorted(c.__qualname__ for c in Answer.__subclasses__())
categories = {c.__qualname__: category(c.about) for c in Answer.__subclasses__()}
def found(i, cls):
    return any(k in t for k in cls.__mro__ for j, t in answered().items() if j.__qualname__ == i)
cells = {i: {c.__qualname__: found(i, c)
             for c in (RelationalCondition, OdeCondition, VarPath, NodePath)}
         for i in interfaces}
print(json.dumps({'table': {i: t for i, t in table.items() if t}, 'interfaces': interfaces,
                  'categories': categories, 'cells': cells}))
'''


@pytest.mark.slow
def test_the_table_of_what_the_shipped_features_answer():
    import json
    out = subprocess.run([sys.executable, '-c', TABLE], capture_output=True, text=True,
                         check=True)
    got = json.loads(out.stdout)
    assert got['table'] == SHIPPED
    assert set(got['interfaces']) == (NODE_INTERFACES | PLACE_INTERFACES | WRITTEN_INTERFACES
                                      | PURPOSE_INTERFACES)
    by = lambda c: {i for i, k in got['categories'].items() if k == c}
    assert (by('node'), by('place'), by('written'), by('purpose')) == (
        NODE_INTERFACES, PLACE_INTERFACES, WRITTEN_INTERFACES, PURPOSE_INTERFACES)
    empty = {(i, c) for i, row in got['cells'].items() for c, ok in row.items() if not ok
             and i not in WRITTEN_INTERFACES | PURPOSE_INTERFACES
             and (c in ('RelationalCondition', 'OdeCondition')) == (i in NODE_INTERFACES)}
    assert empty == set(UNANSWERED)


def _owner(module: str) -> str:
    '''Who a module is: a feature (`cottax.<feature>`, its consumers' answers included), a
    consumer (`cottax.consumers.<consumer>`), or core.'''
    parts = module.split('.')
    if parts[1] == 'consumers':
        return '.'.join(parts[:3])
    return 'cottax.core' if parts[1] == 'core' else '.'.join(parts[:2])


def test_an_answer_lives_with_the_class_it_answers():
    '''Every shipped answer is registered by the owner of the class it answers: a feature for
    its own, a consumer for its own; and for a class core defines, which imports no consumer,
    by the consumer asking.'''
    import cottax.assemblies.jax  # noqa: F401  every shipped answer
    from cottax.core.frozen import answered
    stray = {}
    for interface, table in answered().items():
        if not interface.__module__.startswith('cottax.'):
            continue                                        # a test's own interface
        asking = _owner(interface.__module__)
        for cls, found in table.items():
            if not cls.__module__.startswith('cottax.'):
                continue                                    # a test's own toy
            owner = _owner(cls.__module__)
            expected = asking if owner == 'cottax.core' else owner
            if _owner(type(found).__module__) != expected:
                stray[f'{interface.__qualname__}[{cls.__qualname__}]'] = type(found).__module__
    assert len(answered()) >= 9 and stray == {}
