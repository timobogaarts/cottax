'''
A graph written out and read back: a pickle proved loadable elsewhere, and JSON over the
formulation that reads back equal. Each export is a value: its checks whether it
writes, `data` what it writes.
'''

import dataclasses
import json
import pickle
import sys

import pytest
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

from cottax.core import Function, answer_for_labels, answer_for_places
from cottax.core.frozen import answers, violations
from cottax.core.maths.plan import Plan
from cottax.relational.contracts import RootFinding
from cottax.core.maths.definition import relabelled
from cottax.core.maths.kinds import FunctionalNode
from cottax.consumers.json import read_json, write_json
from cottax.consumers.json.interface import LabelCodec, Spelling
from cottax.consumers.pickle import DefinedInMain, Unpicklable, write_pickle
from cottax.consumers.json.interface import PlaceCodec
from cottax.consumers.json.refusals import SpelledTwice
from cottax.consumers.json.refusals import UnknownKind, Unexportable
from cottax.core.frozen.answer import Unanswered
from cottax.core.maths.dispatch import NothingToAsk
import cottax.assemblies.jax as cx
from cottax.assemblies.jax import N as NAME, flat
from cottax.relational.consumers.execution.jax_solvers import Newton
from cottax.pytree.consumers.json import PytreeCodec, PytreeLabelCodec, spelled
from cottax.pytree.crossings import key_kind
from cottax.pytree.path import NodePath, VarPath
from cottax.core.realisation import ImplementedFunction, is_implemented
from cottax.relational import Eq, Le, Max, Min, Objective, Rel, Relation, RelationalCondition
from cottax.relational.consumers.json import RELS
from graphs import graph_of
from test_cutting import D1, D2, OPT, call, mdf, optimise, V, N
from test_cx import double, square



def solved():
    '''A realised graph: module-level bodies and a Newton with scalar settings.'''
    graph   = cx.absolute.declare(flat.residual(square, 'u'), flat.function(double, 'w')).graph
    problem = cx.absolute.node(cx.solve(NAME['square']))
    return (Plan(graph) + cx.absolute.assign(problem, RootFinding(steps=False))
            + cx.absolute.implement({problem: Newton(rtol=1e-10, atol=1e-10)})).graph


def sellar_mdf():
    '''Nested and minted, with lambda bodies: exportable only as structure.'''
    sellar = graph_of({
        OPT     : optimise(V('f'), (V('z'),), (V('g'),)),
        D1      : call(['z', 'y2'], ['y1']),
        D2      : call(['z', 'y1'], ['y2']),
        N('obj'): call(['z', 'y1', 'y2'], ['f']),
        N('con'): call(['y1'], ['g']),
    })
    return (Plan(sellar) + mdf()).graph


def written(g, implementations=True) -> dict:
    '''The file as another process reads it.'''
    return json.loads(json.dumps(write_json(g, implementations)))


# ---------------------------------------------------------------- pickle

def test_a_realised_graph_is_proved_picklable_and_loads_back_equal():
    g = solved()
    assert pickle.loads(write_pickle(g)) == g


def test_a_structural_graph_needs_no_executability():
    '''A cycle with no problem is no executable graph, and pickles all the same.'''
    x, y = VarPath((DictKey('x'),)), VarPath((DictKey('y'),))
    g = graph_of({NodePath((DictKey('a'),)): Function((x,), (y,)),
                  NodePath((DictKey('b'),)): Function((y,), (x,))})
    write_pickle(g)


def test_a_lambda_is_refused_by_name():
    g = sellar_mdf()
    found = list(violations(write_pickle, g))
    assert {type(v) for v in found} == {Unpicklable}
    assert D1 in {v.node for v in found}


def test_something_in_main_pickles_here_and_is_refused(monkeypatch):
    def body(x): return x
    body.__module__, body.__qualname__ = '__main__', 'exported_body'
    monkeypatch.setattr(sys.modules['__main__'], 'exported_body', body, raising=False)
    x, y = VarPath((DictKey('x'),)), VarPath((DictKey('y'),))
    g = graph_of({NodePath((DictKey('a'),)): ImplementedFunction(Function((x,), (y,)), body)})
    pickle.dumps(g)
    assert [type(v) for v in violations(write_pickle, g)] == [DefinedInMain]


# ---------------------------------------------------------------- json

def test_a_realised_graph_reads_back_equal():
    g = solved()
    assert read_json(written(g)) == g


def test_a_problems_parameters_are_written_and_read_back():
    g = cx.absolute.declare({'sum': flat.function(lambda z, x: z + x, 'f'),
                     'opt': flat.optimise(maximise='f', design={'z': ('z_lo', 0)})}).graph
    assert g[cx.absolute.node('opt')].parameters == (flat.place('z_lo'),)
    opt = cx.absolute.node('opt')
    assert read_json(written(g, implementations=False))[opt] == g[opt]


@dataclasses.dataclass(frozen=True)
class Scaled:
    """A body with a setting: written as its class and fields, not by name."""
    by : float

    def __call__(self, u):
        return self.by * u


def test_a_dataclass_body_is_written_as_its_class_and_fields():
    graph = cx.absolute.declare({'scaled': flat.function(Scaled(2.0), 'w')}).graph
    data  = written(graph)
    (entry,) = data['nodes']
    assert entry['definition']['implementation'] == {
        'class': 'test_export:Scaled', 'fields': {'by': 2.0}}
    assert read_json(data) == graph


def test_a_lambda_is_refused_and_the_structure_is_written_without_it():
    g = sellar_mdf()
    refused = list(violations(write_json, g, True))
    assert {type(v) for v in refused} == {Unexportable}
    assert {v.node for v in refused} == {n for n in g.nodes if is_implemented(g[n])}
    data = written(g, implementations=False)
    back = read_json(data)
    assert back.nodes == g.nodes and back.within == g.within and g.within
    assert not any(is_implemented(back[n]) for n in back.nodes)
    assert written(back) == data


def test_a_dict_key_and_an_index_stay_two_segments():
    '''Both spell `[0]`; the file keeps them apart.'''
    a, b = VarPath((DictKey(0),)), VarPath((SequenceKey(0),))
    g = graph_of({NodePath((DictKey('f'),)): Function((a,), (b,))})
    assert read_json(written(g)) == g


@dataclasses.dataclass(frozen=True)
class RowKey:
    '''A key kind of a study's own, registered once: walked, written as is, spelled.'''
    row : str


key_kind(RowKey, lambda key, node: node.rows[key.row])
spelled(RowKey, 'row', lambda key: key.row)


def test_a_key_kind_registered_is_a_segment_as_written_and_round_trips():
    place = cx.absolute.place(cx.V[RowKey('a')].x)
    assert place.segments == (RowKey('a'), GetAttrKey('x'))      # not wrapped as a dict key
    g = graph_of({NodePath((DictKey('f'),)): Function((place,), (VarPath((DictKey('y'),)),))})
    assert '{"row": "a"}' in json.dumps(written(g))
    assert read_json(written(g)) == g


def test_loading_imports_nothing():
    data = written(solved())
    data['nodes'][0]['definition']['implementation'] = {'ref': 'not_a_module_anyone_imported:f'}
    with pytest.raises(ImportError, match='not imported'):
        read_json(data)


def test_a_slot_under_another_role_is_refused():
    '''A slot's role is written for a reader; it must be the driver's own.'''
    data = written(solved())
    driven = next(e['definition']['node'] for e in data['nodes']
                  if e['definition'].get('node', {}).get('kind') == 'driven')
    driven['data'][0]['role'] = 'Steps'
    with pytest.raises(ValueError, match='data roles disagree'):
        read_json(data)


def test_every_rule_is_asked_of_the_file():
    '''Two owners of one place: refused by the graph, not the reader.'''
    data = written(solved(), implementations=False)
    data['nodes'].append(dict(data['nodes'][-1], name=[{'dict': 'again'}]))
    with pytest.raises(ValueError, match='owned by both'):
        read_json(data)


def test_another_format_is_refused():
    with pytest.raises(ValueError, match='version'):
        read_json({'format': 'cottax.graph', 'version': 0})


def test_a_codec_nobody_registered_is_refused_on_reading():
    data = dict(written(solved()), places='dotted')
    with pytest.raises(ValueError, match="PlaceCodec 'dotted', which is not registered: import"):
        read_json(data)
    data = dict(written(solved()), labels='dotted')
    with pytest.raises(ValueError, match="LabelCodec 'dotted', which is not registered: import"):
        read_json(data)


def test_the_codecs_are_the_ones_registered_for_the_places_and_the_node_names():
    '''Writing finds each codec by its class, reading by the header's tag.'''
    g = solved()
    assert answer_for_places(PlaceCodec, g) == PytreeCodec()
    assert answer_for_labels(LabelCodec, g) == PytreeLabelCodec()
    assert (written(g)['places'], written(g)['labels']) == ('pytree', 'pytree')


class Word(str):
    '''A place no codec is registered for: its class names it.'''

    @property
    def segments(self):
        return (str(self),)


def test_a_place_class_with_no_codec_is_refused_by_name():
    g = graph_of({NodePath((DictKey('f'),)): Function((Word('x'),), (Word('y'),))})
    (refused,) = violations(write_json, g)
    assert isinstance(refused, Unanswered) and 'test_export.Word' in refused.message
    assert 'no PlaceCodec' in refused.message
    assert 'import the module that registers one' in refused.message
    assert [type(v) for v in violations(write_json, graph_of({Word('f'): Function(
        (Word('x'),), (Word('y'),))}))] == [Unanswered, Unanswered]       # the node names too


class Term(Word):
    '''A place with a codec registered: `TermCodec`.'''


@answers(Term)
class TermCodec(PlaceCodec):
    '''Writes a term without a space, and refuses one with.'''
    tag = 'test-terms'
    def write_place(self, place, w):
        if ' ' in place:
            raise ValueError(f'{place!r} has a space')
        return str(place)
    def read_place(self, data, r):   return Term(data)


def test_a_new_place_class_names_its_nodes_by_a_label_class_already_registered():
    '''A place class brings its own `PlaceCodec`; the node names keep `NodePath`'s.'''
    g    = graph_of({NodePath((DictKey('f'),)): Function((Term('x'),), (Term('y'),))})
    data = write_json(g, implementations=False)
    assert (data['places'], data['labels']) == ('test-terms', 'pytree')
    assert read_json(data) == g


def test_a_registered_codec_writes_and_reads_its_places_and_what_it_refuses_is_unexportable():
    f      = NodePath((DictKey('f'),))
    spaced = graph_of({f: Function((Term('x y'),), (Term('y'),))})
    assert [type(v) for v in violations(write_json, spaced)] == [Unexportable]


def test_a_place_the_codec_found_cannot_write_is_refused_by_name():
    '''A graph is over one place class: the codec is found for one place, and a place of
    another class it cannot write is refused where it is met.'''
    g = graph_of({NodePath((DictKey('f'),)): Function((V('x'),), (Term('y'),))})
    (refused,) = violations(write_json, g)
    assert isinstance(refused, Unexportable) and refused.what == repr(V('x'))


@dataclasses.dataclass(frozen=True)
class Lookalike:
    '''Not a `VarPath` or a `NodePath`, with the segments of one.'''
    segments : tuple


def test_a_lookalike_of_a_pytree_place_or_node_name_is_refused_by_name():
    '''Written by the pytree codec, it would be read back as a `VarPath` or `NodePath`.'''
    a = Lookalike((DictKey('a'),))
    (refused,) = violations(write_json, graph_of({NodePath((DictKey('f'),)): Function(
        (V('x'), a), (V('a'),))}))
    assert isinstance(refused, Unexportable) and refused.what == repr(a)
    assert 'test_export.Lookalike, not a VarPath' in refused.message
    (refused,) = violations(write_json, graph_of({NodePath((DictKey('f'),)): Function(
        (V('x'),), (V('y'),)), a: Function((V('y'),), (V('z'),))}))
    assert refused.what == repr(a) and 'not a NodePath' in refused.message


def test_an_empty_graph_names_no_codec():
    assert [type(v) for v in violations(write_json, graph_of({}))] == [NothingToAsk]


def test_a_tag_coded_twice_is_refused():
    '''A second codec for a tag would be chosen by import order; a label codec's tags are
    its own.'''
    class Again(PlaceCodec):
        tag = 'pytree'
        def write_place(self, place, w): ...
        def read_place(self, data, r):   ...

    with pytest.raises(TypeError, match="'pytree' is written by the codec .*PytreeCodec"):
        answers(Word)(Again)


# ---------------------------------------------------------------- extending the format
# A registration lasts for the session, so each test registers for finals of its own.

class Local(FunctionalNode):
    """A function: the level of this module's own finals, never instantiated."""

    reads : tuple
    owns  : tuple

    def relabel(self, renaming):
        return type(self)(relabelled(self.reads, renaming), relabelled(self.owns, renaming))


class Unspelled(Local):
    """A final nobody registers a spelling for."""


class BrokenNode(Local):
    """A final spelled by `Broken`."""


class Stepped(FunctionalNode):
    """A kind the shipped spellings do not know: a function with a time step."""

    reads : tuple
    owns  : tuple
    dt    : float

    def relabel(self, renaming):
        return Stepped(relabelled(self.reads, renaming), relabelled(self.owns, renaming), self.dt)


class SteppedSpelling(Spelling[Stepped]):
    """Registered by the test that needs it, not here."""

    kind = 'stepped'

    def write(self, node, w):
        return {'reads': w.places(node.reads), 'owns': w.places(node.owns), 'dt': node.dt}

    def read(self, data, r):
        return Stepped(r.places(data['reads']), r.places(data['owns']), data['dt'])


def one(node):
    return graph_of({NodePath((DictKey('step'),)): node})


X, Y = VarPath((DictKey('x'),)), VarPath((DictKey('y'),))


def stepped():
    return one(Stepped((X,), (Y,), 0.1))


def test_a_kind_with_no_spelling_is_refused_until_one_is_registered():
    g    = stepped()
    file = {'kind': 'stepped', 'reads': [[{'dict': 'x'}]], 'owns': [[{'dict': 'y'}]], 'dt': 0.1}
    with pytest.raises(TypeError, match='no spelling'):
        write_json(g)
    entry = {'name': [{'dict': 'step'}], 'definition': file}
    with pytest.raises(ValueError, match="'stepped' is no node kind"):
        read_json(dict(written(one(Function((X,), (Y,)))), nodes=[entry]))
    answers(Stepped)(SteppedSpelling)
    data = written(g)
    assert data['nodes'][0]['definition'] == file
    assert read_json(data) == g


def test_a_final_with_no_spelling_is_refused_by_name_and_class():
    g = one(Unspelled((X,), (Y,)))
    (refused,) = violations(write_json, g)
    assert isinstance(refused, Unexportable) and refused.node == NodePath((DictKey('step'),))
    assert refused.what == 'test_export.Unspelled'
    assert 'import the module that registers one' in refused.message


def test_a_kind_nobody_registered_is_refused_on_reading():
    data = written(one(Function((X,), (Y,))))
    data['nodes'][0]['definition']['kind'] = 'unspelled'
    with pytest.raises(ValueError) as e:
        read_json(data)
    assert str(e.value) == UnknownKind(NodePath((DictKey('step'),)), 'unspelled').message
    assert 'import the module that registers it' in str(e.value)


def test_a_kind_spelled_twice_is_refused():
    class SameWord(Spelling[Unspelled]):
        kind = 'function'
        def write(self, node, w): return {}
        def read(self, data, r): raise NotImplementedError

    with pytest.raises(TypeError) as e:
        answers(Unspelled)(SameWord)
    assert str(e.value) == SpelledTwice("'function'", 'cottax.consumers.json.format.FunctionSpelling',
                                        f'test_export.{SameWord.__qualname__}').message
    # Nothing of a refused registration is kept.
    assert written(one(Function((X,), (Y,))))['nodes'][0]['definition']['kind'] == 'function'
    assert [type(v) for v in violations(write_json, one(Unspelled((X,), (Y,))))] == [Unexportable]


class Approx(Rel):
    """A relation the shipped spellings do not know."""
    symbol = '~'


def test_a_new_relation_is_one_line_in_rels(monkeypatch):
    u, r = VarPath((DictKey('u'),)), VarPath((DictKey('r'),))
    g = graph_of({NodePath((DictKey('body'),)): Function((u,), (r,)),
                  NodePath((DictKey('near'),)): RelationalCondition((Relation(r, None, Approx),), (), (u,), ())})
    with pytest.raises(TypeError, match='no spelling in `RELS`'):
        write_json(g)
    monkeypatch.setitem(RELS, 'approx', Approx)
    data = written(g)
    assert [x['op'] for x in data['nodes'][1]['definition']['relations']] == ['approx']
    assert read_json(data) == g


def test_the_file_words_of_relations_and_senses_are_pinned():
    u, v, f, g = (VarPath((DictKey(k),)) for k in 'uvfg')
    graph = graph_of({NodePath((DictKey('body'),)): Function((u, v), (f, g)),
                      NodePath((DictKey('down'),)): RelationalCondition(
                          (Relation(g, None, Le), Relation(g, u, Eq)), (Objective(f, Min),), (u,), ()),
                      NodePath((DictKey('up'),)): RelationalCondition(
                          (), (Objective(f, Max),), (v,), ())})
    data = written(graph)
    down, up = (data['nodes'][i]['definition'] for i in (1, 2))
    assert [x['op'] for x in down['relations']] == ['le', 'eq']
    assert [o['sense'] for o in down['objectives'] + up['objectives']] == ['min', 'max']
    assert read_json(data) == graph


def test_a_word_no_table_holds_is_refused_naming_the_table():
    u, v, f, g = (VarPath((DictKey(k),)) for k in 'uvfg')
    graph = graph_of({NodePath((DictKey('body'),)): Function((u, v), (f, g)),
                      NodePath((DictKey('down'),)): RelationalCondition(
                          (Relation(g, u, Eq),), (Objective(f, Min),), (u,), ())})
    for key, field, table in (('relations', 'op', 'RELS'), ('objectives', 'sense', 'SENSES')):
        data = written(graph)
        data['nodes'][1]['definition'][key][0][field] = 'nope'
        with pytest.raises(ValueError, match=f"'nope' is no entry of `{table}`"):
            read_json(data)


def test_an_implementation_around_a_condition_is_refused_as_such():
    data = written(solved(), implementations=False)
    entry = next(e for e in data['nodes'] if e['definition']['kind'] == 'driven')
    entry['definition'] = {'kind': 'implemented', 'node': entry['definition']['condition'],
                           'implementation': {'ref': 'test_cx:square'}}
    with pytest.raises(TypeError, match='nothing implements a RelationalCondition'):
        read_json(data)


@answers(BrokenNode)
class Broken(Spelling[BrokenNode]):
    kind = 'broken'

    def write(self, node, w):
        return {'dt': node.no_such_field}

    def read(self, data, r):
        raise NotImplementedError


def test_a_bug_in_a_spelling_raises_rather_than_being_refused():
    with pytest.raises(AttributeError, match='no_such_field'):
        write_json(one(BrokenNode((X,), (Y,))))


def test_a_place_the_codec_refuses_is_refused_by_name():
    x, y = VarPath((DictKey((1, 2)),)), VarPath((DictKey('y'),))
    g = graph_of({NodePath((DictKey('f'),)): Function((x,), (y,))})
    (refused,) = violations(write_json, g)
    assert isinstance(refused, Unexportable) and 'codec refuses' in refused.reason
