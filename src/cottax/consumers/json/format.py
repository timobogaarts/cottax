'''A graph as JSON: `write_json` and `read_json`.'''

import dataclasses
import sys
from collections.abc import Callable, Iterator
from typing import Any

from ...core import (
    AbstractGraph, Function, Label, NestingGraph, NodeDefinition, Place, Violation,
    answer_for_labels, answer_for_places,
)
from ...core.frozen import Frozen, OrderedMap, answer, answered, answers, checked, dotted, violations
from ...core.driven import Driven
from ...core.realisation import (
    Implemented, ImplementedDriven, ImplementedFunction, NotImplementable, implemented_as,
)


from .interface import LabelCodec, PlaceCodec, Spelling
from .interface_types import Json, Reader, Writer
from .refusals import (
    Inconsistent, NotAGraph, NotLoaded, Unexportable, UnknownCodec, UnknownKind,
)


FORMAT, VERSION = 'cottax.graph', 1

def loaded(name: str) -> object:
    '''What `module:qualname` names, among the modules already imported: a file imports
    nothing unless a `resolve` handed in does.'''
    module, _, qualname = name.partition(':')
    if module not in sys.modules:
        raise NotLoaded(name, module).raised
    return _lookup(module, qualname)


def by_word[A: Spelling | PlaceCodec | LabelCodec](interface: type[A], word: str) -> dict[str, A]:
    '''Each registered `interface` answer by its `word`: what a file names it by.'''
    return {getattr(a, word): a for a in answered().get(interface, {}).values()}


# ---------------------------------------------------------------- reading and writing

def read_json(data: dict, resolve: Callable[[str], object] = loaded) -> NestingGraph:
    '''
    The graph `data` spells, every rule asked of it: places and node names by the codecs
    registered under the file's tags, each node by the spelling registered for its kind; a
    name resolved by `resolve`.
    '''
    if (data.get('format'), data.get('version')) != (FORMAT, VERSION):
        raise NotAGraph(data.get('format'), data.get('version'), FORMAT, VERSION).raised
    codecs = []
    for interface, header in ((PlaceCodec, 'places'), (LabelCodec, 'labels')):
        if (codec := by_word(interface, 'tag').get(data.get(header))) is None:
            raise UnknownCodec(interface.__qualname__, data.get(header)).raised
        codecs.append(codec)
    places, labels = codecs
    names = NodeReader(places, labels, resolve, None)
    definitions = []
    for entry in data['nodes']:
        r    = NodeReader(places, labels, resolve, labels.read_label(entry['name'], names))
        definitions.append((r.name, r.node(entry['definition'])))
    within = ((names.label(i), names.label(o)) for i, o in data['within'])
    return NestingGraph(AbstractGraph(OrderedMap(definitions)), OrderedMap(within))


def check_codec[N: Label, V: Place](graph: NestingGraph[N, V], implementations: bool) -> Iterator[Violation]:
    '''One registered codec writes every node name, and one every place. A graph with no
    node has no place either, so it is refused once.'''
    yield from violations(answer_for_labels, LabelCodec, graph)
    if graph.nodes:
        yield from violations(answer_for_places, PlaceCodec, graph)


def check_nodes_write[N: Label, V: Place](graph: NestingGraph[N, V], implementations: bool) -> Iterator[Violation]:
    '''Each node writes: the check is the writing, so the two cannot disagree.'''
    if any(True for _ in check_codec(graph, implementations)):
        return
    places, labels = answer_for_places(PlaceCodec, graph), answer_for_labels(LabelCodec, graph)
    for name in graph.nodes:
        try:
            _json_for_node(graph, name, places, labels, implementations)
        except _Refused as refused:
            yield refused.violation


def _json_for_node[N: Label, V: Place](graph: NestingGraph[N, V], name: N, places: PlaceCodec[V],
                                       labels: LabelCodec[N], implementations: bool) -> Json:
    '''The binding at `name`; raises `_Refused` where it does not write.'''
    w = NodeWriter(places, labels, implementations, name)
    return {'name': w.label(name), 'definition': w.node(graph[name])}


@checked(check_codec, check_nodes_write)
def write_json[N: Label, V: Place](graph: NestingGraph[N, V], implementations: bool = True) -> dict:
    '''A graph written as JSON, ready for `json.dump`: each binding its name and its
    definition, places by the registered codecs. With `implementations=False`, the structure
    alone. Each node is written twice: once by the check, once here.'''
    places, labels = answer_for_places(PlaceCodec, graph), answer_for_labels(LabelCodec, graph)
    label = NodeWriter(places, labels, implementations, None).label
    return {'format': FORMAT, 'version': VERSION,
            'places': places.tag, 'labels': labels.tag,
            'nodes': [_json_for_node(graph, n, places, labels, implementations) for n in graph.nodes],
            'within': [[label(i), label(o)] for i, o in graph.within.items()]}


# ---------------------------------------------------------------- what a spelling writes and reads through

class _Refused(Exception):
    '''Carries a violation out of the writer, to the check that yields it.'''

    def __init__(self, violation: Violation) -> None:
        self.violation = violation


class NodeWriter[N: Label, V: Place](Frozen):
    '''A `Writer`: the node bound at `name`, through the codecs.'''

    place_codec     : PlaceCodec[V]
    label_codec     : LabelCodec[N]
    implementations : bool
    name            : N

    def places(self, places: tuple[V, ...]) -> list:    return [self.place(p) for p in places]

    def place(self, place: V) -> Json:
        '''Through the codec; what it refuses is refused, anything else in a spelling raises.'''
        try:
            return self.place_codec.write_place(place, self)
        except Exception as e:
            raise self.refuse(repr(place), f'the codec refuses it ({type(e).__name__}: {e})') from e

    def label(self, name: N) -> Json:
        try:
            return self.label_codec.write_label(name, self)
        except Exception as e:
            raise self.refuse(repr(name), f'the codec refuses it ({type(e).__name__}: {e})') from e

    def refuse(self, what: str, reason: str) -> _Refused:
        return _Refused(Unexportable(self.name, what, reason))

    def node(self, node: NodeDefinition[V]) -> dict:
        '''`node` by its final's spelling; without implementations, a wrapped one as its node.'''
        if isinstance(node, Implemented) and not self.implementations:
            return self.node(node.node)
        if (spelling := answer(Spelling, node, None)) is None:
            raise self.refuse(dotted(type(node)),
                              'no spelling is registered for its class: import the module that '
                              'registers one (a feature registers its own when its `consumers` '
                              'is imported)')
        return {'kind': spelling.kind, **spelling.write(node, self)}

    def implementation(self, obj: object) -> dict:
        '''By name, or a dataclass as a value.'''
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return self.value(obj)
        if (ref := reference(obj)) is None:
            raise self.refuse(repr(obj), 'neither reachable by a module-level name nor a dataclass '
                              'of scalars; pickle it, or export without implementations')
        return {'ref': ref}

    def value(self, obj: object) -> dict:
        '''A dataclass as its class and fields, each a JSON scalar or such a value again.'''
        if (cls := reference(type(obj))) is None:
            raise self.refuse(type(obj).__qualname__, 'a class with no module-level name')
        def field(v: object) -> Json:
            if v is None or isinstance(v, (bool, int, float, str)):
                return v
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return self.value(v)
            raise self.refuse(f'{type(obj).__qualname__} holding {type(v).__qualname__}',
                              'a field that is neither a JSON scalar nor a dataclass of them')
        return {'class': cls, 'fields': {f.name: field(getattr(obj, f.name))
                                         for f in dataclasses.fields(obj) if f.init}}


class NodeReader[N: Label, V: Place](Frozen):
    '''A `Reader`: the node bound at `name`, through the place codec.'''

    place_codec : PlaceCodec[V]
    label_codec : LabelCodec[N]
    resolve     : Callable[[str], object]
    name        : N

    def place(self, data: Json) -> V:                   return self.place_codec.read_place(data, self)
    def label(self, data: Json) -> N:                   return self.label_codec.read_label(data, self)
    def places(self, data: list) -> tuple[V, ...]:      return tuple(self.place(p) for p in data)

    def node(self, data: dict) -> NodeDefinition[V]:
        if (spelling := by_word(Spelling, 'kind').get(data['kind'])) is None:
            raise UnknownKind(self.name, data['kind']).raised
        return spelling.read(data, self)

    def implementation(self, data: dict) -> object:
        return self.resolve(data['ref']) if 'ref' in data else self.value(data)

    def value(self, data: dict) -> Any:
        cls = self.resolve(data['class'])
        return cls(**{k: self.value(v) if isinstance(v, dict) else v for k, v in data['fields'].items()})


# ---------------------------------------------------------------- names

def reference(obj: object) -> str | None:
    '''`module:qualname` if that finds `obj` again, else `None`: a lambda, a local, anything
    in `__main__` or bound under another name.'''
    module, qualname = getattr(obj, '__module__', None), getattr(obj, '__qualname__', None)
    if not isinstance(module, str) or not isinstance(qualname, str) or module == '__main__':
        return None
    try:
        found = _lookup(module, qualname)
    except (KeyError, AttributeError):
        return None
    return f'{module}:{qualname}' if found is obj else None


def _lookup(module: str, qualname: str) -> object:
    found: object = sys.modules[module]
    for part in qualname.split('.'):
        found = getattr(found, part)
    return found


# ---------------------------------------------------------------- core's own spellings

@answers(Function)
class FunctionSpelling(Spelling[Function]):
    kind = 'function'

    def write[V: Place](self, node: Function[V], w: Writer[Any, V]) -> dict:
        return {'reads': w.places(node.reads), 'owns': w.places(node.owns)}

    def read[V: Place](self, data: dict, r: Reader[Any, V]) -> Function[V]:
        return Function(r.places(data['reads']), r.places(data['owns']))


@answers(Driven)
class DrivenSpelling(Spelling[Driven]):
    '''The condition, the driver as a value, and each slot with its role's name.'''

    kind = 'driven'

    def write[V: Place](self, node: Driven[V], w: Writer[Any, V]) -> dict:
        def slots(roles: tuple[type, ...], slotted: tuple[tuple[V, ...], ...]) -> list:
            return [{'role': role.__qualname__, 'places': w.places(ps)} for role, ps in zip(roles, slotted)]
        return {'condition': w.node(node.condition), 'contract': w.value(node.contract),
                'data': slots(node.contract.takes, node.data),
                'reports': slots(node.contract.reports, node.reports)}

    def read[V: Place](self, data: dict, r: Reader[Any, V]) -> Driven[V]:
        driver = r.value(data['contract'])
        for which, roles in (('data', driver.takes), ('reports', driver.reports)):
            if [s['role'] for s in data[which]] != [role.__qualname__ for role in roles]:
                raise Inconsistent(r.name, which).raised
        return Driven(r.node(data['condition']), driver,
                      tuple(r.places(s['places']) for s in data['data']),
                      tuple(r.places(s['places']) for s in data['reports']))


@answers(ImplementedFunction, ImplementedDriven)
class ImplementedSpelling(Spelling[Implemented]):
    '''The node and what runs it; the final is the node's level's (`implemented_as`).'''

    kind = 'implemented'

    def write[V: Place](self, node: Implemented[V], w: Writer[Any, V]) -> dict:
        return {'node': w.node(node.node), 'implementation': w.implementation(node.implementation)}

    def read[V: Place](self, data: dict, r: Reader[Any, V]) -> Implemented[V]:
        inner = r.node(data['node'])
        if (final := implemented_as(inner)) is None:
            raise NotImplementable(r.name, type(inner).__name__).raised
        return final(inner, r.implementation(data['implementation']))
