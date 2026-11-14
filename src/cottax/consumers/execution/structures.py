'''
What a run takes and where a caller keeps it: `slots`, each boundary input with the key it is
given under and its place in a caller's structure; `read_from` reads them there, `valued_once`
holds the values found to one per input, `Assessed` keeps them for asking; `write_back` writes what a run computed where a
structure holds it. Exact places alone: what a name means, and where a datum's unknown sits in
a structure (`at`), is the caller's.
'''

from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any, NamedTuple

from ...core import GraphWithValues, Label, NestingGraph, Place, ProvedGraph, answer_for_places, comparable
from ...core.driven import PerUnknown, is_driven, role_slots
from ...core.frozen import Frozen, OrderedMap
from .interface import DataStructure
from .refusals import NoneTakes, NoSlotFor, Unvalued, ValuedTwice


class Inputs:
    '''The key for a variable's own value, by name or in a structure, beside a driver's roles:
    `s.run(model, {Inputs: {'x': 1.0}}, read_from={Inputs: state})`. Not a role.'''


class Slot[V: Place, N: Label](NamedTuple):
    '''A place a run takes a value at, under `key`, for `unknown` of `problem`, at `at` in a
    structure: an input at itself; a per-problem datum for no unknown, at none.'''
    place   : V
    key     : type
    problem : N | None = None
    unknown : V | None = None
    at      : V | None = None

    @property
    def how(self) -> str:
        '''How a caller gives it, by name.'''
        if self.key is Inputs:
            return f'`values[Inputs]` at {self.place!r}'
        return (f'`values[{self.key.__name__}]` at '
                f'{self.unknown if self.unknown is not None else f"any unknown of {self.problem!r}"!r} '
                f'({self.place!r})')


def slots[N: Label, V: Place](graph: NestingGraph[N, V], at: Callable[[V], V]) -> tuple[Slot[V, N], ...]:
    '''Every place a run of `graph` takes a value at, by what the caller calls it: each driver
    datum, also one it computes (a warm start), then each boundary input. A place two problems
    take is there twice.'''
    roles = dict.fromkeys(r for n in graph.enclosers if is_driven(graph[n])
                          for r in graph[n].contract.takes)
    return (*(Slot(place, role, problem, unknown, None if unknown is None else at(unknown))
              for role in roles for place, problem, unknown in role_slots(graph, role)),
            *(Slot(p, Inputs, at=p) for p in graph.graph.boundary_inputs))


def read_from[N: Label, V: Place](graph: NestingGraph[N, V], table: tuple[Slot[V, N], ...],
                                  structures: Mapping[type, Any]) -> tuple[tuple[V, Any, str], ...]:
    '''
    Each value a structure of `structures` gives an open slot of `table`, where its key reads
    it: under `Inputs`, an input where the structure holds it; under a per-unknown role, the
    datum at its unknown's place. What a structure holds beyond is never asked; a slot holding
    `None` gives nothing. Refused: a per-problem role, a role no driver takes.
    '''
    structure = answer_for_places(DataStructure, graph)
    open_ = set(graph.graph.boundary_inputs)
    found = []
    for key, state in structures.items():
        if key is not Inputs and not issubclass(key, PerUnknown):
            raise NoSlotFor(key.__name__).raised
        if not (taken := [s for s in table if s.key is key]) and key is not Inputs:
            raise NoneTakes(key.__name__).raised
        for s in taken:
            if s.place in open_ and structure.holds(state, s.at) and \
                    (v := structure.read(state, (s.at,))[s.at]) is not None:
                found.append((s.place, v, f'read_from[{key.__name__}] at {s.at!r}'
                              + ('' if s.unknown in (None, s.at) else f', for {s.unknown!r}')))
    return tuple(found)


def unasked[N: Label, V: Place](graph: NestingGraph[N, V], table: tuple[Slot[V, N], ...],
                                structures: Mapping[type, Any]) -> tuple[V, ...]:
    '''
    What `read_from` never asks of `structures`: each place one holds a value at that no open
    slot of its key reads, at it, inside it or holding it. A place one key reads and another
    does not, the same structure given under both, is asked.
    '''
    structure = answer_for_places(DataStructure, graph)
    open_ = set(graph.graph.boundary_inputs)
    held, asked = {}, set()
    for key, state in structures.items():
        ats = [s.at for s in table if s.key is key and s.place in open_]
        for place in structure.places(state):
            held[place] = None
            if any(comparable(place, at) for at in ats):
                asked.add(place)
    return tuple(p for p in held if p not in asked)


def unvalued[N: Label, V: Place](table: tuple[Slot[V, N], ...], inputs: tuple[V, ...],
                                 given: Collection[V]) -> tuple[Slot[V, N], ...]:
    '''The slot to give each input not in `given`, once however many rows it has: its first.'''
    return tuple(next(s for s in table if s.place == p) for p in inputs if p not in given)


def valued_once[N: Label, V: Place](table: tuple[Slot[V, N], ...], inputs: tuple[V, ...],
                                    candidates: Iterable[tuple[V, Any, str]]) -> OrderedMap[V, Any]:
    '''The one value at each input: refused, a place proposed twice (`ValuedTwice`, naming both
    sources) and an input none is proposed for (`Unvalued`, saying how to give it).'''
    given, sources = {}, {}
    for place, value, source in candidates:
        if place in sources:
            raise ValuedTwice(place, (sources[place], source)).raised
        given[place], sources[place] = value, source
    if missing := unvalued(table, inputs, given):
        raise Unvalued('; '.join(s.how for s in missing)).raised
    return OrderedMap((p, given[p]) for p in inputs)


class Assessed[N: Label, V: Place](Frozen):
    '''
    A model's values looked up and not yet held to one per input: what `valued` refuses is
    asked here (`missing`, `twice`), or refused by asking for them (`valued()`); and what the
    structures read from hold that no run takes (`unread`, `overwritten`), which is not refused.
    '''

    graph      : NestingGraph[N, V] | ProvedGraph
    proof      : type
    table      : tuple[Slot[V, N], ...]
    candidates : tuple[tuple[V, Any, str], ...]
    structures : Mapping[type, Any]

    @property
    def nesting(self) -> NestingGraph[N, V]:
        return self.graph.graph if isinstance(self.graph, ProvedGraph) else self.graph

    @property
    def _inputs(self) -> tuple[V, ...]:
        return self.nesting.graph.boundary_inputs

    @property
    def _proposed(self) -> dict[V, tuple[str, ...]]:
        '''Who proposed each place, in the order proposed.'''
        proposed: dict[V, tuple[str, ...]] = {}
        for place, _, source in self.candidates:
            proposed[place] = (*proposed.get(place, ()), source)
        return proposed

    @property
    def missing(self) -> tuple[Slot[V, N], ...]:
        '''The slot to give each input that has no value, `.how` saying how.'''
        return unvalued(self.table, self._inputs, self._proposed)

    @property
    def twice(self) -> OrderedMap[V, tuple[str, ...]]:
        '''Each place proposed more than once, with every source proposing it.'''
        return OrderedMap((p, s) for p, s in self._proposed.items() if len(s) > 1)

    @property
    def sources(self) -> OrderedMap[V, str]:
        '''Where each place's value comes from; one given twice, the first.'''
        return OrderedMap((p, s[0]) for p, s in self._proposed.items())

    def _computed(self, place: V) -> bool:
        return any(comparable(place, o) for o in self.nesting.graph.owned_variables)

    @property
    def unread(self) -> tuple[V, ...]:
        '''What the structures hold that nothing reads and the graph does not compute.'''
        return tuple(p for p in unasked(self.nesting, self.table, self.structures) if not self._computed(p))

    @property
    def overwritten(self) -> tuple[V, ...]:
        '''What the structures hold that nothing reads and the graph computes: given, then discarded.'''
        return tuple(p for p in unasked(self.nesting, self.table, self.structures) if self._computed(p))

    def valued(self) -> GraphWithValues:
        '''The graph, proved, with its one value per input; refused as `Running.valued` says.'''
        found = valued_once(self.table, self._inputs, self.candidates)
        return GraphWithValues(self.graph if isinstance(self.graph, self.proof) else self.proof(self.nesting), found)


def write_back[V: Place](graph: NestingGraph[Any, V], state: Any,
                         values: Mapping[V, Any]) -> Any:
    '''`state` with each place `graph` computes written where the state holds it, its value
    from `values` (a run's); an input is never written.'''
    structure = answer_for_places(DataStructure, graph)
    return structure.write(state, OrderedMap(
        (p, values[p]) for p in graph.graph.owned_variables
        if p in values and structure.holds(state, p)))


def write_role[V: Place](graph: NestingGraph[Any, V], state: Any, values: Mapping[V, Any],
                         role: type, at: Callable[[V], V]) -> Any:
    '''`state` with the value of each place a driver takes or reports a per-unknown `role` at,
    written at `at(unknown)` where the state holds it: a structure keyed as the unknowns are.
    Refused: a per-problem role, one no driver has.'''
    if not issubclass(role, PerUnknown):
        raise NoSlotFor(role.__name__).raised
    if not (found := role_slots(graph, role)):
        raise NoneTakes(role.__name__).raised
    structure = answer_for_places(DataStructure, graph)
    return structure.write(state, OrderedMap(
        (at(u), values[p]) for p, _, u in found if p in values and structure.holds(state, at(u))))


__all__ = ['Inputs', 'Slot', 'slots', 'read_from', 'unasked', 'unvalued', 'valued_once', 'Assessed', 'write_back', 'write_role']
