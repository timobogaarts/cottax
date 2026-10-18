'''
What a run takes and where a caller keeps it: `slots`, each boundary input with the key it is
given under and its place in a caller's structure; `read_from` reads them there, `valued_once`
holds the values found to one per input; `write_back` writes what a run computed where a
structure holds it. Exact places alone: what a name means, and where a datum's unknown sits in
a structure (`at`), is the caller's.
'''

from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

from ...core import NestingGraph, Place, answer_for_places
from ...core.driven import PerUnknown, is_driven, role_slots
from ...core.frozen import OrderedMap
from .interface import DataStructure
from .refusals import NoneTakes, NoSlotFor, Unvalued, ValuedTwice


class Inputs:
    '''The key for a variable's own value, by name or in a structure, beside a driver's roles:
    `s.run(model, {Inputs: {'x': 1.0}}, read_from={Inputs: state})`. Not a role.'''


class Slot(NamedTuple):
    '''A place a run takes a value at, under `key`, for `unknown` of `problem`, at `at` in a
    structure: an input at itself; a per-problem datum for no unknown, at none.'''
    place   : Any
    key     : type
    problem : Any = None
    unknown : Any = None
    at      : Any = None


def slots[V: Place](graph: NestingGraph[Any, V], at: Callable[[V], V]) -> tuple[Slot, ...]:
    '''Every place a run of `graph` takes a value at, by what the caller calls it: each driver
    datum, also one it computes (a warm start), then each boundary input. A place two problems
    take is there twice.'''
    roles = dict.fromkeys(r for n in graph.problems if is_driven(graph[n])
                          for r in graph[n].contract.takes)
    return (*(Slot(place, role, problem, unknown, None if unknown is None else at(unknown))
              for role in roles for place, problem, unknown in role_slots(graph, role)),
            *(Slot(p, Inputs, at=p) for p in graph.graph.boundary_inputs))


def read_from[V: Place](graph: NestingGraph[Any, V], table: tuple[Slot, ...],
                        structures: Mapping[type, Any]) -> tuple[tuple[Any, Any, str], ...]:
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


def valued_once[V: Place](table: tuple[Slot, ...], inputs: tuple[V, ...],
                          candidates: Iterable[tuple[Any, Any, str]]) -> OrderedMap[V, Any]:
    '''The one value at each input: refused, a place proposed twice (`ValuedTwice`, naming both
    sources) and an input none is proposed for (`Unvalued`, saying how to give it).'''
    given, sources = {}, {}
    for place, value, source in candidates:
        if place in sources:
            raise ValuedTwice(place, (sources[place], source)).raised
        given[place], sources[place] = value, source
    if missing := [next(s for s in table if s.place == p) for p in inputs if p not in given]:
        raise Unvalued('; '.join(              # each as a caller gives it
            f'`values[Inputs]` at {s.place!r}' if s.key is Inputs else f'`values[{s.key.__name__}]` '
            f'at {s.unknown if s.unknown is not None else f"any unknown of {s.problem!r}"!r} ({s.place!r})'
            for s in missing)).raised
    return OrderedMap((p, given[p]) for p in inputs)


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


__all__ = ['Inputs', 'Slot', 'slots', 'read_from', 'valued_once', 'write_back', 'write_role']
