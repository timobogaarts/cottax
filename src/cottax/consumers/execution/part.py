'''`Driving` and `Running`: driving a model's problems, and running it on values written by
name or read from a caller's structures, as a surface's methods.'''

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from ...core import GraphWithValues, Label, Minting, NestingGraph, PlaceToPlace, ProvedGraph, is_problem
from ...core.driven import AssignByRule, DriverContract, Role, is_driven, role_slots
from ...core.frozen import AbstractVar, OrderedMap, answer
from ..surface.interface import Part
from ..surface.interface_types import RolePlaces, Surface
from .interface_types import Model
from .refusals import NoRoleAt, NotAnInput, NotAnUnknown, NotDriverData, TakesNoValue
from .drive import Drive
from .structures import (Inputs, Slot, read_from as _read_from, slots, valued_once, write_back as _write_back,
                         write_role as _write_role)
from .run import CallableGraph, run as _run


class Driving(Part):
    '''Driving problems: one by its contract (`assign`), or every undriven one by the first of
    the implementations accepting it (`drive`), tried under `contracts`; each role where
    `role_places` puts it.'''

    contracts   : AbstractVar[tuple[DriverContract, ...]]
    '''The contracts a problem is driven under, the first an implementation implements.'''
    role_places : AbstractVar[RolePlaces]
    '''Where a driver's data and reports go, a rule per role.'''

    def assign(self, problem: Label, contract: DriverContract) -> AssignByRule:
        '''The problem driven by `contract`, each of its roles where `role_places` puts it,
        read off the problem when the op is applied.'''
        return AssignByRule(problem, contract, self.role_places)

    def drive(self, *implementations: Any) -> Drive:
        '''Every undriven problem driven by the first of `implementations` accepting it, under
        the first of the contracts it implements; a problem driven already implemented by the
        first covering its contract.'''
        return Drive(implementations, self.contracts, self.role_places)


class Results(Mapping):
    '''Every place's value after a run, read by any name the surface reads: `out['y_1']`,
    `out[V.y_1]`, or the place itself; a driver's datum or report by role at an unknown,
    `out[Trajectory, 'x']`. `out.into(state)` writes them into a structure, `out.into(history,
    Trajectory)` a role's at each unknown's place.'''

    def __init__(self, values: OrderedMap, surface: Surface, graph: NestingGraph) -> None:
        self._values, self._surface, self._graph = values, surface, graph

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], type) \
                and issubclass(key[0], Role):
            return self._values[self._at_role(*key)]
        exact = isinstance(key, self._surface.locator.places)
        return self._values[key if exact else self._surface.place(key, role='a result')]

    def _at_role(self, role: type, written: Any) -> Any:
        '''Where the problem owning the unknown written has `role`: at that unknown, or, per
        problem, at its one place.'''
        unknown = self._surface.place(written, role=f'out[{role.__name__}, ...]')
        owner = self._graph.graph.owners.get(unknown)
        for place, problem, at in role_slots(self._graph, role):
            if problem == owner and at in (unknown, None):
                return place
        raise NoRoleAt(role.__name__, unknown).raised

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f'Results({self._values!r})'

    def into(self, state: Any, key: type = Inputs) -> Any:
        '''`state` with each place the run computed written where the state holds it, an input
        never; under a per-unknown role, that role's value at each unknown's place (at what a
        copy copies), as `read_from` reads it.'''
        if key is Inputs:
            return _write_back(self._graph, state, self._values)
        return _write_role(self._graph, state, self._values, key, self._surface._at(self._graph))


class Running(Part):
    '''
    A model run on values written by name (`values`) or read from a caller's structures
    (`read_from`), each keyed by `Inputs` or a driver's role: `run(model, values, read_from)`,
    through `valued`. A datum of a per-unknown role is read in a structure at its unknown's
    place, or, the unknown a copy by this surface's `copies`, at what it copies.
    '''

    copies : AbstractVar[PlaceToPlace]
    '''Where a cut's copy of a place goes: a copy's datum is read at what it copies.'''

    def valued(self, model: NestingGraph | ProvedGraph | Model, values: Mapping[type, Mapping[Any, Any]] = OrderedMap(),
               read_from: Mapping[type, Any] = OrderedMap(), *,
               proof: type = CallableGraph) -> GraphWithValues:
        '''
        `model` (a graph, a proof, or a `Model`: a `Declared`), proved a `proof` unless it is one, with a
        value at each boundary input, from exactly one source: the declaration's own values;
        `values`, by name, `{Inputs: {'x': 1.}, Start: {'z_1': 5., s.copy('y_2'): 1.}}`, a
        driver's data by role and unknown, a per-problem role at any one unknown of its
        problem; or `read_from`, a structure per key, `{Inputs: state, Start: state}`.
        Refused: a value nothing reads, a place given twice, a key that is no problem's
        unknown, a role its driver does not take, and any input left without a value.
        '''
        declared, graph, nesting = _model(model)
        inputs = nesting.graph.boundary_inputs
        table = slots(nesting, self._at(nesting))
        given = [(p, v, 'the declaration') for p, v in (declared.given.items() if declared is not None else ())
                 if p in inputs]
        by_name = [(self._slot(nesting, table, key, written), value,
                    f'values[{key.__name__}][{written!r}]')
                   for key, written_values in values.items()
                   for written, value in written_values.items()]
        found = valued_once(table, inputs, (*given, *by_name, *_read_from(nesting, table, read_from)))
        return GraphWithValues(graph if isinstance(graph, proof) else proof(nesting), found)

    def _at(self, graph: NestingGraph) -> Callable[[Any], Any]:
        '''Where an unknown's datum sits in a structure: at what it copies, by this surface's
        `copies`, else at the unknown itself.'''
        copied = {self.copies(p): p for p in graph.graph.variables
                  if answer(Minting, p, None) is not None}
        return lambda unknown: copied.get(unknown, unknown)

    def _slot(self, graph: NestingGraph, table: tuple[Slot, ...], key: type, written: Any) -> Any:
        '''The open slot of `table` a name written under `key` gives: an input itself, or the
        datum of the problem owning the unknown written, at that unknown or, per problem, at
        any. Refused with why it gives none.'''
        if key is Inputs:
            place = self.place(written, role='an input')
            if place not in graph.graph.boundary_inputs:
                raise NotAnInput((place,)).raised
            return place
        unknown = self.place(written, role=f'`values[{key.__name__}]`')
        at = graph.graph.owners.get(unknown)
        node = graph[at] if at is not None else None
        if node is None or not is_problem(node) or unknown not in node.unknowns:
            why = ('which nothing owns' if node is None else
                   f'which {at!r} computes and does not determine')
            raise NotAnUnknown(key.__name__, unknown, why).raised
        found = [t for t in table if t.key is key and t.problem == at and t.unknown in (unknown, None)]
        if not found:
            why = ('which no driver answers' if not is_driven(node) else
                   f'whose driver takes {", ".join(r.__name__ for r in node.contract.takes)}')
            raise NotDriverData(at, key.__name__, why).raised
        if found[0].place not in graph.graph.boundary_inputs:
            raise TakesNoValue(at, key.__name__, found[0].place).raised
        return found[0].place

    def run(self, model: NestingGraph | ProvedGraph | Model, values: Mapping[type, Mapping[Any, Any]] = OrderedMap(),
            read_from: Mapping[type, Any] = OrderedMap()) -> Results:
        '''`model` run, valued as `valued` says: every place's value, read by name.'''
        return Results(_run(self.valued(model, values, read_from)), self, _model(model)[2])


def _model(model: NestingGraph | ProvedGraph | Model
           ) -> tuple[Model | None, NestingGraph | ProvedGraph, NestingGraph]:
    '''A graph, a proof or a `Model`: the model if it is one, the graph or proof, and the
    nesting graph under it.'''
    declared = None if isinstance(model, (NestingGraph, ProvedGraph)) else model
    graph = model.graph if declared is not None else model
    return declared, graph, graph.graph if isinstance(graph, ProvedGraph) else graph
