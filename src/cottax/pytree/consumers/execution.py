'''Registers a pytree as the `DataStructure` of a `VarPath`, each place at its key path.'''

from collections.abc import Iterable, Mapping
from typing import Any

from ...core.frozen import OrderedMap, answers, dotted
from ...consumers.execution.interface import DataStructure
from ...consumers.execution.refusals import ForeignPlace
from ...core import NestingGraph, ProvedGraph, mint_of
from ..crossings import check_pytree_readable, complete_at, read_at, resolves, write_at
from ..path import VarPath


def own(places: Iterable[object]) -> None:
    '''Refuses a place that is not a `VarPath`: another class's with the same keys would be
    read and written at the `VarPath`'s.'''
    if foreign := tuple(p for p in places if not isinstance(p, VarPath)):
        raise ForeignPlace(foreign, tuple(dotted(type(p)) for p in foreign), 'PytreeStructure',
                           'VarPath').raised


@answers(VarPath)
class PytreeStructure(DataStructure[VarPath]):
    '''A caller's pytree, indexed by places: a place is held where its key path resolves. A
    minted place never does, its root no key of a caller's.'''

    def holds(self, data: Any, place: VarPath) -> bool:
        own((place,))
        return resolves(data, place.segments)

    def read(self, data: Any, places: Iterable[VarPath]) -> OrderedMap[VarPath, Any]:
        places = tuple(places)
        own(places)
        check_pytree_readable(data, [p.segments for p in places])
        return OrderedMap((p, read_at(data, p.segments)) for p in places)

    def write(self, data: Any, values: Mapping[VarPath, Any]) -> Any:
        own(values)
        return write_at(data, {p.segments: v for p, v in values.items()})


def completed(graph: NestingGraph[Any, VarPath] | ProvedGraph[Any, VarPath], tree: Any) -> Any:
    '''
    `tree` with `None` at each place of `graph` it lacks, a minted one aside, so a run has a
    slot for every value it writes back. Only a dict grows a key; a `None` the run must read,
    an input or a start, is still refused there by name.
    '''
    nesting = graph.graph if isinstance(graph, ProvedGraph) else graph
    places = tuple(p for p in nesting.graph.variables if mint_of(p) is None)
    own(places)
    return complete_at(tree, [p.segments for p in places])
