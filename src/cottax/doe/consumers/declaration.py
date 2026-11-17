'''The DOE declaration, `DoeDeclaring`, chosen as a part of a study's surface, and what it
declares, `Sample`.'''

from collections.abc import Iterator, Mapping
from typing import Any

from ...core import Label, NodeDefinition, Place, Violation, minted
from ...core.frozen import OrderedMap, choice, violations
from ...consumers.declaration.interface import Declaring
from ...consumers.declaration.interface_types import (
    Declaration, PlaceWithValue, Tags, given_in, merged_tags,
)
from ...consumers.surface.interface import DerivedNames, Part
from ...consumers.surface.interface_types import Named
from ..doe_condition import DoeCondition, Recorded, Varied
from ..mints import Points, Record


def check_condition(sample: 'Sample') -> Iterator[Violation]:
    '''Its condition is a condition node: refused here, where it is written, not when declared.'''
    yield from violations(DoeCondition, *sample._condition_fields)


class Sample[N: Label, V: Place](Declaration[N, V]):
    '''A design of experiments declared alone, with no body: each factor with where its plan
    is read, each response with where its record is written.'''

    plans   : OrderedMap[V, PlaceWithValue[V]]
    '''Each factor and its plan: named, or at `^points.x`, a value written given there.'''
    records : OrderedMap[V, V]
    '''Each response and its record: named, or at `^record.h`.'''

    checks = (check_condition,)

    holds = ()

    @property
    def outputs(self) -> tuple[V, ...]:
        '''The factors: what it determines.'''
        return tuple(self.plans)

    @property
    def given(self) -> OrderedMap[V, Any]:
        return given_in(self.plans.values())

    @property
    def tags(self) -> Tags[V]:
        return merged_tags((p.place, p.tags) for p in self.plans.values())

    @property
    def _condition_fields(self) -> tuple:
        return (tuple(Varied(x, p.place) for x, p in self.plans.items()),
                tuple(Recorded(h, r) for h, r in self.records.items()))

    @property
    def condition(self) -> DoeCondition[V]:
        return DoeCondition(*self._condition_fields)

    def nodes(self, at: N) -> tuple[tuple[N, NodeDefinition[V]], ...]:
        return ((at, self.condition),)

    def nodes_when_held(self, at: N) -> tuple[tuple[N, NodeDefinition[V]], ...]:
        '''Nothing: nothing holds a DOE.'''
        return ()

    def loose_at(self, names: DerivedNames[N, V]) -> N | None:
        '''None: a DOE goes in a slot.'''
        return None


type Over[P]    = tuple[Named[P], ...] | list[Named[P]] | Mapping[Named[P], Any]
# The factors, either a sequence of names (each plan at `^points.x`), or a mapping of each
# factor to its plan: a name, or a value given at `^points.x`.

type Records[P] = tuple[Named[P], ...] | list[Named[P]] | Mapping[Named[P], Named[P]]
# The responses, either a sequence of names (each record at `^record.h`), or a mapping of each
# response to the name its record is written at.


def plans_of[V](surface: Declaring, over: Over) -> OrderedMap[V, PlaceWithValue[V]]:
    '''`over=`: each factor and where its plan is read, `^points.x` where none is named, a value
    written given there.'''
    if not isinstance(over, Mapping):
        return OrderedMap((x, PlaceWithValue(minted(Points(), x)))
                          for x in surface.places(_names('over', over)))
    def plan(x: V, written: Any) -> PlaceWithValue[V]:
        at = surface.term(x, written, Points())
        return PlaceWithValue(minted(Points(), x), written) if at is None else at
    return OrderedMap((x, plan(x, w)) for x, w in ((surface.place(k), w) for k, w in over.items()))


def records_of[V](surface: Declaring, record: Records) -> OrderedMap[V, V]:
    '''`record=`: each response and where its record is written, `^record.h` where none is
    named.'''
    if not isinstance(record, Mapping):
        return OrderedMap((h, minted(Record(), h)) for h in surface.places(_names('record', record)))
    return OrderedMap((surface.place(h), surface.place(r)) for h, r in record.items())


def _names(keyword: str, written: Any) -> Any:
    if isinstance(written, str):
        raise TypeError(f'{keyword}={written!r}: names, or a mapping -- write one as '
                        f'`({written!r},)`')
    return written


@choice(Part)
class DoeDeclaring(Declaring):
    '''The DOE declaration: a design of experiments over the model, in a slot.'''

    def sample(self, *, over: Over, record: Records) -> Sample:
        '''Each factor of `over` set to every point of its plan, each response of `record`
        recorded there: `{'sweep': sample(over={'z': 'zs'}, record={'f': 'fs'})}`. Each is
        names, or a mapping: a plan named, or a value given at `^points.z`, else read there; a
        record named, else written at `^record.f`.'''
        return Sample(plans_of(self, over), records_of(self, record))
