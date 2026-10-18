'''
A declaration interface for a surface for types of conditional nodes.
'''

from collections.abc import Callable, Mapping
from typing import Any, Self
import dataclasses

from ...core import Place, PlaceToPlace
from ...core.frozen import choice, name_of
from ..surface.interface import Part
from .conventions import Conventions, Shipped
from .declared import Declared, declare
from .interface_types import Body, Explicit, PlaceWithValue, declares, parameters_of


@choice(Part)
class Declaring(Part):
    '''
    Part declaring multiple nodes (residuals / implicit models / rates)

    Includes conventions on what means what here (i.e. does it read all function parameter
    strictly?)

    class Mine(ShippedSurface, MyDeclaring), a declaring part of a study's own beside the
    shipped ones; `surface.reading(conventions)` reads its arguments otherwise.
    '''

    conventions : Conventions = Shipped()
    '''What each argument that may be a value means: cottax's unless a study says otherwise.'''

    function = declares(
        Explicit,
        '''A function computing its outputs, returned in that order: `function(fn, *outputs)`
        where it is used, `@function(*outputs)` where it is written. Each parameter is read
        where `reads` says, else by name under the surface `read_in` names it in, else by name
        in this one.''')

    def declare(self, *models: Any) -> Declared:
        '''Walk a model tree yielding a `Declared`: the graph plus already given values and tags.'''
        return declare(*models)

    def reading(self, conventions: Conventions) -> Self:
        '''The same surface, arguments read as `conventions` says.'''
        return dataclasses.replace(self, conventions=conventions)

    # ------------------------------------------------------------ the conventions, asked

    def parameter_reads(self, fn: Body, reads: Any, read_in: Mapping[Callable[[str], Any], Any],
                        *, given: Callable[[str], Any]) -> tuple[PlaceWithValue, ...]:
        '''The place each parameter of `fn` reads, as `conventions.reads` says.'''
        return self.conventions.reads(parameters_of(fn), reads, read_in, place=self.place,
                                      given=given, of=name_of(fn))

    def term(self, x: Any, written: Any, value_at: PlaceToPlace) -> PlaceWithValue | None:
        '''What is written where a place about `x` is due, as `conventions.term` says: `None`
        for zero, a value at `value_at(x)`, or a place.'''
        return self.conventions.term(x, written, place=self.place, value_at=value_at)

    def per_place(self, among: tuple[Any, ...], written: Any) -> dict[Place, Any]:
        '''What is written for each of `among`, as `conventions.per_place` says.'''
        return self.conventions.per_place(among, written, place=self.place)


__all__ = ['Declaring']
