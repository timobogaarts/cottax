'''
A surface converts whatever is input to cottax native names.

Of a place:

1. DerivedNames: how a class of names gives the other class, and a name one deeper.

Of a type that wants to be a place or a node (`Written`):

1. How it is converted from what it can be given from the surface (itself + the naming rules)
'''

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from ...core import (
    Combine, Cut, Delete, Determine, Label, Nest, Place, PlaceToPlace, Rename, Unnest,
)
from ...core.frozen import AbstractClassVar, AbstractVar, Answer, Choice, Frozen, OrderedMap
from ...core.realisation import Implement
from .interface_types import Step, Surface


# ============================================================================ registered

class DerivedNames[N: Label, V: Place](Answer):
    '''
    How a class of names gives the other class, and a name one deeper; how it wears a purpose
    is core's `Minting`.
    '''

    about = (Place, Label)

    @abstractmethod
    def beside(self, place: V, name: str) -> V:
        '''`place` one name deeper: where a read written as a value sits, under its purpose.'''

    @abstractmethod
    def node_at(self, place: V) -> N:
        '''The node name a place gives: where a closure over it binds.'''

    @abstractmethod
    def place_at(self, node: N) -> V:
        '''The place a node name gives: where a driver's per-problem data sits.'''

    @abstractmethod
    def node_in_tree(self, steps: tuple[Step, ...]) -> N:
        '''The node name a position in a tree of models gives: the steps to it from the root.'''


class Written[N: Label, V: Place](Answer):
    '''
    What a study writes where a name is due that is neither a string nor an exact name: the
    places it stands for, and the node name it takes. Registered by the owner of its class
    (pytree for a `V` / `N` builder, declaration for a wrapper or a problem); a class nobody
    answers is no name.
    '''

    about = object

    @abstractmethod
    def places(self, written: Any) -> tuple[V, ...] | None:
        '''The places `written` stands for; `None` where it is no place.'''

    @abstractmethod
    def node(self, written: Any, names: DerivedNames[N, V]) -> N | None:
        '''The node name `written` takes, `names` the surface's rules; `None` where it is no
        node name.'''


# ============================================================================ chosen

class Locator[V: Place, N: Label](Frozen, Choice):
    '''
    How a bare string is converted to a cottax native Place or Label.

    Handed in as:

    surface.located(Under(V.plasma, GetAttrKey)), a surface whose strings are attributes under
    `.plasma`; `cx.Scope(V.plasma, GetAttrKey)` is the same.
    '''

    places : AbstractClassVar[type[Place]]
    '''The class of place it locates: what the names derived from one are found by.'''
    nodes  : AbstractClassVar[type[Label]]
    '''The class of node name it locates.'''

    @abstractmethod
    def place(self, s: str) -> V:
        '''A string written where a place is due.'''

    @abstractmethod
    def node(self, s: str) -> N:
        '''A string written where a node is due.'''


class Part(Surface, Choice):
    '''
    A level of a surface adding methods: each takes exact names, its parameters annotated
    `Place` or `Label` read by the surface before it runs, and returns a declaration, an op or
    a run. What it defaults to is an `AbstractVar` a final binds. Two parts defining one name
    are refused where they are joined.

    Handed in as:

    class Mine(ShippedSurface, MyPart), the shipped parts and one of a study's own, then
    Mine(locator, conventions) -- or a surface joining only the parts it names.
    '''


class Editing(Part):
    '''Core's own edits, written by name: the ops a study writes by hand, each named after the op it builds; a copy goes
    where `copies` says.'''

    copies : AbstractVar[PlaceToPlace]
    '''Where a cut's copy of a place goes: `^copy.var`.'''

    def copy(self, var: Place) -> Place:
        '''Where a cut of `var` puts its copy, as this surface's cuts and closings mint it.'''
        return self.copies(var)

    def cut(self, var: Place, readers: tuple[Label, ...], *,
            copy_of: PlaceToPlace | None = None) -> Cut:
        '''`readers` (one, or several) read a copy of `var`; its producer stays.'''
        return Cut(var, readers, copy_of or self.copies)

    def rename(self, node: Label, onto: Mapping[Place, Place]) -> Rename:
        '''The node reads and owns each place of `onto` at the place it maps to.'''
        return Rename(node, tuple(onto.items()))

    def determine(self, requirement: Label, *unknowns: Place) -> Determine:
        '''The requirement determines `unknowns`: a problem.'''
        return Determine(requirement, unknowns)

    def nest(self, inner: Label, outer: Label) -> Nest:
        '''The problem `inner` is answered inside every iteration of `outer`.'''
        return Nest(inner, outer)

    def unnest(self, inner: Label) -> Unnest:
        '''The problem `inner` answered at its own level again.'''
        return Unnest(inner)

    def combine(self, at: Label, nodes: tuple[Label, ...], kind: type) -> Combine:
        '''`nodes` folded into one at `at`, by `kind`'s `+`.'''
        return Combine(at, nodes, kind)

    def delete(self, nodes: tuple[Label, ...]) -> Delete:
        '''`nodes` (one, or several) unbound.'''
        return Delete(nodes)

    def implement(self, implementations: Mapping[Label, object]) -> Implement:
        '''Each node run by what it maps to.'''
        return Implement(OrderedMap(implementations))


__all__ = ['DerivedNames', 'Written', 'Locator', 'Part', 'Editing']
