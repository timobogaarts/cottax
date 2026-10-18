'''
JSON interface asks of a place type:

1.Spelling of places (`PlaceCodec`) and of node names (`LabelCodec`)

Of a feature type:

2. Spelling of node kinds
'''

from abc import abstractmethod
from typing import Any

from ...core import Label, NodeDefinition, Place
from ...core.frozen import AbstractClassVar, Answer
from .interface_types import Json, Reader, Writer
from .refusals import one_per_kind, one_per_tag


# ============================================================================ registered

class Spelling[T: NodeDefinition](Answer):
    '''NodeDefinition <-> JSON.'''

    about = NodeDefinition

    kind : AbstractClassVar[str]
    '''How it is referred to in the file.'''

    checks_answer = (one_per_kind,)

    @abstractmethod
    def write[V: Place](self, node: T, w: Writer[Any, V]) -> dict:
        '''The node's fields; recursive through wrapped nodes (Driven/Implemented)'''

    @abstractmethod
    def read[V: Place](self, data: dict, r: Reader[Any, V]) -> T:
        '''The node `data` spells: recursive through wrapped nodes (Driven/Implemented)'''


class PlaceCodec[V: Place](Answer):
    '''Place <-> JSON, for the place classes it answers for. `tag` names it in the file
    header.'''

    about = Place
    tag : AbstractClassVar[str]

    checks_answer = (one_per_tag,)

    @abstractmethod
    def write_place(self, place: V, w: Writer[Any, V]) -> Json:
        '''`place`; a value it holds (a `Mint`) through `w.value`.'''
    @abstractmethod
    def read_place(self, data: Json, r: Reader[Any, V]) -> V:
        '''The place `data` spells; a value it holds through `r.value`.'''


class LabelCodec[N: Label](Answer):
    '''Node name <-> JSON, for the label classes it answers for. `tag` names it in the file
    header.'''

    about = Label
    tag : AbstractClassVar[str]

    checks_answer = (one_per_tag,)

    @abstractmethod
    def write_label(self, name: N, w: Writer[N, Any]) -> Json:
        '''`name`; a value it holds (a `Mint`) through `w.value`.'''
    @abstractmethod
    def read_label(self, data: Json, r: Reader[N, Any]) -> N:
        '''The node name `data` spells; a value it holds through `r.value`.'''


__all__ = ['Spelling', 'PlaceCodec', 'LabelCodec']
