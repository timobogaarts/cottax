'''
What a spelling writes and reads a node through (`Writer`, `Reader`), and `Json`, what it
writes.
'''

from typing import Any, Protocol

from ...core import Label, NodeDefinition, Place


type Json = Any


# ============================================================================ of a node

class Writer[N: Label, V: Place](Protocol):
    '''What a spelling writes a node through, for the node bound at `name`.'''

    name : N

    def place(self, place: V) -> Json: ...
    def places(self, places: tuple[V, ...]) -> list: ...
    def label(self, name: N) -> Json: ...
    def node(self, node: NodeDefinition[V]) -> dict:
        '''A node it holds, by its own final's spelling.'''
    def implementation(self, obj: object) -> dict:
        '''What runs a node: by name, or a dataclass as a value.'''
    def value(self, obj: object) -> dict:
        '''A dataclass as its class and fields: a driver, say.'''
    def refuse(self, what: str, reason: str) -> Exception:
        '''What to raise where `what` is not written: the export's check reports it.'''


class Reader[N: Label, V: Place](Protocol):
    '''What a spelling reads a node through, for the node bound at `name`.'''

    name : N

    def place(self, data: Json) -> V: ...
    def places(self, data: list) -> tuple[V, ...]: ...
    def label(self, data: Json) -> N: ...
    def node(self, data: dict) -> NodeDefinition[V]: ...
    def implementation(self, data: dict) -> object: ...
    def value(self, data: dict) -> Any: ...


__all__ = ['Json', 'Writer', 'Reader']
