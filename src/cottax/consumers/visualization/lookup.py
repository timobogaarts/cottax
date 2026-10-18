'''
What a drawing asks of a node or a place: the answer registered for its class, or, for one
nobody answered, the default: a grey problem with no slots, a place written as its `str`.
'''

from typing import Any

from ...core.frozen import answer
from ...core import Label, Mint, Place
from .interface import NodeDrawing, PlaceDrawing
from .interface_types import Drawn


class Undrawn(NodeDrawing):
    '''No category and no slots: a grey problem.'''

    def drawn(self, node: Any) -> Drawn:
        return Drawn(None, ())


class AsStr(PlaceDrawing):
    '''Its `str`, every name its own, in no group.'''

    def spell_plain(self, place: Label) -> str:
        return str(place)

    def spelling(self, place: Label) -> str:
        return str(place)

    def minted_from(self, place: Label) -> tuple[Mint, Place] | None:
        return None

    def namespace_of(self, place: Label, depth: int | None = None) -> tuple[str, ...]:
        return ()


UNDRAWN, AS_STR = Undrawn(), AsStr()


def drawn(node: Any) -> Drawn:
    return answer(NodeDrawing, node, UNDRAWN).drawn(node)


def spell_plain(place: Label) -> str:
    return answer(PlaceDrawing, place, AS_STR).spell_plain(place)


def spelling(place: Label) -> str:
    return answer(PlaceDrawing, place, AS_STR).spelling(place)


def minted_from(place: Label) -> tuple[Mint, Place] | None:
    return answer(PlaceDrawing, place, AS_STR).minted_from(place)


def namespace_of(place: Label, depth: int | None = None) -> tuple[str, ...]:
    return answer(PlaceDrawing, place, AS_STR).namespace_of(place, depth)


__all__ = ['drawn', 'spell_plain', 'spelling', 'minted_from', 'namespace_of']
