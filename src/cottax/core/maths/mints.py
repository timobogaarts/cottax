'''
A purpose a name is derived for: a `Mint` is a value, its class the purpose, and calling it on
a name gives the derived name (`Copy()(y)` is `^copy.y`), so it is the `PlaceToPlace` an op is
handed. How a class of names wears one is its `Minting`, registered by the place.
'''

import dataclasses
from abc import abstractmethod
from typing import Any

from ..frozen import Answer, Frozen, answer
from .places import Label, Place


class Mint(Frozen):
    '''A purpose, by its class: what a name is derived for. Its spelling (`^copy`) labels a
    derived name; nothing finds a purpose by it.'''

    def __new__(cls, *_: Any, **__: Any) -> 'Mint':
        if cls is Mint:
            raise TypeError('Mint is abstract: a purpose is a final of its own '
                            '(`class Copy(Mint)`)')
        return super().__new__(cls)

    def __call__[P](self, name: P) -> P:
        '''`name` in this purpose's namespace: the derived name.'''
        return minted(self, name)

    def __hash__(self) -> int:
        '''Its class's and its fields': two purposes without a field are unequal, and hash so.'''
        return hash((type(self), *(getattr(self, f.name) for f in dataclasses.fields(self))))

    def __str__(self) -> str:
        return f'^{type(self).__name__.lower()}'

    def __repr__(self) -> str:
        return str(self)


class Minting[P](Answer):
    '''How a class of names wears a purpose.'''

    about = (Place, Label)

    @abstractmethod
    def minted(self, m: Mint, name: P) -> P:
        '''`name` in `m`'s namespace, over whatever namespace it is in already.'''

    @abstractmethod
    def mint_of(self, name: P) -> Mint | None:
        '''The purpose `name` wears outermost; `None` for a name nobody minted.'''

    @abstractmethod
    def unminted(self, name: P) -> P:
        '''`name` with its outermost purpose dropped; `name` itself if it wears none.'''


def minted[P](m: Mint, name: P) -> P:
    '''`name` in `m`'s namespace, as its class's `Minting` says: refused where none does.'''
    return answer(Minting, name).minted(m, name)


def mint_of(name: Any) -> Mint | None:
    '''The purpose `name` wears outermost; `None` for one nobody minted, or whose class has no
    `Minting`.'''
    found = answer(Minting, name, None)
    return None if found is None else found.mint_of(name)


def unminted[P](name: P) -> P:
    '''`name` with its outermost purpose dropped; itself where it wears none.'''
    return name if mint_of(name) is None else answer(Minting, name).unminted(name)


# ---------------------------------------------------------------- the purposes the maths' ops derive names for

class Copy(Mint):
    '''A cut's copy of a variable: `^copy.y`.'''


class Closure(Mint):
    '''Where a closure of cuts binds: `^closure.<the cycle's first member>`.'''
