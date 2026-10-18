'''
`Conventions`, what a declaring part's arguments mean, and cottax's own readings: a
function's reads (`ParameterReads`), a term written where a place is due (`term_of`), what is
written per place (`per_place_of`).
'''

from collections.abc import Callable, Mapping
import numbers
from typing import Any, Protocol

import numpy as np

from ...core import Place, PlaceToPlace
from ...core.frozen import Frozen
from ..surface.interface_types import Named
from .interface_types import PlaceWithValue, Tagged, untagged
from .refusals import ReadsUnlike


# ---------------------------------------------------------------- type aliases
class ArrayLike(Protocol):
    def __array__(self, *args: Any, **kwargs: Any) -> Any: ...

type Value      = int | float | complex | ArrayLike   # never a bool: a switch is `Static`.
type Given      = Value | Tagged
# Value, possibly tagged 

type Reads[P]   = Mapping[str, Named[P] | Given] | tuple[Named[P] | Given, ...]
# How reads are given: either a mapping of argname -> Named/Given or a tuple of Named/Given in signature order.

type ReadIn[V]  = Mapping[Callable[[str], V], str | tuple[str, ...]]
# Additional names to read by a rule: a mapping of rule -> name(s), applied to all parameters in the right argument. 

# ---------------------------------------------------------------- cottax's readings

class ParameterReads(Frozen):
    '''cottax's reading of a function's reads (`reads_of`), a parameter `reads` omits read at
    its own name unless `by_name` is off.'''

    by_name : bool = True

    def __call__(self, parameters, reads, read_in, *, place, given, of):
        return reads_of(parameters, reads, read_in, place=place, given=given, of=of,
                        by_name=self.by_name)


def reads_of[V: Place](parameters: tuple[str, ...], reads: Reads, read_in: ReadIn[V], *,
                       place: Callable[[Any], V], given: Callable[[str], V], of: str,
                       by_name: bool) -> tuple[PlaceWithValue[V], ...]:
    '''The reads of the function `of` taking `parameters`, one per parameter in order: where
    `reads` says (a tuple is every parameter's, in order); else by name under the rule
    `read_in` names it with; else, `by_name`, by name through `place`. A read written as a
    value, tagged or not, is minted at `given(parameter)`; a name or a path tagged is that
    place, carrying the tags. Every parameter once, and no other name.'''
    if isinstance(reads, (tuple, list)):
        if len(reads) != len(parameters):
            raise ValueError(f'{of} takes {parameters!r}, but {len(reads)} reads are '
                             f'given: a tuple of reads names every parameter, in order')
        reads = dict(zip(parameters, reads))
    by_rule = {}
    for rule, names in read_in.items():
        names = (names,) if isinstance(names, str) else names
        if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
            raise TypeError(f'read_in: {names!r} -- a name or tuple of names, read by that rule')
        for n in names:
            if n in by_rule or n in reads:
                raise ValueError(f'{of}: {n!r} is read twice -- in `reads` and '
                                 f'`read_in`, or twice in `read_in`')
            by_rule[n] = rule(n)
    written = ({p: p for p in parameters} if by_name else {}) | by_rule | dict(reads)
    if set(written) != set(parameters):
        order = [p for p in parameters if p in written] + [k for k in written if k not in parameters]
        raise ReadsUnlike(of, tuple(order), parameters).raised
    # zero too is a value: a read is never none
    return tuple(written_at(written[p], place=place, minted=lambda: given(p))
                 for p in parameters)


def written_at[V: Place](x: Any, *, place: Callable[[Any], V], minted: Callable[[], V]
                         ) -> PlaceWithValue[V]:
    '''What is written where a place is due, as a place: a name or a path, tagged or not,
    that place carrying the tags; a value, tagged or not, at `minted()`, given it bare and
    the place carrying the tags.'''
    inner, tags = untagged(x)
    if tags and (at := _tagged_place(inner, place)) is not None:
        return PlaceWithValue(at, tags=tags)
    if isinstance(x, Tagged) or _is_value(x):
        return PlaceWithValue(minted(), inner, tags)
    return PlaceWithValue(place(x))


# ---------------------------------------------------------------- a term

def _is_value(x: Any) -> bool:
    '''A number or a numeric array (numpy's, jax's): a value, never a name. A `bool` is
    neither: a switch is `Static`, and a flag where a place is due is a mistake.'''
    if isinstance(x, bool):
        return False
    if isinstance(x, numbers.Number):
        return True
    return hasattr(x, '__array__') and np.asarray(x).dtype.kind in 'iufc'


def _tagged_place[V: Place](inner: Any, place: Callable[[Any], V]) -> V | None:
    '''What a tag wraps, as a place: a string, or a path `place` takes; `None` for anything
    else, a value.'''
    if isinstance(inner, str):
        return place(inner)
    if isinstance(inner, bool) or _is_value(inner):
        return None
    try:
        return place(inner)
    except TypeError:
        return None


def _is_zero(x: Any) -> bool:
    '''Zero, the one value that is structure: a side of zero is no place at all.'''
    return _is_value(x) and np.ndim(x) == 0 and x == 0


def term_of[V: Place](x: V, written: Any, *, place: Callable[[Any], V], value_at: PlaceToPlace[V]
                     ) -> PlaceWithValue[V] | None:
    '''What is written where a place is due, about `x`: zero (`None`), a value (a place minted
    by `value_at` off `x`, given it), or a place; tagged or not, as a read is.'''
    if _is_zero(written):
        return None
    return written_at(written, place=place, minted=lambda: value_at(x))


def per_place_of[V: Place](among: tuple[V, ...], written: Any, *, place: Callable[[Any], V]) -> dict[V, Any]:
    '''What is written for each of `among`: a mapping by name, or one thing for the one place
    there is.'''
    if isinstance(written, Mapping):
        return {place(x): w for x, w in written.items()}
    if len(among) != 1:
        raise TypeError(f'one {written!r} for {len(among)} places: name each place\'s in a mapping')
    return {among[0]: written}


# ---------------------------------------------------------------- what a surface's arguments mean

class Conventions(Frozen):
    '''
    What a declaring part's arguments mean, in three shapes, each a callable and cottax's by
    default, so a study changes one: `Conventions(term=my_term)`, `my_term` calling `term_of`
    for what it does not read itself. Held as a declaring part's `conventions`, and asked
    through its methods of the same names (`parameter_reads`, `term`, `per_place`).
    '''

    reads     : Callable[..., tuple[PlaceWithValue, ...]] = ParameterReads()
    '''`reads(parameters, reads, read_in, *, place, given, of)`: the reads of the function
    `of`, one per parameter in order: `reads` as written (a mapping by parameter, or a tuple
    of every one), `read_in` a rule per name, a value minted at `given(parameter)`.'''
    term      : Callable[..., PlaceWithValue | None] = term_of
    '''`term(x, written, *, place, value_at)`: what is written where a place about `x` is due:
    `None` for zero, a value at `value_at(x)`, or a place.'''
    per_place : Callable[..., dict] = per_place_of
    '''`per_place(among, written, *, place)`: what is written for each of `among`: a mapping
    by name, or one thing for the one place there is.'''


def Shipped(by_name: bool = True) -> Conventions:
    '''cottax's conventions, a parameter read at its own name unless `by_name` is off.'''
    return Conventions(reads=ParameterReads(by_name))
