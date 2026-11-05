'''The arithmetic over a `Programme` at the values of a condition's reads: `canonical`,
every objective minimised and every relation a residual, `box`, `scaled`, both in a scaled
driver's units, and `next_values`.'''

import operator
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .....core import Place, Violation
from .....relational import Max, RelationalCondition, programme
from .....relational.mints import ResidualScale


class Canonical(NamedTuple):
    '''`min objectives` subject to `equalities = 0` and `inequalities <= 0`.'''

    objectives   : tuple[Any, ...]
    equalities   : tuple[Any, ...]
    inequalities : tuple[Any, ...]


class Box(NamedTuple):
    '''Per unknown, in `condition.unknowns` order, its lower and upper bound, `None` where
    it has none.'''

    lower : tuple[Any, ...]
    upper : tuple[Any, ...]


def canonical[V: Place](condition: RelationalCondition[V], unknowns: tuple[Any, ...],
                        values: tuple[Any, ...]) -> Canonical:
    '''The objectives, and every relation but a bound as a residual in its group.'''
    value = _values(condition, unknowns, values)
    form  = programme(condition)
    return Canonical(tuple(_minimised(value(o.place), o.sense) for o in form.objectives),
                     tuple(_residual(value(r.lhs), value(r.rhs)) for r in form.equalities),
                     tuple(_residual(value(r.lhs), value(r.rhs)) for r in form.inequalities))


def _minimised(value: Any, sense: type) -> Any:
    '''An objective as one to minimise: a maximised one negated.'''
    return jax.tree.map(operator.neg, value) if sense is Max else value


def box[V: Place](condition: RelationalCondition[V], unknowns: tuple[Any, ...],
                  values: tuple[Any, ...]) -> Box:
    '''Each unknown's bounds: the tightest of its lower bounds, and of its upper ones.'''
    value = _values(condition, unknowns, values)
    form  = programme(condition)
    bound = lambda side: 0.0 if side is None else value(side)
    return Box(tuple(_tightest(jnp.maximum, map(bound, form.lower[u])) for u in condition.unknowns),
               tuple(_tightest(jnp.minimum, map(bound, form.upper[u])) for u in condition.unknowns))


def _tightest(fold, bounds) -> Any:
    '''The bounds folded in the order met, the first as it is; `None` for none.'''
    tightest = None
    for i, b in enumerate(bounds):
        tightest = b if i == 0 else fold(tightest, b)
    return tightest


def scaled[V: Place](condition: RelationalCondition[V], unknowns: tuple[Any, ...],
                     values: tuple[Any, ...], column: tuple[Any, ...]) -> tuple[Canonical, Box]:
    '''`canonical` and `box` in `y = x / column`, for `unknowns` the values `x` at which
    `values` were read: each relation's residual over the magnitude of the place it attaches
    as its `ResidualScale` (one attaching none read as it is), the box over `column`, its
    sides swapped where a column is negative (a side then `-inf` / `inf` where unbounded).
    Objectives as they are.'''
    value = _values(condition, unknowns, values)
    form  = programme(condition)
    def held(r):
        residual = _residual(value(r.lhs), value(r.rhs))
        if (scale := r.attached.get(ResidualScale)) is None:
            return residual
        return jax.tree.map(lambda e, s: e / jnp.abs(s), residual, value(scale))
    lower, upper = box(condition, unknowns, values)
    def scaled_box(lo, hi, c):
        if lo is None and hi is None:
            return None, None
        lo = -jnp.inf if lo is None else lo
        hi =  jnp.inf if hi is None else hi
        return jnp.where(c > 0, lo, hi) / c, jnp.where(c > 0, hi, lo) / c
    sides = tuple(scaled_box(*b) for b in zip(lower, upper, column, strict=True))
    return (Canonical(tuple(_minimised(value(o.place), o.sense) for o in form.objectives),
                      tuple(map(held, form.equalities)), tuple(map(held, form.inequalities))),
            Box(tuple(lo for lo, _ in sides), tuple(hi for _, hi in sides)))


def next_values[V: Place](condition: RelationalCondition[V], unknowns: tuple[Any, ...],
                          values: tuple[Any, ...]) -> tuple[Any, ...]:
    '''For `u = g(u)`: `g` per unknown, in `condition.unknowns` order -- the update an
    iteration applies. One relation per unknown is `FixedPointIterator.accepts`'s.'''
    value  = _values(condition, unknowns, values)
    update = {r.lhs: r.rhs for r in condition.relations}
    return tuple(value(update[u]) for u in condition.unknowns)


def _values[V: Place](condition: RelationalCondition[V], unknowns: tuple[Any, ...],
                      values: tuple[Any, ...]):
    '''The value at a place: an unknown's, or a read's (a condition or a parameter).'''
    at = dict(zip(condition.unknowns, unknowns)) | dict(zip(condition.reads, values))
    return lambda p: None if p is None else at[p]


def _residual(lhs: Any, rhs: Any) -> Any:
    '''`lhs - rhs` leaf-wise, an absent side being zero.'''
    if rhs is None:
        return lhs
    if lhs is None:
        return jax.tree.map(operator.neg, rhs)
    return jax.tree.map(operator.sub, lhs, rhs)


# ----------------------------------------------------------------- violation finals

class NotSquare(Violation):
    implementation : str
    got            : int
    unknowns       : int
    error    = ValueError
    template = ('{implementation} needs as many numbers in the residuals as in the unknowns '
                '-- this problem gives {got} for {unknowns}')
