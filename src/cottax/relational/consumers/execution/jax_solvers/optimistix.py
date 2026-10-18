'''Newton, Picard and BFGS: optimistix answering a `RootFinder`, a `FixedPointIterator` and a `Minimiser`.'''

from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import optimistix as optx

from .....core import Place
from .....core.driven import DriverContract, Start, Steps
from .....core.frozen import choice
from .....relational.contracts import FixedPointIterator, Minimiser, RootFinder
from .....consumers.execution.interface import DriverImplementation
from .....consumers.execution.interface_types import (
    ByRole, Context, Posed, Unknowns,
)
from .canonical_nlp import NotSquare, canonical, next_values


def _size(tree: Any) -> int:
    '''How many numbers: one vector unknown may answer several scalar relations.'''
    return sum(jnp.size(leaf) for leaf in jax.tree.leaves(tree))


class Optimistix[D: DriverContract](DriverImplementation[D]):
    '''An optimistix solve from a `Start`, reporting its `Steps`. The unknowns are
    optimistix's `y` as they are, a tuple, and the context its `args`.'''

    needs    = frozenset({Start})
    provides = frozenset({Steps})

    rtol      : float = 1e-4
    atol      : float = 1e-4
    max_steps : int   = 256

    @abstractmethod
    def solve[V: Place](self, posed: Posed[V], context: Context,
                        data: ByRole) -> optx.Solution:
        '''The optimistix solve answering `posed`: its function takes `args=(posed,
        context)`.'''

    def __call__[V: Place](self, posed: Posed[V], context: Context,
                           data: ByRole) -> tuple[Unknowns, ByRole]:
        solution = self.solve(posed, context, data)
        steps = {Steps: (solution.stats['num_steps'],)}
        return tuple(solution.value), steps if Steps in posed.reports else {}


type Args = tuple[Posed, Context]
'''What each solver function is handed besides the unknowns: the static half, then the
values.'''


@choice(DriverImplementation)
class Newton(Optimistix[RootFinder]):
    '''`optx.Newton` on the equality residuals, as many numbers as the unknowns.'''

    def solve[V: Place](self, posed: Posed[V], context: Context,
                        data: ByRole) -> optx.Solution:
        solver = optx.Newton(rtol=self.rtol, atol=self.atol)
        return optx.root_find(_residuals, solver, data[Start], args=(posed, context),
                              max_steps=self.max_steps)


def _residuals(u: Unknowns, args: Args) -> tuple[Any, ...]:
    '''Newton's function: the equality residuals.'''
    posed, context = args
    equalities = canonical(posed.condition, u, posed.conditions(u, context)).equalities
    if _size(equalities) != _size(u):
        raise NotSquare('Newton', _size(equalities), _size(u)).raised
    return equalities


@choice(DriverImplementation)
class Picard(Optimistix[FixedPointIterator]):
    '''`optx.FixedPointIteration`: `u <- g(u)`, no Jacobian taken.'''

    def solve[V: Place](self, posed: Posed[V], context: Context,
                        data: ByRole) -> optx.Solution:
        solver = optx.FixedPointIteration(rtol=self.rtol, atol=self.atol)
        return optx.fixed_point(_update, solver, data[Start], args=(posed, context),
                                max_steps=self.max_steps)


def _update(u: Unknowns, args: Args) -> tuple[Any, ...]:
    '''Picard's function: the next value of every unknown.'''
    posed, context = args
    return next_values(posed.condition, u, posed.conditions(u, context))


@choice(DriverImplementation)
class BFGS(Optimistix[Minimiser]):
    '''`optx.BFGS` on the one objective.'''

    def solve[V: Place](self, posed: Posed[V], context: Context,
                        data: ByRole) -> optx.Solution:
        solver = optx.BFGS(rtol=self.rtol, atol=self.atol)
        return optx.minimise(_cost, solver, data[Start], args=(posed, context),
                             max_steps=self.max_steps)


def _cost(u: Unknowns, args: Args) -> Any:
    '''BFGS's function: the one objective.'''
    posed, context = args
    (objective,) = canonical(posed.condition, u, posed.conditions(u, context)).objectives
    return objective
