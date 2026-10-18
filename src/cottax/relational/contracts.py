'''
Driver contracts:

1. Generic levels: `RootFinder`, `FixedPointIterator`, `Minimiser`, `ConstrainedMinimiser`.
2. Finals over these levels with steps: `RootFinding`, `FixedPointIteration`, `Minimisation`, `ConstrainedMinimisation`.
3. `CONTRACTS`, the defaults: one of each final, without steps.
'''

from ..core import Place
from ..core.driven import DriverContract, Role, Start, Steps
from .classification import is_equalities, is_fixed_point, is_optimise, is_unconstrained_optimise


# ---------------------------------------------------------------- the levels

class RootFinder[V: Place](DriverContract[V]):
    '''Answers equalities, a fixed point among them.'''
    accepts = staticmethod(is_equalities)


def is_iteration(node: object) -> bool:
    '''A fixed point with each unknown on the left of exactly one relation: one update
    `u <- g(u)` per unknown.'''
    if not is_fixed_point(node):
        return False
    lhs = [r.lhs for r in node.relations]
    return all(lhs.count(u) == 1 for u in node.unknowns)


class FixedPointIterator[V: Place](DriverContract[V]):
    '''Answers `u = g(u)` by updating each unknown to its `g(u)`.'''
    accepts = staticmethod(is_iteration)


class Minimiser[V: Place](DriverContract[V]):
    '''Answers one objective with no relation beside it.'''
    accepts = staticmethod(is_unconstrained_optimise)


class ConstrainedMinimiser[V: Place](DriverContract[V]):
    '''Answers one objective, with or without relations to respect.'''
    accepts = staticmethod(is_optimise)

# ---------------------------------------------------------------- the generic finals

class RootFinding[V: Place](RootFinder[V]):
    '''From a `Start`; `steps` reports `Steps`.'''
    steps : bool

    @property
    def takes(self) -> tuple[type[Role], ...]:   return (Start,)
    @property
    def reports(self) -> tuple[type[Role], ...]: return (Steps,) if self.steps else ()


class FixedPointIteration[V: Place](FixedPointIterator[V]):
    '''From a `Start`; `steps` reports `Steps`.'''
    steps : bool

    @property
    def takes(self) -> tuple[type[Role], ...]:   return (Start,)
    @property
    def reports(self) -> tuple[type[Role], ...]: return (Steps,) if self.steps else ()


class Minimisation[V: Place](Minimiser[V]):
    '''From a `Start`; `steps` reports `Steps`.'''
    steps : bool

    @property
    def takes(self) -> tuple[type[Role], ...]:   return (Start,)
    @property
    def reports(self) -> tuple[type[Role], ...]: return (Steps,) if self.steps else ()

class ConstrainedMinimisation[V: Place](ConstrainedMinimiser[V]):
    '''From a `Start`; `steps` reports `Steps`. Its bounds are relations of the problem
    over parameters, not data of the driver: they change the answer.'''
    steps   : bool

    @property
    def takes(self) -> tuple[type[Role], ...]:   return (Start,)
    @property
    def reports(self) -> tuple[type[Role], ...]: return (Steps,) if self.steps else ()


CONTRACTS = (RootFinding(steps=False), FixedPointIteration(steps=False),
             Minimisation(steps=False), ConstrainedMinimisation(steps=False))
'''The default contracts: an implementation answers a problem by the first one it implements.'''
