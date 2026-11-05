'''
`scipy.optimize.minimize` as a `HostMinimiser`: `SLSQP`, and `ScipyMinimize` for any method,
refusing a problem it would not honour: scipy itself only warns of what it drops, and answers
without it.
'''

from abc import abstractmethod
from collections.abc import Callable, Iterator
from typing import NamedTuple

import jax
import numpy as np
from scipy.optimize import minimize

from .....core import Violation, first
from .....core.driven import Steps
from .....core.frozen import choice
from .....relational import RelationalCondition, programme
from .....relational.contracts import ConstrainedMinimiser
from .....consumers.execution.interface import DriverImplementation
from .....consumers.execution.interface_types import ByRole, Context, Posed, Unknowns
from .host import HostMinimiser, HostProblem


class Honours(NamedTuple):
    '''What a method of `minimize` does with a problem.'''

    equalities   : bool
    inequalities : bool
    bounds       : bool
    gradient     : bool
    '''Takes a gradient, so a Jacobian is compiled; off, none ever is.'''
    steps_option : str = 'maxiter'


METHODS = {
    'SLSQP'        : Honours(True,  True,  True,  True),
    'trust-constr' : Honours(True,  True,  True,  True),
    'COBYQA'       : Honours(True,  True,  True,  False),
    'COBYLA'       : Honours(True,  True,  True,  False),
    'L-BFGS-B'     : Honours(False, False, True,  True),
    'TNC'          : Honours(False, False, True,  True, 'maxfun'),
    'Nelder-Mead'  : Honours(False, False, True,  False),
    'Powell'       : Honours(False, False, True,  False),
    'BFGS'         : Honours(False, False, False, True),
    'CG'           : Honours(False, False, False, True),
}
'''Each method on offer, as scipy 1.18 behaves. Those that want a Hessian are not: nothing here
computes one.'''


def check_method(method: str, condition: RelationalCondition) -> Iterator[Violation]:
    '''`method` one on offer, and honouring all of `condition`: dropped, a constraint or a bound
    would leave an answer that breaks it, with nothing to say so but a warning.'''
    if method not in METHODS:
        yield NoSuchMethod(method, tuple(METHODS))
        return
    form, honours = programme(condition), METHODS[method]
    bounds = [b for side in (form.lower, form.upper) for bs in side.values() for b in bs]
    for name, stated, honoured in (('equalities', form.equalities, honours.equalities),
                                   ('inequalities', form.inequalities, honours.inequalities),
                                   ('bounds', bounds, honours.bounds)):
        if stated and not honoured:
            yield MethodDropsIt(method, name, len(stated))


class Scipy(HostMinimiser[ConstrainedMinimiser]):
    '''`scipy.optimize.minimize` by `method`: the equalities as `eq`, the inequalities negated
    as `ineq`, the box as `bounds`, and a Jacobian only where the method takes one. Under a
    contract taking `Scales`, scipy sees everything read in that `Scaling`, its tolerance
    included.'''

    reported = {Steps: int}
    provides = frozenset(reported)

    tol       : float = 1e-8
    max_steps : int   = 200
    strict    : bool  = True
    '''Raise when scipy does not report success; off, the answer is taken as it stands.'''
    jacobian  : Callable = jax.jacfwd
    '''How the Jacobian is taken: `jax.jacfwd`, `jax.jacrev`, or any hashable function of a
    function taking `has_aux=True`, such as one pulling rows back one at a time to save memory.'''

    @property
    @abstractmethod
    def method(self) -> str:
        '''The method of `minimize`: a final's class attribute, or its field.'''

    @property
    def asks(self) -> frozenset:
        return frozenset({'values', 'jacobians'} if METHODS[self.method].gradient else {'values'})

    def __call__(self, posed: Posed, context: Context, data: ByRole) -> tuple[Unknowns, ByRole]:
        first(check_method(self.method, posed.condition))
        return super().__call__(posed, context, data)

    def minimise(self, problem: HostProblem) -> tuple[np.ndarray, dict]:
        honours = METHODS[self.method]
        jac = lambda take: {'jac': lambda y: take(problem.jacobians(y))} if honours.gradient else {}
        constraints = [{'type': kind, 'fun': lambda y, take=take: take(problem.values(y)), **jac(take)}
                       for kind, take in (('eq', lambda c: c.equalities), ('ineq', lambda c: -c.inequalities))
                       if take(problem.values(problem.y0)).size]
        box = np.stack([problem.lower, problem.upper])
        bounds = (None if not np.isfinite(box).any() else
                  [tuple(None if not np.isfinite(b) else float(b) for b in pair) for pair in box.T])
        result = minimize(lambda y: problem.values(y).objectives.item(), problem.y0,
                          jac=jac(lambda c: c.objectives[0]).get('jac'), method=self.method,
                          bounds=bounds, constraints=constraints, tol=self.tol,
                          options={honours.steps_option: self.max_steps})
        if self.strict and not result.success:
            raise RuntimeError(f'{self.method} did not converge: {result.message}')
        return result.x, {Steps: result.get('nit', result.nfev)}


@choice(DriverImplementation)
class SLSQP(Scipy):
    '''`minimize(method='SLSQP')`, so every iterate stays inside the box.'''

    method = 'SLSQP'


@choice(DriverImplementation)
class ScipyMinimize(Scipy):
    '''`minimize` by the method named, one of `METHODS`. `max_steps` is scipy's `maxiter`,
    `maxfun` for `TNC`; `Steps` is its `nit`, the evaluations for `COBYLA`, which counts none.

    `tol` is each method's own test, so answers differ by more than it. `trust-constr` is an
    interior point: `tol` ends it on a barrier still in force, inside an active inequality or
    bound, by about 4e-4 at `tol=1e-8` on a unit problem and 1.6e-5 at `1e-12`.'''

    method : str = 'SLSQP'


# ----------------------------------------------------------------- violation finals

class NoSuchMethod(Violation):
    method   : str
    offered  : tuple
    error    = ValueError
    template = ('{method!r}: not a `minimize` method on offer, of {offered!r}. Those that want a '
                'Hessian are not: nothing here computes one')


class MethodDropsIt(Violation):
    method   : str
    group    : str
    count    : int
    error    = ValueError
    template = ('{method} has no way to hold {group}, and this problem states {count}: scipy would '
                'answer without them')
