r'''
The jax backend of the driver contracts: Newton, Picard, BFGS and any method of
`scipy.optimize.minimize`, each chosen per contract in `ImplementDrivers`; `HostMinimiser` for
writing another solver that runs on the host.

Dependency:

canonical_nlp                    (canonical, box, scaled: `relational.programme` computed over the
 ||-----------------              values at the condition's reads; next_values, Picard's
 ||                ||             step; NotSquare)
 ||                \/
 ||               host           (HostMinimiser, HostProblem: scaling, one flat vector, the
 ||                ||             compiled programs and `pure_callback`, for a solver on the host)
 \/                \/
optimistix      scipy_minimize   (optimistix: Newton for a RootFinder, Picard for a
                                  FixedPointIterator, BFGS for a Minimiser;
                                  scipy_minimize: ScipyMinimize for a ConstrainedMinimiser, and
                                  `METHODS`: what each method honours, refusing the rest)

'''

from .optimistix import BFGS, Newton, Picard
from .host import HostMinimiser, HostProblem
from .scipy_minimize import ScipyMinimize

__all__ = [
    'Newton', 'Picard', 'BFGS',
    'ScipyMinimize', 'HostMinimiser', 'HostProblem',
]
