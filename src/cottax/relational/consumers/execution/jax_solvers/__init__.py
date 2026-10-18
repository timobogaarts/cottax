r'''
The jax backend of the driver contracts: Newton, Picard, BFGS and SLSQP, each chosen per
contract in `ImplementDrivers`.

Dependency:

canonical_nlp                    (canonical, box: `relational.programme` computed over the
 ||-----------------              values at the condition's reads; next_values, Picard's
 ||                               step; NotSquare)
 \/               \/
optimistix       scipy_slsqp     (optimistix: Newton for a RootFinder, Picard for a
                                  FixedPointIterator, BFGS for a Minimiser;
                                  scipy_slsqp: SLSQP for a ConstrainedMinimiser, through
                                  pure_callback)

'''

from .optimistix import BFGS, Newton, Picard
from .scipy_slsqp import SLSQP

__all__ = [
    'Newton', 'Picard', 'BFGS',
    'SLSQP',
]
