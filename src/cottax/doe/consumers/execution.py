'''Offers `JaxDoe`, what runs a `Sampler` in jax, chosen per contract in `ImplementDrivers`.'''

import jax
import jax.numpy as jnp

from ...core import Place, Violation
from ...core.frozen import choice
from ...consumers.execution.interface import DriverImplementation
from ...consumers.execution.interface_types import ByRole, Context, Posed, Unknowns
from ..contracts import Sampler


@choice(DriverImplementation)
class JaxDoe(DriverImplementation[Sampler]):
    '''`jax.lax.map` of the condition map over the points of the plans, each response's values
    stacked into its record. `batch_size` points are evaluated at once, vmapped; `None` one at
    a time. The factors are left at the last point, which holds no answer (pending a check of
    the condition kind). A plan is an array of points along its leading axis, every plan as
    many, refused otherwise when traced.'''

    needs    = frozenset()
    provides = frozenset()

    batch_size : int | None = None

    def __call__[V: Place](self, posed: Posed[V], context: Context,
                           data: ByRole) -> tuple[Unknowns, ByRole]:
        doe   = posed.condition
        plans = posed.fixed(doe.plans, context)
        lengths = tuple(tuple(jnp.shape(leaf)[:1] for leaf in jax.tree.leaves(p)) for p in plans)
        axes = {axis for found in lengths for axis in found}
        if len(axes) != 1 or axes & {(), (0,)}:                    # unlike, a scalar, or empty
            raise PlansDiffer(tuple(map(str, doe.owns_inside)), lengths).raised
        responses = len(doe.responses)
        records = jax.lax.map(lambda point: tuple(posed.conditions(point, context)[:responses]),
                              plans, batch_size=self.batch_size)
        return (*jax.tree.map(lambda leaf: leaf[-1], plans), *records), {}


class PlansDiffer(Violation):
    factors : tuple
    lengths : tuple
    error    = ValueError
    template = ('{factors} are set from plans whose leading axes are {lengths}: a plan holds '
                'its points along its leading axis, at least one, every plan as many')
