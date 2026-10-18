'''
The shipped assembly: pytree places, the relational and ode languages, the jax solvers and the
drawings; its surface, its defaults and the joins between consumers.
'''

from collections.abc import Callable, Mapping
from typing import Any

from ..core import (
    ByName, Closure, Copy, Mint, PlaceToPlace, GaussSeidel, GaussSeidelMinimal, GaussSeidelOrdered,
    Jacobi, NoCuts,
)
from ..core.frozen import OrderedMap, choice
from ..core.driven import Driven, Start
from ..consumers.architectures import (
    IDF, MDA, MDF, SAND, AtFirstMember, Analysis, Architecture, Bodies, Boxes,
    Architecting, CloseFunctionalCycles, Coupled, Disciplines, IndividualFeasible, Monolithic,
    MultidisciplinaryFeasible, Resolution, Simultaneous, Solves,
)
from ..consumers.execution import CallableGraph, Drive, Driving, Inputs, Results, Running
from ..consumers.execution.jax_jit import JitGraph, Traced
from ..consumers.architectures.interface import Closing
from ..pytree.path import DictKey, GetAttrKey, SequenceKey
from ..pytree.consumers.execution import completed
from ..pytree.consumers.declaration import Static
from ..pytree.consumers.surface import InFull, N, PytreeLocator, Under, V, Var
from ..relational import (
    Eq, FixedPointCut, Le, Max, Min, Objective, Relation, RelationalCondition,
)
from ..relational.contracts import (
    CONTRACTS as _RELATIONAL_CONTRACTS, ConstrainedMinimisation, ConstrainedMinimiser, FixedPointIteration, FixedPointIterator,
    Minimisation, Minimiser, RootFinder, RootFinding,
)
from ..ode import Dt, OdeCondition
from ..ode.contracts import (
    CONTRACTS as _ODE_CONTRACTS, Integration, Integrator, Times, Trajectory,
)
from ..ode.consumers.declaration import OdeDeclaring, Rate
# The answers the shipped place and language register.
from ..ode import consumers as _ode_answers  # noqa: F401
from ..pytree import consumers as _pytree_answers  # noqa: F401
from ..relational import consumers as _relational_answers  # noqa: F401
from ..relational.consumers.architectures import FixedPointClosing
from ..relational.consumers.declaration import (
    EqualityConstraint, FixedPoint, Implicit, InequalityConstraint, Problem, RelationalDeclaring,
    Residual,
)
from ..relational.consumers.surface import RelationalEditing
from ..relational.consumers.execution.jax_solvers import BFGS, SLSQP, Newton, Picard
from ..relational.consumers.execution.jax_solvers.canonical_nlp import (
    Box, NotSquare, box, canonical,
)
from ..consumers.declaration import (
    Conventions, Declared, Description, Explicit, Shipped, Unimplemented, tags_of,
)
from ..consumers.declaration.interface import Declaring
from ..consumers.declaration.interface_types import PlaceWithValue, Tagged
from ..consumers.surface import Purpose, RolePlaces, Surface, core_role_places
from ..consumers.surface import interface_types as _surface_mints
from ..consumers.declaration.interface_types import Solve as _Solve
from ..relational import mints as _relational_mints
from ..ode import mints as _ode_mints
from ..consumers.surface.interface import Editing, Locator, Part
from ..consumers.visualization import (
    DARK, PYXDSM, STYLES, TERRACOTTA, PlainFormat, Style, by_namespace, draw as _draw,
    render as _render, render_html as _render_html, render_matrix_svg as _render_matrix_svg,
    render_xdsm_svg as _render_xdsm_svg, to_dot as _to_dot,
)
from ..consumers.visualization.latex import LatexFormatter


CONTRACTS = (*_RELATIONAL_CONTRACTS, *_ODE_CONTRACTS)
'''The default contracts, every shipped language's: an implementation answers a problem by the
first one it implements.'''

@choice(Part)
class ShippedSurface(RelationalDeclaring, OdeDeclaring, RelationalEditing, Architecting,
                     Driving, Running):
    '''The surface a study writes through: every shipped part, at the pair's defaults.'''

    copies      : PlaceToPlace = Copy()
    contracts   : tuple = CONTRACTS
    role_places : RolePlaces = core_role_places | RolePlaces(OrderedMap((
        (Times, _ode_mints.Times()), (Trajectory, _ode_mints.Trajectory()))))

    @property
    def closing(self) -> Closing:
        '''Each group closed by `copy = var`, the copy where `copies` says and the closure
        at `^closure.<the cycle's first member>`: one mint for a cut and a closing.'''
        return FixedPointClosing(self.copies, AtFirstMember(Closure()))


absolute = ShippedSurface(InFull(), Shipped(by_name=False))
'''No place strings: every place a `V` path or a `VarPath`.'''


def Scope(root: Var, key: type[GetAttrKey] | type[DictKey], by_name: bool = True
          ) -> ShippedSurface:
    '''The shipped surface, a place string one `key` under `root`, and a parameter `reads`
    omits read at its own name unless `by_name` is off.'''
    return absolute.located(Under(root, key)).reading(Shipped(by_name))


flat = Scope(V, DictKey)
'''A plain dict: `flat.function(f, 'y2')` writes `['y2']` and reads each parameter by name.'''


copy, closure, given, start, steps = (
    Purpose(m) for m in (Copy(), Closure(), _surface_mints.Given(), _surface_mints.Start(),
                         _surface_mints.Steps()))
solve = Purpose(_Solve())
cond, require, lower, upper, target = (
    Purpose(m) for m in (_relational_mints.Cond(), _relational_mints.Require(),
                         _relational_mints.Lower(), _relational_mints.Upper(),
                         _relational_mints.Target()))
rate, initial, span, times, trajectory = (
    Purpose(m) for m in (_ode_mints.Rate(), _ode_mints.Initial(), _ode_mints.Span(),
                         _ode_mints.Times(), _ode_mints.Trajectory()))
'''Each shipped purpose, called on a name as written for the name derived from it:
`cx.copy('y_2')` is `^copy['y_2']`, `cx.solve('f')` the problem named after `f`,
`cx.given.beside('y', 'k')` a read written as a value.'''


def _drawn(model: Any, kw: Mapping[str, Any]) -> tuple[Any, dict]:
    '''What a drawing draws of `model` (a `Declared`'s plan), and its arguments with the
    plain formatter unless one is given.'''
    return (model.plan if isinstance(model, Declared) else model,
            {'formatter': PlainFormat(), **kw})


def render_html(model, **kw):
    '''Visualization's `render_html` of a graph, a proof, a `Plan` or a `Declared` (its plan,
    stepped op by op), written plainly unless `formatter=` says otherwise
    (`LatexFormatter(names)`).'''
    model, kw = _drawn(model, kw)
    return _render_html(model, **kw)


def render_xdsm_svg(model, **kw):
    '''Visualization's `render_xdsm_svg`, as `render_html` takes its model and formatter.'''
    model, kw = _drawn(model, kw)
    return _render_xdsm_svg(model, **kw)


def render_matrix_svg(model, **kw):
    '''Visualization's `render_matrix_svg`, as `render_html` takes its model and formatter.'''
    model, kw = _drawn(model, kw)
    return _render_matrix_svg(model, **kw)


def draw(model, **kw):
    '''Visualization's `draw`, as `render_html` takes its model and formatter.'''
    model, kw = _drawn(model, kw)
    return _draw(model, **kw)


def to_dot(model, **kw):
    '''Visualization's `to_dot`, as `render_html` takes its model and formatter.'''
    model, kw = _drawn(model, kw)
    return _to_dot(model, **kw)


def render(model, **kw):
    '''Visualization's `render`, as `render_html` takes its model and formatter.'''
    model, kw = _drawn(model, kw)
    return _render(model, **kw)


__all__ = [
    'V', 'N', 'Description', 'Unimplemented',
    'Explicit', 'Implicit', 'Residual', 'FixedPoint', 'PlaceWithValue', 'InequalityConstraint',
    'EqualityConstraint', 'Problem', 'Tagged',
    'Surface', 'Part', 'Declaring', 'Editing', 'RelationalDeclaring', 'RelationalEditing',
    'Architecting', 'Driving', 'ShippedSurface', 'Drive', 'RolePlaces', 'core_role_places',
    'Mint', 'Copy', 'Closure', 'Purpose',
    'copy', 'closure', 'given', 'start', 'steps', 'solve', 'cond', 'require', 'lower', 'upper',
    'target', 'rate', 'initial', 'span', 'times', 'trajectory',
    'Locator', 'Conventions', 'Shipped', 'PytreeLocator', 'Under', 'InFull',
    'Scope', 'flat', 'absolute', 'Declared', 'tags_of',
    'Static', 'DictKey', 'GetAttrKey', 'SequenceKey',
    'FixedPointClosing', 'FixedPointCut', 'AtFirstMember',
    'GaussSeidel', 'GaussSeidelMinimal', 'GaussSeidelOrdered', 'Jacobi', 'NoCuts', 'ByName',
    'CloseFunctionalCycles', 'Architecture', 'Monolithic', 'Coupled', 'MultidisciplinaryFeasible',
    'IndividualFeasible', 'Simultaneous', 'Analysis', 'MDF', 'IDF', 'SAND', 'MDA', 'Boxes',
    'Bodies', 'Disciplines', 'Solves', 'Resolution',
    'Eq', 'Le', 'Min', 'Max', 'Objective', 'Relation', 'RelationalCondition',
    'RootFinder', 'FixedPointIterator', 'Minimiser', 'ConstrainedMinimiser',
    'RootFinding', 'FixedPointIteration', 'Minimisation', 'ConstrainedMinimisation', 'CONTRACTS',
    'OdeDeclaring', 'Rate', 'Dt', 'OdeCondition', 'Integration', 'Integrator',
    'Times', 'Trajectory',
    'Newton', 'Picard', 'BFGS', 'SLSQP', 'canonical', 'box', 'Box', 'NotSquare', 'JitGraph',
    'Running', 'Results', 'Inputs', 'Traced', 'Start', 'Driven', 'CallableGraph', 'completed',
    'PlainFormat', 'LatexFormatter', 'render_xdsm_svg', 'render_matrix_svg', 'by_namespace',
    'render_html', 'draw', 'render', 'to_dot', 'Style', 'STYLES', 'TERRACOTTA', 'PYXDSM', 'DARK',
]


def __getattr__(name: str) -> Any:
    '''`Diffrax`, imported when asked for: it needs the `ode` extra.'''
    if name == 'Diffrax':
        from ..ode.consumers.execution import Diffrax
        return Diffrax
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
