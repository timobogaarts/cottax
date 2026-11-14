'''
Edits, over a `NestingGraph`: `algebra.py` is what an edit is as a record -- `GraphOp`,
`ComposedOp`, `StagedOp`, `Plan`; `primitive.py` the primitives, each
building a checked graph -- `Rebind` and its finals, `Nest`, `Unnest`; and `ops.py` the ops on `Redefine`,
every edit of what a node reads and owns. Nothing below reads a plan.
'''

from .algebra import ComposedOp, GraphOp, Plan, Stage, StagedOp, refusals
from .primitive import (
    Primitive, Rebind, Rebinding, Combine, Delete, Insert, Nest, Redefine, Unnest,
)
from .ops import Cut, Rename, Replace, Rewire, SetOwnsInside
