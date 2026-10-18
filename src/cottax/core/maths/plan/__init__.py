'''
Edits, over a `NestingGraph`: `algebra.py` is what an edit is as a record -- `GraphOp`,
`ComposedOp`, `StagedOp`, `Plan`; `primitive.py` the primitives, each proved through
`rebound` -- `Rebind` and its finals, `Nest`, `Unnest`; and `ops.py` the ops on `Redefine`,
every edit of what a node reads, owns and determines. Nothing below reads a plan.
'''

from .algebra import ComposedOp, GraphOp, Plan, Stage, StagedOp, refusals
from .primitive import (
    Primitive, Rebind, Rebinding, Combine, Delete, Insert, Nest, Redefine, Unnest,
)
from .ops import Cut, Determine, Rename, Replace, Rewire, Undetermine
