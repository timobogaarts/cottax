'''
Edits, over a `Graph`: `edit.py` is `Edit`, the one builder of a graph, refusing where it does
not fit; `op.py` what an op is --
`GraphOp` (`apply`, `edits`), `PrimitiveOp`, `Composite`, `in_turn`; `plan.py` `Plan`;
`primitive_ops.py` the ops that are one edit. `legacy.py` is temporary. Nothing below reads a plan.
'''

from .edit import Edit
from .op import Composite, GraphOp, PrimitiveOp, in_turn
from .plan import Plan
from .primitive_ops import (Cut, Delete, Insert, Nest, RemoveAttribute, Rename, Replace,
                            SetAttribute, Unnest)
from .legacy import ComposedOp, Rebind, Rebinding, Redefine, Stage, StagedOp
