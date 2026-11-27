'''
Edits over a Graph. 

- Edits are the atomic: changed bindings and nestings
- GraphOps provide conditional edits depending on the graph (and checks that they are valid as intended)
- Plan is record of a base graph + applied ops.
'''

from .edit import Edit
from .op import Composite, GraphOp, PrimitiveOp, in_turn
from .plan import Plan
from .primitive_ops import (Contract, Cut, Delete, Insert, Nest, RemoveAttribute, Rename, Replace,
                            SetAttribute, Unnest)
