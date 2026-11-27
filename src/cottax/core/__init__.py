r'''
The core, in layers, each importing only those before it:

frozen/        the frozen values, what a check yields, answers
maths/         the graph, and what can be said of it alone: places, node definitions, the
               two kinds as the rules read them, the stored graph, the rule groups, the
               dependency and nesting views, the executable proof, the steps
mints          an interface over the places: a purpose a name is derived for (`Mint`), and
               how a class of names wears one (`Minting`), answered by the places
answers        the one `Answer` for a graph, by its place class or its label class
plan/          edits of a graph, as records
conditional/   the condition node: a condition, the driver answering it, assigning one
realisation/   what runs a node: implementations

`from cottax.core import X` is the names of `maths/`, `mints`, `answers`, `plan/` and `conditional/`, and
`frozen`'s `Violation`, `first`, `violations`; the driver's vocabulary (`conditional/`) and
`realisation/` are imported from their own doors.
'''

from .frozen import Violation, first, violations
from .maths import (
    Label, Place, PlaceToPlace, comparable, comparables_in,
    NodeDefinition, Renaming, relabelled, Attribute,
    AbstractConditionalNode, FunctionalNode, conditions, enclosers, owns_inside,
    Graph, View, Box,
    Rules, per_node_check,
    Dependencies, Valid, ValidDependencies, ValidNesting, Nesting, Executable, body, outline,
    Enclosed, Run, Schedule, Step,
)
from .mints import Closure, Copy, Mint, Minting, mint_of, minted, unminted
from .answers import answer_for_labels, answer_for_places
from .plan import (
    Plan, GraphOp, PrimitiveOp, Composite, Edit, in_turn,
    Cut, Delete, Insert, Nest, Rename, Replace, SetAttribute, RemoveAttribute, Unnest,
    ComposedOp, StagedOp, Stage, Rebind, Rebinding, Redefine,   # legacy, phase 2 removes
)
from .conditional import (
    Condition, ConditionalNode, SetOwnsInside, is_driven, kind_of,
)
