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
conditional/   the condition node as data: a condition, and its driver attached
driven/        what a driver is: contracts, roles; assigning one
realisation/   what runs a node: implementations

`from cottax.core import X` is the names of `maths/`, `mints`, `answers`, `plan/` and `conditional/`, and
`frozen`'s `Violation`, `first`, `violations`; `driven/` and `realisation/` are imported
from their own doors.
'''

from .frozen import Violation, first, violations
from .maths import (
    Label, Place, PlaceToPlace, comparable, comparables_in,
    NodeDefinition, Renaming, relabelled, Attribute,
    AbstractConditionalNode, FunctionalNode, conditions, owns_inside,
    Graph,
    Rules, each_node, is_proved, proved, proves, prove, unproved,
    Dependencies, WellFormed, dependencies, Nesting, nesting,
    ExecutableGraph, body_of, outline,
    Run, Solve, Step, steps,
)
from .mints import Closure, Copy, Mint, Minting, mint_of, minted, unminted
from .answers import answer_for_labels, answer_for_places
from .plan import (
    Plan, ComposedOp, GraphOp, StagedOp, refusals,
    Rebind, Rebinding, Cut, Delete, Insert, Nest, Redefine, Rename, Replace, Rewire,
    SetAttribute, RemoveAttribute,
    Unnest,
)
from .conditional import (
    Attachment, Condition, ConditionalNode, SetOwnsInside, is_driven, kind_of,
)
