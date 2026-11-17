r'''
The core, in layers, each importing only those before it:

frozen/        the frozen values, what a check yields, answers
maths/         the graph, and what can be said of it alone: places, node definitions, the
               two kinds as the rules read them, nesting, the executable proof, the steps
mints          an interface over the places: a purpose a name is derived for (`Mint`), and
               how a class of names wears one (`Minting`), answered by the places
plan/          edits of a graph, as records
conditional/   the condition node as data: a condition, and its driver attached
driven/        what a driver is: contracts, roles; assigning one
realisation/   what runs a node: implementations

`from cottax.core import X` is the names of `maths/`, `mints`, `plan/` and `conditional/`, and
`frozen`'s `Violation`, `first`, `violations`; `driven/` and `realisation/` are imported
from their own doors.
'''

from .frozen import Violation, first, violations
from .maths import (
    Label, Place, PlaceToPlace, comparable, comparables_in, prefixes,
    NodeDefinition, Renaming, relabelled, Attribute, misfits, with_attributes, without_attribute,
    AbstractConditionalNode, FunctionalNode, is_encloser, enclosers,
    AbstractGraph,
    NestingGraph, ProvedGraph,
    answer_for_labels, answer_for_places,
    GraphWithValues,
    ExecutableGraph, body_of,
    Run, Solve, Step, SteppedExecutableGraph,
)
from .mints import Closure, Copy, Mint, Minting, mint_of, minted, unminted
from .plan import (
    Plan, ComposedOp, GraphOp, StagedOp, refusals,
    Rebind, Rebinding, Cut, Delete, Insert, Nest, Redefine, Rename, Replace, Rewire,
    SetAttribute, RemoveAttribute,
    Unnest,
)
from .conditional import (
    Attachment, Condition, ConditionalNode, SetOwnsInside, is_driven, kind_of,
)
