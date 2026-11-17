r'''
The mathematics: the graph, and what can be said of it alone -- what a well-formed one is,
said without naming a concrete type. No edit of one (`plan/`), and no concrete condition
node (`conditional/`): a condition node is read through `AbstractConditionalNode`'s names.

Dependency:

places      frozen/          (places: Place, Label, PlaceToPlace, the prefix order;
 ||          ||               frozen/: the frozen values, and what a check yields)
 \/          \/
definition <-'               (NodeDefinition; Attribute, what a node carries opaquely)
 ||-----------
 \/          \/
kinds       graph            (kinds: FunctionalNode; AbstractConditionalNode, what every rule
 ||          ||               reads of a condition node: its four names;
 ||          ||               graph: the bindings and the theory over them; no kind named)
 \/          \/
nesting <----'               (has a graph, and the nesting over it; `contract`, which builds
                              a `FunctionalNode` per block; ProvedGraph, a proof's base)
 ||-------> dispatch         (the one `Answer` for a graph's places, or its node names)
 ||-------> valued           (GraphWithValues: a proved graph and a value at each input)
 ||
 \/
executable                   (the structural proof -- cycles, problems, and every
 ||                           problem's edges against its cycle)
 \/
steps                        (SteppedExecutableGraph: the steps, every interior the same
                              proof; a level below it checks each node)

'''

from .places import (
    Label, Place, PlaceToPlace, comparable, comparables_in, prefixes,
)
from .definition import (
    Attribute, NodeDefinition, Renaming, relabelled, misfits, with_attributes,
    without_attribute,
)
from .kinds import AbstractConditionalNode, FunctionalNode, is_encloser, enclosers
from .graph import AbstractGraph
from .nesting import NestingGraph, ProvedGraph
from .dispatch import answer_for_labels, answer_for_places
from .valued import GraphWithValues
from .executable import ExecutableGraph, body_of
from .steps import Run, Solve, Step, SteppedExecutableGraph
