r'''
The mathematics: the graph, and what can be said of it alone -- what a well-formed one is,
said without naming a concrete type. No edit of one (`plan/`), and no concrete condition
node (`conditional/`): a condition node is read through `AbstractConditionalNode`'s names.

Dependency:

places      frozen/          (places: Place, Label, PlaceToPlace, the prefix order;
 ||          ||               frozen/: the frozen values, what a check yields, `remembered`)
 \/          \/
definition <-'               (NodeDefinition; Attribute, what a node carries opaquely)
 ||
 \/
kinds                        (FunctionalNode; AbstractConditionalNode, what every rule reads
 ||                           of a condition node: its four names)
 \/
graph                        (Graph, the one stored graph: definitions and `within`, checked
 ||                           at nothing; what is read off the bindings)
 \/
rules                        (Rules, each_node: what is proved of a graph, as classes;
 ||                           unproved / prove / proved, a set remembered on the graph)
 \/
wellformed                   (WellFormed: the nesting's three rules and the bindings' two,
 ||                           what a readable graph obeys)
 \/
dependencies                 (`dependencies(graph)`, the view of owners, edges, components,
 ||                           order, reachability)
 \/
nesting                      (`nesting(graph)`, the view of the outermost problems, what each
 ||                           is nested in, its cycle and interior; `contract`)
 \/
executable                   (the structural proof, a group implying WellFormed: cycles,
 ||                           problems, and every problem's edges against its cycle; outline)
 \/
steps                        (steps(graph): Run and Solve, every interior proved
                              executable and holding the graph's node groups)

'''

from .places import (
    Label, Place, PlaceToPlace, comparable, comparables_in,
)
from .definition import (
    Attribute, NodeDefinition, Renaming, relabelled,
)
from .kinds import AbstractConditionalNode, FunctionalNode, conditions, owns_inside
from .graph import Graph
from .rules import Rules, each_node, is_proved, proved, proves, prove, unproved
from .wellformed import WellFormed
from .dependencies import Dependencies, dependencies
from .nesting import Nesting, nesting
from .executable import ExecutableGraph, body_of, outline
from .steps import Run, Solve, Step, steps
