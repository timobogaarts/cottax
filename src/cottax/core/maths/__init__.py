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
rules                        (Rules, per_node_check: what is proved of a graph, as classes;
 ||                           Group.check / Group.prove, the answer remembered on the graph)
 \/
valid                        (Valid: ValidNesting, the nesting's three rules, and
 ||                           ValidDependencies, the bindings' two -- each view asks its own)
 \/
dependencies                 (`Dependencies.of(graph)`, the view of owners, edges, components,
 ||                           order, reachability)
 \/
nesting                      (`Nesting.of(graph)`, the view of the outermost enclosers, what each
 ||                           is nested in, its cycle and interior)
 \/
executable                   (the structural proof, a group implying Valid: cycles,
 ||                           enclosers, and every encloser's edges against its cycle; outline)
 \/
steps                        (`Schedule.of(graph)`, the view of the steps in order: a Run per
                              node, a Loop per cycle with its interior proved executable)

'''

from .places import (
    Label, Place, PlaceToPlace, comparable, comparables_in,
)
from .definition import (
    Attribute, NodeDefinition, Renaming, relabelled,
)
from .kinds import AbstractConditionalNode, FunctionalNode, conditions, enclosers, owns_inside
from .graph import Graph, View
from .rules import Rules, per_node_check
from .valid import Valid, ValidDependencies, ValidNesting
from .dependencies import Dependencies
from .nesting import Nesting
from .executable import Executable, body, outline
from .steps import Loop, Run, Schedule, Step
