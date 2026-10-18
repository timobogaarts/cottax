r'''
The mathematics: what a well-formed value is, said without naming a concrete type.

Dependency:

places      frozen/          (places: Place, Label, PlaceToPlace, the prefix order;
 ||          ||               frozen/: the frozen values, and what a check yields)
 ||-------> mints            (Mint, a purpose a name is derived for; Minting, how a
 ||          ||               class of names wears one)
 \/          \/
definition <-'
 ||-----------
 \/          \/
kinds       graph            (graph: the bindings and the theory over them; no kind named)
 ||          ||
 \/          \/
nesting <----'               (has a graph, and the nesting over it; `contract`, which builds
                              a `Function` per block; ProvedGraph, a proof's base)
 ||-------> dispatch         (the one `Answer` for a graph's places, or its node names)
 ||-------> valued           (GraphWithValues: a proved graph and a value at each input)
 ||------------------------
 \/         \/            \/
plan/      cut_rules     executable      (plan/: algebra -> primitive -> ops, the edits as
                          ||              records; cut_rules: which couplings a cut opens,
                          ||              per functional cycle: CutRule and its finals;
                          ||              executable: the structural proof -- cycles,
                          \/              problems, and every problem's edges against its cycle)
                         steps           (SteppedExecutableGraph: the steps, every interior the
                                          same proof; a level below it checks each node)

'''

from .places import (
    Label, Place, PlaceToPlace, comparables_in, prefixes,
)
from .mints import Closure, Copy, Mint, Minting, mint_of, minted, unminted
from .definition import NodeDefinition, Renaming, relabelled
from .kinds import (
    ConditionalNode, Function, FunctionalNode, is_requirement, is_problem, requirements, problems,
)
from .graph import AbstractGraph
from .nesting import NestingGraph, ProvedGraph
from .dispatch import answer_for_labels, answer_for_places
from .valued import GraphWithValues
from .plan import (
    Plan, ComposedOp, GraphOp, StagedOp, refusals,
    Rebind, Rebinding, Combine, Cut, Delete, Determine, Insert, Nest, Redefine, Rename, Replace, Rewire,
    Undetermine, Unnest,
)
from .executable import ExecutableGraph, body_of
from .steps import Run, Solve, Step, SteppedExecutableGraph
from .cut_rules import (
    ByName, CutRule, GaussSeidel, GaussSeidelMinimal, GaussSeidelOrdered, Jacobi, NoCuts, Sweep,
    Opened, ToCut, coupling_reads, cuts_per_cycle, gauss_seidel, minimal_feedback_order,
    per_functional_cycle,
)
