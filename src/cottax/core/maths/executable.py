'''
The proof that a graph is executable: every entry, at every depth, is answered by one
encloser whose edges lie right against its cycle. A rule group (`ExecutableGraph`), proved
of a graph and remembered on it.
'''

from collections.abc import Iterator

from .dependencies import WellFormed, dependencies
from .graph import Graph
from .nesting import check_nesting_ends_on_one_cycle, nesting
from .kinds import AbstractConditionalNode, owns_inside
from .places import Label, Place
from ..frozen import Violation



def _problems(g: Graph) -> list:
    '''The nodes that iterate a cycle. Asked before `check_no_requirements` holds, so a
    requirement -- a condition node owning nothing inside -- is not one.'''
    return [n for n, d in g.definitions.items() if owns_inside(d)]

def body_of[G: Graph](g: G) -> G:
    '''
    The graph without its problems: what is run, never answered.
    '''
    ps = set(_problems(g))
    return g.subgraph(n for n in g.nodes if n not in ps)


# ---------------------------------------------------------------- the rules: what an executable graph obeys

def check_nestings_share_a_cycle(g: Graph) -> Iterator[Violation]:
    '''
    Both ends of every nesting are on one cycle (`check_nesting_ends_on_one_cycle`): a pair
    that is not is inert to the graph, and here it would name an encloser answered inside
    an iteration that never re-runs it.

    p ~ q  in  N \\ A(q)   ∀ (p, q) ∊ W

    Asked first, and of the whole graph: the level rules below read `within`, and a pair
    off its outer's cycle hides its inner from them (it is neither outermost nor a sibling
    of what shares its cycle), so they pass a graph `steps` cannot read.
    '''
    yield from check_nesting_ends_on_one_cycle(g)


def check_no_requirements(g: Graph) -> Iterator[Violation]:
    '''
    Every condition node owns something inside: no requirement is left.

    N_P = N_C
    '''
    for n, d in g.definitions.items():
        if isinstance(d, AbstractConditionalNode) and not d.owns_inside:
            yield RequirementUnanswered(n, tuple(d.reads_inside))

def check_body_is_dag(g: Graph) -> Iterator[Violation]:
    '''
    With the problems taken out no cycle is left: *functionally acyclic*.

    |S_k| = 1  ∀ S_k ∊ SCC[G \\ N_P]
    '''
    for cycle in dependencies(body_of(g)).cycles:
        held = [p for p in _problems(g) if set(cycle) <= set(nesting(g).cycle_of(p))]
        if held:
            yield StillCyclic(max(held, key=lambda p: len(nesting(g).enclosing(p))), cycle)
        else:
            yield NoProblem(cycle)


def check_one_encloser_per_level(g: Graph) -> Iterator[Violation]:
    '''
    No two problems nested in the same thing -- both outermost, or both answered
    inside one encloser -- are on one cycle. 

    The mirror of `check_nesting_ends_on_one_cycle`: a nested pair *must* share a
    cycle at the outer's level, siblings *must not*.

    q ≁ q'  in  N \\ A(q)   ∀ q ≠ q' ∊ N_P with A(q) = A(q')

    Same parent (`within`) is the same A(·): in a forest, equal ancestor sets means
    equal chains.
    '''
    ps = set(_problems(g))
    for q in _problems(g):
        cycle    = nesting(g).cycle_of(q)
        siblings = tuple(p for p in cycle if p in ps and g.within.get(p) == g.within.get(q))
        if len(siblings) > 1 and siblings[0] == q:                  # said once, by the first
            yield SeveralOutermost(cycle, siblings)


def check_all_conditions_on_cycle(g: Graph) -> Iterator[Violation]:
    '''
    Every encloser iterates something, and every condition it holds is produced there:
    on its cycle.

    |S_p| > 1   and   C_p ⊆ ⋃_{k ∊ S_p} O_k   ∀ p ∊ N_P

    A condition nothing on the cycle produces is a constant to the encloser: no
    unknown can move it. A read nothing produces at all (a boundary input) is one.
    '''
    for p in _problems(g):
        cycle = nesting(g).cycle_of(p)
        d     = g.definitions[p]
        if len(cycle) == 1:
            yield OutsideCycle(p, tuple(d.owns_inside))
        elif constant := tuple(v for v in d.reads_inside if dependencies(g).owners.get(v) not in cycle):
            yield Constant(p, constant)


def check_other_reads_off_cycle(g: Graph) -> Iterator[Violation]:
    '''
    What an encloser reads outside is not produced on its cycle.

    I^out_p ∩ O(S_p) = ∅   ∀ p ∊ N_P
    '''
    for p in _problems(g):
        d    = g.definitions[p]
        held = set(nesting(g).cycle_of(p))
        if inside := tuple(v for v in d.reads_outside if dependencies(g).owners.get(v) in held):
            yield ReadInsideCycle(p, inside)


def check_other_owns_off_cycle(g: Graph) -> Iterator[Violation]:
    '''
    What an encloser owns outside is not read on its cycle.
    O^out_p ∩ I(S_p) = ∅   ∀ p ∊ N_P
    '''
    readers = g.readers
    for p in _problems(g):
        d    = g.definitions[p]
        held = set(nesting(g).cycle_of(p))
        if inside := [(v, r) for v in d.owns_outside
                      for r in readers.get(v, ()) if r in held]:
            yield OwnReadInsideCycle(p, tuple(v for v, _ in inside), tuple(r for _, r in inside))


class ExecutableGraph(WellFormed):
    '''A well-formed graph that can be run: every cycle answered by one problem, the rest
    functionally acyclic.'''
    rules = (check_nestings_share_a_cycle,
             check_no_requirements, check_one_encloser_per_level, check_body_is_dag,
             check_all_conditions_on_cycle, check_other_reads_off_cycle,
             check_other_owns_off_cycle)


def outline(g: Graph) -> str:
    '''The top level in one line: each component in dependency order, a component
    answered by an outermost encloser as `solve(problem)[members]`, one node bare.'''
    outer = set(nesting(g).outermost)
    def one(component) -> str:
        members = ', '.join(str(n) for n in component)
        if held := [p for p in component if p in outer]:
            return f'solve({", ".join(str(p) for p in held)})[{members}]'
        return f'[{members}]' if len(component) > 1 else members
    return ' -> '.join(one(c) for c in dependencies(g).components) or '-'


# ----------------------------------------------------------------- violation finals

class RequirementUnanswered(Violation):
    node       : Label
    conditions : tuple[Place, ...]
    error    = ValueError
    template = ('{node!r} holds {conditions} against nothing: it has no unknowns, so no driver '
                 'can move it -- determine it by a variable on its cycle, or drop it')

class SeveralOutermost(Violation):
    cycle    : tuple[Label, ...]
    problems : tuple[Label, ...]
    error    = ValueError
    template = ('cycle {cycle} declares several problems ({problems}) with nothing saying which '
                 'is outer -- one driver answers one problem, so `Combine` them into a single '
                 'problem over every unknown, or `Nest` the others inside one of them. Which is a modelling '
                 'decision, not something the blocking can read off the graph')

class NoProblem(Violation):
    cycle : tuple[Label, ...]
    error    = ValueError
    template = ('coupled cycle {cycle} declares no problem, so it has no conditions and there '
                 'is nothing to hand a driver -- `Cut` a variable in the cycle and drive the '
                 'iteration from outside, or declare the problem to drive within the cycle')

class StillCyclic(Violation):
    problem : Label
    cycle   : tuple[Label, ...]
    error    = ValueError
    template = ('inside {problem!r}\'s iteration, {cycle} is still cyclic with no problem on '
                 'it, so there is no condition map to build for it. Cut a variable and pick a '
                 'driver (inside / outside), or nest a problem that answers it in {problem!r}')

class OutsideCycle(Violation):
    problem  : Label
    unknowns : tuple[Place, ...]
    error    = ValueError
    template = ('{problem!r} has no body and is in no cycle at its level: nothing it reads '
                 'depends on {unknowns}, so no iteration can determine them -- a problem is '
                 'answered by a condition computed from its unknowns (e.g. r_u(u) = 0), not '
                 'by a value')

class Constant(Violation):
    problem : Label
    places  : tuple[Place, ...]
    error    = ValueError
    template = ('{problem!r} reads {places} as a condition, but nothing in its cycle produces '
                 'it, so no unknown of the problem can move it -- a constant is not a condition; '
                 'compute it from the unknowns, or drop it')

class ReadInsideCycle(Violation):
    problem : Label
    places  : tuple[Place, ...]
    error    = ValueError
    template = ('{problem!r} reads {places} without holding against it, and it is produced '
                 'inside the block it drives -- a driver reads its data before the block runs, '
                 'so it must come from outside (an earlier block, or a boundary input)')

class OwnReadInsideCycle(Violation):
    problem : Label
    places  : tuple[Place, ...]
    readers : tuple[Label, ...]
    error    = ValueError
    template = ('{problem!r} owns {places} without answering it, and it is read by {readers} '
                 'inside the block it drives -- a report is bound after convergence, so a '
                 'reader on the cycle would iterate against a value that does not exist yet')
