'''
MDAO architectures as one op each, over the cut rules and a `Closing` handed in:

    Plan(g) + MDF(GaussSeidel(), cx.closing, opt) + Assign(...)
    Plan(g) + MDA(GaussSeidel(), cx.closing)                       # no optimiser: an analysis
    Plan(g) + CloseFunctionalCycles(GaussSeidel(), cx.closing)     # a bare cut, no architecture
'''

from abc import abstractmethod
from dataclasses import field
from collections.abc import Generator, Iterator, Sequence
from typing import Any, ClassVar

from ...core import (
    Combine, CutRule, FunctionalNode, Label, NestingGraph, Opened, Place, ToCut, Violation,
    is_problem, cuts_per_cycle, requirements, GraphOp, Nest, StagedOp, refusals,
)
from ...core.driven import condition_of
from ...core.frozen import AbstractVar, Frozen, OrderedMap, answer, first
from .closing import closed
from .interface import Closing, ModelReading
from .refusals import BodyShared, NestedAlready, NotCombinable, TopNotAProblem, TopUnbound


# ---------------------------------------------------------------- what a problem is, asked

def not_implicit[N: Label](graph: NestingGraph[N, Any], problem: N) -> tuple[Violation, ...]:
    '''Why the problem is not an implicit model, by its `ModelReading`; refused if none.'''
    return tuple(answer(ModelReading, condition_of(graph[problem])).not_implicit(graph, problem))


def body_of_problem[N: Label](graph: NestingGraph[N, Any], problem: N) -> tuple[N, ...]:
    '''The function nodes that are the problem's own, by its `ModelReading`.'''
    return answer(ModelReading, condition_of(graph[problem])).body(graph, problem)


def combines_as(node: Any) -> type | None:
    '''The kind `node` folds as, `None` where it folds into nothing or nobody answers.'''
    found = answer(ModelReading, condition_of(node), None)
    return None if found is None else found.combines_as(condition_of(node))


def _spelled(kind: type | None) -> str:
    return 'nothing' if kind is None else kind.__name__


# ---------------------------------------------------------------- one problem inside another


def lies_inside[N: Label](graph: NestingGraph[N, Any], inner: N, outer: N,
                          bodies: dict[N, tuple[N, ...]]) -> bool:
    '''`inner` and its whole body on `outer`'s cycle, and its body part of `outer`'s, which
    holds more: it can be placed nowhere but inside `outer`.'''
    own, other = set(bodies[inner]), set(bodies[outer])
    return own < other and {inner, *own} <= set(graph.cycle_of(outer))


def inside[N: Label](graph: NestingGraph[N, Any], bodies: dict[N, tuple[N, ...]]) -> OrderedMap[N, N]:
    '''Each problem lying wholly inside another's body, and the least such other, the one it
    is nested in; in binding order. The pairs are a forest, a body growing up each chain.'''
    out = {}
    for q in bodies:
        outers = [p for p in bodies if p != q and lies_inside(graph, q, p, bodies)]
        if least := [p for p in outers if all(set(bodies[p]) <= set(bodies[o]) for o in outers)]:
            out[q] = least[0]
    return OrderedMap(out)


def chain[N: Label](pairs: OrderedMap[N, N], p: N) -> tuple[N, ...]:
    '''`p` and every problem it lies inside (`inside`'s pairs), inner to outer.'''
    out = [p]
    while out[-1] in pairs:
        out.append(pairs[out[-1]])
    return tuple(out)


def shared_bodies[N: Label](graph: NestingGraph[N, Any], bodies: dict[N, tuple[N, ...]]
                            ) -> Iterator[Violation]:
    '''Two problems sharing a body node, neither inside the other at any depth.'''
    pairs = inside(graph, bodies)
    above = lambda p: set(chain(pairs, p)[1:])
    claimed: dict = {}
    for p, body in bodies.items():
        for b in body:
            if apart := [q for q in claimed.get(b, ()) if q not in above(p) and p not in above(q)]:
                q = apart[-1]
                kind = combines_as(graph[q])
                remedy = ('`Combine` them first' if kind is not None and combines_as(graph[p]) is kind
                          else 'neither lies inside the other, so neither nests in it: '
                               'write them as one model')
                yield BodyShared(str(q), str(p), str(b), remedy)
            claimed.setdefault(b, []).append(p)


def folded_as[N: Label](op: 'Architecture', graph: NestingGraph[N, Any], into: N,
                        nodes: Sequence[N]) -> type:
    '''The kind `into` combines as, every one of `nodes` combining as it; else refused.'''
    kind = combines_as(graph[into])
    for n in nodes:
        if (found := combines_as(graph[n])) is None or found is not kind:
            raise NotCombinable(str(n), _spelled(found), repr(op), str(into)).raised
    return kind


def combined[N: Label](op: 'Architecture', graph: NestingGraph[N, Any], into: N,
                       nodes: Sequence[N]) -> tuple[GraphOp, ...]:
    '''`nodes` combined into `into`, their first, as `folded_as` says: one `Combine`, or
    none for one node.'''
    return (Combine(into, nodes, folded_as(op, graph, into, nodes)),) if nodes[1:] else ()


# ---------------------------------------------------------------- what a final reads, and what an op keeps


class OnCycle[N: Label](Frozen):
    '''The problems on one cycle of the scope, as a final reads them.'''

    closures : tuple[N, ...]
    '''The ones the cutting bound, in binding order.'''
    declared : tuple[N, ...]
    '''The ones the model declared, but those nested inside another's body, in binding order.'''


class Resolution[N: Label, V: Place](Frozen):
    '''
    Architecture applied to a graph in terms of
    GraphOps and the intermediate graphs.
    Useful for extracting which closures were inserted.
    '''

    absorbed  : tuple[GraphOp[N, V], ...]
    '''The requirements the design reaches, combined into the top: one `Combine`, or none.'''
    outside   : tuple[GraphOp[N, V], ...]
    '''What resolved the cycles off the scope's: the analysis's closures and placing.'''
    closures  : tuple[GraphOp[N, V], ...]
    '''What the cutting bound, one closure per group: what the `Closing` returned.'''
    opened    : NestingGraph[N, V]
    '''The graph after those two and before any problem is placed.'''
    combining : tuple[GraphOp[N, V], ...]
    '''The ops placing the problems: each lying inside another's body nested there, then the
    final's own, placing the closures and the declared problems.'''
    graph     : NestingGraph[N, V]
    '''The result.'''

# ---------------------------------------------------------------- which solves are boxes

class Boxes[N: Label, V: Place](Frozen):
    '''Which problems of the graph handed are seen as one function each.'''

    @abstractmethod
    def folded(self, graph: NestingGraph[N, V]) -> tuple[N, ...]:
        '''The problems to fold, in binding order.'''


# ---------------------------------------------------------------- the skeleton


def check_nothing_nested(op: 'Architecture', graph: NestingGraph) -> Iterator[Violation]:
    if graph.within:
        yield NestedAlready(tuple((str(i), str(o)) for i, o in graph.within.items()))


def check_implicit_models(op: 'Architecture', graph: NestingGraph) -> Iterator[Violation]:
    '''
    - An implicit model in its own language (`ModelReading.not_implicit`); where the final
      combines what is declared, combining as the top does
    - No two bodies share a node, unless one problem lies wholly inside the other's body
      (`inside`), where it is nested
    Asked of the graph without the design, nothing nested.
    '''
    own = set(op.design)
    into = combines_as(graph[op.design[0]]) if op.design and op.design[0] in graph else None
    free = NestingGraph(graph.graph.subgraph(n for n in graph.nodes if n not in own), OrderedMap())
    bodies: dict = {}
    for p in free.problems:
        if refused := not_implicit(free, p):
            yield from refused
            continue
        if op.combines_declared and ((kind := combines_as(graph[p])) is None
                                     or into is not None and kind is not into):
            yield NotCombinable(str(p), _spelled(kind), repr(op),
                                ', '.join(map(str, op.design)) or 'a closure')
            continue
        bodies[p] = body_of_problem(free, p)
    yield from shared_bodies(free, bodies)


class Architecture[N: Label, V: Place](StagedOp[N, V]):
    '''
    An MDAO architecture, with a top (`Monolithic`) or without (`Coupled`).

    A `StagedOp`, since each stage is decided off the graph the one before leaves; like
    any composition it proves nothing of its own, every op it applies being checked.

    1. what the level absorbs is combined (`absorbing`), and every cycle then off the
       level's own is resolved (`outside`);
    2. the cycles of the level's `scope` are cut over the boxes the final names, and
       each group is closed with one problem, the **closure**;
    3. every problem lying wholly inside another's body is nested there, at every depth
       (unless the final combines declared problems, which then combines it as any); then
       every problem on each cycle of the scope is placed as the final's `combining` says,
       the closures apart from the problems the model **declared**.
    '''

    rule    : CutRule[N, V]
    '''What is cut on each cycle: the MDA method.'''
    closing : Closing[N, V]
    '''How a group is closed: the op, and where its problem binds.'''

    boxes : AbstractVar[Boxes[N, V]]
    '''Which solves in the scope the rule sees as one function each.'''
    design : AbstractVar[tuple[N, ...]]
    '''The problems that are the level's own and not models: the top, or none.'''
    combines_declared : ClassVar[bool] = True
    '''Whether the final may `Combine` a declared problem into another, so each must
    combine as the top does; a level that only nests them says `False`, and then takes a
    problem combining as nothing.'''

    checks_graph = (check_nothing_nested, check_implicit_models)

    @abstractmethod
    def scope(self, graph: NestingGraph[N, V]) -> NestingGraph[N, V]:
        '''The graph the cutting sees, and whose cycles are placed.'''

    @abstractmethod
    def outside(self, graph: NestingGraph[N, V]) -> tuple[GraphOp[N, V], ...]:
        '''What resolves the cycles the scope leaves out, once absorbed.'''

    @abstractmethod
    def absorbing(self, graph: NestingGraph[N, V]) -> tuple[GraphOp[N, V], ...]:
        '''What is combined before anything is cut.'''

    @abstractmethod
    def combining(self, opened: NestingGraph[N, V], closures: Sequence[N]
                  ) -> Sequence[GraphOp[N, V]]:
        '''The ops placing what lies in the scope, read off `opened`, the graph the
        cutting leaves; `closures` is the problems it bound.'''

    def stages(self, graph: NestingGraph[N, V]
               ) -> Generator[Sequence[GraphOp[N, V]], NestingGraph[N, V], None]:
        '''Absorb; resolve what then lies outside; cut and close the scope of what is
        left; place every problem on its cycles, the closures being the problems the
        cutting bound.'''
        absorbed = yield self.absorbing(graph)
        resolved = yield self.outside(absorbed)
        scope    = self.scope(resolved)
        opened   = yield closed(cuts_per_cycle_boxed(scope, self.boxes, self.rule), self.closing)
        nested   = yield self.nesting_inside(scope)
        yield self.combining(nested, tuple(p for p in opened.problems if p not in set(resolved.problems)))

    def nesting_inside(self, scope: NestingGraph[N, V]) -> tuple[GraphOp[N, V], ...]:
        '''Each problem of the scope lying wholly inside another's body nested there, read
        before anything is cut; none where the final combines declared problems.'''
        if self.combines_declared:
            return ()
        pairs = inside(scope, {p: body_of_problem(scope, p) for p in scope.problems})
        return tuple(Nest(i, o) for i, o in pairs.items())

    def resolution(self, graph: NestingGraph[N, V]) -> Resolution[N, V]:
        '''The whole derivation: each stage's ops and the graph it leaves. Refused as
        `apply` refuses.'''
        first(refusals(self, graph))
        absorbed, outside, closures, nested, combining = self.staged(graph)
        return Resolution(absorbed.ops, outside.ops, closures.ops, closures.graph,
                          (*nested.ops, *combining.ops), combining.graph)

    def cycles_under(self, opened: NestingGraph[N, V], closures: Sequence[N]
                     ) -> tuple[OnCycle[N], ...]:
        '''Per cycle of the scope in `opened`, the problems on it: the closures the
        cutting bound and the ones the model declared, but those already nested.'''
        bound = set(closures)
        under = self.scope(opened)
        out = []
        for cycle in under.graph.cycles:
            problems = [p for p in cycle if is_problem(under[p]) and p not in under.within]
            out.append(OnCycle(tuple(p for p in problems if p in bound),
                               tuple(p for p in problems if p not in bound)))
        return tuple(out)


# ---------------------------------------------------------------- the two scopes


def check_top_bound(op: 'Monolithic', graph: NestingGraph) -> Iterator[Violation]:
    if op.top not in graph.definitions:
        yield TopUnbound(str(op.top))


def check_top_is_problem(op: 'Monolithic', graph: NestingGraph) -> Iterator[Violation]:
    if op.top in graph and not is_problem(graph[op.top]):
        yield TopNotAProblem(str(op.top))


def check_top_combinable(op: 'Monolithic', graph: NestingGraph) -> Iterator[Violation]:
    '''The top is combined with what it absorbs and, by the final, with closures or
    declared problems: it combines as some kind, and what it absorbs as the same.'''
    if op.top in graph and is_problem(graph[op.top]):
        kind = combines_as(graph[op.top])
        for n in (op.top, *reached_requirements(graph, op.top)):
            if (found := combines_as(graph[n])) is None or kind is not None and found is not kind:
                yield NotCombinable(str(n), _spelled(found), repr(op), str(op.top))


def reached_requirements[N: Label, V: Place](graph: NestingGraph[N, V], top: N) -> tuple[N, ...]:
    '''Every requirement whose conditions the design of `top` reaches.'''
    reach  = set(graph.graph.reach(graph[top].unknowns))
    owners = graph.graph.owners
    return tuple(r for r in requirements(graph.definitions)
                 if all(owners.get(c) in reach for c in graph[r].conditions))


class Monolithic[N: Label, V: Place](Architecture[N, V]):
    '''Under a top: its interior is the scope, every requirement the design reaches is
    combined into it first, and every cycle then off the top's is resolved by an
    analysis.'''

    top : N
    '''The problem the design lives in. Required: nothing searches for an optimiser.'''
    analysis : 'Analysis[N, V] | None' = field(default=None, kw_only=True)
    '''What resolves every cycle off the top's; `None` is an `MDA` with this rule and
    closing.'''

    checks_graph = (check_top_bound, check_top_is_problem, check_top_combinable)

    @property
    def design(self) -> tuple[N, ...]:
        return (self.top,)

    def scope(self, graph: NestingGraph[N, V]) -> NestingGraph[N, V]:
        return graph.interior(self.top)

    def outside(self, graph: NestingGraph[N, V]) -> tuple[GraphOp[N, V], ...]:
        '''The analysis over the graph with the top's cycle taken out, its ops applied
        to the whole: a cycle off the top's is a component of its own.'''
        on  = set(graph.cycle_of(self.top))
        off = graph.subgraph(n for n in graph.nodes if n not in on)
        if not off.nodes:
            return ()
        r = (self.analysis or MDA(self.rule, self.closing)).resolution(off)
        return (*r.absorbed, *r.outside, *r.closures, *r.combining)

    def absorbing(self, graph: NestingGraph[N, V]) -> tuple[GraphOp[N, V], ...]:
        '''Every requirement whose conditions the design of `top` reaches, combined
        into `top`: one `Combine`, or nothing.'''
        found = (self.top, *reached_requirements(graph, self.top))
        return combined(self, graph, self.top, found)

    def _repr(self) -> str:
        analysis = '' if self.analysis is None else f', analysis={self.analysis!r}'
        return (f'{type(self).__name__.lower()}({self.boxes!r}, {self.rule!r}, {self.closing!r}, '
                f'{self.top}{analysis})')


class Coupled[N: Label, V: Place](Architecture[N, V]):
    '''No top: the whole graph is the scope and nothing is absorbed. A requirement is
    left where it is, for the proof to refuse.'''

    design = ()

    def scope(self, graph: NestingGraph[N, V]) -> NestingGraph[N, V]:
        return graph

    def outside(self, graph: NestingGraph[N, V]) -> tuple[GraphOp[N, V], ...]:
        return ()

    def absorbing(self, graph: NestingGraph[N, V]) -> tuple[GraphOp[N, V], ...]:
        return ()

    def _repr(self) -> str:
        return f'{type(self).__name__.lower()}({self.boxes!r}, {self.rule!r}, {self.closing!r})'


# ---------------------------------------------------------------- the combinings


class MultidisciplinaryFeasible[N: Label, V: Place](Monolithic[N, V]):
    '''Per cycle, the closures folded into one MDA nested in `top`, and each declared
    problem nested in that MDA -- or in `top`, on a cycle nothing was bound on.'''

    combines_declared = False

    def combining(self, opened, closures):
        return [op for cycle in self.cycles_under(opened, closures)
                for op in one_mda(self, opened, cycle, self.top)]


class IndividualFeasible[N: Label, V: Place](Monolithic[N, V]):
    '''`top` owns the coupling copies and drives the gaps to zero; each declared
    problem is a solve of its own inside its iteration.'''

    combines_declared = False

    def combining(self, opened, closures):
        cycles = self.cycles_under(opened, closures)
        bound  = (self.top, *(c for cycle in cycles for c in cycle.closures))
        return (*combined(self, opened, self.top, bound),
                *(Nest(p, self.top) for cycle in cycles for p in cycle.declared))


class Simultaneous[N: Label, V: Place](Monolithic[N, V]):
    '''`top` absorbs every problem under it.'''

    def combining(self, opened, closures):
        problems = (self.top, *(p for cycle in self.cycles_under(opened, closures)
                                  for p in (*cycle.closures, *cycle.declared)))
        return combined(self, opened, self.top, problems)


class Analysis[N: Label, V: Place](Coupled[N, V]):
    '''Per cycle, the closures folded into one, and each declared problem nested in it.
    A cycle nothing was bound on is left as it is: its one problem closes it.'''

    combines_declared = False

    def combining(self, opened, closures):
        return [op for cycle in self.cycles_under(opened, closures)
                for op in one_mda(self, opened, cycle, None)]


def one_mda[N: Label](op: Architecture, opened: NestingGraph[N, Any], cycle: OnCycle[N],
                      within: N | None) -> tuple[GraphOp, ...]:
    '''A cycle's closures folded into its first, the MDA, nested in `within` if given, and
    each declared problem nested in the MDA; with no closure, each nested in `within`, or
    left as it is.'''
    if not cycle.closures:
        return () if within is None else tuple(Nest(p, within) for p in cycle.declared)
    mda = cycle.closures[0]
    return (*combined(op, opened, mda, cycle.closures),
            *(() if within is None else (Nest(mda, within),)),
            *(Nest(p, mda) for p in cycle.declared))

# ---------------------------------------------------------------- boxes cyclic


def cuts_per_cycle_boxed[N: Label, V: Place](graph: NestingGraph[N, V], boxes: Boxes[N, V],
                                             rule: CutRule[N, V]) -> tuple[Opened[N, V], ...]:
    '''
    `cuts_per_cycle` over the graph with each selected problem contracted with its
    body (`NestingGraph.contract`), together with every problem lying inside it; a body no
    selected problem brings is contracted alone, so nothing inside a body is cut. Every
    box is mapped back to its members: on a cycle, its function nodes in binding order;
    as a reader, those that read the variable -- under `check_implicit_models` always its
    body, since a problem reads nothing from outside its box. In the graph's own names.
    '''
    bodies = {p: body_of_problem(graph, p) for p in graph.problems}
    pairs  = inside(graph, bodies)
    trees: dict = {}
    for p in graph.problems:
        trees.setdefault(chain(pairs, p)[-1], []).append(p)
    folded = set(boxes.folded(graph))
    blocks = tuple((*tree, *bodies[root]) if folded & set(tree) else bodies[root]
                   for root, tree in trees.items() if folded & set(tree) or len(bodies[root]) > 1)
    if not blocks:
        return cuts_per_cycle(graph, rule)
    members = {min(block, key=graph.nodes.index): block for block in blocks}    # a box is named after its first member
    return tuple((tuple(sorted((m for n in cycle for m in members.get(n, (n,))
                                if isinstance(graph[m], FunctionalNode)), key=graph.nodes.index)),
                  tuple({var: tuple(m for r in readers for m in members.get(r, (r,))
                                    if var in graph[m].reads)
                         for var, readers in group.items()}
                        for group in groups))
                 for cycle, groups in cuts_per_cycle(graph.contract(blocks), rule))


def per_functional_cycle_boxed[N: Label, V: Place](graph: NestingGraph[N, V], boxes: Boxes[N, V],
                                                   rule: CutRule[N, V]) -> tuple[ToCut[N, V], ...]:
    '''`cuts_per_cycle_boxed`'s groups, every cycle's in a row.'''
    return tuple(group for _, groups in cuts_per_cycle_boxed(graph, boxes, rule) for group in groups)

# ---------------------------------------------------------------- the boxes


class Bodies[N: Label, V: Place](Boxes[N, V]):
    '''None: the function nodes alone are seen.'''

    def folded(self, graph: NestingGraph[N, V]) -> tuple[N, ...]:
        return ()

    def __repr__(self) -> str:
        return 'bodies'


class Disciplines[N: Label, V: Place](Boxes[N, V]):
    '''Every problem of the graph handed -- a discipline under `check_implicit_models`,
    and under an architecture every implicit model in its scope.'''

    def folded(self, graph: NestingGraph[N, V]) -> tuple[N, ...]:
        return graph.problems

    def __repr__(self) -> str:
        return 'disciplines'


class Solves[N: Label, V: Place](Boxes[N, V]):
    '''These, by name.'''

    names : tuple[N, ...]

    def folded(self, graph: NestingGraph[N, V]) -> tuple[N, ...]:
        return self.names

    def __repr__(self) -> str:
        return f'solves({", ".join(map(str, self.names))})'


# ---------------------------------------------------------------- the four names


class MDF[N: Label, V: Place](MultidisciplinaryFeasible[N, V]):
    '''Multidisciplinary feasible over the disciplines.'''

    boxes = Disciplines()

    def __repr__(self) -> str:
        return self._repr()


class IDF[N: Label, V: Place](IndividualFeasible[N, V]):
    '''Individual discipline feasible over the disciplines.'''

    boxes = Disciplines()

    def __repr__(self) -> str:
        return self._repr()


class SAND[N: Label, V: Place](Simultaneous[N, V]):
    '''Simultaneous analysis and design over the bodies: the states are the
    optimiser's outright, so a solve's own loop needs no cut.'''

    boxes = Bodies()

    def __repr__(self) -> str:
        return self._repr()


class MDA[N: Label, V: Place](Analysis[N, V]):
    '''A multidisciplinary analysis over the disciplines: each implicit model keeps
    its own solver, and one closure per cycle iterates the states between them.'''

    boxes = Disciplines()

    def __repr__(self) -> str:
        return self._repr()
