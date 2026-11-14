'''
How a graph is read for a drawing: each strongly connected component as what it is
(`entries`), the rings a drawing draws (`walk`), and an order for its diagonal (`sequence`).
A graph that cannot be scheduled still has entries, so it is still drawn, its nestings
included.
'''

from collections.abc import Iterator, Sequence
from functools import cached_property

from ...core.frozen import AbstractVar, Frozen, OrderedMap
from ...core import AbstractGraph, ExecutableGraph, Plan, Label, NestingGraph, body_of


def graph_drawn(handed: NestingGraph | ExecutableGraph | Plan) -> NestingGraph:
    '''The graph a drawing draws of what it is handed: a `Plan`'s graph, a proof's graph,
    or the graph itself.'''
    if isinstance(handed, (Plan, ExecutableGraph)):
        return handed.graph
    if isinstance(handed, NestingGraph):
        return handed
    raise TypeError(f'a drawing draws a NestingGraph, an ExecutableGraph or a Plan; '
                    f'got {type(handed).__name__}')


# ---------------------------------------------------------------- what a component is

class Entry(Frozen):
    '''
    One strongly connected component of a graph -- a cycle, or a node on none -- as a
    drawing reads it, its kind the final (`Run`, `Cycle`, `Solve`, `Unresolved`). Built by
    `entries`.
    '''
    subgraph : NestingGraph
    held     : AbstractVar[NestingGraph | None]
    '''The part of the component drawn inside its ring and read as a graph of its own (its
    own entries, its own order); the nodes not in it are drawn on the ring itself. `None`
    where the component is drawn whole.'''

    @property
    def nodes(self) -> tuple[Label, ...]:
        '''The component, in the graph's binding order.'''
        return self.subgraph.nodes


class Run(Entry):
    '''One node, no problem: it runs.'''
    held = None


class Cycle(Entry):
    '''A cycle with no problem on it: nothing iterates it.'''
    held = None


class Solve(Entry):
    '''
    A component with exactly one outermost problem, which answers it. Inside its ring is
    its interior, the component without that problem (`NestingGraph.interior`), where the
    problem's children are the outermost problems.
    '''
    problem  : Label
    interior : NestingGraph

    @property
    def held(self) -> NestingGraph:
        return self.interior


class Unresolved(Entry):
    '''
    A cycle with several outermost problems: no single problem answers it, so it is drawn as
    one grey ring until a `Nest` or a `Combine` decides which does.
    '''
    problems : tuple[Label, ...]

    @cached_property
    def held(self) -> NestingGraph | None:
        '''
        The rest of the cycle once some of its outermost problems are removed. Those removed
        are drawn on the grey ring itself, and what they own is an input to the rest, so the
        cycles through them are broken and the rings inside can be drawn. Removed are the
        leaves among them (no children), so a problem with children keeps its ring; where
        none is a leaf, all of them, so their children keep theirs. A parent-child link whose
        cycle ran only through a removed problem stays, inert: its ends fall into different
        components of the rest, so nothing rings it. `None` where nothing is nested: the
        cycle is drawn whole.
        '''
        links = self.subgraph.within
        if not links:
            return None
        parents = set(links.values())
        removed = {p for p in self.problems if p not in parents} or set(self.problems)
        return self.subgraph.subgraph(n for n in self.nodes if n not in removed)


def entries(g: NestingGraph) -> tuple[Entry, ...]:
    '''
    Each component of `g` as an `Entry`, in dependency order, the members of each in
    binding order.
    '''
    out : list[Entry] = []
    for component in g.graph.components:
        sub   = g.subgraph(component)
        found = sub.outermost_enclosers
        if len(found) > 1:
            out.append(Unresolved(sub, found))
        elif found:
            problem, = found
            out.append(Solve(sub, problem, sub.interior(problem)))
        else:
            # a component of one node is acyclic (a node may not read what it owns);
            # of more it is a cycle, exactly
            out.append((Run if len(component) == 1 else Cycle)(sub))
    return tuple(out)


class Block(Frozen):
    '''
    One entry as a drawing rings it, the number of rings around it (`depth`; a grey ring
    counts, so a problem with no parent may sit one ring in), and whether it is `unsettled`:
    a `Solve` on a cycle whose problem is one of the outermost problems of the grey ring
    directly around it. Its ring could run as it stands, but its problem's place is part of
    that ring's open nesting decision, so it is drawn grey with a closed contour.
    '''
    entry     : Entry
    depth     : int
    unsettled : bool

    @property
    def nodes(self) -> tuple[Label, ...]:
        return self.entry.nodes

    @property
    def held(self) -> NestingGraph | None:
        '''What is ringed inside it: the entry's `held` where that holds a problem, else
        `None`, so a solve whose interior holds no problem rings nothing inside.'''
        held = self.entry.held
        return held if held is not None and held.enclosers else None

    @property
    def problem(self) -> Label | None:
        '''The problem answered at this block's level, or `None`.'''
        return self.entry.problem if isinstance(self.entry, Solve) else None

    @property
    def solved(self) -> bool:
        '''Whether it contains a cycle: a solve ring, else a run.'''
        return not self.entry.subgraph.graph.is_acyclic


def walk(graph: NestingGraph) -> tuple[Block, ...]:
    '''
    Every block a drawing rings, at every depth: the graph's entries, each followed by the
    blocks of what it holds, so an inner ring comes after (paints over) the ring around it.
    '''
    def blocks(g: NestingGraph, depth: int, open_: tuple[Label, ...]) -> Iterator[Block]:
        # `open_`: the outermost problems of the grey ring directly around `g`, if any
        for entry in entries(g):
            ringed = isinstance(entry, Solve) and not entry.subgraph.graph.is_acyclic
            block  = Block(entry, depth, ringed and entry.problem in open_)
            yield block
            if block.held is not None:
                yield from blocks(block.held, depth + 1,
                                  entry.problems if isinstance(entry, Unresolved) else ())
    return tuple(blocks(graph, 0, ()))


# ---------------------------------------------------------------- the order

def draws_feedback(order: Sequence[Label], subgraph: NestingGraph) -> bool:
    '''
    Whether some node in `order` reads a variable a later one owns: an edge drawn above
    the diagonal.
    '''
    at, owners = {name: i for i, name in enumerate(order)}, subgraph.graph.owners
    return any(at[owner] > i
               for i, name in enumerate(order)
               for var in subgraph[name].reads
               if (owner := owners.get(var)) is not None)


def sequence(graph: NestingGraph) -> tuple[Label, ...]:
    '''
    The graph's nodes in an order for one diagonal, entry by entry in dependency order. An
    entry holding something (`Entry.held`): the nodes drawn on its ring first (a `Solve`'s
    problem, an `Unresolved`'s removed problems), in binding order, then what it holds,
    sequenced the same way. Any other: its problems in binding order, then its body as bound
    unless that draws feedback, else in `components_order` (defined on a cycle too). So each
    ring is one contiguous stretch, and the only edges left above the diagonal run through a
    problem, or are a cycle no problem closes.
    '''
    out : list[Label] = []
    for entry in entries(graph):
        if entry.held is not None:
            out.extend(n for n in entry.nodes if n not in entry.held)
            out.extend(sequence(entry.held))
        else:
            sub          = entry.subgraph
            body         = body_of(sub)
            outside_body = [n for n in entry.nodes if n not in body.definitions]
            order        = [n for n in entry.nodes if n in body.definitions]
            if order and draws_feedback(order, body):
                order = list(body.graph.components_order)
            out.extend((*outside_body, *order))
    return tuple(out)


def ordered_graph(handed: NestingGraph | ExecutableGraph | Plan) -> NestingGraph:
    '''The graph rebound in `sequence` order, no node merged and no edge changed: what a
    drawing puts on its diagonal.'''
    graph = graph_drawn(handed)
    return NestingGraph(AbstractGraph(OrderedMap((n, graph[n]) for n in sequence(graph))),
                        graph.within)
