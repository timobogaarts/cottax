'''
How a graph is read for a drawing: each component as what it is (`entries`), the blocks a
drawing rings (`walk`), and an order for its diagonal (`sequence`, `problems_first`). A graph
that cannot be scheduled still has entries, so it is still drawn.
'''

from collections.abc import Callable, Iterable, Iterator, Sequence

from ...core.frozen import Frozen, OrderedMap
from ...core import AbstractGraph, ExecutableGraph, Plan, Label, NestingGraph, body_of, problems


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
    One component of a graph, its kind the final (`Run`, `Cycle`, `Solve`,
    `Unresolved`). Built by `entries`.
    '''
    subgraph : NestingGraph

    @property
    def nodes(self) -> tuple[Label, ...]:
        '''The component, in the graph's binding order.'''
        return self.subgraph.nodes


class Run(Entry):
    '''One node, no problem: it runs.'''


class Cycle(Entry):
    '''A cycle with no problem on it: nothing iterates it.'''


class Solve(Entry):
    '''
    One outermost problem, answered at this component's level, and its interior (the
    component without it, `NestingGraph.interior`), whose own `entries` are the next depth.
    '''
    problem  : Label
    interior : NestingGraph


class Unresolved(Entry):
    '''Several outermost problems with nothing nested: none is answered at this level.'''
    problems : tuple[Label, ...]


def entries(g: NestingGraph) -> tuple[Entry, ...]:
    '''
    Each component of `g` as an `Entry`, in dependency order, the members of each in
    binding order.
    '''
    out : list[Entry] = []
    for component in g.graph.components:
        sub   = g.subgraph(component)
        found = sub.outermost_problems
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
    One entry as a drawing rings it: the `entry`, the problems it sits inside (`within`,
    outermost first; its length the depth), and `held`, the interior rung inside it: a
    `Solve`'s interior when that holds a problem, else `None`.
    '''
    entry  : Entry
    within : tuple[Label, ...]
    held   : NestingGraph | None

    @property
    def depth(self) -> int:
        return len(self.within)

    @property
    def nodes(self) -> tuple[Label, ...]:
        return self.entry.nodes

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
    Every block a drawing rings, at every depth: the graph's entries, each followed by its
    `held` interior's, so an inner ring comes after (paints over) the ring around it.
    `problems_first` relies on this order.
    '''
    def blocks(g: NestingGraph, within: tuple[Label, ...]) -> Iterator[Block]:
        for entry in entries(g):
            held = (entry.interior if isinstance(entry, Solve)
                    and problems(entry.interior.definitions) else None)
            yield Block(entry, within, held)
            if held is not None:
                yield from blocks(held, (*within, entry.problem))
    return tuple(blocks(graph, ()))


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
    The graph's nodes in an order for one diagonal: entry by entry in dependency order; a
    `Solve` its problem then its interior sequenced the same way; any other entry its
    problems in binding order, then its body as bound unless that draws feedback, else in
    `components_order` (defined on a cycle too). So the only edges left above the
    diagonal run through a problem, or are a cycle no problem closes.
    '''
    out : list[Label] = []
    for entry in entries(graph):
        if isinstance(entry, Solve):
            out.append(entry.problem)
            out.extend(sequence(entry.interior))
        else:
            sub      = entry.subgraph
            body     = body_of(sub)
            held_out = [n for n in entry.nodes if n not in body.definitions]
            order    = [n for n in entry.nodes if n in body.definitions]
            if order and draws_feedback(order, body):
                order = list(body.graph.components_order)
            out.extend((*held_out, *order))
    return tuple(out)


def ordered_graph(handed: NestingGraph | ExecutableGraph | Plan) -> NestingGraph:
    '''The graph rebound in `sequence` order, no node merged and no edge changed: what a
    drawing puts on its diagonal.'''
    graph = graph_drawn(handed)
    return NestingGraph(AbstractGraph(OrderedMap((n, graph[n]) for n in sequence(graph))),
                        graph.within)


# ---------------------------------------------------------------- the levels

_LAST = 1 << 30
'''A rank past every other; `_LAST - 1` ranks just before it.'''


def problems_first(names: Sequence[Label],
                   tree: Iterable[tuple[Sequence[Label], Label | None]],
                   is_problem: Callable[[Label], bool]) -> list[Label]:
    '''
    `names` permuted so each block's problem leads it: the problem `tree` pairs with the
    block (`walk`'s blocks, outermost first, each with its `Block.problem`); where that is
    `None`, every problem in the block ahead of every other member. A block's members are
    dealt back into the positions it already holds, so every block stays contiguous and no
    other step moves. `is_problem` says which names are problems. A caller with no rings
    pairs each component with `None`.
    '''
    order = list(names)
    pos   = {name: i for i, name in enumerate(order)}
    for block, lead in tree:
        slots   = sorted(pos[n] for n in block if n in pos)
        members = [order[i] for i in slots]

        def rank(name: Label, lead=lead) -> float:
            # the level's problem at the head; with none, every problem ahead of the
            # rest. A problem answered deeper sorts with the rest.
            if lead is not None:
                return 0 if name == lead else _LAST
            return _LAST - 1 if is_problem(name) else _LAST

        for i, name in zip(slots, sorted(members, key=rank)):
            order[i] = name
    return order
