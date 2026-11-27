'''
TEMPORARY (phase 1 of the plan redesign, handoff §3.3): today's op bases, written on the new
core so the ops outside `plan/` keep working until phase 2 ports them to `PrimitiveOp` / `Composite`
(or plain `_changed` code over `in_turn`). Phase 2 deletes this module. Nothing new is to be
written on it. Each base `needs` `Valid`, as every op did before.
'''

from abc import abstractmethod
from collections.abc import Generator, Mapping, Sequence
from dataclasses import replace

from ..frozen import Frozen, OrderedMap
from ..maths.definition import NodeDefinition
from ..maths.graph import Graph
from ..maths.places import Label, Place
from ..maths.valid import Valid
from .edit import Edit
from .op import Composite, GraphOp, PrimitiveOp, in_turn
from .primitive_ops import blocks_own_something, blocks_partition, contraction
from .primitive_ops import nested_among


class ComposedOp[N: Label, V: Place](Composite[N, V]):
    '''An op made of other ops, applied in order.'''

    needs = Valid

    @abstractmethod
    def composed(self, graph: Graph[N, V]) -> Sequence[GraphOp[N, V]]:
        '''The ops this one is, which may depend on `graph`.'''

    def _ops(self, graph: Graph[N, V]) -> Sequence[GraphOp[N, V]]:
        return self.composed(graph)


class Stage[N: Label, V: Place](Frozen):
    '''One stage of a `StagedOp`: the ops decided, and the graph they leave.'''

    ops   : tuple[GraphOp[N, V], ...]
    graph : Graph[N, V]


class StagedOp[N: Label, V: Place](GraphOp[N, V]):
    '''A composition decided in stages, as `stages`.'''

    needs = Valid

    @abstractmethod
    def stages(self, graph: Graph[N, V]
               ) -> Generator[Sequence[GraphOp[N, V]], Graph[N, V], None]:
        '''Yield a stage's ops; receive the graph they leave.'''

    def _changed(self, graph):
        result, edits, _ = self._run(graph)
        return result, edits

    def staged(self, graph: Graph[N, V]) -> tuple[Stage[N, V], ...]:
        '''Every stage, with the graph it left; the first refusal raised.'''
        self.apply(graph)
        return self._run(graph)[2]

    def _run(self, graph):
        stages, edits, decide, sent = [], [], self.stages(graph), None
        while True:
            try:
                ops = tuple(decide.send(sent))
            except StopIteration:
                return graph, tuple(edits), tuple(stages)
            graph, done = in_turn(ops, graph)
            edits += done
            stages.append(Stage(ops, graph))
            sent = graph


class Rebinding[N: Label, V: Place](Frozen):
    '''Names removed, and names defined: in place where bound, appended where new.'''

    remove : frozenset[N]
    define : OrderedMap[N, NodeDefinition[V]]


class Rebind[N: Label, V: Place](PrimitiveOp[N, V]):
    '''Remove and define names in one step.'''

    needs = Valid

    @abstractmethod
    def rebinding(self, graph: Graph[N, V]) -> Rebinding[N, V]:
        '''What this op removes and defines of `graph`.'''

    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        r = self.rebinding(graph)
        return Edit(unnest=nested_among(graph, r.remove),
                    remove=r.remove,
                    redefine=OrderedMap((n, d) for n, d in r.define.items() if n in graph),
                    insert=OrderedMap((n, d) for n, d in r.define.items() if n not in graph))


class Redefine[N: Label, V: Place](PrimitiveOp[N, V]):
    """Replace definitions in place, carrying what each node carries."""

    needs = Valid

    @abstractmethod
    def redefined(self, graph: Graph[N, V]) -> Mapping[N, NodeDefinition[V]]:
        """The new definition of each node this op touches."""

    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        redefined = OrderedMap(self.redefined(graph))
        return Edit(redefine=OrderedMap((name, replace(after, attributes=graph[name].attributes))
                                        for name, after in redefined.items()))


def _check_blocks_partition(op: 'Contract'):
    return blocks_partition(op.blocks)


def _check_blocks_own_something(op: 'Contract', graph: Graph):
    return blocks_own_something(graph, op.blocks)


class Contract[N: Label, V: Place](PrimitiveOp[N, V]):
    '''`contraction` as an op, until its callers call `contracted`.'''

    needs = Valid

    blocks : tuple[tuple[N, ...], ...]

    checks       = (_check_blocks_partition,)
    checks_graph = (_check_blocks_own_something,)

    def _edit(self, graph: Graph[N, V]) -> Edit[N, V]:
        return contraction(graph, self.blocks)

    def __repr__(self) -> str:
        return 'contract(' + ', '.join('{' + ', '.join(map(str, b)) + '}' for b in self.blocks) + ')'
