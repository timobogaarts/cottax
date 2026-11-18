'''
Graph rules
(`@checked(Executable.check)`, `checks = (Valid.check,)`) 
'''
from collections.abc import Callable, Iterator
from functools import update_wrapper
from typing import Any, ClassVar, NoReturn, Protocol
from ..frozen import Violation, first, remembered
from .definition import NodeDefinition
from .graph import Graph
from .places import Label, Place

class HasGraph(Protocol):
    @property 
    def graph(self) -> Graph: ...    

class Rules:
    '''
    Inheritable rules

        class Executable(Valid):
            rules = (check_no_requirements, ...)

        class Driven(Executable, DrivenNodes):
            \'\'\'The two together: no rules of its own.\'\'\'

    A group's bases are groups, nothing else. Python inheritance already ensures it's a DAG.
    '''

    rules : ClassVar[tuple[Callable[[Graph], Iterator[Violation]], ...]] = ()

    @classmethod
    def implied(cls) -> tuple[type['Rules'], ...]:
        '''This group and every group it implies (in order written for multiple inheritance; depth-first)'''
        found = dict.fromkeys(g for base in cls.__bases__ if base is not Rules
                              for g in base.implied())
        return (*found, cls)

    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f'{cls.__name__} is a group of rules, asked of a graph '
                        f'(`{cls.__name__}.check`, `{cls.__name__}.prove`), not made')

    @classmethod
    def check(cls, graph: Graph | HasGraph, *_: Any, **__: Any) -> Iterator[Violation]:
        '''Every violation of this group and the groups it implies, as data, implied groups
        first (`@checked(X.check)` and `checks = (X.check,)`). A group implying one
        that fails is not asked: its rules may read what the groups it implies make true.'''
        graph = graph if isinstance(graph, Graph) else graph.graph
        for g in cls.implied():
            if not _satisfies(graph, g) and all(_satisfies(graph, b) for b in g.implied()[:-1]):
                yield from _own(graph, g)           # fails on its own: everything it finds

    @classmethod
    def prove(cls, graph: Graph | HasGraph, *_: Any, **__: Any) -> None:
        '''The first violation raised, else nothing: the graph is the same value, proved,
        and asking again costs nothing.'''
        graph = graph if isinstance(graph, Graph) else graph.graph
        if not _satisfies(graph, cls):
            first(cls.check(graph))


class per_node_check[N: Label, V: Place]:
    '''Wrapper to convert a per-node-check to a graph-check.'''

    def __init__(self, node: Callable[[N, NodeDefinition[V]], Iterator[Violation]]):
        self.node = node
        update_wrapper(self, node)

    def __call__(self, graph: Graph[N, V]) -> Iterator[Violation]:
        for name, definition in graph.definitions.items():
            yield from self.node(name, definition)


def _own(graph: Graph, group: type[Rules]) -> Iterator[Violation]:
    '''The violations of `group`'s own rules, not its bases'.'''
    for rule in group.__dict__.get('rules', ()):
        yield from rule(graph)


@remembered
def _satisfies(graph: Graph, group: type[Rules]) -> bool:
    '''Whether `graph` satisfies `group`, decided once and remembered (cached).
    '''
    return all(_satisfies(graph, b) for b in group.implied()[:-1]) and next(_own(graph, group), None) is None
