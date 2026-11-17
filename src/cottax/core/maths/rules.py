'''
What is proved of a graph: rule groups, each a class whose bases are the groups it implies.
What a function or class asks of the graph it is handed is declared with it
(`@checked(proved(Group))`) and read off it (`proves`).

A group is a class (`Rules`): its rules are asked of the graph after those of the groups it
implies, its bases; a rule of each node is lifted to one of the graph (`each_node`). A group that passed is kept in the graph's set of
proved groups, `remembered` on it, outside its equality, hash and pickle, so asking it again
costs nothing and the graph is the same value whether proved or not.

Nothing is copied from one graph to another: a subgraph is a value of its own and is proved
when it is asked, so every entry of the set is a fact of the value it is on.
'''

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, NoReturn

from ..frozen import Frozen, Violation, first, remembered
from .graph import Graph


class Rules:
    '''
    A group of rules asked of the graph, as a class: its own `rules`, `rule(graph) ->
    violations`, and its bases, the groups it implies -- proved first, each once, in the
    order the bases are written. A class is never instantiated: it names the group, and is
    the key of the graph's proofs.

        class ExecutableGraph(WellFormed):
            rules = (check_no_requirements, ...)

        class DrivenGraph(ExecutableGraph, DrivenNodes):
            \'\'\'The two together: no rules of its own.\'\'\'

    One that fails leaves every group implying it unasked: a group's rules may read what the
    groups it implies make true.
    '''

    rules : ClassVar[tuple[Callable[[Graph], Iterator[Violation]], ...]] = ()

    closure : ClassVar[tuple[type['Rules'], ...]] = ()
    '''This group and every group it implies, implied first, each once, this group last.'''

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        found = {g: None for base in cls.__bases__ if issubclass(base, Rules)
                 for g in base.closure}
        cls.closure = (*found, cls)

    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f'{cls.__name__} is a group of rules, asked of a graph, not made')

    @classmethod
    def asked(cls, graph: Graph) -> Iterator[Violation]:
        '''Its own rules, not its bases'.'''
        for rule in cls.__dict__.get('rules', ()):
            yield from rule(graph)


class each_node(Frozen):
    '''A rule of one node, `rule(name, node) -> violations`, asked of every node of the graph:
    a rule of the graph itself.'''

    rule : Callable[[Any, Any], Iterator[Violation]]

    def __call__(self, graph: Graph) -> Iterator[Violation]:
        for name, node in graph.definitions.items():
            yield from self.rule(name, node)

    def __repr__(self) -> str:
        return f'each_node({self.rule.__name__})'


@remembered
def proofs(graph: Graph) -> set[type[Rules]]:
    '''The groups whose own rules found nothing on `graph`. Written by `unproved` alone; a
    group is proved when it and every group it implies are here (`is_proved`).'''
    return set()


def unproved(graph: Graph, group: type[Rules]) -> Iterator[Violation]:
    '''Every violation of `group` and the groups it implies, as data, implied groups first.
    A group whose rules find nothing is added to the graph's proofs; a group implying one that
    failed is not asked.'''
    done, failed = proofs(graph), set()
    for g in group.closure:
        if g in done:
            continue
        if any(base in failed for base in g.__mro__[1:]):
            failed.add(g)
            continue
        found = False
        for violation in g.asked(graph):
            found = True
            yield violation
        (failed if found else done).add(g)


def is_proved(graph: Graph, group: type[Rules]) -> bool:
    '''Whether `group` and everything it implies are among `graph`'s proofs. Asks nothing; a
    group with no rules of its own holds once the groups it implies do.'''
    done = proofs(graph)
    return all(g in done for g in group.closure if 'rules' in g.__dict__)


def prove(graph: Graph, group: type[Rules]) -> None:
    '''Prove `group` of `graph`: the first violation raised, else remembered on it. Nothing
    is returned, since nothing new is made: the graph is the same value, proved.'''
    first(unproved(graph, group))


class proved(Frozen):
    '''`unproved` of the `of`-th argument -- or of its attribute at `at`, a dotted path -- as
    a check `@checked` and `checks` take. Declared where the function or class is, so what it
    proves is read off its signature, not its body (`proves`).'''

    group : type[Rules]
    of    : int = 0
    at    : str = ''

    def __call__(self, *args: Any) -> Iterator[Violation]:
        graph = args[self.of]
        for name in filter(None, self.at.split('.')):
            graph = getattr(graph, name)
        yield from unproved(graph, self.group)

    def __repr__(self) -> str:
        return f'proved({self.group.__name__}{", at=" + repr(self.at) if self.at else ""})'


def proves(declared: Any) -> tuple[type[Rules], ...]:
    '''The groups a function (`@checked`) or a class (`checks`, every level) proves of what
    it is handed, as `proved` declared them.'''
    checks = (declared.per_level('checks') if isinstance(declared, type) and issubclass(declared, Frozen)
              else getattr(declared, 'checks', ()))
    return tuple(c.group for c in checks if isinstance(c, proved))
