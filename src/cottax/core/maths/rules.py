'''
What is proved of a graph: rule groups, values not classes. What a function or class asks of
the graph it is handed is declared with it (`@checked(proved(Group))`) and read off it
(`proves`).

A group's checks are asked of the graph (`Rules`) or of each node (`NodeRules`), after the
groups it implies. A group that passed is kept in the graph's set of proved groups,
`remembered` on it, outside its equality, hash and pickle, so asking it again costs nothing
and the graph is the same value whether proved or not.

Nothing is copied from one graph to another: a subgraph is a value of its own and is proved
when it is asked, so every entry of the set is a fact of the value it is on.
'''

from collections.abc import Callable, Iterator
from typing import Any

from ..frozen import Frozen, Violation, first, remembered
from .graph import Graph


class Rules(Frozen):
    '''A group of rules asked of the whole graph: `check(graph) -> violations`.'''

    name    : str
    rules : tuple[Callable[[Graph], Iterator[Violation]], ...] = ()
    implies : tuple['Rules | NodeRules', ...] = ()
    '''Proved first, each once however often it is implied.'''

    def asked(self, graph: Graph) -> Iterator[Violation]:
        for check in self.rules:
            yield from check(graph)

    def __repr__(self) -> str:
        return self.name


class NodeRules(Frozen):
    '''A group of rules asked of each node: `check(name, node) -> violations`.'''

    name    : str
    rules : tuple[Callable[[Any, Any], Iterator[Violation]], ...] = ()
    implies : tuple['Rules | NodeRules', ...] = ()

    def asked(self, graph: Graph) -> Iterator[Violation]:
        for check in self.rules:
            for name, node in graph.definitions.items():
                yield from check(name, node)

    def __repr__(self) -> str:
        return self.name


type Group = Rules | NodeRules


@remembered
def closure(group: Group) -> tuple[Group, ...]:
    '''`group` and every group it implies, transitively: implied first, each once, `group`
    last -- the order they are asked in.'''
    found = {g: None for implied in group.implies for g in closure(implied)}
    return (*found, group) if group not in found else tuple(found)


@remembered
def proofs(graph: Graph) -> set[Group]:
    '''The groups whose own rules found nothing on `graph`. Written by `unproved` alone; a
    group is proved when it and every group it implies are here (`is_proved`).'''
    return set()


def unproved(graph: Graph, group: Group) -> Iterator[Violation]:
    '''Every violation of `group` and the groups it implies, as data, implied groups first.
    A group whose rules find nothing is added to the graph's proofs.'''
    done = proofs(graph)
    for g in closure(group):
        if g in done:
            continue
        found = False
        for violation in g.asked(graph):
            found = True
            yield violation
        if not found:
            done.add(g)


def is_proved(graph: Graph, group: Group) -> bool:
    '''Whether `group` and everything it implies are among `graph`'s proofs. Asks nothing.'''
    return all(g in proofs(graph) for g in closure(group))


def prove(graph: Graph, group: Group) -> None:
    '''Prove `group` of `graph`: the first violation raised, else remembered on it. Nothing
    is returned, since nothing new is made: the graph is the same value, proved.'''
    first(unproved(graph, group))


def proved(group: Group, of: int = 0, at: str = '') -> Callable[..., Iterator[Violation]]:
    '''`unproved` of the `of`-th argument -- or of its attribute at `at`, a dotted path --
    as a check `@checked` and `checks` take. Declared where the function or class is, so what
    it proves is read off its signature, not its body (`proves`).'''
    def check(*args: Any) -> Iterator[Violation]:
        graph = args[of]
        for name in filter(None, at.split('.')):
            graph = getattr(graph, name)
        yield from unproved(graph, group)
    check.__name__ = check.__qualname__ = f'proved({group.name}{", at=" + repr(at) if at else ""})'
    check.group = group                                             # type: ignore[attr-defined]
    return check


def proves(declared: Any) -> tuple[Group, ...]:
    '''The groups a function (`@checked`) or a class (`checks`, every level) proves of what
    it is handed, as `proved` declared them.'''
    checks = (declared.per_level('checks') if isinstance(declared, type) and issubclass(declared, Frozen)
              else getattr(declared, 'checks', ()))
    return tuple(c.group for c in checks if hasattr(c, 'group'))
