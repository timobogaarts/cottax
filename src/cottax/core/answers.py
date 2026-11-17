'''
The one `Answer` for a whole graph: the one registered for its place class, or its label
class. An interface of core's, like `mints`, not maths: a lookup in the registry keyed by
the classes a graph is over, and no statement about the graph.
'''

from collections.abc import Iterator, Sequence
from typing import Any

from .frozen import Answer, Violation, answer, checked, dotted
from .frozen.answer import Unanswered
from .maths.graph import Graph


def _refusals(interface: type[Answer], asked: Sequence[Any], what: str) -> Iterator[Violation]:
    '''Why no `interface` answer is found for the class of `asked[0]`.'''
    if not asked:
        yield NothingToAsk(interface.__qualname__, what)
    elif answer(interface, asked[0], None) is None:
        yield Unanswered(interface.__qualname__, dotted(type(asked[0])))


def check_places_answered(interface: type[Answer], graph: Graph) -> Iterator[Violation]:
    yield from _refusals(interface, graph.variables, 'place')


def check_labels_answered(interface: type[Answer], graph: Graph) -> Iterator[Violation]:
    yield from _refusals(interface, graph.nodes, 'node name')


@checked(check_places_answered)
def answer_for_places[A: Answer](interface: type[A], graph: Graph) -> A:
    '''The `interface` answer registered for the class of `graph`'s places, asked of one: a
    graph is over one place class. Its refusals are `violations(answer_for_places, ...)`.'''
    return answer(interface, graph.variables[0])


@checked(check_labels_answered)
def answer_for_labels[A: Answer](interface: type[A], graph: Graph) -> A:
    '''The `interface` answer registered for the class of `graph`'s node names, asked of one.
    Its refusals are `violations(answer_for_labels, ...)`.'''
    return answer(interface, graph.nodes[0])


class NothingToAsk(Violation):
    interface : str
    what      : str
    error    = ValueError
    template = 'a graph with no {what} has no class to find its {interface} by'
