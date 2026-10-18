'''`DrivenGraph`: an executable graph whose every problem is driven.'''

from collections.abc import Iterator

from ...core import Label, NodeDefinition, Place, SteppedExecutableGraph, Violation, is_problem
from .nodes import is_driven


def check_driven[N: Label, V: Place](name: N, node: NodeDefinition[V]) -> Iterator[Violation]:
    '''Every problem carries a contract, placed.'''
    if is_problem(node) and not is_driven(node):
        yield Undriven(name)


class DrivenGraph[N: Label, V: Place](SteppedExecutableGraph[N, V]):
    '''Executable graph with additionally all problems driven.'''

    checks_node = (check_driven,)


# ----------------------------------------------------------------- violation finals

class Undriven(Violation):
    node : object
    error    = ValueError
    template = ('{node!r} carries no driver, so nothing answers it -- `Assign` one')
