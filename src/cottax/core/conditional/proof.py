'''`Driven`: an executable graph whose every problem is driven. That each driver fits its
condition is the node's own rule (`node.check_driver_fits`), so it is not asked here.'''

from collections.abc import Iterator

from ..frozen import Violation
from ..maths import (
    AbstractConditionalNode, Executable, Label, NodeDefinition, Place, Rules, per_node_check,
)
from .node import is_driven


@per_node_check
def check_driven[N: Label, V: Place](name: N, node: NodeDefinition[V]) -> Iterator[Violation]:
    '''Every condition node carries a contract, placed. Asked after `Executable`, so
    each owns something inside: none is a requirement left loose.'''
    if isinstance(node, AbstractConditionalNode) and not is_driven(node):
        yield Undriven(name)


class DrivenNodes(Rules):
    '''Every problem driven.'''
    rules = (check_driven,)


class Driven(Executable, DrivenNodes):
    '''An executable graph whose every problem is driven.'''


# ----------------------------------------------------------------- violation finals

class Undriven(Violation):
    node : object
    error    = ValueError
    template = ('{node!r} carries no driver, so nothing answers it -- `Assign` one')
