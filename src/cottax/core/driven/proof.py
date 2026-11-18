'''`Driven`: an executable graph whose every problem is driven, by a driver that fits
it. The driven layer's two node rules (`DrivenNodes`); `Assign` asks the second before the
graph is touched.'''

from collections.abc import Iterator

from ...core import (
    Executable, Label, NodeDefinition, Place, Rules, Violation, per_node_check, is_driven, AbstractConditionalNode,
)


@per_node_check
def check_driven[N: Label, V: Place](name: N, node: NodeDefinition[V]) -> Iterator[Violation]:
    '''Every condition node carries a contract, placed. Asked after `Executable`, so
    each owns something inside: none is a requirement left loose.'''
    if isinstance(node, AbstractConditionalNode) and not is_driven(node):
        yield Undriven(name)


@per_node_check
def check_driver_fits[N: Label, V: Place](name: N, node: NodeDefinition[V]) -> Iterator[Violation]:
    '''A driver fits the condition it is on: its contract accepts it, and each role's slot
    holds as many places as the role takes on it. The condition passed its own rules when
    it was built.'''
    if is_driven(node):
        yield from node.driver.violations_on(node.condition)


class DrivenNodes(Rules):
    '''Every problem driven, by a driver that fits it.'''
    rules = (check_driven, check_driver_fits)


class Driven(Executable, DrivenNodes):
    '''An executable graph whose every problem is driven.'''


# ----------------------------------------------------------------- violation finals

class Undriven(Violation):
    node : object
    error    = ValueError
    template = ('{node!r} carries no driver, so nothing answers it -- `Assign` one')
