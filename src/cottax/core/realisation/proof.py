'''
What a graph answers to before anything runs it: every node carrying what runs it, over
`driven.DrivenGraph`. It does not ask whether a backend *can* run what it finds -- that is
the backend's proof, a level under this.
'''

from collections.abc import Iterator

from ...core import Label, NodeDefinition, Place, Violation
from ..driven import DrivenGraph
from .implemented import is_implemented


def check_implemented[N: Label, V: Place](name: N, node: NodeDefinition[V]) -> Iterator[Violation]:
    '''Every node carries what runs it -- a function its body, a problem its driver's
    implementation. An executable graph holds no bare condition, so that is every node.'''
    if not is_implemented(node):
        yield NoImplementation(name)


class RealisedGraph[N: Label, V: Place](DrivenGraph[N, V]):
    '''A driven graph with what runs every node: what a backend's proof strengthens.'''

    checks_node = (check_implemented,)


# ----------------------------------------------------------------- violation finals

class NoImplementation(Violation):
    node : object
    error    = ValueError
    template = ('{node!r} carries nothing that runs it -- `Implement` it. Drawing, rewriting, '
                'proving and exporting a graph holding one is fine; only running it is not')
