'''Registers how an architecture places an ODE (`ModelReading`), with why one is no implicit model.'''

from ...core import FunctionalNode, Violation
from ...core.frozen import answers
from ...consumers.architectures.interface import ModelReading
from ..ode_condition import OdeCondition


@answers(OdeCondition)
class OdeModel(ModelReading):
    '''
    Implicit model is here the entire function body: a cut in it does not work.
    In algebraic systems a cut is always the same (since hat(y) = y algebraically), but in rate-based systems where the
    body is integrated, that is not the same (hat(y) might change during the integration).
    '''

    def not_implicit(self, graph, problem):
        owners = graph.graph.owners
        if unwritten := tuple(r for r in graph[problem].conditions
                              if r not in owners or not isinstance(graph[owners[r]], FunctionalNode)):
            return (RateUnwritten(str(problem), tuple(map(str, unwritten))),)
        return ()

    def body(self, graph, problem):
        return tuple(n for n in graph.cycle_of(problem) if isinstance(graph[n], FunctionalNode))

    def combines_as(self, condition):
        return None


class RateUnwritten(Violation):
    problem : str
    rates   : tuple
    error    = ValueError
    template = ('{problem} is not an implicit model: no function node writes {rates!r} -- a '
                'rate is what its body computes from the states')
