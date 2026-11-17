'''Registers how an architecture reads a DOE (`ModelReading`): no implicit model, so a preset
refuses one.'''

from ...core import Violation
from ...core.frozen import answers
from ...consumers.architectures.interface import ModelReading
from ..doe_condition import DoeCondition


@answers(DoeCondition)
class DoeModel(ModelReading):
    '''No implicit model, no body, combining as nothing. Work in progress: no preset places a
    DOE yet, so one meeting it refuses it.'''

    def not_implicit(self, graph, problem):
        return (DoeNotPlaced(str(problem)),)

    def body(self, graph, problem):
        return ()

    def combines_as(self, condition):
        return None


class DoeNotPlaced(Violation):
    problem : str
    error    = ValueError
    template = ('{problem} is a design of experiments, which no preset places yet -- nest what '
                'it samples in it by hand (`fixed_point_cut`, `nest`)')
