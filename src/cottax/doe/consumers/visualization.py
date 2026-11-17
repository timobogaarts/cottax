'''Registers how a DOE is drawn (`NodeDrawing`), and its category (`SAMPLE`). Its mints are
written around their spelling, `points(x)` and `record(h)`.'''

from ...core.frozen import answers
from ...consumers.visualization.interface import NodeDrawing
from ...consumers.visualization.interface_types import Drawn, Kind, Slot, Text
from ..doe_condition import DoeCondition


SAMPLE = Kind('sample', 'sample (h_i = F(x_i))',
              'each factor set to every point of its plan, each response recorded there', False,
              ('#e6cbdd', '#8e4775'))
'''A DOE's one category, in plum.'''

RESPONSE = Text('hᵢ =', 'h_i =')
PLAN     = Text('xᵢ =', 'x_i =')
'''What labels the responses' and the plans' slots.'''


@answers(DoeCondition)
class DoeDrawing(NodeDrawing):
    '''`SAMPLE`: the responses and the plans, a slot each.'''

    def drawn(self, node: DoeCondition) -> Drawn:
        return Drawn(SAMPLE, (Slot((RESPONSE,), node.reads_inside), Slot((PLAN,), node.plans)))
