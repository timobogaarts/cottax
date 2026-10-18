'''Registers how an ODE condition is drawn (`NodeDrawing`), its category (`INTEGRATE`), and how
the names its vocabulary mints are written (`MintDrawing`).'''

from ...core.frozen import answers
from ...consumers.visualization.interface import MintDrawing, NodeDrawing
from ...consumers.visualization.interface_types import Around, Drawn, Kind, Slot, Text
from ..mints import Initial, Rate
from ..ode_condition import OdeCondition


INTEGRATE = Kind('integrate', "integrate (x' = r)",
                 'each state integrated at its rate, from its initial value over a span', False,
                 ('#b9dbe0', '#2f7682'))
'''An ODE's one category, in teal.'''

@answers(Rate)
class RateDrawing(MintDrawing):
    r'''A state's rate: `\dot{x}`.'''

    plain = Around('rate({})')
    latex = Around(r'\dot{{{}}}')


@answers(Initial)
class InitialDrawing(MintDrawing):
    '''A state's initial value: `x₀`.'''

    plain = Around('{}₀')
    latex = Around(r'{{{}}}_0')

RATE    = Text('ẋ =', r'\dot{x} =')
INITIAL = Text('x(t₀) =', r'x(t_0) =')
SPAN    = Text('t ∈', r't \in')
'''What labels the rates', the initial values' and the span's slots.'''


@answers(OdeCondition)
class OdeDrawing(NodeDrawing):
    '''`INTEGRATE`: the rates, the initial values and the span, a slot each.'''

    def drawn(self, node: OdeCondition) -> Drawn:
        return Drawn(INTEGRATE, (Slot((RATE,), node.rates), Slot((INITIAL,), node.initials),
                                 Slot((SPAN,), (node.span,))))
