'''Registers the JSON `Spelling` of an ODE condition.'''

from typing import Any

from ...core import Place
from ...core.frozen import answers
from ...consumers.json.interface import Spelling
from ...consumers.json.interface_types import Reader, Writer
from ..ode_condition import Dt, OdeCondition


@answers(OdeCondition)
class OdeSpelling(Spelling[OdeCondition]):
    '''The states, their rates and their initial values, then the span.'''

    kind = 'ode'

    def write[V: Place](self, node: OdeCondition[V], w: Writer[Any, V]) -> dict:
        return {'states': w.places(node.states), 'rates': w.places(node.rates),
                'initials': w.places(node.initials), 'span': w.place(node.span)}

    def read[V: Place](self, data: dict, r: Reader[Any, V]) -> OdeCondition[V]:
        return OdeCondition(tuple(Dt(*d) for d in zip(r.places(data['states']), r.places(data['rates']),
                                                      r.places(data['initials']), strict=True)),
                            r.place(data['span']))
