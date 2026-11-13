'''Registers the JSON `Spelling` of an ODE condition.'''

from typing import Any

from ...core import Place
from ...core.frozen import answers
from ...consumers.json.interface import Spelling
from ...consumers.json.interface_types import Reader, Writer
from ..ode_condition import Dt, OdeCondition


@answers(OdeCondition)
class OdeSpelling(Spelling[OdeCondition]):
    '''The states, their rates and their initial values, then the span; the kinks, the jumps
    and the time where there are any, read as none where absent.'''

    kind = 'ode'

    def write[V: Place](self, node: OdeCondition[V], w: Writer[Any, V]) -> dict:
        written = {'states': w.places(node.states), 'rates': w.places(node.rates),
                   'initials': w.places(node.initials), 'span': w.place(node.span)}
        breaks = {k: w.places(at) for k, at in (('kinks', node.kinks), ('jumps', node.jumps))
                  if at}
        return written | breaks | ({} if node.time is None else {'time': w.place(node.time)})

    def read[V: Place](self, data: dict, r: Reader[Any, V]) -> OdeCondition[V]:
        return OdeCondition(tuple(Dt(*d) for d in zip(r.places(data['states']), r.places(data['rates']),
                                                      r.places(data['initials']), strict=True)),
                            r.place(data['span']), tuple(r.places(data.get('kinks', []))),
                            tuple(r.places(data.get('jumps', []))),
                            r.place(data['time']) if 'time' in data else None)
