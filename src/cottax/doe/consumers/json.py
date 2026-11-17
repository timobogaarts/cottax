'''Registers the JSON `Spelling` of a DOE.'''

from typing import Any

from ...core import Place
from ...core.frozen import answers
from ...consumers.json.interface import Spelling
from ...consumers.json.interface_types import Reader, Writer
from ..doe_condition import DoeCondition, Recorded, Varied


@answers(DoeCondition)
class DoeSpelling(Spelling[DoeCondition]):
    '''The factors and their plans, then the responses and their records.'''

    kind = 'doe'

    def write[V: Place](self, node: DoeCondition[V], w: Writer[Any, V]) -> dict:
        return {'factors': w.places(node.owns_inside), 'plans': w.places(node.plans),
                'responses': w.places(node.reads_inside), 'records': w.places(node.records)}

    def read[V: Place](self, data: dict, r: Reader[Any, V]) -> DoeCondition[V]:
        pairs = lambda a, b: zip(r.places(data[a]), r.places(data[b]), strict=True)
        return DoeCondition(tuple(Varied(*p) for p in pairs('factors', 'plans')),
                            tuple(Recorded(*p) for p in pairs('responses', 'records')))
