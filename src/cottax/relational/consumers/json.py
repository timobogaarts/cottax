'''Registers the JSON `Spelling` of a relational condition.'''

from typing import Any

from ...core import Place, Violation
from ...core.frozen import answers
from ...consumers.json.interface import Spelling
from ...consumers.json.interface_types import Reader, Writer
from ..relational_condition import Eq, Le, Max, Min, Objective, Relation, RelationalCondition


RELS = {'eq': Eq, 'le': Le}
'''Each relation by its word in the file: a new `Rel` is one line here.'''
SENSES = {'min': Min, 'max': Max}
'''Each objective's sense by its word in the file.'''
TABLES = {'RELS': RELS, 'SENSES': SENSES}
'''Each table a word in the file is read through, by its name.'''


def word_of(table: dict[str, type], marker: type) -> str | None:
    '''The word `table` writes `marker` as, if any.'''
    return next((k for k, m in table.items() if m is marker), None)


@answers(RelationalCondition)
class RelationalSpelling(Spelling[RelationalCondition]):
    '''The relations, each `op` by its word in `RELS`; the objectives, each `sense` by its
    word in `SENSES`; the unknowns and parameters.'''

    kind = 'relational'

    def write[V: Place](self, node: RelationalCondition[V], w: Writer[Any, V]) -> dict:
        def relation(x: Relation[V]) -> dict:
            if (op := word_of(RELS, x.op)) is None:
                raise w.refuse(x.op.__name__, 'a relation with no spelling in `RELS`')
            side = lambda p: None if p is None else w.place(p)          # a side of `None` is zero
            return {'lhs': side(x.lhs), 'op': op, 'rhs': side(x.rhs)}
        def objective(o: Objective[V]) -> dict:
            if (sense := word_of(SENSES, o.sense)) is None:
                raise w.refuse(o.sense.__name__, 'a sense with no spelling in `SENSES`')
            return {'place': w.place(o.place), 'sense': sense}
        return {'relations': [relation(x) for x in node.relations],
                'objectives': [objective(o) for o in node.objectives],
                'unknowns': w.places(node.unknowns),
                'parameters': w.places(node.parameters)}

    def read[V: Place](self, data: dict, r: Reader[Any, V]) -> RelationalCondition[V]:
        def marker(table: str, word: str) -> type:
            if (found := TABLES[table].get(word)) is None:
                raise UnknownWord(r.name, table, word).raised
            return found
        side = lambda p: None if p is None else r.place(p)
        return RelationalCondition(
            tuple(Relation(side(x['lhs']), side(x['rhs']), marker('RELS', x['op']))
                  for x in data['relations']),
            tuple(Objective(r.place(o['place']), marker('SENSES', o['sense']))
                  for o in data['objectives']),
            r.places(data['unknowns']), r.places(data['parameters']))


class UnknownWord(Violation):
    node  : object
    table : str
    word  : str
    error    = ValueError
    template = '{node!r}: {word!r} is no entry of `{table}`'
