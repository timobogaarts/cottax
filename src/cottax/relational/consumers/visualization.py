'''Registers how a relational condition is drawn (`NodeDrawing`), its categories (`KINDS`), and
how the names its vocabulary mints are written (`MintDrawing`).'''

from ...core.frozen import answers
from ...consumers.visualization.interface import MintDrawing, NodeDrawing
from ...consumers.visualization.interface_types import Around, Drawn, Kind, Slot, Text
from ..classification import (fixed_point_shaped, is_feasibility, is_fixed_point, is_optimise,
                              is_root_find)
from ..mints import Cond
from ..relational_condition import Eq, Le, Max, Min, RelationalCondition


OPTIMISE    = Kind('optimise', 'optimise',
                   'one objective, minimised subject to the relations', True,
                   ('#dbcfec', '#77609a'))
ROOT_FIND   = Kind('root-find', 'root-find (r = 0)',
                   'every relation vanishes, one per unknown: r(u) = 0', False,
                   ('#ead3a0', '#96771f'))
FIXED_POINT = Kind('fixed-point', 'fixed-point (u = g(u))',
                   'every relation is the next value of its unknown: u = g(u)', False,
                   ('#c4dcb2', '#4f7a37'))
FEASIBILITY = Kind('feasibility', 'feasibility (h = 0, g <= 0)',
                   'no objective: any point satisfying the relations will do', True,
                   ('#dcdcdc', '#5e5e5e'))

KINDS = (OPTIMISE, ROOT_FIND, FIXED_POINT, FEASIBILITY)
'''Every category a relational condition draws under.'''

@answers(Cond)
class CondDrawing(MintDrawing):
    '''The condition on a place: a residual, `R(u)`.'''

    plain = Around('R({})')
    latex = Around(r'\mathrm{R}({})')

SLOT_TEXT = {Min: Text('min', r'\min'), Max: Text('max', r'\max'), '= 0': Text('= 0', '= 0'),
             '≤ 0': Text('≤ 0', r'\leq 0'), '0 ≤': Text('0 ≤', r'0 \leq')}
'''What labels a slot of each kind, plain and in LaTeX: an objective's by its sense.'''

EQUALS, EQUALS_OWN = Text(' = ', ' = '), Text('= ', '= ')
AT_MOST, AT_MOST_OWN, OWN_AT_MOST = (Text(' ≤ ', r' \leq '), Text('≤ ', r'\leq '),
                                     Text(' ≤', r' \leq'))


def kind_of(node: RelationalCondition) -> Kind | None:
    '''
    The category `node` draws under: a problem's by its shape, a requirement's the one it
    would take with its left sides as unknowns; `None` for none (two objectives).
    '''
    if node.unknowns:
        return (OPTIMISE if is_optimise(node) else FIXED_POINT if is_fixed_point(node)
                else ROOT_FIND if is_root_find(node) else FEASIBILITY if is_feasibility(node)
                else None)
    if node.objectives:
        return OPTIMISE if len(node.objectives) == 1 else None
    if not node.relations:
        return None
    if any(r.op is not Eq for r in node.relations):
        return FEASIBILITY
    lefts = tuple(r.lhs for r in node.relations if r.lhs is not None)
    return FIXED_POINT if fixed_point_shaped(node.relations, lefts) else ROOT_FIND


def slots_of(node: RelationalCondition) -> tuple[Slot, ...]:
    '''
    A condition node's reads by what each one says, in the order its driver is handed them:
    the objectives, then the relations against zero -- vanishing, bounded above, bounded
    below -- then each relation between two places, labelled by the side the node owns
    (`= u`, `≤ u`, `u ≤`) or, owning neither or both, as written.
    '''
    out = [Slot((SLOT_TEXT[o.sense],), (o.place,)) for o in node.objectives]
    eq0, le, ge, two = [], [], [], []
    for rel in node.relations:
        reads = tuple(t for t in rel.terms if t not in node.unknowns)
        owned = tuple(t for t in rel.terms if t in node.unknowns)
        if None in (rel.lhs, rel.rhs):
            (eq0 if rel.op is Eq else ge if rel.lhs is None else le).extend(reads)
        elif rel.op is Eq:
            two.append(Slot((EQUALS_OWN, *owned) if len(owned) == 1
                            else (rel.lhs, EQUALS, rel.rhs), reads))
        elif len(owned) == 1:                           # a bound: `lo ≤ u` is `≤ u`, `u ≤ hi` is `u ≤`
            two.append(Slot((AT_MOST_OWN, rel.rhs) if rel.rhs in owned
                            else (rel.lhs, OWN_AT_MOST), reads))
        else:
            two.append(Slot((rel.lhs, AT_MOST, rel.rhs), reads))
    for label, found in (('= 0', eq0), ('≤ 0', le), ('0 ≤', ge)):
        if found:
            out.append(Slot((SLOT_TEXT[label],), tuple(found)))
    return (*out, *two)                             # a consistency last, at the right


@answers(RelationalCondition)
class RelationalDrawing(NodeDrawing):
    '''Its category (`kind_of`) and its slots (`slots_of`).'''

    def drawn(self, node: RelationalCondition) -> Drawn:
        return Drawn(kind_of(node), slots_of(node))
