'''What a picture reads off a node: the category it draws under, each ring's, and a key's entry.'''

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ...core import ConditionalNode, NodeDefinition
from ...core.driven import condition_of
from .interface_types import Drawn, Kind
from .lookup import drawn
from .refusals import NamedAlike
from .sequencing import Block, Cycle, Solve, Unresolved
from .style import Style

STATED_TEXT  = 'requirement, no unknowns named'
'''What a key says of a condition node with no category.'''
STATED_TITLE = 'asserted, determining nothing -- `Determine` it to name its unknowns'
'''What hovering says of one.'''


def by_name(kinds: Iterable[Kind | None]) -> dict[str, Kind]:
    '''Each category among `kinds` by its name, in the order first given (`None` skipped).
    Two different categories with one name are refused: a picture would draw one as the
    other.'''
    out : dict[str, Kind] = {}
    for kind in kinds:
        if kind is not None and (held := out.setdefault(kind.name, kind)) != kind:
            raise NamedAlike(kind.name, held, kind).raised
    return out


def drawn_of(node: NodeDefinition) -> Drawn | None:
    '''How `node` is drawn as a condition node (a driven node through its condition),
    `None` for a node that runs.'''
    if not isinstance(node, ConditionalNode):
        return None
    return drawn(condition_of(node))


def stated_shape(node: NodeDefinition) -> str | None:
    '''
    The category name a requirement would take once it determines something, `None` for
    anything else.
    '''
    found = drawn_of(node)
    if found is None or node.unknowns or found.kind is None:
        return None
    return found.kind.name


def block_type(block: Block) -> str | None:
    '''
    The problem type answered at `block`: a `Solve`'s problem's, `'unanswered'` for
    several problems with nothing nested, `'cycle'` for a bare function cycle, `None` for
    a block merely run.
    '''
    entry = block.entry
    return (problem_type_of(entry.subgraph[entry.problem]) if isinstance(entry, Solve) else
            'unanswered' if isinstance(entry, Unresolved) else
            'cycle' if isinstance(entry, Cycle) else None)


def answered(btype: str | None) -> bool:
    '''Whether a driver answers a solve ring of type `btype` (`block_type`): neither
    several problems with nothing nested nor a bare function cycle.'''
    return btype not in ('unanswered', 'cycle')


def ring_stroke(btype: str | None, kinds: Mapping[str, Kind], style: Style) -> str:
    '''
    The contour of a solve ring of type `btype`, `kinds` the categories drawn by name:
    the hue of the driver answering it, `stated`'s grey for a problem with no category,
    `unanswered` where no driver answers it.
    '''
    if not answered(btype):
        return style.unanswered
    return style.paint(kinds[btype].colour if btype in kinds else None)[1]


def category_of(node: NodeDefinition) -> Kind | None:
    '''The category `node` draws under: `None` for a node that runs and for one drawn
    `'stated'` (`problem_type_of`).'''
    found = drawn_of(node)
    return found.kind if found is not None and node.unknowns else None


def problem_type_of(node: NodeDefinition) -> str | None:
    '''
    The category name a node draws under: `None` for a node that runs, `'stated'` for a
    condition node with no unknowns or no category.
    '''
    found = drawn_of(node)
    if found is None:
        return None
    return found.kind.name if node.unknowns and found.kind is not None else 'stated'


def key_entry(kind: Kind | None, style: Style) -> dict:
    '''
    One entry of a key: a problem of category `kind` (`None`: a condition with none) as a
    hexagon filled and closed, as a driven one is drawn in `style`, and what the key
    (`text`) and hovering it (`title`) say.
    '''
    if kind is None:
        fill, stroke = style.stated
        return {'shape': 'hex', 'fill': fill, 'stroke': stroke, 'text': STATED_TEXT,
                'title': STATED_TITLE}
    return {'shape': 'hex', 'fill': style.fill(kind.colour, True),
            'stroke': style.paint(kind.colour)[1], 'text': kind.legend, 'title': kind.title}
