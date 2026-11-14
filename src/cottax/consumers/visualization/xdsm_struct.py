'''The XDSM's struct: what a drawing of a graph says, in plain names, and its rings and key.'''

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from ...core import NestingGraph
from ...core import ExecutableGraph
from ...core import Plan
from ...core.driven import is_driven
from ...core import Label, Place

from .interface import Formatter
from ...core.frozen import first
from .lookup import namespace_of, spell_plain
from .refusals import spelled_apart
from .interface_types import Text
from .kinds import (answered, block_type, by_name, drawn_of, key_entry, problem_type_of,
                    stated_shape)
from .naming import PlainFormat, marked, shown
from .sequencing import Block, graph_drawn, ordered_graph, walk
from .style import Style


# ============================================================================ the rings
RING_LEVELS = 3
'''How many depths have a colour of their own before the palette repeats.'''


def ring_class(kind: str, depth: int) -> str:
    '''
    The CSS class a ring is drawn with: a solve's by its depth (`group-nested` below the
    top), a block run in order `group-ff` at every depth.
    '''
    nested = ' group-nested' if depth else ''
    return 'group-ff' if kind == 'run' else f'group group-l{depth % RING_LEVELS}{nested}'


RING_TEXT = {
    ('run',   0): 'run in order',
    ('solve', 0): 'solved together',
    ('solve', 1): 'solved one level in',
    ('solve', 2): 'solved two levels in',
}
'''What a key entry says of a ring, by `(kind, depth)`.'''


def ring_text(kind: str, depth: int) -> str:
    '''What a key entry says of a ring; past `RING_TEXT`, `solved <depth> levels in`.'''
    return (RING_TEXT['run', 0] if kind == 'run'
            else RING_TEXT.get((kind, depth), f'solved {depth} levels in'))


def key_entries(struct: dict, sty: Style) -> list[dict]:
    '''
    The key of one XDSM: each problem type drawn, as `key_entry` writes one, then its rings
    (`ring_entries`), each once, with the `class` the figure draws it in.
    '''
    shown = set(struct['problem_types'].values())
    out = [{**key_entry(kind, sty), 'class': 'prob'}
           for name, kind in struct['kinds'].items() if name in shown]
    if 'stated' in shown:
        out.append({**key_entry(None, sty), 'class': 'prob prob-stated'})
    return out + ring_entries(struct.get('blocks', []), sty)


def ring_entries(rings: Iterable[dict], sty: Style) -> list[dict]:
    '''
    The key's entries for rings, each a record with `ring` (`'solve'` or `'run'`), `depth`
    and `type`: each depth of solve ring, a block run in order where the style draws one, a
    ring no single problem solves (cut open) and a ring whose nesting is still open (grey, closed:
    `Block.unsettled`), each once, with the `class` an XDSM draws it in. The N2 keys its
    boxes with the same entries.
    '''
    rings = list(rings)
    out : list[dict] = []
    for depth in sorted({b['depth'] for b in rings if b['ring'] == 'solve'}):
        out.append({'shape': 'band', 'fill': sty.band(depth) or sty.surface,
                    'stroke': sty.band_stroke, 'text': ring_text('solve', depth),
                    'class': ring_class('solve', depth)})
    if sty.run_ring is not None and any(b['ring'] == 'run' for b in rings):
        out.append({'shape': 'band', 'fill': sty.surface, 'stroke': sty.run_ring,
                    'text': ring_text('run', 0), 'class': ring_class('run', 0)})
    if any(b['ring'] == 'solve' and not answered(b['type']) for b in rings):
        out.append({'shape': 'open', 'fill': sty.band(0) or sty.surface,
                    'stroke': sty.unanswered, 'text': 'no/too many drivers',
                    'class': ring_class('solve', 0)})
    if any(b['type'] == 'unsettled' for b in rings):
        out.append({'shape': 'band', 'fill': sty.band(0) or sty.surface,
                    'stroke': sty.unanswered, 'text': 'nesting still open',
                    'class': ring_class('solve', 0)})
    return out


# ============================================================================ builder
def problems_first(names: Sequence[Label], groups: Iterable[Sequence[Label]],
                   is_problem: Callable[[Label], bool]) -> list[Label]:
    '''
    `names` reordered so that within each group the problems come first: a group's members
    are put back into the positions the group already holds, so nothing outside it moves.
    `is_problem` says which names are problems. For a diagonal in binding order, each
    component a group; one in `sequencing.sequence` order has its problems first already.
    '''
    order = list(names)
    pos   = {name: i for i, name in enumerate(order)}
    for group in groups:
        slots   = sorted(pos[n] for n in group if n in pos)
        members = [order[i] for i in slots]
        for i, name in zip(slots, sorted(members, key=lambda n: not is_problem(n))):
            order[i] = name
    return order


def xdsm_struct(
    graph  : NestingGraph | ExecutableGraph | Plan,
    *,
    blocks : bool | ExecutableGraph = False,
    formatter : Formatter = PlainFormat(),
    problem_first : bool = True,
) -> dict:
    '''
    What an XDSM of `graph` says, in plain names: every step and variable keyed by its plain
    spelling (`spell_plain`); two spelled alike are refused (`SpelledAlike`).

    `blocks`: ring every solve and rebind the diagonal into run order
    (`sequencing.ordered_graph`); an `ExecutableGraph` rings as that proof has it.
    `formatter`: how each name is shown (`shown_steps`, `shown_vars`).
    `problem_first`: problems first on the diagonal. With `blocks` the run order puts each
    level's problems first (`ordered_graph`); without, each component's problems are moved to
    its front (`problems_first`). Off keeps the graph's order either way.

    Keys: `steps` (the diagonal), `reads` / `writes` (step to variables), `producer`
    (variable to its step; a read with none is a boundary input), `shown_steps` /
    `shown_vars` (`(markup, mark)` per name), `problem_types` (a condition node to its
    category's name, `'stated'` without one), `stated_shapes` (a requirement to the category it will take),
    `driven` (the driven steps), `kinds` (each category present, its `Kind`); when present,
    `slots` (a condition node's slots, `(label, reads)` each), `step_groups` (step to
    its enclosing name, as the N2 groups it) and `blocks` (one record per ring, outermost
    first: `steps`, `ring` `'solve'` or `'run'`, `depth`, `type`).
    '''
    if isinstance(graph, ExecutableGraph):                          # drawn as its own rings
        blocks = graph
    graph = graph_drawn(graph)
    if not graph.nodes:
        raise ValueError('nothing to draw: the graph has no nodes')
    first(chain(spelled_apart(graph.nodes), spelled_apart(graph.graph.variables)))

    if isinstance(blocks, ExecutableGraph):
        if set(blocks.graph.nodes) != set(graph.nodes):
            raise ValueError('the blocking is not a blocking of this graph')
        graph, blocks = blocks.graph, True
    ringed : tuple[Block, ...] | None = None
    if blocks:
        # the diagonal in run order, each level's problems first, unless `problem_first` is
        # off: a hand order then stands
        graph  = ordered_graph(graph) if problem_first else graph
        ringed = walk(graph)
        names  = list(graph.nodes)
    else:
        # the diagonal in binding order, each component's problems moved to its front
        names = problems_first(
            graph.nodes, graph.graph.components, lambda n: problem_type_of(graph[n]) is not None,
        ) if problem_first else list(graph.nodes)

    def step(name: Label) -> str:
        return spell_plain(name)

    def var(v: Place) -> str:
        return spell_plain(v)

    def show(path: Label) -> str:
        return marked(shown(path, formatter))

    def group_of(name: Label) -> str | None:
        # the enclosing name (`.fem_model.construct` under `fem_model`), as the N2 reads it
        return '.'.join(found) if (found := namespace_of(name)) else None

    struct = {
        'steps'   : [step(n) for n in names],
        'reads'   : {step(n): list(dict.fromkeys(var(v) for v in graph[n].reads))
                     for n in names},
        'writes'  : {step(n): [var(v) for v in graph[n].owns] for n in names},
        'producer': {var(v): step(owner) for v, owner in graph.graph.owners.items()},
        'shown_steps': {step(n): shown(n, formatter) for n in names},
        'shown_vars' : {var(v): shown(v, formatter) for v in graph.graph.variables},
        'problem_types'  : {step(n): s for n in names
                     if (s := problem_type_of(graph[n])) is not None},
        'stated_shapes'  : {step(n): s for n in names
                     if (s := stated_shape(graph[n])) is not None},
        'driven'  : {step(n) for n in names if is_driven(graph[n])},
    }
    struct['kinds'] = by_name(d.kind for n in names if (d := drawn_of(graph[n])) is not None)
    # every condition node's slots; `render_xdsm_svg` draws a requirement's and a `slotted`
    # category's, or all
    def label(parts: tuple) -> str:
        return ''.join(formatter.text(p) if isinstance(p, Text) else show(p) for p in parts)

    def slots_of(name: Label) -> list:
        node  = graph[name]
        found = drawn_of(node)
        if found is None:
            return []
        return [(label(slot.label), [var(v) for v in slot.reads]) for slot in found.slots]

    if slots := {step(n): sl for n in names if (sl := slots_of(n))}:
        struct['slots'] = slots
    rank = {n: i for i, n in enumerate(names)}
    if groups := {step(n): g for n in names if (g := group_of(n)) is not None}:
        struct['step_groups'] = groups
    if ringed is not None:
        struct['blocks'] = [{'steps': [step(n) for n in sorted(b.nodes, key=rank.__getitem__)],
                             'ring': 'solve' if b.solved else 'run', 'depth': b.depth,
                             'type': block_type(b)} for b in ringed]

    return struct


__all__ = ['key_entries', 'ring_class', 'ring_entries', 'ring_text', 'xdsm_struct']
