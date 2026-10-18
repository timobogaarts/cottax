'''
The XDSM as a view of an interactive page (`interactive.render_html`): `render_xdsm_svg`'s
picture, what hovering it says, and what an op changed in it.
'''

from __future__ import annotations

from .page import TIP_VARS, Pictures, changed
from .sequencing import graph_drawn
from .style import TERRACOTTA, style_of
from .lookup import spell_plain
from .interface import Formatter
from .naming import PlainFormat, marked, markup_html
from .xdsm import render_xdsm_svg
from .xdsm_struct import key_entries, xdsm_struct

NAMES_CSS = (':host([data-nonames]) .lbl,:host([data-nonames]) .dlbl,'
             ':host([data-nonames]) .deco{display:none}')
'''What the **Names** toggle hides in an XDSM: every name, and a minted name's caret with
it. The boxes stay where they are, cut to the names they would hold.'''


def _rows(struct: dict) -> list[dict]:
    '''What hovering a step says: its name, type and ports, as HTML.'''
    steps, vars_ = struct['shown_steps'], struct['shown_vars']

    def html_of(name, table):
        return markup_html(marked(tuple(table[name])))

    def ports(names):
        seen = list(dict.fromkeys(names))
        return [html_of(v, vars_) for v in seen[:TIP_VARS]], len(seen)

    rows = []
    for s in struct['steps']:
        reads, nr = ports(struct['reads'][s])
        writes, nw = ports(struct['writes'][s])
        rows.append({'name': html_of(s, steps),
                     'group': markup_html(struct.get('step_groups', {}).get(s, '')),
                     'kind': struct['problem_types'].get(s),
                     'driven': s in struct['driven'],
                     'reads': reads, 'nr': nr, 'writes': writes, 'nw': nw})
    return rows


def xdsm_view(graph, pictures: Pictures, before=None, *, blocks=False,
              formatter: Formatter = PlainFormat(), **kw) -> dict:
    '''One frame's XDSM, as a page view (`page.page`): its picture added to `pictures`,
    what hovering a step says, and the steps changed since `before`. A proof is drawn as
    its graph, ringed as proved. `kw` is `render_xdsm_svg`'s.'''
    struct = xdsm_struct(graph, blocks=blocks, formatter=formatter,
                         problem_first=kw.get('problem_first', True))
    # the key is the page's, drawn where panning cannot take it
    svg = str(render_xdsm_svg(struct, formatter=formatter, **{**kw, 'legend': False}))
    steps = {name: i for i, name in enumerate(struct['steps'])}
    lit = [steps[w] for n in changed(before, graph_drawn(graph)) if (w := spell_plain(n)) in steps]
    sty = style_of(kw.get('style', TERRACOTTA))
    return {'svg': pictures.add(svg), 'rows': _rows(struct), 'changed': lit,
            'legend': key_entries(struct, sty)}


__all__ = ['NAMES_CSS', 'xdsm_view']
