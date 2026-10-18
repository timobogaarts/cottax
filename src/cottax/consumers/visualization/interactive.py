'''
A graph, or a `Plan`, as one interactive page: its XDSM and / or its N2 as views.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...core import ExecutableGraph, NestingGraph, Plan
from .matrix import NAMES_CSS as N2_NAMES, by_namespace, colour_toggle, n2_view
from .page import HtmlDoc, Pictures, page, prefixes
from .sequencing import graph_drawn
from .style import TERRACOTTA, Style, style_of
from .interface import Formatter
from .naming import PlainFormat
from .xdsm_html import NAMES_CSS as XDSM_NAMES, xdsm_view

VIEWS = {'xdsm': 'XDSM', 'n2': 'N2'}
'''The pictures a page can carry, by the name `views` takes and the name a button shows.'''


def render_html(
    graph     : NestingGraph | ExecutableGraph | Plan,
    *,
    views     : Sequence[str] = ('xdsm', 'n2'),
    names     : bool = True,
    colour    : bool = True,
    blocks    : bool | ExecutableGraph = False,
    style     : Style | str = TERRACOTTA,
    formatter : Formatter = PlainFormat(),
    xdsm      : Mapping[str, Any] | None = None,
    n2        : Mapping[str, Any] | None = None,
    write     : bool = False,
    file_name : str = 'graph',
    outdir    : str = '.',
) -> HtmlDoc:
    '''
    `graph` as an interactive page: drag to pan, Ctrl+scroll to zoom, **Fit**; hover a step
    or a coupling to name it and light its row and column. A `Plan` is a frame per prefix
    under a slider, labelled by its op, what the op changed haloed.

    `views`: the pictures each frame carries, `'xdsm'` and / or `'n2'`, the first opened.
    `names`, `colour`: the **Names** and (with an N2) **Colour** toggles' starting state.
    `blocks`: ring the XDSM's solves (an `ExecutableGraph`: as that proof has them).
    `xdsm`: `render_xdsm_svg`'s own arguments for the XDSM view.
    `n2`: the N2 view's (`cell`, `font_size`, `order`, `groups` -- `by_namespace()` unless
    said --, `grouped`, `colours`).
    `write=True` also writes `<outdir>/<file_name>.html`, filling the window.
    '''
    unknown = [v for v in views if v not in VIEWS]
    if not views or unknown:
        raise ValueError(f'views are some of {list(VIEWS)}, the first opened; got {list(views)}')
    sty = style_of(style)
    xdsm_kw = dict(xdsm or {})
    n2_kw = dict(n2 or {})
    if n2_kw.get('groups') is None:
        n2_kw['groups'] = by_namespace()
    pictures, frames, before = Pictures(), [], None
    for label, tip, g in prefixes(graph):
        node = graph_drawn(g)
        drawn = []
        for v in views:
            if v == 'xdsm':
                drawn.append(xdsm_view(g, pictures, before, blocks=blocks, formatter=formatter,
                                       style=sty, **xdsm_kw))
            else:
                drawn.append(n2_view(node, pictures, before, style=sty, formatter=formatter,
                                     **n2_kw))
        frames.append({'label': label, 'tip': tip, 'views': drawn})
        before = node
    hide = (XDSM_NAMES if 'xdsm' in views else '') + (N2_NAMES if 'n2' in views else '')
    toggles = [('names', 'Names', names, hide)]
    if 'n2' in views:
        toggles.append(colour_toggle(sty, colour))
    return page(pictures.svgs, frames, style=sty, views=[VIEWS[v] for v in views],
                toggles=toggles, note='drag to pan · ctrl+scroll to zoom · hover a box',
                write=write, file_name=file_name, outdir=outdir)


__all__ = ['VIEWS', 'render_html']
