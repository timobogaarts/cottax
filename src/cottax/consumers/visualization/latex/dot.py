r'''
Node-link drawings for a LaTeX document: `visualization.to_dot` with the labels set in
LaTeX, written by `dot` as the PDF the document includes.

A label a `LatexFormatter` marked (`naming.MARKUP`) is written as graphviz's HTML-like label,
set by `dot` with no LaTeX pass. The boundary is collapsed to the XDSM's two boxes,
`\mathcal{I}` and `\mathcal{O}`.

Needs `graphviz` (the Python package and `dot`).
'''

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile

from ..visualization import to_dot as _to_dot
from ..interface import Formatter
from .text import LatexFormatter
from .xdsm import INPUTS, LATIN_MODERN, OUTPUTS, _panel_kw


def to_dot(graph, *, formatter: Formatter | None = None, font: str | None = LATIN_MODERN,
           font_size: int | None = None, edge_font_size: int | None = None,
           nodesep: float | None = None, ranksep: float | None = None,
           node_margin: str | None = None, **kw) -> str:
    '''
    `visualization.to_dot` set in LaTeX, with a figure's defaults: the boundary collapsed
    to `\\mathcal{I}` / `\\mathcal{O}`, no page margin (the cairo back end otherwise puts
    36 pt of blank around a PDF), and `font` for the drawing face. `font_size`,
    `edge_font_size`, `nodesep`, `ranksep` and `node_margin` (`"x,y"` in inches) retune
    the type and the spacing; `None` keeps `to_dot`'s.
    '''
    kw.setdefault('collapse_boundary', True)
    kw.setdefault('input_label', INPUTS)
    kw.setdefault('output_label', OUTPUTS)
    kw.setdefault('margin', 0)
    tuned = {k: v for k, v in (('font_size', font_size), ('edge_font_size', edge_font_size),
                               ('nodesep', nodesep), ('ranksep', ranksep),
                               ('node_margin', node_margin)) if v is not None}
    return _to_dot(graph, formatter=formatter or LatexFormatter(), font=font, **tuned, **kw)


def render(graph, **kw):
    '''A `graphviz.Source`, which a notebook shows inline. Arguments are `to_dot`'s.'''
    import graphviz
    return graphviz.Source(to_dot(graph, **kw))


def save(graph, path, **kw) -> pathlib.Path:
    '''The drawing written as the format `path`'s suffix names. Arguments are `to_dot`'s.'''
    return _write(to_dot(graph, **kw), path)


def _extent(src: str) -> tuple[float, float]:
    '''A source's drawn size in points: the PDF page, since `to_dot` writes no margin.'''
    svg = _pipe(src, 'svg').decode()
    _x, _y, w, h = re.search(r'viewBox="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"', svg).groups()
    return float(w), float(h)


DOT_PAD = 4 / 72
'''`dot`'s own margin around a drawing, in inches: what a padded panel's `pad` adds to.'''


def save_panels(panels, *, formats=('pdf',), canvas=None, **kw) -> list[pathlib.Path]:
    '''
    Several drawings as one figure's panels, at **one zoom**: each padded (`dot`'s own
    `pad`, evenly, so a smaller graph sits centred) to the largest page. `panels` is
    `(stem, graph)` or `(stem, graph, kwargs)`; `canvas=(w, h)` in points pins the page.
    '''
    srcs = [(pathlib.Path(stem), to_dot(g, **_panel_kw(kw, rest)))
            for stem, g, *rest in panels]
    sizes = [_extent(src) for _stem, src in srcs]
    if canvas is None:
        canvas = (max(w for w, _h in sizes), max(h for _w, h in sizes))
    paths = []
    for (stem, src), (w, h) in zip(srcs, sizes):
        if canvas[0] < w - 0.5 or canvas[1] < h - 0.5:
            raise ValueError(f'{stem}: canvas {canvas} is smaller than the drawing ({w}, {h})')
        pad = (DOT_PAD + (canvas[0] - w) / 2 / 72, DOT_PAD + (canvas[1] - h) / 2 / 72)
        src = src.replace('{\n', f'{{\n  pad="{pad[0]:.4f},{pad[1]:.4f}";\n', 1)
        paths += [_write(src, stem.with_suffix('.' + fmt)) for fmt in formats]
    return paths


def save_row(graphs, path, *, labels=True, sep=24, **kw) -> pathlib.Path:
    '''
    Several drawings as one figure, side by side, labelled `(a)`, `(b)`, ... inside it
    (`labels=False` drops them, a sequence of strings replaces them). `sep` is the gap in
    points; `formatter` may be one per graph.
    '''
    fmts = kw.pop('formatter', None)
    if fmts is None or isinstance(fmts, Formatter):
        fmts = [fmts] * len(graphs)
    if labels is True:
        labels = [f'({chr(ord("a") + i)})' for i in range(len(graphs))]
    elif labels is False:
        labels = [None] * len(graphs)
    font = kw.get('font', LATIN_MODERN)
    header, panels = [], []
    for i, (graph, fmt, label) in enumerate(zip(graphs, fmts, labels)):
        attrs, body = _split_panel(to_dot(graph, formatter=fmt, **kw), i)
        header += [a for a in attrs if a not in header]
        cap = (f'    label="{label}"; labelloc="b"; labeljust="c";\n'
               f'    fontname="{font}"; fontsize=10;\n') if label else ''
        panels.append(f'  subgraph cluster_p{i} {{\n'
                      f'    color="none"; penwidth=0; margin=8; sortv={i};\n'
                      f'{cap}{body}  }}')
    # each panel its own component, packed in one row in the order given, `sep` points apart
    header += [f'pack={sep};', 'packmode="array_u1";']
    src = 'digraph cottax {\n  ' + '\n  '.join(header) + '\n' + '\n'.join(panels) + '\n}\n'
    return _write(src, path)


def _split_panel(src: str, index: int) -> tuple[list[str], str]:
    '''One source as (graph attributes, body), its vertex ids and clusters prefixed per
    panel so two panels cannot collide.'''
    inner = src[src.index('{') + 1:src.rindex('}')]
    inner = re.sub(r'\bn(\d+)\b', lambda m: f'p{index}n{m.group(1)}', inner)
    inner = re.sub(r'\bcluster_(\d+)\b', lambda m: f'cluster_p{index}_{m.group(1)}', inner)
    attrs, body = [], []
    for line in inner.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if '->' in stripped or '[' in stripped or stripped.startswith(('subgraph', '}', 'style=')):
            body.append('  ' + line.rstrip())
            continue
        attrs += [a.strip() + ';' for a in stripped.split(';') if a.strip()]
    return attrs, '\n'.join(body) + '\n'


def _write(src: str, path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.write_bytes(_pipe(src, path.suffix.lstrip('.').lower() or 'svg'))
    return path


def _pipe(src: str, fmt: str) -> bytes:
    '''`dot` run on `src` for `fmt`, with the fonts of `_fonts_conf` in view.'''
    import graphviz
    conf = _fonts_conf()
    before = os.environ.get('FONTCONFIG_FILE')
    if conf:
        os.environ['FONTCONFIG_FILE'] = conf
    try:
        return graphviz.Source(src).pipe(format=fmt)
    finally:
        if conf:
            if before is None:
                del os.environ['FONTCONFIG_FILE']
            else:
                os.environ['FONTCONFIG_FILE'] = before


def _fonts_conf() -> str | None:
    '''
    A fontconfig file for `dot` adding TeX Live's Latin Modern OpenType directory (found
    with `kpsewhich`) to the config `dot` would read, since pango takes no Type 1 font; or
    `None`, leaving `dot`'s alone, when `FONTCONFIG_FILE` is set or nothing is found.
    '''
    if os.environ.get('FONTCONFIG_FILE'):
        return None
    dot = shutil.which('dot')
    base = next((c for c in [
        pathlib.Path(dot).resolve().parents[1] / 'etc' / 'fonts' / 'fonts.conf' if dot else None,
        pathlib.Path('/etc/fonts/fonts.conf'),
    ] if c and c.is_file()), None)
    kpsewhich = shutil.which('kpsewhich')
    if base is None or kpsewhich is None:
        return None
    found = subprocess.run([kpsewhich, 'lmroman10-regular.otf'],
                           capture_output=True, text=True).stdout.strip()
    if not found:
        return None
    conf = pathlib.Path(tempfile.gettempdir()) / f'cottax-dot-fonts-{os.getuid()}.conf'
    conf.write_text('<?xml version="1.0"?>\n'
                    '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
                    '<fontconfig>\n'
                    f'  <include>{base}</include>\n'
                    f'  <dir>{pathlib.Path(found).parent}</dir>\n'
                    '</fontconfig>\n')
    return str(conf)


__all__ = ['render', 'save', 'save_panels', 'save_row', 'to_dot']
