'''
Pictures of a graph: XDSM, N2, node-link, and a page framing them.
'''

from .style import DARK, PYXDSM, STYLES, TERRACOTTA, Style
from .naming import PlainFormat
from .xdsm import render_xdsm_svg
from .interactive import render_html
from .matrix import by_namespace, render_matrix_svg
from .visualization import draw, render, to_dot


__all__ = [
    # how every drawing is coloured, and how a name is written
    'Style', 'STYLES', 'TERRACOTTA', 'PYXDSM', 'DARK', 'PlainFormat',

    # xdsm, the N2 matrix (`by_namespace`, its groups), either or both as a page
    'render_xdsm_svg', 'render_matrix_svg', 'by_namespace', 'render_html',

    # graphs directly
    'draw', 'render', 'to_dot',
]
