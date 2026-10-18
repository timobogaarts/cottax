'''
Drawings for a LaTeX document: names set as LaTeX (`text`), and figures sized as the page
says: the XDSM (`xdsm`), the N2 (`matrix`) and node-link through `dot` (`dot`). The three
figure modules import `cairosvg` / `graphviz` where used, so the formatter alone needs
nothing installed.
'''

from .text import LatexFormatter, latex_to_markup, subscripted, tex

__all__ = ['LatexFormatter', 'latex_to_markup', 'subscripted', 'tex']
