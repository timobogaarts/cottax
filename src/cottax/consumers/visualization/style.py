'''
How every drawing is coloured and marked: a `Style`, and the three shipped (`STYLES`).
'''

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping

from ...core.frozen import Frozen

type Pair = tuple[str, str]
'''`(fill, stroke)`.'''


def mix(colour: str, towards: str, t: float) -> str:
    '''`colour` taken `t` of the way to `towards`, per channel in sRGB.'''
    a = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(towards[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * t):02x}' for x, y in zip(a, b))


def _oklab(colour: str) -> tuple[float, float, float]:
    '''`colour` in Oklab: lightness, then the two opponent axes.'''
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
               for c in (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)))
    l, m, s = ((0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3),
               (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3),
               (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _srgb(lightness: float, a: float, b: float) -> str:
    '''An Oklab colour as `#rrggbb`, clipped to the gamut per channel.'''
    l, m, s = ((lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3,
               (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3,
               (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3)
    rgb = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
           -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
           -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
    gamma = (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
             for c in (min(max(c, 0.0), 1.0) for c in rgb))
    return '#' + ''.join(f'{round(c * 255):02x}' for c in gamma)


class Ground(Frozen):
    '''
    The ground a style is drawn on, as the rule taking a category's colour -- its
    `(fill, stroke)` on a light ground, `Kind.colour` -- to the pair drawn on this one. A
    category is coloured once, by its language, and every ground draws it.
    '''

    @abstractmethod
    def paint(self, colour: Pair) -> Pair:
        ...

    @property
    @abstractmethod
    def scheme(self) -> str:
        '''The CSS `color-scheme` it is: `'light'` or `'dark'`.'''


class Light(Ground):
    '''A light ground: the pair as the category gives it.'''

    def paint(self, colour: Pair) -> Pair:
        return colour

    @property
    def scheme(self) -> str:
        return 'light'


class Dark(Ground):
    '''
    A dark ground: each colour keeps its hue (in Oklab), takes a lightness of its own and
    its chroma scaled -- the fill dark and muted, the stroke light, so a hexagon reads on
    the ground and its contour on the hexagon.
    '''

    fill_lightness   : float
    fill_chroma      : float
    stroke_lightness : float
    stroke_chroma    : float

    def paint(self, colour: Pair) -> Pair:
        def relit(c: str, lightness: float, chroma: float) -> str:
            _, a, b = _oklab(c)
            return _srgb(lightness, a * chroma, b * chroma)
        fill, stroke = colour
        return (relit(fill, self.fill_lightness, self.fill_chroma),
                relit(stroke, self.stroke_lightness, self.stroke_chroma))

    @property
    def scheme(self) -> str:
        return 'dark'


class Style(Frozen):
    '''
    The colours and marks of every drawing in this package. A problem's hue is its
    category's; value is structure; fill and contour are how far a problem is decided.

    `surface` is the ground, `ink` a box's label, `muted` a boundary label or a key entry.
    `model`, `variable` and `boundary` are `(fill, stroke)` for a box that runs, a
    coupling and the two boxes standing for the world outside; `variable_ink` a variable's
    name. `edge` a coupling line. `chip` is a condition slot's `(fill, stroke)`, drawn
    translucent over its problem.

    A category's colour is its own (`Kind.colour`), and `ground` paints it (`paint`): the
    hexagon of a problem of that category, and the contour of the ring it answers. A
    renderer holding a category's name takes the colour from its struct. `stated` is a
    condition that determines nothing and whose type is unknown; one whose type is known
    is drawn hollow in that type's stroke. `unanswered` is the contour of a ring no driver
    answers (a bare cycle, or several problems with nothing nested).

    `bands` fills a solve ring, one entry per nesting depth (cycling); empty draws the ring
    as a contour alone. `band_stroke` is the contour of a band's key sample, `dashed`
    whether a solve ring's contour is dashed. `run_ring` is the contour of a block merely
    run in order, `None` to draw none.

    `undriven` says how a problem without a driver is drawn: `'tint'` fills it `tint` of
    the way to its type's fill (`driven_tint` once driven) and cuts its contour, and a ring
    nothing answers, open by `cut_em` in the middle of top and bottom; `'solid'` draws
    every problem alike. `bold` sets every name in a heavier weight. `font` is a CSS family
    list, `None` for the renderer's own. `ring_gap` is a solve ring's clearance as a
    fraction of the gap between two boxes, `nest_em` what a nested ring and a driver's
    chevron stand off a ring (in ems, floored at that clearance), `ring_radius_em` a
    ring's corner.

    `groups` is the categorical palette a drawing colours groups of nodes with; it keeps
    clear of the categories' hues.
    '''

    name           : str
    surface        : str
    ink            : str
    muted          : str
    model          : Pair
    variable       : Pair
    variable_ink   : str
    boundary       : Pair
    edge           : str
    chip           : Pair
    ground         : Ground
    stated         : Pair
    unanswered     : str
    bands          : tuple[str, ...]
    band_stroke    : str
    dashed         : bool
    run_ring       : str | None
    undriven       : str
    tint           : float
    driven_tint    : float
    cut_em         : float
    bold           : bool
    font           : str | None
    ring_gap       : float
    nest_em        : float
    ring_radius_em : float
    groups         : tuple[str, ...]

    def paint(self, colour: Pair | None) -> Pair:
        '''The `(fill, stroke)` of a problem of a category coloured `colour`, or
        `stated`'s for none.'''
        return self.stated if colour is None else self.ground.paint(colour)

    def fill(self, colour: Pair | None, driven: bool) -> str:
        '''What a problem of a category coloured `colour` is filled with, driven or not;
        `stated`'s grey, tinted as any, for none.'''
        fill = self.paint(colour)[0]
        if self.undriven != 'tint':
            return fill
        return mix(fill, self.surface, 1 - (self.driven_tint if driven else self.tint))

    def band(self, depth: int) -> str | None:
        '''The fill of a solve ring at `depth`, or `None` for a contour alone.'''
        return self.bands[depth % len(self.bands)] if self.bands else None

    @property
    def cuts(self) -> bool:
        '''Whether a problem without a driver has its contour cut open.'''
        return self.undriven == 'tint' and self.cut_em > 0


GROUP_HUES = ('#4c78a8', '#f58518', '#3fa7b8', '#e45756', '#cc79a7', '#5c9ecf',
              '#f2a35e', '#ff9da6', '#8ab6d6', '#bab0ac', '#d98c8c', '#6b8fb8')
'''Mid-luminance, readable on white and on black, and clear of green, ocher and violet,
which a problem's category may wear.'''


TERRACOTTA = Style(
    name='terracotta',
    surface='#ffffff', ink='#3d3330', muted='#7d716c',
    model=('#f8ece7', '#b5583f'),                 # terracotta: something that runs
    variable=('#e9f0f6', '#5b83a6'),              # steel: a variable
    variable_ink='#39617f',
    boundary=('#f1ece9', '#8a7d76'),              # warm neutral: outside the model
    edge='#b9aea9',
    chip=('#fff', '#5e5e5e'),
    ground=Light(),                               # each category's own hue
    stated=('#d9d9d9', '#5e5e5e'),                # graphite: no driver, no hue
    unanswered='#9d9d9d',
    bands=('#f0ede8', '#e5dfd3', '#d9d1c1'),      # L* 94, 89, 84: one step per level
    band_stroke='#bdb4a9', dashed=False, run_ring=None,
    undriven='tint', tint=0.65, driven_tint=0.8, cut_em=0.5,
    bold=False, font=None,
    ring_gap=1.0, nest_em=0.7, ring_radius_em=0.3,
    groups=GROUP_HUES,
)
'''The default: terracotta models and steel variables, a solve a neutral band ringed in its
driver's hue, a problem without a driver tinted with its contour cut open.'''

PYXDSM = Style(
    name='pyxdsm',
    surface='#ffffff', ink='#1a1a1a', muted='#555555',
    model=('#e4f2e0', '#3f9c35'),
    variable=('#f3ddd0', '#c1704f'),
    variable_ink='#8a3a1f',
    boundary=('#eef1f5', '#8c9bb0'),
    edge='#8a8a8a',
    chip=('#fff', '#5e5e5e'),
    ground=Light(),
    stated=('#f1f2f4', '#8c9bb0'),
    unanswered='#8c9bb0',
    bands=(), band_stroke='#e08a2b', dashed=True, run_ring='#8c9bb0',
    undriven='solid', tint=1.0, driven_tint=1.0, cut_em=0.0,
    bold=True, font=None,
    ring_gap=0.5, nest_em=0.0, ring_radius_em=0.6,
    groups=GROUP_HUES,
)
'''pyXDSM's green / orange / white, the convention in that literature: rings as dashed
contours in their driver's hue, every problem drawn alike, in its category's own colour.'''

DARK = Style(
    name='dark',
    surface='#1a1a19', ink='#f0f0f0', muted='#aaaaaa',
    model=('#4a2c22', '#d9795c'),
    variable=('#1f3347', '#6f9cc4'),
    variable_ink='#a9c8e4',
    boundary=('#2b2f33', '#8c959e'),
    edge='#6b6a65',
    chip=('#1a1a19', '#aaaaaa'),
    ground=Dark(fill_lightness=0.34, fill_chroma=0.77,
                stroke_lightness=0.74, stroke_chroma=1.13),
    stated=('#333333', '#a0a0a0'),
    unanswered='#6e6e6e',
    bands=('#242322', '#2e2c2a', '#383532'),
    band_stroke='#6e6a66', dashed=False, run_ring=None,
    undriven='tint', tint=0.65, driven_tint=0.8, cut_em=0.5,
    bold=False, font=None,
    ring_gap=1.0, nest_em=0.7, ring_radius_em=0.3,
    groups=GROUP_HUES,
)
'''`TERRACOTTA`'s reading on a dark ground, each category's colour relit for it.'''

STYLES : Mapping[str, Style] = {s.name: s for s in (TERRACOTTA, PYXDSM, DARK)}
'''The styles by name, for any drawing's `style`.'''


def style_of(style: 'Style | str') -> Style:
    '''`style`, or the one of that name.'''
    if isinstance(style, Style):
        return style
    try:
        return STYLES[style]
    except KeyError:
        raise ValueError(f'no style named {style!r}: one of {sorted(STYLES)}') from None
