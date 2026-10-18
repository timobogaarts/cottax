#!/usr/bin/env python3
"""Generate the cottax logo.

The wordmark is Latin Modern Roman 10 Italic in dark red, letters untouched,
with a dataflow arrow in every gap between adjacent letters.

The arrows live in the tuning file as two hand-editable objects apiece, both
rendered verbatim: the *ribbon* (id="arrow-i"), a filled outline like the
letters themselves -- so it can taper, match a glyph's stroke width at the
junction, be as un-constant as it likes -- and the *head* (id="head-i"), a
clean filled triangle the script seeds at the ribbon's tip, which is the one
shape that is miserable to draw by hand.  Drag / rotate the head as a whole
object to align it; node-edit the ribbon like any glyph:

    python assets/make_logo.py --tuning     # writes assets/cottax-tuning.svg
    ... edit ribbons, drag heads in Inkscape, save in place ...
    python assets/make_logo.py              # picks the file up automatically

A gap whose arrow-i is missing from the file falls back to the parametric
starter in CONFIG (which otherwise only seeds the file); delete the file and
rerun --tuning to restart wholesale.  Arrows render *under* the letters, so
ends drawn into the ink vanish behind it -- overlap generously and let the
glyph outline cut the junction.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path as FsPath

import numpy as np
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

# One parametric starter arrow per gap -- ONLY used to seed the tuning file
# (and for gaps whose id is missing from it); a drawn arrow ignores all of it.
# Coordinates in font units (1000/em), y from the baseline; start.x measured
# from the left glyph's right edge, end.x from the right glyph's left edge.
#   start, end : the spine's endpoints
#   bow        : perpendicular bulge of the arc; positive bows upward
#   widths     : shaft half-width at the tail and at the head
#   head       : (length, half-width) of the arrowhead
ARROW = dict(bow=40, widths=(6, 12), head=(55, 34))          # shared defaults

CONFIG = dict(
    font="/usr/share/texmf/fonts/opentype/public/lm/lmroman10-italic.otf",
    word="cottax",
    # extra advance after each letter, font units -- one per gap (a single
    # number applies to all).  Tightened after the o, which LM sets wide.
    gap=[150, 90, 150, 150, 150],
    padding=70,   # around the drawn bbox, font units

    # letters in the dark red, arrows a step lighter, shadow shapes (drawn
    # like the arrows, e.g. closing a terminal's curl) between the two --
    # they belong more to the letters than to the arrows
    light=dict(bg=None, ink="#AD381B", arrow="#C2695E" , shadow="#A64D48"),# shadow="#A63A34"),"#C25B4E"
    dark=dict(bg=None, ink="#E0756A", arrow="#F0A896", shadow="#E88E80"),

    arrows=[  # c->o, o->t, t->t, t->a, a->x
        dict(ARROW, start=(-14, 100), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210), widths = (6, 6)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),
        dict(ARROW, start=(10, 210), end=(-10, 210)),

    ],

    tuning_svg="cottax-tuning.svg",
    png_px=1600,
    icon_png_px=512,
    icon_glyphs=(0, 1),   # the c->o pair
    icon_arrows=(0,),
    icon_shadows=(),      # shadow ids drawn in the icon (none by default)
)


# --------------------------------------------------------------------------
# a tiny path model:  subpath = (start, [seg, ...]),  seg = ('L', p) | ('C', a, b, p)
# --------------------------------------------------------------------------

@dataclass
class Path:
    subpaths: list = field(default_factory=list)

    def moveto(self, p):
        self.subpaths.append((tuple(p), []))
        return self

    def lineto(self, p):
        self.subpaths[-1][1].append(("L", tuple(p)))
        return self

    def curveto(self, a, b, p):
        self.subpaths[-1][1].append(("C", tuple(a), tuple(b), tuple(p)))
        return self

    def polygon(self, pts):
        self.moveto(pts[0])
        for p in pts[1:]:
            self.lineto(p)
        return self

    def translate(self, tx, ty):
        f = lambda p: (p[0] + tx, p[1] + ty)
        out = Path()
        for start, segs in self.subpaths:
            out.subpaths.append(
                (f(start), [(s[0], *[f(q) for q in s[1:]]) for s in segs])
            )
        return out

    def affine(self, m):
        """This path through a 3x3 affine matrix."""
        f = lambda p: tuple((m @ [p[0], p[1], 1.0])[:2])
        out = Path()
        for start, segs in self.subpaths:
            out.subpaths.append(
                (f(start), [(s[0], *[f(q) for q in s[1:]]) for s in segs])
            )
        return out

    def extend(self, other):
        self.subpaths.extend(other.subpaths)
        return self

    def points(self):
        for start, segs in self.subpaths:
            yield start
            for s in segs:
                yield from s[1:]

    def bbox(self):
        pts = np.array(list(self.points()))
        return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()

    def sampled(self, per_seg=24):
        """Each subpath as a dense polyline, for nearest-point/winding queries."""
        out = []
        t = np.linspace(0, 1, per_seg)[:, None]
        for start, segs in self.subpaths:
            pts, cur = [np.array([start], float)], np.asarray(start, float)
            for s in segs:
                if s[0] == "L":
                    p = np.asarray(s[1], float)
                    pts.append((1 - t[1:]) * cur + t[1:] * p)
                    cur = p
                else:
                    a, b, p = (np.asarray(q, float) for q in s[1:])
                    tt = t[1:]
                    pts.append((1 - tt) ** 3 * cur + 3 * (1 - tt) ** 2 * tt * a
                               + 3 * (1 - tt) * tt ** 2 * b + tt ** 3 * p)
                    cur = p
            out.append(np.vstack(pts))
        return out

    def to_svg_d(self, prec=2, close=True):
        n = lambda v: f"{v:.{prec}f}".rstrip("0").rstrip(".")
        out = []
        for start, segs in self.subpaths:
            out.append(f"M{n(start[0])} {n(start[1])}")
            for s in segs:
                if s[0] == "L":
                    out.append(f"L{n(s[1][0])} {n(s[1][1])}")
                else:
                    out.append("C" + " ".join(f"{n(q[0])} {n(q[1])}" for q in s[1:]))
            if close:
                out.append("Z")
        return "".join(out)

    def to_mpl(self):
        from matplotlib.path import Path as MPath

        verts, codes = [], []
        for start, segs in self.subpaths:
            verts.append(start)
            codes.append(MPath.MOVETO)
            for s in segs:
                if s[0] == "L":
                    verts.append(s[1])
                    codes.append(MPath.LINETO)
                else:
                    verts.extend(s[1:])
                    codes.extend([MPath.CURVE4] * 3)
            verts.append((0.0, 0.0))
            codes.append(MPath.CLOSEPOLY)
        return MPath(np.array(verts, float), np.array(codes))


# --------------------------------------------------------------------------
# glyphs
# --------------------------------------------------------------------------

def glyph_path(font, glyphset, ch):
    """The glyph outline as a Path, in font units, y up, origin at the pen."""
    pen = RecordingPen()
    glyphset[font.getBestCmap()[ord(ch)]].draw(pen)
    path = Path()
    for op, args in pen.value:
        if op == "moveTo":
            path.moveto(args[0])
        elif op == "lineTo":
            path.lineto(args[0])
        elif op == "curveTo":
            path.curveto(*args[-3:])
        elif op != "closePath":
            raise ValueError(f"unexpected pen op {op!r} in {ch!r}")
    return path


def layout(cfg):
    """Place the word.  Returns a list of glyph Paths in word coordinates."""
    font = TTFont(cfg["font"])
    gs = font.getGlyphSet()
    cmap = font.getBestCmap()
    gaps = cfg["gap"]
    if not isinstance(gaps, (list, tuple)):
        gaps = [gaps] * len(cfg["word"])
    placed, x = [], 0.0
    for i, ch in enumerate(cfg["word"]):
        placed.append(glyph_path(font, gs, ch).translate(x, 0))
        x += gs[cmap[ord(ch)]].width + (gaps[i] if i < len(gaps) else 0)
    return placed


# --------------------------------------------------------------------------
# spines:  ctrl = np.array [P0, C1, C2, P1, C1, C2, P2, ...]  (3k+1 points)
# --------------------------------------------------------------------------

def default_spine(spec, left_edge, right_edge):
    """The parametric arc from CONFIG, as one cubic segment."""
    p0 = np.array(spec["start"], float) + [left_edge, 0]
    p1 = np.array(spec["end"], float) + [right_edge, 0]
    chord = p1 - p0
    perp = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
    q = (p0 + p1) / 2 + perp * spec["bow"]          # quadratic control
    return np.array([p0, p0 + 2 / 3 * (q - p0), p1 + 2 / 3 * (q - p1), p1])


def sample_spine(ctrl, n=240):
    ctrl = np.asarray(ctrl, float)
    nseg = (len(ctrl) - 1) // 3
    t = np.linspace(0, 1, max(n // nseg, 24) + 1)[:, None]
    pts = []
    for i in range(nseg):
        p0, c1, c2, p1 = ctrl[3 * i: 3 * i + 4]
        tt = t[1:] if i else t
        pts.append((1 - tt) ** 3 * p0 + 3 * (1 - tt) ** 2 * tt * c1
                   + 3 * (1 - tt) * tt ** 2 * c2 + tt ** 3 * p1)
    return np.vstack(pts)


def _fit_cubics(pts, nseg):
    """A dense polyline as `nseg` cubic Beziers, each interpolating four
    arclength-spaced samples -- few nodes, so the outline stays editable."""
    pts = np.asarray(pts, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    at = lambda u: np.array([np.interp(u * s[-1], s, pts[:, 0]),
                             np.interp(u * s[-1], s, pts[:, 1])])
    segs = []
    for i in range(nseg):
        q = [at((i + j / 3) / nseg) for j in range(4)]
        segs.append(((-5 * q[0] + 18 * q[1] - 9 * q[2] + 2 * q[3]) / 6,
                     (2 * q[0] - 9 * q[1] + 18 * q[2] - 5 * q[3]) / 6, q[3]))
    return at(0.0), segs


def ribbon(spine, widths, head=None):
    """A stroke swept along a sampled centerline, as a filled outline of a
    handful of smooth Beziers.  With `head`, the last `head_len` of the
    centerline becomes an arrowhead aligned with the curve's end; without,
    the shaft runs the whole centerline (a hand-placed head will cover it)."""
    spine = np.asarray(spine, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(spine, axis=0), axis=1))])
    if head is None:
        shaft = len(spine)
    else:
        head_len, head_w = head
        shaft = max(np.searchsorted(s, s[-1] - head_len), 2)

    tang = np.gradient(spine, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])

    w = np.interp(s[:shaft] / max(s[shaft - 1], 1e-9), [0, 1], widths)[:, None]
    left = spine[:shaft] + nrm[:shaft] * w
    right = spine[:shaft] - nrm[:shaft] * w

    nseg = max(1, round(s[shaft - 1] / 250))      # ~one cubic per 250 units
    p = Path()
    start, segs = _fit_cubics(right, nseg)
    p.moveto(start)
    for c1, c2, q in segs:
        p.curveto(c1, c2, q)
    if head is not None:
        base, bt = spine[shaft - 1], tang[shaft - 1]
        bn = np.array([-bt[1], bt[0]])
        p.lineto(base - bn * head_w).lineto(spine[-1]).lineto(base + bn * head_w)
    start_l, segs_l = _fit_cubics(left[::-1], nseg)
    p.lineto(start_l)
    for c1, c2, q in segs_l:
        p.curveto(c1, c2, q)
    return p                                      # Z closes the tail cap


def head_glyph(tip, direction, head):
    """The arrowhead: a clean triangle, tip at `tip`, pointing along
    `direction` -- seeded by the script, aligned by hand."""
    head_len, head_w = head
    d = np.asarray(direction, float)
    d = d / max(np.linalg.norm(d), 1e-9)
    base = np.asarray(tip, float) - d * head_len
    n = np.array([-d[1], d[0]])
    return Path().polygon([base - n * head_w, tip, base + n * head_w])


def build_arrows(cfg, placed, tune=None):
    """One arrow per gap as a list of (Path, gradient) pieces: the drawn
    ribbon and the drawn head, verbatim, each keeping its own fade.  A gap
    missing from the tuning file gets the parametric starter."""
    ribbons, heads = (tune[0], tune[1]) if tune is not None else ({}, {})
    
    out = []
    for i, spec in enumerate(cfg["arrows"]):
        
        pieces = []
        if i in ribbons:
            pieces.append(ribbons[i])
        else:
            pts = sample_spine(
                default_spine(spec, placed[i].bbox()[2], placed[i + 1].bbox()[0]))
            pieces.append((ribbon(pts, spec["widths"], spec["head"]), None))
        if i in heads:
            pieces.append(heads[i])
        out.append(pieces)
    return out


# --------------------------------------------------------------------------
# the tuning file: arrow outlines out to Inkscape, and back
# --------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _parse_d(d):
    """SVG path data, verbatim, as a Path (subpaths of lines and cubics)."""
    tokens = re.findall(r"[MmLlHhVvCcSsZz]|" + _NUM.pattern, d)
    path, i, cur, first, cmd, prev_c2 = Path(), 0, np.zeros(2), np.zeros(2), None, None

    def take(n):
        nonlocal i
        vals = [float(t) for t in tokens[i:i + n]]
        i += n
        return vals

    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                cur = first.copy()       # a close returns the pen to the start
                prev_c2 = None
                continue
        rel = cmd.islower()
        C = cmd.upper()
        if C == "M":
            x, y = take(2)
            cur = cur + [x, y] if rel and path.subpaths else np.array([x, y])
            first = cur.copy()
            path.moveto(cur)
            prev_c2 = None
            cmd = "l" if rel else "L"
        elif C in "LHV":
            if C == "L":
                x, y = take(2)
                p = cur + [x, y] if rel else np.array([x, y])
            elif C == "H":
                (x,) = take(1)
                p = np.array([cur[0] + x if rel else x, cur[1]])
            else:
                (y,) = take(1)
                p = np.array([cur[0], cur[1] + y if rel else y])
            path.lineto(p)
            cur, prev_c2 = p, None
        elif C == "C":
            v = take(6)
            a, b, p = (cur + v[k:k + 2] if rel else np.array(v[k:k + 2])
                       for k in (0, 2, 4))
            path.curveto(a, b, p)
            cur, prev_c2 = p, b
        elif C == "S":
            v = take(4)
            b, p = (cur + v[k:k + 2] if rel else np.array(v[k:k + 2]) for k in (0, 2))
            a = 2 * cur - prev_c2 if prev_c2 is not None else cur
            path.curveto(a, b, p)
            cur, prev_c2 = p, b
        else:
            raise ValueError(f"unsupported path command {cmd!r} in tuning SVG")
    return path


def _parse_transform(attr):
    """An SVG transform attribute as a 3x3 matrix."""
    m = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", attr or ""):
        v = [float(t) for t in _NUM.findall(args)]
        t = np.eye(3)
        if name == "matrix":
            t[:2, :] = [[v[0], v[2], v[4]], [v[1], v[3], v[5]]]
        elif name == "translate":
            t[0, 2], t[1, 2] = v[0], v[1] if len(v) > 1 else 0.0
        elif name == "scale":
            t[0, 0], t[1, 1] = v[0], v[1] if len(v) > 1 else v[0]
        elif name == "rotate" and len(v) == 1:
            c, s = np.cos(np.radians(v[0])), np.sin(np.radians(v[0]))
            t[:2, :2] = [[c, -s], [s, c]]
        else:
            raise ValueError(f"unsupported transform {name!r} in tuning SVG")
        m = m @ t
    return m


def _ellipse_path(cx, cy, rx, ry):
    """An ellipse as four cubic arcs, so a drawn <circle>/<ellipse> needs no
    Object-to-Path in Inkscape."""
    k = 0.5522847498
    return (Path()
            .moveto((cx + rx, cy))
            .curveto((cx + rx, cy + k * ry), (cx + k * rx, cy + ry), (cx, cy + ry))
            .curveto((cx - k * rx, cy + ry), (cx - rx, cy + k * ry), (cx - rx, cy))
            .curveto((cx - rx, cy - k * ry), (cx - k * rx, cy - ry), (cx, cy - ry))
            .curveto((cx + k * rx, cy - ry), (cx + rx, cy - k * ry), (cx + rx, cy)))


def _fill_url(el):
    """The id of the gradient this element is filled with, or None."""
    fill = el.get("fill") or ""
    m = re.match(r"url\(#(.+)\)", fill)
    if m is None:
        m = re.search(r"fill\s*:\s*url\(#([^)]+)\)", el.get("style") or "")
    return m.group(1) if m else None


def _resolve_gradient(gid, gradients, to_user, bbox):
    """A drawn linearGradient as (p1, p2) in word coordinates.

    Inkscape splits a dragged gradient over an href chain -- coordinates on
    the outer element, stops on the one it references -- and this follows it.
    Only the fade's *geometry* is kept: what a gradient means here is
    'ink at p1, running to the shape's own colour by p2', so the drawn stops'
    colours and opacities are ignored entirely."""
    el, coords, gt, units = gradients.get(gid), {}, None, None
    seen = set()
    while el is not None and gid not in seen:
        seen.add(gid)
        for k in ("x1", "y1", "x2", "y2"):
            coords.setdefault(k, el.get(k))
        if gt is None:
            gt = el.get("gradientTransform")
        if units is None:
            units = el.get("gradientUnits")
        gid = (el.get("{http://www.w3.org/1999/xlink}href")
               or el.get("href") or "").lstrip("#")
        el = gradients.get(gid)

    def num(k, default):
        v = coords.get(k)
        if v is None:
            return default
        return float(v.rstrip("%")) / 100 if v.endswith("%") else float(v)

    if units == "userSpaceOnUse":
        pts = np.array([[num("x1", 0), num("y1", 0), 1],
                        [num("x2", 1), num("y2", 0), 1]], float)
        m = to_user @ _parse_transform(gt)
        return tuple((pts @ m.T)[:, :2])
    x0, y0, x1, y1 = bbox                        # objectBoundingBox fractions
    p1 = (x0 + num("x1", 0) * (x1 - x0), y0 + num("y1", 0) * (y1 - y0))
    p2 = (x0 + num("x2", 1) * (x1 - x0), y0 + num("y2", 0) * (y1 - y0))
    return (np.array(p1), np.array(p2))


def read_arrows(path):
    """({i: ribbon}, {i: head}, {id: shadow}) -- each value a
    (Path, gradient-or-None) pair, verbatim in word coordinates.

    An element's geometry is in whatever frame Inkscape left it in;
    accumulating the transforms from the root gives viewport coordinates, and
    re-expressing those in the arrows *layer's* frame (the layer carries the
    y-flip) gives word coordinates back -- robust to Inkscape regrouping or
    adding element-level transforms (dragging a head adds one), as long as
    everything stays inside its layer.  A shadow shape (id="shadow-<n>") is
    drawn exactly like an arrow ribbon and rendered in the shadow colour; a
    cut shape (id="cut-<n>") is a hole punched through the whole drawing
    (the page shows through, like a counter); a <circle> or <ellipse> works
    for any id, not only a path.  A gradient dragged onto a shape keeps its
    direction and extent, and renders as ink at the start handle running to
    the shape's own colour at the end -- so drag from the letter outward."""
    root = ET.parse(path).getroot()
    found, layer = {}, [None]
    LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
    gradients = {el.get("id"): el for el in root.iter()
                 if el.tag.rsplit("}", 1)[-1] == "linearGradient"}

    def walk(el, m):
        m = m @ _parse_transform(el.get("transform"))
        if el.get(LABEL) == "arrows":
            layer[0] = m
        tag = el.tag.rsplit("}", 1)[-1]
        pid = el.get("id", "")
        if pid.startswith(("arrow-", "head-", "shadow-", "cut-")):
            if tag == "path":
                found[pid] = (m, _parse_d(el.get("d")), _fill_url(el))
            elif tag in ("circle", "ellipse"):
                cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
                rx = float(el.get("rx") or el.get("r"))
                ry = float(el.get("ry") or el.get("r"))
                found[pid] = (m, _ellipse_path(cx, cy, rx, ry), _fill_url(el))
        for child in el:
            walk(child, m)

    walk(root, np.eye(3))
    if layer[0] is None:
        raise ValueError(f'{path}: no layer labelled "arrows" -- '
                         f"regenerate with --tuning")
    to_word = np.linalg.inv(layer[0])
    ribbons, heads, shadows, cuts = {}, {}, {}, {}
    for pid, (m, p, gid) in found.items():
        q = p.affine(to_word @ m)
        if pid.startswith("cut-"):
            cuts[pid] = q
            continue
        grad = (_resolve_gradient(gid, gradients, to_word @ m, q.bbox())
                if gid else None)
        if pid.startswith("shadow-"):
            shadows[pid] = (q, grad)
            continue
        exact = re.fullmatch(r"(arrow|head)-(\d+)", pid)
        if exact is None:
            # Inkscape renames a colliding id by appending a suffix
            # (arrow-1 copied -> arrow-1-6); render nothing it half-names
            print(f"note: ignoring {pid!r} -- not an exact arrow-<n> / "
                  f"head-<n> id (a stray Inkscape duplicate?)")
            continue
        kind, num = exact.groups()
        (ribbons if kind == "arrow" else heads)[int(num)] = (q, grad)
    return ribbons, heads, shadows, cuts


def write_tuning_svg(path, cfg):
    """Letters plus the current arrows as editable *filled outlines* --
    the same kind of object the letters are."""
    placed = layout(cfg)
    x0, y0, x1, y1 = Path().extend(placed[0]).extend(
        Path(sum([p.subpaths for p in placed[1:]], []))).bbox()
    pad = 3 * cfg["padding"]
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    flip = f'transform="translate({-x0:.2f},{y1:.2f}) scale(1,-1)"'
    ink, arrow = cfg["light"]["ink"], cfg["light"]["arrow"]

    letters = "".join(f'<path d="{p.to_svg_d()}" fill="{ink}" fill-opacity="0.35"/>'
                      for p in placed)
    arrows = []
    for i, spec in enumerate(cfg["arrows"]):
        ctrl = default_spine(spec, placed[i].bbox()[2], placed[i + 1].bbox()[0])
        pts = sample_spine(ctrl)
        s = np.concatenate([[0.0],
                            np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        # the ribbon stops inside the head, so the seeded pair already overlaps
        shaft = ribbon(pts[s <= s[-1] - 0.6 * spec["head"][0]], spec["widths"])
        head = head_glyph(pts[-1], pts[-1] - pts[-8], spec["head"])
        arrows.append(f'<path id="arrow-{i}" d="{shaft.to_svg_d()}" fill="{arrow}"/>'
                      f'<path id="head-{i}" d="{head.to_svg_d()}" fill="{arrow}"/>')

    NS = ('xmlns="http://www.w3.org/2000/svg" '
          'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
          'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd"')
    svg = (
        f'<svg {NS} viewBox="0 0 {x1 - x0:.2f} {y1 - y0:.2f}">'
        f'<g inkscape:groupmode="layer" inkscape:label="letters (locked)" '
        f'sodipodi:insensitive="true" {flip}>{letters}</g>'
        f'<g inkscape:groupmode="layer" inkscape:label="arrows" {flip}>'
        f'{"".join(arrows)}</g></svg>'
    )
    FsPath(path).write_text(svg)
    return path


# --------------------------------------------------------------------------
# assembly and output
# --------------------------------------------------------------------------

def build(cfg, tune=None, glyphs=None, arrows=None, shadows=None):
    """Returns ([(path, role, gradient-or-None), ...], bbox) in word
    coordinates.  Arrows at the bottom, shadow shapes over them, letters on
    top -- a shadow belongs to the letter whose curl it closes, so it covers
    an arrow but the ink still cuts it.  A "cut" layer is no ink at all: a
    hole punched through everything (write_svg masks with it), so it never
    grows the drawn extent."""
    placed = layout(cfg)
    arrs = build_arrows(cfg, placed, tune)
    shadow_map = tune[2] if tune is not None else {}
    cut_map = tune[3] if tune is not None else {}
    glyphs = range(len(placed)) if glyphs is None else glyphs
    arrows = range(len(arrs)) if arrows is None else arrows
    shadows = sorted(shadow_map) if shadows is None else shadows

    layers, extent = [], Path()
    for i in arrows:
        for p, grad in arrs[i]:
            layers.append((p, "arrow", grad))
            extent.extend(p)
    for i in shadows:
        p, grad = shadow_map[i]
        layers.append((p, "shadow", grad))
        extent.extend(p)
    for i in glyphs:
        layers.append((placed[i], "ink", None))
        extent.extend(placed[i])
    for pid in sorted(cut_map):
        layers.append((cut_map[pid], "cut", None))

    pad = cfg["padding"]
    x0, y0, x1, y1 = extent.bbox()
    return layers, (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _square(box):
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    x0 -= (side - (x1 - x0)) / 2
    y0 -= (side - (y1 - y0)) / 2
    return x0, y0, x0 + side, y0 + side


def write_svg(path, cfg, layers, box, theme, square=False):
    x0, y0, x1, y1 = _square(box) if square else box
    w, h = x1 - x0, y1 - y0
    pal = cfg[theme]

    cuts = [p for p, role, _ in layers if role == "cut"]
    layers = [l for l in layers if l[1] != "cut"]

    # font space is y-up; flip once on the group
    mask = ""
    defs = []
    if cuts:
        # a luminance mask: keep everything, black out the cut shapes --
        # a true hole, so the page shows through like a counter
        holes = "".join(f'<path d="{p.to_svg_d()}" fill="black"/>' for p in cuts)
        defs.append(f'<mask id="cuts" maskUnits="userSpaceOnUse" '
                    f'x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}">'
                    f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" '
                    f'height="{h:.2f}" fill="white"/>{holes}</mask>')
        mask = ' mask="url(#cuts)"'
    # the mask rides an inner group INSIDE the flip: mask content resolves in
    # the referencing element's user space, and only in there is that space
    # word coordinates
    body = [f'<g transform="translate({-x0:.2f},{y1:.2f}) scale(1,-1)">'
            f'<g{mask}>' if mask else
            f'<g transform="translate({-x0:.2f},{y1:.2f}) scale(1,-1)">']
    for n, (p, role, grad) in enumerate(layers):
        fill = pal[role]
        if grad is not None:
            # the fade runs from the letters' ink into the shape's own colour
            p1, p2 = grad
            defs.append(
                f'<linearGradient id="fade{n}" gradientUnits="userSpaceOnUse" '
                f'x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}">'
                f'<stop offset="0" stop-color="{pal["ink"]}"/>'
                f'<stop offset="1" stop-color="{fill}"/>'
                f'</linearGradient>')
            fill = f"url(#fade{n})"
        body.append(f'<path d="{p.to_svg_d()}" fill="{fill}" fill-rule="nonzero"/>')
    body.append("</g></g>" if mask else "</g>")

    bg = f'<rect width="{w:.2f}" height="{h:.2f}" fill="{pal["bg"]}"/>' if pal["bg"] else ""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}">'
        f"<title>cottax</title>"
        + (f"<defs>{''.join(defs)}</defs>" if defs else "")
        + bg + "".join(body) + "</svg>"
    )
    FsPath(path).write_text(svg)
    return path


def write_png(path, svg_path, px):
    """Rasterise the written SVG with inkscape -- exact, fades and all."""
    import shutil as _shutil
    import subprocess

    inkscape = _shutil.which("inkscape")
    if inkscape:
        # absolute paths: a snap-confined inkscape resolves relative ones
        # against $HOME, not the working directory
        path, svg_path = FsPath(path).absolute(), FsPath(svg_path).absolute()
        r = subprocess.run(
            [inkscape, str(svg_path), f"--export-filename={path}",
             f"--export-width={px}"],
            capture_output=True, text=True)
        # a snap-confined inkscape exits 0 even when it cannot see the file
        # (anything outside $HOME), so the artefact is the only honest check
        if r.returncode != 0 or not FsPath(path).exists():
            raise RuntimeError(
                f"inkscape wrote no {path}\n{r.stderr}\n"
                f"(a snap inkscape can only reach files under your home)")
        return path
    raise RuntimeError("no inkscape on PATH -- PNG export needs it now that "
                       "fills can fade (matplotlib cannot fill a gradient)")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--outdir", default=str(FsPath(__file__).parent))
    ap.add_argument("--tuning", action="store_true",
                    help="write the Inkscape tuning SVG (refuses to overwrite)")
    args = ap.parse_args()
    out = FsPath(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    tuning = out / cfg["tuning_svg"]

    if args.tuning:
        if tuning.exists():
            raise SystemExit(f"{tuning} exists -- delete it first if you really "
                             f"want to restart from the parametric arrows")
        print(write_tuning_svg(tuning, cfg))
        return

    drawn = None
    if tuning.exists():
        drawn = read_arrows(tuning)
        ribbons, heads, shadows, cuts = drawn
        print(f"from {tuning}: ribbons {sorted(ribbons)}, heads {sorted(heads)}, "
              f"shadows {sorted(shadows)}, cuts {sorted(cuts)}")

    full, box = build(cfg, drawn)
    icon, ibox = build(cfg, drawn, glyphs=cfg["icon_glyphs"],
                       arrows=cfg["icon_arrows"], shadows=cfg["icon_shadows"])

    written = []
    for theme, suffix in (("light", ""), ("dark", "-dark")):
        logo = write_svg(out / f"cottax-logo{suffix}.svg", cfg, full, box, theme)
        icon_svg = write_svg(out / f"cottax-icon{suffix}.svg", cfg, icon, ibox,
                             theme, square=True)
        written += [
            logo,
            write_png(out / f"cottax-logo{suffix}.png", logo, cfg["png_px"]),
            icon_svg,
            write_png(out / f"cottax-icon{suffix}.png", icon_svg,
                      cfg["icon_png_px"]),
        ]
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
