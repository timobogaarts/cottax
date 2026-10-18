'''
The interactive page around pictures rendered in Python, one view per picture: pan, Ctrl+scroll
zoom and **Fit**; hover (a tooltip and the row and column lit, read off a picture's `data-i`,
`data-r`, `data-c`, `data-v`); CSS toggles; frames under a slider, the steps a frame changed
haloed, a picture two frames share shipped once; the key a strip outside what pans.
'''

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Sequence

from .style import Style
from .svg import attr

TIP_VARS = 12
'''How many names one hover lists before it says how many are left.'''


class HtmlDoc(str):
    '''
    HTML as a `str`, which a notebook also renders inline (`_repr_html_`). `path` is the
    `.html` file it was written to, or `None`. The string is the inline widget; the file is
    the same widget in a document of its own, filling the window.
    '''

    path : str | None = None

    def _repr_html_(self) -> str:
        return str(self)


_DOCUMENT = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
html,body{margin:0;padding:0;height:100%;background:__SURFACE__}
body{overflow:hidden}
.cx-wrap{height:100%;height:100dvh}
</style>
</head>
<body>
__BODY__
</body>
</html>
'''

_CSS = '''
:host{display:block;color-scheme:__SCHEME__;background:__SURFACE__;color:__INK__;
  border:1px solid __RULE__;border-radius:8px;overflow:hidden;
  font:13px ui-sans-serif,system-ui,sans-serif;user-select:none;-webkit-user-select:none}
:host([data-fill]){display:flex;flex-direction:column;height:100%;border:0;border-radius:0}
.bar{display:flex;gap:6px;align-items:center;padding:4px 8px;border-bottom:1px solid __RULE__;
  flex:0 0 auto}
.bar button{font:inherit;padding:2px 8px;border:1px solid __RULE__;border-radius:5px;
  background:__SURFACE__;color:inherit;cursor:pointer}
.bar button[aria-pressed="false"]{opacity:.55}
.bar button:disabled{opacity:.3;cursor:default}
.bar .note{margin-left:auto;opacity:.65;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar input[type=range]{flex:0 0 240px;width:240px}
.bar .op{flex:1 1 0;min-width:0;font-family:ui-monospace,monospace;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.key{display:flex;flex-wrap:wrap;gap:4px 14px;padding:3px 8px;border-bottom:1px solid __RULE__;
  flex:0 0 auto;font-size:12px;color:__MUTED__}
.key:empty{display:none}
.key span{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.stage{position:relative;overflow:hidden;height:min(72vh,680px);cursor:grab;background:__SURFACE__}
:host([data-fill]) .stage{flex:1 1 auto;height:auto;min-height:0}
.stage.drag{cursor:grabbing}
.stage svg{position:absolute;left:0;top:0;transform-origin:0 0}
text{pointer-events:none}
[data-i],[data-v]{cursor:default}
.hl{filter:drop-shadow(0 0 2px __ACCENT__)}
.changed{filter:drop-shadow(0 0 3px __ACCENT__) drop-shadow(0 0 1px __ACCENT__)}
.tip{position:fixed;z-index:2147483647;pointer-events:none;background:__SURFACE__;color:__INK__;
  border:1px solid __RULE__;border-radius:6px;padding:6px 9px;font:13px __FACE__;
  white-space:pre;max-width:60ch;max-height:60vh;overflow:hidden;box-shadow:0 2px 10px #0003;
  display:none}
.tip .h{font-weight:700;text-decoration:underline}
.tip .fb{color:__ACCENT__;font-weight:700}
'''

_JS = r'''
(function(){
const HOST = document.getElementById(__UID__);
if (!HOST) return;
const ROOT = HOST.attachShadow({mode: 'open'});
ROOT.innerHTML = __SHELL__;
const SVGS = __SVGS__, FRAMES = __FRAMES__, VIEWS = __VIEWS__;
const stage = ROOT.querySelector('.stage'), tip = ROOT.querySelector('.tip');
let frame = FRAMES.length - 1, view = 0, tf = {s: 1, x: 0, y: 0}, svg = null;
const current = () => FRAMES[frame].views[view];

const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function apply(){ if (svg) svg.style.transform = 'translate('+tf.x+'px,'+tf.y+'px) scale('+tf.s+')'; }
function fit(){
  if (!svg) return;
  const w = +svg.getAttribute('width'), h = +svg.getAttribute('height');
  const W = stage.clientWidth || w, H = stage.clientHeight || h;
  tf.s = Math.min(W / w, H / h, 3);
  tf.x = (W - w * tf.s) / 2; tf.y = Math.max(0, (H - h * tf.s) / 2); apply();
}
// The key, outside what pans and zooms: a sample of each thing, in its own colours.
function swatch(e){
  const a = 'fill="'+e.fill+'" stroke="'+e.stroke+'" stroke-width="1.4"';
  const s = e.shape === 'hex'  ? '<polygon '+a+' points="3,1 15,1 18,6 15,11 3,11 0,6" transform="translate(0.5,0.5)"/>'
          : e.shape === 'square' ? '<rect '+a+' x="3" y="1" width="11" height="11" rx="2"/>'
          : e.shape === 'open' ? '<rect fill="'+e.fill+'" x="1" y="1" width="17" height="11" rx="2"/>'
              + '<path fill="none" stroke="'+e.stroke+'" stroke-width="1.4" d="M7,1 H1 V12 H7 M12,1 H18 V12 H12"/>'
          : '<rect '+a+' x="1" y="1" width="17" height="11" rx="2"/>';
  const t = e.title ? ' title="' + esc(e.title).replace(/"/g,'&quot;') + '"' : '';
  return '<span'+t+'><svg width="20" height="13" viewBox="0 0 20 13">' + s + '</svg>' + esc(e.text) + '</span>';
}
function show(i){
  frame = Math.max(0, Math.min(FRAMES.length - 1, i));
  const f = FRAMES[frame], v = f.views[view];
  stage.innerHTML = SVGS[v.svg];
  ROOT.querySelector('.key').innerHTML = (v.legend || []).map(swatch).join('');
  svg = stage.querySelector('svg');
  for (const k of (v.changed || []))
    for (const el of ROOT.querySelectorAll('[data-i="'+k+'"]')) el.classList.add('changed');
  const slider = ROOT.querySelector('[data-act="frame"]');
  if (slider){
    slider.value = frame;
    const op = ROOT.querySelector('.op');
    op.textContent = frame + '/' + (FRAMES.length - 1) + '  ' + f.label;
    op.title = f.tip;
    ROOT.querySelector('[data-act="prev"]').disabled = frame === 0;
    ROOT.querySelector('[data-act="next"]').disabled = frame === FRAMES.length - 1;
  }
  apply();
}

// ---- hover: what the picture says an element is
function list(head, vs, n){
  if (!n) return '';
  // a name arrives as HTML, set by the drawing's formatter; nothing else is markup
  return '<div class="h">' + head + '</div>' + vs.map(v => '  ' + v).join('\n')
    + (n > vs.length ? '\n  … ' + (n - vs.length) + ' more' : '') + '\n';
}
function place(e){
  tip.style.display = 'block';
  tip.style.left = Math.max(4, Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 4)) + 'px';
  tip.style.top = Math.max(4, Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 4)) + 'px';
}
function light(r, c){
  for (const el of ROOT.querySelectorAll('.hl')) el.classList.remove('hl');
  const on = sel => { for (const el of ROOT.querySelectorAll(sel)) el.classList.add('hl'); };
  for (const k of [r, c]) if (k !== undefined && k >= 0){
    on('[data-i="'+k+'"]'); on('[data-r="'+k+'"]'); on('[data-c="'+k+'"]');
  }
}
stage.addEventListener('mousemove', e => {
  const t = e.target.closest ? e.target.closest('[data-i],[data-v]') : null;
  const rows = current().rows;
  if (t && t.dataset.i !== undefined){
    const r = rows[+t.dataset.i];
    tip.innerHTML = r.name + (r.group ? '\n' + r.group : '')
      + (r.kind ? '\n' + esc(r.kind) + (r.driven ? ', driven' : ', no driver') : '')
      + '\n' + list('reads', r.reads, r.nr) + list('writes', r.writes, r.nw);
    light(+t.dataset.i); place(e);
  } else if (t && t.dataset.v !== undefined){
    const r = +t.dataset.r, c = +t.dataset.c, vs = t.dataset.v.split('\n');
    const from = r >= 0 ? rows[r].name : 'outside', to = c >= 0 ? rows[c].name : 'outside';
    tip.innerHTML = (r >= 0 && c >= 0 && r > c ? '<span class="fb">feedback</span>\n' : '')
      + from + ' → ' + to + '\n'
      + list(vs.length + (vs.length === 1 ? ' variable' : ' variables'), vs.slice(0, __TIPN__), vs.length);
    light(r, c); place(e);
  } else { tip.style.display = 'none'; light(); }
});
stage.addEventListener('mouseleave', () => { tip.style.display = 'none'; light(); });

// ---- pan and zoom
// Ctrl (or Cmd) + scroll zooms, and a trackpad pinch arrives as one; a plain scroll is
// left to the page, or a notebook could not be scrolled past a diagram.
stage.addEventListener('wheel', e => {
  if (!(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  const b = stage.getBoundingClientRect(), px = e.clientX - b.left, py = e.clientY - b.top;
  const s = Math.min(Math.max(tf.s * Math.exp(-e.deltaY * 0.0015), 0.02), 40);
  tf.x = px - (px - tf.x) * s / tf.s; tf.y = py - (py - tf.y) * s / tf.s; tf.s = s; apply();
}, {passive: false});
let drag = null;
stage.addEventListener('mousedown', e => { drag = {x: e.clientX - tf.x, y: e.clientY - tf.y}; stage.classList.add('drag'); });
addEventListener('mouseup', () => { drag = null; stage.classList.remove('drag'); });
addEventListener('mousemove', e => { if (drag){ tf.x = e.clientX - drag.x; tf.y = e.clientY - drag.y; apply(); } });

// ---- the bar
ROOT.querySelector('[data-act="fit"]').addEventListener('click', fit);
// the other picture of the same frame: the zoom is kept, and Fit is a click away
const other = ROOT.querySelector('[data-act="view"]');
if (other) other.addEventListener('click', () => {
  view = (view + 1) % VIEWS.length;
  other.textContent = VIEWS[(view + 1) % VIEWS.length];
  show(frame); fit();
});
for (const b of ROOT.querySelectorAll('[data-toggle]'))
  b.addEventListener('click', () => {
    const off = HOST.toggleAttribute('data-no' + b.dataset.toggle);
    b.setAttribute('aria-pressed', String(!off));
  });
if (FRAMES.length > 1){
  ROOT.querySelector('[data-act="frame"]').addEventListener('input', e => show(+e.target.value));
  ROOT.querySelector('[data-act="prev"]').addEventListener('click', () => show(frame - 1));
  ROOT.querySelector('[data-act="next"]').addEventListener('click', () => show(frame + 1));
}
show(frame); fit();
if (typeof ResizeObserver !== 'undefined') new ResizeObserver(fit).observe(stage);
})();
'''


FRAME_LABEL_CHARS = 80
'''How much of an op's `repr` a frame's label shows; the whole of it is the tooltip.'''


def frame_label(op) -> tuple[str, str]:
    '''An op as a frame names it: its own `repr`, shortened, and the whole for a tooltip.'''
    full = repr(op)
    return (full if len(full) <= FRAME_LABEL_CHARS else full[:FRAME_LABEL_CHARS - 1] + '…'), full


def prefixes(graph) -> list[tuple[str, str, object]]:
    '''`(label, tip, graph)` per frame: one for a graph, one per prefix of a `Plan` -- its
    base, then each op applied, labelled by that op.'''
    from ...core import Plan
    if not isinstance(graph, Plan):
        return [('', '', graph)]
    plans, at = [], graph
    while at is not None:
        plans.append(at)
        at = at.trunk
    plans.reverse()
    return [(('base', f'{len(p.graph.nodes)} nodes') if not p.ops else frame_label(p.ops[-1]))
            + (p.graph,) for p in plans]


def changed(before, after) -> set:
    '''The nodes `after` defines that `before` did not define so: what an op changed.'''
    if before is None:
        return set()
    return {n for n in after.nodes if n not in before or before[n] != after[n]}


class Pictures:
    '''The pictures a page ships, each once however many frames show it.'''

    def __init__(self) -> None:
        self.svgs: list[str] = []
        self._index: dict[str, int] = {}

    def add(self, svg: str) -> int:
        if svg not in self._index:
            self._index[svg] = len(self.svgs)
            self.svgs.append(svg)
        return self._index[svg]


def _js(value) -> str:
    '''JSON a `<script>` can hold: `</script>` in a name cannot end it early.'''
    return json.dumps(value).replace('<', '\\u003c')


def page(svgs: Sequence[str], frames: Sequence[dict], *, style: Style,
         views: Sequence[str] = ('',), toggles: Sequence[tuple[str, str, bool, str]] = (),
         note: str = '', write: bool = False, file_name: str = 'page',
         outdir: str = '.') -> HtmlDoc:
    '''
    The page. `svgs` are the pictures, `frames` what the slider steps through, each
    `{'label', 'tip', 'views'}` -- one view per name in `views` (the XDSM and the N2 of
    one graph, say, swapped by a button), each `{'svg': index, 'rows', 'changed',
    'legend'}`: `legend` the key, drawn in a strip the pictures pan and zoom under, each
    entry `{'shape': 'hex' | 'band' | 'open' | 'square', 'fill', 'stroke', 'text'}` and
    an optional `title`, what hovering it says; `rows`
    a list per step of `{'name', 'group', 'kind', 'driven', 'reads', 'nr', 'writes',
    'nw'}` in that picture's order, `changed` the step indices to halo. `toggles` are
    `(key, label, on, css)`: a button flipping `data-no<key>` on the host, and the rules
    drawn while it is off (written against `:host([data-no<key>])`). `note` is a line at
    the end of the bar.
    '''
    uid = 'cx-' + uuid.uuid4().hex[:8]
    css = (_CSS.replace('__SCHEME__', style.ground.scheme)
           .replace('__SURFACE__', style.surface).replace('__INK__', style.ink)
           .replace('__RULE__', style.boundary[1]).replace('__ACCENT__', '#d9480f')
           .replace('__MUTED__', style.muted)
           .replace('__FACE__', style.font or 'ui-sans-serif,system-ui,sans-serif'))
    css += ''.join(rules for *_, rules in toggles)
    buttons = ''.join(f'<button type="button" data-toggle="{key}" aria-pressed="{str(on).lower()}">'
                      f'{label}</button>' for key, label, on, _ in toggles)
    if len(views) > 1:                          # the button names the picture it goes to
        buttons = f'<button type="button" data-act="view">{attr(views[1])}</button>' + buttons
    n = len(frames)
    slider = ''
    if n > 1:
        last = frames[-1]
        slider = (f'<div class="bar">'
                  # the two arrows side by side and the slider a fixed width, so nothing
                  # moves under the pointer while the label changes length
                  f'<button type="button" data-act="prev">&#9664;</button>'
                  f'<button type="button" data-act="next" disabled>&#9654;</button>'
                  f'<input type="range" data-act="frame" min="0" max="{n - 1}" value="{n - 1}" step="1">'
                  f'<span class="op" title="{attr(last["tip"])}">{n - 1}/{n - 1}  '
                  f'{attr(last["label"])}</span></div>')
    shell = (f'<style>{css}</style>'
             f'<div class="bar"><button type="button" data-act="fit">Fit</button>{buttons}'
             f'<span class="note">{attr(note)}</span></div>{slider}'
             f'<div class="key"></div><div class="stage"></div><div class="tip"></div>')
    script = _JS
    for token, value in (('__UID__', _js(uid)), ('__SHELL__', _js(shell)),
                         ('__SVGS__', _js(list(svgs))), ('__FRAMES__', _js(list(frames))),
                         ('__VIEWS__', _js(list(views))),
                         ('__TIPN__', str(TIP_VARS))):
        script = script.replace(token, value)
    off = ''.join(f' data-no{key}' for key, _, on, _ in toggles if not on)

    def fragment(fill: bool) -> str:
        return (f'<div id="{uid}" class="cx-wrap"{off}{" data-fill" if fill else ""}></div>'
                f'<script>{script}</script>')

    doc = HtmlDoc(fragment(False))
    if write:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, file_name + '.html')
        with open(path, 'w') as fh:
            fh.write(_DOCUMENT.replace('__TITLE__', attr(file_name))
                     .replace('__SURFACE__', style.surface).replace('__BODY__', fragment(True)))
        doc.path = path
    return doc
