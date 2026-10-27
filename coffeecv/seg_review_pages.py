"""The ML-5 review page's two pages (coffeecv/seg_review.py, ticket ML-5 P6): HTML, CSS and the key and click
handlers, kept apart from the session logic that serves them. Each page takes two substitutions, `__INFO__` (the
photo's state as JSON) and `__RULE__` (the acceptance rule, HTML-escaped).

    LABEL_PAGE    labelling: A/D/1-5 verdicts, hold g/r + click for include/exclude points, hold b + drag for a
                  fence (ticket ML-9 P1), hold e/w + drag to erase/fill with a round brush (ticket ML-11), U undo,
                  Enter redraw
    PAIRED_PAGE   the blind paired mode (D22): A/D the left mask, J/L the right one
"""

STYLE = """<style>
body{font-family:sans-serif;margin:0;background:#222;color:#eee;display:flex;height:100vh}
#left{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;gap:8px}
.wrap{position:relative;display:inline-block;line-height:0}
.wrap img{max-height:100vh;cursor:crosshair}
.pt{position:absolute;width:12px;height:12px;border-radius:50%;border:2px solid #fff;transform:translate(-50%,-50%);
    pointer-events:none}
.pt.include{background:#1db31d}.pt.exclude{background:#e01b1b}
.fence{position:absolute;border:2px dashed #ffd400;box-sizing:border-box;pointer-events:none}
#paint{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;opacity:.45}
.brush{position:absolute;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;display:none;
    box-sizing:border-box}
.brush.erase{border:1px solid #ff8080;background:rgba(255,112,112,.15)}
.brush.fill{border:1px solid #80ff80;background:rgba(112,255,112,.15)}
.radius{margin:6px 3px}.radius input{width:4em;font-size:15px}
#right{width:560px;padding:12px;overflow:auto;background:#2b2b2b}
.zoom{width:512px;height:512px;background:#111;display:block;margin:8px 0}
button{font-size:15px;margin:3px;padding:6px 10px}
.acc{background:#2e7d32;color:#fff}.dec{background:#c62828;color:#fff}
.rule{font-size:13px;line-height:1.4;background:#333;padding:8px;border-radius:4px}
.state{font-size:18px;margin:8px 0}.msg{background:#444;padding:6px;margin-top:6px;min-height:1em}
.keys{font-size:12px;color:#aaa}
</style>"""

LABEL_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ML-5 labels</title>""" + STYLE + """
</head><body>
<div id="left"><div class="wrap" id="wrap"><img id="photo" style="max-width:calc(100vw - 600px)"><canvas id="paint">
</canvas><div class="brush" id="brush"></div></div></div>
<div id="right">
 <div id="head"></div>
 <div class="state" id="state"></div>
 <button class="acc" onclick="decide('accept')">Accept (A)</button>
 <button class="dec" onclick="decide('decline')">Decline (D)</button>
 <button onclick="redraw()">Redraw (Enter)</button> <button onclick="undo()">Undo (U)</button><br>
 <div id="reasons"></div>
 <div class="radius">brush radius <input id="radius" type="number" min="1" max="100" step="1"> screen pixels
   (1&ndash;100)</div>
 <div class="msg" id="msg"></div>
 <button onclick="go(info.i-1)">&larr; prev</button><button onclick="go(info.i+1)">next &rarr;</button>
 <button onclick="location='/'">next to do (N)</button>
 <div class="keys">hold g + click: include point (green) &middot; hold r + click: exclude point (red) &middot;
   hold b + drag: fence (yellow), the redrawn mask is cut to it &middot; hold e + drag: erase (light red), hold
   w + drag: fill (light green), a brush of the radius above, painted after the fence &middot; U / Backspace: undo
   the last point, fence or stroke &middot; Enter: redraw &middot; Space: toggle outline &middot; plain click: full
   resolution</div>
 <img id="zoom" class="zoom">
 <div class="rule"><b>Accept rule (D13)</b><br>__RULE__</div>
</div>
<script>
const info = __INFO__;
let outline = 1, last = null, held = null;
// Every point so far, in the order placed (those that drew the shown mask first: includes, then excludes).
let points = [...info.points.include.map(p => ({x: p[0], y: p[1], kind: 'include'})),
              ...info.points.exclude.map(p => ({x: p[0], y: p[1], kind: 'exclude'}))];
// The fence that cut the shown mask, if any.
let fence = info.fence, drag = null;
// Every brush stroke so far (ticket ML-11): r in photo pixels, the path in fractions of the photo (D1). Each shows
// as a tint (D7), those that drew the shown mask too (D10), so the owner checks the mask against them; those drawn
// here are `here`.
let strokes = info.strokes.map(s => ({...s, here: false})), paint = null, mouse = null;
// The undo order: points, fences and strokes as placed (loaded ones first).
const history = [...points.map(() => ({kind: 'point'})), ...(fence ? [{kind: 'fence', prev: null}] : []),
                 ...strokes.map(() => ({kind: 'stroke'}))];
const photo = document.getElementById('photo'), zoom = document.getElementById('zoom');
const wrap = document.getElementById('wrap'), msgBox = document.getElementById('msg');
const canvas = document.getElementById('paint'), brush = document.getElementById('brush');
const radiusBox = document.getElementById('radius'), RADIUS_KEY = 'seg_review.brush_radius';
const BRUSH = {e: 'erase', w: 'fill'}, TINT = {erase: '#ff7070', fill: '#70ff70'};
// The radius in screen pixels, 1-100; it starts at 5 and the browser keeps the last one set (D8).
const clampRadius = v => Number.isFinite(v) ? Math.min(100, Math.max(1, Math.round(v))) : 5;
let radius = clampRadius(parseFloat(localStorage.getItem(RADIUS_KEY)));
function msg(t){ msgBox.textContent = t; }
function canPoint(){ return info.status === 'points' || (info.status === 'judge' && info.round > 1); }
function src(){ const v = `outline=${outline}&m=${info.mask_sha256.slice(0, 12)}`;
  photo.src = `/img/${info.i}/view.jpg?${v}`;
  if(last) zoom.src = `/img/${info.i}/zoom.jpg?x=${last[0]}&y=${last[1]}&${v}`; }
function drawPoints(){
  wrap.querySelectorAll('.pt, .fence').forEach(d => d.remove());
  for(const p of points){ const d = document.createElement('div'); d.className = 'pt ' + p.kind;
    d.style.left = (p.x * 100) + '%'; d.style.top = (p.y * 100) + '%'; wrap.appendChild(d); }
  const f = drag ? box(drag) : fence;
  if(f){ const d = document.createElement('div'); d.className = 'fence';
    d.style.left = (f[0] * 100) + '%'; d.style.top = (f[1] * 100) + '%';
    d.style.width = ((f[2] - f[0]) * 100) + '%'; d.style.height = ((f[3] - f[1]) * 100) + '%'; wrap.appendChild(d); }
}
function drawStrokes(){
  // Every stroke so far and the one being drawn, opaque on a canvas the CSS makes see-through, so overlaps do not
  // darken and a later stroke covers an earlier one, as on the server.
  const b = photo.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(b.width * dpr); canvas.height = Math.round(b.height * dpr);
  const g = canvas.getContext('2d'), sx = canvas.width, sy = canvas.height;
  g.lineCap = g.lineJoin = 'round';
  for(const s of [...strokes, ...(paint ? [paint] : [])]){
    const r = s.r * sx / info.width;                        // photo pixels to canvas pixels
    g.strokeStyle = g.fillStyle = TINT[s.mode]; g.beginPath();
    if(s.path.length === 1){ g.arc(s.path[0][0] * sx, s.path[0][1] * sy, r, 0, 2 * Math.PI); g.fill(); continue; }
    s.path.forEach(([x, y], k) => k ? g.lineTo(x * sx, y * sy) : g.moveTo(x * sx, y * sy));
    g.lineWidth = 2 * r; g.stroke();
  }
}
function showBrush(){
  // The cursor while e or w is held over the photo: a circle of the brush's size, at least a few pixels wide.
  const b = photo.getBoundingClientRect(), w = wrap.getBoundingClientRect();
  const on = !!BRUSH[held] && !!mouse && mouse.x >= b.left && mouse.x <= b.right && mouse.y >= b.top &&
             mouse.y <= b.bottom;
  brush.style.display = on ? 'block' : 'none';
  if(!on) return;
  const d = Math.max(2 * radius, 4) + 'px';
  brush.className = 'brush ' + BRUSH[held];
  Object.assign(brush.style, {width: d, height: d, left: (mouse.x - w.left) + 'px', top: (mouse.y - w.top) + 'px'});
}
function counts(){
  return `${points.length} points${fence ? ', a fence' : ''}${strokes.length ? `, ${strokes.length} strokes` : ''}`;
}
function told(){ msg(`${counts()}; Enter redraws`); }
function render(){
  document.getElementById('head').textContent =
    `${info.i+1} / ${info.n}  ${info.item}  (${info.source}, ${info.split})`;
  const what = {judge: 'judge this mask', points: `declined${info.reason ? ' ('+info.reason+')' : ''}: place points, a fence or strokes, then Enter`,
                accepted: '<b>accepted</b>', dropped: `<b>dropped</b>: ${info.reason}`, pending: 'not prepared'}[info.status];
  document.getElementById('state').innerHTML =
    `session ${info.session}: accepted ${info.accepted}, dropped ${info.dropped}, to do ${info.todo}<br>` +
    `round ${info.round} / ${info.rounds}, drawn by ${info.model}: ${what}`;
  document.getElementById('reasons').innerHTML = info.reasons.map((r,k)=>
    `<button class="dec" onclick="decide('decline', '${r}')">${k+1}: ${r}</button>`).join('');
  drawPoints();
}
function go(i){ if(i>=0 && i<info.n) location = `/item/${i}`; }
async function post(url, body){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  if(!r.ok){ msg(j.error || ('error ' + r.status)); return null; }
  return j;
}
async function decide(decision, reason=null){
  const r = await post(`/decide/${info.i}`, {decision, reason});
  if(!r) return;
  if((r.status === 'accepted' || r.status === 'dropped') && r.next !== null) go(r.next); else location.reload();
}
async function redraw(){
  if(!canPoint()){ msg('decline the mask first (D or 1-5), then place points'); return; }
  const pick = k => points.filter(p => p.kind === k).map(p => [p.x, p.y]);
  msg('redrawing...');
  const r = await post(`/redraw/${info.i}`, {include: pick('include'), exclude: pick('exclude'), fence,
                                             strokes: strokes.map(s => ({mode: s.mode, r: s.r, path: s.path}))});
  if(r) location.reload();
}
function undo(){
  const h = history.pop();
  if(!h) return;
  if(h.kind === 'fence') fence = h.prev;
  else if(h.kind === 'stroke'){
    // A stroke from an earlier round is painted into the shown mask, which the page cannot unpaint (D9).
    const s = strokes.pop();
    drawStrokes();
    if(!s.here){ msg(`undone; Enter redraws to show it (${counts()})`); return; }
  }
  else points.pop();
  drawPoints(); told();
}
const r4 = v => Math.round(Math.min(Math.max(v, 0), 1) * 1e4) / 1e4;
function at(e){ const b = photo.getBoundingClientRect();
  return [r4((e.clientX-b.left)/b.width), r4((e.clientY-b.top)/b.height)]; }
const box = d => [Math.min(d[0], d[2]), Math.min(d[1], d[3]), Math.max(d[0], d[2]), Math.max(d[1], d[3])];
photo.ondragstart = () => false;
photo.onmousedown = e => {
  if(e.button !== 0 || (held !== 'b' && !BRUSH[held])) return;
  if(!canPoint()){ msg(`decline the mask first (D or 1-5), then ${held === 'b' ? 'draw a fence' : 'paint'}`); return; }
  e.preventDefault();
  if(held === 'b'){ drag = [...at(e), ...at(e)]; return; }
  // The radius goes to the record in photo pixels, at the view's scale now (D1).
  const scale = info.width / photo.getBoundingClientRect().width;
  paint = {mode: BRUSH[held], r: Math.round(radius * scale * 100) / 100, path: [at(e)], last: [e.clientX, e.clientY]};
  drawStrokes();
};
function endPaint(){
  if(!paint) return;
  strokes.push({mode: paint.mode, r: paint.r, path: paint.path, here: true}); history.push({kind: 'stroke'});
  paint = null; drawStrokes(); told();
}
document.onmousemove = e => {
  mouse = {x: e.clientX, y: e.clientY}; showBrush();
  if(drag){ drag = [drag[0], drag[1], ...at(e)]; drawPoints(); }
  // A path point only where the mouse moved a screen pixel or more (ticket R5).
  if(paint && Math.hypot(e.clientX - paint.last[0], e.clientY - paint.last[1]) >= 1){
    paint.path.push(at(e)); paint.last = [e.clientX, e.clientY]; drawStrokes(); }
};
document.onmouseup = e => {
  endPaint();
  if(!drag) return;
  const f = box([drag[0], drag[1], ...at(e)]);
  drag = null;
  if(f[2] - f[0] >= 0.01 && f[3] - f[1] >= 0.01){ history.push({kind: 'fence', prev: fence}); fence = f; }
  drawPoints(); told();
};
photo.onclick = e => {
  const [x, y] = at(e);
  if(held === 'b' || BRUSH[held]) return;
  if(held){
    if(!canPoint()){ msg('decline the mask first (D or 1-5), then place points'); return; }
    points.push({x, y, kind: held === 'g' ? 'include' : 'exclude'}); history.push({kind: 'point'});
    drawPoints(); told(); return;
  }
  last = [x, y]; src();
};
function holdKey(k){ k = k.toLowerCase(); return 'grbew'.includes(k) && k.length === 1 ? k : null; }
function hold(h){ held = h; photo.style.cursor = !h ? '' : BRUSH[h] ? 'none' : 'cell'; showBrush(); }
function release(){ endPaint(); hold(null); }
// The radius field keeps its keys: a digit typed there declines nothing (ticket R2); Enter or Esc leaves it.
radiusBox.value = radius;
radiusBox.oninput = () => { const v = parseFloat(radiusBox.value);
  if(Number.isFinite(v)){ radius = clampRadius(v); localStorage.setItem(RADIUS_KEY, radius); showBrush(); } };
radiusBox.onchange = () => { radiusBox.value = radius; };
radiusBox.onkeydown = e => { e.stopPropagation();
  if(e.key === 'Enter' || e.key === 'Escape'){ radiusBox.value = radius; radiusBox.blur(); e.preventDefault(); } };
radiusBox.onkeyup = e => e.stopPropagation();
document.onkeydown = e => {
  if(e.ctrlKey || e.metaKey || e.altKey || e.target === radiusBox) return;
  const h = holdKey(e.key);
  if(h){ if(h !== held){ endPaint(); hold(h); } e.preventDefault(); return; }
  if(e.key===' '){ outline = 1-outline; src(); e.preventDefault(); }
  else if(e.key==='a'||e.key==='A') decide('accept');
  else if(e.key==='d'||e.key==='D') decide('decline');
  else if(e.key>='1' && e.key<='5') decide('decline', info.reasons[+e.key-1]);
  else if(e.key==='u'||e.key==='U'||e.key==='Backspace'){ undo(); e.preventDefault(); }
  else if(e.key==='Enter'){ redraw(); e.preventDefault(); }
  else if(e.key==='ArrowLeft') go(info.i-1);
  else if(e.key==='ArrowRight') go(info.i+1);
  else if(e.key==='n'||e.key==='N') location='/';
};
document.onkeyup = e => { if(holdKey(e.key) === held) release(); };
window.onblur = release;
photo.onload = drawStrokes;
window.onresize = () => { drawStrokes(); showBrush(); };
render(); src();
</script></body></html>"""


PAIRED_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ML-5 paired review</title>""" + STYLE + """
</head><body>
<div id="left"></div>
<div id="right">
 <div id="head"></div>
 <div class="state" id="state"></div>
 <div id="buttons"></div>
 <div class="msg" id="msg"></div>
 <button onclick="go(info.i-1)">&larr; prev</button><button onclick="go(info.i+1)">next &rarr;</button>
 <button onclick="location='/'">next unjudged (N)</button>
 <div class="keys">A / D: accept / decline the left mask (the only one when both are identical) &middot;
   J / L: accept / decline the right mask &middot; Space: toggle outlines &middot; click a photo: both masks at
   full resolution there</div>
 <div id="zooms"></div>
 <div class="rule"><b>Accept rule (D13)</b><br>__RULE__</div>
</div>
<script>
const info = __INFO__;
let outline = 1, last = null;
const left = document.getElementById('left'), zooms = document.getElementById('zooms');
const label = {left: 'left', right: 'right', both: 'both masks (identical)'};
const width = info.sides.length === 2 ? 'calc((100vw - 620px) / 2)' : 'calc(100vw - 600px)';
left.innerHTML = info.sides.map(s => `<div class="wrap"><img id="img_${s}" style="max-width:${width}"></div>`).join('');
zooms.innerHTML = info.sides.map(s => `<div>${label[s]}</div><img id="zoom_${s}" class="zoom">`).join('');
function src(){ for(const s of info.sides){
  document.getElementById('img_' + s).src = `/img/${info.i}/${s}.jpg?outline=${outline}`;
  if(last) document.getElementById('zoom_' + s).src =
    `/img/${info.i}/${s}/zoom.jpg?x=${last[0]}&y=${last[1]}&outline=${outline}`; } }
function render(){
  document.getElementById('head').textContent = `${info.i+1} / ${info.n}  ${info.item}  (${info.source})`;
  document.getElementById('state').innerHTML = `judged ${info.judged} / ${info.total} masks` +
    info.sides.map(s => `<br>${label[s]}: ` + (info.decisions[s] ? `<b>${info.decisions[s]}</b>` : '<i>unjudged</i>') +
      (info.revealed ? ` (${info.revealed[s].join(' = ')})` : '')).join('') +
    (info.complete ? '<br>session complete: <a href="/reveal" style="color:#9cf">models and verdicts</a>' : '');
  document.getElementById('buttons').innerHTML = info.sides.map(s =>
    `<div>${label[s]}: <button class="acc" onclick="decide('${s}', 'accept')">Accept</button>` +
    `<button class="dec" onclick="decide('${s}', 'decline')">Decline</button>` +
    info.reasons.map(r => `<button class="dec" onclick="decide('${s}', 'decline', '${r}')">${r}</button>`).join('') +
    '</div>').join('');
}
function go(i){ if(i>=0 && i<info.n) location = `/item/${i}`; }
async function decide(side, decision, reason=null){
  const r = await fetch(`/decide/${info.i}`, {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({side, decision, reason})});
  const j = await r.json();
  if(!r.ok){ document.getElementById('msg').textContent = j.error || ('error ' + r.status); return; }
  if(j.photo_done && j.next !== null) go(j.next); else location.reload();
}
const leftSide = () => info.sides.includes('both') ? 'both' : 'left';
left.onclick = e => { if(e.target.tagName !== 'IMG') return; const b = e.target.getBoundingClientRect();
  last = [(e.clientX-b.left)/b.width, (e.clientY-b.top)/b.height]; src(); };
document.onkeydown = e => {
  if(e.ctrlKey || e.metaKey || e.altKey) return;
  if(e.key===' '){ outline = 1-outline; src(); e.preventDefault(); }
  else if(e.key==='a'||e.key==='A') decide(leftSide(), 'accept');
  else if(e.key==='d'||e.key==='D') decide(leftSide(), 'decline');
  else if((e.key==='j'||e.key==='J') && info.sides.includes('right')) decide('right', 'accept');
  else if((e.key==='l'||e.key==='L') && info.sides.includes('right')) decide('right', 'decline');
  else if(e.key==='ArrowLeft') go(info.i-1);
  else if(e.key==='ArrowRight') go(info.i+1);
  else if(e.key==='n'||e.key==='N') location='/';
};
render(); src();
</script></body></html>"""
