"""The ML-5 review page's two pages (coffeecv/seg_review.py, ticket ML-5 P6): HTML, CSS and the key and click
handlers, kept apart from the session logic that serves them. Each page takes two substitutions, `__INFO__` (the
photo's state as JSON) and `__RULE__` (the acceptance rule, HTML-escaped).

    LABEL_PAGE    labelling: A/D/1-5 verdicts, hold g/r + click for include/exclude points, U undo, Enter redraw
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
<div id="left"><div class="wrap" id="wrap"><img id="photo" style="max-width:calc(100vw - 600px)"></div></div>
<div id="right">
 <div id="head"></div>
 <div class="state" id="state"></div>
 <button class="acc" onclick="decide('accept')">Accept (A)</button>
 <button class="dec" onclick="decide('decline')">Decline (D)</button>
 <button onclick="redraw()">Redraw (Enter)</button> <button onclick="undo()">Undo point (U)</button><br>
 <div id="reasons"></div>
 <div class="msg" id="msg"></div>
 <button onclick="go(info.i-1)">&larr; prev</button><button onclick="go(info.i+1)">next &rarr;</button>
 <button onclick="location='/'">next to do (N)</button>
 <div class="keys">hold g + click: include point (green) &middot; hold r + click: exclude point (red) &middot;
   U / Backspace: undo the last point &middot; Enter: redraw &middot; Space: toggle outline &middot;
   plain click: full resolution</div>
 <img id="zoom" class="zoom">
 <div class="rule"><b>Accept rule (D13)</b><br>__RULE__</div>
</div>
<script>
const info = __INFO__;
let outline = 1, last = null, held = null;
// Every point so far, in the order placed (those that drew the shown mask first: includes, then excludes).
let points = [...info.points.include.map(p => ({x: p[0], y: p[1], kind: 'include'})),
              ...info.points.exclude.map(p => ({x: p[0], y: p[1], kind: 'exclude'}))];
const photo = document.getElementById('photo'), zoom = document.getElementById('zoom');
const wrap = document.getElementById('wrap'), msgBox = document.getElementById('msg');
function msg(t){ msgBox.textContent = t; }
function canPoint(){ return info.status === 'points' || (info.status === 'judge' && info.round > 1); }
function src(){ const v = `outline=${outline}&m=${info.mask_sha256.slice(0, 12)}`;
  photo.src = `/img/${info.i}/view.jpg?${v}`;
  if(last) zoom.src = `/img/${info.i}/zoom.jpg?x=${last[0]}&y=${last[1]}&${v}`; }
function drawPoints(){
  wrap.querySelectorAll('.pt').forEach(d => d.remove());
  for(const p of points){ const d = document.createElement('div'); d.className = 'pt ' + p.kind;
    d.style.left = (p.x * 100) + '%'; d.style.top = (p.y * 100) + '%'; wrap.appendChild(d); }
}
function render(){
  document.getElementById('head').textContent =
    `${info.i+1} / ${info.n}  ${info.item}  (${info.source}, ${info.split})`;
  const what = {judge: 'judge this mask', points: `declined${info.reason ? ' ('+info.reason+')' : ''}: place points, then Enter`,
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
  const r = await post(`/redraw/${info.i}`, {include: pick('include'), exclude: pick('exclude')});
  if(r) location.reload();
}
function undo(){ if(points.length){ points.pop(); drawPoints(); msg(`${points.length} points; Enter redraws`); } }
const r4 = v => Math.round(Math.min(Math.max(v, 0), 1) * 1e4) / 1e4;
photo.onclick = e => {
  const b = photo.getBoundingClientRect(), x = (e.clientX-b.left)/b.width, y = (e.clientY-b.top)/b.height;
  if(held){
    if(!canPoint()){ msg('decline the mask first (D or 1-5), then place points'); return; }
    points.push({x: r4(x), y: r4(y), kind: held === 'g' ? 'include' : 'exclude'});
    drawPoints(); msg(`${points.length} points; Enter redraws`); return;
  }
  last = [x, y]; src();
};
function holdKey(k){ return k === 'g' || k === 'G' ? 'g' : k === 'r' || k === 'R' ? 'r' : null; }
document.onkeydown = e => {
  if(e.ctrlKey || e.metaKey || e.altKey) return;
  const h = holdKey(e.key);
  if(h){ held = h; photo.style.cursor = 'cell'; e.preventDefault(); return; }
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
document.onkeyup = e => { if(holdKey(e.key) === held){ held = null; photo.style.cursor = ''; } };
window.onblur = () => { held = null; photo.style.cursor = ''; };
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
