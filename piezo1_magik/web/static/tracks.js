// Track linking and gap filling on the demo detections: candidate graph -> oracle links (perfect classifier on the
// graph, standing in for MAGIK) -> greedy_linking -> gap filling, scored against the ground truth.
import { api, app, esc, list, route } from './core.js';
import { colour, fmt, getDemo } from './data.js';
import { lengthCategories } from './track_engine.js';

const K = 4;
const mean = v => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN);

function field(id, label, value, step, hint = '') {
  return `<label for="${id}">${label}</label><input id="${id}" name="${id}" type="number" min="${step === 1 ? 1 : 0.1}" step="${step}" value="${value}">${hint}`;
}

route('tracks', async () => {
  const [d, info] = await Promise.all([getDemo(), api('/info')]);
  const [T, H, W] = d.shape, nm = d.pixel_size_nm, p0 = info.track_defaults;
  const st = { view: 'filled', res: null };
  app().innerHTML = `
  <h1>Track linking and gap filling</h1>
  <p class="lead">Stage 2 around the network: detections become graph nodes, candidate edges join detections up to
    <code>max_temporal_gap</code> frames and <code>max_spatial_distance</code> px apart, MAGIK scores every edge, greedy linking keeps the
    best edges one-to-one, and gap filling joins fragments across longer gaps by linear extrapolation.</p>
  <p class="demo-label">Synthetic demo, not a measurement: the ${d.tracks.track_id.length} ground-truth detections of the demo movie stand in for DECODE's output.
    MAGIK is not run here: every candidate edge joining consecutive detections of the same true track gets probability 1, all others 0,
    i.e. a perfect classifier limited to the candidate graph, the best MAGIK could do on it. Everything after that is the repo's own linking,
    gap filling and statistics code.</p>
  <div class="side"><form class="params" id="tf">
    <fieldset><legend>Candidate graph</legend>
      ${field('max_temporal_gap', 'max_temporal_gap (frames)', p0.max_temporal_gap, 1)}
      ${field('max_spatial_distance', 'max_spatial_distance (px)', p0.max_spatial_distance, 0.5, '<p class="hint" id="h1"></p>')}</fieldset>
    <fieldset><legend>Gap filling</legend>
      ${field('max_gap', 'max_gap (frames)', p0.max_gap, 1)}
      ${field('max_distance', 'max_distance from the extrapolated position (px)', p0.max_distance, 0.5, '<p class="hint" id="h2"></p>')}</fieldset>
    <p class="hint">Defaults: graph from configs/magik_training.yaml, gap filling from 08_fill_track_gaps.py.</p>
    <button type="submit">Run</button><p id="tmsg" class="small"></p></form>
  <div>
    <div class="seg" role="group" aria-label="colour detections by">${list([['gt', 'ground truth'], ['linked', 'after linking'], ['filled', 'after gap filling']],
      ([k, l]) => `<button type="button" data-v="${k}" class="${k === st.view ? 'on' : ''}">${l}</button>`)}</div>
    <div class="imgwrap" style="margin-top:8px"><canvas id="tv" width="${W * K}" height="${H * K}" aria-label="detections joined into tracks over the PIEZO1 maximum projection"></canvas></div>
    <p class="small mute">PIEZO1 maximum projection over ${T} frames, ${W} × ${H} px (${(W * nm / 1000).toFixed(2)} µm square). Lines join consecutive detections of one track; grey dots are detections left unlinked.</p>
  </div></div>
  <div id="tsum"></div>`;
  const cv = document.getElementById('tv'), off = document.createElement('canvas');
  off.width = W; off.height = H;
  const img = off.getContext('2d').createImageData(W, H);
  for (let i = 0; i < H * W; i++) {
    let m = 0;
    for (let t = 0; t < T; t++) m = Math.max(m, d.piezo1[t * H * W + i]);
    img.data[4 * i] = m * 0.55; img.data[4 * i + 1] = m * 0.2; img.data[4 * i + 2] = m * 0.3; img.data[4 * i + 3] = 255;
  }
  off.getContext('2d').putImageData(img, 0, 0);

  function draw() {
    const g = cv.getContext('2d'), r = st.res, n = r.nodes;
    g.imageSmoothingEnabled = false; g.drawImage(off, 0, 0, W * K, H * K);
    const label = i => (st.view === 'gt' ? n.gt[i] : r[st.view].assignment[String(i)]);
    const groups = new Map();
    n.frame.forEach((_, i) => { const t = label(i); if (t === undefined) return; if (!groups.has(t)) groups.set(t, []); groups.get(t).push(i); });
    const P = i => [(n.x[i] + 0.5) * K, (n.y[i] + 0.5) * K];
    for (const [t, idx] of groups) {
      idx.sort((a, b) => n.frame[a] - n.frame[b]);
      g.strokeStyle = colour(t); g.lineWidth = 1.6; g.beginPath();
      idx.forEach((i, k) => { const [x, y] = P(i); if (k) g.lineTo(x, y); else g.moveTo(x, y); });
      g.stroke();
      g.fillStyle = colour(t); for (const i of idx) { const [x, y] = P(i); g.fillRect(x - 1.5, y - 1.5, 3, 3); }
    }
    g.fillStyle = '#8a8fa3';
    n.frame.forEach((_, i) => { if (label(i) === undefined) { const [x, y] = P(i); g.fillRect(x - 1.5, y - 1.5, 3, 3); } });
  }
  function summary() {
    const r = st.res, gtLen = {};
    r.nodes.gt.forEach(t => { gtLen[t] = (gtLen[t] || 0) + 1; });
    const gtSpans = Object.keys(gtLen).map(t => { const f = r.nodes.frame.filter((_, i) => String(r.nodes.gt[i]) === t); return Math.max(...f) - Math.min(...f) + 1; });
    const col = s => [s.num_tracks, fmt(mean(s.lengths), 1), fmt(mean(s.temporal_spans), 1), fmt(mean(s.gaps), 2), fmt(mean(s.fragmentation), 2),
      `${s.categories.short} / ${s.categories.medium} / ${s.categories.long}`, s.mixed, r.n_nodes - Object.keys(s.assignment).length];
    const gl = Object.values(gtLen), gc = lengthCategories(gl);
    const L = col(r.linked), F = col(r.filled), G = [r.n_gt_tracks, fmt(mean(gl), 1), fmt(mean(gtSpans), 1),
      fmt(mean(gtSpans.map((s, k) => s - gl[k])), 2), '1.00', `${gc.short} / ${gc.medium} / ${gc.long}`, 0, 0];
    const names = ['tracks', 'mean length (detections)', 'mean temporal span (frames)', 'mean gaps inside a track (frames)',
      'mean fragmentation (result tracks per true track)', 'short ≤3 / medium 4-10 / long >10 (detections)', 'tracks mixing true tracks', 'detections left unlinked'];
    document.getElementById('tsum').innerHTML = `
      <h2>Result <span class="badge ${r.computed_by === 'python' ? 'py' : 'js'}">computed by ${esc(r.computed_by)}</span></h2>
      <p>${r.n_nodes} detections, ${r.n_edges.toLocaleString()} candidate edges, of which ${r.n_true_edges} join consecutive detections of a true track
        (${r.n_nodes - r.n_gt_tracks} such pairs exist; the rest fall outside the graph limits). Gap filling ran ${r.filled.iterations} passes.</p>
      <div class="scroll"><table class="tbl compare"><thead><tr><th></th><th>ground truth</th><th>after linking</th><th>after gap filling</th></tr></thead>
      <tbody>${list(names, (nm2, k) => `<tr><td class="wrap">${nm2}</td><td class="n">${G[k]}</td><td class="n">${L[k]}</td><td class="n">${F[k]}</td></tr>`)}</tbody></table></div>
      <p class="small mute">Lengths, spans, gaps, fragmentation and categories: <code>piezo1_magik/tracking/track_stats.py</code> (as in 07_analyze_tracks.py).
        "Tracks mixing true tracks" and "unlinked" are demo-page counts. Try a small max_spatial_distance (e.g. 0.3 px) with max_gap 10 to see gap filling join the wrong fragments.</p>`;
  }
  function hints() {
    for (const [id, h] of [['max_spatial_distance', 'h1'], ['max_distance', 'h2']]) {
      document.getElementById(h).textContent = `= ${(+document.getElementById(id).value * nm / 1000).toFixed(2)} µm at ${nm} nm/px`;
    }
  }
  async function run() {
    const f = new FormData(document.getElementById('tf')), p = {}, msg = document.getElementById('tmsg');
    for (const k of ['max_temporal_gap', 'max_spatial_distance', 'max_gap', 'max_distance']) p[k] = Number(f.get(k));
    if (!Number.isInteger(p.max_temporal_gap) || p.max_temporal_gap < 1 || !Number.isInteger(p.max_gap) || p.max_gap < 1
      || !(p.max_spatial_distance > 0) || !(p.max_distance > 0)) {
      msg.textContent = 'Gaps must be whole numbers of frames (at least 1) and distances positive.'; msg.className = 'small err'; return;
    }
    msg.textContent = 'Running...'; msg.className = 'small mute';
    try { st.res = await api('/tracks', { body: p }); msg.textContent = ''; draw(); summary(); } catch (e) { msg.textContent = e.message; msg.className = 'small err'; }
  }
  document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
    st.view = b.dataset.v; document.querySelectorAll('.seg button').forEach(x => x.classList.toggle('on', x === b)); if (st.res) draw();
  }));
  document.getElementById('tf').addEventListener('input', hints);
  document.getElementById('tf').addEventListener('submit', e => { e.preventDefault(); run(); });
  hints();
  await run();
});
