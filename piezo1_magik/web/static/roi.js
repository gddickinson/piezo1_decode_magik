// ROI calcium analysis on the synthetic demo: CalciumROIAnalyzer (Python in the local app, roi_engine.js on the
// static site) on every ground-truth track; movie viewer with ROI boxes, trace + dF/F0 charts, events table.
import { api, app, esc, list, route } from './core.js';
import { drawFrame, lineChart } from './charts.js';
import { colour, fmt, getDemo } from './data.js';
import { roundHalfEven } from './roi_engine.js';

const K = 4;  // display pixels per image pixel

function form(p) {
  const odd = Array.from({ length: 8 }, (_, i) => 2 * i + 1);
  return `<form class="params" id="roiform">
    <fieldset><legend>ROI</legend>
      <label for="roi_size">roi_size (px, odd; side of the square)</label>
      <select id="roi_size" name="roi_size">${list(odd, n => `<option value="${n}" ${n === p.roi_size ? 'selected' : ''}>${n} × ${n} px</option>`)}</select>
      <p class="hint" id="roi_um"></p></fieldset>
    <fieldset><legend>Baseline F₀</legend>
      <label for="baseline_method">baseline_method</label>
      <select id="baseline_method" name="baseline_method">${list(['percentile', 'rolling', 'mean'], m => `<option ${m === p.baseline_method ? 'selected' : ''}>${m}</option>`)}</select>
      <label for="baseline_percentile">baseline_percentile (percentile method)</label>
      <input id="baseline_percentile" name="baseline_percentile" type="number" min="0" max="100" step="1" value="${p.baseline_percentile}"></fieldset>
    <fieldset><legend>Events (detect_peaks)</legend>
      <label for="threshold">threshold (× std of ΔF/F₀)</label><input id="threshold" name="threshold" type="number" min="0" step="0.1" value="${p.threshold}">
      <label for="min_duration">min_duration (frames)</label><input id="min_duration" name="min_duration" type="number" min="1" step="1" value="${p.min_duration}">
      <label for="min_distance">min_distance between peaks (frames)</label><input id="min_distance" name="min_distance" type="number" min="1" step="1" value="${p.min_distance}"></fieldset>
    <button type="submit">Run analysis</button></form>`;
}

function readForm() {
  const f = new FormData(document.getElementById('roiform')), p = {};
  for (const k of ['roi_size', 'baseline_percentile', 'threshold', 'min_duration', 'min_distance']) {
    const v = Number(f.get(k));
    if (!Number.isFinite(v)) throw new Error(`${k} must be a number`);
    p[k] = v;
  }
  p.baseline_method = f.get('baseline_method');
  if (!Number.isInteger(p.min_duration) || p.min_duration < 1 || !Number.isInteger(p.min_distance) || p.min_distance < 1) {
    throw new Error('min_duration and min_distance must be whole numbers of frames, at least 1');
  }
  return p;
}

const std = v => { const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); };

route('roi', async () => {
  const [d, info] = await Promise.all([getDemo(), api('/info')]);
  const [T, H, W] = d.shape, nm = d.pixel_size_nm;
  const st = { frame: 18, track: d.true_events[0]?.track_id ?? 0, channel: 'calcium', res: null };
  app().innerHTML = `
  <h1>ROI calcium analysis</h1>
  <p class="lead">Stage 3: for every tracked punctum, the mean calcium intensity in a small square ROI gives F(t), F₀ is a low
    percentile of F, ΔF/F₀ = (F − F₀)/F₀, and peaks above threshold × std(ΔF/F₀) are calcium events.</p>
  <p class="demo-label">${esc(d.description)} The ROI analysis uses the ground-truth tracks.</p>
  <div class="side"><div>${form(info.roi_defaults)}<p id="roimsg" class="small"></p></div>
  <div>
    <div class="row"><div><label for="frame">Frame <b id="fval"></b></label><input id="frame" type="range" min="0" max="${T - 1}" value="${st.frame}"></div>
      <div><label for="chan">Channel</label><select id="chan"><option value="calcium">calcium (ADU)</option><option value="piezo1">PIEZO1 (display-scaled)</option></select></div></div>
    <div class="imgwrap"><canvas id="view" width="${W * K}" height="${H * K}" aria-label="demo movie frame with ROI squares at tracked puncta"></canvas></div>
    <p class="small mute">${W} × ${H} px = ${(W * nm / 1000).toFixed(2)} × ${(H * nm / 1000).toFixed(2)} µm at ${nm} nm/px. Squares: ROIs of the tracks visible in this frame; the selected track is bold.</p>
  </div></div>
  <h2>Track</h2><div class="track-pick" id="pick" role="group" aria-label="select track"></div>
  <div id="charts"></div>
  <h2>Detected events</h2><div id="events"></div>
  <h2>Reading the result</h2>
  <ul class="small">
    <li>The generator clips directed tracks at the image border; there <code>extract_roi</code> pads the part of the ROI outside the image with zeros, so F drops and F₀ (a low percentile) drops with it. Watch those tracks for events that are artefacts of the padding.</li>
    <li>The event threshold is threshold × std(ΔF/F₀) measured from zero, while F₀ is the 10th percentile, so ΔF/F₀ sits above zero on average and noise peaks can cross a threshold of 2. The dashed boxes mark where events were really placed.</li>
  </ul>`;
  const view = document.getElementById('view'), off = document.createElement('canvas');
  const caScale = v => (v - 300) / (900 - 300) * 255;

  function rowsOf(id) {
    const r = [];
    d.tracks.track_id.forEach((t, i) => { if (t === id) r.push({ frame: d.tracks.frame[i], x: d.tracks.x[i], y: d.tracks.y[i] }); });
    return r;
  }
  function drawView() {
    document.getElementById('fval').textContent = `${st.frame} / ${T - 1}`;
    if (st.channel === 'calcium') drawFrame(off, d.calcium, H, W, st.frame, caScale, [120, 255, 170]);
    else drawFrame(off, d.piezo1, H, W, st.frame, v => v, [255, 110, 140]);
    const g = view.getContext('2d');
    g.imageSmoothingEnabled = false; g.drawImage(off, 0, 0, W * K, H * K);
    const rs = st.res ? st.res.params.roi_size : info.roi_defaults.roi_size, half = Math.floor(rs / 2);
    d.tracks.frame.forEach((f, i) => {
      if (f !== st.frame) return;
      const id = d.tracks.track_id[i], xi = roundHalfEven(d.tracks.x[i]), yi = roundHalfEven(d.tracks.y[i]);
      g.strokeStyle = colour(id); g.lineWidth = id === st.track ? 3 : 1.2;
      g.strokeRect((xi - half) * K, (yi - half) * K, rs * K, rs * K);
      g.fillStyle = colour(id); g.font = '12px monospace'; g.fillText(String(id), (xi + half + 1) * K, (yi - half) * K + 10);
    });
  }
  function drawCharts() {
    const r = st.res.tracks.find(t => t.track_id === st.track);
    const truth = d.true_events.filter(e => e.track_id === st.track).map(e => [e.start_frame, e.end_frame, 'truth']);
    const bands = r.events.map(e => [e.frame, e.end_frame, 'ev']);
    const valid = r.dff.filter(v => v !== null && !Number.isNaN(v));
    const thr = valid.length ? st.res.params.threshold * std(valid) : undefined;
    const n = rowsOf(st.track).length;
    document.getElementById('charts').innerHTML = `
      <div class="two"><div class="panel"><h3>F(t), track ${st.track}</h3>${lineChart([{ y: r.trace, colour: colour(st.track) }],
        { xlabel: 'frame', ylabel: 'F (ADU)', bands: [...bands, ...truth], label: `fluorescence trace of track ${st.track}` })}
        <p class="small mute">F₀ = ${fmt(r.baseline, 1)} ADU · ${n} of ${T} frames have a detection (gaps are blinks)</p></div>
      <div class="panel"><h3>ΔF/F₀, track ${st.track}</h3>${lineChart([{ y: r.dff, colour: colour(st.track), marks: r.events.map(e => e.peak_frame) }],
        { xlabel: 'frame', ylabel: 'ΔF/F₀', thr, bands: [...bands, ...truth], label: `dF/F0 of track ${st.track}` })}
        <p class="small mute">dashed line: threshold ${thr === undefined ? '' : fmt(thr)} · shaded: detected events · dashed boxes: events placed in the demo</p></div></div>`;
  }
  function drawEvents() {
    const all = st.res.tracks.flatMap(t => t.events.map(e => ({ ...e, track_id: t.track_id })));
    const real = e => d.true_events.some(x => x.track_id === e.track_id && e.peak_frame >= x.start_frame && e.peak_frame <= x.end_frame);
    const hits = all.filter(real).length;
    const found = d.true_events.filter(x => all.some(e => e.track_id === x.track_id && e.peak_frame >= x.start_frame && e.peak_frame <= x.end_frame)).length;
    document.getElementById('events').innerHTML = `
      <p>${all.length} events detected; ${hits} peak inside an event placed in the demo, ${all.length - hits} elsewhere.
        ${found} of ${d.true_events.length} placed events were found. <span class="badge ${st.res.computed_by === 'python' ? 'py' : 'js'}">computed by ${esc(st.res.computed_by)}</span></p>
      <div class="scroll"><table class="tbl"><thead><tr><th>track</th><th>start (frame)</th><th>peak (frame)</th><th>end (frame)</th><th>duration (frames)</th><th>amplitude (ΔF/F₀)</th><th>placed event?</th></tr></thead>
      <tbody>${list(all, e => `<tr class="click" data-t="${e.track_id}"><td><span class="sw" style="background:${colour(e.track_id)}"></span>${e.track_id}</td><td>${e.frame}</td><td>${e.peak_frame}</td><td>${e.end_frame}</td><td>${e.duration}</td><td>${fmt(e.amplitude)}</td><td>${real(e) ? 'yes' : 'no'}</td></tr>`)}</tbody></table></div>`;
    document.querySelectorAll('#events tr[data-t]').forEach(tr => tr.addEventListener('click', () => select(+tr.dataset.t)));
  }
  function drawPick() {
    document.getElementById('pick').innerHTML = list(st.res.tracks, t => `<button type="button" data-t="${t.track_id}" class="${t.track_id === st.track ? 'on' : ''}" aria-pressed="${t.track_id === st.track}">
      <span class="sw" style="background:${colour(t.track_id)}"></span>track ${t.track_id} · ${t.events.length} ev</button>`);
    document.querySelectorAll('#pick button').forEach(b => b.addEventListener('click', () => select(+b.dataset.t)));
  }
  function select(id) {
    st.track = id;
    const first = rowsOf(id)[0];
    if (first && !d.tracks.frame.some((f, i) => f === st.frame && d.tracks.track_id[i] === id)) { st.frame = first.frame; document.getElementById('frame').value = st.frame; }
    drawPick(); drawCharts(); drawView();
  }
  async function run(p) {
    const msg = document.getElementById('roimsg');
    msg.textContent = 'Running...'; msg.className = 'small mute';
    try {
      st.res = await api('/roi', { body: p });
      msg.textContent = ''; drawPick(); drawCharts(); drawEvents(); drawView();
    } catch (e) { msg.textContent = e.message; msg.className = 'small err'; }
  }
  const umHint = () => { const n = +document.getElementById('roi_size').value; document.getElementById('roi_um').textContent = `${n} px = ${(n * nm / 1000).toFixed(2)} µm at ${nm} nm/px`; };
  document.getElementById('roi_size').addEventListener('change', umHint); umHint();
  document.getElementById('frame').addEventListener('input', e => { st.frame = +e.target.value; drawView(); });
  document.getElementById('chan').addEventListener('change', e => { st.channel = e.target.value; drawView(); });
  document.getElementById('roiform').addEventListener('submit', e => {
    e.preventDefault();
    try { run(readForm()); } catch (err) { const m = document.getElementById('roimsg'); m.textContent = err.message; m.className = 'small err'; }
  });
  drawView();
  await run(info.roi_defaults);
});
