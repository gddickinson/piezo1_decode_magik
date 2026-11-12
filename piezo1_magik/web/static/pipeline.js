// Pipeline: the three stages from the README and the code, where each runs, and (local app) the network sizes.
import { api, app, esc, isStatic, list, needsServer, route } from './core.js';

route('pipeline', async () => {
  const p = await api('/pipeline');
  app().innerHTML = `
  <h1>Pipeline</h1>
  <p class="lead">Input: a dual-channel TIFF stack (PIEZO1 + calcium), loaded by <code>04_run_pipeline.py</code> as (2, T, H, W) or (T, 2, H, W).
    Positions are in pixels (pixel-centre convention, x = column, y = row; ${p.facts.pixel_size_nm} nm per pixel in the PSF models),
    times in frames (the frame interval is ${esc(p.facts.frame_interval)}).</p>
  <div class="flow" aria-label="pipeline flow">
    <span class="box">PIEZO1 channel</span><span class="arr">→</span>
    <span class="box local">1 DECODE</span><span class="arr">→</span><span class="box">detections</span><span class="arr">→</span>
    <span class="box local">2 MAGIK</span><span class="arr">→</span><span class="box web">linking + gap filling</span><span class="arr">→</span>
    <span class="box">tracks</span><span class="arr">+ calcium channel →</span><span class="box web">3 ROI analysis</span><span class="arr">→</span><span class="box">traces, events</span>
  </div>
  <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>needs PyTorch + trained checkpoint (local only)</span>
    <span><span class="sw" style="background:var(--accent2)"></span>runs in this site too</span></div>
  <div class="stages">${list(p.stages, s => `
    <section class="panel stage-card" id="${s.id}">
      <h2><span class="num">STAGE ${s.n}</span>${esc(s.title)}</h2>
      <p><span class="badge ${s.id === 'roi' ? 'js' : 'local'}">${esc(s.where)}</span></p>
      <dl class="kv"><dt>input</dt><dd>${esc(s.input)}</dd><dt>output</dt><dd>${esc(s.output)}</dd>
        <dt>code</dt><dd>${list(s.code, c => `<a href="https://github.com/gddickinson/piezo1_decode_magik/blob/main/${c}" target="_blank" rel="noopener">${esc(c)}</a>`, '<br>')}</dd></dl>
      <ul>${list(s.points, x => `<li>${esc(x)}</li>`)}</ul>
      ${s.id === 'roi' ? '<a class="btn alt" href="#/roi">Try it on the demo</a>' : s.id === 'magik' ? '<a class="btn alt" href="#/tracks">Try linking + gap filling</a>' : ''}
    </section>`)}</div>
  <h2>The networks in this code</h2>
  <div id="models" class="panel">${isStatic() ? needsServer('Building the DECODE and MAGIK networks to count their parameters') : '<p class="mute">Building the networks...</p>'}</div>
  <h2>Training and evaluation</h2>
  <p>Both networks are trained on synthetic movies with ground truth from <code>01_generate_synthetic_data.py</code>
    (<a href="#/psf">Gaussian PSF</a>, Poisson and read noise, optional blinking). DECODE: <code>02_train_decode.py</code>, evaluated by <code>02c</code>.
    MAGIK: <code>03_train_magik.py</code> builds its graphs from DECODE's detections on those movies, evaluated by <code>03c</code>, <code>07</code> (track lengths, gaps,
    fragmentation) and <code>08</code> (gap filling). <code>09_run_complete_pipeline.py</code> runs DECODE + MAGIK (+ gap filling) on a movie;
    <code>04_run_pipeline.py</code> runs all three stages. Trained checkpoints are not part of the repository.</p>`;
  if (!isStatic()) {
    const box = document.getElementById('models');
    try {
      const m = await api('/models');
      box.innerHTML = `<dl class="kv"><dt>DECODENet (base_channels 32)</dt><dd>${m.decode.parameters.toLocaleString()} parameters; outputs on a 1×3×64×64 input:
        ${list(Object.entries(m.decode.outputs), ([k, v]) => `${esc(k)} ${v.join('×')}`, ', ')}</dd>
        <dt>MAGIKNet (hidden 64, 3 layers)</dt><dd>${m.magik.parameters.toLocaleString()} parameters</dd><dt>PyTorch</dt><dd>${esc(m.torch)}</dd></dl>
        <p class="small mute">${esc(m.note)}</p>`;
    } catch (e) { box.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }
});
