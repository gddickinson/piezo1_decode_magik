// About: what the project is, how to run it (scripts, web app), what the static site does, credits.
import { api, app, esc, isStatic, list, route } from './core.js';

route('about', async () => {
  const i = await api('/info');
  app().innerHTML = `
  <h1>About</h1>
  <p class="lead">PIEZO1 DECODE-MAGIK is George Dickinson's (UC Irvine) modular pipeline for tracking PIEZO1 puncta and reading calcium
    signals at them: DECODE localization, MAGIK graph-network tracking and ROI-based calcium analysis. MIT licence.</p>
  <h2>Run the pipeline (Python)</h2>
  <pre>git clone https://github.com/gddickinson/piezo1_decode_magik.git
cd piezo1_decode_magik
conda create -n piezo1_magik python=3.11 && conda activate piezo1_magik
pip install -r requirements.txt
pip install -e .

python scripts/01_generate_synthetic_data.py --output data/synthetic --num_samples 200 --with_blinking
python scripts/02_train_decode.py --config configs/decode_training.yaml --data data/synthetic --output checkpoints/decode
python scripts/03_train_magik.py --config configs/magik_training.yaml --data data/synthetic \\
    --decode checkpoints/decode/best_model.pth --output checkpoints/magik
python scripts/04_run_pipeline.py --decode checkpoints/decode/best_model.pth \\
    --magik checkpoints/magik/best_model.pth --input movie.tif --output results/</pre>
  <h2>Run the web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py            # http://127.0.0.1:8209 (local only)</pre>
  <p>In the local app the ROI and track analyses run in the repo's Python code, and the Pipeline page builds the (untrained) networks with
    PyTorch to report their size. ${isStatic() ? 'On this static site they run in your browser as JS ports that the tests check against the Python output.' : ''}
    DECODE and MAGIK themselves need trained checkpoints and run through the scripts above.</p>
  <h2>What is on this site</h2>
  <ul><li>Pipeline and parameter pages generated from the code and <code>configs/</code>.</li>
    <li>A synthetic demo movie (${i.demo.frames} frames, ${i.demo.width} × ${i.demo.height} px) made by the repo's generator; it is labelled as synthetic wherever it appears.</li>
    <li>No measured data and no trained models: the performance figures in the README ("expected performance") are targets, not results reproduced here.</li></ul>
  <h2>Citations (from the README)</h2>
  <ul>${list(i.facts.citations, c => `<li>${esc(c)}</li>`)}</ul>
  <p class="small mute">Built with AI assistance from Claude (Anthropic).</p>`;
});
