// Entry point: lazy page modules, the mode label, a subtle blinking-puncta background, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = {
  'home.js': ['home'], 'pipeline.js': ['pipeline'], 'params.js': ['params'], 'roi.js': ['roi'],
  'tracks.js': ['tracks'], 'psf.js': ['psf'], 'about.js': ['about'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: ROI and track analysis run in your browser (JS ports tested against the Python code); DECODE and MAGIK need the local Python install'
  : 'local app: ROI and track analysis run in Python on this computer';

// Background: a few faint puncta drifting and blinking. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const ps = Array.from({ length: 24 }, () => ({ x: Math.random(), y: Math.random(), on: Math.random() > 0.2, ca: 0 }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 90) return;
    last = t;
    g.fillStyle = 'rgba(5,6,12,.25)'; g.fillRect(0, 0, canvas.width, canvas.height);
    for (const p of ps) {
      p.x = (p.x + (Math.random() - 0.5) * 0.002 + 1) % 1; p.y = (p.y + (Math.random() - 0.5) * 0.002 + 1) % 1;
      if (Math.random() < 0.05) p.on = !p.on;
      if (Math.random() < 0.004) p.ca = 1;
      p.ca *= 0.93;
      const X = p.x * canvas.width, Y = p.y * canvas.height;
      if (p.ca > 0.05) { g.fillStyle = `rgba(71,240,160,${0.25 * p.ca})`; g.beginPath(); g.arc(X, Y, 26, 0, 7); g.fill(); }
      if (p.on) { g.fillStyle = 'rgba(255,90,122,.8)'; g.beginPath(); g.arc(X, Y, 2.2, 0, 7); g.fill(); }
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
