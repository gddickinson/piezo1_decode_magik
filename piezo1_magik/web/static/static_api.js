// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py, and runs the ROI
// and track analyses in the browser (roi_engine.js, track_engine.js: JS ports tested against the Python code).
// DECODE and MAGIK themselves need PyTorch and trained checkpoints, so /models says it needs the local app.
import { getDemo } from './data.js';
import { runRoi } from './roi_engine.js';
import { runTracks } from './track_engine.js';

const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export const NEEDS_SERVER = 'needs the full app running locally (Python server with PyTorch): see the README, "Run the web app"';

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'roi') { const d = await getDemo(); return runRoi(d.calcium, d.shape, d.tracks, body || {}); }
  if (clean === 'tracks') { const d = await getDemo(); return runTracks(d.tracks, body || {}); }
  if (clean === 'models') throw new Error('Building the DECODE and MAGIK networks ' + NEEDS_SERVER + '.');
  return file(clean);
}
