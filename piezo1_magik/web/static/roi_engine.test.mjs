// Node test: roi_engine.js reproduces CalciumROIAnalyzer and scipy.signal.find_peaks
// (tests/fixtures/engine_cases.json, written by scripts/make_engine_fixture.py). Run: node roi_engine.test.mjs
import assert from 'node:assert/strict';
import { detectPeaks, findPeaks, noiseLevels, roundHalfEven, runRoi } from './roi_engine.js';
import { close, decodeU16, fixture } from './test_util.mjs';

const fx = fixture();
let checks = 0;

assert.deepEqual([0.5, 1.5, 2.5, -0.5, -1.5, 2.4999, 2.5001].map(roundHalfEven), [0, 2, 2, 0, -2, 2, 3]);

for (const c of fx.peaks) {
  assert.deepEqual(findPeaks(c.x, c.height, c.distance), c.peaks, `find_peaks ${JSON.stringify(c).slice(0, 80)}`);
  checks++;
}

for (const c of fx.noise) {
  const dff = c.dff.map(v => (v === null ? NaN : v)), lv = noiseLevels(dff, 2.0, c.window), tag = `noise window ${c.window}`;
  c.level.forEach((v, k) => { close(lv.level[k], v, `${tag} level[${k}]`); close(lv.sigma[k], c.sigma[k], `${tag} sigma[${k}]`); checks++; });
  const ev = detectPeaks(dff, 2.0, 2, 3, c.window);
  assert.deepEqual(ev.map(e => [e.frame, e.peak_frame, e.end_frame, e.duration]),
    c.events.map(e => [e.frame, e.peak_frame, e.end_frame, e.duration]), `${tag} events`);
  checks++;
}
assert.throws(() => noiseLevels([0, 1, 2], 2, -1), /noise_window/);

const movie = decodeU16(fx.roi.movie_b64);
for (const run of fx.roi.runs) {
  const js = runRoi(movie, fx.roi.shape, fx.roi.tracks, run.params);
  assert.equal(js.tracks.length, run.tracks.length);
  run.tracks.forEach((py, i) => {
    const j = js.tracks[i], tag = `${JSON.stringify(run.params)} track ${py.track_id}`;
    assert.equal(j.track_id, py.track_id);
    close(j.baseline, py.baseline, `${tag} baseline`);
    py.trace.forEach((v, k) => { close(j.trace[k], v, `${tag} F[${k}]`); checks++; });
    py.dff.forEach((v, k) => { close(j.dff[k], v, `${tag} dff[${k}]`); checks++; });
    assert.deepEqual(j.edge, py.edge, `${tag} edge flags`);
    assert.equal(j.events.length, py.events.length, `${tag} number of events`);
    py.events.forEach((e, k) => {
      for (const key of ['frame', 'peak_frame', 'duration', 'end_frame']) assert.equal(j.events[k][key], e[key], `${tag} ${key}`);
      close(j.events[k].amplitude, e.amplitude, `${tag} amplitude`);
      checks++;
    });
  });
}
assert.throws(() => runRoi(movie, fx.roi.shape, fx.roi.tracks, { roi_size: 4 }), /odd/);
console.log(`roi_engine: ${checks} checks ok`);
