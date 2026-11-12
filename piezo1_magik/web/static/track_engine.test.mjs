// Node test: track_engine.js reproduces piezo1_magik.tracking and the demo helpers of piezo1_magik/web/demo.py
// (tests/fixtures/engine_cases.json). Run: node track_engine.test.mjs
import assert from 'node:assert/strict';
import { greedyLinking, runTracks } from './track_engine.js';
import { close, fixture } from './test_util.mjs';

const fx = fixture();
let checks = 0;

for (const c of fx.links) {
  const m = greedyLinking(c.probs, c.src, c.dst, 0.4);
  assert.deepEqual(Object.fromEntries([...m].map(([k, v]) => [String(k), v])), c.result, 'greedy_linking');
  checks++;
}

for (const run of fx.tracks.runs) {
  const js = runTracks(fx.tracks.input, run.params), tag = JSON.stringify(run.params);
  for (const k of ['n_nodes', 'n_edges', 'n_true_edges', 'n_gt_tracks']) assert.equal(js[k], run[k], `${tag} ${k}`);
  for (const k of ['frame', 'gt']) assert.deepEqual(js.nodes[k], run.nodes[k], `${tag} nodes.${k}`);
  run.nodes.x.forEach((v, i) => close(js.nodes.x[i], v, `${tag} x`));
  for (const stage of ['linked', 'filled']) {
    const a = js[stage], b = run[stage];
    for (const k of ['lengths', 'temporal_spans', 'gaps', 'fragmentation', 'categories', 'assignment']) {
      assert.deepEqual(a[k], b[k], `${tag} ${stage}.${k}`); checks++;
    }
    assert.equal(a.num_tracks, b.num_tracks); assert.equal(a.mixed, b.mixed, `${tag} ${stage}.mixed`);
  }
  assert.equal(js.filled.iterations, run.filled.iterations, `${tag} iterations`);
}
console.log(`track_engine: ${checks} checks ok`);
