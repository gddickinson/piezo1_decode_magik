// Track post-processing: a JS port of piezo1_magik/tracking/linking.py and track_stats.py (candidate graph, greedy
// one-to-one linking, fragments, gap filling, track statistics) plus the demo helpers of piezo1_magik/web/demo.py
// (oracle links from ground truth, mixed-track count). Tested against Python by track_engine.test.mjs.
// Positions in pixels, times in frames.

export const TRACK_DEFAULTS = { max_temporal_gap: 2, max_spatial_distance: 30.0, max_gap: 10, max_distance: 30.0 };

// build_candidate_graph: nodes in frame order; edges i -> j > i with 0 < gap <= maxGap and distance <= maxDist
export function buildCandidateGraph(frames, xs, ys, maxGap, maxDist) {
  const src = [], dst = [], feats = [], n = frames.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const gap = frames[j] - frames[i];
      if (gap <= 0 || gap > maxGap) continue;
      const dx = xs[j] - xs[i], dy = ys[j] - ys[i], d = Math.sqrt(dx ** 2 + dy ** 2);
      if (d <= maxDist) { src.push(i); dst.push(j); feats.push([dx, dy, d, gap]); }
    }
  }
  return { src, dst, feats };
}

// greedy_linking -> Map(node -> track id). Iteration orders follow Python's: sets of small ints iterate in
// ascending order, dicts in insertion order.
export function greedyLinking(probs, src, dst, threshold = 0.5) {
  const idx = [];
  probs.forEach((p, k) => { if (p > threshold) idx.push(k); });
  if (!idx.length) return new Map();
  idx.sort((a, b) => probs[b] - probs[a] || a - b);
  const sL = new Set(), dL = new Set(), fwd = new Map(), bwd = new Map();
  for (const k of idx) {
    const s = src[k], d = dst[k];
    if (!sL.has(s) && !dL.has(d)) { sL.add(s); dL.add(d); fwd.set(s, d); bwd.set(d, s); }
  }
  const all = [...new Set([...fwd.keys(), ...bwd.keys()])].sort((a, b) => a - b);
  const n2t = new Map();
  let c = 0;
  for (const start of all.filter(n => !bwd.has(n))) {
    if (n2t.has(start)) continue;
    let cur = start;
    while (cur !== undefined) { n2t.set(cur, c); cur = fwd.get(cur); }
    c++;
  }
  for (const n of all) if (!n2t.has(n)) n2t.set(n, c++);
  return n2t;
}

// build_track_fragments (node features: frames[], xs[], ys[])
export function buildFragments(n2t, frames, xs, ys) {
  const by = new Map();
  for (const [node, t] of n2t) { if (!by.has(t)) by.set(t, []); by.get(t).push(node); }
  const out = [];
  for (const [t, nodes] of by) {
    const d = nodes.map(n => [frames[n], xs[n], ys[n], n]);
    d.sort((a, b) => a[0] - b[0]);  // stable, like Python's sort
    const fr = d.map(r => r[0]), pos = d.map(r => [r[1], r[2]]);
    out.push({ track_id: t, nodes: d.map(r => r[3]), frames: fr, positions: pos, start_frame: fr[0],
      end_frame: fr[fr.length - 1], start_pos: pos[0], end_pos: pos[pos.length - 1], length: d.length });
  }
  return out;
}

function predict(f, target) {
  let vx = 0, vy = 0;
  const dt = f.end_frame - f.start_frame;
  if (f.frames.length >= 2 && dt !== 0) { vx = (f.end_pos[0] - f.start_pos[0]) / dt; vy = (f.end_pos[1] - f.start_pos[1]) / dt; }
  const k = target - f.end_frame;
  return [f.end_pos[0] + vx * k, f.end_pos[1] + vy * k];
}

// can_merge_fragments -> score or null
export function mergeScore(f1, f2, maxGap, maxDist) {
  if (f1.end_frame >= f2.start_frame) return null;
  const gap = f2.start_frame - f1.end_frame;
  if (gap > maxGap) return null;
  const [px, py] = predict(f1, f2.start_frame);
  const d = Math.sqrt((px - f2.start_pos[0]) ** 2 + (py - f2.start_pos[1]) ** 2);
  return d > maxDist ? null : d + gap * 2;
}

// merge_track_fragments -> { fragments, iterations }
export function mergeFragments(fragments, maxGap = 10, maxDist = 30, maxIter = null) {
  const active = fragments.map(f => ({ ...f })), merged = new Set();
  const limit = maxIter ?? fragments.length + 1;
  let it = 0;
  while (it < limit) {
    it++;
    let best = null;
    for (let i = 0; i < active.length; i++) {
      if (merged.has(i)) continue;
      for (let j = 0; j < active.length; j++) {
        if (merged.has(j) || i === j) continue;
        const q = mergeScore(active[i], active[j], maxGap, maxDist);
        if (q !== null && (best === null || q < best[2])) best = [i, j, q];  // first minimum, like a stable sort
      }
    }
    if (!best) break;
    const [i, j] = best, a = active[i], b = active[j];
    active[i] = { track_id: a.track_id, nodes: a.nodes.concat(b.nodes), frames: a.frames.concat(b.frames),
      positions: a.positions.concat(b.positions), start_frame: a.start_frame, end_frame: b.end_frame,
      start_pos: a.start_pos, end_pos: b.end_pos, length: a.length + b.length };
    merged.add(j);
  }
  return { fragments: active.filter((_, i) => !merged.has(i)), iterations: it };
}

export function fragmentsToMap(frags) {
  const m = new Map();
  frags.forEach((f, t) => f.nodes.forEach(n => m.set(n, t)));
  return m;
}

// analyze_track_lengths (track order: first appearance in the map)
export function trackLengths(n2t, frames) {
  const by = new Map();
  for (const [n, t] of n2t) { if (!by.has(t)) by.set(t, []); by.get(t).push(frames[n]); }
  const lengths = [], spans = [], gaps = [];
  for (const fs of by.values()) {
    fs.sort((a, b) => a - b);
    lengths.push(fs.length);
    spans.push(fs[fs.length - 1] - fs[0] + 1);
    gaps.push(fs[fs.length - 1] - fs[0] + 1 - new Set(fs.map(Math.trunc)).size);
  }
  return { lengths, temporal_spans: spans, gaps, num_tracks: by.size };
}

// analyze_fragmentation (GT tracks in order of first appearance)
export function fragmentation(n2t, gt) {
  const by = new Map();
  gt.forEach((t, n) => { if (!by.has(t)) by.set(t, []); by.get(t).push(n); });
  return [...by.values()].map(nodes => new Set(nodes.filter(n => n2t.has(n)).map(n => n2t.get(n))).size);
}

export const lengthCategories = L => ({ short: L.filter(n => n <= 3).length,
  medium: L.filter(n => n > 3 && n <= 10).length, long: L.filter(n => n > 10).length });

// demo helpers (piezo1_magik/web/demo.py)
export function oracleProbs(gt, src, dst) {
  const order = gt.map((_, i) => i).sort((a, b) => gt[a] - gt[b] || a - b), next = new Map();
  for (let k = 0; k + 1 < order.length; k++) if (gt[order[k]] === gt[order[k + 1]]) next.set(order[k], order[k + 1]);
  return src.map((s, k) => (next.get(s) === dst[k] ? 1 : 0));
}

export function mixedTracks(n2t, gt) {
  const by = new Map();
  for (const [n, t] of n2t) { if (!by.has(t)) by.set(t, new Set()); by.get(t).add(gt[n]); }
  return [...by.values()].filter(s => s.size > 1).length;
}

function summarize(n2t, frames, gt) {
  const s = trackLengths(n2t, frames);
  return { ...s, fragmentation: fragmentation(n2t, gt), categories: lengthCategories(s.lengths),
    mixed: mixedTracks(n2t, gt), assignment: Object.fromEntries([...n2t].map(([k, v]) => [String(k), v])) };
}

// The demo analysis (same shape as POST /api/tracks). tracks: {track_id[], frame[], x[], y[]}
export function runTracks(tracks, params = {}) {
  const p = { ...TRACK_DEFAULTS, ...params };
  const idx = tracks.frame.map((_, i) => i).sort((a, b) => tracks.frame[a] - tracks.frame[b]
    || tracks.track_id[a] - tracks.track_id[b] || a - b);
  const frames = idx.map(i => tracks.frame[i]), xs = idx.map(i => tracks.x[i]), ys = idx.map(i => tracks.y[i]);
  const gt = idx.map(i => tracks.track_id[i]);
  const g = buildCandidateGraph(frames, xs, ys, p.max_temporal_gap, p.max_spatial_distance);
  const probs = oracleProbs(gt, g.src, g.dst);
  const linked = greedyLinking(probs, g.src, g.dst, 0.5);
  const m = mergeFragments(buildFragments(linked, frames, xs, ys), p.max_gap, p.max_distance);
  const filled = fragmentsToMap(m.fragments);
  return { params: p, computed_by: 'javascript', n_nodes: frames.length, n_edges: g.src.length,
    n_true_edges: probs.reduce((a, b) => a + b, 0), n_gt_tracks: new Set(gt).size,
    nodes: { frame: frames, x: xs, y: ys, gt },
    linked: summarize(linked, frames, gt), filled: { ...summarize(filled, frames, gt), iterations: m.iterations } };
}
