// ROI calcium analysis: a JS port of piezo1_magik/analysis/calcium_roi.py (CalciumROIAnalyzer) including the parts
// of numpy/pandas/scipy it relies on (np.round half-to-even, np.percentile linear, pandas centred rolling min,
// scipy.signal.find_peaks with height + distance). Tested against the Python code by roi_engine.test.mjs.
// Coordinates: pixel centres, x = column, y = row (as in the Python code). Movie: flat Float64Array/Uint16Array
// of T*H*W values, frame-major.

export const ROI_DEFAULTS = { roi_size: 5, baseline_method: 'percentile', baseline_percentile: 10.0, threshold: 2.0,
  min_duration: 2, min_distance: 5 };

// np.round: round half to even
export function roundHalfEven(v) {
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// CalciumROIAnalyzer.roi_mean: mean over the in-image pixels of the roi_size x roi_size square around the rounded
// centre, and their number; [NaN, 0] when the square misses the image or the centre is not finite.
export function roiMean(movie, shape, frame, x, y, roiSize) {
  const [, H, W] = shape, half = Math.floor(roiSize / 2);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return [NaN, 0];
  const y0 = roundHalfEven(y) - half, x0 = roundHalfEven(x) - half;
  const ya = Math.max(0, y0), yb = Math.min(H, y0 + roiSize), xa = Math.max(0, x0), xb = Math.min(W, x0 + roiSize);
  if (yb <= ya || xb <= xa) return [NaN, 0];
  let s = 0;
  const base = frame * H * W;
  for (let r = ya; r < yb; r++) for (let c = xa; c < xb; c++) s += movie[base + r * W + c];
  const n = (yb - ya) * (xb - xa);
  return [s / n, n];
}

// extract_trace_flags: F per frame (NaN where the track has no detection) and edge flags. track = [{frame, x, y}]
export function extractTrace(movie, shape, track, roiSize) {
  const T = shape[0], trace = new Array(T).fill(NaN), edge = new Array(T).fill(false);
  for (const p of track) {
    const f = Math.trunc(p.frame);
    if (f >= 0 && f < T) {
      const [m, n] = roiMean(movie, shape, f, p.x, p.y, roiSize);
      trace[f] = m; edge[f] = n < roiSize * roiSize;
    }
  }
  return { trace, edge };
}

// np.percentile, default 'linear' method (with numpy's lerp form)
export function percentile(values, q) {
  const v = [...values].sort((a, b) => a - b), n = v.length;
  const pos = (q / 100) * (n - 1), lo = Math.floor(pos), hi = Math.min(lo + 1, n - 1), t = pos - lo;
  const a = v[lo], b = v[hi], diff = b - a;
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

function median(values) {
  const v = values.filter(x => !Number.isNaN(x)).sort((a, b) => a - b), n = v.length;
  if (!n) return NaN;
  return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

// pandas Series.rolling(w, center=True).min(): window [i - w//2, i - w//2 + w - 1], NaN unless fully inside
export function rollingMinCentered(values, w) {
  const n = values.length, out = new Array(n).fill(NaN);
  if (w < 1) return out;
  const off = Math.floor(w / 2);
  for (let i = 0; i < n; i++) {
    const a = i - off, b = a + w - 1;
    if (a < 0 || b >= n) continue;
    let m = Infinity;
    for (let k = a; k <= b; k++) m = Math.min(m, values[k]);
    out[i] = m;
  }
  return out;
}

const mean = v => v.reduce((s, x) => s + x, 0) / v.length;

// compute_baseline: F0
export function computeBaseline(trace, method = 'percentile', pct = 10.0) {
  const valid = trace.filter(x => !Number.isNaN(x));
  if (!valid.length) return 0.0;
  if (method === 'percentile') return percentile(valid, pct);
  if (method === 'rolling') return median(rollingMinCentered(valid, Math.min(50, Math.floor(valid.length / 5))));
  return mean(valid);
}

// compute_delta_f_over_f
export function deltaFOverF(trace, method, pct) {
  const F0 = computeBaseline(trace, method, pct);
  if (F0 === 0) return trace.map(() => 0);
  return trace.map(f => (f - F0) / F0);
}

// scipy.signal.find_peaks(x, height=hmin, distance=d): indices
export function findPeaks(x, hmin, distance) {
  const n = x.length, peaks = [];
  let i = 1;
  while (i < n - 1) {
    if (x[i - 1] < x[i]) {
      let ahead = i + 1;
      while (ahead < n - 1 && x[ahead] === x[i]) ahead++;
      if (x[ahead] < x[i]) { peaks.push(Math.floor((i + ahead - 1) / 2)); i = ahead; }
    }
    i++;
  }
  let p = peaks.filter(k => x[k] >= hmin);
  if (distance != null && p.length > 1) {
    const d = Math.ceil(distance), keep = new Array(p.length).fill(true);
    const order = p.map((_, k) => k).sort((a, b) => x[p[a]] - x[p[b]] || a - b);
    for (let r = order.length - 1; r >= 0; r--) {
      const j = order[r];
      if (!keep[j]) continue;
      for (let k = j - 1; k >= 0 && p[j] - p[k] < d; k--) keep[k] = false;
      for (let k = j + 1; k < p.length && p[k] - p[j] < d; k++) keep[k] = false;
    }
    p = p.filter((_, k) => keep[k]);
  }
  return p;
}

// Baseline level and robust noise of a dF/F0 trace, as in detect_peaks: median and 1.4826 x MAD of the finite values
export function noiseLevels(dff, threshold = 2.0) {
  const valid = dff.filter(v => v !== null && Number.isFinite(v));
  if (!valid.length) return null;
  const level = median(valid), sigma = 1.4826 * median(valid.map(v => Math.abs(v - level)));
  return { level, sigma, thr: level + threshold * sigma, boundary: level + threshold * sigma / 2 };
}

// detect_peaks: events [{frame, peak_frame, amplitude, duration, end_frame}]; threshold above the trace's median
// in units of the robust noise sigma
export function detectPeaks(dff, threshold = 2.0, minDuration = 2, minDistance = 5) {
  const n = noiseLevels(dff, threshold);
  if (!n || n.sigma === 0 || Number.isNaN(n.sigma)) return [];
  const vt = dff.map(v => (v === null || !Number.isFinite(v) ? n.level : v)), events = [];
  for (const pk of findPeaks(vt, n.thr, minDistance)) {
    let start = pk, end = pk;
    while (start > 0 && vt[start - 1] > n.boundary) start--;
    while (end < vt.length - 1 && vt[end + 1] > n.boundary) end++;
    const duration = end - start + 1, last = events[events.length - 1];
    if (last && last.frame === start && last.end_frame === end) {  // same span: one event, at its highest peak
      if (vt[pk] > vt[last.peak_frame]) { last.peak_frame = pk; last.amplitude = dff[pk]; }
      continue;
    }
    if (duration >= minDuration) events.push({ frame: start, peak_frame: pk, amplitude: dff[pk], duration, end_frame: end });
  }
  return events;
}

// The demo analysis (same shape as POST /api/roi): every track, its trace, dF/F0, F0 and events.
export function runRoi(movie, shape, tracks, params = {}) {
  const p = { ...ROI_DEFAULTS, ...params };
  if (!Number.isInteger(p.roi_size) || p.roi_size < 1 || p.roi_size % 2 === 0) {
    throw new Error(`roi_size must be a positive odd number of pixels, got ${p.roi_size}`);
  }
  const ids = [...new Set(tracks.track_id)].sort((a, b) => a - b);
  const out = ids.map(id => {
    const rows = [];
    tracks.track_id.forEach((t, i) => { if (t === id) rows.push({ frame: tracks.frame[i], x: tracks.x[i], y: tracks.y[i] }); });
    const { trace, edge } = extractTrace(movie, shape, rows, p.roi_size);
    const dff = deltaFOverF(trace, p.baseline_method, p.baseline_percentile);
    return { track_id: id, baseline: computeBaseline(trace, p.baseline_method, p.baseline_percentile), trace, dff, edge,
      events: detectPeaks(dff, p.threshold, p.min_duration, p.min_distance) };
  });
  return { params: p, computed_by: 'javascript', tracks: out };
}
