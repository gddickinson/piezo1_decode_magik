// Small SVG chart helpers (no libraries): line chart with optional threshold line and shaded intervals.
import { esc } from './core.js';

const nice = (lo, hi) => (hi > lo ? [lo, hi] : [lo - 1, hi + 1]);

// series: [{ y: [...], cls/colour, label }]; opts: { xlabel, ylabel, thr, bands: [[x0, x1, cls]], h }
export function lineChart(series, opts = {}) {
  const W = 640, H = opts.h || 220, L = 52, R = 10, T = 12, B = 34;
  const n = Math.max(...series.map(s => s.y.length));
  const vals = series.flatMap(s => s.y.filter(v => v !== null && !Number.isNaN(v)));
  if (opts.thr !== undefined) vals.push(opts.thr);
  const [lo, hi] = nice(Math.min(...vals), Math.max(...vals));
  const X = i => L + (i / Math.max(1, n - 1)) * (W - L - R), Y = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const ticks = [lo, (lo + hi) / 2, hi];
  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label || 'chart')}">`;
  for (const [a, b, cls] of opts.bands || []) svg += `<rect class="${cls}" x="${X(a) - 3}" y="${T}" width="${X(b) - X(a) + 6}" height="${H - T - B}"/>`;
  for (const t of ticks) svg += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${L - 6}" y="${Y(t) + 4}" text-anchor="end">${t.toPrecision(3)}</text>`;
  svg += `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/><line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>`;
  for (let i = 0; i < n; i += Math.max(1, Math.ceil(n / 8))) svg += `<text x="${X(i)}" y="${H - B + 14}" text-anchor="middle">${i}</text>`;
  if (opts.xlabel) svg += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(opts.xlabel)}</text>`;
  if (opts.ylabel) svg += `<text x="12" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(T + H - B) / 2})">${esc(opts.ylabel)}</text>`;
  if (opts.thr !== undefined) svg += `<line class="thr" x1="${L}" x2="${W - R}" y1="${Y(opts.thr)}" y2="${Y(opts.thr)}"/>`;
  for (const s of series) {
    let d = '', pen = false;
    s.y.forEach((v, i) => {
      if (v === null || Number.isNaN(v)) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`; pen = true;
    });
    svg += `<path d="${d}" fill="none" stroke="${s.colour}" stroke-width="1.6"/>`;
    for (const i of s.marks || []) if (s.y[i] !== null) svg += `<circle cx="${X(i)}" cy="${Y(s.y[i])}" r="4" fill="none" stroke="${s.colour}" stroke-width="2"/>`;
  }
  return svg + '</svg>';
}

// Draw a T*H*W frame (Uint8 or scaled Uint16) into a canvas, grey or tinted. scale(v) -> 0..255
export function drawFrame(canvas, data, H, W, t, scale = v => v, tint = [255, 255, 255]) {
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext('2d'), img = g.createImageData(W, H), base = t * H * W;
  for (let i = 0; i < H * W; i++) {
    const v = Math.max(0, Math.min(255, scale(data[base + i])));
    img.data[4 * i] = v * tint[0] / 255; img.data[4 * i + 1] = v * tint[1] / 255; img.data[4 * i + 2] = v * tint[2] / 255; img.data[4 * i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
}
