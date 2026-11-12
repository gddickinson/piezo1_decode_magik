// Demo dataset loader (shared by pages and static_api.js): fetches /demo once and decodes the base64 movies.
import { api } from './core.js';

let cached = null;
function b64(s) {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// { ...json, calcium: Uint16Array T*H*W (ADU), piezo1: Uint8Array T*H*W (display-scaled) }
export async function getDemo() {
  if (!cached) {
    cached = api('/demo').then(d => {
      const raw = b64(d.calcium_b64);
      const dv = new DataView(raw.buffer), calcium = new Uint16Array(raw.length / 2);
      for (let i = 0; i < calcium.length; i++) calcium[i] = dv.getUint16(2 * i, true);
      return { ...d, calcium, piezo1: b64(d.piezo1_u8_b64) };
    });
  }
  return cached;
}

export const PALETTE = ['#ff5a7a', '#47f0a0', '#5ab4ff', '#ffd23f', '#c77dff', '#ff9f43', '#2ee6ff', '#f5f5f5',
  '#9cff57', '#ff6fd8'];
export const colour = i => PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];
export const um = (px, nm) => (px * nm / 1000);
export const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? 'NaN' : Number(v).toFixed(d));
