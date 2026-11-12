// Shared helpers for the node engine tests: load the Python fixture, compare numbers with a tolerance.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

export const fixture = () => JSON.parse(readFileSync(new URL('../../../tests/fixtures/engine_cases.json', import.meta.url), 'utf8'));

// Python NaN arrives as null (JSON); tolerance rtol/atol 1e-9 unless given
export function close(js, py, what, tol = 1e-9) {
  if (py === null) return assert.ok(js === null || Number.isNaN(js), `${what}: expected NaN, got ${js}`);
  assert.ok(Math.abs(js - py) <= tol * Math.max(1, Math.abs(py)), `${what}: ${js} vs ${py}`);
}

export function decodeU16(b64) {
  const buf = Buffer.from(b64, 'base64');
  return new Uint16Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}
