/**
 * Small, dependency-free color & math helpers for visualizations.
 */

export function clamp(x, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, x));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

function mix(c1, c2, t) {
  return [
    Math.round(lerp(c1[0], c2[0], t)),
    Math.round(lerp(c1[1], c2[1], t)),
    Math.round(lerp(c1[2], c2[2], t)),
  ];
}

function rgb([r, g, b], a = 1) {
  return a >= 1 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${a})`;
}

// A perceptual-ish ramp from deep ink -> indigo -> cyan -> white.
const HEAT_STOPS = [
  [11, 14, 26], // ink
  [36, 51, 175], // indigo
  [58, 85, 245], // brand
  [34, 211, 238], // cyan
  [233, 250, 255], // near white
];

/** value in [0,1] -> css color along the heat ramp. */
export function heat(t, alpha = 1) {
  t = clamp(t);
  const seg = t * (HEAT_STOPS.length - 1);
  const i = Math.min(Math.floor(seg), HEAT_STOPS.length - 2);
  return rgb(mix(HEAT_STOPS[i], HEAT_STOPS[i + 1], seg - i), alpha);
}

// Diverging ramp: rose (negative) -> slate (zero) -> emerald (positive).
const NEG = [251, 113, 133];
const MID = [51, 65, 85];
const POS = [52, 211, 153];

/** value in [-1,1] -> diverging color. */
export function diverge(t, alpha = 1) {
  t = clamp(t, -1, 1);
  if (t < 0) return rgb(mix(MID, NEG, -t), alpha);
  return rgb(mix(MID, POS, t), alpha);
}

/** Numerically stable softmax over an array. */
export function softmax(arr, temperature = 1) {
  const t = Math.max(1e-6, temperature);
  const scaled = arr.map((x) => x / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Seeded pseudo-random generator (deterministic across renders). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function fmt(x, d = 2) {
  if (!isFinite(x)) return "–";
  return x.toFixed(d);
}
