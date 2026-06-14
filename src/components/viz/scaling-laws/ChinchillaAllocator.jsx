import { useMemo, useState } from "react";
import { clamp } from "../scales";
import { Slider, Stat } from "../ui";
import LineChart from "../LineChart";

/**
 * ChinchillaAllocator — an interactive compute-budget allocator.
 *
 * Top slider: total training compute C (log scale, 1e18 .. 1e25 FLOPs).
 *
 * From C we derive the Chinchilla-OPTIMAL split (N*, D*) using the constraint
 * C = 6 N D together with the empirical ~20-tokens-per-parameter rule:
 *
 *     N* = sqrt(C / (6 * R)),   D* = R * N*,    with R = 20.
 *
 * This satisfies 6 N* D* = 6 * R * (N*)^2 = C exactly, and the ratio D* to N*
 * equals R = 20, i.e. both N* and D* scale as C^0.5 — the central Chinchilla
 * result.
 *
 * Second slider: pick a DIFFERENT parameter count N at the SAME budget. The
 * data is then forced to D = C / (6N) to stay on budget. We score every choice
 * with a Chinchilla-style parametric loss surface
 *
 *     L(N, D) = E + A / N^a + B / D^a.
 *
 * The real paper (Hoffmann et al. 2022) fits irreducible E = 1.69 and the
 * form L = E + 406.4/N^0.34 + 410.7/D^0.28. We use the SAME shape but a single
 * symmetric exponent a, with B = A * RATIO^a, so that the loss minimum along
 * the budget line lands exactly at D/N = RATIO for EVERY budget — which is the
 * paper's headline ("scale N and D together, ~20 tokens/param"). Sweeping N at
 * fixed C traces a U-shaped curve: too small a model is over-trained
 * (data-bottlenecked), too large is under-trained (parameter-bottlenecked).
 * Fully deterministic — no randomness, no animation.
 */

// Tokens-per-parameter target for the compute-optimal split (the Chinchilla rule).
const RATIO = 20;

// Chinchilla-style parametric-loss coefficients. E and the loss magnitudes are
// realistic; the symmetric exponent a pins the budget-constrained optimum at
// exactly D/N = RATIO (see header).
const E = 1.69;
const A = 400;
const EXP = 0.3;
const B = A * Math.pow(RATIO, EXP); // ≈ 982.6, so the U-floor sits at 20:1

function lossOf(N, D) {
  return E + A / Math.pow(N, EXP) + B / Math.pow(D, EXP);
}

// Pretty-print a big count as 1.4e9 -> "1.4B", 7e8 -> "700M", 1.4e12 -> "1.4T".
function human(x) {
  if (!isFinite(x) || x <= 0) return "–";
  const units = [
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "k"],
  ];
  for (const [div, suf] of units) {
    if (x >= div) {
      const v = x / div;
      return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)}${suf}`;
    }
  }
  return x.toFixed(0);
}

// Pretty-print FLOPs as a power of ten with a mantissa, e.g. 6.0e22.
function flops(x) {
  if (!isFinite(x) || x <= 0) return "–";
  const e = Math.floor(Math.log10(x));
  const m = x / Math.pow(10, e);
  return `${m.toFixed(1)}e${e}`;
}

export default function ChinchillaAllocator() {
  // C is controlled in log10 space so the slider is uniform across decades.
  const [logC, setLogC] = useState(22); // 1e22 FLOPs (~ a few-hundred-GPU run)
  const C = Math.pow(10, logC);

  // Compute-optimal split for this budget.
  const Nstar = Math.sqrt(C / (6 * RATIO));
  const Dstar = RATIO * Nstar;
  const Lstar = lossOf(Nstar, Dstar);

  // The user's chosen model size, expressed as a multiplier on N* in log space
  // so "0" always means "exactly optimal" regardless of the budget.
  const [logMult, setLogMult] = useState(0); // log10(N / N*), range [-1.3, 1.3]
  const N = Nstar * Math.pow(10, logMult);
  const D = C / (6 * N); // forced onto the budget constraint
  const L = lossOf(N, D);
  const ratio = D / N; // tokens per parameter at the user's choice
  const excess = L - Lstar; // loss penalty vs. the optimum

  // U-curve: loss vs N at fixed C, with D = C/(6N) everywhere on the budget.
  const curve = useMemo(() => {
    const pts = [];
    const STEPS = 90;
    const lo = -1.3; // N as low as N*/20
    const hi = 1.3; //  N as high as N* * 20
    for (let i = 0; i <= STEPS; i++) {
      const lm = lo + ((hi - lo) * i) / STEPS;
      const n = Nstar * Math.pow(10, lm);
      const d = C / (6 * n);
      pts.push([n, lossOf(n, d)]);
    }
    return pts;
  }, [C, Nstar]);

  // A single marker series for the user's current pick (a 1-point "line").
  const youPoint = [[N, L]];

  // y-window padded a little around the visible curve for a clean U.
  const ys = curve.map((p) => p[1]);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  return (
    <div className="space-y-5">
      {/* ---- budget control ---- */}
      <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
        <Slider
          label="total compute budget  C  (FLOPs)"
          min={18}
          max={25}
          step={0.1}
          value={logC}
          onChange={setLogC}
          format={(v) => `10^${v.toFixed(1)} ≈ ${flops(Math.pow(10, v))}`}
        />
        <div className="mt-2 text-xs text-slate-500">
          For scale: GPT-3 used ≈ 3.1e23 FLOPs; a 7B model on 2T tokens is ≈ 8.4e22 FLOPs.
        </div>
      </div>

      {/* ---- compute-optimal readouts ---- */}
      <div>
        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-emerald-300/80">
          Chinchilla-optimal split for this budget
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="params N*" value={human(Nstar)} accent="text-emerald-200" />
          <Stat label="tokens D*" value={human(Dstar)} accent="text-emerald-200" />
          <Stat label="tokens / param" value={`${RATIO.toFixed(0)}×`} accent="text-emerald-200" />
          <Stat label="optimal loss" value={Lstar.toFixed(3)} accent="text-emerald-200" />
        </div>
      </div>

      {/* ---- misallocation control ---- */}
      <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
        <Slider
          label="your model size  N  (same budget, so D = C / 6N)"
          min={-1.3}
          max={1.3}
          step={0.02}
          value={logMult}
          onChange={setLogMult}
          accent="violet"
          format={(v) =>
            Math.abs(v) < 0.02 ? "optimal" : `${v > 0 ? "×" : "÷"}${Math.pow(10, Math.abs(v)).toFixed(1)} N*`
          }
        />
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="your params N" value={human(N)} accent="text-violet-200" />
          <Stat label="your tokens D" value={human(D)} accent="text-violet-200" />
          <Stat
            label="tokens / param"
            value={`${ratio >= 100 ? ratio.toFixed(0) : ratio.toFixed(1)}×`}
            accent={clamp(Math.abs(Math.log10(ratio / RATIO)), 0, 1) < 0.15 ? "text-emerald-200" : "text-amber-200"}
          />
          <Stat
            label="loss penalty"
            value={excess < 1e-3 ? "0.000" : `+${excess.toFixed(3)}`}
            accent={excess < 1e-3 ? "text-emerald-200" : "text-rose-200"}
          />
        </div>
        <div className="mt-2 text-xs leading-relaxed text-slate-500">
          {excess < 1e-3 ? (
            <>You are spending the budget optimally — N and D sit at the bottom of the U.</>
          ) : ratio > RATIO ? (
            <>
              Too <strong className="text-amber-200">small</strong> a model: it is{" "}
              <em>over-trained</em> ({ratio >= 100 ? ratio.toFixed(0) : ratio.toFixed(0)} tokens/param), data-bottlenecked. Cheap to serve, but
              it leaves loss on the table for this budget.
            </>
          ) : (
            <>
              Too <strong className="text-amber-200">big</strong> a model: it is{" "}
              <em>under-trained</em> ({ratio.toFixed(1)} tokens/param), starved of data. You burned compute on
              parameters you couldn't feed.
            </>
          )}
        </div>
      </div>

      {/* ---- the U-curve ---- */}
      <LineChart
        height={320}
        xScale="log"
        yScale="linear"
        xLabel="model parameters  N  (log scale, fixed budget C)"
        yLabel="test loss  L(N, C/6N)"
        series={[
          { label: "loss along the budget", color: "#22d3ee", points: curve },
          { label: "your choice", color: "#a855f7", points: youPoint },
        ]}
        annotations={[{ x: Nstar, label: "optimum ≈ 20 tok/param", color: "#34d399" }]}
        fmtX={(v) => human(v)}
        fmtY={(v) => v.toFixed(2)}
        yTicks={5}
      />
      <p className="text-xs leading-relaxed text-slate-500">
        The curve is the <span className="text-cyan-300">Chinchilla parametric loss</span>{" "}
        <span className="font-mono">L = E + A/N^α + B/D^β</span> evaluated along the budget line{" "}
        <span className="font-mono">D = C/6N</span>. Its floor — marked in{" "}
        <span className="text-emerald-300">green</span> — lands almost exactly on the{" "}
        <span className="font-mono">20:1</span> token-to-parameter ratio, <em>independent of the budget</em>.
        Slide either control: the bottom of the U glides along, but its <em>shape</em> never changes. That is
        a scaling law you can plan a data-center around.
      </p>
    </div>
  );
}
