import { useMemo, useState } from "react";
import { fmt, heat, clamp } from "../scales";
import { Slider, Stat } from "../ui";
import LineChart from "../LineChart";

/**
 * Cross-entropy, felt as a slider.
 *
 * A tiny vocabulary of 6 words. One of them ("sat") is the TRUE next token.
 * The user drags a single knob: the probability mass p the model assigns to the
 * correct token. The remaining (1 - p) is spread over the other five words in a
 * fixed, deterministic ratio, so the whole thing is reproducible and the bars
 * always form a valid probability distribution.
 *
 * The loss for one example with a one-hot target is simply  L = -log(p_true).
 * We display it in nats, alongside perplexity = exp(L) = 1/p — the "effective
 * branching factor". The -log curve is drawn with the shared LineChart and a
 * live marker tracks the current p, so the explosion as p -> 0 is visceral.
 */

const VOCAB = ["the", "cat", "sat", "on", "mat", "ran"];
const TRUE_INDEX = 2; // "sat"

// Fixed, deterministic weights for how the leftover mass (1 - p) is divided
// among the five *incorrect* tokens. Order matches VOCAB minus the true token.
const WRONG_WEIGHTS = [0.30, 0.24, 0.20, 0.16, 0.10];

function distributionFor(p) {
  const leftover = 1 - p;
  const probs = new Array(VOCAB.length).fill(0);
  let w = 0;
  for (let i = 0; i < VOCAB.length; i++) {
    if (i === TRUE_INDEX) continue;
    probs[i] = WRONG_WEIGHTS[w] * leftover;
    w++;
  }
  probs[TRUE_INDEX] = p;
  return probs;
}

export default function CrossEntropyViz() {
  const [p, setP] = useState(0.6);

  const probs = useMemo(() => distributionFor(p), [p]);
  const loss = -Math.log(p); // nats, natural log
  const lossBits = -Math.log2(p); // bits
  const ppl = Math.exp(loss); // = 1 / p

  // The -log(p) curve, sampled densely so the asymptote reads cleanly.
  const curve = useMemo(() => {
    const pts = [];
    for (let i = 1; i <= 100; i++) {
      const x = i / 100; // p from 0.01 .. 1.00
      pts.push([x, -Math.log(x)]);
    }
    return pts;
  }, []);

  // A short verdict that names the four corners of the confidence/correctness grid.
  const verdict =
    p >= 0.85
      ? { text: "Confident & correct — tiny loss.", color: "#34d399" }
      : p >= 0.4
        ? { text: "Hedging — moderate loss.", color: "#f59e0b" }
        : p >= 0.08
          ? { text: "Doubtful about the truth — loss climbing.", color: "#fb7185" }
          : { text: "Confident & wrong — catastrophic loss.", color: "#fb7185" };

  return (
    <div className="space-y-6">
      {/* Predicted distribution over the vocabulary */}
      <div>
        <div className="mb-2 flex items-center justify-between text-xs uppercase tracking-wider text-slate-500">
          <span>Predicted P(next token)</span>
          <span>
            true token ={" "}
            <span className="font-mono text-emerald-300">“{VOCAB[TRUE_INDEX]}”</span>
          </span>
        </div>
        <div className="space-y-1.5">
          {VOCAB.map((word, i) => {
            const isTrue = i === TRUE_INDEX;
            const pct = probs[i] * 100;
            return (
              <div key={word} className="flex items-center gap-3">
                <span
                  className={`w-12 shrink-0 text-right font-mono text-sm ${
                    isTrue ? "text-emerald-300" : "text-slate-400"
                  }`}
                >
                  {word}
                </span>
                <div className="relative h-7 flex-1 overflow-hidden rounded-md bg-ink-800/70">
                  <div
                    className="flex h-full items-center justify-end rounded-md px-2 transition-all duration-200"
                    style={{
                      width: `${Math.max(1.5, pct)}%`,
                      background: isTrue ? "#34d399" : heat(0.2 + probs[i] * 0.6),
                    }}
                  >
                    <span
                      className={`font-mono text-[11px] font-semibold ${
                        isTrue ? "text-ink-950" : "text-ink-950"
                      }`}
                    >
                      {pct.toFixed(1)}%
                    </span>
                  </div>
                  {isTrue && (
                    <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-bold uppercase tracking-wide text-ink-950/70">
                      target
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* The knob */}
      <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
        <Slider
          label="Probability on the correct token  p(true)"
          min={0.01}
          max={0.99}
          step={0.01}
          value={p}
          onChange={setP}
          accent="cyan"
          format={(v) => v.toFixed(2)}
        />
        <p className="mt-2 text-xs font-medium" style={{ color: verdict.color }}>
          {verdict.text}
        </p>
      </div>

      {/* Live readouts */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="p(true)" value={fmt(p, 3)} accent="text-cyan-200" />
        <Stat label="Loss −log p (nats)" value={fmt(loss, 3)} accent="text-rose-200" />
        <Stat label="Loss (bits)" value={fmt(lossBits, 3)} accent="text-amber-200" />
        <Stat label="Perplexity eᴸ = 1/p" value={fmt(ppl, 2)} accent="text-violet-200" />
      </div>

      {/* The -log curve with a live marker */}
      <div>
        <div className="mb-2 text-xs uppercase tracking-wider text-slate-500">
          The penalty curve  L = −log p
        </div>
        <LineChart
          height={260}
          xLabel="p assigned to the true token"
          yLabel="loss (nats)"
          series={[
            { label: "−log p", color: "#fb7185", points: curve },
            // a tiny two-point "series" used purely to drop a marker dot at current p
            {
              label: "you are here",
              color: "#22d3ee",
              points: [
                [clamp(p, 0.01, 0.99), loss],
                [clamp(p, 0.01, 0.99), loss],
              ],
            },
          ]}
          annotations={[{ x: clamp(p, 0.01, 0.99), label: `p=${p.toFixed(2)}`, color: "#22d3ee" }]}
          fmtX={(v) => v.toFixed(2)}
          fmtY={(v) => v.toFixed(1)}
        />
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        Cross-entropy for one example with a one-hot target collapses to{" "}
        <span className="font-mono text-slate-400">−log p(true)</span>. Notice the asymmetry: pushing{" "}
        <span className="font-mono">p</span> from 0.9 → 1.0 barely moves the loss, but letting it
        slip from 0.1 → 0.01 multiplies the penalty. Being <em>confidently wrong</em> about the truth
        is punished without mercy — that gradient is what teaches the model humility.
      </p>
    </div>
  );
}
