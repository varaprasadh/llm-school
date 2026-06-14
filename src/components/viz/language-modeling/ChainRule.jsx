import { useMemo, useState } from "react";
import { heat, fmt } from "../scales";
import { SegmentedControl } from "../ui";

/**
 * ChainRule — visualizes the autoregressive factorization
 *   P(w1, ..., wT) = ∏ P(w_t | w_<t)
 * over a fixed short sentence. Each factor is a hoverable chip; hovering token t
 * highlights its conditioning context (the tokens it is allowed to look at) and
 * shows an illustrative per-token probability plus the running sequence
 * probability and the running sum of −log p (the cross-entropy total).
 *
 * A SegmentedControl switches the Markov order between bigram (n=2, condition on
 * the single previous token), trigram (n=3, the previous two), and full history
 * (condition on everything before) — visually shrinking/growing the highlighted
 * context window. All probabilities are FIXED illustrative numbers so the figure
 * is deterministic across renders.
 */

const TOKENS = ["the", "cat", "sat", "on", "the", "mat"];

// Illustrative P(w_t | w_<t) under each Markov order. Index t aligns with TOKENS.
// Numbers are hand-chosen to tell a story: more context generally sharpens the
// distribution, so the full-history probabilities are mostly >= the bigram ones.
// (These are pedagogical, not measured — a real model would estimate them.)
const PROBS = {
  // n = 2: condition on the previous token only
  2: [0.07, 0.18, 0.31, 0.62, 0.55, 0.12],
  // n = 3: condition on the previous two tokens
  3: [0.07, 0.21, 0.44, 0.66, 0.58, 0.34],
  // full history: condition on all preceding tokens
  full: [0.07, 0.24, 0.52, 0.71, 0.6, 0.58],
};

const ORDER_OPTIONS = [
  { value: "2", label: "Bigram (n=2)" },
  { value: "3", label: "Trigram (n=3)" },
  { value: "full", label: "Full history" },
];

// How many preceding tokens position t may condition on, given the order.
function contextSpan(order, t) {
  if (t === 0) return 0; // P(w1) has no context
  if (order === "full") return t; // all preceding tokens
  const n = parseInt(order, 10); // bigram -> 1 previous, trigram -> 2 previous
  return Math.min(n - 1, t);
}

// Display-ready label for the conditioning set of position t (1-based, x-notation),
// e.g. "x1" or "x2x3" or "x<5". Returns null when there is no context (the first token).
function condLabel(order, t) {
  const span = contextSpan(order, t);
  if (span === 0) return null;
  const first = t - span + 1; // 1-based position of first conditioning token
  const last = t; // 1-based position of last conditioning token (= previous token)
  if (order === "full") return `x<${t + 1}`;
  if (span === 1) return `x${last}`;
  return `x${first}…x${last}`;
}

export default function ChainRule() {
  const [order, setOrder] = useState("full");
  const [hover, setHover] = useState(null); // hovered token index, or null

  const probs = PROBS[order];

  // Running products / sums up to and including each position.
  const cum = useMemo(() => {
    let prod = 1;
    let nll = 0;
    return probs.map((p) => {
      prod *= p;
      nll += -Math.log(p);
      return { prod, nll };
    });
  }, [probs]);

  const seqProb = cum[cum.length - 1].prod;
  const seqNll = cum[cum.length - 1].nll;
  const perplexity = Math.exp(seqNll / TOKENS.length);

  // The position whose factor / context we are spotlighting (hover, else last).
  const focus = hover == null ? TOKENS.length - 1 : hover;
  const span = contextSpan(order, focus);
  const ctxStart = focus - span; // first highlighted context index (0-based)

  // Is token index i part of the highlighted conditioning context of `focus`?
  const isContext = (i) => span > 0 && i >= ctxStart && i <= focus - 1;
  const isFocus = (i) => i === focus;

  return (
    <div className="space-y-5">
      {/* Order selector */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SegmentedControl
          label="Conditioning context"
          options={ORDER_OPTIONS}
          value={order}
          onChange={setOrder}
        />
        <p className="max-w-[18rem] text-xs leading-relaxed text-slate-500">
          Hover a token to spotlight the context it conditions on. Switch the order to
          grow or shrink that window.
        </p>
      </div>

      {/* The sentence as token chips */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="mb-3 text-[10px] uppercase tracking-wider text-slate-500">
          Sequence x₁ … x₆
        </div>
        <div className="flex flex-wrap items-end gap-2 font-mono">
          {TOKENS.map((tok, i) => {
            const focused = isFocus(i);
            const ctx = isContext(i);
            return (
              <div key={i} className="flex flex-col items-center gap-1">
                <span className="text-[10px] text-slate-500">
                  w<sub>{i + 1}</sub>
                </span>
                <span
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                  className={`cursor-default rounded-md px-3 py-1.5 text-base transition-all duration-150 ${
                    focused
                      ? "bg-brand-500/40 text-brand-50 ring-2 ring-brand-400"
                      : ctx
                        ? "bg-cyan-500/20 text-cyan-100 ring-1 ring-cyan-400/60"
                        : "bg-ink-700/60 text-slate-300"
                  }`}
                >
                  {tok}
                </span>
              </div>
            );
          })}
        </div>
        <div className="mt-3 flex flex-wrap gap-4 text-[11px] text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm bg-brand-500/60 ring-1 ring-brand-400" />
            predicted token x_t
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm bg-cyan-500/40 ring-1 ring-cyan-400/60" />
            conditioning context x_&lt;t
          </span>
        </div>
      </div>

      {/* The factorization, factor by factor */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="mb-3 text-[10px] uppercase tracking-wider text-slate-500">
          P(x₁ … x₆) = ∏ P(x_t | x_&lt;t)
        </div>
        <div className="flex flex-wrap items-stretch gap-1.5">
          {TOKENS.map((tok, i) => {
            const p = probs[i];
            const cl = condLabel(order, i);
            const focused = isFocus(i);
            return (
              <div key={i} className="flex items-stretch gap-1.5">
                <button
                  type="button"
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                  className={`flex min-w-[6.5rem] flex-col items-center rounded-lg border px-2 py-2 text-center transition-all duration-150 ${
                    focused
                      ? "border-brand-400 bg-brand-500/15"
                      : "border-white/10 bg-ink-800/50 hover:border-white/25"
                  }`}
                >
                  <span className="font-mono text-xs text-slate-200">
                    P(<span className="text-brand-200">{tok}</span>
                    {cl ? (
                      <>
                        {" | "}
                        <span className="text-cyan-200">{cl}</span>
                      </>
                    ) : null}
                    )
                  </span>
                  {/* probability bar */}
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-ink-700/70">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{ width: `${Math.max(4, p * 100)}%`, background: heat(0.3 + p * 0.6) }}
                    />
                  </div>
                  <span className="mt-1 font-mono text-[11px] text-slate-400">
                    {fmt(p, 2)}
                  </span>
                </button>
                {i < TOKENS.length - 1 && (
                  <span className="flex items-center font-mono text-lg text-slate-600">·</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Running tallies up to the focused position */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <RunningCard
          label={`Running product · up to x${sub(focus + 1)}`}
          value={fmt(cum[focus].prod, 5)}
          help="P(x₁…x_t): multiply the factors so far. It shrinks fast — sequences are improbable."
          accent="text-brand-200"
        />
        <RunningCard
          label={`Running −log p · up to x${sub(focus + 1)}`}
          value={`${fmt(cum[focus].nll, 3)} nats`}
          help="Σ −log P(x_t | x_<t): adds instead of multiplies. This is the total surprise / cross-entropy."
          accent="text-cyan-200"
        />
        <RunningCard
          label="Full-sequence perplexity"
          value={fmt(perplexity, 2)}
          help="exp(mean −log p): the model's effective branching factor. Lower is better."
          accent="text-emerald-300"
        />
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        The product over all six factors gives the whole-sequence probability{" "}
        <span className="font-mono text-slate-300">P(x₁…x₆) = {fmt(seqProb, 6)}</span>. Notice
        that more context (trigram → full history) generally raises each factor, lowering the
        total surprise and the perplexity — that is exactly the advantage a neural model buys by
        conditioning on the entire history.
      </p>
    </div>
  );
}

function sub(n) {
  const map = { 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆" };
  return map[n] || n;
}

function RunningCard({ label, value, help, accent }) {
  return (
    <div className="rounded-lg border border-white/10 bg-ink-900/50 p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`mt-0.5 font-mono text-xl ${accent}`}>{value}</div>
      <p className="mt-1 text-[11px] leading-snug text-slate-500">{help}</p>
    </div>
  );
}
