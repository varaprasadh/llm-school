import { useMemo, useState } from "react";
import { Toggle, Button, Stat } from "../ui";
import { heat } from "../scales";

/**
 * KVCacheViz — steps through autoregressive decoding and makes the cost of
 * recomputation (no cache) vs. caching (cache) viscerally obvious.
 *
 * The model has already processed a short prompt of PROMPT tokens. We then
 * generate one token at a time. At every decode step the model produces a NEW
 * query that must attend over ALL tokens generated so far.
 *
 *  • WITHOUT a KV cache: to compute attention for the new token, the model
 *    re-projects keys & values for EVERY previous token (they were thrown away
 *    last step), so it recomputes the whole growing prefix. The number of
 *    key/value projections at step t is the full length so far. Summed over the
 *    sequence this is the quadratic ½n(n+1) — wasted work, because those K/V are
 *    identical to last step's.
 *
 *  • WITH a KV cache: past keys & values are kept in memory. The new token
 *    projects its OWN key & value once (1 unit), appends them to the cache, and
 *    its single query attends over the cached K/V. Work per step is constant ⇒
 *    O(n) total.
 *
 * Everything is deterministic: the grid is a pure function of the step index.
 * No randomness, no clocks.
 */

const PROMPT = 4; // tokens in the prompt (the "prefill")
const GEN = 8; // tokens we will generate, one per decode step
const TOTAL = PROMPT + GEN; // total sequence length at the end

// Short illustrative token strings so the columns read like real text.
const TOKENS = [
  "The", "cat", "sat", "on", "the", "mat", "and", "it", "purred", "soft", "-ly", ".",
];

const W = 720;
const CELL = 42; // px per token column
const GAP = 5;
const LEFT = 92; // gutter for the row label
const TOP = 30; // gutter for the column labels

export default function KVCacheViz() {
  const [cache, setCache] = useState(true);
  // step = how many tokens have been GENERATED (0..GEN). step 0 = prompt only.
  const [step, setStep] = useState(GEN);

  // Length of the sequence currently "live" = prompt + tokens generated so far.
  const liveLen = PROMPT + step;
  // Index of the brand-new token's column (the one decoded at this step), or -1
  // when we are still showing just the prompt.
  const newCol = step === 0 ? -1 : liveLen - 1;

  // ── Cost accounting ───────────────────────────────────────────────────────
  // We count "key/value projections" — the unit of redundant work the cache
  // eliminates. Prefill (processing the prompt) costs PROMPT for both modes; we
  // attribute only the *decode* cost to the comparison so the contrast is clean.
  const { noCacheTotal, cacheTotal, noCacheThisStep, cacheThisStep } = useMemo(() => {
    let nc = 0; // cumulative K/V projections, no cache
    let c = 0; // cumulative K/V projections, with cache
    let ncStep = 0;
    let cStep = 0;
    for (let s = 1; s <= step; s++) {
      const lenAtStep = PROMPT + s; // sequence length when generating token s
      // No cache: re-project K/V for the entire prefix every step.
      nc += lenAtStep;
      // Cache: project only the one new token's K/V.
      c += 1;
      if (s === step) {
        ncStep = lenAtStep;
        cStep = 1;
      }
    }
    return { noCacheTotal: nc, cacheTotal: c, noCacheThisStep: ncStep, cacheThisStep: cStep };
  }, [step]);

  const savings = noCacheTotal === 0 ? 0 : 1 - cacheTotal / noCacheTotal;

  const svgH = TOP + 2 * (CELL + GAP) + 26;

  // Classify every cell in the two K/V rows for the CURRENT step.
  //  - "new":        the brand-new token (always computed, both modes)
  //  - "recompute":  re-projected this step in NO-cache mode (wasted)
  //  - "cached":     already in memory in cache mode (reused, free)
  //  - "future":     not yet generated (inactive)
  function cellState(col) {
    if (col >= liveLen) return "future";
    if (col === newCol) return "new";
    if (col < PROMPT) return cache ? "cached" : "recompute-prompt";
    return cache ? "cached" : "recompute";
  }

  const COLORS = {
    new: "#f59e0b", // amber — freshly computed
    recompute: "#fb7185", // rose — wasted recomputation (generated tokens)
    "recompute-prompt": "#fb7185", // rose — wasted recomputation (prompt tokens)
    cached: "#34d399", // emerald — reused from cache (free)
    future: "#10182a", // inert
  };

  function Row({ y, label }) {
    return (
      <g>
        <text
          x={LEFT - 10}
          y={y + CELL / 2 + 4}
          textAnchor="end"
          fontSize="11"
          fill="#94a3b8"
          fontFamily="JetBrains Mono, monospace"
        >
          {label}
        </text>
        {Array.from({ length: TOTAL }, (_, col) => {
          const st = cellState(col);
          const x = LEFT + col * (CELL + GAP);
          const fill = COLORS[st] || COLORS.future;
          const isProcessed = st !== "future";
          return (
            <g key={col}>
              <rect
                x={x}
                y={y}
                width={CELL}
                height={CELL}
                rx={6}
                fill={fill}
                opacity={st === "future" ? 0.5 : st === "cached" ? 0.85 : 1}
                stroke={st === "new" ? "#fff" : "rgba(255,255,255,0.08)"}
                strokeWidth={st === "new" ? 2 : 1}
                className="transition-all duration-300"
              />
              {isProcessed && (st === "recompute" || st === "recompute-prompt") && (
                // little "redo" glyph to scream "wasted work"
                <text
                  x={x + CELL / 2}
                  y={y + CELL / 2 + 4}
                  textAnchor="middle"
                  fontSize="13"
                  fill="#3b0d17"
                  fontFamily="JetBrains Mono, monospace"
                >
                  ↻
                </text>
              )}
              {isProcessed && st === "cached" && (
                <text
                  x={x + CELL / 2}
                  y={y + CELL / 2 + 4}
                  textAnchor="middle"
                  fontSize="12"
                  fill="#073b2a"
                  fontFamily="JetBrains Mono, monospace"
                >
                  ✓
                </text>
              )}
              {isProcessed && st === "new" && (
                <text
                  x={x + CELL / 2}
                  y={y + CELL / 2 + 4}
                  textAnchor="middle"
                  fontSize="12"
                  fill="#3b2a07"
                  fontFamily="JetBrains Mono, monospace"
                >
                  ✦
                </text>
              )}
            </g>
          );
        })}
      </g>
    );
  }

  return (
    <div className="space-y-5">
      {/* controls */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <Toggle label="KV cache" checked={cache} onChange={setCache} />
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => setStep((s) => Math.min(GEN, s + 1))} disabled={step >= GEN}>
            Decode next ▶
          </Button>
          <Button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step <= 0}>
            ◀ Back
          </Button>
          <Button onClick={() => setStep(0)}>↻ Reset</Button>
        </div>
        <span className="text-xs text-slate-400">
          {step === 0
            ? `prompt of ${PROMPT} tokens loaded — press “Decode next”`
            : `decode step ${step}/${GEN}: generating token #${liveLen}`}
        </span>
      </div>

      {/* the grid */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="overflow-x-auto scrollbar-thin">
          <svg width={W} height={svgH} className="block min-w-[640px]">
            {/* column / token labels */}
            {TOKENS.slice(0, TOTAL).map((tok, col) => {
              const x = LEFT + col * (CELL + GAP);
              const inPrompt = col < PROMPT;
              const live = col < liveLen;
              return (
                <text
                  key={col}
                  x={x + CELL / 2}
                  y={TOP - 12}
                  textAnchor="middle"
                  fontSize="11"
                  fill={live ? (inPrompt ? "#7c93ff" : "#cbd5e1") : "#475569"}
                  fontFamily="JetBrains Mono, monospace"
                >
                  {tok}
                </text>
              );
            })}
            {/* a divider between prompt (prefill) and generated tokens */}
            <line
              x1={LEFT + PROMPT * (CELL + GAP) - GAP / 2}
              y1={TOP - 24}
              x2={LEFT + PROMPT * (CELL + GAP) - GAP / 2}
              y2={TOP + 2 * (CELL + GAP)}
              stroke="#5b7dff"
              strokeWidth={1}
              strokeDasharray="3 3"
              opacity={0.5}
            />
            <text
              x={LEFT + (PROMPT * (CELL + GAP)) / 2}
              y={svgH - 6}
              textAnchor="middle"
              fontSize="10"
              fill="#7c93ff"
              fontFamily="JetBrains Mono, monospace"
            >
              ← prompt (prefill)
            </text>
            <text
              x={LEFT + PROMPT * (CELL + GAP) + (GEN * (CELL + GAP)) / 2}
              y={svgH - 6}
              textAnchor="middle"
              fontSize="10"
              fill="#94a3b8"
              fontFamily="JetBrains Mono, monospace"
            >
              generated, one per step →
            </text>

            <Row y={TOP} label="K[layer]" />
            <Row y={TOP + CELL + GAP} label="V[layer]" />
          </svg>
        </div>

        {/* legend */}
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: COLORS.new }} />
            new token (computed once)
          </span>
          {cache ? (
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-3 w-3 rounded-sm"
                style={{ background: COLORS.cached }}
              />
              read from cache (free)
            </span>
          ) : (
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-3 w-3 rounded-sm"
                style={{ background: COLORS.recompute }}
              />
              recomputed this step (wasted)
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ background: COLORS.future, opacity: 0.5 }}
            />
            not yet generated
          </span>
        </div>
      </div>

      {/* running counters */}
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/[0.05] p-3">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-rose-200">
            Without KV cache
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Stat label="this step" value={`${noCacheThisStep}`} accent="text-rose-200" />
            <Stat label="total K/V projections" value={`${noCacheTotal}`} accent="text-rose-200" />
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
            Re-projects K/V for the entire prefix every step → grows each step. Quadratic overall.
          </p>
        </div>
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/[0.05] p-3">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-emerald-200">
            With KV cache
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Stat label="this step" value={`${cacheThisStep}`} accent="text-emerald-200" />
            <Stat label="total K/V projections" value={`${cacheTotal}`} accent="text-emerald-200" />
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
            Projects only the one new token; reuses cached K/V. Constant per step → linear overall.
          </p>
        </div>
      </div>

      <div
        className="rounded-lg px-3 py-2 text-center text-sm"
        style={{ background: heat(0.25, 0.5), color: "#e9faff" }}
      >
        After {step} decode step{step === 1 ? "" : "s"}, the cache has done{" "}
        <span className="font-mono text-emerald-200">{cacheTotal}</span> projections vs.{" "}
        <span className="font-mono text-rose-200">{noCacheTotal}</span> — a{" "}
        <span className="font-mono text-amber-200">{Math.round(savings * 100)}%</span> reduction in
        redundant work, and it widens every single step.
      </div>
    </div>
  );
}
