import { useMemo, useState } from "react";
import { SegmentedControl, Toggle, Stat } from "../ui";
import { fmt } from "../scales";
import Heatmap from "../Heatmap";

/**
 * BenchmarkBoard — an interactive benchmark comparison.
 *
 * A small set of clearly-fictional-but-realistic models scored across several
 * benchmarks. Two views, both deterministic (no randomness, fixed data):
 *
 *  1) "Leaderboard" — grouped horizontal bars. The user picks which benchmark
 *     to SORT by (SegmentedControl). The whole point: the #1 model changes
 *     depending on which column you rank on. We label the live rank and show
 *     how each model's rank *moved* relative to ranking by the overall average.
 *
 *  2) "Heatmap" — the same scores as a coloured matrix (reusing the shared
 *     Heatmap), so you can see at a glance that no single model dominates every
 *     column. A toggle highlights the sorted column.
 *
 * Plus a small LLM-as-judge pairwise panel that shows the SAME pair of answers
 * scored two ways — and how swapping their order flips the verdict (position
 * bias), illustrating why judge protocols randomize order.
 *
 * All numbers are hand-authored and illustrative. They are NOT real model
 * scores; the models are invented. The pedagogy is "no single number is good".
 */

// ----- fictional models and their (illustrative) benchmark scores ----------
// Columns are percentages 0–100 except Arena which is an Elo-style rating.
// Designed so the ranking genuinely reorders across columns.
const BENCHMARKS = [
  { key: "mmlu", label: "MMLU", unit: "%", blurb: "knowledge & reasoning, 57 subjects" },
  { key: "gsm8k", label: "GSM8K", unit: "%", blurb: "grade-school math word problems" },
  { key: "humaneval", label: "HumanEval", unit: "%", blurb: "Python code, pass@1" },
  { key: "hellaswag", label: "HellaSwag", unit: "%", blurb: "commonsense sentence completion" },
  { key: "truthfulqa", label: "TruthfulQA", unit: "%", blurb: "resisting popular falsehoods" },
  { key: "arena", label: "Arena Elo", unit: "", blurb: "human pairwise chat preference" },
];

const MODELS = [
  // A well-rounded generalist: strong everywhere, best on chat, not the top coder.
  { name: "Aria-13B", color: "#5b7dff", scores: { mmlu: 71.4, gsm8k: 68.0, humaneval: 49.2, hellaswag: 84.6, truthfulqa: 52.8, arena: 1187 } },
  // A code specialist: dominates HumanEval, mediocre on truthfulness & chat.
  { name: "Cobalt-Coder", color: "#22d3ee", scores: { mmlu: 66.1, gsm8k: 71.5, humaneval: 73.8, hellaswag: 79.2, truthfulqa: 41.0, arena: 1124 } },
  // A math/reasoning model: best GSM8K, strong MMLU, weak code.
  { name: "Pythia-Reason", color: "#a855f7", scores: { mmlu: 73.9, gsm8k: 79.3, humaneval: 38.5, hellaswag: 81.0, truthfulqa: 48.6, arena: 1142 } },
  // The "truthful & safe" model: tops TruthfulQA, average raw capability.
  { name: "Verum-Safe", color: "#34d399", scores: { mmlu: 64.8, gsm8k: 55.2, humaneval: 34.1, hellaswag: 78.4, truthfulqa: 63.5, arena: 1109 } },
  // A small fast model: nothing top, decent commonsense — the "cheap baseline".
  { name: "Lumen-3B", color: "#f59e0b", scores: { mmlu: 58.3, gsm8k: 47.6, humaneval: 27.9, hellaswag: 80.7, truthfulqa: 44.2, arena: 1063 } },
];

// Per-column min/max for bar normalization (so bars use the column's range,
// which makes the within-column differences readable rather than absolute).
function colRange(key) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const m of MODELS) {
    const v = m.scores[key];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}

// Average rank across all benchmarks → a single "overall" ordering we compare
// against, to show how a per-benchmark sort shuffles the table.
const overallOrder = (() => {
  // rank within each column (1 = best), then average the ranks per model.
  const avgRank = MODELS.map((m) => {
    let sum = 0;
    for (const b of BENCHMARKS) {
      const better = MODELS.filter((o) => o.scores[b.key] > m.scores[b.key]).length;
      sum += better + 1;
    }
    return { name: m.name, avg: sum / BENCHMARKS.length };
  });
  avgRank.sort((a, b) => a.avg - b.avg);
  return avgRank.map((r) => r.name);
})();

function LeaderboardView({ sortKey }) {
  const [lo, hi] = colRange(sortKey);
  const range = hi - lo || 1;

  const sorted = useMemo(
    () => [...MODELS].sort((a, b) => b.scores[sortKey] - a.scores[sortKey]),
    [sortKey]
  );

  const meta = BENCHMARKS.find((b) => b.key === sortKey);
  const topModel = sorted[0];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Stat label={`sorting by ${meta.label}`} value={meta.label} accent="text-brand-200" />
        <Stat
          label="this benchmark's #1"
          value={topModel.name}
          accent="text-amber-200"
        />
      </div>

      <div className="space-y-2.5">
        {sorted.map((m, i) => {
          const v = m.scores[sortKey];
          const frac = (v - lo) / range;
          const overallRank = overallOrder.indexOf(m.name) + 1;
          const liveRank = i + 1;
          const delta = overallRank - liveRank; // + means it climbed vs overall
          return (
            <div key={m.name} className="flex items-center gap-3">
              <span className="w-6 shrink-0 text-right font-mono text-sm text-slate-500">
                {liveRank}
              </span>
              <span
                className="w-28 shrink-0 truncate text-sm font-medium"
                style={{ color: m.color }}
                title={m.name}
              >
                {m.name}
              </span>
              <div className="relative h-7 flex-1 overflow-hidden rounded-md bg-ink-900/60">
                <div
                  className="h-full rounded-md transition-all duration-500"
                  style={{
                    width: `${Math.max(6, 12 + frac * 88)}%`,
                    background: `${m.color}cc`,
                  }}
                />
                <span className="absolute inset-y-0 right-2 flex items-center font-mono text-xs text-slate-200">
                  {meta.unit === "%" ? `${fmt(v, 1)}%` : fmt(v, 0)}
                </span>
              </div>
              <span
                className="w-16 shrink-0 text-right font-mono text-[11px]"
                style={{
                  color: delta > 0 ? "#34d399" : delta < 0 ? "#fb7185" : "#64748b",
                }}
                title="rank change vs. ranking by overall average"
              >
                {delta === 0 ? "—" : delta > 0 ? `▲${delta}` : `▼${-delta}`}
              </span>
            </div>
          );
        })}
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        Bars are scaled to this column's range so differences are visible. The right column shows how
        each model's rank <em>moved</em> versus ranking by the overall average (
        <span className="text-emerald-300">▲ climbed</span>,{" "}
        <span className="text-rose-300">▼ dropped</span>). Switch the benchmark and watch the order
        reshuffle — <strong>{meta.label}</strong> rewards {meta.blurb}.
      </p>
    </div>
  );
}

function HeatmapView({ sortKey, highlight }) {
  // Normalize every column to 0–1 so colours are comparable across very
  // different scales (percent vs. Elo). We show the *raw* value as the label.
  const colIndex = BENCHMARKS.findIndex((b) => b.key === sortKey);

  const sortedModels = useMemo(
    () => [...MODELS].sort((a, b) => b.scores[sortKey] - a.scores[sortKey]),
    [sortKey]
  );

  const ranges = BENCHMARKS.map((b) => colRange(b.key));

  // Build a per-column-normalized matrix in [0,1] for colour, plus a parallel
  // matrix of display strings.
  const norm = sortedModels.map((m) =>
    BENCHMARKS.map((b, c) => {
      const [lo, hi] = ranges[c];
      return (m.scores[b.key] - lo) / (hi - lo || 1);
    })
  );

  const labels = sortedModels.map((m) =>
    BENCHMARKS.map((b) => (b.unit === "%" ? `${Math.round(m.scores[b.key])}` : `${m.scores[b.key]}`))
  );

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto scrollbar-thin">
        {/* We render our own labelled grid so cells can show the RAW score
            while colour encodes the per-column-normalized value. */}
        <HeatGrid
          norm={norm}
          labels={labels}
          rowLabels={sortedModels.map((m) => m.name)}
          rowColors={sortedModels.map((m) => m.color)}
          colLabels={BENCHMARKS.map((b) => b.label)}
          highlightCol={highlight ? colIndex : -1}
        />
      </div>
      <p className="text-xs leading-relaxed text-slate-500">
        Colour encodes each column's <em>relative</em> standing (best in column = brightest), because
        the columns live on different scales. Read down a column: every benchmark has a different
        bright cell. There is no row that is bright all the way across — which is exactly the point.
      </p>
    </div>
  );
}

// A small bespoke grid (not the shared Heatmap) so we can colour by a
// normalized value but print the raw score, and tint row labels per model.
function HeatGrid({ norm, labels, rowLabels, rowColors, colLabels, highlightCol }) {
  const cell = 64;
  const gap = 4;
  const heatCell = (t) => {
    // ink → brand → cyan ramp; mirror the site's `heat` feel but emphasize mid.
    const r = Math.round(20 + t * (34 - 20) + Math.max(0, t - 0.5) * 2 * (233 - 34));
    const g = Math.round(28 + t * (211 - 28));
    const b = Math.round(54 + t * (238 - 54));
    return `rgb(${r}, ${g}, ${b})`;
  };
  return (
    <div className="inline-block">
      {/* column header */}
      <div className="flex" style={{ gap, marginLeft: 96 }}>
        {colLabels.map((l, c) => (
          <div
            key={c}
            className={`flex items-end justify-center pb-1 text-[11px] font-medium ${
              c === highlightCol ? "text-brand-200" : "text-slate-500"
            }`}
            style={{ width: cell }}
          >
            {l}
          </div>
        ))}
      </div>
      {norm.map((row, r) => (
        <div key={r} className="flex items-center" style={{ gap, marginBottom: gap }}>
          <div
            className="pr-2 text-right text-[12px] font-medium"
            style={{ width: 96, color: rowColors[r] }}
            title={rowLabels[r]}
          >
            <span className="block truncate">{rowLabels[r]}</span>
          </div>
          {row.map((t, c) => (
            <div
              key={c}
              className="flex items-center justify-center rounded-md transition-all duration-500"
              style={{
                width: cell,
                height: 40,
                background: heatCell(t),
                outline: c === highlightCol ? "2px solid #e9faff" : "none",
                opacity: highlightCol < 0 || c === highlightCol ? 1 : 0.55,
              }}
            >
              <span
                className="font-mono text-[12px] font-semibold"
                style={{ color: t > 0.55 ? "#0b0e1a" : "#cbd5e1" }}
              >
                {labels[r][c]}
              </span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// ----- LLM-as-judge pairwise panel -----------------------------------------
// Two fixed candidate answers to one prompt. A "judge" assigns a quality, but a
// naive judge also adds a length bonus and a position bonus to whichever answer
// is shown FIRST. We let the user (a) swap the presentation order and (b) toggle
// the debiasing (averaging both orders). With bias on, order decides the winner.

const JUDGE_PROMPT = "Explain why the sky is blue, briefly.";
const ANSWER_A = {
  id: "A",
  who: "Model-Concise",
  text: "Sunlight scatters off air molecules; shorter (blue) wavelengths scatter most, so the sky looks blue.",
  // hidden ground-truth quality the judge is *trying* to estimate
  quality: 0.82,
  words: 18,
};
const ANSWER_B = {
  id: "B",
  who: "Model-Verbose",
  text: "That is a wonderful question! The sky's blue colour arises from a phenomenon known as Rayleigh scattering, in which the shorter-wavelength blue light from the sun is scattered far more strongly by the tiny molecules of the atmosphere than the longer red wavelengths, and so on...",
  quality: 0.74, // actually slightly worse content, but much longer
  words: 56,
};

function judgeScore(ans, isFirst, { positionBias, verbosityBias }) {
  let s = ans.quality;
  if (verbosityBias) s += 0.0018 * ans.words; // longer ⇒ a few free points
  if (positionBias && isFirst) s += 0.06; // the first answer gets a bump
  return s;
}

function JudgePanel() {
  const [firstId, setFirstId] = useState("A"); // which answer is shown first
  const [debias, setDebias] = useState(false); // average both orders + drop biases

  const biases = { positionBias: !debias, verbosityBias: !debias };

  // Score in the presented order.
  const aFirst = firstId === "A";
  const sA = judgeScore(ANSWER_A, aFirst, biases);
  const sB = judgeScore(ANSWER_B, !aFirst, biases);

  // If debiasing, also evaluate the swapped order and average (and biases are
  // off anyway, so order no longer matters — we show the stable verdict).
  let finalA = sA;
  let finalB = sB;
  if (debias) {
    const sA2 = judgeScore(ANSWER_A, false, biases);
    const sB2 = judgeScore(ANSWER_B, true, biases);
    finalA = (sA + sA2) / 2;
    finalB = (sB + sB2) / 2;
  }

  const winner = finalA === finalB ? "tie" : finalA > finalB ? "A" : "B";

  const card = (ans, score, isFirst) => {
    const win = winner === ans.id;
    return (
      <div
        key={ans.id}
        className={`flex-1 rounded-xl border p-3 transition-colors ${
          win ? "border-amber-400/60 bg-amber-400/[0.06]" : "border-white/10 bg-ink-900/50"
        }`}
      >
        <div className="mb-1 flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300">
            {ans.who}{" "}
            <span className="font-mono text-[10px] text-slate-500">
              ({ans.words} words{isFirst ? ", shown first" : ", shown second"})
            </span>
          </span>
          {win && <span className="text-[10px] font-semibold text-amber-300">JUDGE PICKS ✓</span>}
        </div>
        <p className="mb-2 text-[12px] leading-relaxed text-slate-400">{ans.text}</p>
        <div className="flex items-center gap-2">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${score * 100}%`, background: win ? "#f59e0b" : "#5b7dff" }}
            />
          </div>
          <span className="font-mono text-[11px] text-slate-400">{fmt(score, 3)}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-white/5 bg-ink-900/40 px-3 py-2 text-sm text-slate-400">
        <span className="text-slate-500">prompt:</span>{" "}
        <span className="text-slate-200">{JUDGE_PROMPT}</span>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row">
        {aFirst
          ? [card(ANSWER_A, finalA, true), card(ANSWER_B, finalB, false)]
          : [card(ANSWER_B, finalB, true), card(ANSWER_A, finalA, false)]}
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <SegmentedControl
          label="presentation order (which answer is shown first)"
          value={firstId}
          onChange={setFirstId}
          options={[
            { value: "A", label: "Concise first" },
            { value: "B", label: "Verbose first" },
          ]}
        />
        <Toggle
          label="debias judge (randomize order + drop length/position bonuses)"
          checked={debias}
          onChange={setDebias}
        />
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        {debias ? (
          <>
            With debiasing on, the judge scores both orderings and averages, and ignores raw length.
            The verdict is now <strong>stable</strong>: whichever answer is shown first, the same one
            wins. This is the protocol real LLM-judge harnesses use.
          </>
        ) : (
          <>
            This naive judge quietly rewards the answer shown <em>first</em> (+0.06) and longer answers
            (+~0.002/word). Flip the presentation order above and watch the winner change — the{" "}
            <em>content</em> never moved. That is <strong>position bias</strong> and{" "}
            <strong>verbosity bias</strong>, and it is why you must never trust a single-order judge.
          </>
        )}
      </p>
    </div>
  );
}

// ----- wrapper -------------------------------------------------------------

export default function BenchmarkBoard() {
  const [view, setView] = useState("leaderboard");
  const [sortKey, setSortKey] = useState("mmlu");
  const [highlight, setHighlight] = useState(true);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SegmentedControl
          value={view}
          onChange={setView}
          options={[
            { value: "leaderboard", label: "Leaderboard" },
            { value: "heatmap", label: "Score matrix" },
            { value: "judge", label: "LLM-as-judge" },
          ]}
        />
      </div>

      {view !== "judge" && (
        <div className="flex flex-wrap items-end justify-between gap-3">
          <SegmentedControl
            label="rank / sort by benchmark"
            value={sortKey}
            onChange={setSortKey}
            options={BENCHMARKS.map((b) => ({ value: b.key, label: b.label }))}
          />
          {view === "heatmap" && (
            <Toggle label="highlight sorted column" checked={highlight} onChange={setHighlight} />
          )}
        </div>
      )}

      {view === "leaderboard" && <LeaderboardView sortKey={sortKey} />}
      {view === "heatmap" && <HeatmapView sortKey={sortKey} highlight={highlight} />}
      {view === "judge" && <JudgePanel />}

      <p className="text-[11px] leading-relaxed text-slate-600">
        Models and scores here are <strong>invented for illustration</strong> — they are not real
        systems or real benchmark results. The lesson is structural: ranking depends entirely on the
        metric you choose.
      </p>
    </div>
  );
}
