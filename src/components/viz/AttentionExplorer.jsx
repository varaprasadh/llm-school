import { useMemo, useState } from "react";
import { softmax, heat, fmt } from "./scales";
import { SegmentedControl, Toggle } from "./ui";

/**
 * Interactive self-attention explorer.
 *
 * A fixed sentence is tokenized. Each "head" carries a hand-authored matrix of
 * raw compatibility scores (what Q·Kᵀ would produce). The pipeline that turns
 * those scores into attention weights — scale by 1/√d_k, apply the causal mask,
 * softmax — runs live so you can watch each step and toggle masking on/off.
 */

const TOKENS = ["The", "cat", "sat", "on", "the", "mat", "because", "it"];
const D_K = 16; // pretend head dimension, for the 1/√d_k scaling

// Raw scores[query][key] per head. Hand-tuned to be interpretable.
const HEADS = {
  "Previous token": rawPrev(),
  "First / BOS": rawFirst(),
  "Coreference": rawCoref(),
};

function rawPrev() {
  // each token most compatible with the one just before it
  const n = TOKENS.length;
  const m = Array.from({ length: n }, () => Array(n).fill(0.2));
  for (let i = 0; i < n; i++) {
    if (i - 1 >= 0) m[i][i - 1] = 6;
    m[i][i] = 1.5;
  }
  return m;
}
function rawFirst() {
  const n = TOKENS.length;
  const m = Array.from({ length: n }, () => Array(n).fill(0.3));
  for (let i = 0; i < n; i++) {
    m[i][0] = 5.5;
    m[i][i] = 1.2;
  }
  return m;
}
function rawCoref() {
  const n = TOKENS.length;
  const m = Array.from({ length: n }, () => Array(n).fill(0.4));
  for (let i = 0; i < n; i++) m[i][i] = 1.5;
  // "it" (7) strongly attends to "cat" (1) and "mat" (5)
  m[7][1] = 6.5;
  m[7][5] = 4.5;
  m[7][7] = 1;
  // "because" (6) links the clause back to "sat" (2)
  m[6][2] = 4.5;
  // "sat" (2) attends to subject "cat" (1)
  m[2][1] = 4.0;
  // "mat" (5) attends to "sat"/"on"
  m[5][2] = 3.0;
  m[5][3] = 2.5;
  return m;
}

export default function AttentionExplorer() {
  const [headName, setHeadName] = useState("Coreference");
  const [causal, setCausal] = useState(true);
  const [query, setQuery] = useState(7); // default: "it"

  const raw = HEADS[headName];
  const n = TOKENS.length;

  // Live pipeline for every query row -> attention weight matrix.
  const weights = useMemo(() => {
    return raw.map((row, i) => {
      const scaled = row.map((s) => s / Math.sqrt(D_K));
      const masked = scaled.map((s, j) => (causal && j > i ? -Infinity : s));
      return softmax(masked);
    });
  }, [raw, causal]);

  const qWeights = weights[query];
  const qScaled = raw[query].map((s) => s / Math.sqrt(D_K));

  // Geometry for the arc diagram.
  const W = 640;
  const tokenGap = W / n;
  const xOf = (i) => tokenGap * i + tokenGap / 2;
  const baseY = 30;

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-5">
        <SegmentedControl
          label="Attention head"
          options={Object.keys(HEADS)}
          value={headName}
          onChange={setHeadName}
        />
        <Toggle label="Causal mask (can't see the future)" checked={causal} onChange={setCausal} />
        <div className="text-xs text-slate-500">
          Click a token to make it the <span className="text-brand-300">query</span>.
        </div>
      </div>

      {/* Arc diagram */}
      <div className="overflow-x-auto scrollbar-thin">
        <svg width={W} height={150} className="mx-auto block min-w-[560px]">
          {/* arcs from query to each key */}
          {TOKENS.map((_, j) => {
            const w = qWeights[j];
            if (!isFinite(w) || w < 0.01) return null;
            const x1 = xOf(query);
            const x2 = xOf(j);
            const midY = baseY + 70 + Math.min(60, Math.abs(x1 - x2) * 0.25);
            return (
              <path
                key={j}
                d={`M ${x1} ${baseY + 14} Q ${(x1 + x2) / 2} ${midY} ${x2} ${baseY + 14}`}
                fill="none"
                stroke={heat(0.3 + w * 0.7)}
                strokeWidth={1 + w * 10}
                strokeLinecap="round"
                opacity={0.85}
              />
            );
          })}
          {/* token chips */}
          {TOKENS.map((t, i) => {
            const isQuery = i === query;
            const masked = causal && i > query;
            return (
              <g
                key={i}
                onClick={() => setQuery(i)}
                style={{ cursor: "pointer" }}
                opacity={masked ? 0.3 : 1}
              >
                <rect
                  x={xOf(i) - 26}
                  y={baseY}
                  width={52}
                  height={28}
                  rx={7}
                  fill={isQuery ? "#3a55f5" : "#1c2438"}
                  stroke={isQuery ? "#8eabff" : "#334155"}
                />
                <text
                  x={xOf(i)}
                  y={baseY + 19}
                  textAnchor="middle"
                  fontSize="13"
                  fontFamily="JetBrains Mono, monospace"
                  fill={isQuery ? "#fff" : "#cbd5e1"}
                >
                  {t}
                </text>
                <text x={xOf(i)} y={baseY - 8} textAnchor="middle" fontSize="10" fill="#475569">
                  {i}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Computation panel for the selected query */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="mb-3 text-sm text-slate-300">
          How much does{" "}
          <span className="rounded bg-brand-500/30 px-1.5 py-0.5 font-mono text-brand-100">
            “{TOKENS[query]}”
          </span>{" "}
          (query {query}) attend to each token?
        </div>
        <div className="space-y-1.5">
          {TOKENS.map((t, j) => {
            const masked = causal && j > query;
            const w = qWeights[j];
            return (
              <div key={j} className="flex items-center gap-3 text-sm">
                <span className="w-24 shrink-0 text-right font-mono text-slate-400">
                  {t}
                </span>
                <span className="w-16 shrink-0 text-right font-mono text-xs text-slate-600">
                  {masked ? "−∞" : fmt(qScaled[j], 2)}
                </span>
                <div className="relative h-5 flex-1 overflow-hidden rounded bg-ink-800/70">
                  <div
                    className="h-full rounded transition-all duration-300"
                    style={{
                      width: masked ? "0%" : `${Math.max(1, w * 100)}%`,
                      background: heat(0.3 + (isFinite(w) ? w : 0) * 0.7),
                    }}
                  />
                </div>
                <span className="w-14 shrink-0 text-right font-mono text-xs text-slate-300">
                  {masked ? "—" : `${(w * 100).toFixed(0)}%`}
                </span>
              </div>
            );
          })}
        </div>
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-[11px] text-slate-500">
          <span>
            raw score → ÷√d_k ({fmt(Math.sqrt(D_K), 1)}) → {causal ? "mask future →" : ""} softmax →
            weight
          </span>
          <span>Weights sum to 100% across visible keys.</span>
        </div>
      </div>

      {/* Full attention matrix */}
      <div>
        <div className="mb-2 text-xs uppercase tracking-wider text-slate-500">
          Full attention matrix · rows = queries, columns = keys
        </div>
        <div className="overflow-x-auto scrollbar-thin">
          <table className="border-separate" style={{ borderSpacing: 3 }}>
            <thead>
              <tr>
                <th />
                {TOKENS.map((t, j) => (
                  <th key={j} className="pb-1 font-mono text-[10px] font-normal text-slate-500">
                    {t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {weights.map((row, i) => (
                <tr key={i}>
                  <td className="pr-2 text-right font-mono text-[10px] text-slate-500">
                    {TOKENS[i]}
                  </td>
                  {row.map((w, j) => {
                    const masked = causal && j > i;
                    return (
                      <td key={j}>
                        <div
                          onClick={() => setQuery(i)}
                          className="flex h-8 w-9 cursor-pointer items-center justify-center rounded"
                          style={{
                            background: masked ? "transparent" : heat(w),
                            border: masked ? "1px dashed #27314a" : "none",
                            outline: i === query ? "2px solid #8eabff" : "none",
                          }}
                          title={`${TOKENS[i]} → ${TOKENS[j]} = ${masked ? "masked" : fmt(w, 3)}`}
                        >
                          <span
                            className="font-mono text-[9px]"
                            style={{ color: w > 0.5 ? "#0b0e1a" : "#94a3b8" }}
                          >
                            {masked ? "" : (w * 100).toFixed(0)}
                          </span>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
