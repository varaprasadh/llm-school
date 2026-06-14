import { useMemo, useState } from "react";
import { Slider } from "../ui";

/**
 * HeadSplit — anatomy of multi-head attention as a dataflow diagram.
 *
 * A single d_model-dimensional token vector is sliced into `h` equal,
 * colored segments (one per head). Each segment is routed into its own little
 * attention box (its own Q/K/V projections, its own softmax) running in a
 * subspace of size d_k = d_model / h. The per-head outputs are then
 * concatenated back into a single d_model vector and passed through the output
 * projection W_O.
 *
 * Everything is deterministic and driven by the head-count slider. h is
 * constrained to the divisors of d_model so that d_k = d_model / h is always a
 * whole number (you can't split 12 dimensions into 5 equal heads).
 */

const D_MODEL = 12;
const DIVISORS = [1, 2, 3, 4, 6, 12]; // h values for which d_model / h is integral

// One distinct hue per head index, cycled if h is large (here h ≤ 12).
const HEAD_COLORS = [
  "#5b7dff", // brand
  "#22d3ee", // cyan
  "#34d399", // emerald
  "#a855f7", // violet
  "#f59e0b", // amber
  "#fb7185", // rose
  "#818cf8", // indigo
  "#2dd4bf", // teal
  "#facc15", // yellow
  "#f472b6", // pink
  "#4ade80", // green
  "#38bdf8", // sky
];

// Snap an arbitrary slider value to the nearest allowed divisor of d_model.
function snapToDivisor(v) {
  let best = DIVISORS[0];
  let bestDist = Infinity;
  for (const d of DIVISORS) {
    const dist = Math.abs(d - v);
    if (dist < bestDist) {
      bestDist = dist;
      best = d;
    }
  }
  return best;
}

export default function HeadSplit() {
  const [h, setH] = useState(3);
  const dK = D_MODEL / h;

  // Layout geometry.
  const W = 720;
  const H = 360;
  const cellW = 30; // width of one scalar cell in a vector strip
  const stripW = D_MODEL * cellW; // full d_model strip width
  const stripX = (W - stripW) / 2; // left edge so the strip is centered

  const inputY = 26;
  const boxRowY = 150; // top of the per-head attention boxes
  const boxH = 64;
  const concatY = 262;
  const outY = 320;

  // Cell ranges [start, end) for each head along the d_model axis.
  const heads = useMemo(() => {
    const out = [];
    for (let k = 0; k < h; k++) {
      out.push({ start: k * dK, end: (k + 1) * dK, color: HEAD_COLORS[k % HEAD_COLORS.length] });
    }
    return out;
  }, [h, dK]);

  // Render one horizontal strip of `count` cells starting at x, each filled
  // according to `colorFor(localIndex)`.
  const Strip = ({ x, y, count, colorFor, label, sub }) => (
    <g>
      {Array.from({ length: count }, (_, i) => (
        <rect
          key={i}
          x={x + i * cellW}
          y={y}
          width={cellW - 2}
          height={26}
          rx={3}
          fill={colorFor(i)}
          stroke="rgba(255,255,255,0.14)"
          strokeWidth={0.75}
        />
      ))}
      {label && (
        <text x={x - 10} y={y + 18} textAnchor="end" fontSize="11" fill="#94a3b8" fontFamily="JetBrains Mono, monospace">
          {label}
        </text>
      )}
      {sub && (
        <text x={x + (count * cellW) / 2} y={y - 8} textAnchor="middle" fontSize="10" fill="#64748b">
          {sub}
        </text>
      )}
    </g>
  );

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-6">
        <div className="min-w-[260px] flex-1">
          <Slider
            label="Number of heads h"
            min={1}
            max={12}
            step={1}
            value={h}
            onChange={(v) => setH(snapToDivisor(v))}
            format={(v) => `h = ${v}`}
          />
          <div className="mt-1 text-[11px] text-slate-500">
            h must divide d_model = {D_MODEL}, so it snaps to {DIVISORS.join(", ")}.
          </div>
        </div>
        <div className="flex gap-2 text-center">
          <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">d_model</div>
            <div className="font-mono text-lg text-slate-200">{D_MODEL}</div>
          </div>
          <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">heads h</div>
            <div className="font-mono text-lg text-brand-200">{h}</div>
          </div>
          <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">d_k = d_model / h</div>
            <div className="font-mono text-lg text-cyan-300">{dK}</div>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto scrollbar-thin">
        <svg width={W} height={H} className="mx-auto block min-w-[640px]">
          <defs>
            <marker id="hs-arrow" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#475569" />
            </marker>
          </defs>

          {/* 1) Input token vector, colored by which head each dimension belongs to */}
          <Strip
            x={stripX}
            y={inputY}
            count={D_MODEL}
            colorFor={(i) => {
              const head = heads.find((hd) => i >= hd.start && i < hd.end);
              return head ? head.color : "#334155";
            }}
            label="x"
            sub={`one token, d_model = ${D_MODEL}`}
          />

          {/* Slice boundary ticks between heads */}
          {heads.slice(1).map((hd, k) => (
            <line
              key={k}
              x1={stripX + hd.start * cellW - 1}
              y1={inputY - 4}
              x2={stripX + hd.start * cellW - 1}
              y2={inputY + 30}
              stroke="#e2e8f0"
              strokeWidth={1.5}
            />
          ))}

          {/* 2) Per-head attention boxes */}
          {heads.map((hd, k) => {
            const segCells = hd.end - hd.start;
            const segCenterX = stripX + (hd.start + segCells / 2) * cellW;
            // Spread the boxes evenly across the available width.
            const boxW = Math.min(120, (W - 40) / h - 12);
            const boxX = 20 + (k + 0.5) * ((W - 40) / h) - boxW / 2;
            const boxCenterX = boxX + boxW / 2;
            return (
              <g key={k}>
                {/* routing line: input segment -> head box */}
                <path
                  d={`M ${segCenterX} ${inputY + 30} C ${segCenterX} ${inputY + 70}, ${boxCenterX} ${boxRowY - 40}, ${boxCenterX} ${boxRowY}`}
                  fill="none"
                  stroke={hd.color}
                  strokeWidth={1.5}
                  opacity={0.7}
                  markerEnd="url(#hs-arrow)"
                />
                {/* the head's attention box */}
                <rect
                  x={boxX}
                  y={boxRowY}
                  width={boxW}
                  height={boxH}
                  rx={8}
                  fill={`${hd.color}1f`}
                  stroke={hd.color}
                  strokeWidth={1.25}
                />
                <text x={boxCenterX} y={boxRowY + 20} textAnchor="middle" fontSize="11" fill="#e2e8f0" fontWeight="600">
                  head {k + 1}
                </text>
                <text x={boxCenterX} y={boxRowY + 37} textAnchor="middle" fontSize="9" fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
                  softmax(QKᵀ/√{dK})V
                </text>
                <text x={boxCenterX} y={boxRowY + 52} textAnchor="middle" fontSize="9" fill="#94a3b8">
                  subspace dim {dK}
                </text>

                {/* routing line: head box -> its slot in the concatenated vector */}
                <path
                  d={`M ${boxCenterX} ${boxRowY + boxH} C ${boxCenterX} ${concatY - 40}, ${segCenterX} ${concatY - 40}, ${segCenterX} ${concatY}`}
                  fill="none"
                  stroke={hd.color}
                  strokeWidth={1.5}
                  opacity={0.7}
                  markerEnd="url(#hs-arrow)"
                />
              </g>
            );
          })}

          {/* 3) Concatenated per-head outputs (same coloring, back to d_model) */}
          <Strip
            x={stripX}
            y={concatY}
            count={D_MODEL}
            colorFor={(i) => {
              const head = heads.find((hd) => i >= hd.start && i < hd.end);
              return head ? head.color : "#334155";
            }}
            label="concat"
            sub={`stack heads side by side → back to ${D_MODEL} dims`}
          />

          {/* 4) Output projection W_O mixes all heads together */}
          <line
            x1={stripX + stripW / 2}
            y1={concatY + 30}
            x2={stripX + stripW / 2}
            y2={outY}
            stroke="#475569"
            strokeWidth={1.5}
            markerEnd="url(#hs-arrow)"
          />
          <rect
            x={stripX + stripW / 2 - 26}
            y={concatY + 36}
            width={52}
            height={20}
            rx={5}
            fill="#1c2438"
            stroke="#64748b"
          />
          <text
            x={stripX + stripW / 2}
            y={concatY + 50}
            textAnchor="middle"
            fontSize="11"
            fill="#e2e8f0"
            fontFamily="JetBrains Mono, monospace"
          >
            W_O
          </text>

          {/* output vector — uniform color: heads are now blended */}
          <Strip
            x={stripX}
            y={outY}
            count={D_MODEL}
            colorFor={() => "#8eabff"}
            label="output"
            sub={`d_model = ${D_MODEL}, all heads mixed`}
          />
        </svg>
      </div>

      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4 text-sm text-slate-300">
        With <span className="font-mono text-brand-200">h = {h}</span>, the {D_MODEL}-dimensional
        vector is cut into {h} colored slice{h === 1 ? "" : "s"} of{" "}
        <span className="font-mono text-cyan-300">{dK}</span> dimension{dK === 1 ? "" : "s"} each.
        Every slice gets its <em>own</em> attention computed in its <em>own</em> subspace, in
        parallel. The {h} results are concatenated back into {D_MODEL} dimensions and{" "}
        <span className="font-mono text-slate-200">W_O</span> mixes them into the final output.
        {h === 1 && " At h = 1 this is exactly ordinary single-head attention."}
        {h === D_MODEL &&
          " At h = d_model every head owns a single dimension — maximal specialization, smallest subspaces."}
      </div>
    </div>
  );
}
