import { useMemo, useState } from "react";
import { SegmentedControl, Slider } from "../ui";
import { heat } from "../scales";

/**
 * PositionalViz — two views of how position is injected into a transformer.
 *
 *  (a) "Sinusoidal" — a live heatmap of the classic sinusoidal positional-
 *      encoding matrix PE[pos, i], computed from the real formula:
 *        PE[pos, 2k]   = sin(pos / 10000^(2k/d))
 *        PE[pos, 2k+1] = cos(pos / 10000^(2k/d))
 *      Rows are positions, columns are embedding dimensions. Low-index columns
 *      oscillate fast (fine "clock hands"); high-index columns oscillate slowly.
 *
 *  (b) "RoPE" — the rotary view. Two token vectors are each rotated by an angle
 *      proportional to their position (angle = pos · θ). Drag the positions and
 *      watch: the *individual* angles change, but the *angle between* the two
 *      vectors — which is what their dot product depends on — depends only on
 *      the position difference. That is the whole point of RoPE: it encodes
 *      relative position.
 *
 * Everything is deterministic and computed live from the formulas.
 */

const D_MODEL = 32; // embedding dimension for the heatmap (even, for sin/cos pairs)

// Sinusoidal PE matrix: shape [seqLen, d].
function sinusoidalPE(seqLen, d) {
  const M = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const row = new Array(d);
    for (let i = 0; i < d; i++) {
      const pair = Math.floor(i / 2); // dimensions come in (sin, cos) pairs
      const freq = 1 / Math.pow(10000, (2 * pair) / d);
      row[i] = i % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq);
    }
    M.push(row);
  }
  return M;
}

export default function PositionalViz() {
  const [view, setView] = useState("sinusoidal");

  return (
    <div className="space-y-5">
      <SegmentedControl
        options={[
          { value: "sinusoidal", label: "Sinusoidal heatmap" },
          { value: "rope", label: "RoPE rotation" },
        ]}
        value={view}
        onChange={setView}
      />
      {view === "sinusoidal" ? <SinusoidalView /> : <RopeView />}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* (a) Sinusoidal heatmap                                              */
/* ------------------------------------------------------------------ */

function SinusoidalView() {
  const [seqLen, setSeqLen] = useState(32);
  const matrix = useMemo(() => sinusoidalPE(seqLen, D_MODEL), [seqLen]);

  // Draw as raw SVG so we control cell size for the (possibly tall) matrix.
  // Values are in [-1, 1]; map to the heat ramp via (v+1)/2.
  const cellW = 16;
  const cellH = Math.max(7, Math.min(16, 380 / seqLen));
  const gridW = D_MODEL * cellW;
  const gridH = seqLen * cellH;
  const padL = 44;
  const padT = 24;

  return (
    <div className="space-y-4">
      <div className="max-w-sm">
        <Slider
          label="Sequence length (positions)"
          min={4}
          max={64}
          step={1}
          value={seqLen}
          onChange={(v) => setSeqLen(Math.round(v))}
          format={(v) => `${Math.round(v)} tokens`}
          accent="cyan"
        />
      </div>

      <div className="overflow-x-auto scrollbar-thin">
        <svg
          width={gridW + padL + 16}
          height={gridH + padT + 24}
          className="block"
        >
          {/* column axis label */}
          <text x={padL + gridW / 2} y={12} textAnchor="middle" fontSize="11" fill="#64748b">
            embedding dimension i (0 … {D_MODEL - 1})  →  faster left, slower right
          </text>
          {/* row axis label */}
          <text
            x={12}
            y={padT + gridH / 2}
            textAnchor="middle"
            fontSize="11"
            fill="#64748b"
            transform={`rotate(-90, 12, ${padT + gridH / 2})`}
          >
            position pos
          </text>

          {/* cells */}
          {matrix.map((row, pos) =>
            row.map((v, i) => (
              <rect
                key={`${pos}-${i}`}
                x={padL + i * cellW}
                y={padT + pos * cellH}
                width={cellW - 1}
                height={cellH - 1}
                fill={heat((v + 1) / 2)}
              >
                <title>{`pos=${pos}, dim=${i}: ${v.toFixed(3)}`}</title>
              </rect>
            ))
          )}

          {/* a few position tick labels */}
          {matrix.map((_, pos) =>
            pos % Math.ceil(seqLen / 8) === 0 ? (
              <text
                key={pos}
                x={padL - 6}
                y={padT + pos * cellH + cellH * 0.8}
                textAnchor="end"
                fontSize="9"
                fill="#475569"
                fontFamily="JetBrains Mono, monospace"
              >
                {pos}
              </text>
            ) : null
          )}
        </svg>
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: heat(0) }} />
          −1
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: heat(0.5) }} />
          0
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: heat(1) }} />
          +1
        </span>
        <span>
          Each row is a unique "fingerprint" for one position. Left columns are fast clock hands;
          right columns barely move across the whole sequence.
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* (b) RoPE 2D rotation demo                                           */
/* ------------------------------------------------------------------ */

function RopeView() {
  // Two tokens at positions m and n. A single 2D feature pair is rotated by
  // angle = position · θ. We expose θ via a "frequency" control too.
  const [m, setM] = useState(2);
  const [n, setN] = useState(5);
  const [thetaDeg, setThetaDeg] = useState(30); // θ in degrees per position step

  const theta = (thetaDeg * Math.PI) / 180;

  // The unrotated base feature vector for the pair (same content for both
  // tokens, so any difference in the plot is purely positional). Fixed,
  // deterministic direction.
  const base = { x: 1, y: 0 };

  const rotate = (v, ang) => ({
    x: v.x * Math.cos(ang) - v.y * Math.sin(ang),
    y: v.x * Math.sin(ang) + v.y * Math.cos(ang),
  });

  const angM = m * theta;
  const angN = n * theta;
  const vM = rotate(base, angM);
  const vN = rotate(base, angN);

  // Relative angle between the two rotated vectors = (n - m) · θ.
  const relAngleRad = (n - m) * theta;
  const relAngleDeg = ((relAngleRad * 180) / Math.PI) % 360;
  const cosRel = Math.cos(relAngleRad); // ∝ qₘ · kₙ for this pair

  // Plot geometry.
  const size = 320;
  const cx = size / 2;
  const cy = size / 2;
  const R = size / 2 - 30; // axis radius
  const px = (v) => cx + v.x * R;
  const py = (v) => cy - v.y * R; // SVG y is down

  const Arrow = ({ v, color, label }) => (
    <g>
      <line x1={cx} y1={cy} x2={px(v)} y2={py(v)} stroke={color} strokeWidth={3} markerEnd={`url(#rope-${color.replace("#", "")})`} />
      <circle cx={px(v)} cy={py(v)} r={4} fill={color} />
      <text x={px(v) + (v.x >= 0 ? 8 : -8)} y={py(v) - 6} fontSize="12" fill={color} fontWeight="600" textAnchor={v.x >= 0 ? "start" : "end"}>
        {label}
      </text>
    </g>
  );

  const colM = "#22d3ee";
  const colN = "#f59e0b";

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-3">
        <Slider label="Token A position m" min={0} max={12} step={1} value={m} onChange={(v) => setM(Math.round(v))} format={(v) => `m = ${Math.round(v)}`} accent="cyan" />
        <Slider label="Token B position n" min={0} max={12} step={1} value={n} onChange={(v) => setN(Math.round(v))} format={(v) => `n = ${Math.round(v)}`} accent="violet" />
        <Slider label="Angle per step θ" min={5} max={90} step={5} value={thetaDeg} onChange={(v) => setThetaDeg(Math.round(v))} format={(v) => `${Math.round(v)}°`} />
      </div>

      <div className="flex flex-col items-center gap-5 lg:flex-row lg:items-start lg:justify-center">
        <svg width={size} height={size} className="shrink-0 rounded-xl border border-white/10 bg-ink-900/40">
          <defs>
            <marker id={`rope-${colM.replace("#", "")}`} markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill={colM} />
            </marker>
            <marker id={`rope-${colN.replace("#", "")}`} markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill={colN} />
            </marker>
          </defs>

          {/* unit circle + axes */}
          <circle cx={cx} cy={cy} r={R} fill="none" stroke="#27314a" strokeWidth={1} />
          <line x1={cx - R} y1={cy} x2={cx + R} y2={cy} stroke="#27314a" strokeWidth={1} />
          <line x1={cx} y1={cy - R} x2={cx} y2={cy + R} stroke="#27314a" strokeWidth={1} />

          {/* relative-angle wedge between the two vectors */}
          <path
            d={describeWedge(cx, cy, R * 0.42, angM, angN)}
            fill="rgba(168,85,247,0.18)"
            stroke="#a855f7"
            strokeWidth={1}
          />

          <Arrow v={vM} color={colM} label={`A (pos ${m})`} />
          <Arrow v={vN} color={colN} label={`B (pos ${n})`} />

          <text x={cx} y={size - 8} textAnchor="middle" fontSize="10" fill="#64748b">
            one rotated 2-D feature pair
          </text>
        </svg>

        <div className="grid w-full max-w-xs grid-cols-1 gap-2 text-sm">
          <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">A rotated by</div>
            <div className="font-mono text-cyan-300">m·θ = {(m * thetaDeg).toFixed(0)}°</div>
          </div>
          <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">B rotated by</div>
            <div className="font-mono text-amber-300">n·θ = {(n * thetaDeg).toFixed(0)}°</div>
          </div>
          <div className="rounded-lg border border-violet-400/30 bg-violet-500/[0.08] px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-violet-300/80">
              relative angle (n−m)·θ
            </div>
            <div className="font-mono text-violet-200">
              {(((n - m) * thetaDeg) % 360).toFixed(0)}°
            </div>
          </div>
          <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">
              dot product ∝ cos(rel)
            </div>
            <div className="font-mono text-slate-200">{cosRel.toFixed(3)}</div>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4 text-sm text-slate-300">
        Slide A and B while keeping their <em>gap</em> fixed (e.g. m = 2, n = 5, then m = 6, n = 9):
        both arrows swing around the circle, but the{" "}
        <span className="text-violet-300">purple wedge between them</span> — the relative angle{" "}
        <span className="font-mono">(n − m)·θ = {relAngleDeg.toFixed(0)}°</span> — never changes, so
        the dot product <span className="font-mono">cos((n − m)·θ) = {cosRel.toFixed(3)}</span> never
        changes. RoPE makes attention scores depend only on <strong>relative</strong> position.
      </div>
    </div>
  );
}

// SVG arc wedge from angle a0 to a1 (radians, math convention, CCW positive),
// centered at (cx,cy) with radius r. Used to shade the relative angle.
function describeWedge(cx, cy, r, a0, a1) {
  const x0 = cx + r * Math.cos(a0);
  const y0 = cy - r * Math.sin(a0);
  const x1 = cx + r * Math.cos(a1);
  const y1 = cy - r * Math.sin(a1);
  let delta = a1 - a0;
  // Normalize to (-π, π] for a clean minor-arc wedge.
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta <= -Math.PI) delta += 2 * Math.PI;
  const largeArc = Math.abs(delta) > Math.PI ? 1 : 0;
  const sweep = delta < 0 ? 1 : 0; // SVG sweep flag (y-down flips orientation)
  return `M ${cx} ${cy} L ${x0} ${y0} A ${r} ${r} 0 ${largeArc} ${sweep} ${x1} ${y1} Z`;
}
