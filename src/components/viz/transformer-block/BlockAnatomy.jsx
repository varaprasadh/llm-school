import { useMemo, useState } from "react";
import { Slider, SegmentedControl } from "../ui";
import { heat } from "../scales";

/**
 * BlockAnatomy — two related visuals for the non-attention half of a
 * transformer block, drawn as bespoke SVG (matching the house style of
 * AttentionExplorer / TransformerStack).
 *
 * (a) "anatomy"  — an interactive pre-norm block. Data flows
 *        x -> LN -> Multi-Head Attention -> ⊕(residual) ->
 *             LN -> FFN -> ⊕(residual).
 *     Every node is clickable; a side panel reveals its role, formula, and the
 *     tensor shape flowing through it. The two residual "skip" arrows are drawn
 *     distinctly (curved, dashed, brand-colored) so the gradient highway is
 *     visible at a glance.
 *
 * (b) "residuals" — a deterministic depth demo. A <Slider> sets network DEPTH
 *     (1..50). We show how a signal's magnitude evolves layer by layer WITH a
 *     residual connection (stays ~constant, because each layer adds to an
 *     identity path) versus WITHOUT one (a plain deep stack whose per-layer
 *     gain compounds, so the signal either vanishes or explodes). Fully
 *     deterministic — closed-form curves, no randomness in render.
 */

// ---- Side-panel info for the anatomy nodes, keyed by node id. -------------
const NODE_INFO = {
  in: {
    name: "Residual stream in",
    color: "#64748b",
    shape: "(B, T, d)",
    formula: "x",
    lines: [
      "The running representation entering the block — one d-dim vector per token.",
      "The block will read this, compute two updates, and add them back on. It is never overwritten.",
    ],
  },
  ln1: {
    name: "LayerNorm (pre-attention)",
    color: "#22d3ee",
    shape: "(B, T, d)",
    formula: "\\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}\\,\\gamma + \\beta",
    lines: [
      "Normalizes each token's vector to zero mean and unit variance across the d features,",
      "then rescales with learned γ (gain) and β (shift). Sits INSIDE the residual branch (pre-norm).",
    ],
  },
  attn: {
    name: "Multi-Head Attention",
    color: "#a855f7",
    shape: "(B, T, d)",
    formula: "\\text{MHA}(\\hat{x})",
    lines: [
      "The communication step: every token looks at the others (causally) and pulls in relevant info.",
      "This is the ONLY place tokens mix. Covered in Chapters 8–9 — here it is a black box.",
    ],
  },
  add1: {
    name: "Residual add ⊕",
    color: "#5b7dff",
    shape: "(B, T, d)",
    formula: "x \\leftarrow x + \\text{MHA}(\\text{LN}(x))",
    lines: [
      "Add the attention output back onto the UN-normalized input that entered the block.",
      "The +x term is the gradient highway: it gives every layer a direct, identity path to the loss.",
    ],
  },
  ln2: {
    name: "LayerNorm (pre-FFN)",
    color: "#22d3ee",
    shape: "(B, T, d)",
    formula: "\\hat{x}' = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}\\,\\gamma + \\beta",
    lines: [
      "A second normalization, on the updated residual stream, before the feed-forward network.",
      "Again pre-norm: the clean residual path runs straight through, the norm lives on the branch.",
    ],
  },
  ffn: {
    name: "Feed-Forward Network",
    color: "#f59e0b",
    shape: "(B, T, d)  ·  hidden 4d",
    formula: "\\text{FFN}(\\hat{x}') = W_2\\,\\text{act}(W_1\\hat{x}' + b_1) + b_2",
    lines: [
      "Two linear layers with a nonlinearity: expand d → 4d, apply GELU/SwiGLU, project 4d → d.",
      "Applied to each token INDEPENDENTLY. This is where ~2/3 of a block's parameters live.",
    ],
  },
  add2: {
    name: "Residual add ⊕",
    color: "#5b7dff",
    shape: "(B, T, d)",
    formula: "x \\leftarrow x + \\text{FFN}(\\text{LN}(x))",
    lines: [
      "Add the FFN output back onto the stream. The block's job is done — shape is unchanged.",
      "Two reads, two adds: x ← x + Attn(LN x); x ← x + FFN(LN x). That is the whole block.",
    ],
  },
  out: {
    name: "Residual stream out",
    color: "#34d399",
    shape: "(B, T, d)",
    formula: "x_{\\ell}",
    lines: [
      "The refined stream, same shape as the input, ready for the next identical block.",
      "Stack N of these and you have the body of a transformer.",
    ],
  },
};

// The vertical order of nodes in the anatomy diagram (top -> bottom in reading
// order is bottom -> top, but we lay out top-first since SVG y grows down).
// We render bottom (input) to top (output); list is in flow order.
const FLOW = ["in", "ln1", "attn", "add1", "ln2", "ffn", "add2", "out"];

export default function BlockAnatomy() {
  const [mode, setMode] = useState("anatomy");
  return (
    <div className="space-y-5">
      <SegmentedControl
        label="View"
        options={[
          { value: "anatomy", label: "Block anatomy" },
          { value: "residuals", label: "Why residuals" },
        ]}
        value={mode}
        onChange={setMode}
      />
      {mode === "anatomy" ? <Anatomy /> : <ResidualDemo />}
    </div>
  );
}

/* ===================================================================== */
/*  (a) Interactive pre-norm block anatomy                                */
/* ===================================================================== */

function Anatomy() {
  const [selected, setSelected] = useState("ffn");
  const info = NODE_INFO[selected];

  // ---- Layout. Build rows top-first (logits side up), assign y. -----------
  const W = 360;
  const cx = W / 2;
  const boxW = 230;
  const gap = 20;

  // Heights per node kind.
  const H = { in: 44, ln1: 40, attn: 56, add1: 38, ln2: 40, ffn: 64, add2: 38, out: 44 };

  // Order rows from TOP (out) to BOTTOM (in) for y assignment.
  const orderTopFirst = [...FLOW].reverse();
  let y = 14;
  const placed = {};
  const rows = orderTopFirst.map((id) => {
    const h = H[id];
    const top = y;
    y += h + gap;
    const row = { id, top, h, mid: top + h / 2 };
    placed[id] = row;
    return row;
  });
  const totalH = y;

  // Residual skip arrows: from BEFORE ln1 (in) bypassing attn into add1, and
  // from after add1 (the stream at ln2 level) bypassing ffn into add2.
  // We draw them as curved paths on the right side of the column.
  const skipX = cx + boxW / 2 + 26; // x of the skip lane

  const isSel = (id) => id === selected;

  return (
    <div className="flex flex-col gap-5 lg:flex-row">
      {/* ---- SVG diagram ---- */}
      <div className="lg:w-[56%]">
        <div className="mb-3 text-xs text-slate-500">
          Click any node to inspect its role, formula, and tensor shape. The two{" "}
          <span style={{ color: "#5b7dff" }}>blue dashed arrows</span> are the residual skips —
          the identity path that information and gradients ride straight through the block.
        </div>
        <div className="overflow-x-auto scrollbar-thin">
          <svg width={W + 60} height={totalH} className="mx-auto block" style={{ maxWidth: "100%" }}>
            <defs>
              <marker
                id="ba-skiphead"
                markerWidth="9"
                markerHeight="9"
                refX="6"
                refY="3"
                orient="auto"
              >
                <path d="M0,0 L6,3 L0,6 Z" fill="#5b7dff" />
              </marker>
              <marker
                id="ba-flowhead"
                markerWidth="8"
                markerHeight="8"
                refX="5"
                refY="2.6"
                orient="auto"
              >
                <path d="M0,0 L5,2.6 L0,5.2 Z" fill="#3a4a72" />
              </marker>
            </defs>

            {/* Straight flow connectors between adjacent boxes (the main path). */}
            {FLOW.slice(0, -1).map((id, i) => {
              const a = placed[id];
              const b = placed[FLOW[i + 1]];
              return (
                <line
                  key={`flow-${id}`}
                  x1={cx}
                  y1={a.top} /* a is above b in flow => a.top is its upper edge */
                  x2={cx}
                  y2={b.top + b.h}
                  stroke="#27314a"
                  strokeWidth={2.5}
                  markerEnd="url(#ba-flowhead)"
                />
              );
            })}

            {/* Residual skip #1: in -> add1 (bypassing ln1 + attn). */}
            <ResidualSkip
              fromY={placed.in.mid}
              toY={placed.add1.mid}
              x0={cx + boxW / 2 - 8}
              lane={skipX}
              targetX={cx + 18}
              highlight={isSel("add1") || isSel("in")}
            />
            {/* Residual skip #2: add1 -> add2 (bypassing ln2 + ffn). */}
            <ResidualSkip
              fromY={placed.add1.mid}
              toY={placed.add2.mid}
              x0={cx + boxW / 2 - 8}
              lane={skipX + 16}
              targetX={cx + 18}
              highlight={isSel("add2") || isSel("add1")}
            />

            {/* Nodes. */}
            {rows.map((r) => {
              const meta = NODE_INFO[r.id];
              const sel = isSel(r.id);
              const isAdd = r.id === "add1" || r.id === "add2";
              const isStream = r.id === "in" || r.id === "out";
              const label = nodeLabel(r.id);

              if (isAdd) {
                // Draw the residual add as a circled ⊕.
                return (
                  <g
                    key={r.id}
                    onClick={() => setSelected(r.id)}
                    style={{ cursor: "pointer" }}
                  >
                    <circle
                      cx={cx}
                      cy={r.mid}
                      r={15}
                      fill={sel ? "#1c2438" : "#10162a"}
                      stroke={sel ? "#8eabff" : "#5b7dff"}
                      strokeWidth={sel ? 2.5 : 1.8}
                    />
                    <text
                      x={cx}
                      y={r.mid + 6}
                      textAnchor="middle"
                      fontSize="18"
                      fill={sel ? "#cdd9ff" : "#8eabff"}
                    >
                      ⊕
                    </text>
                    <text
                      x={cx - 24}
                      y={r.mid + 4}
                      textAnchor="end"
                      fontSize="9.5"
                      fontFamily="JetBrains Mono, monospace"
                      fill="#475569"
                    >
                      add
                    </text>
                  </g>
                );
              }

              return (
                <g key={r.id} onClick={() => setSelected(r.id)} style={{ cursor: "pointer" }}>
                  <rect
                    x={cx - boxW / 2}
                    y={r.top}
                    width={boxW}
                    height={r.h}
                    rx={9}
                    fill={sel ? "#1c2438" : isStream ? "#11172a" : "#141a2e"}
                    stroke={sel ? meta.color : "rgba(255,255,255,0.10)"}
                    strokeWidth={sel ? 2 : 1}
                    strokeDasharray={isStream ? "5 4" : undefined}
                  />
                  {/* left color tab */}
                  <rect
                    x={cx - boxW / 2}
                    y={r.top}
                    width={5}
                    height={r.h}
                    rx={2}
                    fill={meta.color}
                    opacity={sel ? 1 : 0.7}
                  />
                  <text
                    x={cx - 2}
                    y={r.id === "ffn" || r.id === "attn" ? r.top + 22 : r.mid + 4}
                    textAnchor="middle"
                    fontSize="12.5"
                    fontWeight="600"
                    fill={sel ? "#f1f5f9" : "#cbd5e1"}
                  >
                    {label}
                  </text>

                  {/* FFN: draw the d -> 4d -> d expansion glyph. */}
                  {r.id === "ffn" && (
                    <text
                      x={cx - 2}
                      y={r.top + 46}
                      textAnchor="middle"
                      fontSize="10"
                      fontFamily="JetBrains Mono, monospace"
                      fill="#f59e0b"
                    >
                      d → 4d → d
                    </text>
                  )}
                  {r.id === "attn" && (
                    <text
                      x={cx - 2}
                      y={r.top + 44}
                      textAnchor="middle"
                      fontSize="9.5"
                      fontFamily="JetBrains Mono, monospace"
                      fill="#a855f7"
                    >
                      tokens mix
                    </text>
                  )}

                  {/* shape badge */}
                  <text
                    x={cx + boxW / 2 - 9}
                    y={r.top + 13}
                    textAnchor="end"
                    fontSize="8.5"
                    fontFamily="JetBrains Mono, monospace"
                    fill="#475569"
                  >
                    {isStream ? "(B,T,d)" : ""}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* ---- Side panel ---- */}
      <div className="lg:w-[44%]">
        <div className="sticky top-4 rounded-xl border border-white/10 bg-ink-900/60 p-4">
          <div className="mb-1 flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: info.color }} />
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              Selected node
            </span>
          </div>
          <div className="mb-2 text-lg font-semibold text-slate-100">{info.name}</div>

          <div className="mb-3 inline-flex items-center gap-2 rounded-lg border border-white/10 bg-ink-800/70 px-3 py-1.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">shape</span>
            <span className="font-mono text-sm text-cyan-200">{info.shape}</span>
          </div>

          <div className="mb-3 overflow-x-auto rounded-lg border border-white/10 bg-ink-950/60 px-3 py-2 scrollbar-thin">
            <Formula tex={info.formula} />
          </div>

          <ul className="space-y-1.5 text-sm leading-relaxed text-slate-300">
            {info.lines.map((l, i) => (
              <li key={i} className="flex gap-2">
                <span className="mt-1.5 inline-block h-1 w-1 shrink-0 rounded-full bg-slate-600" />
                <span>{l}</span>
              </li>
            ))}
          </ul>

          <div className="mt-4 rounded-lg border border-brand-500/30 bg-brand-500/[0.06] p-3">
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-brand-200">
              The block in two lines
            </div>
            <div className="font-mono text-xs leading-relaxed text-slate-300">
              x ← x + Attn(LN(x))
              <br />
              x ← x + FFN(LN(x))
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** A single residual "skip" arrow: leaves the stream at fromY, bows out into a
 * lane on the right, and re-enters at the ⊕ at toY. Drawn distinctly (dashed,
 * brand-colored, arrowhead) so the identity path is unmistakable. */
function ResidualSkip({ fromY, toY, x0, lane, targetX, highlight }) {
  const d = `M ${x0} ${fromY}
             C ${lane} ${fromY}, ${lane} ${toY}, ${targetX} ${toY}`;
  return (
    <path
      d={d}
      fill="none"
      stroke="#5b7dff"
      strokeWidth={highlight ? 3 : 2}
      strokeDasharray="5 4"
      strokeLinecap="round"
      opacity={highlight ? 1 : 0.7}
      markerEnd="url(#ba-skiphead)"
    />
  );
}

function nodeLabel(id) {
  switch (id) {
    case "in":
      return "x  (residual stream)";
    case "ln1":
      return "LayerNorm";
    case "attn":
      return "Multi-Head Attention";
    case "ln2":
      return "LayerNorm";
    case "ffn":
      return "Feed-Forward (MLP)";
    case "out":
      return "x′  (to next block)";
    default:
      return id;
  }
}

/** Minimal renderer for the short formulas in the side panel. We avoid pulling
 * KaTeX into the viz component (the chapter prose already renders rich math);
 * here we display the LaTeX-ish string in a readable monospace form by
 * stripping the most common TeX wrappers. */
function Formula({ tex }) {
  const pretty = useMemo(() => prettyTex(tex), [tex]);
  return <span className="font-mono text-[13px] leading-relaxed text-slate-200">{pretty}</span>;
}

function prettyTex(s) {
  return s
    .replace(/\\hat\{([^}]*)\}/g, "$1̂")
    .replace(/\\mu/g, "μ")
    .replace(/\\sigma/g, "σ")
    .replace(/\\epsilon/g, "ε")
    .replace(/\\gamma/g, "γ")
    .replace(/\\beta/g, "β")
    .replace(/\\ell/g, "ℓ")
    .replace(/\\leftarrow/g, "←")
    .replace(/\\text\{([^}]*)\}/g, "$1")
    .replace(/\\sqrt\{([^}]*)\}/g, "√($1)")
    .replace(/\\frac\{([^}]*)\}\{([^}]*)\}/g, "($1)/($2)")
    .replace(/\\,/g, " ")
    .replace(/[{}]/g, "")
    .replace(/\^2/g, "²")
    .replace(/_1/g, "₁")
    .replace(/_2/g, "₂")
    .replace(/\\cdot/g, "·")
    .trim();
}

/* ===================================================================== */
/*  (b) Why residuals: depth demo                                         */
/* ===================================================================== */

// Per-layer multiplicative gain for a "plain" (no-residual) deep stack. A
// real layer's Jacobian rarely has gain exactly 1; we model a representative
// 0.85× attenuation per layer to make the vanishing-signal effect vivid and
// deterministic. (Pick >1 and it would explode instead — the point is that a
// product of L gains is unstable, while 1 + small stays near 1.)
const PLAIN_GAIN = 0.85;

function ResidualDemo() {
  const [depth, setDepth] = useState(24);

  // Closed-form magnitude after passing through ℓ layers, starting at 1.0.
  // WITHOUT residual: signal *= PLAIN_GAIN each layer  ->  PLAIN_GAIN^ℓ.
  // WITH residual: y = x + f(x). With f a normalized branch contributing a
  // small, bounded amount, the stream magnitude grows mildly and predictably
  // rather than collapsing; we model it as staying ~1 (slow sqrt-like drift),
  // which matches the empirical observation that pre-norm residual streams keep
  // a stable scale across dozens of layers.
  const series = useMemo(() => {
    const withRes = [];
    const without = [];
    for (let l = 0; l <= depth; l++) {
      without.push([l, Math.pow(PLAIN_GAIN, l)]);
      // residual stream: starts at 1, each block adds a bounded contribution;
      // norm keeps the *input* to each branch unit-scale, so the running norm
      // drifts up like sqrt(1 + l * c) — bounded, never collapsing.
      withRes.push([l, Math.sqrt(1 + l * 0.04)]);
    }
    return { withRes, without };
  }, [depth]);

  // Tail magnitudes for the headline stats.
  const lastWith = series.withRes[series.withRes.length - 1][1];
  const lastWithout = series.without[series.without.length - 1][1];

  // ---- bespoke SVG plot (so the contrast is hand-tunable & vivid) ----------
  const W = 560;
  const Hgt = 280;
  const m = { top: 18, right: 18, bottom: 40, left: 50 };
  const iw = W - m.left - m.right;
  const ih = Hgt - m.top - m.bottom;
  const yMax = 1.6; // fixed so the "with" line sits comfortably mid-frame
  const sx = (l) => (l / Math.max(1, depth)) * iw;
  const sy = (v) => ih - (Math.min(v, yMax) / yMax) * ih;

  const pathOf = (pts) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p[0]).toFixed(2)} ${sy(p[1]).toFixed(2)}`).join(" ");

  // A small column of shrinking/steady bars to dramatize the same fact.
  const barLayers = Math.min(depth, 12);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end gap-5">
        <div className="w-64">
          <Slider
            label="Network depth (layers)"
            min={1}
            max={50}
            value={depth}
            onChange={(v) => setDepth(Math.round(v))}
            accent="cyan"
            format={(v) => `${v}`}
          />
        </div>
        <div className="text-xs text-slate-500">
          How does a signal's magnitude survive a deep stack — with vs. without the residual{" "}
          <span style={{ color: "#5b7dff" }}>+x</span>?
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {/* line plot, 2 cols */}
        <div className="lg:col-span-2 overflow-x-auto scrollbar-thin">
          <svg viewBox={`0 0 ${W} ${Hgt}`} className="block w-full min-w-[420px]">
            <g transform={`translate(${m.left},${m.top})`}>
              {/* y gridlines */}
              {[0, 0.4, 0.8, 1.2, 1.6].map((v) => (
                <g key={v} transform={`translate(0,${sy(v)})`}>
                  <line x1={0} x2={iw} stroke="#1c2438" />
                  <text x={-8} y={4} textAnchor="end" fontSize={10} fill="#64748b">
                    {v.toFixed(1)}
                  </text>
                </g>
              ))}
              {/* the "healthy" band around 1.0 */}
              <rect x={0} y={sy(1.15)} width={iw} height={sy(0.85) - sy(1.15)} fill="#34d39915" />
              <line x1={0} x2={iw} y1={sy(1)} y2={sy(1)} stroke="#34d399" strokeDasharray="2 4" opacity={0.5} />

              {/* x ticks */}
              {[0, 0.25, 0.5, 0.75, 1].map((f) => {
                const l = Math.round(f * depth);
                return (
                  <g key={f} transform={`translate(${sx(l)},${ih})`}>
                    <line y1={0} y2={5} stroke="#475569" />
                    <text y={18} textAnchor="middle" fontSize={10} fill="#64748b">
                      {l}
                    </text>
                  </g>
                );
              })}

              {/* WITHOUT residual (decays) */}
              <path d={pathOf(series.without)} fill="none" stroke="#fb7185" strokeWidth={2.5} />
              {/* WITH residual (steady) */}
              <path d={pathOf(series.withRes)} fill="none" stroke="#22d3ee" strokeWidth={2.5} />

              {/* endpoint dots */}
              <circle cx={sx(depth)} cy={sy(lastWithout)} r={4} fill="#fb7185" stroke="#0b0e1a" strokeWidth={1.5} />
              <circle cx={sx(depth)} cy={sy(Math.min(lastWith, yMax))} r={4} fill="#22d3ee" stroke="#0b0e1a" strokeWidth={1.5} />

              <text x={iw / 2} y={ih + 34} textAnchor="middle" fontSize={12} fill="#94a3b8">
                layer depth
              </text>
              <text
                transform={`translate(${-38},${ih / 2}) rotate(-90)`}
                textAnchor="middle"
                fontSize={12}
                fill="#94a3b8"
              >
                signal magnitude
              </text>
            </g>
          </svg>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: "#22d3ee" }} />
              with residual (y = x + f(x))
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: "#fb7185" }} />
              without residual (plain stack)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: "#34d399" }} />
              healthy ≈ 1.0
            </span>
          </div>
        </div>

        {/* shrinking-bars dramatization + stats, 1 col */}
        <div className="flex flex-col gap-3">
          <div className="rounded-xl border border-white/10 bg-ink-900/50 p-3">
            <div className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
              signal through {barLayers} layers
            </div>
            <div className="flex items-end gap-3">
              {/* without residual: bars shrink */}
              <div className="flex flex-1 flex-col items-center gap-1">
                <div className="flex h-28 items-end gap-[3px]">
                  {Array.from({ length: barLayers }).map((_, i) => {
                    const h = Math.max(2, Math.pow(PLAIN_GAIN, i) * 100);
                    return (
                      <div
                        key={i}
                        style={{ height: `${h}%`, width: 6, background: heat(0.25 + 0.5 * Math.pow(PLAIN_GAIN, i)) }}
                        className="rounded-sm"
                      />
                    );
                  })}
                </div>
                <span className="text-[10px] text-rose-300">no residual</span>
              </div>
              {/* with residual: bars stay tall */}
              <div className="flex flex-1 flex-col items-center gap-1">
                <div className="flex h-28 items-end gap-[3px]">
                  {Array.from({ length: barLayers }).map((_, i) => (
                    <div
                      key={i}
                      style={{ height: "92%", width: 6, background: "#22d3ee" }}
                      className="rounded-sm"
                    />
                  ))}
                </div>
                <span className="text-[10px] text-cyan-300">with residual</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <DemoStat
              label={`signal @ ${depth}`}
              value={lastWithout < 0.001 ? lastWithout.toExponential(1) : lastWithout.toFixed(3)}
              accent="text-rose-300"
              sub="no residual"
            />
            <DemoStat
              label={`signal @ ${depth}`}
              value={lastWith.toFixed(3)}
              accent="text-cyan-300"
              sub="with residual"
            />
          </div>
          <div className="rounded-lg border border-white/10 bg-ink-900/40 p-3 text-[11px] leading-relaxed text-slate-400">
            Without the skip connection, each layer multiplies the signal by a gain (here{" "}
            <span className="font-mono text-slate-300">{PLAIN_GAIN}</span>); over{" "}
            <span className="font-mono text-slate-300">{depth}</span> layers that compounds to{" "}
            <span className="font-mono text-rose-300">{PLAIN_GAIN}^{depth}</span> — the signal (and
            its gradient) vanishes. The residual <span style={{ color: "#5b7dff" }}>+x</span> keeps an
            identity path, so the magnitude stays near 1 no matter how deep you go.
          </div>
        </div>
      </div>
    </div>
  );
}

function DemoStat({ label, value, accent, sub }) {
  return (
    <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`font-mono text-lg ${accent}`}>{value}</div>
      <div className="text-[10px] text-slate-600">{sub}</div>
    </div>
  );
}
